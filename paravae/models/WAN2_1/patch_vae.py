import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair, _triple
from einops import rearrange
from torch.nn.common_types import _size_2_t, _size_3_t
from typing import Optional, List, Union
from pathlib import Path
from safetensors.torch import load_file as load_sft
from torch.utils.checkpoint import checkpoint

from paravae.dist.distributed_env import DistributedEnv
from paravae.dist.split_gather import split_forward_gather_backward, gather_forward_split_backward

CACHE_T = 2

class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)
    

class RMS_norm(nn.Module):
    
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias
        
class Upsample(nn.Upsample):
    
    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)
    

class ResidualBlock(nn.Module):
    
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if layer.__class__.__name__ == "CausalConv3d" and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity

class PatchZeroPad2d(nn.Module):
    '''
    ZeroPad2d for patched input. For each patch, pad left and right; 
    but only pad up for first patch and only pad down for last patch.
    '''
    def __init__(
        self, 
        padding
    ):
        super(PatchZeroPad2d, self).__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif len(padding) == 4:
            self.padding = tuple(padding)
        else:
            raise ValueError("padding must be an int or a 4-element tuple")

    def forward(self, x):
        if not dist.is_initialized() or DistributedEnv.get_group_world_size() == 1:
            return F.pad(x, self.padding, mode='constant', value=0)

        group_world_size = DistributedEnv.get_group_world_size()
        rank_in_group = DistributedEnv.get_rank_in_vae_group()

        adjusted_padding = list(self.padding)
        # Only first patch need up padding
        if rank_in_group != 0:
            adjusted_padding[2] = 0 
        # Only last patch need down padding
        if rank_in_group != group_world_size - 1:
            adjusted_padding[3] = 0

        return F.pad(x, adjusted_padding, mode='constant', value=0)


class PatchConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  
        device=None,
        dtype=None
    ):

        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in PatchConv2d"
        else:
            for i in dilation:
                assert i == 1, "dilation is not supported in PatchConv2d"
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode, device, dtype
        )
        
    def _adjust_padding_for_patch(self, padding, rank, world_size):
        if isinstance(padding, tuple):
            padding = list(padding)
        elif isinstance(padding, int):
            padding = [padding] * 4

        if rank == 0:
            padding[-1] = 0
        elif rank == world_size - 1:
            padding[-2] = 0
        else:
            padding[-2:] = [0, 0]
        return tuple(padding)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        bs, channels, h, w = input.shape

        group_world_size, global_rank, rank_in_group, local_rank = _get_world_size_and_rank()

        if (group_world_size == 1):
            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            return F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
            
        else:
            # 1. get the meta data of input tensor and conv operation
            patch_height_list = [torch.zeros(1, dtype=torch.int64, device=f"cuda:{local_rank}") for _ in range(group_world_size)]
            dist.all_gather(patch_height_list, torch.tensor([h], dtype=torch.int64, device=f"cuda:{local_rank}"), group=DistributedEnv.get_vae_group())
            patch_height_index = _calc_patch_height_index(patch_height_list)
            halo_width = _calc_halo_width_in_h_dim(rank_in_group,  patch_height_index, self.kernel_size[0], self.padding[0], self.stride[0])
            prev_bottom_halo_width: int = 0
            next_top_halo_width: int = 0
            if rank_in_group != 0:
                prev_bottom_halo_width = _calc_bottom_halo_width(rank_in_group - 1, patch_height_index, self.kernel_size[0], self.padding[0], self.stride[0])
            if rank_in_group != group_world_size - 1:
                next_top_halo_width = _calc_top_halo_width(rank_in_group + 1, patch_height_index, self.kernel_size[0], self.padding[0], self.stride[0])
                next_top_halo_width = max(0, next_top_halo_width)
            
            assert halo_width[0] <= h and halo_width[1] <= h, "halo width is larger than the height of input tensor"

            # 2. get the halo region from other ranks
            # up to down
            to_next = None
            to_prev = None
            top_halo_recv = None
            bottom_halo_recv = None
            global_rank_of_next, global_rank_of_prev  = None, None
            if next_top_halo_width > 0:
                # isend to next
                bottom_halo_send = input[:, :, -next_top_halo_width:, :].contiguous()
                global_rank_of_next = DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
                to_next = dist.isend(bottom_halo_send, global_rank_of_next, group=DistributedEnv.get_vae_group())
                
            if halo_width[0] > 0:
                # recv from prev
                assert patch_height_index[rank_in_group] - halo_width[0] >= patch_height_index[rank_in_group-1], \
                    "width of top halo region is larger than the height of input tensor of last rank"
                top_halo_recv = torch.empty([bs, channels, halo_width[0], w], dtype=input.dtype, device=f"cuda:{local_rank}")
                global_rank_of_prev = DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
                dist.recv(top_halo_recv, global_rank_of_prev, group=DistributedEnv.get_vae_group())

            # down to up
            if prev_bottom_halo_width > 0:
                # isend to prev
                top_halo_send = input[:, :, :prev_bottom_halo_width, :].contiguous()
                if global_rank_of_prev is None:
                    global_rank_of_prev = DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
                to_prev = dist.isend(top_halo_send, global_rank_of_prev, group=DistributedEnv.get_vae_group())
            
            if halo_width[1] > 0:
                # recv from next
                assert patch_height_index[rank_in_group+1] + halo_width[1] < patch_height_index[rank_in_group+2], \
                    "width of bottom halo region is larger than the height of input tensor of next rank"
                bottom_halo_recv = torch.empty([bs, channels, halo_width[1], w], dtype=input.dtype, device=f"cuda:{local_rank}")
                if global_rank_of_next is None:
                    global_rank_of_next = DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
                dist.recv(bottom_halo_recv, global_rank_of_next, group=DistributedEnv.get_vae_group())
        
            # Remove redundancy at the top of the input
            if halo_width[0] < 0:
                input = input[:, :, -halo_width[0]:, :]
            # concat the halo region to the input tensor            
            if top_halo_recv is not None:
                input = torch.cat([top_halo_recv, input], dim=-2)
            if bottom_halo_recv is not None:
                input = torch.cat([input, bottom_halo_recv], dim=-2)
            
            # wait for the communication to finish
            if to_next is not None:
                to_next.wait()
            if to_prev is not None:
                to_prev.wait()

            # 3. do convolution and postprocess
            conv_res: Tensor
            padding = self._adjust_padding_for_patch(self._reversed_padding_repeated_twice, rank=rank_in_group, world_size=group_world_size)
            bs, channels, h, w = input.shape
            if self.padding_mode != 'zeros':
                conv_res = F.conv2d(F.pad(input, padding, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            else:
                if self.stride[0] == 1 and self.padding[0] == 1 and self.kernel_size[0] == 3:
                    conv_res = F.conv2d(input, weight, bias, self.stride,
                                self.padding, self.dilation, self.groups)
                    if halo_width[1] == 0:
                        conv_res = conv_res[:, :, halo_width[0]:, :].contiguous()
                    else:
                        conv_res = conv_res[:, :, halo_width[0]:-halo_width[1], :]
                else:
                    conv_res = F.conv2d(F.pad(input, padding, "constant", 0.0),
                                    weight, bias, self.stride,
                                    _pair(0), self.dilation, self.groups)
            return conv_res
            

class PatchCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ):
        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in PatchCausalConv3d"
        else:
            for i in dilation:
                assert i == 1, "dilation is not supported in PatchCausalConv3d"
        
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype
        )

        # casual
        self._causal_padding = (self.padding[2], self.padding[2],  
                                self.padding[1], self.padding[1],  
                                2 * self.padding[0], 0)      

    # in 3d case, padding is a tuple of 6 integers: [left_pad, right_pad, top_pad, bottom_pad, front_pad, back_pad]
    def _adjust_padding_for_patch(self, padding, rank, world_size):
        if isinstance(padding, tuple):
            _padding = list(padding)
        elif isinstance(padding, int):
            _padding = [padding] * 6  # [left, right, top, bottom, front, back]
        if rank == 0:
            _padding[3] = 0  # bottom no pad
        elif rank == world_size - 1:
            _padding[2] = 0  # top no pad
        else:
            _padding[2] = 0  # top and bottom no pad
            _padding[3] = 0
        return tuple(_padding)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        bs, channels, d, h, w = input.shape
        group_world_size, global_rank, rank_in_group, local_rank = _get_world_size_and_rank()

        if group_world_size == 1:
            if self.padding_mode != 'zeros':
                return F.conv3d(F.pad(input, self._causal_padding, mode=self.padding_mode),
                                weight, bias, self.stride, _triple(0), self.dilation, self.groups)
            
            return F.conv3d(F.pad(input, self._causal_padding), weight, bias, self.stride,
                            _triple(0), self.dilation, self.groups)
            
        else:
            patch_height_list = [torch.zeros(1, dtype=torch.int64, device=f"cuda:{local_rank}") for _ in range(group_world_size)]
            dist.all_gather(patch_height_list, torch.tensor([h], dtype=torch.int64, device=f"cuda:{local_rank}"), group=DistributedEnv.get_vae_group())
            patch_height_index = _calc_patch_height_index(patch_height_list)
            halo_width = _calc_halo_width_in_h_dim(rank_in_group, patch_height_index, self.kernel_size[1], self.padding[1], self.stride[1])

            prev_bottom_halo_width: int = 0
            next_top_halo_width: int = 0
            if rank_in_group != 0:
                prev_bottom_halo_width = _calc_bottom_halo_width(rank_in_group - 1, patch_height_index, self.kernel_size[1], self.padding[1], self.stride[1])
            if rank_in_group != group_world_size - 1:
                next_top_halo_width = _calc_top_halo_width(rank_in_group + 1, patch_height_index, self.kernel_size[1], self.padding[1], self.stride[1])
                next_top_halo_width = max(0, next_top_halo_width)
                
            assert halo_width[0] <= h and halo_width[1] <= h, "halo width is larger than the height of input tensor"

            to_next = None
            to_prev = None
            top_halo_recv = None
            bottom_halo_recv = None
            global_rank_of_next, global_rank_of_prev = None, None

            if next_top_halo_width > 0:
                bottom_halo_send = input[:, :, :, -next_top_halo_width:, :].contiguous()
                global_rank_of_next = DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
                to_next = dist.isend(bottom_halo_send, global_rank_of_next, group=DistributedEnv.get_vae_group())

            if halo_width[0] > 0:
                # recv from prev
                assert patch_height_index[rank_in_group] - halo_width[0] >= patch_height_index[rank_in_group-1], \
                    "width of top halo region is larger than the height of input tensor of last rank"
                top_halo_recv = torch.empty(
                    [bs, channels, d, halo_width[0], w], dtype=input.dtype, device=f"cuda:{local_rank}"
                )
                global_rank_of_prev = DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
                dist.recv(top_halo_recv, global_rank_of_prev, group=DistributedEnv.get_vae_group())

            if prev_bottom_halo_width > 0:
                top_halo_send = input[:, :, :, :prev_bottom_halo_width, :].contiguous()
                if global_rank_of_prev is None:
                    global_rank_of_prev = DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
                to_prev = dist.isend(top_halo_send, global_rank_of_prev, group=DistributedEnv.get_vae_group())

            if halo_width[1] > 0:
                bottom_halo_recv = torch.empty(
                    [bs, channels, d, halo_width[1], w], dtype=input.dtype, device=f"cuda:{local_rank}"
                )
                if global_rank_of_next is None:
                    global_rank_of_next = DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
                dist.recv(bottom_halo_recv, global_rank_of_next, group=DistributedEnv.get_vae_group())

            if halo_width[0] < 0:
                input = input[:, :, :, -halo_width[0]:, :]

            if top_halo_recv is not None:
                input = torch.cat([top_halo_recv, input], dim=-2)
            if bottom_halo_recv is not None:
                input = torch.cat([input, bottom_halo_recv], dim=-2)

            if to_next is not None:
                to_next.wait()
            if to_prev is not None:
                to_prev.wait()

            padding = self._adjust_padding_for_patch(self._causal_padding, rank=rank_in_group, world_size=group_world_size)

            if self.padding_mode != 'zeros':
                conv_res = F.conv3d(F.pad(input, padding, mode=self.padding_mode),
                                    weight, bias, self.stride, _triple(0), self.dilation, self.groups)
            else:
                conv_res = F.conv3d(F.pad(input, padding, "constant", 0.0),
                                    weight, bias, self.stride, _triple(0), self.dilation, self.groups)
            return conv_res
        
    def forward(self, x, cache_x=None):
        self._causal_padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1],  2 * self.padding[0], 0)
        padding = list(self._causal_padding)
        if cache_x is not None and self._causal_padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        
        # update _causal_padding
        self._causal_padding = tuple(padding)
        x = self._conv_forward(x, self.weight, self.bias)
        self._causal_padding = (self.padding[2], self.padding[2],  self.padding[1], self.padding[1],  2 * self.padding[0], 0)
        
        return x
    

class PatchResample(nn.Module):
    '''
    Resample for patched input. Convert all Conv2d, Zeropad2d and CausalConv3d to patch version.
    '''

    def __init__(
        self, 
        dim, 
        mode
    ):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                PatchConv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                PatchConv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = PatchCausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                PatchZeroPad2d((0, 1, 0, 1)),
                PatchConv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                PatchZeroPad2d((0, 1, 0, 1)),
                PatchConv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = PatchCausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)
        

class PatchResidualBlock(nn.Module):
    '''
    ResidualBlock for patched input. Convert all CausalConv3d to patch version.
    '''
    
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        dropout=0.0
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), 
            nn.SiLU(),
            PatchCausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), 
            nn.SiLU(), 
            nn.Dropout(dropout),
            PatchCausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = PatchCausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, PatchCausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h
    

class PatchEncoder3d(nn.Module):
    '''
    Encoder3d for patched input. Convert CausalConv3d in conv1, downsamples and head to patch version.
    '''
    
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = PatchCausalConv3d(3, dims[0], 3, padding=1)
        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(PatchResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(PatchResample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            PatchCausalConv3d(out_dim, z_dim, 3, padding=1))
        
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # patch
        x = split_forward_gather_backward(None, x, 3)
        
        # conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
                        
        # downsamples
        for layer in self.downsamples:    
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
            continue
        
        # depatch
        x = gather_forward_split_backward(None, x, 3)

        ## middle
        for layer in self.middle:
            if layer.__class__.__name__ == "ResidualBlock" and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
        
        ## head
        for layer in self.head:
            if isinstance(layer, PatchCausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
                
        return x
    

class PatchDecoder3d(nn.Module):
    '''
    Decoder3d for patched input. Convert CausalConv3d in upsamples and head to patch version.
    '''
    
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(PatchResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(PatchResample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            PatchCausalConv3d(out_dim, 3, 3, padding=1))
        
    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if layer.__class__.__name__ == "ResidualBlock" and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
               
        x = split_forward_gather_backward(None, x, 3)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)
                        
        ## head
        for layer in self.head:
            if isinstance(layer, PatchCausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        
        x = gather_forward_split_backward(None, x, 3)
        
        return x
    
    
def count_conv3d(model):
    '''
    Get number of CausalConv3d and PatchCasualConv3d in WanVae.
    '''
    count = 0
    for m in model.modules():
        # PatchCausalConv3d for initialized from self, and CausalConv3dAdapter for initialized from adapter
        if m.__class__.__name__ == "CausalConv3d" or m.__class__.__name__ == "PatchCausalConv3d" or m.__class__.__name__ == "CausalConv3dAdapter":
            count += 1
    return count


class PatchWanVAE_(nn.Module):
    '''
    WanVAE for patched input, convert Encoder3d and Decoder3d to patch version.
    '''

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.spatial_compression_ratio = 2 ** len(self.temperal_downsample)

        # modules
        self.encoder = PatchEncoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = PatchDecoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)
        
        self.use_slicing = False
        self.use_tiling = False
        
        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        
        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        
        self.gradient_checkpointing = True
        self.training = True
        
    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
    ) -> None:
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        
    def disable_tiling(self) -> None:
        self.use_tiling = False

    def enable_slicing(self) -> None:
        self.use_slicing = True
    
    def disable_slicing(self) -> None:
        self.use_slicing = False

    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            mu, log_var = checkpoint(self.encode, x, use_reentrant=False)
        else:
            mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        
        if self.training:
            x = checkpoint(self.decode, z, use_reentrant=False)
            # x = self.decode(z)
        else:
            x = self.decode(z)
        
        return x, mu, log_var, z
    
    def encode(self, x, scale=None):
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice, scale) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
            
        mu, log_var = h.chunk(2, dim=1)
        if scale:
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
            return mu
        else:
            return mu, log_var
                       
    def _encode(self, x):
        _, _, num_frame, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)
        
        self.clear_cache()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                    )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                    )
                out = torch.cat([out, out_], 2)
        
        out = self.conv1(out)
        self.clear_cache()
        return out
    
    def decode(self, z, scale=None):
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice, scale).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, scale)
            
        return decoded

    def _decode(self, z, scale):
        _, _, num_frame, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        
        # z: [b,c,t,h,w]
        if scale:
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                z = z / scale[1] + scale[0]

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z)
        
        self.clear_cache()
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    )
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    )
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out
    
    def tiled_encode(self, x: torch.Tensor):
        _, _, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        # The minimal distance between two spatial tiles
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                self.clear_cache()
                time = []
                frame_range = 1 + (num_frames - 1) // 4
                for k in range(frame_range):
                    self._enc_conv_idx = [0]
                    if k == 0:
                        tile = x[:, :, :1, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                    else:
                        tile = x[
                            :,
                            :,
                            1 + 4 * (k - 1) : 1 + 4 * k,
                            i : i + self.tile_sample_min_height,
                            j : j + self.tile_sample_min_width,
                        ]
                    tile = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                    tile = self.conv1(tile)
                    time.append(tile)
                    
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc
    
    def tiled_decode(self, z: torch.Tensor):
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                self.clear_cache()
                time = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width]
                    tile = self.conv2(tile)
                    decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
                    time.append(decoded)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
        
        return dec
    
    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        #cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def load_state_dict(model, ckpt_path, device="cuda", strict=False, assign=True):
    if Path(ckpt_path).suffix == ".safetensors":
        state_dict = load_sft(ckpt_path, device)
    else:
        state_dict = torch.load(ckpt_path, map_location=device)#"cpu")

    missing, unexpected = model.load_state_dict(
        state_dict, strict=strict, assign=assign
    )
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    return model

def _video_patch_vae(pretrained_path=None, z_dim=16, device='cpu', **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    # with torch.device('meta'):
    model = PatchWanVAE_(**cfg)

    # load checkpoint
    if pretrained_path:
        logging.info(f'loading {pretrained_path}')
        # model.load_state_dict(
        #     torch.load(pretrained_path, map_location=device), assign=True)
        model = load_state_dict(model, pretrained_path, device=device, strict=True, assign=True)
    
    torch.cuda.empty_cache()

    return model


def _get_world_size_and_rank():
    group_world_size = DistributedEnv.get_group_world_size()
    global_rank = DistributedEnv.get_global_rank()
    rank_in_group = DistributedEnv.get_rank_in_vae_group()
    local_rank = DistributedEnv.get_local_rank()
    return group_world_size, global_rank, rank_in_group, local_rank

def _calc_patch_height_index(patch_height_list: List[Tensor]):
    height_index = []
    cur = 0
    for t in patch_height_list:
        height_index.append(cur)
        cur += t.item()
    height_index.append(cur)
    return height_index

def _calc_bottom_halo_width(rank, height_index, kernel_size, padding = 0, stride = 1):
    assert rank >= 0, "rank should not be smaller than 0"
    assert rank < len(height_index) - 1, "rank should be smaller than the length of height_index - 1"
    assert padding >= 0, "padding should not smaller than 0"
    assert stride > 0, "stride should be larger than 0"

    if rank == DistributedEnv.get_group_world_size() - 1:
        return 0
    nstep_before_bottom = (height_index[rank + 1] + padding - (kernel_size - 1) // 2 + stride - 1) // stride
    assert nstep_before_bottom > 0, "nstep_before_bottom should be larger than 0"
    bottom_halo_width =  (nstep_before_bottom - 1) * stride + kernel_size - padding - height_index[rank + 1]
    return max(0, bottom_halo_width)

def _calc_top_halo_width(rank, height_index, kernel_size, padding = 0, stride = 1):
    assert rank >= 0, "rank should not be smaller than 0"
    assert rank < len(height_index) - 1, "rank should be smaller than the length of height_index - 1"
    assert padding >= 0, "padding should not smaller than 0"
    assert stride > 0, "stride should be larger than 0"

    if rank == 0:
        return 0
    nstep_before_top = (height_index[rank] + padding - (kernel_size - 1) // 2 + stride - 1) // stride
    top_halo_width = height_index[rank] - (nstep_before_top * stride - padding)
    return top_halo_width

def _calc_halo_width_in_h_dim(rank, height_index, kernel_size, padding = 0, stride = 1):
    ''' 
        Calculate the width of halo region in height dimension. 
        The halo region is the region that is used for convolution but not included in the output.
        return value: (top_halo_width, bottom_halo_width)
    '''
    halo_width = [
        _calc_top_halo_width(rank, height_index, kernel_size, padding, stride),
        _calc_bottom_halo_width(rank, height_index, kernel_size, padding, stride)
    ]
    if rank == 0:
        halo_width[0] = 0
    elif rank == DistributedEnv.get_group_world_size() - 1:
        halo_width[1] = 0
    return tuple(halo_width)
