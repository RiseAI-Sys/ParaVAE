import torch.nn as nn
from paravae.models.WAN2_1.patch_vae import PatchZeroPad2d, PatchConv2d, PatchCausalConv3d, \
    PatchResample, PatchResidualBlock, PatchEncoder3d, PatchDecoder3d, PatchWanVAE_


class ZeroPad2dAdapter(PatchZeroPad2d):
    '''
    Convert nn.ZeroPad2d to PatchZeroPad2d
    '''
    def __init__(
        self, 
        zeropad2d: nn.ZeroPad2d,
    ):
        super().__init__(
            padding=zeropad2d.padding
        )
    

class Conv2dAdapter(PatchConv2d):
    '''
    Convert nn.Conv2d to PatchConv2d
    '''
    def __init__(
        self, 
        conv2d: nn.Conv2d,
    ):
        super().__init__(
            in_channels=conv2d.in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size,
            stride=conv2d.stride,
            padding=conv2d.padding,
            dilation=conv2d.dilation,
            groups=conv2d.groups,
            bias=conv2d.bias is not None,
            padding_mode=conv2d.padding_mode,
            device=conv2d.weight.device,
            dtype=conv2d.weight.dtype,
        )
        for i in conv2d.dilation:
            assert i == 1, "dilation is not supported in Conv2dAdapter"
            
        self.weight.data = conv2d.weight.data
        self.bias.data = conv2d.bias.data


class CausalConv3dAdapter(PatchCausalConv3d):
    '''
    Convert nn.Conv3d to PatchCausalConv3d
    '''
    def __init__(
        self, 
        conv3d: nn.Conv3d,
    ):
        for i in conv3d.dilation:
            assert i == 1, "dilation is not supported in CausalConv3dAdapter"
            
        original_padding = (conv3d._padding[4]//2, conv3d._padding[2], conv3d._padding[0]) 
        
        super().__init__(
            in_channels=conv3d.in_channels,
            out_channels=conv3d.out_channels,
            kernel_size=conv3d.kernel_size,
            stride=conv3d.stride,
            padding=original_padding,
            dilation=conv3d.dilation,
            groups=conv3d.groups,
            bias=conv3d.bias is not None,
            padding_mode=conv3d.padding_mode,
            device=conv3d.weight.device,
            dtype=conv3d.weight.dtype
        )

        self.weight.data = conv3d.weight.data.clone()
        self.bias.data = conv3d.bias.data.clone()
    

class ResampleAdapter(PatchResample):
    '''
    Convert Resample to PatchResample
    '''
    def __init__(
        self, 
        resample,
    ):
        super().__init__(
            dim=resample.dim, 
            mode=resample.mode
        )
        
        modules = list(resample.resample.children())
        for i, layer in enumerate(modules):
            if isinstance(layer, nn.Conv2d):  
                new_conv = Conv2dAdapter(layer)
                modules[i] = new_conv
            elif isinstance(layer, nn.ZeroPad2d):  
                new_conv = ZeroPad2dAdapter(layer)
                modules[i] = new_conv
        self.resample = nn.Sequential(*modules)
        
        if self.mode == 'upsample3d' or self.mode == 'downsample3d':
            self.time_conv = CausalConv3dAdapter(resample.time_conv)
    

class ResidualBlockAdapter(PatchResidualBlock):
    '''
    Convert ResidualBlock to PatchResidualBlock
    '''
    def __init__(
        self, 
        residualBlock,
    ):
        super().__init__(
            in_dim=residualBlock.in_dim, 
            out_dim=residualBlock.out_dim
        )
        
        # Convert CausalConv3d in the ResidualBlock.residual to patch version
        modules = list(residualBlock.residual.children())
        for i, layer in enumerate(modules):
            if layer.__class__.__name__ == "CausalConv3d": 
                new_conv = CausalConv3dAdapter(layer)
                modules[i] = new_conv
        self.residual = nn.Sequential(*modules)

        # Convert CausalConv3d in the ResidualBlock.shortcut to patch version
        if residualBlock.shortcut.__class__.__name__ == "CausalConv3d":
            new_shortcut = CausalConv3dAdapter(residualBlock.shortcut)
            self.shortcut = new_shortcut
        else:
            self.shortcut = residualBlock.shortcut
    

class Encoder3dAdapter(PatchEncoder3d):
    '''
    Convert Encoder3d to PatchEncoder3d
    '''
    def __init__(
        self, 
        encoder3d
    ):
        super().__init__(
            dim=encoder3d.dim, 
            z_dim=encoder3d.z_dim,
            dim_mult=encoder3d.dim_mult,
            num_res_blocks=encoder3d.num_res_blocks,
            attn_scales=encoder3d.attn_scales,
            temperal_downsample=encoder3d.temperal_downsample,
            dropout=0.0
        )
        
        # init block
        self.conv1 = CausalConv3dAdapter(encoder3d.conv1)
        
        # downsampe blocks
        # Convert ResidualBlock, Resample in the Encoder3d.downsamples to patch version
        modules = list(encoder3d.downsamples.children())
        for i, layer in enumerate(modules):
            # if isinstance(layer, ResidualBlock):
            if layer.__class__.__name__ == "ResidualBlock":
                new_conv = ResidualBlockAdapter(layer)
                modules[i] = new_conv
            # if isinstance(layer, Resample): 
            if layer.__class__.__name__ == "Resample":
                new_conv = ResampleAdapter(layer)
                modules[i] = new_conv
        self.downsamples = nn.Sequential(*modules)
        
        # middle blocks
        self.middle = encoder3d.middle
        
        # patch head
        # Convert CausalConv3d in the Encoder3d.head to patch version
        modules = list(encoder3d.head.children())
        for i, layer in enumerate(modules):
            # if isinstance(layer, CausalConv3d): 
            if layer.__class__.__name__ == "CausalConv3d":
                new_conv = CausalConv3dAdapter(layer)
                modules[i] = new_conv
        self.head = nn.Sequential(*modules)
    
    
class Decoder3dAdapter(PatchDecoder3d):
    '''
    Convert Decoder3d to PatchDecoder3d
    '''
    def __init__(
            self, 
            decoder3d,
    ):
        
        super().__init__(
            dim=decoder3d.dim, 
            z_dim=decoder3d.z_dim,
            dim_mult=decoder3d.dim_mult,
            num_res_blocks=decoder3d.num_res_blocks,
            attn_scales=decoder3d.attn_scales,
            temperal_upsample=decoder3d.temperal_upsample,
            dropout=0.0
        )
        
        # init block
        self.conv1 = decoder3d.conv1

        # middle blocks
        self.middle = decoder3d.middle
        
        # upsampe blocks
        # Convert ResidualBlock, Resample in the Decoder3d.upsamples to patch version
        modules = list(decoder3d.upsamples.children())
        for i, layer in enumerate(modules):
            # if isinstance(layer, ResidualBlock):  
            if layer.__class__.__name__ == "ResidualBlock":
                new_conv = ResidualBlockAdapter(layer)
                modules[i] = new_conv
            # elif isinstance(layer, Resample): 
            if layer.__class__.__name__ == "Resample":
                new_conv = ResampleAdapter(layer)
                modules[i] = new_conv
        self.upsamples = nn.Sequential(*modules)
        
        # output blocks
        # Convert CausalConv3d in the Encoder3d.head to patch version
        modules = list(decoder3d.head.children())
        for i, layer in enumerate(modules):
            # if isinstance(layer, CausalConv3d): 
            if layer.__class__.__name__ == "CausalConv3d":
                new_conv = CausalConv3dAdapter(layer)
                modules[i] = new_conv
        self.head = nn.Sequential(*modules)
    
    
class WanVAEAdapter(PatchWanVAE_):
    def __init__(
        self, 
        wanVae_
    ):
        super().__init__(
            dim=wanVae_.dim, 
            z_dim=wanVae_.z_dim,
            dim_mult=wanVae_.dim_mult,
            num_res_blocks=wanVae_.num_res_blocks,
            attn_scales=wanVae_.attn_scales,
            temperal_downsample=wanVae_.temperal_downsample,
            dropout=0.0
        )     
        self.encoder = Encoder3dAdapter(wanVae_.encoder)
        self.conv1 = wanVae_.conv1
        self.conv2 = wanVae_.conv2
        self.decoder = Decoder3dAdapter(wanVae_.decoder)