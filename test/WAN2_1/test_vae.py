import torch
import torch.distributed as dist
import random
import argparse
import time

from paravae.dist.distributed_env import DistributedEnv

from paravae.models.WAN2_1.vae import WanVAE_
from paravae.models.WAN2_1.adapter import WanVAEAdapter

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# @torch.no_grad()
def main():
    '''
    For memory test: torchrun --nproc_per_node=2 test/WAN2.1/test_vae.py --memory_test
    For correctness test: torchrun --nproc_per_node=2 test/WAN2.1/test_vae.py --correctness_test
    For tiling test: torchrun --nproc_per_node=2 test/WAN2.1/test_vae.py --tiling_test
    '''
    set_seed()
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=5, help="Depth of input tensor")
    parser.add_argument("--height", type=int, default=512, help="Height of input tensor")
    parser.add_argument("--width", type=int, default=512, help="Width of input tensor")
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--enable_slicing", action="store_true")
    parser.add_argument("--memory_test", action="store_true")
    parser.add_argument("--tiling_test", action="store_true")
    parser.add_argument("--correctness_test", action="store_true")
    args = parser.parse_args()
    
    memory_test = False
    tiling_test = False
    correctness_test = False

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    DistributedEnv.initialize(None)
    device = torch.device(f"cuda:{rank}")
    
    data_type = torch.float32
    
    base_vae = WanVAE_().to(data_type).to(device=device)    
    approximate_patch_vae = WanVAE_().to(data_type).to(device=device)
    approximate_patch_vae.load_state_dict(base_vae.state_dict())
    patch_vae = WanVAEAdapter(base_vae).to(data_type).to(device=device) 
    
    approximate_patch_vae.enable_approximate_patch()   
        
    if args.enable_slicing:
        approximate_patch_vae.enable_slicing()
        patch_vae.model.enable_slicing()
    if args.enable_tiling:
        approximate_patch_vae.enable_tiling()
        patch_vae.model.enable_tiling()
    if args.memory_test:
        memory_test = True
    if args.tiling_test:
        tiling_test = True
    if args.correctness_test:
        correctness_test = True
            
    hidden_state = torch.randn(1, 3, args.depth, args.height, args.width, device=device, dtype=data_type, requires_grad=True)
    warmup_hidden_state = torch.randn(1, 3, 9, 64, 64, device=device, dtype=data_type, requires_grad=True)
    scale = [0, 1] 
    
    if memory_test:
        # base_vae with grad
        ## warmup
        for i in range(10):
            latent = base_vae.encode(warmup_hidden_state, scale)
            pred = base_vae.decode(latent, scale)
            loss = pred.mean()
            loss.backward()
        
        ## run   
        for i in range(10):
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            latent = base_vae.encode(hidden_state, scale)
            pred = base_vae.decode(latent, scale)
            loss = pred.mean()
            loss.backward()
            
            peak_memory = torch.cuda.max_memory_allocated(device=device)
            if rank == 0:
                print(f"distwanvae_base_vae: resolution: {args.depth}x{args.height}x{args.width}, time: {time.time() - start_time} sec, peak memory: {peak_memory / 2 ** 30} GB")

        # approximate_patch_vae with grad
        ## warmup
        for i in range(10):
            latent = approximate_patch_vae.encode(warmup_hidden_state, scale)
            pred = approximate_patch_vae.decode(latent, scale)
            loss = pred.mean()
            loss.backward()
         
        ## run   
        for i in range(10):
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            latent = approximate_patch_vae.encode(hidden_state, scale)
            pred = approximate_patch_vae.decode(latent, scale)
            loss = pred.mean()
            loss.backward()
            
            for param in approximate_patch_vae.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=DistributedEnv.get_vae_group())
            
            peak_memory = torch.cuda.max_memory_allocated(device=device)
            if rank == 0:
                print(f"distwanvae_approximate_patch_vae: resolution: {args.depth}x{args.height}x{args.width}, time: {time.time() - start_time} sec, peak memory: {peak_memory / 2 ** 30} GB")

        # patch_vae with grad
        ## warmup
        for i in range(10):
            patch_latent = patch_vae.encode(warmup_hidden_state, scale)
            patch_pred = patch_vae.decode(patch_latent, scale)
            patch_loss = patch_pred.mean()
            patch_loss.backward()
        
        ## run
        for i in range(10):
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
    
            patch_latent = patch_vae.encode(hidden_state, scale)
            patch_pred = patch_vae.decode(patch_latent, scale)
            patch_loss = patch_pred.mean()
            patch_loss.backward()
            
            for param in patch_vae.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=DistributedEnv.get_vae_group())
            
            peak_memory = torch.cuda.max_memory_allocated(device=device)
            if rank == 0:
                print(f"distwanvae_patch_vae: resolution: {args.depth}x{args.height}x{args.width}, time: {time.time() - start_time} sec, peak memory: {peak_memory / 2 ** 30} GB")

    # correctness verification with grad (because of backward)
    if correctness_test:
        ## base_vae
        base_latent = base_vae.encode(hidden_state, scale)
        base_pred = base_vae.decode(base_latent, scale)
        base_loss = base_pred.mean()
        base_loss.backward()
        
        base_weight_grad = base_vae.decoder.head[2].weight.grad.clone()
        
        ## approximate_patch_vae
        standard_latent = approximate_patch_vae.encode(hidden_state, scale)
        standard_pred = approximate_patch_vae.decode(standard_latent, scale)    
        standard_loss = standard_pred.mean()
        standard_loss.backward()
        for param in approximate_patch_vae.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=DistributedEnv.get_vae_group())
                
        standard_weight_grad = approximate_patch_vae.decoder.head[2].weight.grad.clone()
        
        ## patch_vae
        patch_latent = patch_vae.encode(hidden_state, scale)
        patch_pred = patch_vae.decode(patch_latent, scale)
        patch_loss = patch_pred.mean()
        patch_loss.backward()
        for param in patch_vae.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=DistributedEnv.get_vae_group())
        
        patch_weight_grad = patch_vae.model.decoder.head[2].weight.grad.clone()
                
        if rank == 0:            
            print("⭕️ approximate patch latent max error:", (base_latent - standard_latent).abs().max().item())
            print("⭕️ approximate patch latent mean error:", (base_latent - standard_latent).abs().mean().item())
            print("⭕️ approximate patch pred max error:", (base_pred - standard_pred).abs().max().item())
            print("⭕️ approximate patch pred mean error:", (base_pred - standard_pred).abs().mean().item())
            print("✅ approximate patch grad max error:", (base_weight_grad - standard_weight_grad).abs().max().item())
            print("✅ approximate patch grad mean error:", (base_weight_grad - standard_weight_grad).abs().mean().item())
        
            print("⭕️ patch latent max error:", (base_latent - patch_latent).abs().max().item())
            print("⭕️ patch latent mean error:", (base_latent - patch_latent).abs().mean().item())
            print("⭕️ patch pred max error:", (base_pred - patch_pred).abs().max().item())
            print("⭕️ patch pred mean error:", (base_pred - patch_pred).abs().mean().item())
            print("✅ patch grad max error:", (base_weight_grad - patch_weight_grad).abs().max().item())
            print("✅ patch grad mean error:", (base_weight_grad - patch_weight_grad).abs().mean().item())
            
    # tiling slicing test without grad
    if tiling_test:
        hidden_state = torch.randn(1, 3, 21, 1024, 2048, device=device, dtype=data_type, requires_grad=False)
        warmup_hidden_state = torch.randn(1, 3, 9, 64, 64, device=device, dtype=data_type, requires_grad=False)
        approximate_patch_vae.enable_tiling()
        patch_vae.model.enable_tiling()
        
        # warmup
        for i in range(3):
            encoded = base_vae.encode(warmup_hidden_state, scale)
            base_vae.decode(encoded, scale)
            encoded = approximate_patch_vae.encode(warmup_hidden_state, scale)
            approximate_patch_vae.decode(encoded, scale)
            encoded = patch_vae.encode(warmup_hidden_state, scale)
            patch_vae.decode(encoded, scale)
            
        
        # approximate_patch_vae
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            encoded = approximate_patch_vae.encode(hidden_state, scale)
            standard_decoded = approximate_patch_vae.decode(encoded, scale)
        peak_memory = torch.cuda.max_memory_allocated(device=device)
        if rank == 0:
            print(f"approximate_patch_vae_block + tiling: resolution: 1x3x21x1024x2048, time: {time.time() - start_time} sec, peak memory: {peak_memory / 2 ** 30} GB")
            
        # patch_vae
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            encoded = patch_vae.encode(hidden_state, scale)
            patch_decoded = patch_vae.decode(encoded, scale) 
        peak_memory = torch.cuda.max_memory_allocated(device=device)
        if rank == 0:
            print(f"patch_vae_block + tiling: resolution: 1x3x21x1024x2048, time: {time.time() - start_time} sec, peak memory: {peak_memory / 2 ** 30} GB")
            
        
        # base_vae
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            encoded = base_vae.encode(hidden_state, scale)
            base_decoded = base_vae.decode(encoded, scale)
        peak_memory = torch.cuda.max_memory_allocated(device=device)
        if rank == 0:
            print(f"base_vae_block: resolution: 1x3x9x1024x1024, time: {time.time() - start_time} sec, peak memory: {peak_memory / 2 ** 30} GB")
            
        if rank == 0:            
            print("⭕️ approximate patch with tiling decoded max error:", (standard_decoded - base_decoded).abs().max().item())
            print("⭕️ approximate patch with tiling decoded mean error:", (standard_decoded - base_decoded).abs().mean().item())
            print("⭕️ patch decoded with tiling max error:", (patch_decoded - base_decoded).abs().max().item())
            print("⭕️ patch decoded with tiling mean error:", (patch_decoded - base_decoded).abs().mean().item())


if __name__ == "__main__":
    main()