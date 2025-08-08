# ParaVAE: A Parallelism Distributed 3D VAE for Efficient VAE Training and Inference with Slicing & Tiling Optimization
ParaVAE is a high-performance distributed framework designed to accelerate 3D VAE training and inference in large-scale generative AI workflows. Built for modern multi-GPU computing environments, it reduces the memory footprint of the image and video model training and generating process. The framework excels in applications like diffusion models, video generation, with native support for WAN2.1 VAE and modular extensibility.

# Installation
```bash
git clone https://github.com/RiseAI-Sys/ParaVAE.git
cd ParaVAE
pip install -e .
```

# Usage
1. Evaluate the peak GPU memory consumption when training (with grad) of base vae, approximate patch vae (without halo exchange), and patch vae (with halo exchange) with 2 GPU.
```bash
torchrun --nproc_per_node=2 --master-port=29501 test/WAN2_1/test_vae.py --memory_test
```

2. Evaluate the peak GPU memory consumption and inference time when inferencing for video generation (without grad) of base vae, base vae with tiling, approximate patch vae (without halo exchange), approximate patch vae with tiling, patch vae (with halo exchange), and patch vae with tiling, with 2 GPU.
```bash
cd resources
wget https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B/resolve/master/Wan2.1_VAE.pth
cd ..
torchrun --nproc_per_node=2 --master-port=29501 test/WAN2_1/test_vae_video.py 
```

# Acknowledgement
We learned the design and resued the code from the following projects: [Wan2.1](https://github.com/Wan-Video/Wan2.1), [DistVAE](https://github.com/xdit-project/DistVAE), and [Diffusers](https://github.com/huggingface/diffusers).    