# paravae/setup.py
from setuptools import find_packages, setup


if __name__ == "__main__":
    setup(
        name="paravae",
        packages=find_packages(),
        install_requires=["torch>=2.5.1", "diffusers>=0.34.0.dev0", "einops>=0.8.1", "imageio", "imageio-ffmpeg"],
        description="ParaVAE: A Parallelism Distributed 2D/3D VAE for Efficient VAE Training and Inference with Slicing & Tiling Optimization",
        long_description=None,
        long_description_content_type="text/markdown",
        version="0.1.0",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.10",
    )
