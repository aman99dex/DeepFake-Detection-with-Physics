from setuptools import setup, find_packages

setup(
    name="physforensics",
    version="2.0.0",
    description="Deepfake Detection via Physics-Based Neural Inverse Rendering",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="PhysForensics Research",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.66.0",
        "einops>=0.7.0",
        "kornia>=0.7.0",
        "timm>=0.9.0",
        "facenet-pytorch>=2.5.3",
        "huggingface_hub>=0.20.0",
        "flask>=3.0.0",
    ],
    extras_require={
        "video": ["opencv-python-headless>=4.8.0"],
        "tracking": ["wandb>=0.15.0", "tensorboard>=2.14.0"],
        "datasets": ["kaggle>=1.5.0", "datasets>=2.16.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
