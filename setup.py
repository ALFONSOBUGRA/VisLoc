from setuptools import setup, find_packages

setup(
    name="visloc",
    version="0.1.0",
    description="Visual Localization Package",
    author="Hamit Buğra Bayram",
    author_email="hamitbugrabayram@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "tqdm>=4.66.2",
        "scipy>=1.13.0",
        "opencv-python==4.5.5.64",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "ipdb>=0.13.0",
            "wandb>=0.13.5",
            "matplotlib>=3.6.2",
            "ipywidgets>=8.0.4",
            "jupyterlab>=3.6.1",
            "seaborn>=0.12.2",
            "tensorboard>=2.15.0",
            "tensorboardx>=2.6.2.2",
            "rich>=13.6.0",
            "plotly>=5.18.0",
            "rasterio>=1.3.9",
        ],
        "lint": [
            "black[jupyter]>=23.7.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
            "pre-commit>=2.16.0",
        ],
        "test": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.8.0",
        ],
        "docs": [
            "Sphinx>=7.3.7",
            "myst-nb>=1.0.0",
            "sphinx-autoapi>=1.8.0",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "visloc=main:main",
        ],
    },
) 