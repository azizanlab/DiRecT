from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="diffusion_policy",
    version="1.0.0",
    description="Diffusion Policy for PCD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hao Luan",
    author_email="haoluan@comp.nus.edu.sg",
    packages=find_packages(include=["diffusion_policy", "diffusion_policy.*"]), 
    install_requires=[],
)
