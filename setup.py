from setuptools import find_packages, setup

setup(
    name="geographical-erasure",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "tqdm",
        "pickle5",
        "scipy",
        "pandas",
        "matplotlib",
        "torchvision",
        "setuptools",
        "torch-scatter",
    ],
    dependency_links=[],
)
