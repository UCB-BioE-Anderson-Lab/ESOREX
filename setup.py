from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="esorex",
    version="0.1.0-beta",
    author="J. Christopher Anderson",
    author_email="jcanderson@berkeley.edu",
    description="Interpretable per-enzyme models of substrate specificity using EVODEX abstractions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UCB-BioE-Anderson-Lab/ESOREX",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(),
    package_data={"esorex": ["data/*"]},
    install_requires=[
        "rdkit",
        "pandas",
        "numpy<2",
        "networkx",
        "scikit-learn",
        # Pin to 2.1.0: corrected reactive-center logic; 2.1.1 reverted it for notebook compat.
        "evodex==2.1.0",
        "CGRtools==4.0.41",
    ],
)
