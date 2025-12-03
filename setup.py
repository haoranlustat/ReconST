from setuptools import setup, find_packages

setup(
    name="reconst",
    version="0.1.0",
    description="Gene panel selection for spatial transcriptomics",
    author="ReconST Team",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "scanpy>=1.8.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.7",
)
