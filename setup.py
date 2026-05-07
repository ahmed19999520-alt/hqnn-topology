from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="hqnn-topology",
    version="1.0.0",
    author="HQNN Research Group",
    author_email="research@hqnn.io",
    description="Hyperconnected Quantum Neural Network with Topological Error Correction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hqnn-topology/hqnn-topology",
    packages=find_packages(exclude=["tests*", "notebooks*", "examples*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.0",
            "mypy>=1.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "tensorflow[and-cuda]>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hqnn-demo=examples.run_full_pipeline:main",
        ],
    },
)