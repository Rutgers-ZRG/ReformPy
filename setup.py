from setuptools import setup, find_packages

setup(
    name="reformpy",
    version="1.3.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.4',
        'scipy>=1.10.1',
        'numba>=0.58.1',
        'ase>=3.22.1',
        'libfp>=3.1.2',
        'mpi4py>=3.1.0,<4.0',
    ],
    author="Zhu Research Group",
    author_email="li.zhu@rutgers.edu",
    description="Rational Exploration of Fingerprint-Oriented Relaxation Methodology",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Rutgers-ZRG/ReformPy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
