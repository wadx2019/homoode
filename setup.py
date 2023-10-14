import os
import re
import setuptools


# for simplicity we actually store the version in the __version__ attribute in the source
here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, 'torchdiffeq', '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")


setuptools.setup(
    name="torchdiffeq",
    version=version,
    author="dingsht",
    author_email="dingsht@shanghaitech.edu.cn",
    description="Modified version of torchdiffeq for HomoODE.",
    url="https://github.com/wadx2019/homodiffeq",
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.5.0', 'scipy>=1.4.0'],
    python_requires='~=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)