# VBIndex
Vogt-Bailey index toolbox in Python

## Installation

It is possible to simply copy the folder vb_toobox to your project folder and
proceed from there. If this is the case, be sure you have the following
packages installed

```
multiprocess
nibabel
numpy
scipy
```

The preferred way to install is through pip. It is as easy as

```bash
pip install VBIndex
```

If your pip is properly configured, you can now use the program `vb_index` from
your command line, and import any of the submodules in the `vb_toolbox` in your python 
interpreter.

## Build

We use setuptool and wheel to build the distribution code. The process is
described next. More information can be found
[here](https://packaging.python.org/tutorials/packaging-projects/).

1. Be sure that setuptools, twine, and wheel are up-to-dated

```bash
python3 -m pip install --user --upgrade setuptools wheel twine
```

2. Run the build command

```bash
python3 setup.py sdist bdist_wheel
```

3. Upload the package to pip

```bash
python3 -m twine upload dist/*
```
