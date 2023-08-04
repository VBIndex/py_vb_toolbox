
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![DOI](https://zenodo.org/badge/224148416.svg)](https://zenodo.org/badge/latestdoi/224148416) [![PyPI version](https://badge.fury.io/py/vb-toolbox.svg)](https://badge.fury.io/py/vb-toolbox)


# VBIndex
Vogt-Bailey index toolbox in Python

## Installation

It is possible to simply copy the folder `vb_toolbox` to your project folder and
proceed from there. If you take this approach, be sure you have the following
packages installed

```
multiprocess
nibabel
numpy
scipy
pillow
psutil
```

The preferred way to install the toolbox is by cloning this branch (using `git clone -b Local-gradients-paper` followed by the repository url: https://github.com/VBIndex/py_vb_toolbox)
into a folder called `vb_gradients` and then simply typing (from inside `vb_gradients`)

```bash
pip3 install py_vb_toolbox/
```

after which you should be able to use the program `vb_tool` from
your command line, and import any of the submodules in `vb_toolbox` into your python
interpreter.

## Usage of `vb_tool` CLI

If the VBIndex toolbox was installed as described above, the command line program `vb_tool` should
be available in your terminal. You can test if the program is correctly
installed by typing

```bash
vb_tool -h
```

In your terminal, if you see the following output, the program has been
properly installed.

```
usage: vb_tool [-h] [-j N] [-n norm] [-fb] [-hy] [-vm file] [-m file] [-c file] -s file -d file
              -o file

Calculate the Vogt-Bailey index of a dataset. For more information, check
https://github.com/VBIndex/py_vb_toolbox.

optional arguments:
  -h, --help            show this help message and exit
  -j N, --jobs N        Maximum number of jobs to be used. If abscent, one job
                        per CPU will be spawned
  -n norm, --norm norm  Laplacian normalization to be used. Possibilities are
                        "geig", "unnorm", "rw" and "sym". Defaults to geig.
  -fb, --full-brain     Calculate full brain feature gradient analyses.
  -hy, --hybrid         Calculate VB index with hybrid approach.
  -vm file, --volmask file
                        Nifti file containing the whole brain mask in volumetric space. This flag must be set if computing hybrid VB
  -m file, --mask file  File containing the labels to identify the cortex,
                        rather than the medial brain structures. This flag
                        must be set for normal analyses and full brain
                        analyses.
  -c file, --clusters file
                        File containing the surface clusters. Cluster with
                        index 0 are expected to denote the medial brain
                        structures and will be ignored.

required named arguments:
  -s file, --surface file
                        File containing the surface mesh
  -d file, --data file  File containing the data over the surface
  -o file, --output file
                        Base name for the output files

authors:

Lucas da Costa Campos (lqccampos (at) gmail.com) and Claude J Bajada
(claude.bajada (at) um.edu.mt).

copyright:

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses>.
```

If you copied the program source code, the executable is found in `vb_toolbox/app.py`.
You can test the program using

```bash
python3 vb_toolbox/app.py
```

There are three main uses for the `vb_tool`

1. Searchlight analysis (hybrid)
2. Whole brain feature gradient analysis
3. Feature gradient analysis in a specified set of regions of interest

We currently recommend using the hybrid searchlight analysis as it is the most tested approach. 

### Searchlight analysis

This is the method employed in the paper 'Local gradient analysis of human brain function using the Vogt-Bailey Index' [https://www.biorxiv.org/content/10.1101/2022.10.14.511925v3]. It represents a per vertex VB-index analysis that can be carried out with the following command

```bash
vb_tool --hybrid --surface input_data/surface.surf.gii  --data input_data/data.nii --mask input_data/cortical_mask.shape.gii --volmask input_data/volumetric_mask.nii --output search_light
```

The number of vertices in the surface mesh must match the number of entries in
the cortical mask.

The cortical mask must contain a logical array, with `True` values in the
region on which the analysis will be carried out, and `False` in the regions to
be left out. This is most commonly used to mask out midbrain structures which
would otherwise influence the analysis of the cortical regions.

To evaluate the ReHo metric [Y. Zang et al. NeuroImage, 22 (1):394, 2004. doi: 10.1016/j.neuroimage.2003.12.030.] at each vertex instead of the VB index, rename the file `vb_index_ReHo.py` to `vb_index.py` and place it inside `vb_toolbox` instead of the original `vb_index.py`. This should be done prior to installation. The command to be executed in this case is identical to the one given above.  

### Whole brain analysis

To perform full brain feature gradient analysis and extract the associated VB index, the flag 
`-fb` or `--full-brain` must be set instead of `--hybrid`. Otherwise, the flags are the same as for the searchlight analysis.

```bash
vb_tool --surface input_data/surface.surf.gii  --data input_data/data.func.gii --mask input_data/cortical_mask.shape.gii --full-brain --output full_brain_gradient
```

Be warned, however, that this analysis can take long and require a large amount of
RAM. For data sets with 32k vertices, upwards of 30GB of RAM were used.

### Regions of Interest analysis

Sometimes, one is interested only in a small set of ROIs. In this case, the
feature gradient map and the associated VB index value for each ROI will be
extracted. The way of calling the program is as follows:

```bash
vb_tool --surface input_data/surface.surf.gii  --data input_data/data.func.gii  -c input_data/clusters.shape.gii --output clustered_analysis
```

The cluster file works similarly to the cortical mask employed for the searchlight and full brain methods. 
However, its structure is slightly different. Instead of an array
of logical values, the file must contain an array of integers, where each
integer corresponds to a different cluster. The 0th cluster is special, and
denotes an area which will *not* be analyzed. In this regard, it has a
similar use to the cortical mask.

## General Notes

### Note on the data file

`vb_tool` can handle two separate cases. If there is a single structure in the
file, `vb_tool` will read it as a matrix in which each row relates to a specific
vertex. If there are two or more structures, it will read them as a series of
column vectors in which each entry relates to a vertex. It will then coalesce
them into a single matrix, and run the analysis of all quantities concurrently.

### Notes on parallelism

`vb_tool` uses a high level of parallelism. The number of threads spawned by
`vb_tool` itself can be controlled using the `-j/--jobs` flag. By default, the software
will try to use all the CPUs in your computer at the same time to perform the
analysis. Depending on the BLAS installation on your computer, this might not
be the fastest approach, but will rarely be the slowest. If you are
unsure, keep the default number of jobs.

Due to the job structure of the `vb_tool`, the level of parallelism it can achieve
on its own depends on the specific analysis being carried out.

1. Searchlight analysis: High level of parallelism. Will spawn as many jobs as
   there are CPUs
2. Whole brain analysis: Low level of parallelism. Will only spawn one job
3. Region of Interest analysis: Medium level of parallelism. Will spawn as many
   jobs as there are ROIs, or number of CPUS, whichever is the lowest.

Especially for the whole brain analysis, having a well-optimized BLAS
installation will greatly accelerate the process, and allow for further
parallelism.  Both MKL and OpenBLAS have been shown to support fast analysis. If
you are using the Anaconda distribution, you will have a good BLAS
pre-configured.

## Developer Information

### Build

The following information is only useful for individuals who are actively
contributing to the program.

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

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/KeithGeorgeCiantar"><img src="https://avatars1.githubusercontent.com/u/52758149?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Keith George Ciantar</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=KeithGeorgeCiantar" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/NicoleEic"><img src="https://avatars3.githubusercontent.com/u/25506847?v=4?s=100" width="100px;" alt=""/><br /><sub><b>NicoleEic</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=NicoleEic" title="Code">💻</a></td>
    <td align="center"><a href="http://claude.bajada.info"><img src="https://avatars3.githubusercontent.com/u/16142659?v=4?s=100" width="100px;" alt=""/><br /><sub><b>claudebajada</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/issues?q=author%3Aclaudebajada" title="Bug reports">🐛</a> <a href="#ideas-claudebajada" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-claudebajada" title="Project Management">📆</a> <a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=claudebajada" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/LucasCampos"><img src="https://avatars1.githubusercontent.com/u/2735358?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Lucas Campos</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=LucasCampos" title="Code">💻</a> <a href="https://github.com/VBIndex/py_vb_toolbox/issues?q=author%3ALucasCampos" title="Bug reports">🐛</a> <a href="#ideas-LucasCampos" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-LucasCampos" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/paola-g"><img src="https://avatars.githubusercontent.com/u/7580862?v=4?s=100" width="100px;" alt=""/><br /><sub><b>paola-g</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=paola-g" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
