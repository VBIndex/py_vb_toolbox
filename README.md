<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-8-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![DOI](https://zenodo.org/badge/224148416.svg)](https://zenodo.org/badge/latestdoi/224148416) [![PyPI version](https://badge.fury.io/py/vb-toolbox.svg)](https://badge.fury.io/py/vb-toolbox)


# VBIndex
Vogt-Bailey index toolbox in Python

## Installation

It is possible to simply copy the folder `vb_toobox` to your project folder and
proceed from there. If you take this approach, be sure you have the following
packages installed

```
argparse
numpy
scipy
nibabel
multiprocess
textwrap
sys
warnings
traceback
glob
os
signal
shutil
```

The preferred way to install is through pip. It is as easy as

```bash
pip install vb_toolbox
```

If your pip is properly configured, you can now use the program `vb_tool` from
your command line, and import any of the submodules in `vb_toolbox` into your python
interpreter.

## Usage of `vb_tool` CLI

If the VBIndex toolbox was installed from PyPI via `pip`, the command line program `vb_tool` should
be available in your terminal. You can test if the program is correctly
installed by typing

```bash
vb_tool -h
```
Alternatively, if you have downloaded the source code, you can install the program by
typing

```bash
pip install py_vb_toolbox/
```

In your terminal, if you see the following output, the program has been
properly installed.

```
usage: vb_tool [-h] [-j N] [-n norm] [-fb] [-hy] [-m file] [-c file]
               [-t tolerance] [-mi max iterations] [-debug] -s file -d file -o
               file

Calculate the Vogt-Bailey index of a dataset. For more information, refer to
https://github.com/VBIndex/py_vb_toolbox.

optional arguments:
  -h, --help            Show this help message and exit.
  -j N, --jobs N        Maximum number of jobs to be used. If absent, one job
                        per CPU will be spawned.
  -n norm, --norm norm  Laplacian normalization to be employed. Possibilities are
                        "geig", "unnorm", "rw" and "sym". Defaults to geig for
                        the full brain and ROI analyses, and to unnorm
                        otherwise.
  -fb, --full-brain     Calculate full brain feature gradient analysis.
  -hy, --hybrid         Calculate searchlight VB index with hybrid approach.
  -rh, --reho           Calculate the KCC index for ReHo approach.
  -vol, --volumetric    Calculate VB index based on volumetric data.
  -ta, --temporal-analysis
                        Calculate the time varying VB index of a time window.
  -ws, --window-size    Window size for Temporal Analysis.
  -st, --step           Step for Temporal Analysis.
  -sz, --size           Size for Temporal Analysis.
  -p, --path            Path for temporal folder.
  -vm, --volmask        Nifti file containing the whole brain mask
                        in volumetric space. Only relevant if computing the 
                        volumetric VB.
  -m file, --mask file  File containing the labels to identify the cortex,
                        rather than the medial brain structures. This flag
                        must be set for the searchlight and full brain
                        analyses.
  -c file, --clusters file
                        File specifying the surface clusters. The cluster with
                        index 0 is expected to denote the medial brain
                        structures and will be ignored.
  -t tolerance, --tol tolerance
                        Residual tolerance (stopping criterion) for LOBPCG.
                        Default value = sqrt(10e-18)*n, where n is the number
                        of nodes per graph. Note that the LOBPCG algorithm is only
                        utilised for full-brain analysis.
  -mi max iterations, --maxiter max iterations
                        Maximum number of iterations for LOBPCG. Defaults to
                        50.
  -debug, --debug       Save additional files for debugging.

required named arguments:
  -s file, --surface file
                        File containing the surface mesh.
  -d file, --data file  File containing the data over the surface (or volume
                        if hybrid).
  -o file, --output file
                        Base name for the output files.

authors:

The VB Index Team (See Contributors Section in the main README)

references:

Bajada, C. J., Campos, L. Q. C., Caspers, S., Muscat, R., Parker, G. J., Ralph, M. A. L., ... & Trujillo-Barreto, N. J. (2020). A tutorial and tool for exploring feature similarity gradients with MRI data. NeuroImage, 221, 117140.
Ciantar, K. G., Farrugia, C., Scerri, K., Xu, T., & Bajada, C. J. (2020). Geometric effects of volume-to-surface mapping of fMRI data. bioRxiv.

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

There are five main uses for the `vb_tool`

1. **Searchlight analysis** 
    1. **Surface searchlight** (Searchlight analysis)  
    2. **Hybrid approach** (Hybrid analysis)
    3. **No surface mapping approach** (Volumetric analysis)
    4. **Over time approach** (Temporal analysis)
3. **Full brain feature gradient analysis** (Full Brain analysis)
4. **Feature gradient analysis in a specified set of regions of interest** (ROI analysis)

### 1. Searchlight

#### 1.1 Surface searchlight
The original analysis with surface mapping is called Searchlight analysis. However, it is currently not recommended to be used. The command to use is the following one:

```bash
vb_tool --surface input_data/surface.surf.gii  --data input_data/data.nii --mask input_data/cortical_mask.shape.gii --output search_light
```

Other analysis methods based on the searchlight approach are the following ones

#### 1.2 Hybrid

The per vertex VB-index analysis can be carried out with the following command

```bash
vb_tool --hybrid --surface input_data/surface.surf.gii  --data input_data/data.nii --mask input_data/cortical_mask.shape.gii --output search_light
```

The number of vertices in the surface mesh must match the number of entries in
the mask.

The cortical mask must contain a logical array, with `True` values in the
region on which the analysis will be carried out, and `False` in the regions to
be left out. This is most commonly used to mask out midbrain structures which
would otherwise influence the analysis of the cortical regions.

#### 1.3 Volumetric

In order to run the Searchlight analysis with no surface mapping can be carried out with the following command:

```bash
vb_tool --data input_data/data.nii.gz -vol --output volumetric_output
```

This is the simplest way to run this analysis. It will output two files, -vol.nii.gz and -neigh.nii.gz. The first file is going to store the information from the analysis. On the other hand, the second file is going to store the neighbourhood used for every voxel, so the output should be a cube.

This may take a while since the tool will try to compute the VB index for every voxel and for every voxel both inside and outside of the brain. In order to speed this up, a volumetric mask can be specified:

```bash
vb_tool --data input_data/data.nii.gz --volmask input_data/volumetric_mask.nii.gz --volume --output volumetric_output
```

The mask is an optional parameter that can be specified in the Volumetric and Temporal analysis. 

#### 1.4 Temporal

The purpose of this analysis is to see how the VB index changes over time. For this, a window size is specified, then, an average volume is calculated and the VB index is computed. This analysis can be run along Hybrid or Volumetric analysis: 

```bash
vb_tool --data input_data/data.nii.gz --mask input_data/volumetric_mask.nii.gz --window-size 10 --temporal-analysis [--hybrid/--volume] --output temporal_analysis
```

It is a requirement to choose one of the arguments within square brackets. For example, lets say that the first 5 volumes are taken, for the first iteration, volumes 1,2,3,4,5 are going to be taken into account to compute the VB index, and for the next iteration, volumes 2,3,4,5,6 are going to be taken into account. So, the increment or step by default is 1, however, this can be specified using the step parameter:

```bash
vb_tool --data input_data/data.nii.gz --mask input_data/volumetric_mask.nii.gz --step 10 --temporal-analysis [--hybrid/--volume] --output temporal_analysis
```

It is recommended that the window size is a divisor of the number of volumes that exist in the data. If it is not a divisor, that means that for the last iterations, the window size will be smaller than the one specified due to lack of volumes. At that point, the program will not continue calculating and it will directly output the results.

### 2. Full Brain

To perform full brain feature gradient analysis and extract the associated VB index, the flag 
`-fb` or `--full-brain` must be set instead of `--hybrid`. Otherwise, the flags are the same as for the hybrid analysis.

```bash
vb_tool --surface input_data/surface.surf.gii  --data input_data/data.func.gii --mask input_data/cortical_mask.shape.gii --full-brain --output full_brain_gradient
```

Be warned, however, that this analysis can take long and require a large amount of
RAM. For data sets with 32k vertices, upwards of 30GB of RAM were used.
### 3. Regions of Interest (ROI) analysis

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

## ReHo

The VB Toolbox also supports analysing data with the Regional Homogeneity (ReHo) index. The ReHo index measures the similarity between the Blood Oxygen Level Dependent (BOLD) signal of a voxel with respect to its immediate neighbors.

This can be used along Hybrid, Volumetric or Temporal analysis. For this next example, the command for Volumetric analysis will be used:

```bash
vb_tool --data input_data/data.nii.gz --volmask input_data/volumetric_mask.nii.gz --volume --reho --output volumetric_output
```

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

1. Full brain analysis: Low level of parallelism. Will only spawn one job
2. Region of Interest (ROI) analysis: Medium level of parallelism. Will spawn as many
   jobs as there are ROIs, or number of CPUS, whichever is the lowest.
3. Any other analysis: High level of parallelism. Will spawn as many jobs as
   there are CPUs

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
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/KeithGeorgeCiantar"><img src="https://avatars1.githubusercontent.com/u/52758149?v=4?s=100" width="100px;" alt="Keith George Ciantar"/><br /><sub><b>Keith George Ciantar</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=KeithGeorgeCiantar" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/NicoleEic"><img src="https://avatars3.githubusercontent.com/u/25506847?v=4?s=100" width="100px;" alt="NicoleEic"/><br /><sub><b>NicoleEic</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=NicoleEic" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://claude.bajada.info"><img src="https://avatars3.githubusercontent.com/u/16142659?v=4?s=100" width="100px;" alt="claudebajada"/><br /><sub><b>claudebajada</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/issues?q=author%3Aclaudebajada" title="Bug reports">🐛</a> <a href="#ideas-claudebajada" title="Ideas, Planning, & Feedback">🤔</a> <a href="#projectManagement-claudebajada" title="Project Management">📆</a> <a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=claudebajada" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/LucasCampos"><img src="https://avatars1.githubusercontent.com/u/2735358?v=4?s=100" width="100px;" alt="Lucas Campos"/><br /><sub><b>Lucas Campos</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=LucasCampos" title="Code">💻</a> <a href="https://github.com/VBIndex/py_vb_toolbox/issues?q=author%3ALucasCampos" title="Bug reports">🐛</a> <a href="#ideas-LucasCampos" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-LucasCampos" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/paola-g"><img src="https://avatars.githubusercontent.com/u/7580862?v=4?s=100" width="100px;" alt="paola-g"/><br /><sub><b>paola-g</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=paola-g" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ChristineFarrugia"><img src="https://avatars.githubusercontent.com/u/83232978?v=4?s=100" width="100px;" alt="ChristineFarrugia"/><br /><sub><b>ChristineFarrugia</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=ChristineFarrugia" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jschewts"><img src="https://avatars.githubusercontent.com/u/68106439?v=4?s=100" width="100px;" alt="jschewts"/><br /><sub><b>jschewts</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=jschewts" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://www.kscerri.com/Personal/index.html"><img src="https://avatars.githubusercontent.com/u/153515?v=4?s=100" width="100px;" alt="Kenneth Scerri"/><br /><sub><b>Kenneth Scerri</b></sub></a><br /><a href="#projectManagement-kscerri" title="Project Management">📆</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
