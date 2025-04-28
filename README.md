<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![All Contributors](https://img.shields.io/badge/all_contributors-8-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![DOI](https://zenodo.org/badge/224148416.svg)](https://zenodo.org/badge/latestdoi/224148416) [![PyPI version](https://badge.fury.io/py/vb-toolbox.svg)](https://badge.fury.io/py/vb-toolbox)


# VBIndex
Vogt-Bailey index [1], [2], [3], [4] toolbox in Python

## Installation as a Python package

Clone the repository:
```bash
git clone [text](https://github.com/VBIndex/py_vb_toolbox.git)
```

Proceed to install the toolbox via pip.
```bash
pip install py_vb_toolbox/
```

The installation will automatically install the dependancies specified in the `requirements.txt` file. In your terminal, check to see whether the VB toolbox has been properly installed by running:

```bash
vb_tool
```

If you see the following output, the pre-requisites have been properly installed.

```
usage: app.py [-h] {volumetric} ...

Calculate the Vogt-Bailey index of a dataset. For more information, refer to
https://github.com/VBIndex/py_vb_toolbox.

options:
  -h, --help  show this help message and exit

VB method:
  {volumetric}    Different methods for computing the VB Index
    volumetric    Computes the VB Index on volumetric data using a searchlight approach

authors:

The VB Index Team (See Contributors Section in the main README)

references:

Bajada, C. J., Campos, L. Q. C., Caspers, S., Muscat, R., Parker, G. J., Ralph, M. A. L., ... & Trujillo-
Barreto, N. J. (2020). A tutorial and tool for exploring feature similarity gradients with MRI data.
NeuroImage, 221, 117140.

Ciantar, K. G., Farrugia, C., Galdi, P., Scerri, K., Xu, T., & Bajada, C. J. (2022). Geometric effects of
volume-to-surface mapping of fMRI data. Brain Structure and Function, 227(7), 2457-2464.

Farrugia, C., Galdi, P., Irazu, I. A., Scerri, K., & Bajada, C. J. (2024). Local gradient analysis of human
brain function using the Vogt-Bailey Index. Brain Structure and Function, 229(2), 497-512.

Galea, K., Escudero, A. A., Montalto, N. A., Vella, N., Smith, R. E., Farrugia, C., ... & Bajada, C. J. (2025). 
Testing the Vogt-Bailey Index using task-based fMRI across pulse sequence protocols. bioRxiv, 2025-02.

copyright:

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see
<https://www.gnu.org/licenses>.
```

## Usage

### 1.1 Volumetric VB Index

The VB Index has been tested using a searchlight analysis with no surface mapping. This method for running the VB Index
is referred to as "Volumetric analysis" and can be carried out with the following command:

```bash
vb_tool volumetric --data input_data/data.nii.gz --output volumetric_output
```

This is the simplest way to run this analysis. It will output two files, ```-vol.nii.gz``` and ```-neigh.nii.gz```. The first file is going to store the information from the analysis. On the other hand, the second file is going to store the neighbourhood used for every voxel, so the output should be a cube.

This may take a while since the tool will try to compute the VB index for every voxel and for every voxel both inside and outside of the brain. In order to speed this up, a volumetric mask can be specified:

```bash
vb_tool volumetric --data input_data/data.nii.gz --volmask input_data/volumetric_mask.nii.gz --output volumetric_output
```

#### Volumetric approach with ReHo

The VB Toolbox also supports analysing data with the Regional Homogeneity (ReHo) index [5]. The ReHo index measures the similarity between the Blood Oxygen Level Dependent (BOLD) signal of a voxel with respect to its immediate neighbors. To run ReHo for volumetric analysis:

```bash
vb_tool volumetric --data input_data/data.nii.gz --volmask input_data/volumetric_mask.nii.gz --reho --output volumetric_output
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

## References
[1] C. J. Bajada et al., “A tutorial and tool for exploring feature similarity gradients with MRI data,” NeuroImage, vol. 221, pp. 117140–117140, Jul. 2020, doi: https://doi.org/10.1016/j.neuroimage.2020.117140.

[2] C. Farrugia, P. Galdi, Irati Arenzana Irazu, K. Scerri, and C. J. Bajada, “Local gradient analysis of human brain function using the Vogt-Bailey Index,” Brain structure & function, vol. 229, no. 2, pp. 497–512, Jan. 2024, doi: https://doi.org/10.1007/s00429-023-02751-7

[3] K. G. Ciantar, C. Farrugia, P. Galdi, K. Scerri, T. Xu, and C. J. Bajada, “Geometric effects of volume-to-surface mapping of fMRI data,” Brain Structure and Function, vol. 227, no. 7, pp. 2457–2464, Jul. 2022, doi: https://doi.org/10.1007/s00429-022-02536-4.

[4] K. Galea. A. A. Escudero, N. A. Montalto, N. Vella, R. E. Smith, C. Farrugia, P. Galdi, K. Scerri, L. Butler, and C. J. Bajada, “Testing the Vogt-Bailey Index using task-based fMRI across pulse sequence protocols,”  bioRxiv, pp. 2025-02, 2025.

[5] Y. Zang, T. Jiang, Y. Lu, Y. He, and L. Tian, “Regional homogeneity approach to fMRI data analysis,” NeuroImage, vol. 22, no. 1, pp. 394–400, May 2004, doi: https://doi.org/10.1016/j.neuroimage.2003.12.030.


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Aitor-Alberdi"><img src="https://avatars.githubusercontent.com/u/152187460?v=4?s=100" width="100px;" alt="Aitor-Alberdi"/><br /><sub><b>Aitor-Alberdi</b></sub></a><br /><a href="https://github.com/VBIndex/py_vb_toolbox/commits?author=Aitor-Alberdi" title="Code">💻</a> <a href="#maintenance-Aitor-Alberdi" title="Maintenance">🚧</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

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
