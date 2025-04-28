[![All Contributors](https://img.shields.io/badge/all_contributors-8-orange.svg?style=flat-square)](#contributors-)
[![DOI](https://zenodo.org/badge/224148416.svg)](https://zenodo.org/badge/latestdoi/224148416) [![PyPI version](https://badge.fury.io/py/vb-toolbox.svg)](https://badge.fury.io/py/vb-toolbox)


# VBIndex
Vogt-Bailey index [1], [2], [3], [4] toolbox in Python

## Installation as a Python package

Clone the repository:
```bash
git clone [https://github.com/VBIndex/py_vb_toolbox.git](https://github.com/VBIndex/py_vb_toolbox.git)
cd py_vb_toolbox
````

Proceed to install the toolbox via pip from the main project directory:

```bash
pip install .
```

The installation will automatically install the dependencies specified in the `requirements.txt` file. In your terminal, check to see whether the VB toolbox has been properly installed by running:

```bash
vb_tool --help
```

If you see output similar to the following (specifically showing `vb_tool` usage and the `volumetric` subcommand), the prerequisites have been properly installed.

```
usage: vb_tool [-h] {volumetric} ...

Calculate the Vogt-Bailey Index of a dataset. For more information, refer to
[https://github.com/VBIndex/py_vb_toolbox](https://github.com/VBIndex/py_vb_toolbox).

options:
  -h, --help      show this help message and exit

VB method:
  {volumetric}  Different methods for computing the VB Index
    volumetric    Computes the VB Index on volumetric data using a searchlight
                  approach

authors:
... (rest of help message including authors, references, copyright) ...
```

## Usage

### Volumetric VB Index

This toolbox calculates the VB Index using a searchlight approach directly on volumetric data (NIfTI format). This method is referred to as "Volumetric analysis" and can be carried out with the following command:

```bash
vb_tool volumetric --data input_data/data.nii.gz --output volumetric_output
```

This is the simplest way to run this analysis. It will output two files using the base name provided with `--output` and the normalization method used (e.g., `volumetric_output.unnorm.vbi-vol.nii.gz` and `volumetric_output.unnorm.vbi-neigh.nii.gz`).

  * The `*.vbi-vol.nii.gz` file stores the calculated VB Index (or ReHo value) for each voxel.
  * The `*.vbi-neigh.nii.gz` file stores the number of neighbours included in the searchlight calculation for each voxel.

The analysis computes the index for every voxel within the NIfTI file's dimensions. This can take time. To speed this up and restrict the analysis to brain voxels, provide a volumetric brain mask:

```bash
vb_tool volumetric --data input_data/data.nii.gz --volmask input_data/volumetric_mask.nii.gz --output volumetric_output
```

#### Volumetric approach with ReHo

The VB Toolbox also supports analysing data with the Regional Homogeneity (ReHo) index [5]. The ReHo index measures the similarity between the Blood Oxygen Level Dependent (BOLD) signal of a voxel with respect to its immediate neighbors. To run ReHo for volumetric analysis:

```bash
vb_tool volumetric --data input_data/data.nii.gz --volmask input_data/volumetric_mask.nii.gz --reho --output volumetric_reho_output
```

*(Note: Output filenames will reflect the ReHo analysis, e.g., `volumetric_reho_output.unnorm.vbi-vol.nii.gz`)*

## General Notes

### Note on the data file

The `volumetric` analysis mode expects input data (`--data`) as a 4D NIfTI file (e.g., fMRI time series data in `.nii` or `.nii.gz` format). The optional mask (`--volmask`) should be a 3D NIfTI file coregistered with the data file, containing non-zero values for voxels to be included in the analysis.

### Notes on parallelism

`vb_tool` uses multiprocessing to speed up computations. The number of parallel processes (threads) can be controlled using the `-j` or `--jobs` flag when running the `volumetric` command. For example:

```bash
vb_tool volumetric --data ... --output ... --jobs 4
```

By default, the software will try to use all available CPU cores. Depending on your system and BLAS installation, adjusting the number of jobs might yield better performance. If unsure, keeping the default is usually a safe starting point.

## References

[1] C. J. Bajada et al., “A tutorial and tool for exploring feature similarity gradients with MRI data,” NeuroImage, vol. 221, pp. 117140–117140, Jul. 2020, doi: https://doi.org/10.1016/j.neuroimage.2020.117140.

[2] C. Farrugia, P. Galdi, Irati Arenzana Irazu, K. Scerri, and C. J. Bajada, “Local gradient analysis of human brain function using the Vogt-Bailey Index,” Brain structure & function, vol. 229, no. 2, pp. 497–512, Jan. 2024, doi: https://doi.org/10.1007/s00429-023-02751-7

[3] K. G. Ciantar, C. Farrugia, P. Galdi, K. Scerri, T. Xu, and C. J. Bajada, “Geometric effects of volume-to-surface mapping of fMRI data,” Brain Structure and Function, vol. 227, no. 7, pp. 2457–2464, Jul. 2022, doi: https://doi.org/10.1007/s00429-022-02536-4.

[4] K. Galea. A. A. Escudero, N. A. Montalto, N. Vella, R. E. Smith, C. Farrugia, P. Galdi, K. Scerri, L. Butler, and C. J. Bajada, “Testing the Vogt-Bailey Index using task-based fMRI across pulse sequence protocols,” bioRxiv, pp. 2025-02, 2025.

[5] Y. Zang, T. Jiang, Y. Lu, Y. He, and L. Tian, “Regional homogeneity approach to fMRI data analysis,” NeuroImage, vol. 22, no. 1, pp. 394–400, May 2004, doi: https://doi.org/10.1016/j.neuroimage.2003.12.030.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

\<table\>
\<tbody\>
\<tr\>
\<td align="center" valign="top" width="14.28%"\>\<a href="https://github.com/Aitor-Alberdi"\>\<img src="https://avatars.githubusercontent.com/u/152187460?v=4?s=100" width="100px;" alt="Aitor-Alberdi"/\>\<br /\>\<sub\>\<b\>Aitor-Alberdi\</b\>\</sub\>\</a\>\<br /\>\<a href="https://github.com/VBIndex/py\_vb\_toolbox/commits?author=Aitor-Alberdi" title="Code"\>💻\</a\> \<a href="\#maintenance-Aitor-Alberdi" title="Maintenance"\>🚧\</a\>\</td\>
\<td align="center" valign="top" width="14.28%"\>\<a href="https://github.com/KeithGeorgeCiantar"\>\<img src="https://avatars1.githubusercontent.com/u/52758149?v=4?s=100" width="100px;" alt="Keith George Ciantar"/\>\<br /\>\<sub\>\<b\>Keith George Ciantar\</b\>\</sub\>\</a\>\<br /\>\<a href="https://github.com/VBIndex/py\_vb\_toolbox/commits?author=KeithGeorgeCiantar" title="Code"\>💻\</a\>\</td\>
\<td align="center" valign="top" width="14.28%"\>\<a href="https://github.com/NicoleEic"\>\<img src="https://avatars3.githubusercontent.com/u/25506847?v=4?s=100" width="100px;" alt="NicoleEic"/\>\<br /\>\<sub\>\<b\>NicoleEic\</b\>\</sub\>\</a\>\<br /\>\<a href="https://github.com/VBIndex/py\_vb\_toolbox/commits?author=NicoleEic" title="Code"\>💻\</a\>\</td\>
\<td align="center" valign="top" width="14.28%"\>\<a href="http://claude.bajada.info"\>\<img src="https://avatars3.githubusercontent.com/u/16142659?v=4?s=100" width="100px;" alt="claudebajada"/\>\<br /\>\<sub\>\<b\>claudebajada\</b\>\</sub\>\</a\>\<br /\>\<a href="https://github.com/VBIndex/py\_vb\_toolbox/issues?q=author%3Aclaudebajada" title="Bug reports"\>🐛\</a\> \<a href="\#ideas-claudebajada" title="Ideas, Planning, & Feedback"\>🤔\</a\> \<a href="\#projectManagement-claudebajada" title="Project Management"\>📆\</a\> \<a href="https://github.com/VBIndex/py\_vb\_toolbox/commits?author=claudebajada" title="Code"\>💻\</a\>\</td\>
\<td align="center" valign="top" width="14.28%"\>\<a href="https://github.com/LucasCampos"\>\<img src="https://avatars1.githubusercontent.com/u/2735358?v=4?s=100" width="100px;" alt="Lucas Campos"/\>\<br /\>\<sub\>\<b\>Lucas Campos\</b\>\</sub\>\</a\>\<br /\>\<a href="https://github.com/VBIndex/py\_vb\_toolbox/commits?author=LucasCampos" title="Code"\>💻\</a\> \<a href="https://github.com/VBIndex/py\_vb\_toolbox/issues?q=author%3ALucasCampos" title="Bug reports"\>🐛\</a\> \<a href="\#ideas-LucasCampos" title="Ideas, Planning, & Feedback"\>🤔\</a\> \<a href="\#maintenance-LucasCampos" title="Maintenance"\>🚧\</a\>\</td\>
\<td align="center" valign="top" width="14.28%"\>\<a href="https://github.com/paola-g"\>\<img src="https://avatars.githubusercontent.com/u/7580862?v=4?s=100" width="100px;" alt="paola-g"/\>\<br /\>\<sub\>\<b\>paola-g\</b\>\</sub\>\</a\>\<br /\>\<a href="https://github.com/VBIndex/py\_vb\_toolbox/commits?author=paola-g" title="Code"\>💻\</a\>\</td\>
\<td align="center" valign="top" width="14.28%"\>\<a href="https://github.com/ChristineFarrugia"\>\<img src="https://avatars.githubusercontent.com/u/83232978?v=4?s=100" width="100px;" alt="ChristineFarrugia"/\>\<br /\>\<sub\>\<b\>ChristineFarrugia\</b\>\</sub\>\</a\>\<br /\>\<a href="https://github.com/VBIndex/py\_vb\_toolbox/commits?author=ChristineFarrugia" title="Code"\>💻\</a\>\</td\>
\</tr\>
\<tr\>
\<td align="center" valign="top" width="14.28%"\>\<a href="https://github.com/jschewts"\>\<img src="https://avatars.githubusercontent.com/u/68106439?v=4?s=100" width="100px;" alt="jschewts"/\>\<br /\>\<sub\>\<b\>jschewts\</b\>\</sub\>\</a\>\<br /\>\<a href="https://github.com/VBIndex/py\_vb\_toolbox/commits?author=jschewts" title="Code"\>💻\</a\>\</td\>
\<td align="center" valign="top" width="14.28%"\>\<a href="http://www.kscerri.com/Personal/index.html"\>\<img src="https://avatars.githubusercontent.com/u/153515?v=4?s=100" width="100px;" alt="Kenneth Scerri"/\>\<br /\>\<sub\>\<b\>Kenneth Scerri\</b\>\</sub\>\</a\>\<br /\>\<a href="\#projectManagement-kscerri" title="Project Management"\>📆\</a\>\</td\>
\</tr\>
\</tbody\>
\</table\>

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome\!