import nibabel

def save_nifti(result, affine, header, output_name, n_neigh=None):
    """
    Saves the data from the volumetric analysis into 2 compressed files.

    Parameters
    ----------
    result : numpy array of float32 (nRows, nCols, nSlices)
        Contains the processed results from process_vb_vol_results.
    n_neigh : numpy array of float32 (nRows, nCols, nSlices)
        Contains the amount of neighbours that every voxel has.
    affine : numpy array (M, M), optional
        Matrix used to compute spatial translations and rotations.
    header : Nifty header.
        This header contains crucial information about the structure and metadata 
        associated with the image data stored in the file.
    output_name : string
        Name of the output file(s).

    Returns
    -------
    None.

    """
    
    img_new = nibabel.Nifti1Image(result, affine, header)
    nibabel.save(img_new, output_name + ".vbi-vol.nii.gz")
    if n_neigh is not None and n_neigh.any():
        img_neigh = nibabel.Nifti1Image(n_neigh, affine, header)
        nibabel.save(img_neigh, output_name + ".vbi-neigh.nii.gz")