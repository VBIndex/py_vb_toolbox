from . import io_util as io
import nibabel
import numpy as np
import os
import sys

def header_check(data_path, mask_path, correct_header, output_path):
    
    """
    Parameters
    ----------
    data_path : str
        Volumetric data path
    mask_path : str
        Volumetric mask path

    Returns
    -------
    new_volmask : str
        Path to the newly created volumetric mask
    new_data : str
        Path to the newly created volumetric data

    """
    brain_mask_obj = nibabel.load(mask_path)
    vol_obj = nibabel.load(data_path)

    vol_strides = io.getStrides(vol_obj)
    mask_strides = io.getStrides(brain_mask_obj)
    vol_affine = vol_obj.affine
    mask_affine = brain_mask_obj.affine
    
    data_incorrect = False

    if not np.all(vol_strides[:3] == mask_strides):
        # Data strides not okay
        if correct_header is not None:
            converted_vol = mdf_strides_nib(data_path, vol_strides, mask_strides, output_path)
            print("The input volumetric data file has been modified so that data strides comply with input volumetric mask file")
            new_data = converted_vol
        else:
            print("The Data Strides of the inputted volumetric data and volumetric mask do not comply\nUse --correct_header flag to overcome issue")
            data_incorrect = True
    else:
        # Data strides okay
        new_data = None
            
    if not np.all(vol_affine == mask_affine):
        # Data affine not okay
        if correct_header is not None:
            regridded_mask = mdf_grid_nib(mask_path, data_path, output_path)
            print("The input volumetric mask file has been regridded so that its transform matrix complies with the input volumetric data file")
            new_volmask = regridded_mask
        else:
            print("The Transformation Matrices of the inputted volumetric data and volumetric mask do not comply\nUse --correct_header flag to overcome issue")
            data_incorrect = True
    else:
        # Data affine okay
        new_volmask = None
        
    if data_incorrect == True:
        print("Exiting...")
        sys.exit(1)
        quit()
    
    if correct_header is not None:
        print(f"Corrections performed.\nVolmuetric mask file is: {regridded_mask}\nVolumetric data file is: {converted_vol}\nContinuing with analysis...")
    
    return new_volmask, new_data

def mdf_strides_nib(input_file_path, vol_strides, mask_strides, output_path):
    
    """
    Parameters
    ----------
    input_file_path : path to the file
        to be converted
    vol_strides : numpy array (n,1)
        int64 array of (axis_length,1) size that indicate the data strides of the volumetric data
    mask_strides : numpy array (n,1)
        int64 array of (axis_length,1) size that indicate the data strides of the volumetric mask

    Returns
    -------
    output_file_path : path to
        the converted file to be processed

    """
    vol_img = nibabel.load(input_file_path)
    vol_data = vol_img.get_fdata()
    affine = vol_img.affine.copy()
    
    reordering = np.argsort(np.abs(mask_strides))
    vol_data = np.transpose(vol_data, axes=(*reordering, 3))
    
    prefix = os.path.basename(input_file_path).replace(".nii.gz","")
    output_file_path, _ = io.path_and_file_management(prefix, "convert", False, output_path)
    
    new_img = nibabel.Nifti1Image(vol_data, affine=affine, header=vol_img.header)
    nibabel.save(new_img, output_file_path)
    
    return output_file_path

def mdf_grid_nib(input_file_path, reference_file_path, output_path):
    
    """  
    Parameters
    ----------
    input_file_path : path to the file
        to be regridded
    reference_file_path : path to the file
        to use as transformation matrix reference

    Returns
    -------
    output_file_path : path to
        the regridded file to be processed

    """
    ref_img = nibabel.load(reference_file_path)
    input_img = nibabel.load(input_file_path)
    
    input_data = input_img.get_fdata()
    ref_affine = ref_img.affine
    ref_shape = ref_img.shape[:3]
    
    input_affine = input_img.affine
    transformation_matrix = np.linalg.inv(ref_affine) @ input_affine
    
    resampled_input_data = io.affine_transform(input_data, transformation_matrix[:3, :3], offset=transformation_matrix[:3, 3], output_shape=ref_shape, order=0) #order=0 is nearest neighbor interp
    
    prefix = os.path.basename(input_file_path).replace(".nii.gz","")
    output_file_path, _ = io.path_and_file_management(prefix, "regrid", False, output_path)
    
    regridded_img = nibabel.Nifti1Image(resampled_input_data, ref_affine, header=input_img.header)
    nibabel.save(regridded_img, output_file_path)
    
    return output_file_path

def getStrides(obj):
    
    """    
    Parameters
    ----------
    obj : obj
        nibabel object from wich data strides of the data in it to be extracted.

    Returns
    -------
    data_strides : numpy array (n,1)
        int64 array of (axis_length,1) size that indicate the data strides

    """
    axlen = len(obj.shape)
    affine = obj.affine
    voxel_to_world = affine[:axlen, :axlen]
    stride_directions = np.sign(np.diag(voxel_to_world)).astype(int)
    axis_order = np.argsort(-np.abs(np.diag(voxel_to_world))) + 1
    data_strides = stride_directions * axis_order
    
    return data_strides