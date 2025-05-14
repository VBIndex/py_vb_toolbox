import nibabel
#mdf
import os
import re
from pathlib import Path
#mdf_end

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
        
#mdf

def path_and_file_management(prefix,operation,logbool,output_path):
    """
    Parameters
    ----------
    operation : str
        String representing the operation that is carried out after the use of the function 
        (regrid for mrgrid and convert for mrconvert in this case)
    logbool : boolean
        Boolean variable indicating if the logfile has to be taken into account

    Returns
    -------
    output_file_path : str
        Path to the output file where the modified file will be written
    log_file_path : str
        Path to the log file where the MRtrix3 commands' verbose is written

    """
    # current_path = os.getcwd()
    # new_path = os.path.join(current_path,f"tmp_folder_{operation}")
    new_path = os.path.join(output_path,"tmp",f"tmp_folder_{operation}")
    
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        
        if not os.path.exists(new_path):
            raise Exception(f"[!] {new_path} folder couldn't have been created")
        
        print(f"A folder was created in the output directory: {new_path} Do not remove this folder until analysis is finished")
        filename = f"{prefix}_{operation}ed_file_run_1.nii.gz"
        log_name = f"{prefix}_{operation}_log_run_1.txt"
    else:
        found_file = find_files(f"{prefix}_{operation}ed_file*", new_path)
        filename = file_number_increase(found_file[-1])
        if logbool:
            found_log_file = find_files(f"{prefix}_{operation}_log*", new_path)
            log_name = file_number_increase(found_log_file[-1])
        
    output_file_path = os.path.join(new_path, filename)
    if logbool:
        log_file_path = os.path.join(new_path, log_name)
    else:
        log_file_path = ""
        
    return output_file_path, log_file_path

def file_number_increase(string):
    """
    Parameters
    ----------
    string : str
        Input string to which the number is going to be increased

    Returns
    -------
    updated_string : str
        Output string to which the number has been increased

    """
    matches = list(re.finditer(r'\d+', string))
    
    if not matches:
        return string
    
    last_match = matches[-1]
    start, end = last_match.span()
    new_number = str(int(last_match.group()) + 1)
    
    updated_string = string[:start] + new_number + string[end:]
    
    return updated_string

def find_files(pattern, search_path):
    """
    Parameters
    ----------
    pattern : str
        Pattern that the filename must match to be part of the wanted filenames
    search_path : str
        Path where the wanted filenames are located

    Returns
    -------
    filenames : list
        List of the located filenames that match the inputted pattern

    """
    search_dir = Path(search_path)
    
    filenames = []
    for file in search_dir.rglob(pattern):
        filenames.append(file.name)
        
    filenames = sorted(filenames)
    
    return filenames
#mdf_end