from . import multiprocess_util as mu
from . import math_util as ma
from . import io_util as io

import numpy as np
import traceback
from scipy.stats import rankdata

def main(n_cpus, data, affine, header, norm, brain_mask, output_name, six_f, reho):   
    
    # Determine n_items and n_cpus
    n_items, n_cpus, dn = mu.determine_items_and_cpus(n_cpus, data)

    # Initialize multiprocessing
    pool, counter = mu.initialize_multiprocessing(n_cpus, n_items)
    
    internal_loop_func = vol_internal_loop
    arguments = (data, norm, brain_mask, reho, six_f)
    
    # Run multiprocessing
    results = mu.run_multiprocessing(pool, internal_loop_func, n_items, dn, arguments)
    
    # Process and save results
    process_vb_vol_results(results, data, affine, header, output_name)

def vol_internal_loop(i0, iN, data, norm, brain_mask, reho, six_f):
    
    """Computes the Vogt-Bailey index of voxels in a given range.

       Parameters
       ----------
       i0: integer
           Index of first voxel to be analysed.
       iN: integer
           iN - 1 is the index of the last voxel to be analysed.
       data: numpy array (nRows, nCols, nSlices, N)
           Volumetric data used to calculate the VB index. N is the number of maps.
       norm: string
           Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'.
       brain_mask: (nRows, nCols, nSlices) numpy array
           Whole brain mask. Used to eliminate voxels outside the brain from the analysis.
       residual_tolerance: float
           Residual tolerance (stopping criterion) for LOBPCG. Only relevant for the full brain algorithm.
       max_num_iter: integer
           Maximum number of iterations for LOBPCG. Only relevant for the full brain algorithm.
       reho : Bool.
           Indicates that ReHo approach will be computed
       debug: boolean
           Print the current progress of the system

       Returns
       -------
       loc_result: numpy array
                   Resulting VB index of the voxels in the given range. Will have size (iN-i0, 4).
    """
    
    # Calculate how many vertices we will compute
    diff = iN - i0
    loc_result = np.zeros((diff,5))

    for idx in range(diff):
        #Calculate the real index
        i = idx + i0
        neighborhood, vox_coords = ma.get_neighborhood_vol(data,i,six_f,brain_mask)
        if np.isnan(neighborhood).all():
            
            loc_result[idx,0] = np.nan
            loc_result[idx,1:4] = vox_coords
            loc_result[idx,-1] = np.nan
            continue
        to_keep = np.where(np.std(neighborhood,axis=1)>1e-10)
        neighborhood = np.squeeze(neighborhood[to_keep,:])
        neighborhood = np.atleast_2d(neighborhood)

        # Get neighborhood and its data
        try:

            if len(neighborhood) == 0:
                loc_result[idx,0] = np.nan
                loc_result[idx,1:4] = vox_coords
                loc_result[idx,-1] = 0
                continue
                
            affinity = ma.create_affinity_matrix(neighborhood)
            if reho:
                compute_vol_reho(neighborhood, i, idx, loc_result, affinity, vox_coords)
            else:
                compute_vol(neighborhood, i, idx, loc_result, affinity, vox_coords, norm)
            
        except ma.TimeSeriesTooShortError as error:
            raise error
        except Exception:
            traceback.print_exc()
            loc_result[idx,0] = np.nan
            loc_result[idx,1:4] = vox_coords
            loc_result[idx,-1] = np.nan
                
    return loc_result


def compute_vol(neighborhood, i, idx, loc_result, affinity, vox_coords, norm):
    """
    Computes the neighbourhood for every voxel in the volumetric analysis.

    Parameters
    ----------
    neighborhood : numpy array of float32 (M, N)
        Computed neighbourhood for every voxel.
    i : integer
        Real index.
    idx : integer
        Number of iteration in the main for loop from vb_hybrid_internal_loop function.
    loc_result : numpy array of float32 (M, N)
        Stores all the computed eigenvalues.
    affinity : numpy array of float32 (M, M)
        The affinity matrix computed for the neighbourhood.
    vox_coords : numpy array of int64 (M,)
        The coords of the center voxel.
    residual_tolerance : float
        The target tolerance for the calculation of eigenpairs.
    max_num_iter : integer
        Number of iterations for eigenpair calculation.
    norm : string
        Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'.

    Returns
    -------
    loc_result : numpy array of float32 (M, N)
        Stores all the computed eigenvalues.

    """
    assert affinity.shape[0] == len(neighborhood), 'affinity.shape[0] and len(neighborhood) do not match!'
    
    if affinity.shape[0] > 3:
        #Calculate the second smallest eigenvalue
        _, _, eigenvalue, _ = ma.spectral_reorder(affinity, norm)
        loc_result[idx,0] = eigenvalue
        loc_result[idx,1:4] = vox_coords
        loc_result[idx,-1] = affinity.shape[0]
    else:
        #print("Warning: too few neighbors ({})".format(no_of_voxels), "for vertex:",i)
        loc_result[idx,0] = np.nan
        loc_result[idx,1:4] = vox_coords
        loc_result[idx,-1] = affinity.shape[0]
        
    return loc_result

def compute_vol_reho(neighborhood, i, idx, loc_result, affinity, vox_coords):
    """
    Computes the neighbourhood for every voxel in the volumetric ReHo analysis.

    Parameters
    ----------
    neighborhood : numpy array of float32 (M, N)
        Computed neighbourhood for every voxel.
    i : integer
        Real index.
    idx : integer
        Number of iteration in the main for loop from vb_hybrid_internal_loop function.
    loc_result : numpy array of float32 (M, N)
        Stores all the computed KCC values.
    affinity : numpy array of float32 (M, M)
        The affinity matrix computed for the neighbourhood.
    vox_coords : numpy array of int64 (M,)
        The coords of the center voxel.

    Returns
    -------
    loc_result : numpy array of float32 (M, N)
        Stores all the computed KCC values.

    """
    
    no_of_voxels = np.shape(neighborhood)[0]
    no_of_time_pts = np.shape(neighborhood)[1]
    ranked_neigh = np.ones((no_of_voxels,no_of_time_pts))

    if no_of_voxels >= no_of_time_pts:
        print('neighborhood matrix must be transposed!')
    for s in range(no_of_voxels):
        ranked_neigh[s,:] = rankdata(neighborhood[s,:])
    ranked_sums = np.sum(ranked_neigh,axis=0)
    Rbar = np.sum(ranked_sums)/no_of_time_pts
    R = np.sum((ranked_sums - Rbar)**2)
    KCC = 12.*R / ((no_of_voxels**2)*(no_of_time_pts**3 - no_of_time_pts))
    
    if no_of_voxels > 3:
        loc_result[idx] = KCC
        loc_result[idx,1:4] = vox_coords
        loc_result[idx,-1] = affinity.shape[0]
    else:
        #print("Warning: too few neighbors ({})".format(no_of_voxels), "for vertex:",i)
        loc_result[idx] = np.nan
        loc_result[idx,1:4] = vox_coords
        loc_result[idx,-1] = affinity.shape[0]
        
    return loc_result

def process_vb_vol_results(results, data, affine, header, output_name):
    """
    This function processes the results gotten from the vb_vol_internal_loop.

    Parameters
    ----------
    results : numpy object array of (M,)
        Numpy array that contains the computed vb index for every volumetric data.
    data : Volumetric -> (nRows, nCols, nSlices, N) numpy array
        Volumetric data used to calculate the VB index. N is the number of maps.
    affine : numpy array (M, M), optional
        Matrix used to compute spatial translations and rotations. The default is None.
    header : Nifty header.
        This header contains crucial information about the structure and metadata. 
    output_name : string
        Name of the output file(s).
    debug : boolean, optional
        Outputs ribbon file for debugging. The default is False

    Returns
    -------
    None.

    """
    results_v2 = np.zeros((data.shape[0],data.shape[1],data.shape[2]), dtype=np.float32)
    n_neigh = np.zeros((data.shape[0],data.shape[1],data.shape[2]), dtype=np.float32)

    for i, res in enumerate(results):
        for r in results[i]:
            results_v2[int(r[1]),int(r[2]),int(r[3])] = r[0]
            n_neigh[int(r[1]),int(r[2]),int(r[3])] = r[4]
    
    # Save file
    if output_name is not None:      
        io.save_nifti(result=results_v2, n_neigh=n_neigh, affine=affine, header=header, output_name=output_name)