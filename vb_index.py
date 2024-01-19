#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 VB Index Team
#
# Distributed under terms of the GNU license.

import argparse
import multiprocessing
import nibabel
import numpy as np
import textwrap as _textwrap
import sys
import scipy.linalg as spl
import warnings
import traceback
from scipy.sparse.linalg import lobpcg
from scipy.stats import rankdata
import glob
import os
import signal
import time
import shutil

# Ctrl + C
def def_handler(sig, frame):
    
    print("\n[!] Keyboard Interrupt detected, exiting...")
    sys.exit(1)
    
signal.signal(signal.SIGINT, def_handler)

def compute_vb_metrics(internal_loop_func, n_cpus, data, norm, residual_tolerance, max_num_iter, header=None, brain_mask=None, surf_vertices=None, surf_faces=None, output_name=None, nib_surf=None, k=None, cluster_index=None, cort_index=None, affine=None, reho=False, full_brain=False, debug=False):
    """
    It is responsible for executing the functions in the correct order to achieve the final result.

    Parameters
    ----------
    internal_loop_func : string
        The function that is going to run depending on the analysis to be done.
    n_cpus : integer
        How many CPU cores are available.
    data : Surface -> numpy array (M, N)
        Data to use to calculate the VB index. M must match the number of vertices in the mesh.
           Volumetric -> (nRows, nCols, nSlices, N) numpy array
        Volumetric data used to calculate the VB index. N is the number of maps.
    norm : string
        Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'.
    residual_tolerance : float
        The target tolerance for the calculation of eigenpairs.
    max_num_iter : integer
        Number of iterations for eigenpair calculation.
    header : Nifty header.
        This header contains crucial information about the structure and metadata 
        associated with the image data stored in the file.
    brain_mask : numpy array of float32 (nRows, nCols, nSlices)
        Contains boolean values to indicate the area that belongs to the brain (value 1)
        and the area that belongs to the background(value 0).
    surf_vertices : numpy array (M, 3)
        Vertices of the mesh.
    surf_faces : numpy array (M, 3)
        Faces of the mesh. Used to find the neighborhood of a given vertex.
    output_name : string, optional
        Name of the output file(s). The default is None.
    nib_surf : nibabel object, optional
        Nibabel object containing metadata to be replicated. The default is None.
    k : integer, optional
        Factor determining increase in density of input mesh. The default is None.
    cluster_index : numpy array (M,), optional
       Array containing the cluster which each vertex belongs to. The default is None.
    cort_index : numpy array (M,), optional
        Mask for detection of middle brain structures. The default is None.
    affine : numpy array (M, M), optional
        Matrix used to compute spatial translations and rotations. The default is None.
    reho : Bool. The default is False
        Indicates that ReHo approach will be computed
    full_brain : Bool. The default is False
        Indicates that Full Brain analysis will be computed.
    debug : boolean, optional
        Outputs ribbon file for debugging. The default is False.

    Returns
    -------
    processed_results : Full brain -> tuple (numpy array, numpy array)
                        Searchlight -> numpy array of float32 (M,)
                        Hybrid & ReHo -> numpy object array (M,)
        It stores the results, either the eigenvalue(s), the eigenvector(s) or 
        the number of neighbours of each voxel.

    """

    func_mapping = {
        "vb_cluster": "vb_cluster_internal_loop",
        "vb_index": "vb_index_internal_loop",
        "vb_vol": "vb_vol_internal_loop"
    }
    internal_loop_func = func_mapping.get(internal_loop_func, "vb_hybrid_internal_loop")

    # Determine n_items and n_cpus
    n_items, n_cpus, dn = determine_items_and_cpus(internal_loop_func, surf_vertices, cluster_index, n_cpus, data)

    # Initialize multiprocessing
    pool, counter = initialize_multiprocessing(n_cpus, n_items)

    # Run multiprocessing
    results = run_multiprocessing(pool, internal_loop_func, n_items, dn, surf_vertices, surf_faces, data, norm, residual_tolerance, max_num_iter, cluster_index, cort_index, affine, k, reho, full_brain, brain_mask, debug)

    # Process and save results
    processed_results = process_and_save_results(internal_loop_func, results, output_name, nib_surf, surf_vertices, cluster_index, cort_index, affine, debug, data, n_items, header)

    # Clean up
    cleanup(pool)

    # Return results
    return processed_results

def determine_items_and_cpus(internal_loop_func, surf_vertices, cluster_index, n_cpus, data):
    """
    Divides and parallelizes a job according to the number of CPUs available

    Parameters
    ----------
    internal_loop_func : string
        The function that is going to run depending on the analysis to be done.
    surf_vertices : numpy array (M, 3)
        Vertices of the mesh.
    cluster_index : numpy array (M,)
        Indicates to which cluster each vertex belongs.
    n_cpus : integer
        How many CPU cores are available.
    data : Surface -> numpy array (M, N)
               Data to use to calculate the VB index. M must match the number of vertices in the mesh.
           Volumetric -> (nRows, nCols, nSlices, N) numpy array
               Volumetric data used to calculate the VB index. N is the number of maps.

    Returns
    -------
    n_items : integer
        How many clusters are.
    n_cpus : integer
        How many CPU cores are going to be used.
    dn : integer
        How many elements will be processed for each CPU.
    """
    
    if internal_loop_func == "vb_cluster_internal_loop":
        cluster_labels = np.unique(cluster_index)
        n_items = len(cluster_labels)
    elif internal_loop_func == "vb_vol_internal_loop":
        n_items = data.shape[0]*data.shape[1]*data.shape[2]
    else:
        n_items = len(surf_vertices)
    
    n_cpus = min(n_items, n_cpus)
    dn = max(n_items // n_cpus, 1)  # Ensure at least one item per CPU
    return n_items, n_cpus, dn

def initialize_multiprocessing(n_cpus, n_items):
    """
    It initializes the multiprocessing threads.

    Parameters
    ----------
    n_cpus : integer
        How many CPU cores are going to be used.
    n_items : integer
        How many clusters are.

    Returns
    -------
    pool : Pool object.
        Object where pool information is stored.
    counter : Synchronized object.
        To synchronize pool processes.

    """
    counter = Value('i', 0)
    pool = Pool(processes=n_cpus, initializer=init, initargs=(counter, n_items))
    
    return pool, counter

def run_multiprocessing(pool, internal_loop_func, n_items, dn, surf_vertices, surf_faces, data, norm, residual_tolerance, max_num_iter, cluster_index, cort_index, affine, k, reho, full_brain, brain_mask, debug):
    """
    Initializes the specific function for each analysis and takes care of multiprocessing.

    Parameters
    ----------
    pool : Pool object.
        Object where pool information is stored.
    internal_loop_func : string
        The function that is going to run depending on the analysis to be done.
    n_items : integer
        How many clusters are.
    dn : integer
        How many elements will be processed for each CPU.
    surf_vertices : numpy array (M, 3)
        Vertices of the mesh.
    surf_faces : numpy array (M, 3)
        Faces of the mesh. Used to find the neighborhood of a given vertex.
    data : Surface -> numpy array (M, N)
               Data to use to calculate the VB index. M must match the number of vertices in the mesh.
           Volumetric -> (nRows, nCols, nSlices, N) numpy array
               Volumetric data used to calculate the VB index. N is the number of maps.
    norm : string
        Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'.
    residual_tolerance : float
        The target tolerance for the calculation of eigenpairs.
    max_num_iter : integer
        Number of iterations for eigenpair calculation.
    cluster_index : numpy array (M,)
        Array containing the cluster which each vertex belongs to.
    cort_index : numpy array (M,)
        Mask for detection of middle brain structures.
    affine : numpy array (M, M), optional
        Matrix used to compute spatial translations and rotations.
    k : integer
        Factor determining increase in density of input mesh.
    reho : Bool.
        Indicates that ReHo approach will be computed
    full_brain : Bool.
        Indicates that Full Brain analysis will be computed.        
    brain_mask : numpy array of float32 (nRows, nCols, nSlices)
        Contains boolean values to indicate the area that belongs to the brain (value 1)
        and the area that belongs to the background(value 0).
    debug : boolean
        Outputs ribbon file for debugging.

    Returns
    -------
    t.get(): numpy array.
        It gets the results of all the threads.

    """
    
    def pool_callback(result):
        """
        This function is used to handle pool errors

        Parameters
        ----------
            result : pool object
        Error that occurred during the process.
    
        Returns
        -------
        None.
    
        """
        # Define error handling here
        print("Error occurred in pool execution:", result)
        # Terminate the pool in case of error
        pool.close()
        pool.terminate()
        
    threads = np.array([])
    for i0 in range(0, n_items, dn):
        iN = min(i0 + dn, n_items)
        if internal_loop_func == "vb_cluster_internal_loop":
            threads = np.append(threads, (pool.apply_async(vb_cluster_internal_loop, (full_brain, i0, iN, surf_faces, data, cluster_index, norm, residual_tolerance, max_num_iter), error_callback=pool_callback)))
        elif internal_loop_func == "vb_hybrid_internal_loop":
            threads = np.append(threads, (pool.apply_async(vb_hybrid_internal_loop, (reho, i0, iN, surf_vertices, surf_faces, affine, data, norm, residual_tolerance, max_num_iter, k, debug), error_callback=pool_callback)))         
        elif internal_loop_func == "vb_vol_internal_loop":
            threads = np.append(threads, (pool.apply_async(vb_vol_internal_loop, (i0, iN, data, norm, brain_mask, residual_tolerance, max_num_iter, reho, debug), error_callback=pool_callback)))
        else:
            threads = np.append(threads, (pool.apply_async(vb_index_internal_loop, (i0, iN, surf_faces, data, norm, residual_tolerance, max_num_iter), error_callback=pool_callback)))
                
    
    # Wait for all threads to complete
    pool.close()
    pool.join()

    res = np.empty((len(threads),), dtype=object)

    for r, t in enumerate(threads):
        res[r] = t.get() 

    return res


def process_and_save_results(internal_loop_func, results, output_name, nib_surf, surf_vertices, cluster_index, cort_index, affine, debug, data, n_items, header):
    """
    It contains the logic to process the results depending on the desired analysis

    Parameters
    ----------
    internal_loop_func : string
        The function that is going to run depending on the analysis to be done.
    results : numpy array (M,)
        The results of all the threads.
    output_name : string
        Name of the output file(s).
    nib_surf : nibabel object
        Nibabel object containing metadata to be replicated.
    surf_vertices : numpy array (M, 3)
        Vertices of the mesh.
    cluster_index : numpy array (M,)
        Array containing the cluster which each vertex belongs to.
    cort_index : numpy array (M,)
        Mask for detection of middle brain structures.
    affine : numpy array (M, M), optional
        Matrix used to compute spatial translations and rotations.
    debug : boolean
        Outputs ribbon file for debugging.
    data : Surface -> numpy array (M, N)
               Data to use to calculate the VB index. M must match the number of vertices in the mesh.
           Volumetric -> (nRows, nCols, nSlices, N) numpy array
               Volumetric data used to calculate the VB index. N is the number of maps.
    n_items : integer
        How many clusters are. 
    header : Nifty header.
        This header contains crucial information about the structure and metadata 
        associated with the image data stored in the file.

    Returns
    -------
    Full-brain -> results_eigenvalues : numpy array (M, )
                      Array of the same size as the eigenvectors array but only containing
                      the eigenvalue(s).
                  results_eigenvectors : numpy array (M, 1)
                      Array with the same size as the eigenvalues array, each column corresponds
                      to each value.
                      
    Searchlight -> results_v2 : numpy array (M, )
                       Array of the computed eigenvalues.
                       
    Hybrid -> results_v2 : numpy array of float32 (M,)
                  It contains the computed eigenvalues.
              n_neigh : numpy array of float32 (M,)
                  It contains the amount of neighbours of each voxel.
                  

    """
    
    # Implement the processing of results and saving of files as needed.
    if internal_loop_func == "vb_cluster_internal_loop":
        # Processing for vb_cluster_internal_loop
        # Replicating the logic from the old code to handle eigenvalues and eigenvectors
        return process_vb_cluster_results(results, surf_vertices, cluster_index, output_name, nib_surf, n_items)
    elif internal_loop_func == "vb_index_internal_loop":
        # Processing for vb_index_internal_loop
        # This should handle the results specifically for vb_index_internal_loop
        return process_vb_index_results(results, cort_index, output_name, nib_surf)
    elif internal_loop_func == "vb_vol_internal_loop":
        return process_vb_vol_results(results, data, affine, header, output_name, debug)
    else:
        # Processing for vb_hybrid_internal_loop
        # This should handle the results specifically for vb_hybrid_internal_loop
        return process_vb_hybrid_results(results, cort_index, output_name, nib_surf, affine, debug, data)
        
def cleanup(pool):
    """
    This function is used to terminate de pool once it is no longer needed.

    Parameters
    ----------
    pool : Pool object.
        Object where pool information is stored.

    Returns
    -------
    None.

    """
    pool.terminate()

def process_vb_cluster_results(results, surf_vertices, cluster_index, output_name, nib_surf, n_items):
    """
    It processes the results obtained from the vb_cluster_internal_loop and 
    sneds the correct data to create the output file(s).

    Parameters
    ----------
    results : python list.
        The results of the vb_cluster_internal_loop.
    surf_vertices : numpy array (M, 3)
        Vertices of the mesh.
    cluster_index : numpy array (M, )
        Array containing the cluster which each vertex belongs to.
    output_name : string.
        Name of the output file(s). The default is None.
    nib_surf : nibabel object.
        Nibabel object containing metadata to be replicated.
    n_items : integer
        How many clusters are.

    Returns
    -------
    results_eigenvalues : numpy array (M, )
        Array of the same size as the eigenvectors array but only containing
        the eigenvalue(s).
    results_eigenvectors : numpy array (M, 1)
        Array with the same size as the eigenvalues array, each column corresponds
        to each value.

    """
    # Process results as done in the old code for vb_cluster_internal_loop
    # Replace the following lines with your specific logic for processing
    cluster_labels = np.unique(cluster_index)
    midline_index = cluster_index == 0
    results_v2 = np.empty(((len(cluster_labels)-1),), dtype=object)
    #results_v2 = [[], []]
    results_eigenvectors_l = np.empty(((len(cluster_labels)-1),), dtype=object)
    
    #results_eigenvectors_l = [[], []]
    # Save files if output_name is provided
    for r, rv in enumerate(results):
        if r == 0:
            continue
        results_v2[r-1] = rv[r+r-2]
        results_eigenvectors_l[r-1] = rv[r+r-1]
        
    results_eigenvalues = np.zeros(len(surf_vertices), dtype=np.float32)
    results_eigenvectors = np.array([], dtype=np.float32)
    for i in range(n_items):
        cluster = cluster_labels[i]
        if cluster != 0:
            results_eigenvectors_local = np.zeros(len(surf_vertices), dtype=np.float32)
            idx = np.where(cluster_index == cluster)[0]
            results_eigenvalues[idx] = results_v2[i-1]
            results_eigenvectors_local[idx] = results_eigenvectors_l[i-1]
            results_eigenvectors = np.append(results_eigenvectors_local, results_eigenvectors)

    results_eigenvectors = results_eigenvectors.reshape(-1, 1)

    # Remove the midbrain
    results_eigenvalues[midline_index] = np.nan
    results_eigenvectors[midline_index, :] = np.nan
    
    if output_name is not None:
        save_gifti(nib_surf, results_eigenvalues, output_name + ".vb-cluster.value.shape.gii")
        save_gifti(nib_surf, results_eigenvectors, output_name + ".vb-cluster.vector.shape.gii")
    return results_eigenvalues, results_eigenvectors

def process_vb_index_results(results, cort_index, output_name, nib_surf):
    """
    

    Parameters
    ----------
    results : numpy array.
        The results of the vb_cluster_internal_loop.
    cort_index : numpy array (M, )
        Mask for detection of middle brain structures.
    output_name : string.
        Name of the output file(s).
    nib_surf : nibabel object.
        Nibabel object containing metadata to be replicated.

    Returns
    -------
    results_v2 : numpy array (M, )
        Array of the computed eigenvalues.

    """
    # Process results as done in the old code for vb_index_internal_loop
    # Replace the following lines with your specific logic for processing
    results_v2 = np.array([], dtype=np.float32)
    for r in results:
        results_v2 = np.append(results_v2, r)
    #results = np.array(results)
    results_v2[np.logical_not(cort_index)] = np.nan
    # The problem is that r is float64, when appending, results_v2 becomes
    #float64 aswell.
    # Save files if output_name is provided
    if output_name is not None:
        save_gifti(nib_surf, results_v2, output_name + ".vbi.shape.gii")
    return results_v2

def process_vb_hybrid_results(results, cort_index, output_name, nib_surf, affine, debug, data):
    """
    

    Parameters
    ----------
    results : numpy array
        The results of the vb_hybrid_internal_loop.
    cort_index : numpy array (M, )
        Mask for detection of middle brain structures.
    output_name : string
        Name of the output file(s).
    nib_surf : nibabel object.
        Nibabel object containing metadata to be replicated.
    affine : numpy array (M, M).
        Matrix used to compute spatial translations and rotations.
    debug : boolean
        Outputs ribbon file for debugging.
    data : numpy array (M, N)
        Data to use to calculate the VB index. M must match the number of vertices in the mesh.

    Returns
    -------
    results_v2 : numpy array of float32 (M,)
        It contains the computed eigenvalues.
    n_neigh : numpy array of float32 (M,)
        It contains the amount of neighbours of each voxel.

    """
    # Process results as done in the old code for vb_hybrid_internal_loop
    # Replace the following lines with your specific logic for processing
    results_v2 = np.array([], dtype=np.float32)
    n_neigh = np.array([], dtype=np.float32)
    if debug:
        coords = np.empty((0,3))
    for i, res in enumerate(results):
        for r in res[0]:
            results_v2 = np.append(results_v2, r)
        for r in res[1]:
            n_neigh = np.append(n_neigh, r)
        if debug:
            coords = np.vstack([coords,res[2]])
    results_v2[np.logical_not(cort_index)] = np.nan
    n_neigh[np.logical_not(cort_index)] = np.nan
    if debug: coords = coords.astype(int)

    
    # Save file
    if output_name is not None:
        save_gifti(nib_surf, results_v2, output_name + ".vbi-hybrid.shape.gii")
        save_gifti(nib_surf, n_neigh, output_name + ".neighbors.shape.gii")
        if debug:
            ribbon = np.zeros([data.shape[0],data.shape[1],data.shape[2]])
            ribbon[coords[:,0], coords[:,1], coords[:,2]] = 1
            nibabel.save(nibabel.Nifti1Image(ribbon, affine),output_name+'.ribbon.nii.gz')
            
    return results_v2

def process_vb_vol_results(results, data, affine, header, output_name, debug=False):
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

    if debug:
        coords = np.empty((0,3))

    for i, res in enumerate(results):
        for r in results[i]:
            results_v2[int(r[1]),int(r[2]),int(r[3])] = r[0]
            n_neigh[int(r[1]),int(r[2]),int(r[3])] = r[4]
    if debug:
        coords = np.vstack([coords,results[2]])

    if debug: coords = coords.astype(int)
    
    # Save file
    if output_name is not None:
        
        save_nifti(result=results_v2, n_neigh=n_neigh, affine=affine, header=header, output_name=output_name)

    return results_v2

def open_gifti_surf(filename):
    """
    Helper function to read the surface from a gifti file.

    Parameters
    ----------
    filename: string
    String containing the name of the file to be opened.

    Returns
    -------
    nib: Nibabel object
    vertices: (N, 3) numpy vector
              Vertices of the mesh
    faces: (M, 3) numoy vector
            Faces of the mesh.
    """
    nib = nibabel.load(filename)
    return nib, nib.get_arrays_from_intent('pointset')[0].data,  nib.get_arrays_from_intent('triangle')[0].data

def open_gifti(filename):
    """
    Helper function to read data from a gifti file.

    Parameters
    ----------
    filename: string
    String containing the name of the file to be opened.

    Returns
    -------
    nib: Nibabel object
    data: (M, N) numpy array
          Data in the file.
    """
    nib = nibabel.load(filename)

    # We use the first data as this is agnostic to the intent. In the future we
    # might want to change it.
    return nib, nib.darrays[0].data

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

def save_gifti(og_img, data, filename):
    """
    Helper function to write data into a gifti file.

    Parameters
    ----------
    og_img: Nibabel object
    data: (M, N) numpy array
          Data to write into the file. M must math the number of vertices in the mesh
    filename: string
              String containing the name of the file to be saved
    """
    # For some reason, wc_view demands float32
    data_array = nibabel.gifti.gifti.GiftiDataArray(data)

    # Create a meta object containing the cortex information
    if 'AnatomicalStructurePrimary' in og_img.meta:
        new_meta = nibabel.gifti.gifti.GiftiMetaData(AnatomicalStructurePrimary=og_img.meta['AnatomicalStructurePrimary'])
    else:
        new_meta = nibabel.gifti.gifti.GiftiMetaData()
    new_nib = nibabel.gifti.gifti.GiftiImage(darrays=[data_array], meta=new_meta)

    nibabel.save(new_nib, filename)

counter = None
n = None

from multiprocessing import Pool, Value
from itertools import product

def init(a_counter, a_n):
    """
    Store total number of vertices and counter of vertices computed

    Parameters
    ----------
    a_counter : TYPE
        DESCRIPTION.
    a_n : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    global counter
    global n
    counter = a_counter
    n = a_n

def vb_index_internal_loop(i0, iN, surf_faces, data, norm, residual_tolerance, max_num_iter, debug=False):
    """Computes the Vogt-Bailey index of vertices in a given range

       Parameters
       ----------
       i0: integer
           Index of first vertex to be analysed
       iN: integer
           iN - 1 is the index of the last vertex to be analysed
       surf_faces: (M, 3) numpy array
           Faces of the mesh. Used to find the neighborhood of a given vertex
       data: (M, N) numpy array
           Data to use to calculate the VB index. M must match the number of vertices in the mesh
       norm: string
           Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'
       residual_tolerance : float
           The target tolerance for the calculation of eigenpairs.
       max_num_iter : integer
           Number of iterations for eigenpair calculation.
       debug: boolean
           Print the current progress of the system

       Returns
       -------
       loc_result: (N) numpy array
                   Resulting VB index of the indices in range. Will have length iN - i0
    """

    # Calculate how many vertices we will compute
    diff = iN - i0 # Estos determinan el rango de vertices que seran analizados
    loc_result = np.zeros(diff, dtype=np.float32) # Se hace una matriz de 0 teniendo en cuenta el rango de los vertices

    for idx in range(diff):
        # Calculate the real index
        i = idx + i0

        # Get neighborhood and its data
        try:
            neighbor_idx = np.array(np.sum(surf_faces == i, 1), dtype=bool)
            I = np.unique(surf_faces[neighbor_idx, :])
            neighborhood = data[I] # Va iterando por cada vertice y mira si tiene vecinos
            if len(neighborhood) == 0:
                print("Warning: no neighborhood for vertex:",i)
                loc_result[idx] = np.nan # Si no tiene que pone que no tiene y le ponen un NaN
                continue

            # Calculate the second smallest eigenvalue
            affinity = create_affinity_matrix(neighborhood) # Crea la matriz de afinidad para poder sacar luego el eigenvalue 
            _, _, eigenvalue, _ = spectral_reorder(False, affinity, residual_tolerance, max_num_iter, norm) # Calcula el segundo valor mas pequeno del eigenvalue que te dice el grado

            # return [0]
            # Store the result of this run
            loc_result[idx] = eigenvalue # El resultado de va a terminar siendo el eigenvalue
        except TimeSeriesTooShortError as error:
            raise error 
        except Exception:
            loc_result[idx] = np.nan # Simplemente gestiona errores

        if debug: # Esto es un input para que si esta en True te va printeando info, supongo que para debguear

            global counter
            global n
            with counter.get_lock():
                counter.value += 1
            if counter.value % 1000 == 0:
                print("{}/{}".format(counter.value, n))

    return loc_result


def vb_cluster_internal_loop(full_brain, idx_cluster_0, idx_cluster_N, surf_faces, data, cluster_index, norm, residual_tolerance, max_num_iter, debug=False):
    """Computes the Vogt-Bailey index and Fiedler vector of vertices of given clusters

       Parameters
       ----------
       full_brain : 
           
       idx_cluster_0: integer
           Index of first cluster to be analysed
       idx_cluster_N: integer
           idx_cluster_N - 1 is the index of the last cluster to be analysed
       surf_faces: (M, 3) numpy array
           Faces of the mesh. Used to find the neighborhood of a given vertex
       data: (M, N) numpy array
           Data to use to calculate the VB index and Fiedler vector. M must match the number of vertices in the mesh
       cluster_index: (M) numpy array
           Array containing the cluster which each vertex belongs to
       norm: string
           Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'
       residual_tolerance : float
           The target tolerance for the calculation of eigenpairs.
       max_num_iter : integer
           Number of iterations for eigenpair calculation.
       debug: boolean
           Print the current progress of the system

       Returns
       -------
       loc_result: list of pairs of (float, (N) numpy array)
                   Resulting VB index and Fiedler vector for each of the clusters in range
    """
    diff = idx_cluster_N - idx_cluster_0
    loc_result = np.empty(((diff+1),), dtype=object)
    cluster_labels = np.unique(cluster_index)

    for idx in range(diff):
        #Calculate the real index
        i = idx + idx_cluster_0

        if(cluster_labels[i] == 0):
            #loc_result.append(([], []))
            # This line was commented because it makes no sense to fill
            # a result variable with empty arrays
            continue
        try:
            # Get neighborhood and its data
            neighborhood = data[cluster_index == cluster_labels[i]]

            # Calculate the Fiedler eigenpair
            affinity = create_affinity_matrix(neighborhood)
            _, _, eigenvalue, eigenvector = spectral_reorder(full_brain, affinity, residual_tolerance, max_num_iter, norm)

            # Store the result of this run
            # Warning: It is not true that the eigenvectors will be all the same
            # size, as the clusters might be of different sizes
            #val = eigenvalue
            #vel = eigenvector
            #loc_result.append((val, vel))
            loc_result[0] = eigenvalue
            loc_result[1] = eigenvector
            #lr_1[idx] = val
            #lr_2[idx] = vel
        except TimeSeriesTooShortError as error:
            raise error


        if debug:

            global counter
            global n
            with counter.get_lock():
                counter.value += 1
            if counter.value % 1000 == 0:
                print("{}/{}".format(counter.value, n))

    return loc_result

#def claude_choose_whatever_name_you_want(reho, neighborhood, full_brain residual_tolerance, max_num_iter, norm):
    
#    if reho:
        
#    else:
#        affinity = create_affinity_matrix(neighborhood)
#        _, _, eigenvalue, eigenvector = spectral_reorder(full_brain, affinity, residual_tolerance, max_num_iter, norm)

def get_neighborhood(data, surf_vertices, surf_faces, i, affine, k=3, debug=False):
    """
    Get neighbors in volumetric space given the coordinates of a vertex

    Parameters
    ----------
    data : (M, N) numpy array
        Data to use to calculate the VB index and Fiedler vector. M must match the number of vertices in the mesh
    surf_vertices : numpy array (M, 3)
        Vertices of the mesh.
    surf_faces : (M, 3) numpy array
        Faces of the mesh. Used to find the neighborhood of a given vertex.
    i : integer
        Index of the vertex.
    affine : numpy array (M, M), optional
        Matrix used to compute spatial translations and rotations.
    k : integer
        Factor determining increase in density of input mesh (default k=3)
    debug : boolean, optional.
        Outputs ribbon file for debugging (default = False)

    Returns
    -------
    data : numpy array float32 (M, N, X, Y)
        It stores the neighbours of each voxel.

    """

    # Step 1: Find indices of faces containing vertex i
    neighbor_idx = np.array(np.sum(surf_faces == i, 1), dtype=bool)
    I = np.unique(surf_faces[neighbor_idx, :])

    # Step 2: Generate new vertices in the neighborhood
    new_v = np.array([surf_vertices[i] + (surf_vertices[j] - surf_vertices[i]) * p / k for p in range(1, k)
                      for j in np.setdiff1d(I, i)])

    # Step 3: Combine original and new vertices to create a dense neighborhood
    dense_neigh = np.vstack([surf_vertices[I], new_v])

    # Step 4: Transform dense neighborhood coordinates to voxel space
    neigh_coords = np.round(nibabel.affines.apply_affine(spl.inv(affine), dense_neigh)).astype(int)

    # Step 5: Transform vertex coordinates to voxel space
    v_map = np.round(nibabel.affines.apply_affine(spl.inv(affine), surf_vertices[i])).astype(int)

    # Step 6: Create a cube of voxel coordinates around the vertex
    v_cube = np.array([relative_index for relative_index in product((-1, 0, 1), repeat=3)]) + v_map

    # Step 7: Filter out unique voxel coordinates that are within the cube
    neigh_coords = np.array([point for point in np.unique(neigh_coords, axis=0) if point in v_cube]).astype(int)

    # Step 8: Check if the neighborhood is too small and add second-ring neighbors
    if len(neigh_coords) < 4:
        neighbors = I
        for j in I:
            neighbor_idx = np.array(np.sum(surf_faces == j, 1), dtype=bool)
            J = np.unique(surf_faces[neighbor_idx, :])
            neighbors = np.union1d(neighbors, J)

        # Step 9: Generate new vertices for the second ring of neighbors
        new_v = np.array([surf_vertices[i] + (surf_vertices[j] - surf_vertices[i]) * p / k for p in range(1, k)
                          for j in np.setdiff1d(neighbors, i)])

        # Step 10: Combine original and new vertices for the second ring
        dense_neigh = np.vstack([surf_vertices[neighbors], new_v])

        # Step 11: Transform second-ring neighborhood coordinates to voxel space
        neigh_coords = np.round(nibabel.affines.apply_affine(spl.inv(affine), dense_neigh)).astype(int)

        # Step 12: Filter out unique voxel coordinates that are within the cube
        neigh_coords = np.array([point for point in np.unique(neigh_coords, axis=0) if point in v_cube]).astype(int)

    # Step 13: If in debug mode, return neighborhood data and coordinates
    if debug:
        return data[neigh_coords[:, 0], neigh_coords[:, 1], neigh_coords[:, 2], :], neigh_coords
    else:
        # Step 14: Otherwise, return neighborhood data only
        return data[neigh_coords[:, 0], neigh_coords[:, 1], neigh_coords[:, 2], :]


def vb_hybrid_internal_loop(reho, i0, iN, surf_vertices, surf_faces, affine, data, norm, residual_tolerance, max_num_iter, k=3, debug=False):
    """Computes the Vogt-Bailey index of vertices in a given range

       Parameters
       ----------
       reho: boolean
           Used to compute the ReHo approach.
       i0: integer
           Index of first vertex to be analysed
       iN: integer
           iN - 1 is the index of the last vertex to be analysed
       surf_vertices: (M, 3) numpy array
           Coordinates of vertices of the mesh in voxel space
       surf_faces : (M, 3) numpy array
           Faces of the mesh. Used to find the neighborhood of a given vertex.          
       affine : numpy array (M, M), optional
           Matrix used to compute spatial translations and rotations.
       data: (nRows, nCols, nSlices, N) numpy array
           Volumetric data used to calculate the VB index. N is the number of maps
       norm: string
           Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'
       residual_tolerance : float
           The target tolerance for the calculation of eigenpairs.
       max_num_iter : integer
           Number of iterations for eigenpair calculation.           
       k: integer
           Factor determining increase in density of input mesh (default k=3)
       debug: boolean
           Outputs ribbon file for debugging
     
       Returns
       -------
       loc_result: (N) numpy array
                   Resulting VB index of the indices in range. Will have length iN - i0
    """
    
    # Calculate how many vertices we will compute
    diff = iN - i0
    loc_result = np.zeros(diff, dtype=np.float32)
    loc_neigh = np.zeros(diff, dtype=np.float32)
    if debug: all_coords = np.empty((0,3))


    for idx in range(diff):
        #Calculate the real index
        i = idx + i0

        # Get neighborhood and its data
        try:
            if debug:
                neighborhood, neigh_coords = get_neighborhood(data,surf_vertices,surf_faces,i,affine,k=k,debug=True)
                all_coords = np.vstack([all_coords,neigh_coords])
            else:
                neighborhood = get_neighborhood(data,surf_vertices,surf_faces,i,affine,k=k)
            to_keep = np.where(np.std(neighborhood,axis=1)>1e-10)
            neighborhood = np.squeeze(neighborhood[to_keep,:])
            neighborhood = np.atleast_2d(neighborhood)
            loc_neigh[idx] = len(neighborhood)

            if reho:
                loc_result =  compute_reho(loc_result, idx, i, neighborhood)
            else:
                loc_result =  compute_vb_index(loc_result, idx, i, neighborhood, residual_tolerance, max_num_iter, norm)

        except TimeSeriesTooShortError as error:
            raise error
        except Exception:
            traceback.print_exc()
            loc_result[idx] = np.nan
        

        if debug:

            global counter
            global n
            with counter.get_lock():
                counter.value += 1
            if counter.value % 1000 == 0:
                print("{}/{}".format(counter.value, n))

    if debug:
        return loc_result, loc_neigh, all_coords
    else:
        return loc_result, loc_neigh
	
def vb_vol_internal_loop(i0, iN, data, norm, brain_mask, residual_tolerance, max_num_iter, reho, debug=False):
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
        neighborhood, vox_coords = get_neighborhood_vol(data,i,brain_mask)
        if np.isnan(neighborhood).all():
            
            loc_result[idx,0] = np.nan
            loc_result[idx,1:4] = vox_coords
            loc_result[idx,-1] = np.nan
            continue

        # Get neighborhood and its data
        try:

            if len(neighborhood) == 0:
                loc_result[idx,0] = np.nan
                loc_result[idx,1:4] = vox_coords
                loc_result[idx,-1] = 0
                continue
                
            affinity = create_affinity_matrix(neighborhood)
            if reho:
                compute_vol_reho(neighborhood, i, idx, loc_result, affinity, vox_coords)
            else:
                compute_vol(neighborhood, i, idx, loc_result, affinity, vox_coords, residual_tolerance, max_num_iter, norm)
            
        except TimeSeriesTooShortError as error:
            raise error
        except Exception:
            traceback.print_exc()
            loc_result[idx,0] = np.nan
            loc_result[idx,1:4] = vox_coords
            loc_result[idx,-1] = np.nan
        

        if debug:

            global counter
            global n
            with counter.get_lock():
                counter.value += 1
            if counter.value % 1000 == 0:
                print("{}/{}".format(counter.value, n))
                

#    if debug:
#        return loc_result, loc_neigh, all_coords
#    else:
    return loc_result

def get_neighborhood_vol(data,i,mask=None):
    """
    Get neighbors in volumetric space given the index of a voxel.
    Each voxel is assigned a unique index i by previous functions 
    but this function converts i to coordinates in 3D.

    Parameters
    ----------
    data : numpy array (nRows, nCols, nSlices, N)
        Volumetric data used to calculate the VB index. N is the number of maps.
    i : integer
        Index of the vertex.
    mask : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    data : TYPE
        DESCRIPTION.
    vox_coords : TYPE
        DESCRIPTION.

    """
   
    p0 = data.shape[0]
    q0 = data.shape[1]
    r0 = data.shape[2]
    vox_coords = np.ones(3)
    
    if i < r0:
        vox_coords = np.array([0,0,i])
    elif i < q0*r0:
        vox_coords = np.array([0,i//r0,i%r0])
    else:
        vox_coords = np.array([i//(q0*r0),(i%(q0*r0))//r0,(i%(q0*r0))%r0])
    if (vox_coords[0] or vox_coords[1] or vox_coords[2]) < 0:
        print('WARNING! Negative voxel coordinates!')
            
    neigh_coords = np.array([relative_index for relative_index in product((-1, 0, 1), repeat=3)])+vox_coords
    neigh_coords = neigh_coords.astype(int)
    for j in range(np.shape(neigh_coords)[1]):
        new_neigh = np.delete(neigh_coords, neigh_coords[:,j] < 0, axis = 0)
        neigh_coords = new_neigh

    if mask is not None:
        if mask[vox_coords[0],vox_coords[1],vox_coords[2]] == 1:
            neigh_coords = neigh_coords[(neigh_coords[:,0] < p0) & (neigh_coords[:,1] < q0) & (neigh_coords[:,2] < r0)]
            data = data[neigh_coords[:,0],neigh_coords[:,1],neigh_coords[:,2],:]
            data = np.atleast_2d(data)
        else:
            data = np.nan
    else:
        neigh_coords = neigh_coords[(neigh_coords[:,0] < p0) & (neigh_coords[:,1] < q0) & (neigh_coords[:,2] < r0)]
        data = data[neigh_coords[:,0],neigh_coords[:,1],neigh_coords[:,2],:]
        data = np.atleast_2d(data)
            
        
    return data, vox_coords

def compute_reho(loc_result, idx, i, neighborhood):
    """
    Computes the Kendall's coefficient concordance (KCC) for the ReHo approach.

    Parameters
    ----------
    loc_result : numpy array
        It stores the KCC values.
    diff : integer
        The amount of vertices that will be compute.
    idx : integer
        Number of iteration in the main for loop from vb_hybrid_internal_loop function.
    i : integer
        Real index.
    reho : boolean
        Used to compute the ReHo approach.
    neighborhood : numpy array of float32 (M, N)
        Computed neighbourhood for every voxel.
    residual_tolerance : float
        The target tolerance for the calculation of eigenpairs.
    max_num_iter : integer
        Number of iterations for eigenpair calculation.
    norm : string
        Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'.

    Returns
    -------
    loc_result : numpy array
        It stores the KCC values.

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
    else:
        print("Warning: too few neighbors ({})".format(no_of_voxels), "for vertex:",i)
        loc_result[idx] = np.nan
        
    return loc_result

def compute_vb_index(loc_result, idx, i, neighborhood, residual_tolerance, max_num_iter, norm):
    """
    Computes the eigenvalue for every voxel.

    Parameters
    ----------
    loc_result : numpy array
        It stores the KCC values.
    idx : integer
        Number of iteration in the main for loop from vb_hybrid_internal_loop function.
    i : integer
        Real index.
    neighborhood : numpy array of float32 (M, N)
        Computed neighbourhood for every voxel.
    residual_tolerance : float
        The target tolerance for the calculation of eigenpairs.
    max_num_iter : integer
        Number of iterations for eigenpair calculation.
    norm : string
        Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'.

    Returns
    -------
    loc_result : numpy array
        It stores the KCC values.

    """
    
    if len(neighborhood) == 0:
        print("Warning: no neighborhood for vertex:",i)
        loc_result[idx] = np.nan
        
    else:
    
        affinity = create_affinity_matrix(neighborhood)
    
        if affinity.shape[0] > 3:
            # Calculate the second smallest eigenvalue
            _, _, eigenvalue, _ = spectral_reorder(False, affinity, residual_tolerance, max_num_iter, norm)
            loc_result[idx] = eigenvalue
        else:
            print("Warning: too few neighbors ({})".format(affinity.shape[0]), "for vertex:",i)
            loc_result[idx] = np.nan
        
    return loc_result

def compute_vol(neighborhood, i, idx, loc_result, affinity, vox_coords, residual_tolerance, max_num_iter, norm):
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
        _, _, eigenvalue, _ = spectral_reorder(False, affinity, residual_tolerance, max_num_iter, norm)
        loc_result[idx,0] = eigenvalue
        loc_result[idx,1:4] = vox_coords
        loc_result[idx,-1] = affinity.shape[0]
    else:
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
        print("Warning: too few neighbors ({})".format(no_of_voxels), "for vertex:",i)
        loc_result[idx] = np.nan
        loc_result[idx,1:4] = vox_coords
        loc_result[idx,-1] = affinity.shape[0]
        
    return loc_result
    
class TimeSeriesTooShortError(Exception):
    """Raised when the time series in the input data have less than three elements"""
    pass
    

def force_symmetric(M):
    """Force the symmetry of a given matrix.

    The symmetric version is computed by first replicating the diagonal and
    upper triangular components of the matrix, and mirroring the upper diagonal
    into the lower diagonal them into the lower triangular.

    Parameters
    ----------
    M: (N, N) numpy array
       Matrix to be made symmetric

    Returns
    -------
    M_sym: Symmetric version of M
    """
    # One diag extracts the diagonal into an array, two
    # diags turns the array into a diagonal matrix.
    diag_M = np.diag(np.diag(M))
    triu_M = np.triu(M, 1)

    return triu_M + diag_M + triu_M.transpose()
    

def get_fiedler_eigenpair(method, full_brain, Q, D=None, is_symmetric=True, tol='def_tol', maxiter=50):

    """Solve the general eigenproblem to find the Fiedler vector and the corresponding eigenvalue.

    Solves the general eigenproblem Qx = lambda*Dx, and returns the Fiedler vector and associated eigenvalue. 
    The former is unitary in the norm induced by the matrix D. If D is not set, the identity matrix is assumed.

    Parameters
    ----------

    Q: (M, M) numpy array
       Main matrix
    D: (M, M) numpy array
       Matrix that accounts for node degree bias. If not set, this function will solve the
       standard eigenproblem (Qx = lambda*x)
    is_symmetric: boolean
                  Indicates whether Q *and* D are symmetric. If set, the program will use a 
                  much faster, but less general, algorithm

    Returns
    -------
    vbi_value: floating-point number
                         The second smallest eigenvalue
    fiedler_vector: (M) numpy array
                    The Fiedler vector
    """
    if is_symmetric:
        if full_brain == True:
            X = np.random.rand(Q.shape[0],2)
            tol_standard = np.sqrt(1e-15) * Q.shape[0]
            if tol == 'def_tol':
                tol = tol_standard*(10**(-3))
            eigenvalues, eigenvectors = lobpcg(Q, X, B=D, M=None, Y=None, tol=tol, maxiter=maxiter, largest=False, verbosityLevel=0, retLambdaHistory=False, retResidualNormsHistory=False)
        else:
            eigenvalues, eigenvectors = spl.eigh(Q, D, check_finite=False)
          
    else:
        
        eigenvalues, eigenvectors = spl.eig(Q, D, check_finite=False)
    
    eigenvalues = np.array(eigenvalues, dtype=np.float32)
    eigenvectors = np.array(eigenvectors, dtype=np.float32)
    
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    sort_eigen = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_eigen]
    
    dim = Q.shape[0]
    if method == 'unnorm':
        normalisation_factor = dim
    else:
        normalisation_factor = dim/(dim-1.)

    vbi_value = eigenvalues[1]/normalisation_factor
    vbi_value = vbi_value.astype(np.float32)
    
    fiedler_vector = eigenvectors[:, sort_eigen[1]]
    if D is None:
        D = np.identity(dim)
    n = np.matmul(fiedler_vector, np.matmul(D, fiedler_vector))
    fiedler_vector = fiedler_vector/np.sqrt(n)

    return vbi_value, fiedler_vector
    

def spectral_reorder(full_brain, B, residual_tolerance, max_num_iter, method='unnorm'):
    """Computes the spectral reorder of the matrix B

    Parameters
    ----------
    B: (M, M) array_like
       Square matrix to be reordered
    method: string
            Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'

    Returns
    -------
    sorted_B: (M, M) numpy array
              Reordered matrix
    sort_idx: (M) numpy array
              Reordering mask applied to B
    eigenvalue: floating-point number
                Second-smallest eigenvalue
    eigenvector: (M) numpy array
                 The Fiedler vector
    """

    # Fix the input
    assert B.shape[0] == B.shape[1], "Matrix B must be square!"

    min_b = np.min(B)
    assert min_b >= -1.0, "This function only accepts matrices with a mininum negative value of -1"

    if min_b < 0:
        warnings.warn("""
        The value 1 is being added to your similarity matrix to ensure positivity.
        This may cause issues with interpretation. Consider inputing a positive matrix""",
        warnings.UserWarning)
        C = B + 1
    else:
        C = B

    # Actual decomposition

    # create the laplacian matrix (Q).
    # For all non-diagonal elements, populate matrix Q with (minus) the corresponding
    # values of matrix C (i.e. Q[i,j] = -C[i,j]).
    # For each diagonal element Q[i,i], sum across the corresponding row of C (excluding the
    # diagonal element C[i,i]) and set Q[i,i] equal to that sum. 

    triuC = np.triu(C,1) # Extract upper triangular elements;
    C = triuC + triuC.transpose() # Reconstruct a symmetric weighted adjacency matrix eliminating possible small errors in off-diagonal elements
    D =  np.diag(np.sum(C, axis=-1)) # Compute the Degree matrix

    Q = D - C; # Compute un-normalised Laplacian

    method = method.lower()

    if method == 'geig':
        # Method using generalised spectral decomposition of the
        # un-normalised Laplacian (see Shi and Malik, 2000)

        vbi_value, eigenvector = get_fiedler_eigenpair(method, full_brain, Q, D, tol=residual_tolerance, maxiter=max_num_iter)

    elif method == 'sym': 
        # Method using the eigen decomposition of the Symmetric Normalized
        # Laplacian. Note that results should be the same as 'geig'
        T = np.sqrt(D)
        L = spl.solve(T, Q)/np.diag(T) #Compute the normalized laplacian
        L = force_symmetric(L) # Force symmetry

        vbi_value, eigenvector = get_fiedler_eigenpair(method, full_brain, L, tol=residual_tolerance, maxiter=max_num_iter)
        eigenvector = spl.solve(T, eigenvector) # automatically normalized (i.e. eigenvector.transpose() @ (D @ eigenvector) = 1)

    elif method == 'rw':
        # Method using eigen decomposition of Random Walk Normalised Laplacian
        L = spl.solve(D, Q)

        vbi_value, eigenvector = get_fiedler_eigenpair(method, full_brain, L, is_symmetric=False, tol=residual_tolerance, maxiter=max_num_iter)
        n = np.matmul(eigenvector.transpose(), np.matmul(D, eigenvector))
        eigenvector = eigenvector/np.sqrt(n)

    elif method == 'unnorm':

        vbi_value, eigenvector = get_fiedler_eigenpair(method, full_brain, Q, tol=residual_tolerance, maxiter=max_num_iter)

    else:
        raise NameError("""Method '{}' not allowed. \n
        Please choose one of the following: 'sym', 'rw', 'geig', 'unnorm'.""".format(method))

    v2 = eigenvector # Fiedler vector
    sort_idx = np.argsort(v2) # Find the reordering index
    sorted_B = B[sort_idx,:] # Reorder the original matrix
    sorted_B = sorted_B[:,sort_idx] # Reorder the original matrix

    return sorted_B, sort_idx, vbi_value, eigenvector


def create_affinity_matrix(neighborhood, eps=np.finfo(float).eps, verbose=False):
    """
    Computes the affinity matrix of a given neighborhood

    Parameters
    ----------
    neighborhood: (M, N) numpy array
    eps: float
         Small value which will replace negative numbers

    Returns
    -------
    affinity: (M, M) numpy array
              Affinity matrix of the neighborhood
    """
    # The following line is necessary for when neighborhood is an (M,) array (i.e. it is 1D), in 
    # which case it is converted to an (M,1) array. 
    neighborhood = np.atleast_2d(neighborhood)
    # Here, the affinity matrix should have n_neighbors x data_size shape
    if neighborhood.shape[1] < 3:
        raise TimeSeriesTooShortError("Time series have less than 3 entries. Your analysis will be compromised!\n")

    # Create a mean centered neighborhood
    neighborhood_mean = np.mean(neighborhood, axis=-1)
    neighborhood_mc = neighborhood - neighborhood_mean.reshape(-1, 1)
    neighborhood_mc[np.abs(neighborhood_mc)<eps] = eps

    if verbose:
        print("neighborhood_mean")
        print(neighborhood_mean)

        print("neighborhood_mc")
        print(neighborhood_mc)

    # Normalise the mean centered neighborhood
    neighborhood_w = np.sqrt(np.sum(neighborhood_mc**2, axis=-1)).reshape(-1, 1)
    neighborhood_scaled = neighborhood_mc/neighborhood_w

    affinity = np.dot(neighborhood_scaled, neighborhood_scaled.transpose())
    affinity[affinity > 1.0] = 1.0
    affinity[affinity < -1.0] = -1.0


    # "Linearlise" the affinity ensure positive correlations are between 0 to 1
    # what i am doing here is to change cosines to angles but to ensure that
    # this remains a similarity rather than dissimilarity I am treating the
    # values as the sine rather than cosine. I.e. identical signals will be 90
    # rather than 0. 90/90 == 1.
    A = np.arcsin(affinity)/np.pi*180.0
    A = A/90.0
    # Remove negative correlations
    A[A<0] = eps

    if verbose:
        print("affinity")
        print(affinity)
        print("A")
        print(A)

    return A


class MultilineFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        text = self._whitespace_matcher.sub(' ', text).strip()
        paragraphs = text.split('|n ')
        multiline_text = ''
        for paragraph in paragraphs:
            formatted_paragraph = _textwrap.fill(paragraph, width, initial_indent=indent, subsequent_indent=indent) + '\n\n'
            multiline_text = multiline_text + formatted_paragraph
        return multiline_text

def create_temp_folder():
    
    if os.path.exists("/tmp"):
        if os.path.exists("/tmp/temp_folder"):
            shutil.rmtree("/tmp/temp_folder")            
        os.mkdir("/tmp/temp_folder")
        print("\n[+] A temporary folder has been created in /tmp")
    else:
        os.mkdir("temp_folder")        
        print("\n[+] A temporary folder has been created in the actual folder, do not remove it until the analysis ends")
        
    if not os.path.exists("/tmp/temp_folder") and not os.path.exists("temp_folder"):
        raise Exception("[!] Folder could't have been created")
   
def clean_screen():
    
    os.system('cls' if os.name == 'nt' else 'clear')
   
def compute_temporal_analysis(window_size, steps, size, path, analysis_type, affine, n_cpus, nib, norm, cort_index, residual_tolerance, max_num_iter, output_name, reho, header=None, brain_mask=None, data=None, surf_vertices=None, nib_surf=None, k=None, surf_faces=None, debug=None):
    
    create_temp_folder()
    
    print("Running Temporal Analysis")
    
    if analysis_type == "vb_hybrid":
        temp = np.zeros([surf_vertices.shape[0], size], dtype=np.float32)
    else:
        temp = np.zeros([nib.shape[0],nib.shape[1],nib.shape[2],size], dtype=np.float32)
      
    data = np.array(nib.dataobj)
    time.sleep(2) 
    
    j = 0
    
    for i in range(0,data.shape[3]-steps,steps):
        
        clean_screen()
        print(f"[+] Iteration number: {i}")       
        print(f"[+] Progress: {round(i/(data.shape[3]-steps)*100, 2)}%")
        
#        if i == 130:
#            break
        
        vol = np.array(nib.dataobj[:,:,:,i:i+window_size])
        if analysis_type == "vb_hybrid":
            result = compute_vb_metrics(analysis_type, surf_vertices=surf_vertices, surf_faces=surf_faces, affine=affine, n_cpus=n_cpus, data=vol, norm=norm, cort_index=cort_index, residual_tolerance=residual_tolerance, max_num_iter=max_num_iter, output_name=output_name, nib_surf=nib_surf, k=k, reho=reho, debug=debug)
            temp[:,j] = result
        else:
            result = compute_vb_metrics(analysis_type, n_cpus=n_cpus, data=vol, affine=affine, header=header, norm=norm, cort_index=cort_index, brain_mask=brain_mask, residual_tolerance=residual_tolerance, max_num_iter=max_num_iter, output_name=output_name, reho=reho)
            temp[:,:,:,j] = result
        
        if j == size-1:
            if not glob.glob(path+"*"):
                if analysis_type == "vb_hybrid":                
                    save_gifti(nib_surf, temp, path+f"conc{i}.shape.gii") 
                else:
                    save_nifti(result=temp, affine=nib.affine, header=nib.header, output_name=path+f"conc{i}")
            else:
                if analysis_type == "vb_hybrid": 
                    save_gifti(nib_surf, temp, path+f"conc{i}.shape.gii")       
                    concatenate_gifti_images(path, nib_surf, data, path+f"Conc{i}.shape.gii")
                else:
                    save_nifti(result=temp, affine=nib.affine, header=nib.header, output_name=path+f"conc{i}")
                    concatenate_nifti_images(path, path+f"Conc{i}.vbi-vol.nii.gz")                    
            j = 0
            if analysis_type == "vb_hybrid":
                temp = np.zeros([surf_vertices.shape[0], size], dtype=np.float32)
            else:
                temp = np.zeros([nib.shape[0],nib.shape[1],nib.shape[2],size], dtype=np.float32)
        else:
            j += 1
            
    if not np.all(temp==0):
        if analysis_type == "vb_hybrid":    
            save_gifti(nib_surf, temp, path+"conc_f.shape.gii")                      
            concatenate_gifti_images(path, nib_surf, data, f"{output_name}.shape.gii")
        else:
            save_nifti(result=temp, affine=nib.affine, header=nib.header, output_name=path+f"conc{i}")
            concatenate_nifti_images(path, f"{output_name}.vbi-vol.nii.gz")
    else:        
        image = glob.glob(path+"*")
        if analysis_type == "vb_hybrid":  
            os.rename(image[0], f"{output_name}.shape.gii")
        else:
            os.rename(image[0], f"{output_name}.vbi-vol.nii.gz")
    
        
def concatenate_gifti_images(path, nib_surf, data, filename):
    
    images = glob.glob(path+"*")
    image1 = nibabel.load(images[0])
    image2 = nibabel.load(images[1])
    
    d_image1 = image1.darrays[0].data
    d_image2 = image2.darrays[0].data
    
    concat_data = np.concatenate((d_image1, d_image2), axis=1)
    save_gifti(nib_surf, concat_data, filename)   
    
    for file in images:
        os.system(f"rm {file}")
    
def concatenate_nifti_images(path, filename):

    images = glob.glob(path+"*")
    concat_img = nibabel.funcs.concat_images(images,axis=3)
    nibabel.save(concat_img, filename)
    
    for file in images:
        os.system(f"rm {file}")

def create_parser():
    authors = '''authors:
    |n   The VB Index Team (See Contributors Section in the main README)'''
    copyright = '''copyright:|n
        This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.|n
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.|n
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses>.
    '''
    references = '''references:|n
        Bajada, C. J., Campos, L. Q. C., Caspers, S., Muscat, R., Parker, G. J., Ralph, M. A. L., ... & Trujillo-Barreto, N. J. (2020). A tutorial and tool for exploring feature similarity gradients with MRI data. NeuroImage, 221, 117140.|n
        Ciantar, K. G., Farrugia, C., Scerri, K., Xu, T., & Bajada, C. J. (2020). Geometric effects of volume-to-surface mapping of fMRI data. bioRxiv.
    '''
    
    parser = argparse.ArgumentParser(description='Calculate the Vogt-Bailey index of a dataset. For more information, refer to https://github.com/VBIndex/py_vb_toolbox.',
                                     epilog=authors + " |n " + references + " |n " + copyright,
                                     formatter_class=MultilineFormatter)
    parser.add_argument('-j', '--jobs', metavar='N', type=int, nargs=1,
                        default=[multiprocessing.cpu_count()], help="""Maximum
                        number of jobs to be used. If absent, one job per CPU
                        will be spawned.""")

    parser.add_argument('-n', '--norm', metavar='norm', type=str, nargs=1,
                        help="""Laplacian normalization to be
                        employed. Possibilities are "geig", "unnorm", "rw" and
                        "sym". Defaults to geig for the full brain and ROI analyses, and to unnorm otherwise.""")

    parser.add_argument('-fb', '--full-brain', action='store_true',
                        help="""Calculate full brain feature gradient analysis.""")

    parser.add_argument('-hy', '--hybrid', action='store_true',
                        help="""Calculate searchlight VB index with hybrid approach.""")
    
    parser.add_argument('-rh', '--reho', action='store_true',
                        help="""Calculate the KCC index for ReHo approach.""")
    
    parser.add_argument('-vol', '--volume', action='store_true',
                        help="""Do not map results to surface.""")
    
    parser.add_argument('-ta', '--temporal-analysis', action='store_true',
                        help="""FALTA POR PONER.""")
    
    parser.add_argument('-ws', '--window-size', metavar='integer', type=int, nargs=1, default=10,
                        help="""Window size for Temporal Analysis.""")
    
    parser.add_argument('-st', '--step', metavar='integer', type=int, nargs=1, default=1,
                        help="""Step for Temporal Analysis.""")
    
    parser.add_argument('-sz', '--size', metavar='integer', type=int, nargs=1, default=3,
                        help="""Size for Temporal Analysis.""")
    
    parser.add_argument('-p', '--path', metavar='integer', type=str, nargs=1, default="/tmp/temp_folder/",
                        help="""Path for temporal folder""")
                        
    parser.add_argument('-vm', '--volmask', metavar='file', type=str,
                        nargs=1, default=None, help="""Nifti file containing the whole brain mask
                        in volumetric space. Only relevant if computing the volumetric VB.""")

    parser.add_argument('-m', '--mask', metavar='file', type=str,
                               nargs=1, help="""File containing the labels to
                               identify the cortex, rather than the medial
                               brain structures. This flag must be set for
                               the searchlight and full brain analyses.""")

    parser.add_argument('-c', '--clusters', metavar='file', type=str, nargs=1, default=None,
                        help="""File specifying the surface clusters. The cluster
                        with index 0 is expected to denote the medial brain
                        structures and will be ignored.""")
                        
    parser.add_argument('-t', '--tol', metavar='tolerance', type=float, nargs=1,
                        default=["def_tol"], help="""Residual tolerance (stopping criterion) for LOBPCG. 
                        Default value = sqrt(10e-18)*n, where n is the number of nodes per graph. Note that
                        the LOBPCG algorithm is only utilised for full-brain analysis.""")
                        
    parser.add_argument('-mi', '--maxiter', metavar='max iterations', type=int, nargs=1, default=[50],
                        help="""Maximum number of iterations for LOBPCG. Defaults to 50.""")
    
    parser.add_argument('-debug', '--debug', action='store_true',
                        help="""Save additional files for debugging.""")

    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument('-s', '--surface', metavar='file', type=str,
                               nargs=1, help="""File containing the surface
                                              mesh.""", required=False)

    requiredNamed.add_argument('-d', '--data', metavar='file', type=str,
                               nargs=1, help="""File containing the data over
                                              the surface (or volume if hybrid).""", required=True)


    requiredNamed.add_argument('-o', '--output', metavar='file', type=str,
                               nargs=1, help="""Base name for the
                                              output files.""", required=True)

    return parser


def main():
    """
    Main function where argument parser and main logic is handled.

    Returns
    -------
    None.

    """
    
    parser = create_parser()
    args = parser.parse_args()

    n_cpus = args.jobs[0]
    if not args.volume:
        nib_surf, vertices, faces = open_gifti_surf(args.surface[0])

        # Get the text contents from the file
        surf_text = open(args.surface[0], 'r', encoding='latin-1')

        hemi = None

        # Check whether the file is left or right cortex
        for line in surf_text:
            if 'CortexLeft' in line:
                hemi = 'CortexLeft'
                break
            elif 'CortexRight' in line:
                hemi = 'CortexRight'
                break
            
        # Add the cortex information to the beginning of the meta data
        if hemi:
            nib_surf.meta['AnatomicalStructurePrimary'] = hemi

    nib = nibabel.load(args.data[0],mmap=False)
    if (args.hybrid or args.volume):
        data = np.array(nib.dataobj)
        affine = nib.affine
        header = nib.header
    else:
        if len(nib.darrays) > 1:
            data = np.array([n.data for n in nib.darrays]).transpose()
        else:
            data = nib.darrays[0].data
            
    if (args.norm is not None and args.norm[0] == 'rw'):
        print('Warning: this method makes use of the Random Walk Normalized Laplacian, and has not been tested rigorously yet.')
    if (args.norm is not None and args.norm[0] == 'sym'):
        print('Warning: this method makes use of the Symmetric Normalized Laplacian, and has not been tested rigorously yet.')
            
    if (args.norm is not None and args.norm[0] == 'rw'):
        print('Warning: this method makes use of the Random Walk Normalized Laplacian, and has not been tested rigorously yet.')
    if (args.norm is not None and args.norm[0] == 'sym'):
        print('Warning: this method makes use of the Symmetric Normalized Laplacian, and has not been tested rigorously yet.')

    if args.full_brain:
        print("Running full brain analysis")
        if args.mask is None:
            sys.stderr.write("A mask file must be provided through the --mask flag. See --help")
            sys.exit(2)
            quit()
        # Read labels
        _, labels = open_gifti(args.mask[0])
        cort_index = np.array(labels, bool)
        Z = np.array(cort_index, dtype=int)
        if args.norm is None:
            L_norm = 'geig'
        else:
            L_norm = args.norm[0]
        try:
            result = compute_vb_metrics("vb_cluster", surf_vertices=vertices, surf_faces=faces, n_cpus=n_cpus, data=data, norm=L_norm, residual_tolerance=args.tol[0], max_num_iter=args.maxiter[0], output_name=args.output[0] + "." + L_norm, nib_surf=nib_surf, cluster_index=Z, full_brain=True, debug=args.debug)
            #full_brain=True
        except Exception as error:
            sys.stderr.write(str(error))
            sys.exit(2)
            quit()
            
    elif args.volume:
        if args.reho:
            print("Running ReHo approach with no surface mapping")
        else:
            print("Running searchlight analysis with no surface mapping")
        if args.mask:
            _, labels = open_gifti(args.mask[0])
            cort_index = np.array(labels, bool)
        else:
            cort_index = None
        if args.norm is None:
            L_norm = 'unnorm'
        else:
            L_norm = args.norm[0]
        if args.volmask:
            brain_mask = nibabel.load(args.volmask[0],mmap=False)
            brain_mask = np.array(brain_mask.dataobj)
        else:
            brain_mask = None
        if args.temporal_analysis:
            if not type(args.window_size) == int:
                window_size = args.window_size[0]
            else:
                window_size = args.window_size
            if not type(args.step) == int:
                steps = args.step[0]
            else:
                steps = args.step
            if not type(args.size) == int:
                size = args.size[0]
            else:
                size = args.size
            try:
                compute_temporal_analysis(window_size=window_size, steps=steps, size=size, path=args.path, analysis_type="vb_vol", n_cpus=n_cpus, nib=nib, data=data, affine=affine, header=header, norm=L_norm, cort_index=cort_index, brain_mask=brain_mask, residual_tolerance=args.tol[0], max_num_iter=args.maxiter[0], output_name=args.output[0], reho=args.reho)
                sys.exit(1)
            except Exception as error:
                sys.stderr.write(str(error))
                sys.exit(2)
                quit()
        try:
                result = compute_vb_metrics(internal_loop_func="vb_vol", n_cpus=n_cpus, data=data, affine=affine, header=header, norm=L_norm, cort_index=cort_index, brain_mask=brain_mask, residual_tolerance=args.tol[0], max_num_iter=args.maxiter[0], output_name=args.output[0] + "." + L_norm, reho=args.reho)
            
        except Exception as error:
            sys.stderr.write(str(error))
            sys.exit(2)
            quit()    

    elif args.clusters is None:
        if args.hybrid:
            if args.norm is None:
                L_norm = 'unnorm'
            else:
                L_norm = args.norm[0]
            _, labels = open_gifti(args.mask[0])
            cort_index = np.array(labels, bool)
            if args.temporal_analysis:
                if not type(args.window_size) == int:
                    window_size = args.window_size[0]
                else:
                    window_size = args.window_size
                if not type(args.step) == int:
                    steps = args.step[0]
                else:
                    steps = args.step
                if not type(args.size) == int:
                    size = args.size[0]
                else:
                    size = args.size
                try:
                    compute_temporal_analysis(window_size=window_size, steps=steps, size=size, path=args.path, analysis_type="vb_hybrid", surf_vertices=vertices, surf_faces=faces, affine=affine, n_cpus=n_cpus, nib=nib, norm=L_norm, cort_index=cort_index, residual_tolerance=args.tol[0], max_num_iter=args.maxiter[0], output_name=args.output[0], nib_surf=nib_surf, k=3, reho=args.reho, debug=args.debug)
                    sys.exit(1)
                except Exception as error:
                    sys.stderr.write(str(error))
                    sys.exit(2)
                    quit()
            if args.reho:
                print("Running ReHo approach")
            else:
                print("Running searchlight analysis with hybrid approach")
            if args.mask is None:
                sys.stderr.write("A mask file must be provided through the --mask flag. See --help")
                sys.exit(2)
                quit()
          
            # Read labels
            try:
                result = compute_vb_metrics("vb_hybrid", surf_vertices=vertices, surf_faces=faces, affine=affine, n_cpus=n_cpus, data=data, norm=L_norm, cort_index=cort_index, residual_tolerance=args.tol[0], max_num_iter=args.maxiter[0], output_name=args.output[0], nib_surf=nib_surf, k=3, reho=args.reho, debug=args.debug)
            except Exception as error:
                sys.stderr.write(str(error))
                sys.exit(2)
                quit()
        else:
            print("Running searchlight analysis")
            if args.mask is None:
                sys.stderr.write("A mask file must be provided through the --mask flag. See --help")
                sys.exit(2)
                quit()
            # Read labels
            _, labels = open_gifti(args.mask[0])
            cort_index = np.array(labels, bool)
            if args.norm is None:
                L_norm = 'unnorm'
            else:
                L_norm = args.norm[0]
            try:
                result = compute_vb_metrics("vb_index", surf_vertices=vertices, surf_faces=faces, n_cpus=n_cpus, data=data, norm=L_norm, cort_index=cort_index, residual_tolerance=args.tol[0], max_num_iter=args.maxiter[0], output_name=args.output[0] + "." + L_norm, nib_surf=nib_surf)
            except Exception as error:
                sys.stderr.write(str(error))
                sys.exit(2)
                quit()           
                     
    else:
        print("Running ROI analysis")
        if args.clusters is None:
            sys.stderr.write("A cluster file must be provided through the --clusters flag. See --help")
            sys.exit(2)
            quit()
        nib, Z = open_gifti(args.clusters[0])
        Z = np.array(Z, dtype=np.int)
        if args.norm is None:
            L_norm = 'geig'
        else:
            L_norm = args.norm[0]
        try:
            result = compute_vb_metrics("vb_cluster", False, vertices, faces, n_cpus, data, L_norm, args.tol[0], args.maxiter[0], args.output[0] + "." + L_norm, nib_surf, k=3, cluster_index=Z, cort_index=cort_index, affine=affine, debug=args.debug)
        except Exception as error:
            sys.stderr.write(str(error))
            sys.exit(2)
            quit()


if __name__ == "__main__":
    main()