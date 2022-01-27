#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Lucas Costa Campos <rmk236@gmail.com>
#
# Distributed under terms of the GNU license.

import nibabel
import numpy as np
import scipy.linalg as spl
import traceback
import vb_toolbox.io as io
import vb_toolbox.numerics as m

counter = None
n = None

from multiprocessing import Pool, Value, Lock
import multiprocessing
from itertools import product

def init(a_counter, a_n):
    """Store total number of vertices and counter of vertices computed"""
    global counter
    global n
    counter = a_counter
    n = a_n

def vb_index_internal_loop(i0, iN, surf_faces, data, norm, residual_tolerance, max_num_iter, print_progress=False):
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
       print_progress: boolean
           Print the current progress of the system

       Returns
       -------
       loc_result: (N) numpy array
                   Resulting VB index of the indices in range. Will have length iN - i0
    """

    # Calculate how many vertices we will compute
    diff = iN - i0
    loc_result = np.zeros(diff)

    for idx in range(diff):
        # Calculate the real index
        i = idx + i0

        # Get neighborhood and its data
        # TODO: Make this elegant
        try:
            neighbour_idx = np.array(np.sum(surf_faces == i, 1), np.bool)
            I = np.unique(surf_faces[neighbour_idx, :])
            neighborhood = data[I]
            if len(neighborhood) == 0:
                print("Warning: no neighborhood")
                return [0]

            # Calculate the second smallest eigenvalue
            affinity = m.create_affinity_matrix(neighborhood)
            _, _, eigenvalue, _ = m.spectral_reorder(affinity, residual_tolerance, max_num_iter, norm)

            # return [0]
            # Store the result of this run
            loc_result[idx] = eigenvalue
        except m.TimeSeriesTooShortError as error:
            raise error
        except Exception:
            loc_result[idx] = np.nan

        if print_progress:

            global counter
            global n
            with counter.get_lock():
                counter.value += 1
            if counter.value % 1000 == 0:
                print("{}/{}".format(counter.value, n))

    return loc_result

def vb_index(surf_vertices, surf_faces, n_cpus, data, norm, cort_index, residual_tolerance, max_num_iter, output_name=None, nib_surf=None):
    """Computes the Vogt-Bailey index of vertices for the whole mesh

       Parameters
       ----------
       surf_vertices: (M, 3) numpy array
             Vertices of the mesh
       surf_faces: (M, 3) numpy array
             Faces of the mesh. Used to find the neighborhood of a given vertex
       n_cpus: integer
             How many CPUS are available to run the calculation
       data: (M, N) numpy array
             Data to use to calculate the VB index. M must match the number of vertices in the mesh
       norm: string
             Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'
       cort_index: (M) numpy array
             Mask for detection of middle brain structures
       output_name: string
             Root of file to save the results to. If specified, nib_surf must also be provided
       nib_surf: nibabel object
             Nibabel object containing metadata to be replicated

       Returns
       -------
       result: (N) numpy array
               Resulting VB index of the indices in range
    """
    
    # Calculate how many vertices each process is going to be responsible for
    n_items = len(surf_vertices)
    n_cpus = min(n_items, n_cpus)
    dn = n_items//(n_cpus)

    # Init multiprocessing components
    counter = Value('i', 0)
    pool = Pool(initializer = init, initargs = (counter, n_items))

    def callback(result):
        pool.close()
        pool.terminate()

    # vb_index_internal_loop(0, n_items, surf_faces, data, norm)
    # Spawn the threads that are going to do the real work
    threads = []
    for i0 in range(0, n_items, dn):
        iN = min(i0+dn, n_items)
        threads.append(pool.apply_async(vb_index_internal_loop, (i0, iN, surf_faces, data, norm, residual_tolerance, max_num_iter), error_callback=callback))


    # Gather the results from the threads we just spawned
    results = []
    for i, res in enumerate(threads):
        res_ = res.get()
        for r in res_:
            results.append(r)
    results = np.array(results)
    results[np.logical_not(cort_index)] = np.nan

    # Save file
    if output_name is not None:
        io.save_gifti(nib_surf, results, output_name + ".vbi.shape.gii")

    # Cleanup
    pool.close()
    pool.terminate()
    pool.join()

    return results

def vb_cluster_internal_loop(idx_cluster_0, idx_cluster_N, surf_faces, data, cluster_index, norm, residual_tolerance, max_num_iter, print_progress=False):
    """Computes the Vogt-Bailey index and Fiedler vector of vertices of given clusters

       Parameters
       ----------
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
       print_progress: boolean
           Print the current progress of the system

       Returns
       -------
       loc_result: list of pairs of (float, (N) numpy array)
                   Resulting VB index and Fiedler vector for each of the clusters in range
    """

    # Calculate how many clusters we will work with
    diff = idx_cluster_N - idx_cluster_0
    loc_result = []
    cluster_labels = np.unique(cluster_index)

    for idx in range(diff):
        #Calculate the real index
        i = idx + idx_cluster_0

        if(cluster_labels[i] == 0):
            loc_result.append(([], []))
            continue
        try:
            # Get neighborhood and its data
            neighborhood = data[cluster_index == cluster_labels[i]]

            # Calculate the Fiedler eigenpair
            affinity = m.create_affinity_matrix(neighborhood)
            _, _, eigenvalue, eigenvector = m.spectral_reorder(affinity, residual_tolerance, max_num_iter, norm)

            # Store the result of this run
            # Warning: It is not true that the eigenvectors will be all the same
            # size, as the clusters might be of different sizes
            val = eigenvalue
            vel = eigenvector
            loc_result.append((val, vel))
        except m.TimeSeriesTooShortError as error:
            raise error


        if print_progress:

            global counter
            global n
            with counter.get_lock():
                counter.value += 1
            if counter.value % 1000 == 0:
                print("{}/{}".format(counter.value, n))

    return loc_result

def vb_cluster(surf_vertices, surf_faces, n_cpus, data, cluster_index, norm, residual_tolerance, max_num_iter, output_name = None, nib_surf=None):
    """Computes the clustered Vogt-Bailey index and Fiedler vector of vertices for the whole mesh

       Parameters
       ----------
       surf_vertices: (M, 3) numpy array
           Vertices of the mesh
       surf_faces: (M, 3) numpy array
           Faces of the mesh. Used to find the neighborhood of a given vertex
       n_cpus: integer
           How many CPUS are available to run the calculation
       data: (M, N) numpy array
           Data to use to calculate the VB index and Fiedler vector. M must match the number of vertices in the mesh
       cluster_index: (M) numpy array
           Array containing the cluster which each vertex belongs to
       norm: string
           Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'
       cort_index: (M) numpy array
           Mask for detection of middle brain structures
       output_name: string
           Root of file to save the results to. If specified, nib_surf must also be provided
       nib_surf: nibabel object
           Nibabel object containing metadata to be replicated

       Returns
       -------
       results_eigenvalues: (M) numpy array
                            Resulting VB index of the clusters
       results_eigenvectors: (M, N) numpy array
                            Resulting Fiedler vectors of the clusters
    """

    # Find the cluster indices, and the midbrain structures
    cluster_labels = np.unique(cluster_index)
    midline_index = cluster_index == 0

    # Calculate how many clusters each process is going to be responsible for
    n_items = len(cluster_labels)
    n_cpus = min(n_items, n_cpus)
    dn = n_items//(n_cpus)

    # Init multiprocessing components
    counter = Value('i', 0)
    pool = Pool(initializer = init, initargs = (counter, n_items))

    def callback(result):
        pool.close()
        pool.terminate()

    # Spawn the threads that are going to do the real work
    threads = []
    for i0 in range(0, n_items, dn):
        iN = min(i0+dn, n_items)
        threads.append(pool.apply_async(vb_cluster_internal_loop, (i0, iN, surf_faces, data, cluster_index, norm, residual_tolerance, max_num_iter), error_callback=callback))


    # Gather the results from the threads we just spawned
    results = []
    results_eigenvectors_l = []
    for i, res in enumerate(threads):
        res_ = res.get()
        for r, rv in res_:
            results.append(r)
            results_eigenvectors_l.append(rv)
    results = np.array(results)

    # Now we need to push the data back to the original vertices
    results_eigenvalues = np.zeros(len(surf_vertices))
    results_eigenvectors = []
    for i in range(n_items):
        cluster = cluster_labels[i]
        if cluster != 0:
            results_eigenvectors_local = np.zeros(len(surf_vertices))
            idx = np.where(cluster_index == cluster)[0]
            results_eigenvalues[idx] = results[i]
            results_eigenvectors_local[idx] = results_eigenvectors_l[i]
            results_eigenvectors.append(results_eigenvectors_local)

    results_eigenvectors = np.array(results_eigenvectors).transpose()

    # Remove the midbrain
    results_eigenvalues[midline_index] = np.nan
    results_eigenvectors[midline_index, :] = np.nan

    # Save file
    if output_name is not None:
        io.save_gifti(nib_surf, results_eigenvalues, output_name + ".vb-cluster.value.shape.gii")
        io.save_gifti(nib_surf, results_eigenvectors, output_name + ".vb-cluster.vector.shape.gii")

    # Cleanup
    pool.close()
    pool.terminate()
    pool.join()

    return results_eigenvalues, results_eigenvectors
	
def get_neighborhood(data,p,mask,n=1):
    """Get neighbors in volumetric space given the coordinates of a vertex"""
    neigh_coords = np.array([relative_index for relative_index in product((-1, 0, 1), repeat=3)])+p
    neigh_coords = neigh_coords.astype(int)
    masked_neigh = np.where(mask[neigh_coords[:,0],neigh_coords[:,1], neigh_coords[:,2]])[0]

    return data[neigh_coords[masked_neigh,0],neigh_coords[masked_neigh,1], neigh_coords[masked_neigh,2],:]

def vb_hybrid_internal_loop(i0, iN, surf_vertices, brain_mask, data, norm, residual_tolerance, max_num_iter, print_progress=False):
    """Computes the Vogt-Bailey index of vertices in a given range

       Parameters
       ----------
       i0: integer
           Index of first vertex to be analysed
       iN: integer
           iN - 1 is the index of the last vertex to be analysed
       surf_vertices: (M, 3) numpy array
           Coordinates of vertices of the mesh in voxel space
       brain_mask: (nRows, nCols, nSlices) numpy array
           Whole brain mask. Used to mask volumetric data
       data: (nRows, nCols, nSlices, N) numpy array
           Volumetric data used to calculate the VB index. N is the number of maps
       norm: string
           Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'
       print_progress: boolean
           Print the current progress of the system

       Returns
       -------
       loc_result: (N) numpy array
                   Resulting VB index of the indices in range. Will have length iN - i0
    """
    
    # Calculate how many vertices we will compute
    diff = iN - i0
    loc_result = np.zeros(diff)

    for idx in range(diff):
        #Calculate the real index
        i = idx + i0

        # Get neighborhood and its data
        # print(data.shape)
        try:
            neighborhood = get_neighborhood(data,surf_vertices[i,:],brain_mask)
            if len(neighborhood) == 0:
                print("Warning: no neighborhood")
                loc_result[idx] = np.nan
                continue
            affinity = m.create_affinity_matrix(neighborhood)
            
            if affinity.shape[0] > 3:
                #tr_row, tr_col = np.triu_indices(affinity.shape[0], k=1)
            
                # Calculate the second smallest eigenvalue
                _, _, eigenvalue, _ = m.spectral_reorder(affinity, residual_tolerance, max_num_iter, norm)
                # return [0]
                # Store the result of this run
                loc_result[idx] = eigenvalue
                #loc_result[idx] = np.mean(affinity[tr_row, tr_col])
            else:
                loc_result[idx] = np.nan
        except m.TimeSeriesTooShortError as error:
            raise error
        except Exception:
            traceback.print_exc()
            loc_result[idx] = np.nan
        

        if print_progress:

            global counter
            global n
            with counter.get_lock():
                counter.value += 1
            if counter.value % 1000 == 0:
                print("{}/{}".format(counter.value, n))

    return loc_result
	
def vb_hybrid(surf_vertices, brain_mask, affine, n_cpus, data, norm, cort_index, residual_tolerance, max_num_iter, output_name=None, nib_surf=None):
    """Computes the Vogt-Bailey index of vertices for the whole mesh

       Parameters
       ----------
       surf_vertices: (M, 3) numpy array
           Vertices of the mesh
       brain_mask: (nRows, nCols, nSlices) numpy array
           Whole brain mask. Used to mask volumetric data
       n_cpus: integer
           How many CPUS are available to run the calculation
       data: (nRows, nCols, nSlices, N) numpy array
           Volumetric data used to calculate the VB index. N is the number of maps
       norm: string
           Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'
       cort_index: (M) numpy array
           Mask for detection of middle brain structures
       output_name: string
           Root of file to save the results to. If specified, nib_surf must also be provided
       nib_surf: nibabel object
           Nibabel object containing metadata to be replicated

       Returns
       -------
       result: (N) numpy array
               Resulting VB index of the indices in range
    """
    
    # Convert vertex coordinates to voxel coordinates
    vox_coords = np.round(nibabel.affines.apply_affine(np.linalg.inv(affine),surf_vertices))

    # Calculate how many vertices each process is going to be responsible for
    n_items = len(surf_vertices)
    n_cpus = min(n_items, n_cpus)
    dn = n_items//(n_cpus)

    # Init multiprocessing components
    counter = Value('i', 0)
    pool = Pool(initializer = init, initargs = (counter, n_items))

    def callback(result):
        pool.close()
        pool.terminate()

    # Spawn the threads that are going to do the real work
    threads = []
    for i0 in range(0, n_items, dn):
        iN = min(i0+dn, n_items)
        threads.append(pool.apply_async(vb_hybrid_internal_loop, (i0, iN, vox_coords, brain_mask, data, norm, residual_tolerance, max_num_iter), error_callback=callback))


    # Gather the results from the threads we just spawned
    results = []
    for i, res in enumerate(threads):
        res_ = res.get()
        for r in res_:
            results.append(r)
    results = np.array(results)
    results[np.logical_not(cort_index)] = np.nan
    
    # Save file
    if output_name is not None:
        io.save_gifti(nib_surf, results, output_name + ".vbi-hybrid.shape.gii")

    # Cleanup
    pool.close()
    pool.terminate()
    pool.join()

    return results
