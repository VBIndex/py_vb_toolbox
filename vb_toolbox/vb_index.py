#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Lucas Costa Campos <l.campos@fz-juelich.de>
#
# Distributed under terms of the GPL license.

import nibabel
import numpy as np
import scipy.linalg as spl

import vb_toolbox.io as io
import vb_toolbox.math as m

counter = None
n = None

from multiprocessing import Pool, Value, Lock
import multiprocessing

def init(a_counter, a_n):
    """Store total number of vertices and counter of vertices computed"""
    global counter
    global n
    counter = a_counter
    n = a_n

def vb_index_internal_loop(i0, iN, surf_faces, data, norm, print_progress=False):
    """Computes the Voigt-Bailey index of vertices in a given range"""

    # print(f"Doing {i0} to {iN}")

    # Calculate how many vertices we will compute
    diff = iN - i0
    loc_result = np.zeros(diff)

    for idx in range(diff):
        #Calculate the real index
        i = idx + i0

        # Get neighborhood and its data
        # TODO: Make this elegant
        neighbour_idx = np.array(np.sum(surf_faces == i, 1), np.bool)
        I = np.unique(surf_faces[neighbour_idx, :])
        neighborhood = data[I]
        if len(neighborhood) == 0:
            print("Warning: no neighborhood")

        # __import__('pdb').set_trace()
        # Calculate the eigenvalues
        affinity = m.create_affinity_matrix(neighborhood)
        _, _, _, _, eigenvalues, _ = m.spectral_reorder(affinity, norm)
        normalisation_factor = np.average(eigenvalues[1:])

        # Store the result of this run
        loc_result[idx] = eigenvalues[1]/normalisation_factor

        if print_progress:

            global counter
            global n
            with counter.get_lock():
                counter.value += 1
            print("{}/{}".format(counter.value, n))

    return loc_result

def vb_index(surf_vertices, surf_faces, nib_surf, n_cpus, data, norm, cort_index, output_name = None):
    """Computes the Voigt-Bailey index of vertices for the whole mesh"""

    #To check with CJB: Why can't we use rw or sym?
    if norm == "unnorm" or norm == "geig":
        pass
    else:
        raise NameError("""Norm '{}' not allowed. \n
        Please choose one of the following: 'geig', 'unnorm'.""".format(norm))

    # Calculate how many vertices each process is going to be responsible for
    n_items = len(surf_vertices)
    n_cpus = min(n_items, n_cpus)
    dn = n_items//(n_cpus)

    # vb_index_internal_loop(0, n_items, surf_faces, data, norm)
    # Init multiprocessing components
    counter = Value('i', 0)
    pool = Pool(initializer = init, initargs = (counter, n_items))

    # r = vb_index_internal_loop(0, n_items, surf_faces, data, norm)
    # Spawn the threads that are going to do the real work
    threads = []
    for i0 in range(0, n_items, dn):
        iN = min(i0+dn, n_items)
        threads.append(pool.apply_async(vb_index_internal_loop, (i0, iN, surf_faces, data, norm)))


    # Gather the results from the threads we just spawned
    results = []
    for i, res in enumerate(threads):
        for r in res.get():
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

def vb_cluster_internal_loop(idx_cluster_0, idx_cluster_N, surf_faces, data, cluster_index, norm, print_progress=False):
    """Computes the Voigt-Bailey index of vertices in a given range"""

    print(f"Doing {idx_cluster_0} to {idx_cluster_N}")

    # Calculate how many vertices we will compute
    diff = idx_cluster_N - idx_cluster_0
    loc_result = []

    for idx in range(diff):
        #Calculate the real index
        i = idx + idx_cluster_0

        # Get neighborhood and its data
        neighborhood = data[cluster_index == (i+1)]

        # Calculate the eigenvalues
        affinity = m.create_affinity_matrix(neighborhood)
        _, _, _, _, eigenvalues, eigenvectors = m.spectral_reorder(affinity, norm)
        normalisation_factor = sum(eigenvalues)/len(eigenvalues-1)

        # Store the result of this run
        # Warning: It is not true that the eigenvectors will be all the same
        # size, as the clusters might be of different sizes
        val = eigenvalues[1]/normalisation_factor
        vel = eigenvectors[1]
        loc_result.append((val, vel))

        if print_progress:

            global counter
            global n
            with counter.get_lock():
                counter.value += 1
            print("{}/{}".format(counter.value, n))

    return loc_result

def vb_cluster(surf_vertices, surf_faces, nib_surf, n_cpus, data, cluster_index, norm, cort_index, output_name = None):
    """Computes the Voigt-Bailey index of vertices for the whole mesh"""

    print(cort_index)
    #To check with CJB: Why can't we use rw or sym?
    if norm == "unnorm" or norm == "geig":
        pass
    else:
        raise NameError("""Norm '{}' not allowed. \n
        Please choose one of the following: 'geig', 'unnorm'.""".format(norm))

    # Calculate how many vertices each process is going to be responsible for
    n_items = len(np.unique(cluster_index))
    n_cpus = min(n_items, n_cpus)
    # vb_cluster_internal_loop(0, n_items, surf_faces, data, cluster_index, norm)
    # return []
    dn = n_items//(n_cpus)

    # Init multiprocessing components
    counter = Value('i', 0)
    pool = Pool(initializer = init, initargs = (counter, n_items))

    # Spawn the threads that are going to do the real work
    threads = []
    for i0 in range(0, n_items, dn):
        iN = min(i0+dn, n_items)
        threads.append(pool.apply_async(vb_cluster_internal_loop, (i0, iN, surf_faces, data, cluster_index, norm)))


    # Gather the results from the threads we just spawned
    results = []
    results_eigenvectors = []
    for i, res in enumerate(threads):
        for r, rv in res.get():
            results.append(r)
            results_eigenvectors.append(rv)
    results = np.array(results)

    # Now we need to push the data back into the original vertices
    results_vertices = results[cluster_index-1]
    results_vertices_eigenvectors = []
    for i in range(len(surf_vertices)):
        results_vertices_eigenvectors.append(results_eigenvectors[cluster_index[i]-1])

    # Remove the corpus collusum
    results_vertices[np.logical_not(cort_index)] = np.nan
    for i in range(len(results_vertices_eigenvectors)):
        if not cort_index[i]:
            results_vertices_eigenvectors[i][:] = np.nan

    #Now we need to make the eigenvectors into a nice matrix
    max_size = 0
    for v in results_vertices_eigenvectors:
        max_size = max(max_size, len(v))

    matrix_eigenvectors = np.ones((len(surf_vertices), max_size))
    matrix_eigenvectors[:,:] = np.nan

    for idx, v in enumerate(results_vertices_eigenvectors):
        matrix_eigenvectors[idx, :len(v)] = v

    # Save file
    if output_name is not None:
        io.save_gifti(nib_surf, results_vertices, output_name + "vb-cluster.value.shape.gii")
        io.save_gifti(nib_surf, matrix_eigenvectors, output_name + "vb-cluster.vector.shape.gii")

    # Cleanup
    pool.close()
    pool.terminate()
    pool.join()

    return results_vertices, results_vertices_eigenvectors
