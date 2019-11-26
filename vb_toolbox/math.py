#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Lucas Costa Campos <l.campos@fz-juelich.de>
#
# Distributed under terms of the GPL license.

import numpy as np
import scipy.linalg as spl
import warnings

def force_symmetric(M):
    """Force the symmetry of a given matrix.

    The symmetric version is computed by first replicating the diagonal and
    upper triangular components of the matrix, and mirroring the upper diagonal
    into the lower diagonal them into the lower triangular.
    """
    # One diag extracs the diagonal into an array, two
    # diags turns the array into a diagonal matrix.
    diag_M = np.diag(np.diag(M))
    triu_M = np.triu(M, 1)

    return triu_M + diag_M + triu_M.transpose()

def solve_general_eigenproblem(Q, D=None):

    """Solve the general eigenproblem.

    Solves the general eigenproblem Qx = lDx. The eigenvectos returned are
    unitary in the norm induced by the matrix D, and are sorted in increasing
    eigenvalue.
    """

    if D is None:
        eigenvalues, eigenvectors = spl.eig(Q, check_finite=False)
    else:
        eigenvalues, eigenvectors = spl.eig(Q, D, check_finite=False)

    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    if D is not None:
        # By default, scipy returns eigenvectos normalised in the Frobenius
        # norm, while Matlab returns them normalised by the norm induced by D.
        # We will now convert to the Matlab version for easier comparison.
        for i in range(len(eigenvectors)):
            e = eigenvectors[:, i]
            n = np.matmul(e.transpose(), np.matmul(D, e))
            eigenvectors[:, i] = e/np.sqrt(n)


    # Sort eigen pairs in increasing order of eigenvalues
    sort_eigen = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_eigen]
    eigenvectors = eigenvectors[:, sort_eigen]

    return eigenvalues, eigenvectors, sort_eigen

def spectral_reorder(B, method = 'geig'):
    """Computes the spectral reorder of the matrix"""

    # Fix the input
    assert B.shape[0] == B.shape[1], "Matrix B must be square!"

    min_b = np.min(B)
    assert min_b >= -1.0, "This function only accepts matrices with a mininum negative value of -1"

    # __import__('pdb').set_trace()
    if min_b < 0:
        warnings.warn("""
        The value 1 is being added to your similarity matrix to ensure positivity.
        This may cause issues with interpretation. Consider inputing a positive matrix""",
        warning.UserWarning)
        C = B + 1
    else:
        C = B

    # Actual decomposition

    # create the laplacian matrix (Q).
    # For all non diagonal elements, populate matrix Q with the negative
    # value of matrix C
    # For all the diagonal element, sum across the rows (excluding the
    # diagonal element) and populate the diagonal of Q with that sum

    triuC = np.triu(C,1) # Extract upper triangular elements;
    C = triuC + triuC.transpose() # Reconstruct a symmetric weighted adjacency matrix eliminating possible small errors in off-diagonal elements
    D =  np.diag(np.sum(C, axis=-1)); # Compute the Degree matrix

    Q = D - C; #Compute un-normalised Laplacian

    method = method.lower()

    if method == 'geig':
        # Method using generalised spectral decomposition of the
        # un-normalised Laplacian (see Shi and Malik, 2000)

        eigenvalues, eigenvectors, sort = solve_general_eigenproblem(Q, D)
        # eigenvectors = -eigenvectors

    elif method == 'sym':

        # Method using the eigen decomposition of the Symmetric Normalized
        # Laplacian. Note that results should be the same as 'geig'
        T = np.sqrt(D)
        L = spl.solve(T, Q)/np.diag(T) #Compute the normalized laplacian
        L = force_symmetric(L) # Force symmetry

        eigenvalues, eigenvectors, sort = solve_general_eigenproblem(L)
        eigenvectors = spl.solve(T, eigenvectors) # renormalize

    elif method == 'rw':
      # Method using eigen decomposition of Random Walk Normalised Laplacian
      # This method has not been rigorously tested yet

        # T = D
        # L = spl.solve(T, Q)
        L = spl.solve(D, Q)

        eigenvalues, eigenvectors, sort = solve_general_eigenproblem(L)
        # eigenvectors = -eigenvectors

    elif method == 'unnorm':

        eigenvalues, eigenvectors, sort = solve_general_eigenproblem(Q)

    else:
        raise NameError("""Method '{}' not allowed. \n
        Please choose one of the following: 'sym', 'rw', 'geig', 'unnorm'.""".format(method))

    v2 = eigenvectors[:, 1] # Get Fiedler vector
    sort_idx = np.argsort(v2) # Find the reordering index
    sorted_B = B[sort_idx,:] # Reorder the original matrix
    sorted_B = sorted_B[:,sort_idx] # Reorder the original matrix
    v3D = eigenvectors[:, 1:4] # Extract first three non-null vectors
    return sorted_B, sort_idx, v2, v3D, eigenvalues, eigenvectors

def create_affinity_matrix(neighborhood, eps=np.finfo(float).eps):
    """Computes the affinity matrix of a given neighborhood"""

    neighborhood_len = neighborhood.shape[0]
    neighborhood = neighborhood.reshape(neighborhood_len, -1)
    # Here, the affinity matrix should have n_neighbors x data_size shape
    # Create a mean centered neighborhood
    neighborhood_mean = np.mean(neighborhood, axis=-1)
    neighborhood_mc = neighborhood - neighborhood_mean.reshape(-1, 1)
    neighborhood_mc[np.abs(neighborhood_mc)<eps] = eps

    # __import__('pdb').set_trace()
    # Normalise the mean centered neighborhood
    neighborhood_w = np.sqrt(np.sum(neighborhood_mc**2, axis=-1)).reshape(-1, 1)
    neighborhood_scaled = neighborhood_mc/neighborhood_w

    # affinity = np.dot(neighborhood.transpose(), neighborhood)
    affinity = np.dot(neighborhood_scaled, neighborhood_scaled.transpose())
    affinity[affinity > 1.0] = 1.0

    # "Linearlise" the affinity ensure positive correlations are between 0 to 1
    # what i am doing here is to change cosines to angles but to ensure that
    # this remains a similarity rather than dissimilarity I am treating the
    # values as the sine rather than cosine. I.e. identical signals will be 90
    # rather than 0. 90/90 == 1.
    A = np.arcsin(affinity)/np.pi*180.0
    A = A/90.0

    # Remove negative correlations
    A[A<0] = eps

    # __import__('pdb').set_trace()

    return A
