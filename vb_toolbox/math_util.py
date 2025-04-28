import numpy as np
import scipy.linalg as spl
import warnings
from itertools import product

class TimeSeriesTooShortError(Exception):
    """Raised when the time series in the input data have less than three elements"""
    pass

#----get_neighborhood_vol----#

# OFFSETS
OFF27 = np.array(list(product((-1,0,1), repeat=3)), dtype=int)
OFF7  = OFF27[np.sum(np.abs(OFF27), axis=1) <= 1]  # centro + 6 caras

def index_to_coords(i, shape):
    p0, q0, r0 = shape
    z   = i % r0
    tmp = i // r0
    y   = tmp % q0
    x   = tmp // q0
    return np.array([x, y, z], dtype=int)

def get_offsets(six_f):
    return OFF7 if six_f else OFF27

def get_neighborhood_vol(data, i, six_f, mask=None):
    """
    data: ndarray (X,Y,Z,N)  
    i: flat index  
    six_f: bool (True → 7-neighborhood; False → 27-neighborhood)  
    mask: ndarray bool/0-1 or None  
    """
    shape_xyz  = data.shape[:3]
    vox_coords = index_to_coords(i, shape_xyz)

   
    if mask is not None and not bool(mask[tuple(vox_coords)]):
        return np.nan, vox_coords

    offsets = get_offsets(six_f)
    neigh   = vox_coords + offsets
    # Filtramos los que queden fuera del volumen
    in_bounds = np.all((neigh >= 0) & (neigh < shape_xyz), axis=1)
    neigh     = neigh[in_bounds]

    if mask is not None:
        valid = mask[neigh[:,0], neigh[:,1], neigh[:,2]].astype(bool)
        neigh = neigh[valid]

    if neigh.size == 0:
        vals = np.empty((0, data.shape[-1]))
    else:
        vals = data[neigh[:,0], neigh[:,1], neigh[:,2], :]
        if mask is not None:
            non_zero = ~np.all(vals == 0, axis=1)
            vals     = vals[non_zero]
    vals = np.atleast_2d(vals)

    return vals, vox_coords

#----get_neighborhood_vol----#

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

def spectral_reorder(B, method='unnorm'):
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

        vbi_value, eigenvector = get_fiedler_eigenpair(method, Q, D)

    elif method == 'unnorm':

        vbi_value, eigenvector = get_fiedler_eigenpair(method, Q)

    else:
        raise NameError("""Method '{}' not allowed. \n
        Please choose one of the following: 'sym', 'rw', 'geig', 'unnorm'.""".format(method))

    v2 = eigenvector # Fiedler vector
    sort_idx = np.argsort(v2) # Find the reordering index
    sorted_B = B[sort_idx,:] # Reorder the original matrix
    sorted_B = sorted_B[:,sort_idx] # Reorder the original matrix

    return sorted_B, sort_idx, vbi_value, eigenvector

def get_fiedler_eigenpair(method, Q, D=None, is_symmetric=True):

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