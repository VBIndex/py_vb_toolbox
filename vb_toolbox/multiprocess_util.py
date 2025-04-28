import numpy as np
from multiprocessing import Pool, Value

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

def determine_items_and_cpus(n_cpus, data):
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
    

    n_items = data.shape[0]*data.shape[1]*data.shape[2]
    
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

def run_multiprocessing(pool, internal_loop_func, n_items, dn, internal_loop_args):
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
        expanded_internal_loop_args = (i0, iN) + internal_loop_args
        threads = np.append(threads, (pool.apply_async(internal_loop_func, expanded_internal_loop_args, error_callback=pool_callback)))
    
    # Wait for all threads to complete
    pool.close()
    pool.join()

    res = np.empty((len(threads),), dtype=object)

    for r, t in enumerate(threads):
        res[r] = t.get() 

    return res

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