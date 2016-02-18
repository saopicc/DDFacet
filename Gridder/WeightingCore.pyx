import numpy as np
cimport numpy as np
cimport cython


# fast version
@cython.boundscheck(False)
def accumulate_weights_onto_grid_1d (np.ndarray[double,ndim=1] grid, np.ndarray[double,ndim=1] weights, np.ndarray[long,ndim=1] index):
    """Given 1D arrays of weights and indices (of the same size), and a 1D array called 'grid',
    accumulates the sum of weights corresponding to each grid element.
    """
    cdef unsigned int i
    for i in xrange(len(weights)):
      grid[<unsigned int>index[i]] += weights[i]




# this version is over x100 times slower. Teaches me to trust cython for-loops over nditers
@cython.boundscheck(False)
def accumulate_weights_onto_grid_using_nditer (grid, weights, index):
    cdef int i
    cdef double w

    it = np.nditer([weights,index], 
                      flags=[ 'buffered' ],
                      op_flags=[['readonly'], ['readonly']],
                      op_dtypes=['float64', 'int'])
    for w, i in it:
      grid[i] += w

    
    
