import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
def accumulate_weights_onto_grid (grid, weights, index):
    cdef int i
    cdef double w

    it = np.nditer([weights,index], 
                      flags=[ 'buffered' ],
                      op_flags=[['readonly'], ['readonly']],
                      op_dtypes=['float64', 'int'])
    for w, i in it:
      grid[i] += w

    
