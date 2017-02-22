'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

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

@cython.boundscheck(False)
def accumulate_weights_onto_grid_1d_withlocks (np.ndarray[double,ndim=1] grid, np.ndarray[double,ndim=1] weights, np.ndarray[long,ndim=1] index):
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

    
    
