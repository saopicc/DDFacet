#!/usr/bin/env python
# -*- coding: utf-8 -*-
# utils.py --- Basic utilities

# Copyright (c) 2011, 2012, 2013  François Orieux <orieux@iap.fr>

# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Commentarr:

"""Implement various utilities functions.
"""

from __future__ import (division, absolute_import, with_statement,
                        print_function)

# code:

import subprocess
import functools
import warnings

import numpy as np
import numpy.linalg as la
import time

__author__ = "François Orieux"
__copyright__ = "Copyright (C) 2011, 2012, 2013 F. Orieux <orieux@iap.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "0.1.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@iap.fr"
__status__ = "development"
__url__ = ""
__keywords__ = "fft"


def system(cmd):
    """
    Invoke a shell command.
    :returns: A tuple of output, err message and return code
    """
    ret = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    out, err = ret.communicate()
    return out, err, ret.returncode

# http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
# class Bunch:
#     def __init__(self, **kwds):
#         self.__dict__.update(kwds)

class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__ = self


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            filename=func.func_code.co_filename,
            lineno=func.func_code.co_firstlineno + 1
        )
        return func(*args, **kwargs)
    return new_func


def vech(matrix):
    """The flattend upper triangular part of the matrix"""
    return matrix[np.triu_indices(matrix.shape[0])]


def uvech(vector):
    """A symetric matrix from the flattened upper triangular part"""
    # Compute the size by resolution of equation size = n(n+1)/2
    size = int((np.sqrt(1 + 8 * len(vector)) - 1) / 2)
    # Fill the upper triangular part
    matrix = np.zeros((size, size))
    matrix[np.triu_indices(size)] = vector.ravel()
    # Fill the lower triangular part
    return matrix.T + matrix - np.diag(matrix.diagonal())


def cov2cor(covariance):
    """The correlation from the covariance matrix"""
    std = np.sqrt(np.diag(covariance))[:, np.newaxis]
    return (covariance / std) / std.T


def fim2cor(fim):
    """The correlation from the Fisher Information Matrix"""
    return cov2cor(la.inv(fim))


def fim2crb(fim):
    """The Cramer Rao Bound from the Fisher Information Matrix"""
    return np.sqrt(np.diag(la.inv(fim)))


def gaussian_kernel(width, sigma=4.0):
    """Return a 2D gaussian kernel"""
    assert isinstance(width, int), 'width must be an integer'
    radius = (width - 1) / 2.0
    axis = np.linspace(-radius, radius, width)
    filterx = np.exp(-axis * axis / (2 * sigma**2))
    return filterx / filterx.sum()


@deprecated
def circshift(inarray, shifts):
    """Shift array circularly.

    Circularly shifts the values in the array `a` by `s`
    elements. Return a copy.

    Parameters
    ----------
    a : ndarray
       The array to shift.

    s : tuple of int
       A tuple of integer scalars where the N-th element specifies the
       shift amount for the N-th dimension of array `a`. If an element
       is positive, the values of `a` are shifted down (or to the
       right). If it is negative, the values of `a` are shifted up (or
       to the left).

    Returns
    -------
    y : ndarray
       The shifted array (elements are copied)

    Examples
    --------
    >>> circshift(np.arange(10), 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    """
    # Initialize array of indices
    idx = []

    # Loop through each dimension of the input matrix to calculate
    # shifted indices
    for dim in range(inarray.ndim):
        length = inarray.shape[dim]
        try:
            shift = shifts[dim]
        except IndexError:
            shift = 0  # no shift if not specify

        # Lets start for fancy indexing. First we build the shifted
        # index for dim k. It will be broadcasted to other dim so
        # ndmin is specified
        index = np.mod(np.array(range(length),
                                ndmin=inarray.ndim) - shift,
                       length)
        # Shape adaptation
        shape = np.ones(inarray.ndim)
        shape[dim] = inarray.shape[dim]
        index = np.reshape(index, shape)

        idx.append(index.astype(int))

    # Perform the actual conversion by indexing into the input matrix
    return inarray[idx]


class Timer(object):
    """A timer context manager

    Used as

    >>> with Timer() as t: print("Do things")
    >>> t.msecs
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start = 0
        self.end = 0

    @property
    def secs(self):
        """The elapsed time in seconds"""
        return self.end - self.start

    @property
    def msecs(self):
        """The elapsed time in milli-seconds"""
        return self.secs * 1000

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        if self.verbose:
            print('elapsed time: {} ms'.format(self.msecs))


def link_generator(source, *filters):
    """Link generator like pipeline

    If source and filters are generator, return the elements of the
    output of the chained generators like unix pipe
    """
    gen = source()
    for filt in filters:
        gen = filt(gen)
    return gen


def atleast_nd(ndim, *arrs):
    """View inputs as arrays with at least ``ndim`` dimensions.

    Parameters
    ----------
    ndim: int
        The number of dimension for the returned arrays.

    arr1, arr2, ... : array_like
        One or more array-like sequences.  Non-array inputs are
        converted to arrays.  Arrays that already have three or more
        dimensions are preserved.

    Returns
    -------
    res1, res2, ... : ndarray
        An array, or tuple of arrays, each with ``a.ndim >= ndim``.
        Copies are avoided where possible, and views with ``ndim`` or
        more dimensions are returned.  Dimensions are *prepend* to
        already present axes.
    """
    res = [np.reshape(np.asanyarray(arr),
                      [1] * (ndim - arr.ndim) + list(arr.shape))
           for arr in arrs]
    if len(res) == 1:
        return res[0]
    else:
        return res


def def_settings(default, settings):
    if settings is None:
        return default
    else:
        return default.update(settings)


def cr_or_up_dset(hdf5_file, path, array):
    """Create or update hdf5 dset

    Parameters
    ----------
    hdf5_file: file
      The file object pointing to hdf5

    path: string
      The path of the dset

    array: array-like
      The array to store or update inside dset

    Returns
    -------
    The dset pointer
    """
    if path in hdf5_file:
        hdf5_file[path][...] = np.asarray(array)
        return hdf5_file[path]
    else:
        dset = hdf5_file.create_dataset(path, data=np.asarray(array))
        return dset
