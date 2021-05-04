#!/usr/bin/env python
# -*- coding: utf-8 -*-
# udft.py --- Unitary fourier transform

# Copyright (c) 2011, 2015  François Orieux <orieux@l2s.centralesupelec.fr>

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

# Commentary:

"""
Unitary discrete Fourier transform and utilities

This module implement unitary discrete Fourier transform, that is
ortho-normal. They are specially usefull for convolution [1]: they
respect the parseval equality, the value of the null frequency is
equal to

.. math:: frac{1}{sqrt{n}} sum_i x_i.

If the pyfftw module is present, his function are used. pyfftw wrap fftw
C library. Otherwise, numpy.fft functions are used.

You must keep in mind that the transform are applied on last
axes. this is a fftw convention for performance reason (C order
array). If you want more sophisticated use, you must use directly the
numpy.fft or pyfftw modules.

References
----------
.. [1] B. R. Hunt "A matrix theory proof of the discrete convolution
       theorem", IEEE Trans. on Audio and Electroacoustics,
       vol. au-19, no. 4, pp. 285-288, dec. 1971
"""

# code:
from __future__ import (division, absolute_import, print_function)

import logging
import warnings
# import multiprocessing as mp
# from functools import partial

import numpy as np

try:
    from pyfftw.interfaces.numpy_fft import fftn
    from pyfftw.interfaces.numpy_fft import ifftn
    from pyfftw.interfaces.numpy_fft import rfftn
    from pyfftw.interfaces.numpy_fft import irfftn
except ImportError:
    logging.info("Installation of the pyfftw package improve preformance"
                 " by using fftw library.")
    from numpy.fft import fftn as fftn
    from numpy.fft import ifftn as ifftn
    from numpy.fft import rfftn as rfftn
    from numpy.fft import irfftn as irfftn


__author__ = "François Orieux"
__copyright__ = "2011, 2015, F. Orieux <orieux@l2s.centralesupelec.com>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "0.3.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@l2s.centralesupelec.fr"
__status__ = "development"
__url__ = ""
__keywords__ = "fft"


def udftn(inarray, ndim=None, *args, **kwargs):
    """N-dim unitary discrete Fourier transform

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    ndim : int, optional
        The `ndim` last axis along wich to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like (same shape than inarray)
    """
    if not ndim:
        ndim = inarray.ndim

    return fftn(inarray, axes=range(-ndim, 0), *args, **kwargs) / np.sqrt(
        np.prod(inarray.shape[-ndim:]))


def uidftn(inarray, ndim=None, *args, **kwargs):
    """N-dim unitary inverse discrete Fourier transform

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    ndim : int, optional
        The `ndim` last axis along wich to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like (same shape than inarray)
    """
    if not ndim:
        ndim = inarray.ndim

    return ifftn(inarray, axes=range(-ndim, 0), *args, **kwargs) * np.sqrt(
        np.prod(inarray.shape[-ndim:]))


def urdftn(inarray, ndim=None, *args, **kwargs):
    """N-dim real unitary discrete Fourier transform

    This transform consider the Hermitian property of the transform on
    real input

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    ndim : int, optional
        The `ndim` last axis along wich to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like (the last ndim as  N / 2 + 1 lenght)
    """
    if not ndim:
        ndim = inarray.ndim

    return rfftn(inarray, axes=range(-ndim, 0), *args, **kwargs) / np.sqrt(
        np.prod(inarray.shape[-ndim:]))


def uirdftn(inarray, ndim=None, *args, **kwargs):
    """N-dim real unitary discrete Fourier transform

    This transform consider the Hermitian property of the transform
    from complex to real real input.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    ndim : int, optional
        The `ndim` last axis along wich to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like (the last ndim as (N - 1) * 2 lenght)
    """
    if not ndim:
        ndim = inarray.ndim

    return irfftn(inarray, axes=range(-ndim, 0), *args, **kwargs) * np.sqrt(
        np.prod(inarray.shape[-ndim:-1]) * (inarray.shape[-1] - 1) * 2)


def udft2(inarray, *args, **kwargs):
    """2-dim unitary discrete Fourier transform

    Compute the discrete Fourier transform on the last 2 axes.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    Returns
    -------
    outarray : array-like (same shape than inarray)

    See Also
    --------
    uidft2, udftn, urdftn
    """
    return udftn(inarray, 2, *args, **kwargs)


def uidft2(inarray, *args, **kwargs):
    """2-dim inverse unitary discrete Fourier transform

    Compute the inverse discrete Fourier transform on the last 2 axes.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    Returns
    -------
    outarray : array-like (same shape than inarray)

    See Also
    --------
    uidft2, uidftn, uirdftn
    """
    return uidftn(inarray, 2, *args, **kwargs)


def urdft2(inarray, *args, **kwargs):
    """2-dim real unitary discrete Fourier transform

    Compute the real discrete Fourier transform on the last 2 axes. This
    transform consider the Hermitian property of the transform from
    complex to real real input.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    Returns
    -------
    outarray : array-like (the last dim as (N - 1) *2 lenght)

    See Also
    --------
    udft2, udftn, urdftn
    """
    return urdftn(inarray, 2, *args, **kwargs)


def uirdft2(inarray, *args, **kwargs):
    """2-dim real unitary discrete Fourier transform

    Compute the real inverse discrete Fourier transform on the last 2 axes.
    This transform consider the Hermitian property of the transform
    from complex to real real input.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    Returns
    -------
    outarray : array-like (the last ndim as (N - 1) *2 lenght)

    See Also
    --------
    urdft2, uidftn, uirdftn
    """
    return uirdftn(inarray, 2, *args, **kwargs)


def quad_norm(inarray, hermitian_sym=True):
    """Return quadratic norm of images in discrete Fourier space

    Parameters
    ----------
    inarray : array-like
        The images are supposed to be in the last two axes

    Returns
    -------
    norm : float

    """
    if hermitian_sym:
        return 2 * np.sum(np.abs(inarray)**2) - \
            np.sum(np.abs(inarray[..., 0])**2)
    else:
        return np.sum(np.abs(inarray)**2)


def crandn(shape):
    """white complex gaussian noise

    Generate directly the unitary discrete Fourier transform of white gaussian
    noise noise field (with given shape) of zero mean and variance
    unity (ie N(0,1)).
    """
    return np.sqrt(0.5) * (np.random.standard_normal(shape) +
                           1j * np.random.standard_normal(shape))


def ir2fr(imp_resp, shape, center=None, real=True):
    """Return the frequency response from impulsionnal responses

    This function make the necessary correct zero-padding, zero
    convention, correct DFT etc. to compute the frequency response
    from impulsionnal responses (IR).

    The IR array is supposed to have the origin in the middle of the
    array.

    The Fourier transform is performed on the last `len(shape)`
    dimensions.

    Parameters
    ----------
    imp_resp : ndarray
       The impulsionnal responses.

    shape : tuple of int
       A tuple of integer corresponding to the target shape of the
       frequency responses, without hermitian property.

    center : tuple of int, optional
       The origin index of the impulsionnal response. The middle by
       default.

    real : boolean (optionnal, default True)
       If True, imp_resp is supposed real, the hermissian property is
       used with rfftn DFT and the output has `shape[-1] / 2 + 1`
       elements on the last axis.

    Returns
    -------
    y : ndarray
       The frequency responses of shape `shape` on the last
       `len(shape)` dimensions.

    Notes
    -----
    - For convolution, the result have to be used with unitary
      discrete Fourier transform for the signal (udftn or equivalent).
    - DFT are always peformed on last axis for efficiency.
    - Results is always C-contiguous.

    See Also
    --------
    udftn, uidftn, urdftn, uirdftn
    """
    if len(shape) > imp_resp.ndim:
        raise ValueError("length of shape must inferior to imp_resp.ndim")

    if not center:
        center = [int(np.floor(length / 2)) for length in imp_resp.shape]

    if len(center) != len(shape):
        raise ValueError("center and shape must have the same length")

    # Place the provided IR at the beginning of the array
    irpadded = np.zeros(shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp

    # Roll, or circshift to place the origin at 0 index, the
    # hypothesis of the DFT
    for axe, shift in enumerate(center):
        irpadded = np.roll(irpadded, -shift,
                           imp_resp.ndim - len(shape) + axe)

    # Perform the DFT on the last axes
    if real:
        return np.ascontiguousarray(rfftn(
            irpadded, axes=list(range(imp_resp.ndim - len(shape),
                                      imp_resp.ndim))))
    else:
        return np.ascontiguousarray(fftn(
            irpadded, axes=list(range(imp_resp.ndim - len(shape),
                                      imp_resp.ndim))))


def ir2tf(imp_resp, shape, real=True):
    warnings.warn("Deprecated, use ir2fr instead", DeprecationWarning)
    return ir2fr(imp_resp=imp_resp, shape=shape, real=real)


def fr2ir(freq_resp, shape, center=None, real=True):
    """Return the impulsionnal responses from frequency responses

    This function make the necessary correct zero-padding, zero
    convention, correct DFT etc. to compute the impulsionnal responses
    from frequency responses.

    The IR array is supposed to have the origin in the middle of the
    array.

    The Fourier transform is performed on the last `len(shape)`
    dimensions.

    Parameters
    ----------
    freq_resp : ndarray
       The frequency responses.

    shape : tuple of int
       A tuple of integer corresponding to the target shape of the
       impulsionnal responses.

    center : tuple of int, optional
       The origin index of the impulsionnal response. The middle by
       default.

    real : boolean (optionnal, default True)
       If True, imp_resp is supposed real, the hermissian property is
       used with irfftn DFT and the input is supposed to `shape[-1] /
       2 + 1` elements on the last axis.

    Returns
    -------
    y : ndarray
       The impulsionnal responses of shape `shape` on the last
       `len(shape)` axes.

    See Also
    --------
    udftn, uidftn, urdftn, uirdftn
    """
    if len(shape) > freq_resp.ndim:
        raise ValueError("length of shape must inferior to freq_resp.ndim")

    if not center:
        center = [int(np.floor(length / 2)) for length in shape]

    if len(center) != len(shape):
        raise ValueError("center and shape must have the same length")

    if real:
        irpadded = irfftn(freq_resp,
                          axes=list(range(freq_resp.ndim - len(shape),
                                          freq_resp.ndim)))
    else:
        irpadded = ifftn(freq_resp,
                         axes=list(range(freq_resp.ndim - len(shape),
                                         freq_resp.ndim)))

    for axe, shift in enumerate(center):
        irpadded = np.roll(irpadded, shift,
                           freq_resp.ndim - len(shape) + axe)

    return np.ascontiguousarray(irpadded[tuple([slice(0, s) for s in shape])])


class DiffOp:
    def __init__(self, ndim, axe):
        assert ndim > 0, ('The number of dimension `ndim` '
                          'must be strictly positive')
        assert axe < ndim, ('The `axe` argument must be inferior to `ndim`')

        self.ndim = ndim
        self.axe = axe

    @property
    def impr(self):
        return np.reshape(np.array([0, -1, 1], ndmin=self.ndim),
                          [1] * self.axe + [3] + [1] *
                          (self.ndim - self.axe - 1))

    def freqr(self, shape):
        return ir2tf(self.impr, shape)


def diff_op(ndim, axe, shape=None, real=False):
    """Return the transfert function of the difference

    Parameters
    ----------
    ndim : int
        The number of dimension

    axe : int
        The axe where where the diff operate

    shape : tuple, shape
        The support on which to compute the transfert function

    real : bool
        If True return the impulsionnal response. If False return the
        frequency response. False by default.

    Returns
    -------
    rep : array_like
        The frequency or impulsionnal response
    """
    assert (not real) & (shape is not None), ('`shape` must be set '
                                              'if `real` is not True')
    assert ndim > 0, 'The number of dimension `ndim` must be strictly positive'
    assert axe < ndim, 'The `axe` argument must be inferior to `ndim`'

    impr = np.reshape(np.array([0, -1, 1], ndmin=ndim),
                      [1] * axe + [3] + [1] * (ndim - axe - 1))
    if real:
        return impr
    else:
        return ir2tf(impr, shape)


def laplacian(ndim, shape, real=False):
    """Return the transfert function of the laplacian

    Laplacian is the second order difference, on line and column.

    Parameters
    ----------
    ndim : int
        The dimension of the laplacian

    shape : tuple, shape
        The support on which to compute the transfert function

    real : bool
        If True return the impulsionnal response. If False return the
        frequency response. False by default.

    Returns
    -------
    rep : array_like
        The frequency or impulsionnal response
    """
    impr = np.zeros([3] * ndim)
    for dim in range(ndim):
        idx = tuple([slice(1, 2)] * dim +
                    [slice(None)] +
                    [slice(1, 2)] * (ndim - dim - 1))
        impr[idx] = np.array([-1.0,
                              0.0,
                              -1.0]).reshape([-1 if i == dim else 1
                                              for i in range(ndim)])
    impr[([slice(1, 2)] * ndim)] = 2.0 * ndim

    if real:
        return impr
    else:
        return ir2tf(impr, shape)
