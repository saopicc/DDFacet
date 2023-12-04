#!/usr/bin/env python
# -*- coding: utf-8 -*-
# improcessing.py --- Image processing

# Copyright (C) 2011, 2016 F. Orieux <orieux@l2s.centralesupelec.fr>

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

"""This module implements function for image processing. The most
important function is actually udeconv that implement an unsupervised
deconvolution described in [1]. Add a citation if you use this work
and see Acknowledgements section in the package ``README.rst``.

.. [1] François Orieux, Jean-François Giovannelli, and Thomas
       Rodet, "Bayesian estimation of regularization and point
       spread function parameters for Wiener-Hunt deconvolution",
       J. Opt. Soc. Am. A 27, 1593-1607 (2010)

http://www.opticsinfobase.org/josaa/abstract.cfm?URI=josaa-27-7-1593
"""

# code:

from __future__ import (division, print_function)

import numpy as np
import numpy.random as npr
import scipy.special

from . import udft
from .udft import urdft2 as dft
from .udft import uirdft2 as idft
from .udft import ir2fr

from . import optim

__author__ = "François Orieux"
__copyright__ = "Copyright (C) 2011, 2016 F. Orieux "
"<orieux@l2s.centralesupelec.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "0.1.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@l2s.centralesupelec.fr"
__status__ = "development"
__url__ = ""
__keywords__ = "image processing, deconvolution"


def deconv(data, imp_resp, reg_val, reg=None, real=True):
    """Wiener-Hunt deconvolution

    return the deconvolution with a wiener-hunt approach (ie with
    Fourier diagonalisation).

    Parameters
    ----------
    data : ndarray
       The data

    imp_resp : ndarray
       The impulsionnal response in real space or the transfer
       function. Differentiation is done with the dtype where
       transfer function is supposed complex.

    reg_val : float
       The regularisation parameter value.

    reg : ndarray, optional
       The regularisation operator. The laplacian by
       default. Otherwise, the same constraints that for `imp_resp`
       apply

    real : boolean, optional
       True by default. Specify if `imp_resp` or `reg` are provided
       with hermitian hypothesis or not. See otb.udft module.

    Returns
    -------
    im_deconv : ndarray
       The deconvolued data

    References
    ----------
    .. [1] François Orieux, Jean-François Giovannelli, and Thomas
           Rodet, "Bayesian estimation of regularization and point
           spread function parameters for Wiener-Hunt deconvolution",
           J. Opt. Soc. Am. A 27, 1593-1607 (2010)

           http://www.opticsinfobase.org/josaa/abstract.cfm?URI=josaa-27-7-1593

       [2] B. R. Hunt "A matrix theory proof of the discrete
           convolution theorem", IEEE Trans. on Audio and
           Electroacoustics, vol. au-19, no. 4, pp. 285-288, dec. 1971
    """
    if not reg:
        reg = udft.laplacian(data.ndim, data.shape)
    if reg.dtype != complex:
        reg = udft.ir2tf(reg, data.shape)

    if imp_resp.shape != reg.shape:
        trans_func = udft.ir2tf(imp_resp, data.shape)
    else:
        trans_func = imp_resp

    wiener_filter = np.conj(trans_func) / (np.abs(trans_func)**2 +
                                           reg_val * np.abs(reg)**2)
    if real:
        return udft.uirdftn(wiener_filter * udft.urdftn(data))
    else:
        return udft.uidftn(wiener_filter * udft.udftn(data))


def udeconv(data, imp_resp, reg=None, user_params={}):
    """Unsupervised Wiener-Hunt deconvolution

    return the deconvolution with a wiener-hunt approach, where the
    hyperparameters are estimated (or automatically tuned from a
    practical point of view). The algorithm is a stochastic iterative
    process (Gibbs sampler).

    This work can be free software. If you use this work add a
    citation to the reference below.

    Parameters
    ----------
    data : ndarray
       The data

    imp_resp : ndarray
       The impulsionnal response in real space or the transfer
       function. Differentiation is done with the dtype where
       transfer function is supposed complex.

    reg : ndarray, optional
       The regularisation operator. The laplacian by
       default. Otherwise, the same constraints that for `imp_resp`
       apply

    user_params : dict
       dictionary of gibbs parameters. See below.

    Returns
    -------
    x_postmean : ndarray
       The deconvolued data (the posterior mean)

    chains : dict
       The keys 'noise' and prior contains the chain list of noise and
       prior precision respectively

    Other parameters
    ----------------
    The key of user_params are

    threshold : float
       The stopping criterion: the norm of the difference between to
       successive approximated solution (empirical mean of object
       sample). 1e-4 by default.

    burnin : int
       The number of sample to ignore to start computation of the
       mean. 100 by default.

    min_iter : int
       The minimum number of iteration. 30 by default.

    max_iter : int
       The maximum number of iteration if `threshold` is not
       satisfied. 150 by default.

    callback : None
       A user provided function to which is passed, if the function
       exists, the current image sample. This function can be used to
       store the sample, or compute other moments than the mean.

    References
    ----------
    .. [1] François Orieux, Jean-François Giovannelli, and Thomas
           Rodet, "Bayesian estimation of regularization and point
           spread function parameters for Wiener-Hunt deconvolution",
           J. Opt. Soc. Am. A 27, 1593-1607 (2010)

    http://www.opticsinfobase.org/josaa/abstract.cfm?URI=josaa-27-7-1593

    See Acknowledgements section in the package ``README.rst`` to know
    how to add a citation.
    """
    params = {'threshold': 1e-4, 'max_iter': 200,
              'min_iter': 30, 'burnin': 15, 'callback': None}
    params.update(user_params)

    if not reg:
        reg = udft.laplacian(data.ndim, data.shape)
    if reg.dtype != complex:
        reg = udft.ir2tf(reg, data.shape)

    if imp_resp.shape != reg.shape:
        trans_func = udft.ir2tf(imp_resp, data.shape)
    else:
        trans_func = imp_resp

    # The mean of the object
    x_postmean = np.zeros(trans_func.shape)
    # The previous computed mean in the iterative loop
    prev_x_postmean = np.zeros(trans_func.shape)

    # Difference between two successive mean
    delta = np.NAN

    # Initial state of the chain
    gn_chain, gx_chain = [1], [1]

    # Parameter of the hyperparameter law. The following value
    # correspond to Jeffery's prior for the hyper parameter. See
    # reference.
    alpha_n, beta_n_bar = (0, 0)
    alpha_x, beta_x_bar = (0, 0)

    # The correlation of the object in Fourier space (if size is big,
    # this can reduce computation time in the loop)
    areg2 = np.abs(reg)**2
    atf2 = np.abs(trans_func)**2

    data_size = data.size
    data = udft.urdftn(data)

    # Gibbs sampling
    for iteration in range(params['max_iter']):
        # Sample of Eq. 27 p(circX^k | gn^k-1, gx^k-1, y).

        # weighing (correlation in direct space)
        precision = gn_chain[-1] * atf2 + gx_chain[-1] * areg2  # Eq. 29
        excursion = udft.crandn(data.shape) / np.sqrt(precision)

        # mean Eq. 30 (RLS for fixed gn, gamma0 and gamma1 ...)
        wiener_filter = gn_chain[-1] * np.conj(trans_func) / precision
        x_mean = wiener_filter * data

        # sample of X in Fourier space
        x_sample = x_mean + excursion
        if params['callback']:
            params['callback'](x_sample)

        # sample of Eq. 31 p(gn | x^k, gx^k, y)
        likelihood = udft.quad_norm(data - x_sample * trans_func)
        gn_chain.append(npr.gamma(alpha_n + data_size / 2,
                                  1 / (beta_n_bar + likelihood / 2)))

        # sample of Eq. 31 p(gx | x^k, gn^k-1, y)
        smoothness = udft.quad_norm(x_sample * reg)
        gx_chain.append(npr.gamma(alpha_x + (data_size - 1) / 2,
                                  1 / (beta_x_bar + smoothness / 2)))

        # current empirical average
        if iteration > params['burnin']:
            x_postmean = prev_x_postmean + x_sample

        if iteration > (params['burnin'] + 1):
            norm = np.sum(np.abs(x_postmean)) / (iteration - params['burnin'])
            current = x_postmean / (iteration - params['burnin'])
            previous = prev_x_postmean / (iteration - params['burnin'] - 1)

            delta = np.sum(np.abs(current - previous)) / norm

        prev_x_postmean = x_postmean

        # stop of the algorithm
        if (iteration > params['min_iter']) and (delta < params['threshold']):
            break

    # Empirical average ≈ POSTMEAN Eq. 44
    x_postmean = x_postmean / (iteration - params['burnin'])
    x_postmean = udft.uirdftn(x_postmean)

    return (x_postmean, {'noise': np.array(gn_chain),
                         'prior': np.array(gx_chain)})


def huber_gy_deconv(data, imp_resp, reg, threshold, n_iter=50):
    """Return the huber circular deconvolution with the Geman & Yang
    formulation and Huber potential.
    """
    trans_fct = udft.ir2fr(imp_resp, data.shape)
    row_filter = udft.ir2fr(np.array([[1, -1]]), data.shape)
    col_filter = udft.ir2fr(np.array([[1, -1]]).T, data.shape)
    transpose_dataf = np.conj(trans_fct) * dft(data)

    hess_inv = 1.0 / (abs(trans_fct)**2 +
                      reg * (abs(row_filter)**2 + abs(col_filter)**2))

    huber_fourier = hess_inv * transpose_dataf

    def aux_min(var_aux):
        return var_aux - np.where(np.abs(var_aux) < threshold,
                                  2 * var_aux,
                                  2 * threshold * np.sign(var_aux))

    for _ in range(n_iter):
        aux_row_sample = aux_min(idft(row_filter * huber_fourier))
        aux_col_sample = aux_min(idft(col_filter * huber_fourier))

        huber_fourier = hess_inv * (
            transpose_dataf +
            reg * np.conj(row_filter) * dft(aux_row_sample) +
            reg * np.conj(col_filter) * dft(aux_col_sample))

    return idft(huber_fourier), aux_row_sample, aux_col_sample


def xi(obj, gauss_prec, laplace_prec):
    rho = laplace_prec / (2 * gauss_prec)
    return np.exp(laplace_prec * obj / 2) * \
        scipy.special.erfc((rho + obj) * np.sqrt(gauss_prec / 2))


def logerf(obj, gauss_prec, laplace_prec):
    return - 2 * np.log(xi(obj, gauss_prec, laplace_prec) +
                        xi(-obj, gauss_prec, laplace_prec))


def deriv_logerf(obj, gauss_prec, laplace_prec):
    xim = xi(-obj, gauss_prec, laplace_prec)
    xip = xi(obj, gauss_prec, laplace_prec)
    return - laplace_prec * (xip - xim) / (xip + xim)


def logerf_gy_deconv(data, imp_resp, noise_prec, gauss_prec, laplace_prec,
                     n_iter=50):
    """Return the l2l1 circular deconvolution with the Geman & Yang
    formulation and LogErf potential.
    """
    trans_fct = udft.ir2fr(imp_resp, data.shape)
    row_filter = udft.ir2fr(np.array([[1, -1]]), data.shape)
    col_filter = udft.ir2fr(np.array([[1, -1]]).T, data.shape)
    transpose_dataf = np.conj(trans_fct) * dft(data)

    hess_inv = 1.0 / (noise_prec * np.abs(trans_fct)**2 +
                      gauss_prec * np.abs(row_filter)**2 +
                      gauss_prec * np.abs(col_filter)**2)

    def aux_min(var_aux):
        return var_aux - deriv_logerf(var_aux, gauss_prec, laplace_prec)

    l2l1_fourier = hess_inv * noise_prec * transpose_dataf
    for _ in range(n_iter):
        aux_row = dft(aux_min(idft(row_filter * l2l1_fourier)))
        aux_col = dft(aux_min(idft(col_filter * l2l1_fourier)))

        l2l1_fourier = hess_inv * (noise_prec * transpose_dataf +
                                   gauss_prec * np.conj(row_filter) * aux_row +
                                   gauss_prec * np.conj(col_filter) * aux_col)

    return idft(l2l1_fourier), idft(aux_row), idft(aux_col)


def logerf_gy_reconstr(data, imp_resp, noise_prec, gauss_prec, laplace_prec,
                       n_iter=50):
    """Return the l2l1 circular deconvolution with the Geman & Yang
    formulation and LogErf potential.
    """
    trans_fct = udft.ir2fr(imp_resp, data.shape)
    row_filter = udft.ir2fr(np.array([[1, -1]]), data.shape)
    col_filter = udft.ir2fr(np.array([[1, -1]]).T, data.shape)
    transpose_dataf = np.conj(trans_fct) * dft(data)

    hess_inv = 1.0 / (noise_prec * np.abs(trans_fct)**2 +
                      gauss_prec * np.abs(row_filter)**2 +
                      gauss_prec * np.abs(col_filter)**2)
    rho = laplace_prec / (2 * gauss_prec)

    def xi(obj):
        return np.exp(laplace_prec * obj / 2) * \
            scipy.special.erfc((rho + obj) * np.sqrt(gauss_prec / 2))

    def deriv(obj):
        xim = xi(-obj)
        xip = xi(obj)
        return -laplace_prec * (xip - xim) / (xip + xim)

    def aux_min(var_aux):
        return var_aux - deriv(var_aux)

    l2l1_fourier = hess_inv * noise_prec * transpose_dataf
    for _ in range(n_iter):
        aux_row = dft(aux_min(idft(row_filter * l2l1_fourier)))
        aux_col = dft(aux_min(idft(col_filter * l2l1_fourier)))

        l2l1_fourier = hess_inv * (noise_prec * transpose_dataf +
                                   gauss_prec * np.conj(row_filter) * aux_row +
                                   gauss_prec * np.conj(col_filter) * aux_col)

    return idft(l2l1_fourier), idft(aux_row), idft(aux_col)


def huber_gr_deconv(data, imp_resp, reg, threshold, n_iter=50):
    """Return the huber circular deconvolution with the Geman & Yang
    formulation and Huber potential.
    """
    trans_fct = ir2fr(imp_resp, data.shape)
    row_filter = udft.ir2tf(np.array([[1, -1]]), data.shape)
    col_filter = udft.ir2tf(np.array([[1, -1]]).T, data.shape)
    transpose_dataf = np.conj(trans_fct) * dft(data)

    var_aux = transpose_dataf

    def aux_min(var_aux):
        aux = np.ones_like(var_aux)
        idx = (var_aux != 0)
        aux[idx] = np.where(
            np.abs(var_aux[idx]) <= threshold,
            2 * var_aux[idx],
            2 * threshold * np.sign(var_aux[idx])) / (2 * var_aux[idx])
        return aux

    def precond(var_aux):
        return var_aux / (abs(trans_fct)**2 +
                          reg * (abs(row_filter)**2 + abs(col_filter)**2))

    def hessian(var_aux):
        out = abs(trans_fct)**2 * var_aux
        out += reg * np.conj(row_filter) * dft(
            aux_row_sample * idft(row_filter * var_aux))
        out += reg * np.conj(col_filter) * dft(
            aux_col_sample * idft(col_filter * var_aux))
        return out

    for _ in range(n_iter):
        aux_row_sample = aux_min(idft(row_filter * var_aux))
        aux_col_sample = aux_min(idft(col_filter * var_aux))

        var_aux, _, _ = optim.conj_grad(hessian, var_aux, transpose_dataf,
                                        precond=precond)

    return idft(var_aux), aux_row_sample, aux_col_sample
