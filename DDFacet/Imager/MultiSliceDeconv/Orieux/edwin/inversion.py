#!/usr/bin/env python
# -*- coding: utf-8 -*-
# inversion.py --- Main functions for inverse problems

# version: 0.1.0
# keywords: inverse problems, MCMC, map
# author: François Orieux <orieux@iap.fr>
# maintainer: François Orieux <orieux@iap.fr>
# url:

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

# Commentary:

"""This module implements function for resolution of inverse problems
within a Bayesian framework.

The main results here is the use of the Perturbation Optimisation (PO)
algorithm publised in [1]. More explicitly these algorithm allow prior
parameter estimation for general linear inverse problems. They have
been developed and used in the following references [1-3]. See
Acknowledgements section of the package ``README.rst`` to know how to
add a citation.

The problem solved is the general inverse problem

.. math:: y = H x + n

where `y` are data modeled by `x` is the unknown, `H` the linear (but
*not* invariant) acquisition process, and `n` an unknown noise. The
algorithms implemented here allow the estimation of `x` both with
hyper parameter value like the noise power or the regularisation power
of the `x`.

.. [1] F. Orieux, O. Féron and J.-F. Giovannelli, "Sampling
   high-dimensional Gaussian distributions for general linear inverse
   problems", IEEE Signal Processing Letters, 2012

   [2] F. Orieux, E. Sepulveda, V. Loriette, B. Dubertret and
   J.-C. Olivo-Marin, "Bayesian Estimation for Optimized Structured
   Illumination Microscopy", IEEE trans. on Image Processing. 2012

   [3] F. Orieux, J.-F. Giovannelli, T. Rodet, and A. Abergel,
   "Estimating hyperparameters and instrument parameters in
   regularized inversion Illustration for Herschel/SPIRE map
   making", Astronomy & Astrohpysics, 2013

"""

# code:

from __future__ import division  # Then 10/3 provide 3.3333 instead of 3
import functools
import warnings
import time

import numpy as np
import numpy.random as npr
from numpy.random import standard_normal as randn

import udft
from udft import uirdft2 as ifft
from udft import urdft2 as fft
import optim
import sampling

__author__ = "François Orieux"
__copyright__ = "Copyright (C) 2011, 2012, 2013 F. Orieux <orieux@iap.fr>"
__credits__ = ["François Orieux", "Olivier Féron",
               "Jean-François Giovannelli", "Thomas Rodet"]
__license__ = "mit"
__version__ = "0.1.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@iap.fr"
__status__ = "development"
__url__ = ""
__keywords__ = "inverse problems, image processing, image reconstruction"


# http://code.activestate.com/recipes/305268/
def chainmap(*maps):
    chained_map = {}
    for mapping in reversed(maps):
        if mapping:
            chained_map.update(mapping)
    return chained_map


def unsupervised_mse(f_draw, f_transpose, f_adequation, f_hessian_proj,
                     init, size, user_gibbs_params={}, cg_params={},
                     user_alpha=None, user_beta_bar=None):
    """Unsupervised Mean Square Error algorithm

    This function implements a gibbs sampler for general linear
    inverse problems where Gaussian precision prior law are
    unknown. This work is based on the Perturbation Optimisation
    described in [1] and it's understanding is necessary. See Notes
    below. Gaussian hypothesis are fundamental in this algorithm.

    All dictionary of callable must have the same key that identify
    each part of the joint posterior law.

    Parameters
    ----------
    f_draw : dict of callable
        Each items is a callable to draw a prior samples

    f_transpose : dict of callable
        Each items is a callable to compute the transpose of the prior
        samples.

    f_adequation : dict of callable
        Each items is a callable to compute the adequation term
        present in the scale parameter of the gamma law

    f_hessian_proj : callable
        Compute the hessian (or covariance) projection of a current
        space point. Used by the conjugate gradient.

    init : array-like
        The starting point of the gibbs sampler

    size : dict of int
        The size of each object (data size for noise, object size for
        prior, ...), with the same key than other dict, to compute the
        shape parameter.

    user_gibbs_params : dict, optional
        Parameters of the gibbs sampler

    cg_params : dict, optional
        Parameters of the conjugate gradient, see ``optim.conj_grad``

    user_gamma_law : dict, optional
        Prior gamma law parameters

    Returns
    -------
    obj_postmean : array-like
        The posterior mean of the object

    var_postesp : array-like
        The posterior mean variance of the object

    chains : dict
        The chains of hyper parameter

    Other parameters
    ----------------
    `user_gibbs_params` accepts this following keys :
    - `threshold` : minimum difference between two successive mean to
      stop the algorithm (1e-4 by default)
    - `max_iter` : the maximum number of iteration (200 by default)
    - `min_iter` : the minimum number of iteration (30 by default)
    - `burnin` : the iteration from which the algorithm start to compute a mean

    `gamma_prior` must be a dictionary of same key than the
    other, where each value is also a dictionary of two key `alpha`
    and `beta` that are the shape and scale parameters (they are 0 by
    default, ie Jeffrey's law).

    References
    ----------
    .. [1] F. Orieux, O. Féron and J.-F. Giovannelli, "Sampling
       high-dimensional Gaussian distributions for general linear inverse
       problems", IEEE Signal Processing Letters, 2012

    See Acknowledgements section of the package ``README.rst`` to know
    how to add a citation.

    Examples
    --------
    See main.py in the archive directory.
    """
    gibbs_params = {'threshold': 1e-4, 'max_iter': 200, 'min_iter': 30,
                    'burnin': 15}
    gibbs_params.update(user_gibbs_params)

    if gibbs_params['min_iter'] > gibbs_params['max_iter']:
        warnings.warn("Maximum iteration ({0}) is lower than"
                      " minimum iteration ({1}). "
                      "Maximum is set "
                      "to mininum".format(gibbs_params['max_iter'],
                                          gibbs_params['min_iter']))
        gibbs_params['max_iter'] = gibbs_params['min_iter']

        warnings.warn("Maximum iteration ({0}) is lower than"
                      " burnin ({1}): "
                      "burnin set to {2}".format(gibbs_params['max_iter'],
                                                 gibbs_params['burnin'],
                                                 gibbs_params['max_iter'] - 1))
        gibbs_params['burnin'] = gibbs_params['max_iter'] - 1

    # Jeffrey's law
    alpha = chainmap({key: 0 for key in f_adequation.keys()}, user_alpha)
    beta_bar = chainmap({key: 0 for key in f_adequation.keys()}, user_beta_bar)

    delta = np.NAN

    obj_sample = init[0].copy()
    chains = init[1]
    current = obj_sample.copy()

    previous_obj_postmean = np.zeros(obj_sample.shape)
    var_postesp = np.zeros(obj_sample.shape)

    info = {'loop_time': [time.time()]}
    if 'callback' in gibbs_params:
        info['callback_obj'] = [gibbs_params['callback'](obj_sample)]
        info['callback_mean'] = [gibbs_params['callback'](current)]

    # Gibbs sampling
    for iteration in range(gibbs_params['max_iter']):
        # Sampling of object
        precisions = dict([(term, values[-1])
                           for term, values in chains.items()])
        obj_sample = sampling.po_draw_mdim_gauss(f_draw,
                                                 f_transpose,
                                                 f_hessian_proj,
                                                 precisions,
                                                 obj_sample,
                                                 cg_params)

        # Sampling of precisions
        if not gibbs_params['Fix_hyp']:
            for term in precisions.keys():
                adequation = f_adequation[term](obj_sample)
                chains[term].append(npr.gamma(alpha[term] + size[term] / 2,
                                              1 / (beta_bar[term] +
                                                   adequation / 2)))

        # Empirical mean
        if iteration >= gibbs_params['burnin']:
            obj_postmean = previous_obj_postmean + obj_sample
            var_postesp = var_postesp + obj_sample**2

            if iteration > gibbs_params['burnin'] + 1:
                norm = np.sum(np.abs(obj_postmean) / (iteration -
                                                      gibbs_params['burnin']))
                current = obj_postmean / (iteration - gibbs_params['burnin'])
                previous_obj = previous_obj_postmean / (iteration -
                                                        gibbs_params['burnin']
                                                        - 1)

                delta = np.sum(np.abs(current - previous_obj)) / norm

            previous_obj_postmean = obj_postmean

        info['loop_time'].append(time.time())
        if 'callback' in gibbs_params:
            info['callback_obj'].append(gibbs_params['callback'](obj_sample))
            info['callback_mean'].append(gibbs_params['callback'](current))

        # Algorithm ending
        if (iteration >
            gibbs_params['min_iter']) and (delta <
                                           gibbs_params['threshold']):
            info['stop_iter'] = iteration
            info['stop_time'] = time.time()
            break

    # Empirical mean = posterior mean
    obj_postmean /= iteration - (gibbs_params['burnin'] - 1)
    var_postesp = var_postesp / (iteration -
                                 gibbs_params['burnin']) - obj_postmean**2
    chains = {key: np.array(val) for key, val in chains.items()}
    info = {key: np.array(val) for key, val in info.items()}
    info['loop_time'] -= info['loop_time'][0]

    return (obj_postmean, var_postesp, chains, info)


def usual_umse(hessian_proj, forward, transpose, init, size, data, reg=None, pres_init=None,
               user_gibbs_params={}, user_cg_params={}):
    """Usual unsupervised Mean Square Error

    The function `unsupervised_mse` is generic and can be used to
    simulate joint posterior law with high dimensional non stationary
    conditional Gaussian law for the object and conditional gamma law
    for the precision.

    This function is a wrapper for convenience, that set the usual
    model with stationary white Gaussian noise and eventually a
    regulator (prior filter). Again refer to [1] for more
    explaination.

    Parameters
    ----------
    hessian_proj : callable
        A callable to compute the hessian projection of the current
        point. Used by the conjugate gradient.

    forward : callable
        A callable to compute the forward model given the current point

    transpose : callable
        A callable to compute the transpose of data

    init : array-like
        The starting point of the gibbs sampler

    size : dict
        The data size with key 'data' and object size with key 'prior'

    data : array-like
        The data in the forward output shape

    reg : array-like, optional
        The regulator filter. Can be complex (transfer function) or
        real (conv kernel). If complex, must consider the hermitian
        property, see ``otb.udft``.

    user_gibbs_params : dict, optional
        See ``unsupervised_umse``

    user_cg_params : dict, optional
        See ``optim.conj_grad``

    Returns
    -------
    obj_postmean : array-like
        The posterior mean of the object

    var_postesp : array-like
        The posterior mean variance of the object

    chains : dict
        The chains of hyper parameters

    References
    ----------
    .. [1] F. Orieux, O. Féron and J.-F. Giovannelli, "Sampling
       high-dimensional Gaussian distributions for general linear inverse
       problems", IEEE Signal Processing Letters, 2012
    """
    if reg is None:
        reg, _ = udft.laplacian(data.ndim, data.shape)
    if reg.dtype != complex:
        reg = udft.ir2tf(reg, data.shape)

    proper_reg = reg.copy()
    proper_reg[np.where(reg == 0)] = 1e-10

    f_draw = {'data':
              lambda precision: data + randn(data.shape) / np.sqrt(precision),
              'prior':
              lambda precision: udft.crandn(reg.shape) / (np.sqrt(precision) *
                                                         proper_reg)}
    f_transpose = {'data': transpose,
                   'prior': lambda mean: np.conj(reg) * reg * mean}
    f_adequation = {'data': lambda obj: np.sum((data - forward(obj))**2),
                    'prior': lambda obj: udft.image_quad_norm(reg * obj)}

    if pres_init is not None:
        init = (init, pres_init)
    else:
        init = (init, {term: [1] for term in f_adequation.keys()})

    cg_params = {'threshold': 1e-7, 'max_iter': 50, 'min_iter': 20}
    cg_params.update(user_cg_params)
    gibbs_params = {'threshold': 1e-4, 'max_iter': 200, 'min_iter': 30,
                    'burnin': 30}
    gibbs_params.update(user_gibbs_params)

    return unsupervised_mse(f_draw, f_transpose, f_adequation, hessian_proj,
                            init, size, user_gibbs_params=gibbs_params,
                            cg_params=cg_params)


def unsupervised_map(f_adequation, f_hessian_proj, transposed,
                     size, obj, user_map_params={}, cg_params={},
                     user_gamma_prior={}):
    """The joint MAP estimate for general linear inverse problems
    """
    map_params = chainmap({'threshold': 1e-4, 'max_iter': 200, 'min_iter': 30},
                          user_map_params)
    # Jeffrey's law
    alpha = chainmap({key: 0 for key in f_adequation.keys()},
                     user_gamma_alpha)
    beta_bar = chainmap({key: 0 for key in f_adequation.keys()},
                        user_gamma_beta_bar)
    prec = {term: [1] for term in f_adequation.keys()}

    def stop(obj):
        if not hasattr(stop, 'previous_obj'):
            stop.previous_obj = obj
        stop.called = stop.called + 1 if hasattr(stop, 'called') else 1
        delta = np.sum(np.abs(ifft(obj - stop.previous_obj))) / \
                np.sum(np.abs(fft(stop.previous_obj)))
        stop.previous_obj = obj
        return ((iteration > map_params['min_iter'] and
                 delta < map_params['threshold']) or
                iteration > map_params['max_iter'])

    while not stop(obj):
        second_member = np.sum((2 * val[-1] * transposed[key]
                                for key, val in prec.items()),
                               axis=0)
        hess = functools.partial(f_hessian_proj,
                                 {key: val[-1] for key, val in prec})
        obj, _, _ = optim.conj_grad(hess, obj, second_member, cg_params)

        for term in prec.keys():
            prec[term].append(
                npr.gamma(alpha[term] + size[term] / 2,
                          1 / (beta_bar[term] + f_adequation[term](obj) / 2)))


    return (obj, prec)
