#!/usr/bin/env python
# -*- coding: utf-8 -*-
# sampling.py --- Stochastic sampling algorithms

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

"""
References
----------
"""

# code:

from __future__ import division  # Then 10/3 provide 3.3333 instead of 3
import warnings
import time

import numpy as np
from numpy.random import standard_normal as randn

from . import optim

__author__ = "François Orieux"
__copyright__ = "2011, 2017 F. Orieux <orieux@l2s.centralesupelec.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "0.1.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@l2s.centralesupelec.fr"
__status__ = "development"
__url__ = ""
__keywords__ = "sampling algorithm, MCMC"


class GibbsSampler():
    def __init__(self, max_iter=150, min_iter=100, burnin=50, threshold=1e-5,
                 callback=None, settings=None):
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.threshold = threshold
        self.callback = callback

        if settings is not None:
            for key, val in settings:
                setattr(self, key, val)

        for key, val in kwargs.items():
            setattr(self, key, [val])

        self.norms = []
        self.loop_time = []

        if self.min_iter > self.max_iter:
            warnings.warn("Maximum iteration ({0}) is lower than "
                          "minimum iteration ({1}). "
                          "Maximum is set to mininum".format(self.max_iter,
                                                             self.min_iter))
            self.max_iter = self.min_iter

    def __enter__(self):
        self.iteration = 0
        if hasattr(self, names):
            for name in self.names:
                setattr(self, name, [])
        return self

    def __exit__(self):
        self.loop_time = np.array(self.loop_time) - self.loop_time[0]
        self.total_time = self.loop_time[-1]

    def stop_iter(self, sample, *args):
        self.loop_time.append(time.time())

        if not hasattr(self, names):
            self.names = ['param_' + str(idx) for idx in len(args)]
            for name in self.names:
                setattr(self, name, [])

        for idx, val in enumerate(args):
            getattr(self, self.names[idx]).append(val)

        if self.callback:
            self.callback(sample, args)

        if self.iteration >= self.max_iter:
            return True

        if ((self.iteration > self.min_iter) and
            (self.residual_norms[-1] <
             self.threshold * self.residual_norms[0])):
            return True

        self.iteration += 1
        return False


def po_draw_mdim_gauss(f_draw, f_transpose, f_hessian_proj, init,
                       cg_settings=None, precond=None):
    """Draw high-dimension gaussian law with PO algorithm

    This algorithm, described in [1], allow to draw (or simulate) very
    high-dimension law thanks to carefull use of optimisation
    algorithm.

    Parameters
    ----------
    f_draw : dict of callable
        Each items is a callable to draw a prior samples

    f_transpose : dict of callable
        Each items is a callable to compute the transpose of the prior
        samples.

    f_hessian_proj : callable
        Compute the hessian (or covariance) projection of a current
        point. Used by the optimization algorithm.

    init : array-like
        The initial point of the optimization algorithm.

    cg_settings : dict
        Conjugate gradient parameters. See ``optim.conj_grad``.

    Returns
    -------
    sample : array-like
        The gaussian sample

    References
    ----------
    .. [1] F. Orieux, O. Féron and J.-F. Giovannelli, "Sampling
       high-dimensional Gaussian distributions for general linear inverse
       problems", IEEE Signal Processing Letters, 2012

    See Acknowledgements section of the package ``README.rst`` to know
    how to add a citation. """
    # Perturbation
    second_member = np.sum((f_transpose[term](f_draw[term])
                            for term in f_transpose.keys()), axis=0)

    # Optimization
    sample, _, _, _ = optim.conj_grad(f_hessian_proj, init, second_member,
                                      cg_settings)

    return sample


def cdsampler(f_hessian_proj, init, second_member, settings):
    """Conjugate direction sampler

    This function implement an algorithm to simulate a gaussian law by
    conjugate direction factorisation.

    """
    sample = init
    hess_proj = f_hessian_proj(sample)
    second_member = randn(hess_proj.shape)
    residual = second_member - hess_proj
    direction = residual.copy()

    for iteration in range(min(init.size, settings["max_iter"])):
        hess_proj = f_hessian_proj(direction)
        variance = np.sum(hess_proj * direction)
        eee = np.sum(hess_proj * sample) / variance
        fff = np.sum(direction * hess_proj) / variance
        coef = randn() / np.sqrt(variance)
        sample += (coef - eee) * direction
        second_member += (coef - fff) * hess_proj
        residual -= (fff - eee) * hess_proj
        direction = residual - np.sum(residual *
                                      hess_proj) * direction / variance

    return (sample, second_member)
