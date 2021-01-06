#!/usr/bin/env python
# -*- coding: utf-8 -*-
# optim.py --- Optimisation algorithm

# Copyright (c) 2011-2016  François Orieux <orieux@l2s.cenralesupelec.fr>

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

"""Numerical optimisation algorithms

These modules implements personnal designed classical optimisation
algorithm like linear conjugate gradient, with or without
preconditionner.

I have implemented these algorithms:
- to be adapted to large inverse problems with high-dimension unknown
  (> 1e6)
- to learn
- to not depends on another package.

References
----------
.. [1] J. R. Shewchuk, "An introduction to the Conjugate Gradient
   Method without the Agonizing Pain", report, 1994
"""

# code:

from __future__ import division  # Then 10/3 provide 3.3333 instead of 3
import time
import warnings
import functools

import numpy as np
import numpy.random as npr
from numpy.random import standard_normal as randn

import matplotlib.pyplot as plt

from . import algotools
from .criterions import LinearMeanSquare as LMS

__author__ = "François Orieux"
__copyright__ = "Copyright (C) 2011-2014 F. Orieux <orieux@iap.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "0.1.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@iap.fr"
__status__ = "development"
__url__ = ""
__keywords__ = "numerical optimisation, conjugage gradient"


def value_res_quad(crit, obj, res):
    """If res=b - Qx is available, crit value is cheaper to compute as
    J(x) = (x^t (-b - res) + c) / 2"""
    return (np.sum(obj * (-crit.second_term - res)) + crit.constant) / 2


class ConjGrad(algotools.IterativeAlg):
    def __init__(self, max_iter=100, min_iter=20, restart=50,
                 threshold=1e-6, speedrun=True, feedbacks=None,
                 name='QuadCG'):
        """Linear Conjugate Gradient

        Attributes
        ----------
        minimizer : array-like
          The solution of the optimization (alias x).

        success : boolean
          Whether or not the optimizer exited successfully.

        status : int
          Termination status of the optimizer. Its value depends on
          the underlying solver. Refer to message for details.

        message : str
          Description of the cause of the termination.

        fun, jac, hess: array-like
          Values of objective function, its Jacobian and its Hessian
          (if available). The Hessians may be approximations, see the
          documentation of the function in question.

        hess_inv : object
          Inverse of the objective function’s Hessian; may be an
          approximation. Not available for all solvers. The type of
          this attribute may be either np.ndarray or
          scipy.sparse.linalg.LinearOperator.

        nfev, njev, nhev : int
          Number of evaluations of the objective functions and of its
          Jacobian and Hessian.

        nit : int
          Number of iterations performed by the optimizer.

        maxcv : float
          The maximum constraint violation.

        """
        super().__init__(max_iter=max_iter, min_iter=min_iter,
                         threshold=threshold, stochastic=False,
                         speedrun=speedrun, feedbacks=feedbacks,
                         name=name)
        self.restart = restart
        self.threshold = threshold

    def stop_res(self, res_norm):
        """Not used"""
        if res_norm[-1] < self.threshold * res_norm[0]:
            return True
        else:
            return False

    def run(self, crit, init=None, precond=None):
        if precond is None:
            def precond(obj):
                return obj

        if init is None:
            minimizer = self._trace(init=crit.second_term, name='Minimizer')
        else:
            minimizer = self._trace(init=init, name='Minimizer')
        residual = self._trace(
            init=crit.second_term - crit.hessian(minimizer.last),
            name='Residual')

        crit_val = algotools.Trace(init=value_res_quad(
            crit, minimizer.last, residual.last), name='Crit')

        step = algotools.Trace(init=0, name='Step')
        res_norm = algotools.Trace(init=np.sum(np.abs(residual.last**2)),
                                   name='Res norm')

        direction = precond(residual.last)

        self.fig_register_trace(minimizer, residual, crit_val, step)
        self.watch_for_stop(minimizer)

        for nit in self.looper:
            hess_proj = crit.hessian(direction)

            step.last = res_norm.last / np.sum(np.real(
                np.conj(direction) * hess_proj))

            minimizer.last = minimizer.last + step.last * direction

            if nit % self.restart == 0:
                residual.last = crit.second_term - crit.hessian(minimizer.last)
            else:
                residual.last = residual.last - step.last * hess_proj

            secant = precond(residual.last)
            res_norm.last = np.sum(np.real(np.conj(residual.last) * secant))
            direction = secant + res_norm[-1] / res_norm[-2] * direction

            crit_val.last = value_res_quad(crit, minimizer.last, residual.last)

        return minimizer.last, algotools.IterRes(self.looper,
                                                 crit_val=crit_val)


class RJPO(algotools.IterativeAlg):
    def __init__(self, target_prob=0.9, max_iter=100, min_iter=20, restart=50,
                 mh_step=True, speedrun=True, feedbacks=None):
        super().__init__(max_iter=max_iter, min_iter=min_iter,
                         speedrun=speedrun, stochastic=False,
                         feedbacks=feedbacks, name='RJPO')
        self.restart = restart
        self.target_prob = target_prob
        self.mh_step = mh_step

    def stop(self, proba):
        if (proba.last <= self.target_prob):
            return False
        else:
            return True

    def run(self, crit, previous, precond=None):
        if precond is None:
            def precond(obj):
                return obj

        minimizer = self._trace(init=-previous, name='Proposition')
        residual = self._trace(
            init=crit.second_term - crit.hessian(minimizer.last),
            name='Res')

        crit_val = algotools.Trace(init=value_res_quad(crit,
                                                       minimizer.last,
                                                       residual.last),
                                   name='Crit')
        step = algotools.Trace(init=0, name='Step')
        res_norm = algotools.Trace(init=np.sum(np.abs(residual.last**2)),
                                   name='Res norm')
        proba = algotools.Trace(init=0, name='Proba')

        direction = precond(residual.last)

        self.fig_register_trace(minimizer, residual, crit_val, proba)
        self.watch_for_stop(proba)

        for nit in self.looper:
            hess_proj = crit.hessian(direction)

            step <<= res_norm.last / np.sum(np.real(
                np.conj(direction) * hess_proj))

            minimizer <<= minimizer.last + step.last * direction

            if nit % self.restart == 0:
                residual <<= crit.second_term - crit.hessian(minimizer.last)
            else:
                residual <<= residual.last - step.last * hess_proj

            secant = precond(residual.last)
            res_norm <<= np.sum(np.real(np.conj(residual.last) * secant))
            direction = secant + res_norm[-1] / res_norm[-2] * direction

            crit_val <<= value_res_quad(crit, minimizer.last, residual.last)

            proba <<= min(1, np.exp(-np.sum(
                residual.last * (previous - minimizer.last))))

        info = algotools.IterRes(self.looper, crit_val=crit_val, proba=proba)

        if self.mh_step:
            if npr.uniform() <= proba.last:
                return minimizer.last, info
            else:
                return previous, info
        else:
            return minimizer.last, info


class TvADMM(algotools.IterativeAlg):
    def __init__(self, optimizer, max_iter=100, min_iter=20,
                 threshold=1e-6, speedrun=True, feedbacks=None):
        super().__init__(max_iter, min_iter, threshold, False,
                         speedrun, feedbacks, name='TV ADMM')
        self.optimizer = optimizer

    def run(self, instr, reg, data, init, rho, hyper):
        obj = self._trace(init=init, name='Obj')
        z = self._trace(init=reg(init), name='z')
        u = self._trace(init=z.last, name='u')
        data_adeq = LMS(instr, data)

        # z.last = np.zeros_like(reg(init))
        # u.last = np.zeros_like(z.last)
        # data_t = self.instr.reverse(data)
        # obj.last = init

        cg_info = []

        self.fig_register_trace(obj, z)
        self.watch_for_stop(obj)

        for _ in self.looper:
            # scd_term = data_t + rho * self.reg.reverse(z.last - u.last)
            # def hessian(obj):
            #     return self.instr.fwrev(obj) + rho * self.reg.fwrev(obj)
            # obj.last, info, _ = conj_grad(
            #     hessian, obj.last, scd_term, settings=self.setup['CG'])

            obj.last, optres = self.optimizer.run(
                data_adeq + LMS(reg, z.last - u.last, precision=rho),
                obj.last)
            cg_info.append(optres)

            coeffs = reg(obj.last)
            z.last = np.maximum(0, coeffs + u.last - hyper / rho) - \
                np.maximum(0, -coeffs - u.last - hyper / rho)
            u.last = u.last + coeffs - z.last

        return obj, z, u, algotools.IterRes(self.looper, cg_info=cg_info)


class FISTA(algotools.IterativeAlg):
    def __init__(self, max_iter=100, min_iter=20, threshold=1e-6,
                 speedrun=True, feedbacks=None):
        super().__init__(max_iter=max_iter, min_iter=min_iter,
                         threshold=threshold, stochastic=False,
                         speedrun=speedrun, feedbacks=feedbacks,
                         name='FISTA')
        self._eps = 0

    def prox(self, obj, alpha):
        # return obj * np.max(0, 1 - alpha / (np.abs(obj) + self._eps))
        return np.where(np.abs(obj) < alpha, 0, obj - np.sign(obj) * alpha)

    def prox_nng(self, obj, alpha):
        return obj * np.max(0, 1 - (alpha / np.abs(obj) + self._eps)**2)

    def prox_hard(self, obj, alpha):
        return np.where(alpha < np.abs(obj), 0, obj)

    def run(self, instr, data, init, hyper):
        def crit(obj):
            return np.sum(np.abs(data - instr(obj))**2) + \
                hyper * np.sum(np.abs(obj))

        previous = init
        obj = self._trace(init=init, name='Obj')
        crit_val = algotools.Trace(init=crit(init), name='Crit')
        step = algotools.Trace(init=1, name='Step')
        beta = 0.5
        datat = instr.t(data)

        self.fig_register_trace(obj, crit_val, step)
        self.watch_for_stop(obj)

        for it in self.looper:
            previous = obj.last.copy()
            grad = instr.fwrev(obj.last) - datat
            while True:
                z = self.prox(obj.last - step.last * grad, hyper)
                if crit(z) <= (crit(obj.last) + np.sum(grad * (z - obj.last)) +
                               (1 / (2 * step.last)) * np.sum(np.abs(
                                   z - obj.last)**2)):
                    break
                step <<= beta * step.last
            obj <<= z + (it + 1) / (it + 5) * (obj.last - previous)
            crit_val <<= crit(obj.last)

        return obj, algotools.IterRes(
            self.looper, crit_val=crit_val, step=step)


def conj_grad(hessian, init, second_term, settings=None,
              precond=None):
    """Linear conjugate gradient

    This function compute the solution of a linear system `Ax = b`
    with a iterative conjugated gradient algorithm. This function is
    specially usefull for problems in great dimension and
    is compatible with variable in *complex* unknowns.

    Parameters
    ----------
    hessian : callable (function-like)
        The callable object to compute the product `Ax`. The signature
        must accept only on parameter `x`.

    init : array_like
        The starting point of the algorithm

    second_term : array_like
        The second term `b` of the system

    precond : callable, optional
        Application of a preconditionner

    Other parameters
    ----------------
    settings : dict, optional
        The dictionnary of parameters for the conjugate gradient
        algorithm. The lookup key are
        - 'cg min iter' and 'cg max iter' for the minumun and maximum
          iteration (10 and 50 by default)
        - 'f crit' for a callable to compute the criterion value at
          current iteration (None by default)
        - 'cg threshold' as the ration between initial and final residual
          norm, as a stoping criterion (1e-5 by default)
        - 'restart' is the frequency in iteration for restarting the
          CG to remove accumulation of numerical error roundoff.

    Returns
    -------
    minimizer : array_like
        The minimum `x` found to satisfy `Ax = b`
    info : dict
        Information about the algorithm. `res_norm` is a list of
        residual norm at each iteration. `loop_time` is a list of
        current time at the end of each iteration. `crit_val` is set
        if `f_crit` function is provided in settings. `end` indicate
        the criterion used to stop the algorithm

    sample : array_like
        A sample under the equivalent gaussian law [2].

    References
    ----------
    .. [1] J. R. Shewchuk, "An introduction to the Conjugate Gradient
       Method without the Agonizing Pain", report, 1994
    .. [2] C. Fox
    """
    setup = {'cg threshold': 1e-6, 'cg max iter': 100, 'cg min iter': 50,
             'f crit': None, 'cg restart': 50, 'cg draw': None}
    setup.update(settings if settings else {})

    if setup['cg min iter'] > setup['cg max iter']:
        warnings.warn("Maximum iteration ({0}) is lower than"
                      " minimum iteration ({1}). "
                      "Maximum is set to mininum".format(setup['cg max iter'],
                                                         setup['cg min iter']))
        setup['cg max iter'] = setup['cg min iter']

    # Gradient at current init
    residual = second_term - hessian(init)
    direction = precond(residual) if precond else residual

    minimizer = init
    sample = np.zeros_like(init)

    info = {'res_norm': [np.sum(np.real(np.conj(residual) * direction))],
            'loop_time': [time.time()],
            'step': [], 'beta': [], 'dirnorm': [], 'dir_q_norm': []}

    if setup['cg draw']:
        fig = plt.figure(setup['cg draw'])
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(minimizer)
        plt.colorbar()
        plt.title('Minimizer')
        plt.show()
        plt.draw()
        fig.canvas.flush_events()

    for iteration in range(init.size):
        if iteration >= setup['cg max iter']:
            info['end'] = 'by iteration'
            return (minimizer, info, sample)

        hess_proj = hessian(direction)
        # a = r^tr/d^tAd
        # Optimal step in direction of direction
        variance = np.sum(np.real(np.conj(direction) * hess_proj))
        step = info['res_norm'][-1] / variance

        # Descent x^(i+1) = x^(i) + ad
        minimizer = minimizer + step * direction
        sample = sample + (randn() / np.sqrt(variance)) * direction

        # r^(i+1) = r^(i) - a*Ad
        if (iteration % setup['cg restart']) == 0:
            residual = second_term - hessian(minimizer)
        else:
            residual = residual - step * hess_proj

        # Conjugate direction
        secant = precond(residual) if precond else residual
        info['res_norm'].append(np.sum(np.real(np.conj(residual) *
                                               secant)))
        beta = info['res_norm'][-1] / info['res_norm'][-2]
        direction = secant + beta * direction

        if setup['f crit']:
            info['crit_val'].append(setup['f crit'](minimizer))
        info['loop_time'].append(time.time())
        info['step'].append(step)
        info['dir_q_norm'].append(variance)
        info['dirnorm'].append(np.sum(direction**2))
        info['beta'].append(beta)

        if setup['cg draw']:
            # # fig = plt.figure(setup['cg draw'])
            # setup['cg draw'].clf()
            # # setup['cg draw'].clear()
            # ax = setup['cg draw'].add_subplot(1, 2, 1)
            # im = ax.imshow(minimizer)
            # # setup['cg draw'].colorbar(im)
            # ax2 = setup['cg draw'].add_subplot(1, 2, 2)
            # im = ax2.imshow(residual)
            # # setup['cg draw'].colorbar(im)
            # setup['cg draw'].suptitle('conj_grad iter {}'.format(iteration))
            # # plt.draw()
            # # setup['cg draw'].canvas.flush_events()
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(minimizer)
            plt.colorbar()
            plt.title('Minimizer')
            plt.subplot(1, 2, 2)
            plt.imshow(residual)
            plt.colorbar()
            plt.title('Residual')
            plt.suptitle('conj_grad iter {}'.format(iteration))

            plt.draw()
            fig.canvas.flush_events()

        # Stopping criterion
        if (iteration > setup['cg min iter']) and (
                info['res_norm'][-1] < setup['cg threshold'] *
                info['res_norm'][0]):
            info['end'] = 'by criterion'
            return (minimizer, info, sample)

    info['end'] = 'by all direction optimization'
    return (minimizer, info, sample)


def rjpo(hessian, previous, second_term, settings=None, precond=None):
    setup = {'cg threshold': 1e-6, 'cg max iter': 100, 'cg min iter':
             50, 'f crit': None, 'cg restart': 50, 'cg draw': None,
             'target prob': 0.9, 'MH step': False}
    setup.update(settings if settings else {})

    if setup['cg min iter'] > setup['cg max iter']:
        warnings.warn("Maximum iteration ({0}) is lower than"
                      " minimum iteration ({1}). "
                      "Maximum is set to mininum".format(setup['cg max iter'],
                                                         setup['cg min iter']))
        setup['cg max iter'] = setup['cg min iter']

    # Gradient at current init
    init = -previous
    residual = second_term - hessian(init)
    direction = precond(residual) if precond else residual

    minimizer = init
    sample = np.zeros_like(init)

    info = {'res_norm': [np.sum(np.real(np.conj(residual) * direction))],
            'loop_time': [time.time()],
            'step': [], 'beta': [], 'dirnorm': [], 'dir_q_norm': [],
            'prob': []}

    if setup['cg draw']:
        fig = plt.figure(setup['cg draw'])
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(minimizer)
        plt.colorbar()
        plt.title('Minimizer')
        plt.show()
        plt.draw()
        fig.canvas.flush_events()

    for iteration in range(init.size):
        if iteration >= setup['cg max iter']:
            info['end'] = 'by iteration'
            info['nit'] = iteration
            return (minimizer, info, sample)

        hess_proj = hessian(direction)
        # a = r^tr/d^tAd
        # Optimal step in direction of direction
        variance = np.sum(np.real(np.conj(direction) * hess_proj))
        step = info['res_norm'][-1] / variance

        # Descent x^(i+1) = x^(i) + ad
        minimizer = minimizer + step * direction
        sample = sample + (randn() / np.sqrt(variance)) * direction

        # r^(i+1) = r^(i) - a*Ad
        if (iteration % setup['cg restart']) == 0:
            residual = second_term - hessian(minimizer)
        else:
            residual = residual - step * hess_proj

        # Conjugate direction
        secant = precond(residual) if precond else residual
        info['res_norm'].append(np.sum(np.real(np.conj(residual) *
                                               secant)))
        beta = info['res_norm'][-1] / info['res_norm'][-2]
        direction = secant + beta * direction

        if setup['f crit']:
            info['crit_val'].append(setup['f crit'](minimizer))
        info['loop_time'].append(time.time())
        info['step'].append(step)
        info['dir_q_norm'].append(variance)
        info['dirnorm'].append(np.sum(direction**2))
        info['beta'].append(beta)

        if setup['cg draw']:
            # # fig = plt.figure(setup['cg draw'])
            # setup['cg draw'].clf()
            # # setup['cg draw'].clear()
            # ax = setup['cg draw'].add_subplot(1, 2, 1)
            # im = ax.imshow(minimizer)
            # # setup['cg draw'].colorbar(im)
            # ax2 = setup['cg draw'].add_subplot(1, 2, 2)
            # im = ax2.imshow(residual)
            # # setup['cg draw'].colorbar(im)
            # setup['cg draw'].suptitle('conj_grad iter {}'.format(iteration))
            # # plt.draw()
            # # setup['cg draw'].canvas.flush_events()
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(minimizer)
            plt.colorbar()
            plt.title('Minimizer')
            plt.subplot(1, 2, 2)
            plt.imshow(residual)
            plt.colorbar()
            plt.title('Residual')
            plt.suptitle('conj_grad iter {}'.format(iteration))

            plt.draw()
            fig.canvas.flush_events()

        # # Stopping criterion
        # if (iteration > setup['cg min iter']) and (
        #         info['res_norm'][-1] < setup['cg threshold'] *
        #         info['res_norm'][0]):
        #     info['end'] = 'by criterion'
        #     return (minimizer, info, sample)
        # Stopping criterion
        info['prob'].append(min(1, np.exp(-np.sum(
            residual * (previous - minimizer)))))
        # import pdb; pdb.set_trace()
        if (info['prob'][-1] >= setup['target prob']):
            # print('end by prob : {} / iter : {}'.format(info['prob'],
            # iteration))
            info['end'] = 'by prob'
            info['nit'] = iteration
            break

    if setup['MH step']:
        if npr.uniform() <= info['prob']:
            return minimizer, info, None
        else:
            return previous, info, None
    else:
        return minimizer, info, None

    # info['end'] = 'by all direction optimization'
    # return (minimizer, info, sample)


def pre_conj_grad(hessian, init, second_term, f_precond,
                  settings=None):
    setup = {'cg threshold': 1e-5, 'cg max iter': 50, 'cg min iter': 10,
             'f crit': None, 'restart': 50}
    setup.update(settings if settings else {})

    if setup['cg min iter'] > setup['cg max iter']:
        warnings.warn("Minimum iteration ({0}) is lower than"
                      " maximum iteration ({1}). "
                      "Maximum is set to mininum".format(setup['cg min iter'],
                                                         setup['cg max iter']))
        setup['cg max iter'] = setup['cg min iter']

    # Gradient at current init
    residual = second_term - hessian(init)
    direction = f_precond(residual)

    minimizer = init

    info = {'res_norm': [np.sum(np.real(np.conj(residual) *
                                        direction))],
            'loop_time': [time.time()]}

    for iteration in range(init.size):
        hess_proj = hessian(direction)
        # a = r^tr/d^tAd
        # Optimal step in direction of direction
        step = info['res_norm'][-1] / np.sum(np.real(np.conj(direction) *
                                                     hess_proj))

        # Descent x^(i+1) = x^(i) + ad
        minimizer = minimizer + step * direction

        # r^(i+1) = r^(i) - a*Ad (think residual as gradient in data space)
        if (iteration % setup['restart']) == 0:
            residual = second_term - hessian(minimizer)
        else:
            residual = residual - step * hess_proj

        # Conjugate direction with preconditionner
        secant = f_precond(residual)
        info['res_norm'].append(np.sum(np.real(np.conj(residual) *
                                               secant)))
        if info['res_norm'][-1] == 0:
            del info['res_norm'][-1]
            return minimizer, info, 'Minimum reached'

        beta = info['res_norm'][-1] / info['res_norm'][-2]
        direction = secant + beta * direction

        if setup['f crit']:
            info['crit_val'].append(setup['f crit'](minimizer))
        info['loop_time'].append(time.time())

        # Stopping criterion
        if (iteration > setup['cg min iter']) and (info['res_norm'][-1] <
                                                   setup['cg threshold'] *
                                                   info['res_norm'][0]):
            return (minimizer, info, 'End by criterion')

        if iteration >= setup['cg max iter']:
            return (minimizer, info, 'End by iteration')

    return (minimizer, info, 'All direction optimized')


def binsch(crit, interval, epsilon=1e-7, iteration=100):
    """Bisection of scalar cost function
    """
    lower_bound = np.min(interval)
    upper_bound = np.max(interval)
    half_bound = (lower_bound + upper_bound) / 2

    xaxis = [lower_bound, half_bound, upper_bound]

    val_lower_bound = crit(lower_bound)
    val_upper_bound = crit(upper_bound)
    val_half_bound = crit(half_bound)

    yaxis = [val_lower_bound, val_half_bound, val_upper_bound]

    while ((np.abs(upper_bound - lower_bound) >=
            epsilon) and (iteration >= 1)):
        quart_bound = (lower_bound + half_bound) / 2
        three_quart_bound = (half_bound + upper_bound) / 2

        xaxis.append(quart_bound)
        xaxis.append(three_quart_bound)

        val_quart_bound = crit(quart_bound)
        val_three_quart_bound = crit(three_quart_bound)

        yaxis.append(val_quart_bound)
        yaxis.append(val_three_quart_bound)

        argmin = np.argmin([val_lower_bound, val_quart_bound,
                            val_half_bound,
                            val_three_quart_bound, val_upper_bound])

        if (argmin == 1) or (argmin == 2):
            # If the min is lower_bound or quart_bound
            upper_bound = half_bound
            half_bound = quart_bound
            val_upper_bound = val_half_bound
            val_half_bound = val_quart_bound
        elif argmin == 3:
            # If the min is half_bound
            lower_bound = quart_bound
            upper_bound = three_quart_bound
            val_lower_bound = val_quart_bound
            val_upper_bound = val_three_quart_bound
        else:
            # If the min is three_quart_bound or upper_bound
            lower_bound = half_bound
            half_bound = three_quart_bound
            val_lower_bound = val_half_bound
            val_half_bound = val_three_quart_bound

        iteration -= 1

    xaxis = np.sort(xaxis)
    yaxis = np.array(yaxis)[np.argsort(xaxis)]

    return ([lower_bound, upper_bound], xaxis, yaxis)
