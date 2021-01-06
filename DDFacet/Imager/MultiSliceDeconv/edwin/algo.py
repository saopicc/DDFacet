#!/usb/bin/env python3
# -*- coding: utf-8 -*-
# noptim.py --- New optimisation algorithm

from __future__ import (division, absolute_import, print_function)
import time
import warnings

import numpy as np
from numpy.random import standard_normal as randn

__author__ = "François Orieux"
__copyright__ = "2011 - 2015 F. Orieux <orieux@l2s.centralesupelec.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "0.2.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@l2s.centralesupelec.fr"
__status__ = "development"
__url__ = ""
__keywords__ = "numerical optimisation, conjugage gradient"



class Optimization:
    def __init__(self, max_iter=50, min_iter=10, threshold=1e-5, callback=None,
                 restart=50, settings=None):
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.restart = restart
        self.threshold = threshold
        self.callback = callback

        if settings is not None:
            for key, val in settings:
                setattr(self, key, val)

        self.residual_norms = []
        self.loop_time = []

        if self.min_iter > self.max_iter:
            warnings.warn("Maximum iteration ({0}) is lower than "
                          "minimum iteration ({1}). "
                          "Maximum is set to mininum".format(self.max_iter,
                                                             self.min_iter))
            self.max_iter = self.min_iter

    def __enter__(self):
        self.iteration = 0
        return self

    def __exit__(self):
        self.loop_time = np.array(self.loop_time) - self.loop_time[0]
        self.total_time = self.loop_time[-1]
        self.residual_norms = np.array(self.residual_norms)

    def stop_iter(self, residual_norm, minimizer):
        self.residual_norms.append(residual_norm)
        self.loop_time.append(time.time())

        if self.callback:
            self.callback(minimizer)

        if self.iteration >= self.max_iter:
            return True

        if ((self.iteration > self.min_iter) and
            (self.residual_norms[-1] <
             self.threshold * self.residual_norms[0])):
            return True

        if self.iteration > minimizer.size:
            return True

        self.iteration += 1
        return False


class ConjugateGradient:
    def __init__(self, hessian, init, second_term, max_iter,
                 min_iter=1, threshold=1e-5, restart=50, callback=None,
                 precond=None):
        if not callable(hessian):
            raise ValueError("hessian must be callable")
        else:
            self.hessian = hessian

        self.minimizer = init
        self.second_term = second_term

        if type(max_iter) is not int:
            raise ValueError("max_iter must be an integer")
        else:
            self.max_iter = max_iter

        if type(min_iter) is not int:
            raise ValueError("min_iter must be an integer")
        else:
            self.min_iter = min_iter

        if type(restart) is not int:
            raise ValueError("restart must be an integer")
        else:
            self.restart = restart

        if type(threshold) is not float:
            raise ValueError("threshold must be a float")
        elif threshold < 0:
            raise ValueError("threshold must be positive")
        else:
            self.threshold = threshold

        if not callable(callback):
            raise ValueError("callback must be callable")
        else:
            self.callback = callback
        if self.min_iter > self.max_iter:
            warnings.warn("Maximum iteration ({0}) is lower than "
                          "minimum iteration ({1}). "
                          "Maximum set to mininum".format(self.max_iter,
                                                          self.min_iter))
            self.max_iter = self.min_iter

        self._residual = self.second_term - self.hessian(self.minimizer)
        self._direction = (self.precond(self._residual)
                           if self.precond
                           else self._residual)
        self.residual_norm = [np.sum(np.abs(self._residual)**2)]
        self.iteration = 0

    @property
    def step(self):
        return self._step[-1]

    @step.setter()
    def step(self, value):
        self._step.append(value)

    @property
    def residual_norm(self):
        return self._residual_norm[-1]

    @step.setter()
    def residual_norm(self, value):
        self._.append(value)

    @property
    def residual(self):
        # r^(i+1) = b - Qx^(i)
        if (self.iteration % self.restart) == 0:
            return self.second_term - self.hessian(self.minimizer)
        # r^(i+1) = r^(i) - s * Qd
        else:
            self._residual -= self.step * self.projection
            return self._residual

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration == self.max_iter:
            raise StopIteration

        self.projection = self.hessian(self._direction)
        variance = np.sum(np.real(np.conj(self._direction) * self.projection))
        self.step = self.residual_norm[-1] / variance

        self.minimizer += self.step * self._direction

        residual = self.residual()

        secant = self.precond(residual) if self.precond else residual
        self.residual_norm.append(np.sum(np.real(np.conj(residual) * secant)))
        beta = self.residual_norm[-1] / self.residual_norm[-2]
        self._direction = secant + beta * self._direction

        self.iteration += 1

        if self.callback():
            self.callback(self)

        return self.minimizer


def conj_grad2(f_hessian_proj, minimizer, second_term, f_precond=None,
               settings=None):
    # Init gradient
    residual = second_term - f_hessian_proj(minimizer)
    descent = f_precond(residual) if f_precond else residual
    residual_norm = [np.sum(np.abs(residual)**2)]

    sample = np.zeros_like(minimizer)

    for direction, step, var in conjugated_direction():
        minimizer += step * direction
        sample += randn() / np.sqrt(var) * direction

    return (minimizer, sample)


def nconj_grad(f_hessian_proj, minimizer, second_term, restart=50,
               precond=None):
    # Init gradient
    residual = second_term - f_hessian_proj(minimizer)
    sample = np.zeros_like(minimizer)

    direction = precond * residual if precond else residual
    norm = np.sum(np.abs(residual)**2)

    for iteration in range(minimizer.size):
        hess_proj = f_hessian_proj(direction)
        variance = np.sum(np.real(np.conj(direction) * hess_proj))
        step = residual_norm[-1] / variance

        minimizer += step * direction
        sample = sample + randn() / np.sqrt(variance) * direction

        if iteration % restart == 0:
            residual = second_term - f_hessian_proj(minimizer)
        else:
            residual = residual - step * hess_proj

        secant = precond * residual if precond else residual
        norm, previous = np.sum(np.real(np.conj(residual) *
                                        secant)), norm
        direction = secant + norm / previous * direction

    return (minimizer, sample)


class conj_grad(Algorithm):
    residual = second_term - f_hessian_proj(minimizer)
    sample = np.zeros_like(minimizer)

    direction = precond * residual if precond else residual
    norm = np.sum(np.abs(residual)**2)

    for iteration in range(minimizer.size):
        hess_proj = f_hessian_proj(direction)
        variance = np.sum(np.real(np.conj(direction) * hess_proj))
        step = residual_norm[-1] / variance

        minimizer += step * direction
        sample = sample + randn() / np.sqrt(variance) * direction

        if iteration % restart == 0:
            residual = second_term - f_hessian_proj(minimizer)
        else:
            residual = residual - step * hess_proj

        secant = precond * residual if precond else residual
        norm, previous = np.sum(np.real(np.conj(residual) *
                                        secant)), norm
        direction = secant + norm / previous * direction
