#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2015, 2016 François Orieux <orieux@l2s.centralesupelec.fr>

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

import abc
import numbers

import numpy as np
import numpy.random as npr

__author__ = "François Orieux"
__copyright__ = "2016 F. Orieux <orieux@l2s.centralesupelec.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "0.1.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@l2s.centralesupelec.fr"
__status__ = "alpha"
__url__ = "research.orieux.fr"
__keywords__ = "Inverse problems"


class Criterion(abc.ABC):
    def __init__(self, precision=1):
        self.precision = precision

    @abc.abstractmethod
    def value(self, obj):
        raise NotImplemented

    @abc.abstractmethod
    def gradient(self, obj):
        raise NotImplemented

    @abc.abstractmethod
    def hessian(self, obj):
        raise NotImplemented

    def precond(self, obj):
        return obj

    def __mul__(self, value):
        if isinstance(value, numbers.Number):
            self.precision = value
        else:
            raise TypeError("unsupported operand type(s) for +"
                            ": 'criterion' and {}".format(type(value)))
        return self

    def __rmul__(self, value):
        return self.__mul__(value)

    def __add__(self, value):
        if isinstance(self, Composite):
            crit_list = list(self.criterions)
        else:
            crit_list = [self]

        if isinstance(value, Composite):
            crit_list.extend(value.criterions)
            return Composite(crit_list)
        elif isinstance(value, Criterion):
            crit_list.append(value)
            return Composite(crit_list)
        else:
            raise TypeError("unsupported operand type(s) for +"
                            ": 'Criterion' and {}".format(type(value)))

    def __radd__(self, value):
        return self.__add__(value)

    def __call__(self, obj):
        return self.value(obj)


class Composite(Criterion):
    def __init__(self, criterions=None):
        self.criterions = list(criterions) if criterions else []

    def value(self, obj):
        return sum(crit.value(obj) for crit in self.criterions)

    def gradient(self, obj):
        return sum(crit.gradient(obj) for crit in self.criterions)

    def hessian(self, obj):
        return sum(crit.hessian(obj) for crit in self.criterions)

    def __getattr__(self, name):
        return sum(getattr(crit, name) for crit in self.criterions)


class LinearMeanSquare(Criterion):
    def __init__(self, model, data=None, precision=1, metric=None):
        """Metric Q is diagonal

        J(x) = γ(w - Hx)ᵗ Q (w - Hx) / 2

        He = γ HᵗQH

        b = γ HᵗQw

        `model` is H
        `data` is y
        `metric` is diagonal of Q
        `precision` is γ"""
        super().__init__(precision)
        self.model = model
        self.data = data
        self.metric = metric

        # b = HᵗQw
        if data is None:
            self.second_term = 0
        else:
            if metric is None:
                self.second_term = self.precision * self.model.reverse(
                    self.data)
            else:
                self.second_term = self.precision * self.model.reverse(
                    self.metric * self.data)

        # c = wQw
        if self.data is None:
            self.constant = 0
        else:
            if self.metric is None:
                self.constant = self.precision * np.sum(
                    self.data**2)
            else:
                self.constant = self.precision * np.sum(
                    self.metric * self.data**2)

    def value(self, obj):
        if self.data is not None:
            if self.metric is None:
                return self.precision * np.sum(np.abs(
                    self.data - self.model.forward(obj))**2) / 2
            else:
                residual = self.data - self.model.forward(obj)
                return self.precision * np.sum(
                    np.abs(residual.conj() * self.metric * residual)) / 2
        else:
            out = self.model.forward(obj)
            if self.metric is None:
                return self.precision * np.sum(np.abs(out)**2) / 2
            else:
                return self.precision * np.sum(
                    np.abs(out.conj() * self.metric * out)) / 2

    def gradient(self, obj):
        return self.precision * (self.hessian(obj) - 2 * self.second_term)

    def hessian(self, obj):
        """He = HᵗQH"""
        if self.metric is None:
            return self.precision * self.model.fwrev(obj)
        else:
            return self.precision * self.model.reverse(
                self.metric * self.model.forward(obj))

    def perturbed(self):
        if self.data is None:
            if self.metric is None:
                return LinearMeanSquare(
                    self.model,
                    npr.standard_normal(self.model.out_shape) /
                    np.sqrt(self.precision),
                    self.precision,
                    self.metric)
            else:
                return LinearMeanSquare(
                    self.model,
                    npr.standard_normal(self.model.out_shape) /
                    np.sqrt(self.precision * self.metric),
                    self.precision,
                    self.metric)
        else:
            if self.metric is None:
                return LinearMeanSquare(
                    self.model,
                    self.data +
                    npr.standard_normal(self.model.out_shape) /
                    np.sqrt(self.precision),
                    self.precision,
                    self.metric)
            else:
                return LinearMeanSquare(
                    self.model,
                    self.data +
                    npr.standard_normal(self.model.out_shape) /
                    np.sqrt(self.precision * self.metric),
                    self.precision,
                    self.metric)


# class Quadratic(Criterion):
#     def __init__(self, correlation, mean):
#         self.correlation = correlation
#         self.mean = mean
#         self._cst = np.sum(mean * self.hessian(mean))
#         self._st = self.hessian(mean)

#     @property
#     def second_term(self):
#         return self.precision * self._st

#     @property
#     def constant(self):
#         return self.precision * self._cst

#     def value(self, obj):
#         diff = obj - self.mean
#         return self.precision * np.sum(self.diff * self.hessian(diff)) / 2

#     def gradient(self, obj):
#         return self.precision * self.hessian(obj - self.mean)

#     def hessian(self, obj):
#         return self.precision * self.correlation.fwrev(obj)


class Convex(Criterion):
    def min_gy(self, obj):
        return obj - self.gradient(obj)

    def min_gr(self, obj):
        aux = self.inf * np.ones_like(obj)
        idx = (obj != 0)
        aux[idx] = self.gradient(obj[idx]) / (2 * obj[idx])
        return aux


class Huber(Convex):
    def __init__(self, threshold):
        self.threshold = threshold
        self.inf = 1

    def gradient(self, obj):
        return np.where(
            np.abs(obj) <= self.threshold,
            2 * obj,
            2 * self.threshold * np.sign(obj))

    def value(self, obj):
        return np.sum(
            np.where(
                np.abs(obj) <= self.threshold,
                obj**2,
                np.abs(obj)))

    def hessian(self, obj):
        return np.where(np.abs(obj) <= self.threshold, 2, 0)


class TotalVariation(Convex):
    def __init__(self, threshold):
        self.threshold = threshold
        self.inf = 1 / (2 * self.thresold)

    def gradient(self, obj):
        return obj / np.sqrt(self.thresold**2 + obj**2)

    def value(self, obj):
        return np.sum(np.sqrt(self.thresold**2 + obj**2))

    def hessian(self, obj):
        return (1 - obj**2 / (self.thresold**2 + obj**2)) / \
            np.sqrt(self.thresold**2 + obj**2)


class AugumentedLagragian(Criterion):
    def __init__(self, operator, slacks, multipliers, mu=1):
        self.operator = operator
        self.lms = LinearMeanSquare(
            operator, slacks + 2 * multipliers / mu, precision=mu)
        self.second_term = self.lms.second_term
        self.constant = (self.lms.constant -
                         2 * np.sum(multipliers**2) / mu -
                         np.sum(multipliers * slacks))

    def value(self, obj):
        return self.lms.value(obj) + self.constant

    def gradient(self, obj):
        return self.lms.gradient(obj)

    def hessian(self, obj):
        return self.lms.hessian(obj)
