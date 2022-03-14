#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W291, E265

# Copyright (c) 2013, 2016 François Orieux <orieux@l2s.centralesupelec.fr>

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
from math import floor, ceil

import numpy as np
from scipy.ndimage.filters import convolve
import pywt

from . import udft

__author__ = "François Orieux"
__copyright__ = "2015, 2016 F. Orieux <orieux@l2s.centralesupelec.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "0.1.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@l2s.centralesupelec.fr"
__status__ = "alpha"
__url__ = "research.orieux.fr"
__keywords__ = "Inverse problems"


# class DiffIr(enum.Enum):
#     diff_col = np.array([0, -0.5, 0.5])
#     diff_row = np.array([[0], [-0.5], [0.5]])
#     laplacian = np.array([[0, -0.25, 0],
#                           [-0.25, 1, -0.25],
#                           [0, -0.25, 0]])


def diff_ir(ndim, axis=0, order=1):
    """Return a differiental impulsionnal response"""
    if axis >= ndim:
        raise ValueError("axis parameter must be < ndim")
    sig = np.zeros((2 * order + 1, ))
    sig[order] = 1
    ir = np.diff(sig, order)
    ir /= np.sum(np.abs(ir))
    return np.reshape(ir, [1 if a != axis else -1 for a in range(ndim)])


def _is_power2(num):
    """states if a number is a power of two"""
    return num != 0 and ((num & (num - 1)) == 0)


class LinearOperator(abc.ABC):
    def __init__(self, in_shape=None, out_shape=None, name=''):
        self.name = name
        self.in_shape = in_shape
        self.out_shape = out_shape

    @property
    def size_in(self):
        return np.prod(self.in_shape)

    @property
    def size_out(self):
        return np.prod(self.out_shape)

    @abc.abstractmethod
    def forward(self, obj):
        return NotImplemented

    @abc.abstractmethod
    def reverse(self, obj):
        return NotImplemented

    def t(self, obj):
        return self.reverse(obj)

    def fwrev(self, obj):
        return self.reverse(self.forward(obj))

    def __add__(self, op):
        if isinstance(obj, 'LinearOperator'):
            return Add(self, op)
        else:
            raise TypeError('op must be a LinearOperator')

    def __mul__(self, obj):
        if isinstance(obj, 'LinearOperator'):
            return Prod(self, obj)
        else:
            return self.forward(obj)

    def __rmul__(self, obj):
        return self.reverse(obj)

    def __matmul__(self, obj):
        return self.__mul__(obj)

    def __rmatmul__(self, obj):
        return self.__rmul__(obj)

    def __call__(self, obj):
        return self.forward(obj)

    def __repr__(self):
        return "{} ({}): {} → {}".format(
            self.name,
            type(self).__name__,
            str(self.in_shape),
            str(self.out_shape))


class Adjoint(LinearOperator):
    def __init__(self, linearop):
        """A linear operator that behave like the adjoint of `linearop`."""
        self.linearop = linearop

    @property
    def size_in(self):
        return self.linearop.size_out

    @property
    def size_out(self):
        return self.linearop.size_in

    def forward(self, obj):
        return self.linearop.reverse(obj)

    def reverse(self, obj):
        return self.linearop.forward(obj)


class CompositeProd(LinearOperator):
    def __init__(self, operators=None):
        self.operators = list(operators) if operators else []

    def forward(self, obj):
        tmp = obj
        for op in operators:
            tmp = op.forward(tmp)
        return tmp

    def reverse(self, obj):
        tmp = obj
        for op in reversed(operators):
            tmp = op.reverse(tmp)
        return tmp


class Prod(LinearOperator):
    def __init__(self, op_left, op_right):
        self.op_left = op_left
        self.op_right = op_right

    def forward(self, obj):
        return self.op_left.forward(self.op_right.forward(obj))

    def reverse(self, obj):
        return self.op_right.reverse(self.op_left.reverse(obj))


class Add(LinearOperator):
    def __init__(self, op_left, op_right):
        self.op_left = op_left
        self.op_right = op_right

    def forward(self, obj):
        return self.op_left.forward(obj) + self.op_right.forward(obj)

    def reverse(self, obj):
        return self.op_right.reverse(obj) + self.op_left.reverse(obj)


class Identity(LinearOperator):
    def __init__(self, shape, *args, **kwargs):
        super().__init__(in_shape=shape, out_shape=shape, *args, **kwargs)

    def forward(self, obj, ):
        return obj

    def reverse(self, obj):
        return obj

    def fwrev(self, obj):
        return obj


class DirectConvolution(LinearOperator):
    """Not correct"""
    def __init__(self, imp_resp):
        super().__init__()
        self.imp_resp = imp_resp
        self.imp_resp_T = imp_resp[self.imp_resp.ndim * [slice(0, -1, -1), ]]

    def forward(self, obj):
        return convolve(obj, self.imp_resp, self.mode, 'valid')

    def reverse(self, obj):
        return convolve(obj, self.imp_resp_T, self.mode, 'full')


class CircularConvolution(LinearOperator):
    def __init__(self, imp_resp: np.ndarray, shape: tuple,
                 real_in=True, real_out=True, *args, **kwargs):
        super().__init__(in_shape=shape, out_shape=shape, *args, **kwargs)
        self.shape = shape
        self.imp_resp = imp_resp
        self.freq_resp = udft.ir2fr(imp_resp, shape)
        self.ffilter = FrequencyFilter(self.freq_resp)
        self.real_in = real_in
        self.real_out = real_out


    def forward(self, obj):
        if self.real_in:
            tmp = self.ffilter.forward(udft.urdftn(obj))
        else:
            tmp = self.ffilter.forward(obj)

        if self.real_out:
            return udft.uirdftn(tmp, obj.ndim, self.shape)
        else:
            return tmp

    def reverse(self, obj):
        if self.real_out:
            tmp = self.ffilter.reverse(udft.urdftn(obj))
        else:
            tmp = self.ffilter.reverse(obj)

        if self.real_in:
            return udft.uirdftn(tmp, obj.ndim, self.shape)
        else:
            return tmp

    def fwrev(self, obj):
        if self.real_in:
            tmp = self.ffilter.fwrev(udft.urdftn(obj))
        else:
            tmp = self.ffilter.fwrev(obj)

        if self.real_out:
            return udft.uirdftn(tmp, obj.ndim, self.shape)
        else:
            return tmp


class Convolution(LinearOperator):
    """Make non circular convolution with fft"""
    def __init__(self, imp_resp: np.ndarray, out_shape: tuple,
                 name=''):
        in_shape = tuple(len_data + len_ir - 1 for
                         len_ir, len_data in zip(imp_resp.shape, out_shape))
        super().__init__(in_shape=in_shape, out_shape=out_shape, name=name)
        self.circ_conv = CircularConvolution(imp_resp, in_shape)
        self.idx = [slice(floor(len_ir / 2), -floor((len_ir + 1) / 2) + 1)
                    for len_ir in imp_resp.shape]

    @property
    def imp_resp(self):
        return self.circ_conv.imp_resp

    def forward(self, obj):
        tmp = self.circ_conv.forward(obj)
        return tmp[self.idx]

    def reverse(self, obj):
        out = np.zeros(self.in_shape)
        out[self.idx] = obj
        return self.circ_conv.reverse(out)

    def fwrev(self, obj):
        out = np.zeros_like(obj)
        out[self.idx] = self.circ_conv.forward(obj)[self.idx]
        return self.circ_conv.reverse(out)


class FrequencyFilter(LinearOperator):
    def __init__(self, freq_resp):
        super().__init__(in_shape=freq_resp.shape, out_shape=freq_resp.shape)
        self.freq_resp = freq_resp
        self._resp_conj = np.conj(self.freq_resp)
        self._square_mod_resp = np.abs(self.freq_resp)**2

    def forward(self, obj):
        return self.freq_resp * obj

    def reverse(self, obj):
        return self._resp_conj * obj

    def fwrev(self, obj):
        return self._square_mod_resp * obj



class Diff(LinearOperator):
    def __init__(self, axis, in_shape, out_shape, *args, **kwargs):
        super().__init__(in_shape=in_shape, out_shape=out_shape,
                         *args, **kwargs)
        self.axis = axis

    def response(self, ndim):
        ir = np.zeros(ndim * [2])
        index = ndim * [0]
        index[self.axis] = slice(None, None)
        ir[tuple(index)] = [1 / 2, -1 / 2]
        return ir

    def freq_response(self, ndim, shape):
        return udft.ir2fr(self.response(ndim), shape)

    def forward(self, obj):
        """
        -1  1  0  0
         0 -1  1  0
         0  0 -1  1
         0  0  0 -1
        """
        out_shape = [1 if pos == self.axis else in_size
                     for pos, in_size in enumerate(obj.shape)]
        return np.diff(np.concatenate((obj, np.zeros(out_shape)),
                                      axis=self.axis),
                       axis=self.axis)

    def reverse(self, obj):
        """
        -1  0  0  0
         1 -1  0  0
         0  1 -1  0
         0  0  1 -1
        """
        out_shape = [1 if pos == self.axis else in_size
                     for pos, in_size in enumerate(obj.shape)]
        return np.diff(np.concatenate((np.zeros(out_shape), -obj),
                                      axis=self.axis),
                       axis=self.axis)


class CircDiff(CircularConvolution):
    def __init__(self, shape, axis, order=1, real_in=True, real_out=True):
        super().__init__(diff_ir(len(shape), axis, order), shape,
                         real_in=real_in, real_out=real_out)


# \
class CircularLaplacianConvolution(CircularConvolution):
    def __init__(self, dim, shape):
        super().__init__(
            udft.laplacian(dim, shape, real=True), shape)


class LaplacianFilter(FrequencyFilter):
    def __init__(self, dim, shape):
        super().__init__(udft.laplacian(dim, shape))


# \
class UDWT(LinearOperator):
    def __init__(self, wavelet, shape, level, *args, **kwargs):
        super().__init__(shape, shape, *args, **kwargs)
        self.wavelet = wavelet
        self.mode = 'periodization'
        self.shape = shape
        self.level = level
        self.slices = pywt.coeffs_to_array(
            pywt.wavedecn(np.empty(shape),
                          wavelet=wavelet,
                          mode='periodization',
                          level=level))[1]

    def forward(self, obj):
        return pywt.coeffs_to_array(pywt.wavedecn(
            obj, wavelet=self.wavelet, mode=self.mode,
            level=self.level))[0]

    def reverse(self, obj):
        return pywt.waverecn(pywt.array_to_coeffs(obj, self.slices),
                             wavelet=self.wavelet, mode=self.mode)


class SWT2_analy(LinearOperator):
    def __init__(self, wavelet, shape, level, trace_plotter=None):
        """shape is the shape of the image that is analysed"""
        super().__init__(shape,
                         (2 * shape[0],
                          2 * level * shape[1]),
                         trace_plotter)
        self.wavelet = wavelet
        self.shape = shape
        self.level = level

    def forward(self, obj):
        coeffs = pywt.swt2(obj, wavelet=self.wavelet, level=self.level)
        return np.hstack([np.vstack((np.hstack((coeff[0], coeff[1][0])),
                                     np.hstack((coeff[1][1], coeff[1][2]))))
                          for coeff in coeffs])

    def reverse(self, obj):
        coeffs = []
        for lv in range(self.level):
            subobj = obj[:, lv * 2 * self.shape[1]:
                         (lv + 1) * 2 * self.shape[1]]
            coeffs.append([subobj[:self.shape[0], :self.shape[1]],
                           [subobj[:self.shape[0], self.shape[1]:],
                            subobj[self.shape[0]:, :self.shape[1]],
                            subobj[self.shape[0]:, self.shape[1]:], ]])
        return pywt.iswt2(coeffs, self.wavelet)


class SWT2_synth(SWT2_analy):
    def __init__(self, wavelet, shape, level, trace_plotter=None):
        """shape is the shape of the image that is synthetised"""
        super().__init__(wavelet, shape, level, trace_plotter)
        self.out_shape = shape
        self.in_shape = (2 * shape[0], 2 * level * shape[1])

    def forward(self, obj):
        return super().reverse(obj)

    def reverse(self, obj):
        return super().forward(obj)
