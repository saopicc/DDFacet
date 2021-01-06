#!/usr/bin/env python
# -*- coding: utf-8 -*-
# physics.py --- Physics utilities

# keywords: physics, constant, formula
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

"""Physics utilities. Contains constants, formula, etc...
"""

# code:

import numpy as np

try:
    import scipy.constants
    SPEED_LIGHT = scipy.constants.c
    PLANCK_CST = scipy.constants.h
    BOLTZMANN_CST = scipy.constants.k
except ImportError:
    SPEED_LIGHT = 299792458.0
    PLANCK_CST = 6.6260693000000002e-34
    BOLTZMANN_CST = 1.3806505000000001e-23

__author__ = "François Orieux"
__copyright__ = "Copyright (C) 2011, 2012, 2013 F. Orieux <orieux@iap.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "0.1.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@iap.fr"
__status__ = "development"


def planck_law(wavelength, temperature, n_lambda):
    """Return the Planck law value"""
    c_lambda = SPEED_LIGHT / n_lambda
    return 2 * PLANCK_CST * c_lambda**2 / \
        (wavelength**5 * np.exp(BOLTZMANN_CST * c_lambda /
                                (BOLTZMANN_CST * wavelength * temperature))
        - 1)


def sigma2fwhm(sigma):
    """The FWHM (radian) from sigma (radian)"""
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def fwhm2sigma(fwhm):
    """The sigma (radian) from FWHM (radian)"""
    return fwhm / 2 * np.sqrt(2 * np.log(2))


def rad2arcsec(radians):
    """Arcsec from radians"""
    return radians * 180 * 60 * 60 / np.pi


def arcsec2rad(arcsecs):
    """Radians from arcsec"""
    return arcsecs / (180 * 60 * 60 / np.pi)
