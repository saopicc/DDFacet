#!/usr/bin/env python
# -*- coding: utf-8 -*-
# plot.py --- Ploting utilities

# Copyright (c) 2011, 2016  <orieux@l2s.centralesupelec.fr>

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

"""Ploting utilities
"""

# code:

import os

import numpy as np
import matplotlib.pyplot as plt

plt.gray()

__author__ = "François Orieux"
__copyright__ = "Copyright (C) 2011, 2016 <orieux@l2s.centralesupelec.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "0.1.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@l2s.centralesupelec.fr"
__status__ = "development"
__url__ = ""
__keywords__ = "plotting"


def mirror_sym(image):
    return np.hstack((
        np.vstack((image,
                   np.flipud(image))),
        np.vstack((np.fliplr(image),
                   np.fliplr(np.flipud(image))))))


def specshow_woedge(image, **kwargs):
    """Plot the spectrum of an image nicely"""
    return plt.imshow(np.fft.fftshift(np.log(np.abs(np.fft.fft2(
        mirror_sym(image), **kwargs)[::2, ::2]))))


def specshow(image, **kwargs):
    """Plot the spectrum of an image nicely"""
    return plt.imshow(np.fft.fftshift(np.log(np.abs(np.fft.fft2(
        image, **kwargs)))))


def cm2inch(centimeters):
    """Centimeters to inches"""
    return 0.393701 * centimeters


def inch2cm(inches):
    """Inches to centimeters"""
    return inches / 0.393701


def figsize(width):
    """Return figsize mpl parameter from a width in cm"""
    return (cm2inch(width), cm2inch(width) * 6 / 8)


def savefig(path):
    """mpl savefig with common options"""
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0)


def legend():
    """mpl savefig with common options"""
    plt.legend(loc='best', prop={'size': 8})


def pdfcrop(directory):
    os.system(
        "cd " + directory + ";"
        "for file in *.pdf; do "
        "pdfcrop $file $file > /dev/null; "
        "done")


def pngcrop(directory):
    os.system("cd " + directory + "; mogrify -trim *.png")


def pdfmerge(directory):
    os.system(
        "cd " + directory + ";"
        "rm -f all.pdf; "
        "pdfunite *.pdf all.pdf")


# Common setup for matplotlib
params = {'backend': 'pdf',
          'image.interpolation': 'nearest',
          'savefig.dpi': 300,
          'axes.labelsize': 10,
          'grid.color': 'gray',
          'font.size': 8,
          'legend.fontsize': 10,
          'figure.figsize': (cm2inch(8.8), cm2inch(8.8) * 6 / 8),
          'xtick.labelsize': 10,
          'ytick.major.pad': 6,
          'xtick.major.pad': 6,
          'ytick.labelsize': 10,
          'font.family': 'sans-serif'}

# Publication quality from https://github.com/jbmouret/matplotlib_for_papers
params2 = {'backend': 'pdf',
           'image.interpolation': 'nearest',
           'axes.labelsize': 8,
           'font.size': 8,
           'legend.fontsize': 10,
           'xtick.labelsize': 10,
           'ytick.labelsize': 10,
           'text.usetex': False,
           'figure.figsize': (cm2inch(4.4), cm2inch(4.4) * 3 / 4),
           'savefig.dpi': 300,
           'grid.color': 'gray',
           'font.size': 8,
           'ytick.major.pad': 6,
           'xtick.major.pad': 6,
           'font.family': 'sans-serif'}

pgf_with_latex = {'pgf.texsystem': 'pdflatex',
                  'text.usetex': True,
                  'font.family': 'serif',
                  'font.serif': [],
                  'font.sans-serif': [],
                  'font.monospace': [],
                  'axes.labelsize': 10,
                  'text.fontsize': 10,
                  'legend.fontsize': 8,
                  'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'figure.figsize': (cm2inch(8.8), cm2inch(8.8) * 3 / 4),
                  'pgf.preamble': [
                      r'\usepackage[utf8]{inputenc}',
                      r'\usepackage[T1]{fontenc}', ]}

# matplotlib.rcParams.update(PARAMS)
