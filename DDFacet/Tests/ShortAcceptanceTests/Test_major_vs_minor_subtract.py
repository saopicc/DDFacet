#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

import subprocess
import unittest
import os
from os import path, getenv
from DDFacet.Parset.ReadCFG import Parset
import numpy as np
import argparse
import cPickle

# Sets up a test that compares a major cycle subtraction (i.e. R.H.dot(V - R I))
# versus a minor cycle subtraction (i.e. I_D - I_PSF * I) when there are large
# extended components spanning multiple facets. This should be very nearly the
# same when no beam is enabled. The test is also performed with the beam enabled
# to get an idea of the accuracy of the minor cycle when the beam is enabled.

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("outdict")
    p.add_argument('--npix', default=825, type=int)
    p.add_argument('--nchan', default=8, type=int)
    p.add_argument('--ncoeffs', default=4, type=int)
    p.add_argument('--sigma', default=150.0, type=float)
    return p

args = create_parser().parse_args()

# set up a DicoModel
Nchan = args.nchan
Ncoeffs = args.ncoeffs
Npix = args.npix
sigma = args.sigma

DicoModel = {}
loc = (Npix // 2, Npix // 2)
DicoModel["ModelShape"] = (Nchan, 1, Npix, Npix)
DicoModel["ListScales"] = [sigma]
DicoModel["RefFreq"] = 1.96e9
DicoModel["Comp"] = {}
iScale = 1
DicoModel["Comp"][iScale] = {}
DicoModel["Comp"][iScale]['NumComps'] = np.array(1, dtype=np.int16)
DicoModel["Comp"][iScale][loc] = {}  # place component at centre of image
DicoModel["Comp"][iScale][loc]['SolsArray'] = np.zeros(Ncoeffs, dtype=np.float32)
DicoModel["Comp"][iScale][loc]['SolsArray'][0] = 10000.0  # alpha = 0 spi model
DicoModel["Type"] = 'WSCMS'
DicoModel["Scale_Info"] = {}
DicoModel["Scale_Info"][iScale] = {}

# set up kernel and extents etc.
xtmp = np.arange(0.0, Npix)
tmpkern = np.exp(-xtmp ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi * sigma ** 2))
I = int(2 * np.round(np.argwhere(tmpkern >= 1e-6).squeeze()[-1]))
extent = int(np.minimum(I, int(Npix)))
if extent % 2 == 0:
    extent -= 1
volume = 2 * np.pi * sigma ** 2
diff = int((Npix - extent) // 2)
if diff == 0:
    I = slice(None)
else:
    I = slice(diff, -diff)
n = Npix // 2
x_unpadded, y_undpadded = np.mgrid[-n:n:1.0j * Npix, -n:n:1.0j * Npix]
rsq_unpadded = x_unpadded ** 2 + y_undpadded ** 2
kernel = np.exp(-rsq_unpadded[I, I] / (2 * sigma ** 2)) / volume

DicoModel["Scale_Info"][iScale]["sigma"] = sigma
DicoModel["Scale_Info"][iScale]["kernel"] = kernel
DicoModel["Scale_Info"][iScale]["extent"] = extent

cPickle.dump(DicoModel, file(args.outdict, 'w'), 2)

# def run_ddf(parset, image_prefix, stdout_filename, stderr_filename, beam_model="FITS"):
#     """ Execute DDFacet """
#     args = ['DDF.py', parset,
#             '--Output-Name=%s' % image_prefix,
#             '--Beam-Model=%s' % beam_model]
#     stdout_file = open(stdout_filename, 'w')
#     stderr_file = open(stderr_filename, 'w')
#
#     with stdout_file, stderr_file:
#         subprocess.check_call(args, env=os.environ.copy(),
#                               stdout=stdout_file, stderr=stderr_file)


