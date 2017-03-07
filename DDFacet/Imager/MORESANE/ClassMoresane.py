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

# DESCRIBE THE MORESANE CLASS

import numpy as np
import logging
import time

import pymoresane.iuwt as iuwt
import pymoresane.iuwt_convolution as conv
import pymoresane.iuwt_toolbox as tools
import pymoresane.parser as pparser
from pymoresane.beam_fit import beam_fit
from pymoresane.main import FitsImage as FI # importing the class

from scipy.signal import fftconvolve
import pylab as plt
from astropy.io import fits

logger = logging.getLogger(__name__)


class ClassMoresane(FI): # inherits from FitsImage but overriding __init__ to get rid of FITS file processing
    def __init__(self,Dirty,PSF,DictMoresaneParms,GD=None):

        # manage Dirty
        self.Dirty=Dirty
        self.PSF=PSF
        self.DictMoresaneParms=DictMoresaneParms
        self.InitMoresane(self.DictMoresaneParms)

    def InitMoresane(self,DictMoresaneParms):
        # load Moresane parms

        self.singlerun=DictMoresaneParms['SingleRun']
        self.startscale=DictMoresaneParms['StartScale']
        self.stopscale=DictMoresaneParms['StopScale']
        self.subregion=DictMoresaneParms['SubRegion']
        self.scalecount=DictMoresaneParms['ScaleCount']
        self.sigmalevel=DictMoresaneParms['SigmaLevel']
        self.loopgain=DictMoresaneParms['LoopGain']
        self.tolerance=DictMoresaneParms['Tolerance']
        self.accuracy=DictMoresaneParms['Accuracy']
        self.majorloopmiter=DictMoresaneParms['MajorLoopMiter']
        self.minorloopmiter=DictMoresaneParms['MinorLoopMiter']
        self.allongpu=DictMoresaneParms['AllOnGpu']
        self.decommode=DictMoresaneParms['DecomMode']
        self.corecount=DictMoresaneParms['CoreCount']
        self.convdevice=DictMoresaneParms['ConvDevice']
        self.convmode=DictMoresaneParms['ConvMode']
        self.extractionmode=DictMoresaneParms['ExtractionMode']
        self.enforcepositivity=DictMoresaneParms['EnforcePositivity']
        self.edgesuppression=DictMoresaneParms['EdgeSuppression']
        self.edgeoffset=DictMoresaneParms['EdgeOffset']
        self.fluxthreshold=DictMoresaneParms['FluxThreshold']
        self.negcomp=DictMoresaneParms['NegComp']
        self.edgeexcl=DictMoresaneParms['EdgeExcl']
        self.intexcl=DictMoresaneParms['IntExcl']

        # Init
        mask_name= None
        self.mask_name = mask_name

        if self.mask_name is not None:
            self.mask = fits.open("{}".format(mask_name))[0].data
            self.mask = self.mask.reshape(self.mask.shape[-2], self.mask.shape[-1])
            self.mask = self.mask / np.max(self.mask)
            self.mask = fftconvolve(self.mask, np.ones([5, 5]), mode="same")
            self.mask = self.mask / np.max(self.mask)

        self.dirty_data=self.Dirty
        self.psf_data=self.PSF
        self.dirty_data_shape = self.dirty_data.shape
        self.psf_data_shape = self.psf_data.shape

        self.complete = False
        self.model = np.zeros_like(self.Dirty)
        self.residual = np.copy(self.Dirty)
        self.restored = np.zeros_like(self.Dirty)


    def main(self):

        # Proper Moresane run
        if self.singlerun:
            print "Single run"
            self.moresane(self.subregion, self.scalecount, self.sigmalevel, self.loopgain, self.tolerance, self.accuracy,
                          self.majorloopmiter, self.minorloopmiter, self.allongpu, self.decommode, self.corecount,
                          self.convdevice, self.convmode, self.extractionmode, self.enforcepositivity,
                          self.edgesuppression, self.edgeoffset,self.fluxthreshold, self.negcomp, self.edgeexcl, self.intexcl)
        else:
            print "By scale"
            self.moresane_by_scale(self.startscale, self.stopscale, self.subregion, self.sigmalevel, self.loopgain, self.tolerance, self.accuracy,
                                   self.majorloopmiter, self.minorloopmiter, self.allongpu, self.decommode, self.corecount,
                                   self.convdevice, self.convmode, self.extractionmode, self.enforcepositivity,
                                   self.edgesuppression, self.edgeoffset,self.fluxthreshold, self.negcomp, self.edgeexcl, self.intexcl)
        return self.model,self.residual #IslandModel

    def residuals(self):
        return self.residual

    def GiveCLEANBeam(self, PSF, cellsize):
        cellsizeindeg = cellsize * 1. / 3600  # to convert arcsec in degrees
        fakePSFFitsHeader = {"CDELT1": cellsizeindeg, "CDELT2": cellsizeindeg}

        clean_beam, beam_params = beam_fit(PSF, fakePSFFitsHeader)
        return clean_beam, beam_params

    def GiveBeamArea(self, beam_params, cellsize):
        print beam_params
        cellsizedeg = cellsize * 1. / 3600
        bx, by, _ = beam_params  # bx and by in degrees
        px, py = cellsizedeg, cellsizedeg  # in degrees
        BeamArea = np.pi * bx * by / (4 * np.log(2) * px * py)
        return BeamArea
