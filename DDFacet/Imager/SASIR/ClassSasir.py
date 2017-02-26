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

# DESCRIBE THE SASIR CLASS

import numpy as np
import logging
import time
from astropy.io import fits
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)


class ClassSasir: # inherits from FitsImage but overriding __init__ to get rid of FITS file processing
    def __init__(self,Dirty,PSF,DictSasirParms,GD=None):

        # manage Dirty
        self.Dirty=Dirty
        self.PSF=PSF
        self.DictSasirParms=DictSasirParms
        self.InitSasir(self.DictSasirParms)

    def InitSasir(self,DictSasirParms):
        pass
        # load Sasir parms


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
        pass
        # Proper Moresane run
        #if self.singlerun:
        #    print "Single run"
        #    self.moresane(self.subregion, self.scalecount, self.sigmalevel, self.loopgain, self.tolerance, self.accuracy,
        #                  self.majorloopmiter, self.minorloopmiter, self.allongpu, self.decommode, self.corecount,
        #                  self.convdevice, self.convmode, self.extractionmode, self.enforcepositivity,
        #                  self.edgesuppression, self.edgeoffset,self.fluxthreshold, self.negcomp, self.edgeexcl, self.intexcl)
        #else:
        #    print "By scale"
        #    self.moresane_by_scale(self.startscale, self.stopscale, self.subregion, self.sigmalevel, self.loopgain, self.tolerance, self.accuracy,
        #                           self.majorloopmiter, self.minorloopmiter, self.allongpu, self.decommode, self.corecount,
        #                           self.convdevice, self.convmode, self.extractionmode, self.enforcepositivity,
        #                           self.edgesuppression, self.edgeoffset,self.fluxthreshold, self.negcomp, self.edgeexcl, self.intexcl)
        return self.model,self.residual #IslandModel

    def residuals(self):
        pass
        return self.residual

    def GiveCLEANBeam(self,PSF,cellsize):
        pass
        cellsizeindeg=cellsize*1./3600 # to convert arcsec in degrees
        fakePSFFitsHeader={"CDELT1":cellsize,"CDELT2":cellsize}

        clean_beam, beam_params=beam_fit(self.psf_data,fakePSFFitsHeader)
        return clean_beam,beam_params

    def GiveBeamArea(self,beam_params,cellsize):
        pass
        cellsizedeg=cellsize*1./3600
        bx,by,_=beam_params     # bx and by in degrees
        px,py=cellsizedeg,cellsizedeg # in degrees
        BeamArea=np.pi*bx*by/(4*np.log(2)*px*py)
        return BeamArea
