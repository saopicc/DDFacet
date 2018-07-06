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
from astropy.io import fits

logger = logging.getLogger(__name__)


class ClassMoresane(FI): # inherits from FitsImage but overriding __init__ to get rid of FITS file processing
    def __init__(self,dirty,psf,mask=None,GD=None):
        self.dirty_data = dirty
        self.psf_data = psf

        self.mask_name=None
        if mask is not None:
            self.mask_name="NumpyMask"
            self.mask = mask
            self.mask = self.mask.reshape(self.mask.shape[-2], self.mask.shape[-1])
            self.mask = self.mask/np.max(self.mask)
            self.mask = fftconvolve(self.mask,np.ones([5,5]),mode="same")
            self.mask = self.mask/np.max(self.mask)

        self.dirty_data_shape = self.dirty_data.shape
        self.psf_data_shape = self.psf_data.shape

        self.complete = False
        self.model = np.zeros_like(self.dirty_data)
        self.residual = np.copy(self.dirty_data)



    def giveModelResid(self,*args,**kwargs):
        self.moresane(*args,**kwargs)
        return self.model,self.residual

