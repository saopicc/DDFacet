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

import numpy as np
import logging
import time

from DDFacet.Other import MyLogger
from pymoresane.main import FitsImage as FI

class ClassMoresaneSingleSlice(FI):
    def __init__(self,dirty,psf,mask=None,GD=None):
        self.dirty_data = dirty
        self.psf_data = psf

        #MyLogger.setSilent(["pymoresane.main"])
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
        with np.errstate(divide='ignore'): 
            self.moresane(*args,**kwargs)
        return self.model,self.residual

