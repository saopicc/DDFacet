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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import numpy as np
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("MaskMachine")
from astropy.io import fits as pyfits
import scipy.special
import copy
from DDFacet.Imager.ModModelMachine import ClassModModelMachine
from DDFacet.Imager.MSMF import ClassImageDeconvMachineMSMF

def OR(a,b):
    if a is None and b is None:
        raise RuntimeError('Trying to combine two null masks')
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return np.logical_or(a,b)

class ClassMaskMachine():
    def __init__(self,GD):
        self.GD=GD
        self.CurrentMask=None
        self.CurrentNegMask=None
        self.ThresholdMask=None
        #self.setCatalogMask()
        #self.readExternalMaskFromFits()
        self.NoiseMask=None
        self.ExternalMask=None
        self.readExternalMaskFromFits()
        self.DoMask=(self.GD["Mask"]["Auto"] or self.GD["Mask"]["External"])
        self.ImageNoiseMachine=None



    def updateNoiseMap(self):
        if self.Restored:
            print("Computing noise map based on brutal restored", file=log)
            self.NoiseMap=self.giveNoiseMap(self.Restored)
            nx,ny=self.NoiseMap.shape
            self.NoiseMapReShape=self.NoiseMap.reshape((1,1,nx,ny))
        else:
             print("Computing noise map based on residual image", file=log)
        
    def setImageNoiseMachine(self,ImageNoiseMachine):
        self.ImageNoiseMachine=ImageNoiseMachine

    def updateMask(self,DicoResidual):
        if not self.DoMask: return
        print("Computing Mask", file=log)
        
        if self.GD["Mask"]["Auto"] and self.GD["Deconv"]["Mode"] in ["HMP", "SSD", "SSD2"]:
            self.ImageNoiseMachine.calcNoiseMap(DicoResidual)
            self.NoiseMask=(self.ImageNoiseMachine.FluxImage>self.GD["Mask"]["SigTh"]*self.ImageNoiseMachine.NoiseMapReShape)
        elif self.GD["Mask"]["Auto"]:
            raise RuntimeError("Automasking only supported under HMP and SSD. Use Mask-External instead")
        
        if self.NoiseMask is not None: 
            print("  Merging Current mask with Noise-based mask", file=log)
            self.CurrentMask = OR(self.CurrentMask,self.NoiseMask)
        if self.ExternalMask is not None:
            if self.GD["Mask"]["Auto"]:
                print("  Merging Current mask with external mask", file=log)
                self.CurrentMask = OR(self.CurrentMask,self.ExternalMask)
            else:
                self.CurrentMask = self.ExternalMask

        if self.CurrentMask is not None:
            self.CurrentNegMask=self.giveOpposite(self.CurrentMask)

    def readExternalMaskFromFits(self):
        CleanMaskImage=self.GD["Mask"]["External"]
        if not CleanMaskImage: return
        print("  Reading mask image: %s"%CleanMaskImage, file=log)
        with pyfits.open(CleanMaskImage) as f:
            if len(f) != 1:
                raise RuntimeError("Currently only supports external masks with a single HDU")
            MaskImage = f[0].data
            if np.abs(MaskImage).max() < 1.0e-10:
                raise RuntimeError("Provided external mask is empty! Will not continue.")
            if f[0].header["NAXIS"] != 4:
                raise RuntimeError("External Mask must be 4 dimensional: RA x DEC x STOKES x CHAN")
            
        nch,npol,_,_=MaskImage.shape
        MaskArray=np.zeros(MaskImage.shape,np.bool8)
        for ch in range(nch):
            for pol in range(npol):
                MaskArray[ch,pol,:,:]=np.bool8(MaskImage[ch,pol].T[::-1].copy())[:,:]
        self.ExternalMask=MaskArray

    def joinExternalMask(self,Mask):
        self.ExternalMask=OR(self.ExternalMask,Mask)

    def giveOpposite(self,Mask):
        return np.bool8(1-Mask)


    def AdaptMaskShape(self):
        _,_,NMask,_=self._MaskArray.shape
        if NMask!=NDirty:
            print("  Mask do not have the same shape as the residual image", file=log)
            self._MaskArray=self.AdaptArrayShape(self._MaskArray,NDirty)
            self.MaskArray=self._MaskArray[0]
            self.IslandArray=np.zeros_like(self._MaskArray)
            self.IslandHasBeenDone=np.zeros_like(self._MaskArray)


