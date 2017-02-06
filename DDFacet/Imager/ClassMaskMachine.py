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
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("MaskMachine")
from pyrap.images import image
import scipy.special

OR=np.logical_or

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
        self.NoiseMap=None
        self.readExternalMaskFromFits()


    def giveNoiseMap(self,Image):
        print>>log, "Computing noise map"
        Boost=self.Step
        Acopy=Image[0,0,0::Boost,0::Boost].copy()
        SBox=(self.box[0]/Boost,self.box[1]/Boost)

        x=np.linspace(-10,10,1000)
        f=0.5*(1.+scipy.special.erf(x/np.sqrt(2.)))
        n=SBox[0]*SBox[1]
        F=1.-(1.-f)**n
        ratio=np.abs(np.interp(0.5,F,x))

        Noise=-scipy.ndimage.filters.minimum_filter(Acopy,SBox)/ratio

        NoiseMed=np.median(Noise)
        Noise[Noise<NoiseMed]=NoiseMed

        LargeNoise=np.zeros_like(Image[0,0])
        for i in range(Boost):
            for j in range(Boost):
                s00,s01=Noise.shape
                s10,s11=LargeNoise[i::Boost,j::Boost].shape
                s0,s1=min(s00,s10),min(s10,s11)
                LargeNoise[i::Boost,j::Boost][0:s0,0:s1]=Noise[:,:][0:s0,0:s1]
        ind=np.where(LargeNoise==0.)
        LargeNoise[ind]=1e-10
        return LargeNoise

    def updateResidual(self,DicoResidual):
        self.DicoResidual=DicoResidual
        if self.GD["Mask"]["Residual"]:
            self.Box,self.Step,Th=self.GD["Mask"]["Residual"]
            self.box=(self.Box,self.Box)
            _,_,nx,ny=DicoResidual["MeanImage"].shape
            self.NoiseMap=self.giveNoiseMap(DicoResidual["MeanImage"])
            self.NoiseMapReShape=self.NoiseMap.reshape((1,1,nx,ny))
            self.NoiseMask=(DicoResidual["MeanImage"][0,0]>Th*self.NoiseMapReShape)
        self.updateMask()

    def readExternalMaskFromFits(self):
        CleanMaskImage=self.GD["Mask"]["External"]
        if not CleanMaskImage: return
        print>>log, "Reading mask image: %s"%CleanMaskImage
        MaskImage=image(CleanMaskImage).getdata()
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

    def updateMask(self):

        if self.ExternalMask is not None: 
            self.CurrentMask = OR(self.CurrentMask,self.ExternalMask)
        if self.NoiseMask is not None: 
            self.CurrentMask = OR(self.CurrentMask,self.NoiseMask)

        if self.CurrentMask is not None:
            self.CurrentNegMask=self.giveOpposite(self.CurrentMask)

    def AdaptMaskShape(self):
        _,_,NMask,_=self._MaskArray.shape
        if NMask!=NDirty:
            print>>log,"Mask do not have the same shape as the residual image"
            self._MaskArray=self.AdaptArrayShape(self._MaskArray,NDirty)
            self.MaskArray=self._MaskArray[0]
            self.IslandArray=np.zeros_like(self._MaskArray)
            self.IslandHasBeenDone=np.zeros_like(self._MaskArray)

