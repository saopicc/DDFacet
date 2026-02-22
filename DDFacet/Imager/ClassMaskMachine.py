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
log2=logger.getLogger("FilterMachine")
from astropy.io import fits as pyfits
import scipy.special
import copy
from DDFacet.Imager.ModModelMachine import ClassModModelMachine
from DDFacet.Imager.MSMF import ClassImageDeconvMachineMSMF
from astropy.io import fits
import numpy as np
import pylab
from DDFacet.ToolsDir import GeneDist
import scipy.stats
from DDFacet.Array import shared_dict
from scipy.signal import fftconvolve

def OR(a,b):
    if a is None and b is None:
        raise RuntimeError('Trying to combine two null masks')
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return np.logical_or(a,b)

def test():
    DicoDirty = shared_dict.create("AllImages_FMPSF")
    DicoDirty.restore("Cache.ReOrg/J1101_visit1_512ch.ms.N0.F0.D0.ddfcache/LastResidual")
    FM=ClassFilterMachine()
    FM.filterCube(DicoImages=DicoDirty,ThFilterRFI=5.)
    
class ClassFilterMachine():
    def __init__(self):
        pass
    
    def filterCube(self,DicoImages=None,ThFilterRFI=5.):
        ResidualCube=DicoImages["ImageCube"]
        nch,npol, Npix_x, Npix_y=ResidualCube.shape

        nx,ny=Npix_x, Npix_y
    
        NMax=0
        if NMax==0:
            NMax=int(np.sqrt((nx//2)**2+(ny//2)**2))//2
            if NMax%2!=0: NMax-=1
            
        log2.print("Filtering residual cube against RFI in the central %i pixels"%NMax)

        Mask=np.zeros((nx,ny),bool)
        
        #import pylab
        #pylab.clf()
        
        for ich in range(nch):
            A=ResidualCube[ich,0]
            fA=np.fft.fft2(A)
            fAs=np.fft.fftshift(fA)
            
            s_fAs=fAs[nx//2-NMax:nx//2+NMax+1,ny//2-NMax:ny//2+NMax+1]
            
            nxx,nyy=s_fAs.shape
            
            xx,yy=np.mgrid[-(nxx//2):nxx//2+1,-(nyy//2):nyy//2+1]
            dd=np.sqrt(xx**2+yy**2)

        
            dds=dd.flat[:]
            s_fAss=np.abs(s_fAs.flat[:])

            drange=np.linspace(0,dds.max())
            sig=5.
            L=[]
            for id0,d0 in enumerate(drange):
                dds0=d0-5*sig
                dds1=d0+5*sig
                ind=np.where((dds>dds0)&(dds<dds1))[0]
                ddss=dds[ind]
                
                W=np.exp(-(ddss-d0)**2/(2*sig**2))
                
                qq=GeneDist.weighted_quantile(s_fAss[ind], np.array([0.16,0.5,0.84]), sample_weight=W)
                L.append(qq)
            q0,q,q1=np.array(L).T
            
            Th=q0+(q1-q0)*ThFilterRFI
            ThIm=np.interp(dds.ravel(), drange, Th, left=None, right=None).reshape((nxx,nyy))
            Mask_ch=(s_fAss.reshape((nxx,nyy))>ThIm)

            nConv=5
            ConvMask=np.ones((nConv,nConv),np.float32)
            Mask_ch=(fftconvolve(np.float32(Mask_ch),ConvMask,mode="same")>1e-3)
            
            nn=np.count_nonzero(Mask_ch)
            log2.print("  [ch #%i] found %i masked pixels (~%.2f%%)"%(ich,nn,100*(nn/Mask_ch.size)))

            
            # ########################
            # MM=Mask[nx//2-NMax:nx//2+NMax+1,ny//2-NMax:ny//2+NMax+1]
            # Mask[nx//2-NMax:nx//2+NMax+1,ny//2-NMax:ny//2+NMax+1]=(MM|Mask_ch)[:,:]
            # ########################
            Mask.fill(0)
            Mask[nx//2-NMax:nx//2+NMax+1,ny//2-NMax:ny//2+NMax+1]=(Mask_ch)[:,:]
            # ########################
            factBias=(Mask.size-nn)/Mask.size
            fAs[Mask]=0
            sfAs=np.fft.ifftshift(fAs)
            fsfAs=np.fft.ifft2(sfAs)
            ResidualCube[ich,0,:,:]=fsfAs[:,:].real/factBias



            # ind=np.int64(np.random.rand(1000)*A.size)
            # RMS= scipy.stats.median_abs_deviation(A.flat[ind],axis=None,scale="normal")
            # pylab.subplot(1,2,1)
            # pylab.imshow(A,vmin=-5*RMS,vmax=50*RMS)
            # RMS1= scipy.stats.median_abs_deviation(fsfAs.flat[ind],axis=None,scale="normal")
            # pylab.subplot(1,2,2)
            # pylab.imshow(fsfAs.real,vmin=-5*RMS1,vmax=50*RMS1)
            # pylab.show()
            
        #     pylab.subplot(3,1,ich+1)
        #     pylab.scatter(dds,np.abs(s_fAss),label="%i"%ich)
        #     pylab.plot(drange,q0)
        #     pylab.plot(drange,q)
        #     pylab.plot(drange,q1)
        #     pylab.plot(drange,q0+(q1-q0)*5,color="black")
        
        # #F.writeto("%s.FT.fits"%FName,overwrite=True)
        # pylab.legend()
        # pylab.show()
            
        #log2.print("  Flagged %i pixels"%(np.count_nonzero(Mask[nx//2-NMax:nx//2+NMax+1,ny//2-NMax:ny//2+NMax+1])))
        if nch>1:
            WBAND=DicoImages["ImageInfo"]["WBAND"]
            DicoImages["MeanImage"] = np.sum(DicoImages["ImageCube"] * WBAND, axis=0).reshape((1, npol, Npix_x, Npix_y))
        else:
            DicoImages["MeanImage"][:] = DicoImages["ImageCube"][:]


    
    
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
        
        if self.GD["Mask"]["Auto"] and self.GD["Deconv"]["Mode"] in ["HMP", "SSD", "SSD2", "SSD3"]:
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

    def setMask(self,Mask):
        self.CurrentMask = Mask
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
            
        nch,npol,nx,ny=MaskImage.shape
        MaskArray=np.zeros((nch,npol,ny,nx),np.bool_)

        for ch in range(nch):
            for pol in range(npol):
                MaskArray[ch,pol,:,:]=np.bool_(MaskImage[ch,pol].T[::-1].copy())[:,:]
        self.ExternalMask=MaskArray

    def joinExternalMask(self,Mask):
        self.ExternalMask=OR(self.ExternalMask,Mask)

    def giveOpposite(self,Mask):
        return np.bool_(1-Mask)


    def AdaptMaskShape(self):
        _,_,NMask,_=self._MaskArray.shape
        if NMask!=NDirty:
            print("  Mask do not have the same shape as the residual image", file=log)
            self._MaskArray=self.AdaptArrayShape(self._MaskArray,NDirty)
            self.MaskArray=self._MaskArray[0]
            self.IslandArray=np.zeros_like(self._MaskArray)
            self.IslandHasBeenDone=np.zeros_like(self._MaskArray)


