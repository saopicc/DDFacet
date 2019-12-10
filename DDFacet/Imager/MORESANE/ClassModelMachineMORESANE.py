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
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
log=logger.getLogger("ClassModelMachineSSD")
from DDFacet.Array import NpParallel
from DDFacet.Array import ModLinAlg
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyPickle
from DDFacet.Other import reformat
from DDFacet.Imager import ClassFrequencyMachine
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.Imager import ClassModelMachine as ClassModelMachinebase
from DDFacet.ToolsDir import ModFFTW
import scipy.ndimage
from SkyModel.Sky import ModRegFile
from pyrap.images import image
from SkyModel.Sky import ClassSM
import os
import copy

class ClassModelMachine(ClassModelMachinebase.ClassModelMachine):
    def __init__(self,*args,**kwargs):
        ClassModelMachinebase.ClassModelMachine.__init__(self, *args, **kwargs)
        self.RefFreq=None
        self.DicoModel={}
        self.DicoModel["Type"]="MORESANE"

    def setRefFreq(self,RefFreq,Force=False):#,AllFreqs):
        if self.RefFreq is not None and not Force:
            print(ModColor.Str("Reference frequency already set to %f MHz"%(self.RefFreq/1e6)), file=log)
            return
        self.RefFreq=RefFreq
        self.DicoModel["RefFreq"]=RefFreq


    def ToFile(self,FileName,DicoIn=None):
        print("Saving dico model to %s"%FileName, file=log)
        if DicoIn is None:
            D=self.DicoModel
        else:
            D=DicoIn

        D["GD"]=self.GD
        D["ModelShape"]=self.ModelShape
        D["Type"]="MORESANE"

        MyPickle.Save(D,FileName)

    def giveDico(self):
        D=self.DicoModel
        D["GD"]=self.GD
        D["ModelShape"]=self.ModelShape
        D["Type"]="MORESANE"
        return D

    def FromFile(self,FileName):
        print("Reading dico model from file %s"%FileName, file=log)
        self.DicoModel=MyPickle.Load(FileName)
        self.FromDico(self.DicoModel)


    def FromDico(self,DicoModel):
        print("Reading dico model from dico with %i components"%len(DicoModel["Comp"]), file=log)
        #self.PM=self.DicoModel["PM"]
        self.DicoModel=DicoModel
        self.RefFreq=self.DicoModel["RefFreq"]
        self.ModelShape=self.DicoModel["ModelShape"]
            

        

        
    def setModelShape(self,ModelShape):
        self.ModelShape=ModelShape

    def setThreshold(self,Th):
        self.Th=Th

        

    def setModel(self,Image,Order):
        try:
            self.DicoModel[Order]+=Image
        except:
            self.DicoModel[Order]=Image

            
            


    def GiveModelImage(self,FreqIn=None,out=None):
        
        
        RefFreq=self.DicoModel["RefFreq"]
        if FreqIn is None:
            FreqIn=np.array([RefFreq])

        FreqIn=np.array([FreqIn.ravel()]).flatten()


        # print "ModelMachine GiveModelImage:",FreqIn, RefFreq

        _,npol,nx,ny=self.ModelShape
        nchan=FreqIn.size
        if out is not None:
            if out.shape != (nchan,npol,nx,ny) or out.dtype != np.float32:
                raise RuntimeError("supplied image has incorrect type (%s) or shape (%s)" % (out.dtype, out.shape))
            ModelImage = out
        else:
            ModelImage = np.zeros((nchan,npol,nx,ny),dtype=np.float32)

        if 0 in self.DicoModel.keys():
            C0=self.DicoModel[0]
        else:
            C0=0

        if 1 in self.DicoModel.keys():
            C1=self.DicoModel[1]
        else:
            C1=0

        ModelImage[:,:,:,:]=C0*(FreqIn.reshape((-1,1,1,1))/self.RefFreq)**C1
 
        return ModelImage
        

        
    def setListComponants(self,ListScales):
        self.ListScales=ListScales

    def GiveSpectralIndexMap(self, CellSizeRad=1., GaussPars=[(1, 1, 0)], DoConv=True, MaxSpi=100, MaxDR=1e+6,
                             threshold=None):
        dFreq = 1e6
        # f0=self.DicoSMStacked["AllFreqs"].min()
        # f1=self.DicoSMStacked["AllFreqs"].max()
        RefFreq = self.DicoSMStacked["RefFreq"]
        f0 = RefFreq / 1.5
        f1 = RefFreq * 1.5

        M0 = self.GiveModelImage(f0)
        M1 = self.GiveModelImage(f1)
        if DoConv:
            # M0=ModFFTW.ConvolveGaussian(M0,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
            # M1=ModFFTW.ConvolveGaussian(M1,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
            # M0,_=ModFFTW.ConvolveGaussianWrapper(M0,Sig=GaussPars[0][0]/CellSizeRad)
            # M1,_=ModFFTW.ConvolveGaussianWrapper(M1,Sig=GaussPars[0][0]/CellSizeRad)
            M0, _ = ModFFTW.ConvolveGaussianScipy(M0, Sig=GaussPars[0][0] / CellSizeRad)
            M1, _ = ModFFTW.ConvolveGaussianScipy(M1, Sig=GaussPars[0][0] / CellSizeRad)

        # print M0.shape,M1.shape
        # compute threshold for alpha computation by rounding DR threshold to .1 digits (i.e. 1.65e-6 rounds to 1.7e-6)
        if threshold is not None:
            minmod = threshold
        elif not np.all(M0 == 0):
            minmod = float("%.1e" % (np.max(np.abs(M0)) / MaxDR))
        else:
            minmod = 1e-6

        # mask out pixels above threshold
        mask = (M1 < minmod) | (M0 < minmod)
        print("computing alpha map for model pixels above %.1e Jy (based on max DR setting of %g)" % (
              minmod, MaxDR), file=log)
        M0[mask] = minmod
        M1[mask] = minmod
        # with np.errstate(invalid='ignore'):
        #    alpha = (np.log(M0)-np.log(M1))/(np.log(f0/f1))
        # print
        # print np.min(M0),np.min(M1),minmod
        # print
        alpha = (np.log(M0) - np.log(M1)) / (np.log(f0 / f1))
        alpha[mask] = 0

        # mask out |alpha|>MaxSpi. These are not physically meaningful anyway
        mask = alpha > MaxSpi
        alpha[mask] = MaxSpi
        masked = mask.any()
        mask = alpha < -MaxSpi
        alpha[mask] = -MaxSpi
        if masked or mask.any():
            print(ModColor.Str("WARNING: some alpha pixels outside +/-%g. Masking them." % MaxSpi, col="red"), file=log)
        return alpha


