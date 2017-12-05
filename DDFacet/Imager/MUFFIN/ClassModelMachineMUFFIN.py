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
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassModelMachineSSD")
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
        self.DicoModel["Type"]="MUFFIN"
        self.ModelImageCube=0

    def setRefFreq(self,RefFreq,Force=False):#,AllFreqs):
        if self.RefFreq is not None and not Force:
            print>>log,ModColor.Str("Reference frequency already set to %f MHz"%(self.RefFreq/1e6))
            return
        self.RefFreq=RefFreq
        self.DicoModel["RefFreq"]=RefFreq
        
    def setFreqMachine(self,GridFreqs, DegridFreqs):
        # Initialise the Frequency Machine
        self.FreqMachine = ClassFrequencyMachine.ClassFrequencyMachine(GridFreqs, DegridFreqs, self.DicoModel["RefFreq"], self.GD)
        self.GridFreqs=GridFreqs

    def ToFile(self,FileName,DicoIn=None):
        print>>log, "Saving dico model to %s"%FileName
        if DicoIn is None:
            D=self.DicoModel
        else:
            D=DicoIn

        D["GD"]=self.GD
        D["ModelShape"]=self.ModelShape
        D["FreqsCube"]=self.GridFreqs
        D["Type"]="MUFFIN"

        MyPickle.Save(D,FileName)

    def giveDico(self):
        D=self.DicoModel
        D["GD"]=self.GD
        D["ModelShape"]=self.ModelShape
        D["Type"]="MUFFIN"
        return D

    def FromFile(self,FileName):
        print>>log, "Reading dico model from file %s"%FileName
        self.DicoModel=MyPickle.Load(FileName)
        self.FromDico(self.DicoModel)


    def FromDico(self,DicoModel):
        print>>log, "Reading dico model from dico with %i components"%len(DicoModel["Comp"])
        #self.PM=self.DicoModel["PM"]
        self.DicoModel=DicoModel
        self.RefFreq=self.DicoModel["RefFreq"]
        self.ModelShape=self.DicoModel["ModelShape"]
            
        
    def setModelShape(self,ModelShape):
        self.ModelShape=ModelShape

    def setThreshold(self,Th):
        self.Th=Th

    def setMUFFINModel(self,ModelImageCube):
        self.ModelImageCube += ModelImageCube

    def GiveModelImage(self,FreqIn=None):
        
        FreqIn = np.array([FreqIn.ravel()]).flatten()

        iFreqSliceMuffinCube = np.argmin(np.abs(FreqIn.reshape(-1,1)-self.GridFreqs.reshape(1,-1)),axis=1)

        nchan = FreqIn.size
        _,npol,nx,ny=self.ModelShape
        ModelImage = np.zeros((nchan,npol,nx,ny),dtype=np.float32)

        ModelImage_ = self.ModelImageCube[:,:,iFreqSliceMuffinCube]
        ModelImage_ = ModelImage_.transpose(2,1,0)
        
        ModelImage[:,0,:,:] = ModelImage_
 
        return ModelImage
        

        
    def setListComponants(self,ListScales):
        self.ListScales=ListScales

    def GiveSpectralIndexMap(self, threshold=0.1, save_dict=True):
        # Get the model image
        IM = self.GiveModelImage(self.FreqMachine.Freqsp)
        nchan, npol, Nx, Ny = IM.shape

        # Fit the alpha map
        self.FreqMachine.FitAlphaMap(IM[:, 0, :, :], threshold=threshold) # should set threshold based on SNR of final residual

        if save_dict:
            FileName = self.GD['Output']['Name'] + ".Dicoalpha"
            print>>log, "Saving componentwise SPI map to %s"%FileName

            MyPickle.Save(self.FreqMachine.alpha_dict, FileName)

        return self.FreqMachine.weighted_alpha_map.reshape((1, 1, Nx, Ny))


