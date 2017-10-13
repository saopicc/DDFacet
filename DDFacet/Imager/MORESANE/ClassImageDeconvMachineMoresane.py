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

import os
import numpy as np
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.Array import NpShared
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
from pyrap.images import image
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Imager import ClassGainMachine
from DDFacet.Other import MyPickle
import multiprocessing
import time
from DDFacet.Imager.MORESANE.ClassMoresaneSingleSlice import ClassMoresaneSingleSlice
from DDFacet.Array import shared_dict


class ClassImageDeconvMachine():
    def __init__(self,GD=None,ModelMachine=None,RefFreq=None,*args,**kw):
        self.GD=GD
        self.ModelMachine = ModelMachine
        self.RefFreq=RefFreq
        if self.ModelMachine.DicoModel["Type"]!="MORESANE":
            raise ValueError("ModelMachine Type should be MORESANE")


    def SetPSF(self,DicoVariablePSF):
        self.PSFServer=ClassPSFServer(self.GD)
        DicoVariablePSF=shared_dict.attach(DicoVariablePSF.path)#["CubeVariablePSF"]
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF)
        self.PSFServer.setRefFreq(self.ModelMachine.RefFreq)
        self.DicoVariablePSF=DicoVariablePSF
        
    def setMaskMachine(self,MaskMachine):
        self.MaskMachine=MaskMachine

    def GiveModelImage(self,*args): return self.ModelMachine.GiveModelImage(*args)

    def Update(self,DicoDirty,**kwargs):
        """
        Method to update attributes from ClassDeconvMachine
        """
        #Update image dict
        self.SetDirty(DicoDirty)

    def ToFile(self, fname):
        """
        Write model dict to file
        """
        self.ModelMachine.ToFile(fname)

    def FromFile(self, fname):
        """
        Read model dict from file SubtractModel
        """
        self.ModelMachine.FromFile(fname)

    def FromDico(self, DicoName):
        """
        Read in model dict
        """
        self.ModelMachine.FromDico(DicoName)

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe

    def Init(self,**kwargs):
        self.SetPSF(kwargs["PSFVar"])
        self.DicoVariablePSF["PSFSideLobes"]=kwargs["PSFAve"]
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
        self.ModelMachine.setRefFreq(kwargs["RefFreq"])
        # store grid and degrid freqs for ease of passing to MSMF
        #print kwargs["GridFreqs"],kwargs["DegridFreqs"]
        self.GridFreqs=kwargs["GridFreqs"]
        self.DegridFreqs=kwargs["DegridFreqs"]
        self.ModelMachine.setFreqMachine(kwargs["GridFreqs"], kwargs["DegridFreqs"])

    def SetDirty(self,DicoDirty):
        self.DicoDirty=DicoDirty
        self._Dirty=self.DicoDirty["ImageCube"]
        self._MeanDirty=self.DicoDirty["MeanImage"]
        NPSF=self.PSFServer.NPSF
        _,_,NDirty,_=self._Dirty.shape
        off=(NPSF-NDirty)/2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)
        self.ModelMachine.setModelShape(self._Dirty.shape)

    def AdaptArrayShape(self,A,Nout):
        nch,npol,Nin,_=A.shape
        if Nin==Nout: 
            return A
        elif Nin>Nout:
            dx=Nout/2
            B=np.zeros((nch,npol,Nout,Nout),A.dtype)
            print>>log,"  Adapt shapes: %s -> %s"%(str(A.shape),str(B.shape))
            B[:]=A[...,Nin/2-dx:Nin/2+dx+1,Nin/2-dx:Nin/2+dx+1]
            return B
        else:
            stop
            return None

    def updateModelMachine(self,ModelMachine):
        self.ModelMachine=ModelMachine
        if self.ModelMachine.RefFreq!=self.RefFreq:
            raise ValueError("freqs should be equal")

    def updateMask(self,Mask):
        nx,ny=Mask.shape
        self._MaskArray = np.zeros((1,1,nx,ny),np.bool8)
        self._MaskArray[0,0,:,:]=Mask[:,:]

    def Deconvolve(self):

        nch,npol,_,_=self._MeanDirty.shape
        Model=np.zeros_like(self._MeanDirty)
        

        dirty=self._MeanDirty
        _,_,xp,yp=np.where(dirty==np.max(dirty))
        self.PSFServer.setLocation(xp,yp)
        _,psf=self.PSFServer.GivePSF()
        
        Nout=np.min([dirty.shape[-1],psf.shape[-1]])
        psf=self.AdaptArrayShape(psf,Nout)
        dirty=self.AdaptArrayShape(dirty,Nout)
        
        Slice=slice(0,None)
        if dirty.shape[-1]%2!=0:
            Slice=slice(0,-1)
            


        # for ch in range(nch):
        #     CM=ClassMoresaneSingleSlice(dirty[ch,0,Slice,Slice],psf[ch,0,Slice,Slice],mask=None,GD=None)
        #     model,resid=CM.giveModelResid(major_loop_miter=self.GD["MORESANE"]["NMajorIter"],
        #                                  minor_loop_miter=self.GD["MORESANE"]["NMinorIter"],
        #                                  loop_gain=self.GD["MORESANE"]["Gain"],
        #                                  enforce_positivity=self.GD["MORESANE"]["ForcePositive"])
        #     model,resid=CM.giveModelResid()
        #     Model[ch,0,Slice,Slice]=model[:,:]

        CM=ClassMoresaneSingleSlice(dirty[0,0,Slice,Slice],psf[0,0,Slice,Slice],mask=None,GD=None)
        model,resid=CM.giveModelResid(major_loop_miter=self.GD["MORESANE"]["NMajorIter"],
                                      minor_loop_miter=self.GD["MORESANE"]["NMinorIter"],
                                      loop_gain=self.GD["MORESANE"]["Gain"],
                                      enforce_positivity=self.GD["MORESANE"]["ForcePositive"])

        print "!!!!!!!!!!!!!!!!!!!!!!!!!!",np.max(resid)

        # model,resid=CM.giveModelResid(major_loop_miter=100,
        #                               minor_loop_miter=100,
        #                               loop_gain=0.1,
        #                               enforce_positivity=0)

        Model[0,0,Slice,Slice]=model[:,:]
        
        self.ModelMachine.setModel(Model,0)
        #Alpha=np.ones_like(Model)
        


        return "MaxIter", True, True   # stop deconvolution but do update model
