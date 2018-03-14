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
#from DDFacet.Imager.MORESANE.ClassMoresaneSingleSlice import ClassMoresaneSingleSlice
##from DDFacet.Imager.MUFFIN.easy_muffin.easy_muffin_py.deconv3d import EasyMuffin
from easy_muffin_py.deconv3d import EasyMuffin
from DDFacet.Array import shared_dict
from DDFacet.ToolsDir import ClassSpectralFunctions
from scipy.optimize import least_squares
from DDFacet.ToolsDir.GiveEdges import GiveEdges


class ClassImageDeconvMachine():
    def __init__(self,GD=None,ModelMachine=None,RefFreq=None,*args,**kw):
        self.GD=GD
        self.ModelMachine = ModelMachine
        self.RefFreq=RefFreq
        if self.ModelMachine.DicoModel["Type"]!="MUFFIN":
            raise ValueError("ModelMachine Type should be MUFFIN")
        self.MultiFreqMode=(self.GD["Freq"]["NBand"]>1)

    def SetPSF(self,DicoVariablePSF):
        self.PSFServer=ClassPSFServer(self.GD)
        DicoVariablePSF=shared_dict.attach(DicoVariablePSF.path)#["CubeVariablePSF"]
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF)
        self.PSFServer.setRefFreq(self.ModelMachine.RefFreq)
        self.DicoVariablePSF=DicoVariablePSF
        self.setFreqs(self.PSFServer.DicoMappingDesc)

    def setMaskMachine(self,MaskMachine):
        self.MaskMachine=MaskMachine

    def setFreqs(self,DicoMappingDesc):
        self.DicoMappingDesc=DicoMappingDesc
        if self.DicoMappingDesc is None: return
        self.SpectralFunctionsMachine=ClassSpectralFunctions.ClassSpectralFunctions(self.DicoMappingDesc,RefFreq=self.DicoMappingDesc["RefFreq"])#,BeamEnable=False)
        self.SpectralFunctionsMachine.CalcFluxBands()

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
        if "PSFSideLobes" not in self.DicoVariablePSF.keys():
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
            # dx=Nout/2
            # B=np.zeros((nch,npol,Nout,Nout),A.dtype)
            # print>>log,"  Adapt shapes: %s -> %s"%(str(A.shape),str(B.shape))
            # B[:]=A[...,Nin/2-dx:Nin/2+dx+1,Nin/2-dx:Nin/2+dx+1]

            N0=A.shape[-1]
            xc0=yc0=N0/2
            N1=Nout
            xc1=yc1=N1/2
            Aedge,Bedge=GiveEdges((xc0,yc0),N0,(xc1,yc1),N1)
            x0d,x1d,y0d,y1d=Aedge
            x0p,x1p,y0p,y1p=Bedge
            B=A[...,x0d:x1d,y0d:y1d]

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

    	if self._Dirty.shape[-1]!=self._Dirty.shape[-2]:
            # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # print self._Dirty.shape
            # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            return "MaxIter", True, True

        dirty=self._Dirty
        nch,npol,nx,ny=dirty.shape
        Model=np.zeros_like(dirty)

        _,_,xp,yp=np.where(self._MeanDirty==np.max(self._MeanDirty))
        self.PSFServer.setLocation(xp,yp)
        self.iFacet=self.PSFServer.iFacet
        psf,_=self.PSFServer.GivePSF()
        nxPSF=psf.shape[-1]
        nxDirty=dirty.shape[-1]

        Nout=np.min([dirty.shape[-1],psf.shape[-1]])
        dirty=self.AdaptArrayShape(dirty,Nout)
        SliceDirty=slice(0,None)
        if dirty.shape[-1]%2!=0:
            SliceDirty=slice(0,-1)

        d=dirty[:,:,SliceDirty,SliceDirty]
        psf=self.AdaptArrayShape(psf,d.shape[-1])

        SlicePSF=slice(0,None)
        if psf.shape[-1]%2!=0:
            SlicePSF=slice(0,-1)

        p=psf[:,:,SlicePSF,SlicePSF]

        dirty_MUFFIN = np.squeeze(d[:,0,:,:])
        dirty_MUFFIN = dirty_MUFFIN.transpose((2,1,0))

        psf_MUFFIN = np.squeeze(p[:,0,:,:])
        psf_MUFFIN = psf_MUFFIN.transpose((2,1,0))

        EM = EasyMuffin(mu_s=self.GD['MUFFIN']['mu_s'],
                        mu_l=self.GD['MUFFIN']['mu_l'],
                        nb=self.GD['MUFFIN']['nb'],
                        truesky=dirty_MUFFIN,
                        psf=psf_MUFFIN,
                        dirty=dirty_MUFFIN)
        EM.loop(nitermax=self.GD['MUFFIN']['NMinorIter'])


        nxModel=dirty_MUFFIN.shape[0]
        Aedge,Bedge=GiveEdges((nxModel/2,nxModel/2),nxModel,(nxDirty/2,nxDirty/2),nxDirty)
        x0,x1,y0,y1=Bedge

        Model=np.zeros((nxDirty,nxDirty,nch))
        Model[x0:x1,y0:y1,:]=EM.x
        self.ModelMachine.setMUFFINModel(Model)

        # if self._Dirty.shape[-1]!=self._Dirty.shape[-2]:
        #     # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        #     # print self._Dirty.shape
        #     # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        #     return "MaxIter", True, True


        # dirty=self._Dirty
        # nch,npol,_,_=dirty.shape
        # Model=np.zeros_like(dirty)

        # _,_,xp,yp=np.where(self._MeanDirty==np.max(self._MeanDirty))
        # self.PSFServer.setLocation(xp,yp)
        # self.iFacet=self.PSFServer.iFacet

        # psf,_=self.PSFServer.GivePSF()
        
        # Nout=np.min([dirty.shape[-1],psf.shape[-1]])
        # dirty=self.AdaptArrayShape(dirty,Nout)
        # SliceDirty=slice(0,None)
        # if dirty.shape[-1]%2!=0:
        #     SliceDirty=slice(0,-1)

        # d=dirty[:,:,SliceDirty,SliceDirty]
        # psf=self.AdaptArrayShape(psf,d.shape[-1]*2)

        # SlicePSF=slice(0,None)
        # if psf.shape[-1]%2!=0:
        #     SlicePSF=slice(0,-1)

        # p=psf[:,:,SlicePSF,SlicePSF]
        # if p.shape[-1]!=2*d.shape[-1]:
        #     print "!!!!!!!!!!!!!!!!!!!!!!!!!"
        #     print "Could not adapt psf shape to 2*dirty shape!!!!!!!!!!!!!!!!!!!!!!!!!"
        #     print p.shape[-1],d.shape[-1]
        #     print "!!!!!!!!!!!!!!!!!!!!!!!!!"
        #     psf=self.AdaptArrayShape(psf,d.shape[-1])
        #     SlicePSF=SliceDirty

        # for ch in range(nch):
        #     CM=ClassMoresaneSingleSlice(dirty[ch,0,SliceDirty,SliceDirty],psf[ch,0,SlicePSF,SlicePSF],mask=None,GD=None)
        #     model,resid=CM.giveModelResid(major_loop_miter=self.GD["MORESANE"]["NMajorIter"],
        #                                   minor_loop_miter=self.GD["MORESANE"]["NMinorIter"],
        #                                   loop_gain=self.GD["MORESANE"]["Gain"],
        #                                   sigma_level=self.GD["MORESANE"]["SigmaCutLevel"],# tolerance=1.,
        #                                   enforce_positivity=self.GD["MORESANE"]["ForcePositive"])
        #     Model[ch,0,SliceDirty,SliceDirty]=model[:,:]
        
        #     import pylab
        #     pylab.clf()
        #     pylab.subplot(2,2,1)
        #     pylab.imshow(dirty[ch,0,SliceDirty,SliceDirty],interpolation="nearest")
        #     pylab.colorbar()

        #     pylab.subplot(2,2,2)
        #     pylab.imshow(psf[ch,0,SlicePSF,SlicePSF],interpolation="nearest")
        #     pylab.colorbar()

        #     pylab.subplot(2,2,3)
        #     pylab.imshow(model,interpolation="nearest")
        #     pylab.colorbar()

        #     pylab.subplot(2,2,4)
        #     pylab.imshow(resid,interpolation="nearest")
        #     pylab.colorbar()

        #     pylab.draw()
        #     pylab.show()


        # print 
        # print np.max(np.max(Model,axis=-1),axis=-1)
        # print 
        # print 




        #_,_,nx,ny=Model.shape
        #Model=np.mean(Model,axis=0).reshape((1,1,nx,ny))

        #Model.fill(0)
        #Model[:,:,xp,yp]=self._Dirty[:,:,xp,yp]

        
        return "MaxIter", True, True   # stop deconvolution but do update model
