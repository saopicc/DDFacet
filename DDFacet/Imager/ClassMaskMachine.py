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
import copy
from DDFacet.Imager.ModModelMachine import ClassModModelMachine
from DDFacet.Imager.MSMF import ClassImageDeconvMachineMSMF

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
        self.DoMask=(self.GD["Mask"]["Auto"] or self.GD["Mask"]["External"])

    def setMainCache(self,MainCache):
        self.MainCache=MainCache

    def giveNoiseMap(self,Image):
        print>>log, "  Computing noise map..."
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
        if not self.DoMask: return
        print>>log, "Computing Mask"
        self.DicoResidual=DicoResidual

        if self.GD["Mask"]["Auto"]:
            if self.GD["Mask"]["AutoBrutalHMP"]:
                self.doBrutalClean()
                Image=self.Restored
            else:
                Image=DicoResidual["MeanImage"]

            self.Box,self.Step,Th=self.GD["Mask"]["AutoStats"]
            self.box=(self.Box,self.Box)
            _,_,nx,ny=Image.shape
            #self.NoiseMap=self.giveNoiseMap(self.DicoResidual["MeanImage"])
            self.NoiseMap=self.giveNoiseMap(Image)
            self.NoiseMapReShape=self.NoiseMap.reshape((1,1,nx,ny))
            self.NoiseMask=(Image[0,0]>Th*self.NoiseMapReShape)
        self.updateMask()

    def readExternalMaskFromFits(self):
        CleanMaskImage=self.GD["Mask"]["External"]
        if not CleanMaskImage: return
        print>>log, "  Reading mask image: %s"%CleanMaskImage
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
        if self.NoiseMask is not None: 
            print>>log,"  Merging Current mask with Noise-based mask"
            self.CurrentMask = OR(self.CurrentMask,self.NoiseMask)
        if self.ExternalMask is not None: 
            print>>log,"  Merging Current mask with external mask"
            self.CurrentMask = OR(self.CurrentMask,self.ExternalMask)

        if self.CurrentMask is not None:
            self.CurrentNegMask=self.giveOpposite(self.CurrentMask)

    def AdaptMaskShape(self):
        _,_,NMask,_=self._MaskArray.shape
        if NMask!=NDirty:
            print>>log,"  Mask do not have the same shape as the residual image"
            self._MaskArray=self.AdaptArrayShape(self._MaskArray,NDirty)
            self.MaskArray=self._MaskArray[0]
            self.IslandArray=np.zeros_like(self._MaskArray)
            self.IslandHasBeenDone=np.zeros_like(self._MaskArray)


    def setPSF(self,DicoVariablePSF):
        self.DicoVariablePSF=DicoVariablePSF

    def doBrutalClean(self):
        print>>log,"  Running Brutal HMP..."
        ListSilentModules=["ClassImageDeconvMachineMSMF","ClassPSFServer","ClassMultiScaleMachine","GiveModelMachine","ClassModelMachineMSMF"]
        MyLogger.setSilent(ListSilentModules)
        self.DicoDirty=self.DicoResidual
        self.Orig_MeanDirty=self.DicoDirty["MeanImage"].copy()
        self.Orig_Dirty=self.DicoDirty["ImageCube"].copy()
        GD=copy.deepcopy(self.GD)
        # take any reference frequency - doesn't matter
        self.RefFreq=np.mean(self.DicoVariablePSF["freqs"][0])
        self.GD=GD
        #self.GD["Parallel"]["NCPU"]=1
        #self.GD["HMP"]["Alpha"]=[0,0,1]#-1.,1.,5]
        self.GD["HMP"]["Alpha"]=[0,0,1]
        self.GD["Deconv"]["Mode"]="HMP"
        self.GD["Deconv"]["CycleFactor"]=0
        self.GD["Deconv"]["PeakFactor"]=0.01
        self.GD["Deconv"]["RMSFactor"]=3.
        self.GD["Deconv"]["Gain"]=.5
        self.GD["Deconv"]["AllowNegative"]=False
        self.GD["Deconv"]["PSFBox"]="full"
        self.GD["Deconv"]["MaxMinorIter"]=1000
        self.GD["HMP"]["Scales"]=[0,1,2,4,8]
        self.GD["HMP"]["Ratios"]=[]
        #self.GD["MultiScale"]["Ratios"]=[]
        self.GD["HMP"]["NTheta"]=4
        
        #self.GD["HMP"]["AllowResidIncrease"]=False
        self.GD["HMP"]["SolverMode"]="NNLS"

        DicoVariablePSF=self.DicoVariablePSF
        self.NFreqBands=len(DicoVariablePSF["freqs"])
        MinorCycleConfig=dict(self.GD["Deconv"])
        MinorCycleConfig["NCPU"]=self.GD["Parallel"]["NCPU"]
        MinorCycleConfig["NFreqBands"]=self.NFreqBands
        MinorCycleConfig["GD"] = self.GD
        #MinorCycleConfig["RefFreq"] = self.RefFreq
        ModConstructor = ClassModModelMachine(self.GD)
        ModelMachine = ModConstructor.GiveMM(Mode=self.GD["Deconv"]["Mode"])
        ModelMachine.setRefFreq(self.RefFreq)
        MinorCycleConfig["ModelMachine"]=ModelMachine
        #MinorCycleConfig["CleanMaskImage"]=None
        self.MinorCycleConfig=MinorCycleConfig
        self.DeconvMachine=ClassImageDeconvMachineMSMF.ClassImageDeconvMachine(MainCache=self.MainCache,
                                                                               CacheSharedMode=True,
                                                                               ParallelMode=False,
                                                                               CacheFileName="HMP_Masking",
                                                                               **self.MinorCycleConfig)
        

        self.DeconvMachine.Init(PSFVar=self.DicoVariablePSF,PSFAve=self.DicoVariablePSF["EstimatesAvgPSF"][-1])
        self.DeconvMachine.Update(self.DicoDirty,DoSetMask=False)
        self.DeconvMachine.updateRMS()
        # ModConstructor = ClassModModelMachine(self.GD)
        # ModelMachine = ModConstructor.GiveMM(Mode=self.GD["Deconv"]["Mode"])
        # #print "ModelMachine"
        # #time.sleep(30)
        # self.ModelMachine=ModelMachine
        # #self.ModelMachine.DicoSMStacked=self.DicoBasicModelMachine
        # self.ModelMachine.setRefFreq(self.RefFreq,Force=True)
        # self.MinorCycleConfig["ModelMachine"] = ModelMachine
        # #self.ModelMachine.setModelShape(self.SubDirty.shape)
        # #self.ModelMachine.setListComponants(self.DeconvMachine.ModelMachine.ListScales)
        # #self.DeconvMachine.Update(self.DicoSubDirty,DoSetMask=False)
        # #self.DeconvMachine.updateMask(np.logical_not(self.SubMask))
        # self.DeconvMachine.updateModelMachine(ModelMachine)
        self.DeconvMachine.resetCounter()
        self.DeconvMachine.Deconvolve(UpdateRMS=False)

        print>>log,"  Getting model image..."
        ModelImage=ModelMachine.GiveModelImage()[0,0]

        print>>log,"  Convolving..."
        from DDFacet.ToolsDir import Gaussian
        

        Sig_rad=np.max(self.DicoVariablePSF["EstimatesAvgPSF"][1][0:2])
        Sig_pix=Sig_rad/self.DicoDirty["ImageInfo"]["CellSizeRad"]
        Sig_pix=int(np.max([1,Sig_pix]))
        if Sig_pix%2==0: Sig_pix+=1
        Extent_pix=Sig_pix*5
        if Extent_pix%2==0: Extent_pix+=1

        _,_,G=Gaussian.Gaussian(Sig_pix,Extent_pix,1)


        from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric

        N1=G.shape[0]
        N0x,N0y=ModelImage.shape
        indx,indy=np.where(ModelImage!=0)
        ModelConv=np.zeros_like(ModelImage)
        for iComp in range(indx.size):
            xc,yc=indx[iComp],indy[iComp]
            Aedge,Bedge=GiveEdgesDissymetric((xc,yc),(N0x,N0y),(N1/2,N1/2),(N1,N1))
            x0d,x1d,y0d,y1d=Aedge
            x0p,x1p,y0p,y1p=Bedge
            ModelConv[x0d:x1d,y0d:y1d]+=G[x0p:x1p,y0p:y1p]*ModelImage[xc,yc]

#        ModelConv=scipy.signal.convolve2d(ModelImage,G,mode="same")



        self.Restored=ModelConv.reshape(self.DicoDirty["MeanImage"].shape)+self.DicoDirty["MeanImage"]

        self.DicoDirty["MeanImage"][...]=self.Orig_MeanDirty[...]
        self.DicoDirty["ImageCube"][...]=self.Orig_Dirty[...]
        
        MyLogger.setLoud(ListSilentModules)
