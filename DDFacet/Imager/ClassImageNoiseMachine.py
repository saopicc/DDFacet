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

from DDFacet.ToolsDir import ModFFTW

class ClassImageNoiseMachine():
    def __init__(self, GD, ExternalModelMachine=None, DegridFreqs=None, GridFreqs=None, MainCache=None):
        self.GD = copy.deepcopy(GD)
        
        self.MainCache=MainCache
        self.NoiseMap=None
        self.NoiseMapRestored=None
        self.NoiseMapReShape=None
        self._id_InputMap=None
        self.ExternalModelMachine=ExternalModelMachine
        self.DegridFreqs = DegridFreqs
        self.GridFreqs = GridFreqs
        self.NFreqBands = len(GridFreqs)

        # MyLogger.setSilent(ListSilentModules)
        self.RefFreq = ExternalModelMachine.RefFreq
        # self.GD["Parallel"]["NCPU"]=1
        # self.GD["HMP"]["Alpha"]=[0,0,1]#-1.,1.,5]
        self.GD["HMP"]["Alpha"] = [0, 0, 1]
        # self.GD["Deconv"]["Mode"]="HMP"
        # self.GD["Deconv"]["CycleFactor"]=0
        # self.GD["Deconv"]["PeakFactor"]=0.0
        self.GD["Deconv"]["PSFBox"] = "full"
        self.GD["Deconv"]["MaxMinorIter"] = 10000
        self.GD["Deconv"]["RMSFactor"] = 3.
        # self.GD["HMP"]["Scales"]=[0]
        self.GD["HMP"]["Ratios"] = []
        # self.GD["MultiScale"]["Ratios"]=[]
        self.GD["HMP"]["NTheta"] = 4

        # self.GD["Deconv"]["AllowNegative"]=False
        # self.GD["HMP"]["Scales"]=[0,1,2,4,8,16,32,48,64,96,128]
        # self.GD["HMP"]["SolverMode"]="NNLS"
        # self.GD["HMP"]["Support"]=91
        # self.GD["HMP"]["Taper"]=31
        # self.GD["Deconv"]["Gain"]=.3

        self.GD["HMP"]["SolverMode"] = "PI"
        self.GD["HMP"]["Scales"] = [0]
        self.GD["Deconv"]["Gain"] = .1

        if self.NoiseMapReShape is not None:
            print>> log, "Deconvolving on SNR map"
            self.GD["Deconv"]["RMSFactor"] = 0.

        self.GD["HMP"]["AllowResidIncrease"] = 0.1
        # self.GD["HMP"]["SolverMode"]="PI"
        MinorCycleConfig = dict(self.GD["Deconv"])
        MinorCycleConfig["NCPU"] = self.GD["Parallel"]["NCPU"]
        MinorCycleConfig["NFreqBands"] = self.NFreqBands
        MinorCycleConfig["RefFreq"] = self.RefFreq
        MinorCycleConfig["GD"] = self.GD
        # MinorCycleConfig["RefFreq"] = self.RefFreq
        # MinorCycleConfig["CleanMaskImage"]=None
        self.MinorCycleConfig = MinorCycleConfig
        if self.GD["Deconv"]["Mode"] in ["HMP", "SSD"]:
            # for SSD we need to set up the HMP ModelMachine.
            self.GD["Deconv"]["Mode"] = "HMP"
            ModConstructor = ClassModModelMachine(self.GD)
            self.ModelMachine = ModConstructor.GiveMM(Mode=self.GD["Deconv"]["Mode"])
            self.ModelMachine.setRefFreq(self.RefFreq)
            MinorCycleConfig["ModelMachine"] = self.ModelMachine
            self.MinorCycleConfig = MinorCycleConfig
            from DDFacet.Imager.MSMF import ClassImageDeconvMachineMSMF

            self.DeconvMachine = ClassImageDeconvMachineMSMF.ClassImageDeconvMachine(MainCache=self.MainCache,
                                                                                     ParallelMode=True,
                                                                                     CacheFileName="HMP_Masking",
                                                                                     **self.MinorCycleConfig)
        elif self.GD["Deconv"]["Mode"] == "Hogbom":
            from DDFacet.Imager.HOGBOM import ClassImageDeconvMachineHogbom
            self.DeconvMachine = ClassImageDeconvMachineHogbom.ClassImageDeconvMachine(MainCache=self.MainCache,
                                                                                       ParallelMode=True,
                                                                                       CacheFileName="HMP_Masking",
                                                                                       **self.MinorCycleConfig)
        else:
            raise NotImplementedError("Mode %s not compatible with automasking" % self.GD["Deconv"]["Mode"])


    def giveMinStatNoiseMap(self,Image):
        Box,Step=self.GD["Noise"]["MinStats"]
        box=(Box,Box)
        print>>log, "  Computing noise map..."
        Boost=Step
        Acopy=Image[0,0,0::Boost,0::Boost].copy()
        SBox=(box[0]/Boost,box[1]/Boost)

        x=np.linspace(-10,10,1000)
        f=0.5*(1.+scipy.special.erf(x/np.sqrt(2.)))
        n=SBox[0]*SBox[1]
        F=1.-(1.-f)**n
        ratio=np.abs(np.interp(0.5,F,x))

        Noise=-scipy.ndimage.filters.minimum_filter(Acopy,SBox)/ratio

        NPixStats=10000
        IndStats=np.int64(np.linspace(0,Noise.size-1,NPixStats))
        NoiseMed=np.std(Noise.ravel()[IndStats])
        #NoiseMed=np.median(Noise)
        Noise[Noise<NoiseMed]=NoiseMed

        LargeNoise=np.zeros_like(Image[0,0])
        for i in range(Boost):
            for j in range(Boost):
                s00,s01=Noise.shape
                s10,s11=LargeNoise[i::Boost,j::Boost].shape
                s0,s1=min(s00,s10),min(s10,s11)
                LargeNoise[i::Boost,j::Boost][0:s0,0:s1]=Noise[:,:][0:s0,0:s1]
        ind=np.where(LargeNoise==0.)
        LargeNoise[ind]=NoiseMed


        _,_,nx,ny=Image.shape
        self.NoiseMap=LargeNoise
        self.NoiseMapReShape=self.NoiseMap.reshape((1,1,nx,ny))
        return self.NoiseMapReShape

    def calcNoiseMap(self,DicoResidual):
        if self._id_InputMap==id(DicoResidual["MeanImage"]):
            print>>log,"Noise map has already been computed for that image"
            return self.NoiseMapReShape
        else:
            print>>log,"(re-)Computing noise map"
            self._id_InputMap=id(DicoResidual["MeanImage"])

        # self.NoiseMapReShape=self.giveMinStatNoiseMap(DicoResidual["MeanImage"])

        if self.GD["Noise"]["BrutalHMP"]:
            self.giveBrutalRestored(DicoResidual)
            if self.GD["Mask"]["FluxImageType"]=="ModelConv":
                self.FluxImage=self.ModelConv
            elif self.GD["Mask"]["FluxImageType"]=="Restored":
                self.FluxImage=self.Restored
            self.StatImage=self.Restored
        else:
            self.StatImage=DicoResidual["MeanImage"]
            self.FluxImage=DicoResidual["MeanImage"]
        self.NoiseMapReShape=self.giveMinStatNoiseMap(self.StatImage)
        return self.NoiseMapReShape


    def setPSF(self,DicoVariablePSF):
        self.DicoVariablePSF=DicoVariablePSF

    def giveBrutalRestored(self,DicoResidual):
        print>>log,"  Running Brutal deconvolution..."
        ListSilentModules=["ClassImageDeconvMachineMSMF","ClassPSFServer","ClassMultiScaleMachine","GiveModelMachine",
                           "ClassModelMachineMSMF", "ClassImageDeconvMachineHogbom", "ClassModelMachineHogbom"]
        # MyLogger.setSilent(ListSilentModules)
        self.DicoDirty=DicoResidual
        self.Orig_MeanDirty=self.DicoDirty["MeanImage"].copy()
        self.Orig_Dirty=self.DicoDirty["ImageCube"].copy()

        if self.NoiseMapReShape is not None:
            print>>log,"Deconvolving on SNR map"
            self.DeconvMachine.RMSFactor = 0
            
        self.DeconvMachine.Init(PSFVar=self.DicoVariablePSF,PSFAve=self.DicoVariablePSF["EstimatesAvgPSF"][-1],
                                GridFreqs=self.GridFreqs, DegridFreqs=self.DegridFreqs, RefFreq=self.RefFreq)

        if self.NoiseMapReShape is not None:
            self.DeconvMachine.setNoiseMap(self.NoiseMapReShape,PNRStop=self.GD["Mask"]["SigTh"])

        # # #########################
        # # debug
        # MaskImage=image("image_dirin_SSD_test.dirty.fits.mask.fits").getdata()
        # nch,npol,_,_=MaskImage.shape
        # MaskArray=np.zeros(MaskImage.shape,np.bool8)
        # for ch in range(nch):
        #     for pol in range(npol):
        #         MaskArray[ch,pol,:,:]=np.bool8(MaskImage[ch,pol].T[::-1].copy())[:,:]
        # self.DeconvMachine.setMask(np.bool8(1-MaskArray))
        # # #########################

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
        Model=self.ModelMachine.GiveModelImage(DoAbs=True)
        if "Comp" in self.ExternalModelMachine.DicoSMStacked.keys():
            Model+=np.abs(self.ExternalModelMachine.GiveModelImage())
        ModelImage=Model[0,0]

        print>>log,"  Convolving image with beam %s..."%str(self.DicoVariablePSF["EstimatesAvgPSF"][1])
        #from DDFacet.ToolsDir import Gaussian
        

        # Sig_rad=np.max(self.DicoVariablePSF["EstimatesAvgPSF"][1][0:2])
        # Sig_pix=Sig_rad/self.DicoDirty["ImageInfo"]["CellSizeRad"]
        # Sig_pix=np.max([1,Sig_pix])#*2
        # n_pix=int(Sig_pix*4)
        # if n_pix%2==0: n_pix+=1

        # _,_,G=Gaussian.GaussianSymetric(Sig_pix,n_pix)


        # from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric

        # N1=G.shape[0]
        # N0x,N0y=ModelImage.shape
        # indx,indy=np.where(ModelImage!=0)
        # ModelConv=np.zeros_like(ModelImage)
        # for iComp in range(indx.size):
        #     xc,yc=indx[iComp],indy[iComp]
        #     Aedge,Bedge=GiveEdgesDissymetric((xc,yc),(N0x,N0y),(N1/2,N1/2),(N1,N1))
        #     x0d,x1d,y0d,y1d=Aedge
        #     x0p,x1p,y0p,y1p=Bedge
        #     ModelConv[x0d:x1d,y0d:y1d]+=G[x0p:x1p,y0p:y1p]*ModelImage[xc,yc]
            
        # # ModelConv=scipy.signal.convolve2d(ModelImage,G,mode="same")

        ModelConv=ModFFTW.ConvolveGaussian({0:Model},0,0,0, CellSizeRad=self.DicoDirty["ImageInfo"]["CellSizeRad"],
                                           GaussPars_ch=self.DicoVariablePSF["EstimatesAvgPSF"][1])

        #GaussPar=[i*5 for i in self.DicoVariablePSF["EstimatesAvgPSF"][1]]
        #ModelConv+=ModFFTW.ConvolveGaussian(Model, CellSizeRad=self.DicoDirty["ImageInfo"]["CellSizeRad"],
        #                                    GaussPars=[GaussPar])


        self.ModelConv=ModelConv.reshape(self.DicoDirty["MeanImage"].shape)



        self.Restored=self.ModelConv+self.DicoDirty["MeanImage"]

        self.DicoDirty["MeanImage"][...]=self.Orig_MeanDirty[...]
        self.DicoDirty["ImageCube"][...]=self.Orig_Dirty[...]


        self.DeconvMachine.Reset()
        #MyLogger.setLoud(ListSilentModules)
        return self.Restored
