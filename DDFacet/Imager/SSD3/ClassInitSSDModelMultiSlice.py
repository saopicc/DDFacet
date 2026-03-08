from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import numpy as np
from DDFacet.Imager.MultiSliceDeconv import ClassImageDeconvMachineMultiSlice
from DDFacet.Imager.MultiSliceDeconv.ClassImageDeconvMachineMultiSlice import pad_to_square,unpad_to_original
import copy
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Imager.ModModelMachine import ClassModModelMachine
#import multiprocessing
from DDFacet.Other import ClassTimeIt
from DDFacet.Other.progressbar import ProgressBar
import time
from DDFacet.Array import NpShared
from DDFacet.Other import logger
log=logger.getLogger("ClassInitSSDModelMultiSlice")
import traceback
from DDFacet.Other import ModColor
from .ClassConvMachine import ClassConvMachineImages
#from .ClassTaylorToPower import ClassTaylorToPower
from DDFacet.Array import shared_dict
import psutil
from DDFacet.Other.progressbar import ProgressBar
#from DDFacet.Other.AsyncProcessPool import APP

SilentModules=["ClassPSFServer",
               "ClassImageDeconvMachine",
               "GiveModelMachine",
               "ClassImageDeconvMachineMultiSlice",
               "ClassModelMachineMultiSlice",
               #"ClassTaylorToPower",
               "ClassModelMachineSSD"]

class ClassInitSSDModelParallel():
    def __init__(self, GD, NFreqBands, RefFreq, NCPU, MainCache=None,IdSharedMem="",APP=None):
        self.T=ClassTimeIt.ClassTimeIt("ClassInitSSDModelParallel")
        self.T.disable()
        self.GD = copy.deepcopy(GD)
        self.APP=APP
        from DDFacet.Imager.MultiFields.AppendSubFieldInfo import AppendSubFieldInfo
        AppendSubFieldInfo(self)
        
        self.GD["MultiSliceDeconv"]["PolyFitOrder"]=self.GD["SSD3"]["PolyFreqOrder"]
        self.MainCache=MainCache
        self.RefFreq=RefFreq
        self.NCPU = NCPU
        self.IdSharedMem=IdSharedMem
        self.NFreqBands=NFreqBands
        self.Type="MultiSlice"
        self.T.timeit("Init0")
        self.InitMachine = ClassInitSSDModel(self.GD, NFreqBands, RefFreq, MainCache, IdSharedMem,APP=self.APP)
        self.NCPU=(self.GD["Parallel"]["NCPU"] or psutil.cpu_count())
        self.T.timeit("Init1")
        #APP.registerJobHandlers(self)
        self.T.timeit("Init2")

        

    def Init(self, DicoVariablePSF, GridFreqs, DegridFreqs):
        self.T.reinit()
        self.DicoVariablePSF=DicoVariablePSF
        self.GridFreqs=GridFreqs
        self.DegridFreqs=DegridFreqs
        #print("Initialise MultiSlice machine", file=log)
        self.InitMachine.DeconvMachine.SetPSF(self.DicoVariablePSF)
        # self.InitMachine=ClassInitSSDModel(self.GD, self.NFreqBands, self.RefFreq, MainCache=self.MainCache, IdSharedMem=self.IdSharedMem,
        #                                    APP=self.APP)
        self.T.timeit("Init_Init")

    def Reset(self):
        self.DicoVariablePSF = None
        #self.InitMachine.Reset()


    def _initIsland_worker(self, iIsland, Island,
                           DicoVariablePSF, DicoDirty, DicoParm, NCPU, ThSpectralFit):
        self.T.reinit()
        logger.setSilent(["ClassImageDeconvMachineMSMF", "ClassPSFServer", "ClassMultiScaleMachine", "GiveModelMachine", "ClassModelMachineMSMF"])
        self.InitMachine.Init(DicoVariablePSF, DicoParm["GridFreqs"], DicoParm["DegridFreqs"])
        self.T.timeit("_initIsland_worker:Init")
        self.InitMachine.setDirty(DicoDirty)
        self.T.timeit("_initIsland_worker:setDirty")
        # self.InitMachine.DeconvMachine.setNCPU(NCPU)
        #self.InitMachine.setSSDModelImage(DicoParm["ModelImageInt"])
        self.InitMachine.setSSDModelImage(DicoParm["ModelImageApp"])
        self.T.timeit("_initIsland_worker:setSSD")


        #print ":::::::::::::::::::::::",iIsland

        
        ModelImageIsland,NSpectralFit = self.InitMachine.giveModel(Island,ThSpectralFit=ThSpectralFit,iIsland=iIsland)
        # print("OKKKKK")
        # ######################
        # try:
        #     ModelImageIsland = self.InitMachine.giveModel(Island)
        # except Exception as e:
        #     FileOut = "errIsland_%6.6i.npy" % iIsland
        #     print(ModColor.Str("...... error on island %i: %s"%(iIsland,str(e))))
        #     print(ModColor.Str("       saving to file %s" % (FileOut)))
        #     np.save(FileOut, np.array(Island))
        #     self.InitMachine.Reset()
        #     return
        # ######################
        
        self.T.timeit("_initIsland_worker:giveModel")
        #DicoOut["PolyModel"] = ModelImageIsland
        return ModelImageIsland,NSpectralFit
        # self.InitMachine.Reset()

    def giveDicoInitIndiv(self, Island=None,ListIslands=None, iIsland=None,
                          #ModelImage=None,
                          DicoDirty=None, ThSpectralFit=True):
        #DicoInitIndiv = shared_dict.attach("DicoInitIslandMultiSlice%s"%self.StrField)
        ParmDict = shared_dict.attach("ParmDict%s"%self.StrField)
        if Island is None:
            Island=ListIslands[iIsland]
            
        #subdict = DicoInitIndiv.addSubdict(iIsland)

        # NameIsland="errIsland_003718.npy"
        # iIsland=int(NameIsland.split("_")[1].split(".")[0])
        # Island=np.load(NameIsland)
        
        PolyModel,NSpectralFit = self._initIsland_worker(iIsland, 
                                                         Island,
                                                         self.DicoVariablePSF, 
                                                         DicoDirty,
                                                         ParmDict, 
                                                         1,
                                                         ThSpectralFit)
        # DicoInitIndiv.reload()
        
    
        # ParmDict.delete()

        return PolyModel,NSpectralFit



######################################################################################################

class ClassInitSSDModel():
    def __init__(self, GD, NFreqBands, RefFreq, MainCache=None, IdSharedMem="",APP=None):
        GD=copy.deepcopy(GD)
        self.APP=APP
        self.T=ClassTimeIt.ClassTimeIt("ClassInitSSDModel_Single")
        self.T.disable()
        self.RefFreq=RefFreq
        self.GD=GD
        self.GD["Parallel"]["NCPU"]=1
        self.GD["HMP"]["Alpha"]=[-1.,1.,5]

        self.GD["Deconv"]["Mode"]="MultiSlice"
        
        self.GD["Deconv"]["CycleFactor"]=0
        self.GD["Deconv"]["PeakFactor"]=0.0
        self.GD["Deconv"]["RMSFactor"]=self.GD["GAClean"]["RMSFactorInitHMP"]

        self.GD["Deconv"]["Gain"]=1.
        self.GD["Deconv"]["AllowNegative"]=self.GD["GAClean"]["AllowNegativeInitHMP"]
        self.GD["Deconv"]["MaxMinorIter"]=self.GD["GAClean"]["MaxMinorIterInitHMP"]
        
        # self.CTP=ClassTaylorToPower(self.GD["MultiSliceDeconv"]["PolyFitOrder"])
        # self.CTP.ComputeConvertionFunctions()
        
        logger.setSilent(SilentModules)

        self.NFreqBands=NFreqBands
        MinorCycleConfig=dict(self.GD["Deconv"])

        ModConstructor = ClassModModelMachine(self.GD)
        ModelMachine = ModConstructor.GiveMM(Mode="MultiSlice")
        ModelMachine.setRefFreq(self.RefFreq)
        MinorCycleConfig["ModelMachine"]=ModelMachine
        #MinorCycleConfig["CleanMaskImage"]=None
        self.MinorCycleConfig=MinorCycleConfig
        self.T.timeit("Init0")
        self.DeconvMachine=ClassImageDeconvMachineMultiSlice.ClassImageDeconvMachine(MainCache=MainCache,
                                                                                     ParallelMode=False,
                                                                                     RefFreq=self.RefFreq,
                                                                                     CacheFileName="MultiSlice_Init",
                                                                                     IdSharedMem=IdSharedMem,
                                                                                     GD=self.GD,
                                                                                     APP=self.APP,
                                                                                     **self.MinorCycleConfig)
        self.GD["Mask"]["Auto"]=False
        self.GD["Mask"]["External"]=None
        self.T.timeit("Init1")

        #self.Margin=100

    def Reset(self):
        pass
    
    def Init(self,DicoVariablePSF,GridFreqs,DegridFreqs,
                 DoWait=False):
        self.T.reinit()
        self.DicoVariablePSF=DicoVariablePSF
        self.GridFreqs=GridFreqs
        self.DegridFreqs=DegridFreqs


        #self.Margin=20
        
        #print("self.DeconvMachine",id(self.DeconvMachine))
        
        self.DeconvMachine.Init(PSFVar=self.DicoVariablePSF,PSFAve=self.DicoVariablePSF["PSFSideLobes"],
                                GridFreqs=self.GridFreqs,DegridFreqs=self.DegridFreqs,DoWait=DoWait)
        self.T.timeit("_Init")
        

    def setDirty(self, DicoDirty):
        self.T.reinit()
        self.DicoDirty=DicoDirty
        self.Dirty=DicoDirty["ImageCube"]
        self.MeanDirty=DicoDirty["MeanImage"]
        self.DeconvMachine.Update(self.DicoDirty,DoSetMask=False)
        self.T.timeit("setDirty")
        
        #self.DeconvMachine.updateRMS()

    def setSubDirty(self,ListPixParms):
        T=ClassTimeIt.ClassTimeIt("InitSSD.setSubDirty")
        T.disable()

        x,y=np.array(ListPixParms).T
        x0,x1=x.min(),x.max()+1
        y0,y1=y.min(),y.max()+1



        Margin_x=np.min([(x1-x0),200])
        Margin_y=np.min([(y1-y0),200])
        Margin_x=np.max([Margin_x,30])
        Margin_y=np.max([Margin_y,30])


        dx=(x1-x0)+2*Margin_x
        dy=(y1-y0)+2*Margin_y
        
        #dx=3*(x1-x0)
        #dy=3*(y1-y0)
        
        Size=np.max([dx,dy])
        if Size%2==0: Size+=1
        _,_,N0x,N0y=self.Dirty.shape

        xc0,yc0=int((x1+x0)/2.),int((y1+y0)/2.)
        self.xy0=xc0,yc0

        self.DeconvMachine.PSFServer.setLocation(*self.xy0)
        
        
        N1=Size
        xc1=yc1=N1//2
        Aedge,Bedge=GiveEdgesDissymetric(xc0,yc0,N0x,N0y,xc1,yc1,N1,N1)
        x0d,x1d,y0d,y1d=Aedge
        #x0p,x1p,y0p,y1p=Bedge
        self.SubDirty=self.Dirty[:,:,x0d:x1d,y0d:y1d].copy()
        #print(self.SubDirty.shape)
        #print(self.SubDirty.shape)
        T.timeit("0")
        self.blc=(x0d,y0d)
        self.DeconvMachine.PSFServer.setBLC(self.blc)
        _,_,nx,ny=self.SubDirty.shape
        ArrayPixParms=np.array(ListPixParms)
        ArrayPixParms[:,0]-=x0d
        ArrayPixParms[:,1]-=y0d
        self.ArrayPixParms=ArrayPixParms
        self.DicoSubDirty={}
        for key in self.DicoDirty.keys():
            if key in ["ImageCube", "MeanImage",'FacetNorm',"JonesNorm"]:
                self.DicoSubDirty[key]=self.DicoDirty[key][...,x0d:x1d,y0d:y1d].copy()
            else:
                self.DicoSubDirty[key]=self.DicoDirty[key]

        T.timeit("1")
        # ModelImage=np.zeros_like(self.Dirty)
        # ModelImage[:,:,N0//2,N0//2]=10
        # ModelImage[:,:,N0//2+3,N0//2]=10
        # ModelImage[:,:,N0//2-2,N0//2-1]=10
        # self.setSSDModelImage(ModelImage)

        # Mask=np.zeros((nx,ny),np.bool_)
        # Mask[x,y]=1
        # self.SubMask=Mask


        x,y=ArrayPixParms.T
        Mask=np.zeros(self.DicoSubDirty["ImageCube"].shape[-2::],np.bool_)
        Mask[x,y]=1
        self.SubMask=Mask

        self.DeconvMachine.setXY(*self.xy0)
        if self.SSDModelImage is not None:
            self.SubSSDModelImage=self.SSDModelImage[:,:,x0d:x1d,y0d:y1d].copy()
            for ch in range(self.NFreqBands):
                self.SubSSDModelImage[ch,0][np.logical_not(self.SubMask)]=0
            self.addSubModelToSubDirty()
        T.timeit("2")


    def setSSDModelImage(self,ModelImage):
        self.SSDModelImage=ModelImage

    def giveConvModel(self,SubModelImage):
        T=ClassTimeIt.ClassTimeIt("InitSSD.giveConvModel")
        T.disable()
        # Here PSFServer is in **not** peak-normalised mode
        # so we need to get the peak-normalised psf
        # SubModelImage is apparant
        iFacet=self.DeconvMachine.PSFServer.iFacet
        PSF=self.DeconvMachine.PSFServer.DicoVariablePSF["PeakNormed_CubeVariablePSF"][iFacet]
        T.timeit("GivePSF")
        
        # if self.GD["MultiSliceDeconv"]["Type"]=="Orieux":
        #     # Orieux uses a PSF of the same size as the Dirty, so need to pre-convolve with that one other bias appears 
        #     _,_,nx,ny=SubModelImage.shape
        #     s_psf=ClassImageDeconvMachineMultiSlice.giveSliceCut(PSF,nx)
        #     PSF=PSF[:,:,s_psf,s_psf]
            
        ConvModel=ClassConvMachineImages(PSF).giveConvModel(SubModelImage)
        T.timeit("ConvModel")

        # ConvModel=np.zeros_like(SubModelImage)
        # nch,_,N0x,N0y=ConvModel.shape
        # indx,indy=np.where(SubModelImage[0,0]!=0)
        # xc,yc=N0x//2,N0y//2
        # N1=PSF.shape[-1]
        # #T.timeit("0")
        # for i,j in zip(indx.tolist(),indy.tolist()):
        #     ThisPSF=np.roll(np.roll(PSF,i-xc,axis=-2),j-yc,axis=-1)
        #     Aedge,Bedge=GiveEdgesDissymetric((xc,yc),(N0x,N0y),(N1//2,N1//2),(N1,N1))
        #     x0d,x1d,y0d,y1d=Aedge
        #     x0p,x1p,y0p,y1p=Bedge
        #     ConvModel[...,x0d:x1d,y0d:y1d]+=ThisPSF[...,x0p:x1p,y0p:y1p]*SubModelImage[...,i,j].reshape((-1,1,1,1))
        # #T.timeit("1 %s"%(str(ConvModel.shape)))

        return ConvModel
    
    

    def addSubModelToSubDirty(self):
        T=ClassTimeIt.ClassTimeIt("InitSSD.addSubModelToSubDirty")
        T.disable()

        self.IsPadded=False
        

        #if key in ["ImageCube", "MeanImage",'FacetNorm',"JonesNorm"]:            
        DirtyCube=self.DicoSubDirty["ImageCube"]
        nch,npol,_,_=DirtyCube.shape
        if DirtyCube.shape[-1]!=DirtyCube.shape[-2]:
            nch,npol,_,_=DirtyCube.shape
            original_shape = DirtyCube[0,0].shape
            Ldirty=[]
            for ich in range(nch):
                for ipol in range(npol):
                    dirty,blc_trc=pad_to_square(DirtyCube[ich,ipol])
                    Ldirty.append(dirty)
            self.DicoSubDirty["blc_trc"]=blc_trc
            self.DicoSubDirty["original_shape"]=original_shape
            #pad_to_square(DirtyCube[ich,ipol])
            #self.DicoSubDirty["MeanImage"]=self.DicoSubDirty["MeanImage"][:,:,original_shape
            nx,ny=dirty.shape
            dirty=np.array(Ldirty).reshape((nch,npol,nx,ny))
            self.IsPadded=True
            
        
        ConvModel=self.giveConvModel(self.SubSSDModelImage)
        _,_,N0x,N0y=ConvModel.shape
        MeanConvModel=np.mean(ConvModel,axis=0).reshape((1,1,N0x,N0y))

        
        self.DicoSubDirty["ImageCube"]+=ConvModel
        self.DicoSubDirty['MeanImage']+=MeanConvModel
        
        # # ##########################
        if MeanConvModel.max()>0:
            for ich in range(nch):
                Im=self.DicoSubDirty["ImageCube"][ich,0].copy()
                Im[np.logical_not(self.SubMask)]=0
                indx,indy=np.where(Im==Im.max())
                indx=indx[0]
                indy=indy[0]
                M=ConvModel[ich,0,indx,indy]
                R=self.DicoSubDirty["ImageCube"][ich,0,indx,indy]-M
                Alpha=1-R/M
                Min,Max=0.3,3.0
                Alpha=np.max([Min,Alpha])
                Alpha=np.min([Max,Alpha])
                factScale=1./Alpha
                self.DicoSubDirty["ImageCube"][ich]*=factScale
                #print("DSFLMKFSDLKSDF ALPHA [%i,%i] %f"%(self.iIsland,ich,factScale))
                #print("DSFLMKFSDLKSDF ALPHA [%i,%i] %f"%(self.iIsland,ich,factScale))
                #print("DSFLMKFSDLKSDF ALPHA [%i,%i] %f"%(self.iIsland,ich,factScale))

        # # ##########################
        
        

        
        #print "MAX=",np.max(self.DicoSubDirty['MeanImage'])
        T.timeit("2")

        # import pylab
        # pylab.clf()
        # ax=pylab.subplot(1,3,1)
        # pylab.imshow(self.SubSSDModelImage[0,0],interpolation="nearest")
        # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
        # pylab.imshow(PSF[0,0],interpolation="nearest")
        # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
        # pylab.imshow(ConvModel[0,0],interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

            
    def giveModel(self,ListPixParms,ThSpectralFit=True,iIsland=None):
        T=ClassTimeIt.ClassTimeIt("giveModel")
        T.disable()
        self.iIsland=iIsland
        self.setSubDirty(ListPixParms)
        T.timeit("setsub")
        ModConstructor = ClassModModelMachine(self.GD)
        ModelMachine = ModConstructor.GiveMM(Mode=self.GD["Deconv"]["Mode"])
        #print "ModelMachine"
        #time.sleep(30)
        T.timeit("giveMM")
        self.ModelMachine=ModelMachine
        #self.ModelMachine.DicoSMStacked=self.DicoBasicModelMachine
        self.ModelMachine.setRefFreq(self.RefFreq,Force=True)
        
        self.MinorCycleConfig["ModelMachine"] = self.ModelMachine
        self.ModelMachine.setModelShape(self.SubDirty.shape)
        #self.ModelMachine.setListComponants(self.DeconvMachine.ModelMachine.ListScales)
        T.timeit("setlistcomp")
        
        self.DeconvMachine.Update(self.DicoSubDirty,DoSetMask=False,iIsland=iIsland)
        self.DeconvMachine.updateMask(np.logical_not(self.SubMask))
        self.DeconvMachine.updateModelMachine(self.ModelMachine)
        #self.DeconvMachine.resetCounter()
        T.timeit("update")
        #print "update"
        #time.sleep(30)
        
        rep,_,_=self.DeconvMachine.Deconvolve(ThSpectralFit)
        # print("LSFFLJFL",rep)
        # print("LSFFLJFL",rep)
        # print("LSFFLJFL",rep)
        # print("LSFFLJFL",rep)
        if rep=="Edge" or rep=="Skip":
            return np.zeros((self.GD["SSD3"]["PolyFreqOrder"],len(ListPixParms)),np.float32), (0,0)
        
        T.timeit("deconv %s"%str(self.DicoSubDirty["ImageCube"].shape))
        # print "deconv"
        # time.sleep(30)

        ModelImage=self.ModelMachine.DicoModel["CoefImage"]#GiveModelImage()
        if self.ModelMachine.DicoModel["FluxScale"]=="Linear":
            ModelImage=self.CTP.LinPolyCube2LogPolyCube(ModelImage)

        if self.IsPadded:
            Lmodel=[]
            blc_trc=self.DicoSubDirty["blc_trc"]
            original_shape=self.DicoSubDirty["original_shape"]
            nch,npol,_,_=ModelImage.shape
            for ich in range(nch):
                for ipol in range(npol):
                    restored_array = unpad_to_original(ModelImage[ich,ipol], blc_trc)
                    Lmodel.append(restored_array)
            nx,ny=original_shape
            ModelImageIsland=np.array(Lmodel).reshape((nch,npol,nx,ny))
            
        x,y=self.ArrayPixParms.T
        
        ModelImageIsland=ModelImage[:,0,x,y]
        
        # print("LJDFLJDDFJ",ModelImageIsland.max())
        # print("LJDFLJDDFJ",ModelImageIsland.max())
        # print("LJDFLJDDFJ",ModelImageIsland.max())
        # print("LJDFLJDDFJ",ModelImageIsland.max())
        # print("LJDFLJDDFJ",ModelImageIsland.max())
        
        return ModelImageIsland,self.DeconvMachine.NSpectralFit
    


