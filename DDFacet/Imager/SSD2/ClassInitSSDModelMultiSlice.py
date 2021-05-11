from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import numpy as np
from DDFacet.Imager.MultiSliceDeconv import ClassImageDeconvMachineMultiSlice
import copy
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Imager.ModModelMachine import ClassModModelMachine
import multiprocessing
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
from DDFacet.Other.AsyncProcessPool import APP

SilentModules=["ClassPSFServer",
               "ClassImageDeconvMachine",
               "GiveModelMachine",
               "ClassImageDeconvMachineMultiSlice",
               "ClassModelMachineMultiSlice",
               #"ClassTaylorToPower",
               "ClassModelMachineSSD"]

class ClassInitSSDModelParallel():
    def __init__(self, GD, NFreqBands, RefFreq, NCPU, MainCache=None,IdSharedMem=""):
        self.GD = copy.deepcopy(GD)
        
        self.GD["MultiSliceDeconv"]["PolyFitOrder"]=self.GD["SSD2"]["PolyFreqOrder"]
        self.MainCache=MainCache
        self.RefFreq=RefFreq
        self.NCPU = NCPU
        self.IdSharedMem=IdSharedMem
        self.NFreqBands=NFreqBands

        self.InitMachine = ClassInitSSDModel(self.GD, NFreqBands, RefFreq, MainCache, IdSharedMem)
        self.NCPU=(self.GD["Parallel"]["NCPU"] or psutil.cpu_count())
        APP.registerJobHandlers(self)

        

    def Init(self, DicoVariablePSF, GridFreqs, DegridFreqs):
        self.DicoVariablePSF=DicoVariablePSF
        self.GridFreqs=GridFreqs
        self.DegridFreqs=DegridFreqs
        print("Initialise MultiSlice machine", file=log)
        self.InitMachine=ClassInitSSDModel(self.GD, self.NFreqBands, self.RefFreq, MainCache=self.MainCache, IdSharedMem=self.IdSharedMem)

    def Reset(self):
        self.DicoVariablePSF = None
        #self.InitMachine.Reset()


    def _initIsland_worker(self, DicoOut, iIsland, Island,
                           DicoVariablePSF, DicoDirty, DicoParm, NCPU):
        logger.setSilent(["ClassImageDeconvMachineMSMF", "ClassPSFServer", "ClassMultiScaleMachine", "GiveModelMachine", "ClassModelMachineMSMF"])
        self.InitMachine.Init(DicoVariablePSF, DicoParm["GridFreqs"], DicoParm["DegridFreqs"])
        self.InitMachine.setDirty(DicoDirty)
        # self.InitMachine.DeconvMachine.setNCPU(NCPU)
        self.InitMachine.setSSDModelImage(DicoParm["ModelImage"])

        #print ":::::::::::::::::::::::",iIsland

        try:
            ModelImageIsland = self.InitMachine.giveModel(Island)
        except:
            if not self.GD["GAClean"]["ParallelInitHMP"]:
                raise
            print(traceback.format_exc(), file=log)
            FileOut = "errIsland_%6.6i.npy" % iIsland
            print(ModColor.Str("...... error on island %i, saving to file %s" % (iIsland, FileOut)), file=log)
            np.save(FileOut, np.array(Island))
            self.InitMachine.Reset()
            return
        DicoOut["PolyModel"] = ModelImageIsland
        
        self.InitMachine.Reset()

    def giveDicoInitIndiv(self, ListIslands, ModelImage, DicoDirty, ListDoIsland=None):
        DicoInitIndiv = shared_dict.create("DicoInitIsland")
        ParmDict = shared_dict.create("InitSSDModelMultiSlice")
        ParmDict["ModelImage"] = ModelImage
        ParmDict["GridFreqs"] = self.GridFreqs
        ParmDict["DegridFreqs"] = self.DegridFreqs
        
        print("Initialise islands (parallelised over islands)", file=log)
        for iIsland,Island in enumerate(ListIslands):
            if not ListDoIsland or ListDoIsland[iIsland]:
                subdict = DicoInitIndiv.addSubdict(iIsland)
                APP.runJob("InitIsland:%d" % iIsland, self._initIsland_worker,
                           args=(subdict.writeonly(), 
                                 iIsland, 
                                 Island,
                                 self.DicoVariablePSF.readonly(), 
                                 DicoDirty.readonly(),
                                 ParmDict.readonly(), 
                                 1))
        APP.awaitJobResults("InitIsland:*", progress="Init islands MultiSlice")
        DicoInitIndiv.reload()
            
        ParmDict.delete()

        return DicoInitIndiv


    # def giveDicoInitIndiv(self,ListIslands,ModelImage,DicoDirty,ListDoIsland=None,Parallel=True):
    #     #Parallel=False
    #     NCPU=self.NCPU
    #     work_queue = multiprocessing.JoinableQueue()
    #     ListIslands=ListIslands#[300:308]
    #     DoIsland=True
        
        
        
    #     for iIsland in range(len(ListIslands)):
    #         if ListDoIsland is not None:
    #             DoIsland=ListDoIsland[iIsland]
    #         if DoIsland: work_queue.put({"iIsland":iIsland})

    #     result_queue=multiprocessing.JoinableQueue()
    #     NJobs=work_queue.qsize()
    #     workerlist=[]

    #     logger.setSilent(SilentModules)
    #     #MyLogger.setLoud(SilentModules)

    #     #MyLogger.setLoud("ClassImageDeconvMachineMSMF")

    #     print("Launch MultiSlice workers", file=log)
    #     for ii in range(NCPU):
    #         W = WorkerInitMSMF(work_queue,
    #                            result_queue,
    #                            self.GD,
    #                            self.DicoVariablePSF,
    #                            DicoDirty,
    #                            self.RefFreq,
    #                            self.GridFreqs,
    #                            self.DegridFreqs,
    #                            self.MainCache,
    #                            ModelImage,
    #                            ListIslands,
    #                            self.IdSharedMem)
    #         workerlist.append(W)
    #         if Parallel:
    #             workerlist[ii].start()

    #     timer = ClassTimeIt.ClassTimeIt()
    #     pBAR = ProgressBar(Title="  Init islands MultiSlice")
    #     #pBAR.disable()
    #     pBAR.render(0, NJobs)
    #     iResult = 0
    #     if not Parallel:
    #         for ii in range(NCPU):
    #             workerlist[ii].run()  # just run until all work is completed

    #     self.DicoInitIndiv={}
    #     while iResult < NJobs:
    #         DicoResult = None
    #         if result_queue.qsize() != 0:
    #             try:
    #                 DicoResult = result_queue.get()
    #             except:
    #                 pass

    #         if DicoResult == None:
    #             time.sleep(0.5)
    #             continue

    #         if DicoResult["Success"]:
    #             iResult+=1
    #             NDone=iResult

    #             pBAR.render(NDone,NJobs)

    #             iIsland=DicoResult["iIsland"]
    #             NameDico="%sDicoInitIsland%5.5i"%(self.IdSharedMem,iIsland)

    #             Dico=NpShared.SharedToDico(NameDico)
    #             self.DicoInitIndiv[iIsland]=copy.deepcopy(Dico)
    #             NpShared.DelAll(NameDico)



    #     if Parallel:
    #         for ii in range(NCPU):
    #             workerlist[ii].shutdown()
    #             workerlist[ii].terminate()
    #             workerlist[ii].join()
        
    #     #MyLogger.setLoud(["pymoresane.main"])
    #     #MyLogger.setLoud(["ClassImageDeconvMachineMSMF","ClassPSFServer","ClassMultiScaleMachine","GiveModelMachine","ClassModelMachineMSMF"])
    #     return self.DicoInitIndiv

######################################################################################################

class ClassInitSSDModel():
    def __init__(self, GD, NFreqBands, RefFreq, MainCache=None, IdSharedMem=""):
        GD=copy.deepcopy(GD)
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
        self.DeconvMachine=ClassImageDeconvMachineMultiSlice.ClassImageDeconvMachine(MainCache=MainCache,
                                                                                     ParallelMode=False,
                                                                                     RefFreq=self.RefFreq,
                                                                                     CacheFileName="MultiSlice_Init",
                                                                                     IdSharedMem=IdSharedMem,
                                                                                     GD=self.GD,
                                                                                     **self.MinorCycleConfig)
        self.GD["Mask"]["Auto"]=False
        self.GD["Mask"]["External"]=None

        self.Margin=50

    def Reset(self):
        pass
    
    def Init(self,DicoVariablePSF,GridFreqs,DegridFreqs,
                 DoWait=False):
        self.DicoVariablePSF=DicoVariablePSF
        self.GridFreqs=GridFreqs
        self.DegridFreqs=DegridFreqs


        self.Margin=20

        self.DeconvMachine.Init(PSFVar=self.DicoVariablePSF,PSFAve=self.DicoVariablePSF["PSFSideLobes"],
                                GridFreqs=self.GridFreqs,DegridFreqs=self.DegridFreqs,DoWait=DoWait)


    def setDirty(self, DicoDirty):
        self.DicoDirty=DicoDirty
        self.Dirty=DicoDirty["ImageCube"]
        self.MeanDirty=DicoDirty["MeanImage"]
        self.DeconvMachine.Update(self.DicoDirty,DoSetMask=False)
        #self.DeconvMachine.updateRMS()

    def setSubDirty(self,ListPixParms):
        T=ClassTimeIt.ClassTimeIt("InitSSD.setSubDirty")
        T.disable()

        x,y=np.array(ListPixParms).T
        x0,x1=x.min(),x.max()+1
        y0,y1=y.min(),y.max()+1
        dx=x1-x0+self.Margin
        dy=y1-y0+self.Margin
        Size=np.max([dx,dy])
        if Size%2==0: Size+=1
        _,_,N0,_=self.Dirty.shape

        xc0,yc0=int((x1+x0)/2.),int((y1+y0)/2.)
        self.xy0=xc0,yc0
        self.DeconvMachine.PSFServer.setLocation(*self.xy0)

        N1=Size
        xc1=yc1=N1//2
        Aedge,Bedge=GiveEdges(xc0,yc0,N0,xc1,yc1,N1)
        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge
        self.SubDirty=self.Dirty[:,:,x0d:x1d,y0d:y1d].copy()
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

        # Mask=np.zeros((nx,ny),np.bool8)
        # Mask[x,y]=1
        # self.SubMask=Mask


        x,y=ArrayPixParms.T
        Mask=np.zeros(self.DicoSubDirty["ImageCube"].shape[-2::],np.bool8)
        Mask[x,y]=1
        self.SubMask=Mask


        if self.SSDModelImage is not None:
            self.SubSSDModelImage=self.SSDModelImage[:,:,x0d:x1d,y0d:y1d].copy()
            for ch in range(self.NFreqBands):
                self.SubSSDModelImage[ch,0][np.logical_not(self.SubMask)]=0
            self.addSubModelToSubDirty()
        T.timeit("2")


    def setSSDModelImage(self,ModelImage):
        self.SSDModelImage=ModelImage

    def giveConvModel(self,SubModelImage):

        PSF,MeanPSF=self.DeconvMachine.PSFServer.GivePSF()
        ConvModel=ClassConvMachineImages(PSF).giveConvModel(SubModelImage)

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
        ConvModel=self.giveConvModel(self.SubSSDModelImage)
        _,_,N0x,N0y=ConvModel.shape
        MeanConvModel=np.mean(ConvModel,axis=0).reshape((1,1,N0x,N0y))
        self.DicoSubDirty["ImageCube"]+=ConvModel
        self.DicoSubDirty['MeanImage']+=MeanConvModel
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

            
    def giveModel(self,ListPixParms):
        T=ClassTimeIt.ClassTimeIt("giveModel")
        T.disable()
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
        
        self.DeconvMachine.Update(self.DicoSubDirty,DoSetMask=False)
        self.DeconvMachine.updateMask(np.logical_not(self.SubMask))
        self.DeconvMachine.updateModelMachine(self.ModelMachine)
        #self.DeconvMachine.resetCounter()
        T.timeit("update")
        #print "update"
        #time.sleep(30)
        
        rep,_,_=self.DeconvMachine.Deconvolve()
        if rep=="Edge":
            return np.zeros((self.GD["SSD2"]["PolyFreqOrder"],len(ListPixParms)),np.float32)
        
        T.timeit("deconv %s"%str(self.DicoSubDirty["ImageCube"].shape))
        #print "deconv"
        #time.sleep(30)

        ModelImage=self.ModelMachine.DicoModel["CoefImage"]#GiveModelImage()
        if self.ModelMachine.DicoModel["FluxScale"]=="Linear":
            ModelImage=self.CTP.LinPolyCube2LogPolyCube(ModelImage)
        
        x,y=self.ArrayPixParms.T
        
        ModelImageIsland=ModelImage[:,0,x,y]
        return ModelImageIsland
    


