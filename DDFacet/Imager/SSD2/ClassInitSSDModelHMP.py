from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import numpy as np
import copy
import time
import traceback
from DDFacet.Imager.MSMF import ClassImageDeconvMachineMSMF
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.Imager.ModModelMachine import ClassModModelMachine
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import logger
log=logger.getLogger("ClassInitSSDModelHMP")
from DDFacet.Other import ModColor
from DDFacet.Imager.SSD.ClassConvMachine import ClassConvMachineImages
from DDFacet.Imager import ClassMaskMachine
from DDFacet.Array import shared_dict
import psutil
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Other.AsyncProcessPool import APP

class ClassInitSSDModelParallel():
    def __init__(self, GD, NFreqBands, RefFreq, MainCache=None, IdSharedMem=""):
        self.GD=GD
        self.InitMachine = ClassInitSSDModel(GD, NFreqBands, RefFreq, MainCache, IdSharedMem)
        self.NCPU=(self.GD["Parallel"]["NCPU"] or psutil.cpu_count())
        APP.registerJobHandlers(self)

    def Init(self, DicoVariablePSF, GridFreqs, DegridFreqs):
        self.DicoVariablePSF=DicoVariablePSF
        self.GridFreqs=GridFreqs
        self.DegridFreqs=DegridFreqs

        print("Initialise HMP machine", file=log)
        self.InitMachine.Init(DicoVariablePSF, GridFreqs, DegridFreqs)

    def Reset(self):
        self.DicoVariablePSF = None
        self.InitMachine.Reset()

    def _initIsland_worker(self, DicoOut, iIsland, Island,
                           DicoVariablePSF, DicoDirty, DicoParm, FacetCache,NCPU):
        logger.setSilent(["ClassImageDeconvMachineMSMF", "ClassPSFServer", "ClassMultiScaleMachine", "GiveModelMachine", "ClassModelMachineMSMF"])
        self.InitMachine.Init(DicoVariablePSF, DicoParm["GridFreqs"], DicoParm["DegridFreqs"], facetcache=FacetCache)
        self.InitMachine.setDirty(DicoDirty)
        # self.InitMachine.DeconvMachine.setNCPU(NCPU)
        self.InitMachine.setSSDModelImage(DicoParm["ModelImage"])

        #print ":::::::::::::::::::::::",iIsland

        try:
            SModel, AModel = self.InitMachine.giveModel(Island)
        except:
            if not self.GD["GAClean"]["ParallelInitHMP"]:
                raise
            print(traceback.format_exc(), file=log)
            FileOut = "errIsland_%6.6i.npy" % iIsland
            print(ModColor.Str("...... error on island %i, saving to file %s" % (iIsland, FileOut)), file=log)
            np.save(FileOut, np.array(Island))
            self.InitMachine.Reset()
            return
        #DicoOut["S"] = SModel
        #DicoOut["Alpha"] = AModel
        PolyModel=np.zeros((self.GD["SSD2"]["PolyFreqOrder"],SModel.size),SModel.dtype)
        PolyModel[0,:]=SModel
        PolyModel[1,:]=AModel
        DicoOut["PolyModel"] = PolyModel
        self.InitMachine.Reset()

    def giveDicoInitIndiv(self, ListIslands, ModelImage, DicoDirty, ListDoIsland=None):
        DicoInitIndiv = shared_dict.create("DicoInitIsland")
        ParmDict = shared_dict.create("InitSSDModelHMP")
        ParmDict["ModelImage"] = ModelImage
        ParmDict["GridFreqs"] = self.GridFreqs
        ParmDict["DegridFreqs"] = self.DegridFreqs
        
#         ListBigIslands=[]
#         ListSmallIslands=[]
#         ListDoBigIsland=[]
#         ListDoSmallIsland=[]
#         NParallel=0
#         for iIsland,Island in enumerate(ListIslands):
#             if len(Island)>self.GD["SSDClean"]["ConvFFTSwitch"]:
#                 ListBigIslands.append(Island)
#                 ListDoBigIsland.append(ListDoIsland[iIsland])
#                 if ListDoIsland or ListDoIsland[iIsland]:
#                     NParallel+=1
#             else:
#                 ListSmallIslands.append(Island)
#                 ListDoSmallIsland.append(ListDoIsland[iIsland])
#         print>>log,"Initialise big islands (parallelised per island)"
#         pBAR= ProgressBar(Title="Init islands")
#         pBAR.render(0, NParallel)
#         nDone=0
#         for iIsland,Island in enumerate(ListBigIslands):
#             if not ListDoIsland or ListDoBigIsland[iIsland]:
#                 subdict = DicoInitIndiv.addSubdict(iIsland)
#                 # APP.runJob("InitIsland:%d" % iIsland, self._initIsland_worker,
#                 #            args=(subdict.writeonly(), iIsland, Island,
#                 #                  self.DicoVariablePSF.readonly(), DicoDirty.readonly(),
#                 #                  ParmDict.readonly(), self.InitMachine.DeconvMachine.facetcache.readonly(),self.NCPU),serial=True)
#                 self._initIsland_worker(subdict, iIsland, Island,
#                                         self.DicoVariablePSF, DicoDirty,
#                                         ParmDict, self.InitMachine.DeconvMachine.facetcache,
#                                         self.NCPU)
#                 pBAR.render(nDone+1, NParallel)
#                 nDone+=1
# #        APP.awaitJobResults("InitIsland:*", progress="Init islands")
#         print>>log,"Initialise small islands (parallelised over islands)"
#         for iIsland,Island in enumerate(ListSmallIslands):
#             if not ListDoIsland or ListDoSmallIsland[iIsland]:
#                 subdict = DicoInitIndiv.addSubdict(iIsland)
#                 APP.runJob("InitIsland:%d" % iIsland, self._initIsland_worker,
#                            args=(subdict.writeonly(), iIsland, Island,
#                                  self.DicoVariablePSF.readonly(), DicoDirty.readonly(),
#                                  ParmDict.readonly(), self.InitMachine.DeconvMachine.facetcache.readonly(),1))
#         APP.awaitJobResults("InitIsland:*", progress="Init islands")
#         DicoInitIndiv.reload()


        print("Initialise islands (parallelised over islands)", file=log)
        if self.InitMachine.DeconvMachine.facetcache is None:
            print("HMP bases not initialized. Will re-initialize now.", file=log)
        if not self.GD["GAClean"]["ParallelInitHMP"]:
          pBAR = ProgressBar(Title="  Init islands HMP")
          for iIsland,Island in enumerate(ListIslands):
            if not ListDoIsland or ListDoIsland[iIsland]:
                subdict = DicoInitIndiv.addSubdict(iIsland)
                self._initIsland_worker(subdict, 
                                        iIsland, 
                                        Island,
                                        self.DicoVariablePSF, 
                                        DicoDirty,
                                        ParmDict, 
                                        self.InitMachine.DeconvMachine.facetcache,
                                        1)
            pBAR.render(iIsland, len(ListIslands))
        else:
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
                                 self.InitMachine.DeconvMachine.facetcache.readonly() 
                                    if self.InitMachine.DeconvMachine.facetcache is not None else None,
                                 1))
          APP.awaitJobResults("InitIsland:*", progress="Init islands HMP")
          DicoInitIndiv.reload()
        
        ParmDict.delete()

        return DicoInitIndiv

######################################################################################################

class ClassInitSSDModel():
    """
    This class is essentially a wrapper around a single HMP machine. It initializes an HMP machine
    with very specific settings, then uses it deconvolve (init) SSD islands.
    
    The class is initialized once in the main process (to populate the HMP basis function cache),
    then re-initialized in the workers on a per-island basis.
    """
    def __init__(self, GD, NFreqBands, RefFreq, MainCache=None, IdSharedMem=""):
        """Constructs initializer. 
        Note that this should be called pretty much when setting up the imager,
        before APP workers are started, because the object registers APP handlers.
        """
        self.GD = copy.deepcopy(GD)
        self.GD["Parallel"]["NCPU"] = 1
        # self.GD["HMP"]["Alpha"]=[0,0,1]#-1.,1.,5]
        self.GD["HMP"]["Alpha"] = self.GD["GAClean"]["AlphaInitHMP"]
        self.GD["Deconv"]["Mode"] = "HMP"
        self.GD["Deconv"]["CycleFactor"] = 0
        self.GD["Deconv"]["PeakFactor"] = 0.0
        self.GD["Deconv"]["RMSFactor"] = self.GD["GAClean"]["RMSFactorInitHMP"]
        self.GD["Deconv"]["Gain"] = self.GD["GAClean"]["GainInitHMP"]
        self.GD["Deconv"]["AllowNegative"] = self.GD["GAClean"]["AllowNegativeInitHMP"]
        self.GD["Deconv"]["MaxMinorIter"] = int(self.GD["GAClean"]["MaxMinorIterInitHMP"])

        self.GD["HMP"]["Scales"] = self.GD["GAClean"]["ScalesInitHMP"]

        self.GD["HMP"]["Ratios"] =  self.GD["GAClean"]["RatiosInitHMP"]
        # self.GD["MultiScale"]["Ratios"]=[]
        self.GD["HMP"]["NTheta"] = self.GD["GAClean"]["NThetaInitHMP"]

        # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # # self.GD["HMP"]["Scales"] = [0,1,2,4,8,16,24,32,48,64]
        # # self.GD["HMP"]["Taper"] = 32
        # # self.GD["HMP"]["Support"] = 32#self.GD["HMP"]["Scales"][-1]
        # self.GD["Deconv"]["RMSFactor"] = 1.
        # self.GD["Deconv"]["AllowNegative"] = True
        
        self.GD["HMP"]["SolverMode"] = "NNLS"
        # self.GD["MultiScale"]["SolverMode"]="PI"

        self.NFreqBands = NFreqBands
        self.RefFreq = RefFreq
        MinorCycleConfig = dict(self.GD["Deconv"])
        MinorCycleConfig["NCPU"] = self.GD["Parallel"]["NCPU"]
        MinorCycleConfig["NFreqBands"] = self.NFreqBands
        MinorCycleConfig["RefFreq"] = RefFreq

        ModConstructor = ClassModModelMachine(self.GD)
        ModelMachine = ModConstructor.GiveMM(Mode=self.GD["Deconv"]["Mode"])
        ModelMachine.setRefFreq(self.RefFreq)
        MinorCycleConfig["ModelMachine"] = ModelMachine

        self.MinorCycleConfig = MinorCycleConfig
        self.DeconvMachine = ClassImageDeconvMachineMSMF.ClassImageDeconvMachine(MainCache=MainCache,
                                                                                 ParallelMode=True,
                                                                                 CacheFileName="HMP_Init",
                                                                                 IdSharedMem=IdSharedMem,
                                                                                 GD=self.GD,
                                                                                 **MinorCycleConfig)

        self.GD["Mask"]["Auto"]=False
        self.GD["Mask"]["External"]=None
        self.MaskMachine = ClassMaskMachine.ClassMaskMachine(self.GD)

    def Init(self,DicoVariablePSF,GridFreqs,DegridFreqs,
                 DoWait=False,
                 facetcache=None):
        """
        Init method. Note that this will end up being called in one of two modes. In the main process,
        it is called to initialize the HMP machine's basis function cache (so facetcache=None). After this is 
        done, the cache is passed to workers, where the HMP machine is initialized from facetcache.
        
        facetcache: dict of basis functions for the HMP machine.
        """
        self.DicoVariablePSF=DicoVariablePSF
        self.GridFreqs=GridFreqs
        self.DegridFreqs=DegridFreqs

        self.DeconvMachine.setMaskMachine(self.MaskMachine)

        self.Margin=20

        self.DeconvMachine.Init(PSFVar=self.DicoVariablePSF,PSFAve=self.DicoVariablePSF["PSFSideLobes"],
                                facetcache=facetcache,
                                GridFreqs=self.GridFreqs,DegridFreqs=self.DegridFreqs,DoWait=DoWait)


    def Reset(self):
        self.DeconvMachine.Reset()

    def setDirty(self, DicoDirty):
        self.DicoDirty=DicoDirty
        self.Dirty=DicoDirty["ImageCube"]
        self.MeanDirty=DicoDirty["MeanImage"]
        self.DeconvMachine.Update(self.DicoDirty,DoSetMask=False)
        self.DeconvMachine.updateRMS()

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
        #self.DeconvMachine.PSFServer.iFacet=118
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


        # PSF,MeanPSF=self.DeconvMachine.PSFServer.GivePSF()
        # import pylab
        # pylab.clf()
        # ax=pylab.subplot(1,3,1)
        # N=self.DicoSubDirty["MeanImage"].shape[-1]
        # pylab.imshow(self.DicoSubDirty["MeanImage"][0,0],
        #              interpolation="nearest",extent=(-N//2.,N//2.,-N//2.,N//2.),vmin=-0.1,vmax=1.)
        # pylab.colorbar()
        # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
        # N=MeanPSF.shape[-1]
        # pylab.imshow(MeanPSF[0,0],interpolation="nearest",extent=(-N//2.,N//2.,-N//2.,N//2.),vmin=-0.1,vmax=1.)
        # pylab.colorbar()
        # pylab.draw()
        # pylab.show()

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
        #self.DicoSubDirty['MeanImage']+=MeanConvModel
        
        W=np.float32(self.DicoSubDirty["WeightChansImages"])
        W=W/np.sum(W)
        MeanImage=np.sum(self.DicoSubDirty["ImageCube"]*W.reshape((-1,1,1,1)),axis=0).reshape((1,1,N0x,N0y))
        self.DicoSubDirty['MeanImage']=MeanImage
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

        ## OMS: no need for this, surely -- RefFreq is set from ModelMachine in the first place?
        self.ModelMachine.setRefFreq(self.RefFreq,Force=True)

        ## this doesn't seem to be needed or used outside of __init__, so why assign to it?
        # self.MinorCycleConfig["ModelMachine"] = ModelMachine
        self.ModelMachine.setModelShape(self.SubDirty.shape)
        self.ModelMachine.setListComponants(self.DeconvMachine.ModelMachine.ListScales)
        T.timeit("setlistcomp")
        
        self.DeconvMachine.Update(self.DicoSubDirty,DoSetMask=False)
        self.DeconvMachine.updateMask(np.logical_not(self.SubMask))
        self.DeconvMachine.updateModelMachine(ModelMachine)
        self.DeconvMachine.resetCounter()
        T.timeit("update")

        self.DeconvMachine.Deconvolve(UpdateRMS=False)
        #self.DeconvMachine.Plot()
        T.timeit("deconv %s"%str(self.DicoSubDirty["ImageCube"].shape))
        #print "deconv"
        #time.sleep(30)

        ModelImage=self.ModelMachine.GiveModelImage()
        T.timeit("getmodel")

        # import pylab
        # pylab.clf()
        # pylab.subplot(2,2,1)
        # pylab.imshow(self.DicoDirty["MeanImage"][0,0,:,:],interpolation="nearest")
        # pylab.colorbar()
        # pylab.subplot(2,2,2)
        # pylab.imshow(self.DicoSubDirty["MeanImage"][0,0,:,:],interpolation="nearest")
        # pylab.colorbar()
        # pylab.subplot(2,2,3)
        # pylab.imshow(self.SubMask,interpolation="nearest")
        # pylab.colorbar()
        # pylab.subplot(2,2,4)
        # pylab.imshow(ModelImage[0,0],interpolation="nearest")
        # pylab.colorbar()
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop

        x,y=self.ArrayPixParms.T

        # PSF,MeanPSF=self.DeconvMachine.PSFServer.GivePSF()
        # ConvModel=ClassConvMachineImages(PSF).giveConvModel(ModelImage*np.ones((self.NFreqBands,1,1,1)))
        # #T.timeit("Conv1")
        # #print "done1"
        # #ConvModel=self.giveConvModel(ModelImage*np.ones((self.NFreqBands,1,1,1)))
        # # print "done2"
        # # T.timeit("Conv2")
        # # import pylab
        # # pylab.clf()
        # # pylab.subplot(1,3,1)
        # # pylab.imshow(ConvModel[0,0],interpolation="nearest")
        # # pylab.subplot(1,3,2)
        # # pylab.imshow(ConvModel1[0,0],interpolation="nearest")
        # # pylab.subplot(1,3,3)
        # # pylab.imshow((ConvModel-ConvModel1)[0,0],interpolation="nearest")
        # # pylab.colorbar()
        # # pylab.draw()
        # # pylab.show(False)
        # # stop
        
        # ModelOnes=np.zeros_like(ModelImage)
        # ModelOnes[:,:,x,y]=1
        # ConvModelOnes=ClassConvMachineImages(PSF).giveConvModel(ModelOnes*np.ones((self.NFreqBands,1,1,1)))

        # SumConvModel=np.sum(ConvModel[:,:,x,y])
        # SumConvModelOnes=np.sum(ConvModelOnes[:,:,x,y])
        # SumResid=np.sum(self.DeconvMachine._CubeDirty[:,:,x,y])

        # SumConvModel=np.max([SumConvModel,1e-6])

        # factor=(SumResid+SumConvModel)/SumConvModel

        
        # ###############
        #fMult=1.
        #if 1.<factor<2.:
        #    fMult=factor
        fMult=1.
        SModel=ModelImage[0,0,x,y]*fMult
        # ###########"
        # fMult=(np.mean(SumResid))/(np.mean(SumConvModelOnes))
        # SModel=ModelImage[0,0,x,y]+ModelOnes[0,0,x,y]*fMult
        # print fMult
        # print fMult
        # print fMult
        # print fMult
        # print fMult
        # print fMult
        # print fMult
        # ############


        if self.NFreqBands>1:
            AModel=self.ModelMachine.GiveSpectralIndexMap()[0,0,x,y]
        else:
            AModel=np.zeros_like(SModel)
        T.timeit("spec index")

        return SModel,AModel


