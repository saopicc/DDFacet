import numpy as np
from DDFacet.Imager.MORESANE import ClassImageDeconvMachineMoresane
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
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassInitSSDModel")
import traceback
from DDFacet.Other import ModColor
from ClassConvMachine import ClassConvMachineImages
from DDFacet.Imager import ClassMaskMachine

SilentModules=["ClassPSFServer","ClassImageDeconvMachine","GiveModelMachine","ClassModelMachineMoresane","ClassModelMachineSSD","pymoresane.main"]

class ClassInitSSDModelParallel():
    def __init__(self, GD, NFreqBands, RefFreq, NCPU, MainCache=None,IdSharedMem=""):
        self.GD = copy.deepcopy(GD)
        self.MainCache=MainCache
        self.RefFreq=RefFreq
        self.NCPU = NCPU
        self.IdSharedMem=IdSharedMem


    def Init(self, DicoVariablePSF, RefFreq, GridFreqs, DegridFreqs):
        self.DicoVariablePSF=DicoVariablePSF
        self.GridFreqs=GridFreqs
        self.DegridFreqs=DegridFreqs
        print>>log,"Initialise MORESANE machine"
        self.InitMachine=ClassInitSSDModel(self.GD,
                                           self.DicoVariablePSF,
                                           self.RefFreq,
                                           self.GridFreqs,
                                           self.DegridFreqs,
                                           MainCache=self.MainCache,
                                           IdSharedMem=self.IdSharedMem)


    def giveDicoInitIndiv(self,ListIslands,ModelImage,DicoDirty,ListDoIsland=None,Parallel=True):
        NCPU=self.NCPU
        work_queue = multiprocessing.JoinableQueue()
        ListIslands=ListIslands#[300:308]
        DoIsland=True
        
        
        
        for iIsland in range(len(ListIslands)):
            if ListDoIsland is not None:
                DoIsland=ListDoIsland[iIsland]
            if DoIsland: work_queue.put({"iIsland":iIsland})

        result_queue=multiprocessing.JoinableQueue()
        NJobs=work_queue.qsize()
        workerlist=[]

        MyLogger.setSilent(SilentModules)
        #MyLogger.setLoud(SilentModules)

        #MyLogger.setLoud("ClassImageDeconvMachineMSMF")

        print>>log,"Launch MORESANE workers"
        for ii in range(NCPU):
            W = WorkerInitMSMF(work_queue,
                               result_queue,
                               self.GD,
                               self.DicoVariablePSF,
                               DicoDirty,
                               self.RefFreq,
                               self.GridFreqs,
                               self.DegridFreqs,
                               self.MainCache,
                               ModelImage,
                               ListIslands,
                               self.IdSharedMem)
            workerlist.append(W)
            if Parallel:
                workerlist[ii].start()

        timer = ClassTimeIt.ClassTimeIt()
        pBAR = ProgressBar(Title="  MORESANing islands ")
        #pBAR.disable()
        pBAR.render(0, NJobs)
        iResult = 0
        if not Parallel:
            for ii in range(NCPU):
                workerlist[ii].run()  # just run until all work is completed

        self.DicoInitIndiv={}
        while iResult < NJobs:
            DicoResult = None
            if result_queue.qsize() != 0:
                try:
                    DicoResult = result_queue.get()
                except:
                    pass

            if DicoResult == None:
                time.sleep(0.5)
                continue

            if DicoResult["Success"]:
                iResult+=1
                NDone=iResult

                pBAR.render(NDone,NJobs)

                iIsland=DicoResult["iIsland"]
                NameDico="%sDicoInitIsland_%5.5i"%(self.IdSharedMem,iIsland)
                Dico=NpShared.SharedToDico(NameDico)
                self.DicoInitIndiv[iIsland]=copy.deepcopy(Dico)
                NpShared.DelAll(NameDico)



        if Parallel:
            for ii in range(NCPU):
                workerlist[ii].shutdown()
                workerlist[ii].terminate()
                workerlist[ii].join()
        
        #MyLogger.setLoud(["pymoresane.main"])
        #MyLogger.setLoud(["ClassImageDeconvMachineMSMF","ClassPSFServer","ClassMultiScaleMachine","GiveModelMachine","ClassModelMachineMSMF"])
        return self.DicoInitIndiv

######################################################################################################

class ClassInitSSDModel():
    def __init__(self,GD,DicoVariablePSF,RefFreq,GridFreqs,DegridFreqs,
                 MainCache=None,
                 IdSharedMem="",
                 DoWait=False):
        self.DicoVariablePSF=DicoVariablePSF
        GD=copy.deepcopy(GD)
        self.RefFreq=RefFreq
        self.GridFreqs=GridFreqs
        self.DegridFreqs=DegridFreqs
        self.GD=GD
        self.GD["Parallel"]["NCPU"]=1
        #self.GD["HMP"]["Alpha"]=[0,0,1]#-1.,1.,5]
        self.GD["HMP"]["Alpha"]=[-1.,1.,5]
        self.GD["Deconv"]["Mode"]="MORESANE"
        
        self.GD["Deconv"]["CycleFactor"]=0
        self.GD["Deconv"]["PeakFactor"]=0.0
        self.GD["Deconv"]["RMSFactor"]=self.GD["GAClean"]["RMSFactorInitHMP"]

        self.GD["Deconv"]["Gain"]=self.GD["GAClean"]["GainInitHMP"]
        self.GD["Deconv"]["AllowNegative"]=self.GD["GAClean"]["AllowNegativeInitHMP"]
        self.GD["Deconv"]["MaxMinorIter"]=self.GD["GAClean"]["MaxMinorIterInitHMP"]
        
        MyLogger.setSilent(SilentModules)

        self.GD["HMP"]["Scales"]=self.GD["GAClean"]["ScalesInitHMP"]

        self.GD["HMP"]["Ratios"]=[]
        #self.GD["MultiScale"]["Ratios"]=[]
        self.GD["HMP"]["NTheta"]=4
        
        self.GD["HMP"]["SolverMode"]="NNLS"
        #self.GD["MultiScale"]["SolverMode"]="PI"

        self.NFreqBands=len(DicoVariablePSF["freqs"])
        MinorCycleConfig=dict(self.GD["Deconv"])
        MinorCycleConfig["NCPU"]=self.GD["Parallel"]["NCPU"]
        MinorCycleConfig["NFreqBands"]=self.NFreqBands
        MinorCycleConfig["GD"] = self.GD
        MinorCycleConfig["GridFreqs"] = self.GridFreqs
        MinorCycleConfig["DegridFreqs"] = self.DegridFreqs

        #MinorCycleConfig["RefFreq"] = self.RefFreq

        ModConstructor = ClassModModelMachine(self.GD)
        ModelMachine = ModConstructor.GiveMM(Mode="MORESANE")
        ModelMachine.setRefFreq(self.RefFreq)
        MinorCycleConfig["ModelMachine"]=ModelMachine
        #MinorCycleConfig["CleanMaskImage"]=None
        self.MinorCycleConfig=MinorCycleConfig
        self.DeconvMachine=ClassImageDeconvMachineMoresane.ClassImageDeconvMachine(MainCache=MainCache,
                                                                                   ParallelMode=False,
                                                                                   RefFreq=self.RefFreq,
                                                                                   CacheFileName="HMP_Init",
                                                                                   IdSharedMem=IdSharedMem,
                                                                                   **self.MinorCycleConfig)
        self.GD["Mask"]["Auto"]=False
        self.GD["Mask"]["External"]=None
        self.MaskMachine=ClassMaskMachine.ClassMaskMachine(self.GD)
        self.DeconvMachine.setMaskMachine(self.MaskMachine)

        self.Margin=50

        #print "Start 3"
        self.DeconvMachine.Init(PSFVar=self.DicoVariablePSF,PSFAve=self.DicoVariablePSF["PSFSideLobes"],
                                GridFreqs=self.GridFreqs,DegridFreqs=self.DegridFreqs,DoWait=DoWait,RefFreq=self.RefFreq)

        if DoWait:
            print "IINit3"
            time.sleep(10)
            print "Start 4"

        self.DeconvMachine.Update(self.DicoDirty,DoSetMask=False)
        if DoWait:
            print "IINit4"
            time.sleep(10)

        #self.DeconvMachine.updateRMS()

        #self.DicoBasicModelMachine=copy.deepcopy(self.DeconvMachine.ModelMachine.DicoSMStacked)

    def setDirty(self, DicoDirty):
        self.DicoDirty=DicoDirty
        self.Dirty=DicoDirty["ImageCube"]
        self.MeanDirty=DicoDirty["MeanImage"]

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
        xc1=yc1=N1/2
        Aedge,Bedge=GiveEdges((xc0,yc0),N0,(xc1,yc1),N1)
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
        # ModelImage[:,:,N0/2,N0/2]=10
        # ModelImage[:,:,N0/2+3,N0/2]=10
        # ModelImage[:,:,N0/2-2,N0/2-1]=10
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
        # xc,yc=N0x/2,N0y/2
        # N1=PSF.shape[-1]
        # #T.timeit("0")
        # for i,j in zip(indx.tolist(),indy.tolist()):
        #     ThisPSF=np.roll(np.roll(PSF,i-xc,axis=-2),j-yc,axis=-1)
        #     Aedge,Bedge=GiveEdgesDissymetric((xc,yc),(N0x,N0y),(N1/2,N1/2),(N1,N1))
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
        self.ModelMachine.setFreqMachine(self.GridFreqs,self.DegridFreqs)
        self.MinorCycleConfig["ModelMachine"] = ModelMachine
        self.ModelMachine.setModelShape(self.SubDirty.shape)
        #self.ModelMachine.setListComponants(self.DeconvMachine.ModelMachine.ListScales)
        T.timeit("setlistcomp")
        
        self.DeconvMachine.Update(self.DicoSubDirty,DoSetMask=False)
        #self.DeconvMachine.updateMask(np.logical_not(self.SubMask))
        self.DeconvMachine.updateModelMachine(ModelMachine)
        #self.DeconvMachine.resetCounter()
        T.timeit("update")
        #print "update"
        #time.sleep(30)
        self.DeconvMachine.Deconvolve()
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


        AModel=self.ModelMachine.GiveSpectralIndexMap()[0,0,x,y]
        T.timeit("spec index")

        return SModel,AModel



##########################################
####### Workers
##########################################
import os
import signal
           
class WorkerInitMSMF(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 GD,
                 DicoVariablePSF,
                 DicoDirty,
                 RefFreq,
                 GridFreqs,
                 DegridFreqs,
                 MainCache,
                 ModelImage,
                 ListIsland,
                 IdSharedMem):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.GD=GD
        self.DicoVariablePSF=DicoVariablePSF
        self.DicoDirty=DicoDirty
        self.RefFreq=RefFreq
        self.GridFreqs=GridFreqs
        self.DegridFreqs=DegridFreqs
        self.MainCache=MainCache
        self.ModelImage=ModelImage
        self.ListIsland=ListIsland
        self.InitMachine=None
        self.IdSharedMem=IdSharedMem

    def Init(self):

        #print "sleeeping init0"
        #time.sleep(10)
        if self.InitMachine is not None: return
        self.InitMachine=ClassInitSSDModel(self.GD,
                                           self.DicoVariablePSF,
                                           self.RefFreq,
                                           self.GridFreqs,
                                           self.DegridFreqs,
                                           MainCache=self.MainCache,
                                           IdSharedMem=self.IdSharedMem,
                                           DoWait=False)
        self.InitMachine.setSSDModelImage(self.ModelImage)
        self.InitMachine.setDirty(self.DicoDirty)
        #print "sleeeping init1"
        #time.sleep(10)


    def shutdown(self):
        self.exit.set()


    def initIsland(self, DicoJob):
        if self.InitMachine is None:
            self.Init()
        iIsland=DicoJob["iIsland"]
        Island=self.ListIsland[iIsland]
        SModel,AModel=self.InitMachine.giveModel(Island)
        

        DicoInitIndiv={"S":SModel,"Alpha":AModel}
        NameDico="%sDicoInitIsland_%5.5i"%(self.IdSharedMem,iIsland)
        NpShared.DicoToShared(NameDico, DicoInitIndiv)
        self.result_queue.put({"Success": True, "iIsland": iIsland})


    def run(self):
        while not self.kill_received and not self.work_queue.empty():
            
            DicoJob = self.work_queue.get()
            # self.initIsland(DicoJob)
            try:
                self.initIsland(DicoJob)
            except:
                print traceback.format_exc()
                iIsland=DicoJob["iIsland"]
                FileOut="errIsland_%6.6i.npy"%iIsland
                print ModColor.Str("...... on island %i, saving to file %s"%(iIsland,FileOut))
                np.save(FileOut,np.array(self.ListIsland[iIsland]))
                print







