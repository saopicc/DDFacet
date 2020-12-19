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
log=logger.getLogger("ClassInitSSDModel")
from DDFacet.Other import ModColor
from DDFacet.Imager.SSD.ClassConvMachine import ClassConvMachineImages
from DDFacet.Imager import ClassMaskMachine
from DDFacet.Array import shared_dict
import psutil
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Other.AsyncProcessPool import APP

class ClassImageDeconvMachineIsland():
    def __init__(self, GD, NFreqBands, RefFreq, MainCache=None, **MinorCycleConfig):
        self.GD=GD
        
        if self.GD["SSD2"]["IslandDeconvMode"] == "HMP":
            from DDFacet.Imager.MSMF import ClassImageDeconvMachineMSMF
            self.DeconvMachine=ClassImageDeconvMachineMSMF.ClassImageDeconvMachine(GD=self.GD,MainCache=MainCache, **MinorCycleConfig)
            print("Using MSMF algorithm", file=log)
        elif self.GD["SSD2"]["IslandDeconvMode"] == "Hogbom":
            from DDFacet.Imager.HOGBOM import ClassImageDeconvMachineHogbom
            self.DeconvMachine=ClassImageDeconvMachineHogbom.ClassImageDeconvMachine(**MinorCycleConfig)
            print("Using Hogbom algorithm", file=log)
        elif self.GD["SSD2"]["IslandDeconvMode"]=="MORESANE":
            from DDFacet.Imager.MORESANE import ClassImageDeconvMachineMoresane
            self.DeconvMachine=ClassImageDeconvMachineMoresane.ClassImageDeconvMachine(MainCache=MainCache, **MinorCycleConfig)
            print("Using MORESANE algorithm", file=log)
        elif self.GD["SSD2"]["IslandDeconvMode"]=="WSCMS":
            from DDFacet.Imager.WSCMS import ClassImageDeconvMachineWSCMS
            self.DeconvMachine = ClassImageDeconvMachineWSCMS.ClassImageDeconvMachine(MainCache=MainCache,
                                                                                      **MinorCycleConfig)
            print("Using WSCMS algorithm", file=log)
        else:
            raise NotImplementedError("Unknown --Deconvolution-Mode setting '%s'" % self.GD["Deconv"]["Mode"])
        
        self.GD["Mask"]["Auto"]=False
        self.GD["Mask"]["External"]=None
        self.MaskMachine = ClassMaskMachine.ClassMaskMachine(self.GD)
        self.DeconvMachine.setMaskMachine(self.MaskMachine)

        self.NCPU=(self.GD["Parallel"]["NCPU"] or psutil.cpu_count())
        
        APP.registerJobHandlers(self)

    def Init(self, DicoVariablePSF, GridFreqs, DegridFreqs):
        self.DicoVariablePSF=DicoVariablePSF
        self.GridFreqs=GridFreqs
        self.DegridFreqs=DegridFreqs

        print("Initialise island deconv machine", file=log)
        self.DeconvMachine.Init(PSFVar=self.DicoVariablePSF,
                                PSFAve=self.DicoVariablePSF["PSFSideLobes"],
                                facetcache=facetcache,
                                GridFreqs=self.GridFreqs,
                                DegridFreqs=self.DegridFreqs,
                                DoWait=DoWait)

        
    def Reset(self):
        self.DicoVariablePSF = None
        self.DeconvMachine.Reset()

    def _deconvIsland_worker(self, DicoOut, iIsland, Island,
                           DicoVariablePSF, DicoDirty, DicoParm, FacetCache,NCPU):
        logger.setSilent(["ClassImageDeconvMachineMSMF", "ClassPSFServer", "ClassMultiScaleMachine", "GiveModelMachine", "ClassModelMachineMSMF"])
        self.DeconvMachine.Init(DicoVariablePSF, DicoParm["GridFreqs"], DicoParm["DegridFreqs"], facetcache=FacetCache)
        self.DeconvMachine.setDirty(DicoDirty)
        self.DeconvMachine.setSSDModelImage(DicoParm["ModelImage"])
        
        self.DeconvMachine.Reset()


        self.NFreqBands = NFreqBands
        self.RefFreq = RefFreq
        
        MinorCycleConfig = dict(self.GD["Deconv"])
        MinorCycleConfig["NCPU"] = 1#self.GD["Parallel"]["NCPU"]
        MinorCycleConfig["NFreqBands"] = self.NFreqBands
        MinorCycleConfig["RefFreq"] = RefFreq

        ModConstructor = ClassModModelMachine(self.GD)
        ModelMachine = ModConstructor.GiveMM(Mode=self.GD["Deconv"]["Mode"])
        ModelMachine.setRefFreq(self.RefFreq)
        MinorCycleConfig["ModelMachine"] = ModelMachine
        self.MinorCycleConfig = MinorCycleConfig



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
        #print "update"
        #time.sleep(30)
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

