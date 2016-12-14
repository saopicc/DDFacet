import numpy as np
from DDFacet.Imager.MSMF import ClassImageDeconvMachineMSMF
import copy
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Imager.ModModelMachine import ClassModModelMachine

class ClassInitSSDModel():
    def __init__(self,GD,DicoVariablePSF,DicoDirty,RefFreq,MainCache=None):

        self.DicoVariablePSF=DicoVariablePSF
        self.DicoDirty=DicoDirty
        GD=copy.deepcopy(GD)
        self.RefFreq=RefFreq
        self.GD=GD
        self.GD["Parallel"]["NCPU"]=1
        self.GD["MultiFreqs"]["Alpha"]=[0,0,1]#-1.,1.,5]
        #self.GD["MultiFreqs"]["Alpha"]=[-1.,1.,5]
        self.GD["ImagerDeconv"]["MinorCycleMode"]="MSMF"
        self.GD["ImagerDeconv"]["CycleFactor"]=0
        self.GD["ImagerDeconv"]["PeakFactor"]=0.01
        self.GD["ImagerDeconv"]["RMSFactor"]=0
        self.GD["ImagerDeconv"]["Gain"]=0.02
        self.GD["MultiScale"]["Scales"]=[0,1,2,3]
        self.GD["MultiScale"]["SolverMode"]="NNLS"
        self.NFreqBands=len(DicoVariablePSF["freqs"])
        MinorCycleConfig=dict(self.GD["ImagerDeconv"])
        MinorCycleConfig["NCPU"]=self.GD["Parallel"]["NCPU"]
        MinorCycleConfig["NFreqBands"]=self.NFreqBands
        MinorCycleConfig["GD"] = self.GD
        #MinorCycleConfig["RefFreq"] = self.RefFreq

        ModConstructor = ClassModModelMachine(self.GD)
        ModelMachine = ModConstructor.GiveMM(Mode=self.GD["ImagerDeconv"]["MinorCycleMode"])
        ModelMachine.setRefFreq(self.RefFreq)
        MinorCycleConfig["ModelMachine"]=ModelMachine

        
        self.MinorCycleConfig=MinorCycleConfig
        self.DeconvMachine=ClassImageDeconvMachineMSMF.ClassImageDeconvMachine(MainCache=MainCache,**self.MinorCycleConfig)
        self.Margin=100
        self.DicoDirty=DicoDirty
        self.Dirty=DicoDirty["ImagData"]
        self.MeanDirty=DicoDirty["MeanImage"]
        self.DeconvMachine.Init(PSFVar=self.DicoVariablePSF,PSFAve=self.DicoVariablePSF["PSFSideLobes"])
        self.DeconvMachine.Update(self.DicoDirty)
        self.DeconvMachine.updateRMS()
        #self.DicoBasicModelMachine=copy.deepcopy(self.DeconvMachine.ModelMachine.DicoSMStacked)

    def setSubDirty(self,ListPixParms):
        x,y=np.array(ListPixParms).T
        x0,x1=x.min(),x.max()+1
        y0,y1=y.min(),y.max()+1
        dx=x1-x0+self.Margin
        dy=y1-y0+self.Margin
        Size=np.max([dx,dy])
        if Size%2==0: Size+=1
        _,_,N0,_=self.Dirty.shape
        xc0,yc0=int(round(np.mean(x))),int(round(np.mean(y)))
        self.xy0=xc0,yc0
        N1=Size
        xc1=yc1=N1/2
        Aedge,Bedge=GiveEdges((xc0,yc0),N0,(xc1,yc1),N1)
        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge
        self.SubDirty=self.Dirty[:,:,x0d:x1d,y0d:y1d].copy()

        self.blc=(x0d,y0d)
        self.DeconvMachine.PSFServer.setBLC(self.blc)
        _,_,nx,ny=self.SubDirty.shape
        ArrayPixParms=np.array(ListPixParms)
        ArrayPixParms[:,0]-=x0d
        ArrayPixParms[:,1]-=y0d
        Mask=np.zeros((nx,ny),np.bool8)
        self.ArrayPixParms=ArrayPixParms
        x,y=ArrayPixParms.T
        Mask[x,y]=1
        self.SubMask=Mask
        self.DicoSubDirty={}
        for key in self.DicoDirty.keys():
            if key in ['ImagData', "MeanImage",'NormImage',"NormData"]:
                self.DicoSubDirty[key]=self.DicoDirty[key][...,x0d:x1d,y0d:y1d].copy()
            else:
                self.DicoSubDirty[key]=self.DicoDirty[key]

        # ModelImage=np.zeros_like(self.Dirty)
        # ModelImage[:,:,N0/2,N0/2]=10
        # ModelImage[:,:,N0/2+3,N0/2]=10
        # ModelImage[:,:,N0/2-2,N0/2-1]=10
        # self.setSSDModelImage(ModelImage)

        if self.SSDModelImage is not None:
            self.SubSSDModelImage=self.SSDModelImage[:,:,x0d:x1d,y0d:y1d].copy()
            for ch in range(self.NFreqBands):
                self.SubSSDModelImage[ch,0][np.logical_not(self.SubMask)]=0
            self.addSubModelToSubDirty()

    def setSSDModelImage(self,ModelImage):
        self.SSDModelImage=ModelImage

    def addSubModelToSubDirty(self):
        ConvModel=np.zeros_like(self.SubSSDModelImage)
        nch,_,N0x,N0y=ConvModel.shape
        indx,indy=np.where(self.SubSSDModelImage[0,0]!=0)
        xc,yc=N0x/2,N0y/2
        self.DeconvMachine.PSFServer.setLocation(*self.xy0)
        PSF,MeanPSF=self.DeconvMachine.PSFServer.GivePSF()
        N1=PSF.shape[-1]
        for i,j in zip(indx.tolist(),indy.tolist()):
            ThisPSF=np.roll(np.roll(PSF,i-xc,axis=-2),j-yc,axis=-1)
            Aedge,Bedge=GiveEdgesDissymetric((xc,yc),(N0x,N0y),(N1/2,N1/2),(N1,N1))
            x0d,x1d,y0d,y1d=Aedge
            x0p,x1p,y0p,y1p=Bedge
            ConvModel[...,x0d:x1d,y0d:y1d]+=ThisPSF[...,x0p:x1p,y0p:y1p]*self.SubSSDModelImage[...,i,j].reshape((-1,1,1,1))

        MeanConvModel=np.mean(ConvModel,axis=0).reshape((1,1,N0x,N0y))
        self.DicoSubDirty['ImagData']+=ConvModel
        self.DicoSubDirty['MeanImage']+=MeanConvModel
        print "MAX=",np.max(self.DicoSubDirty['MeanImage'])

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
        self.setSubDirty(ListPixParms)

        ModConstructor = ClassModModelMachine(self.GD)
        ModelMachine = ModConstructor.GiveMM(Mode=self.GD["ImagerDeconv"]["MinorCycleMode"])
        self.ModelMachine=ModelMachine
        #self.ModelMachine.DicoSMStacked=self.DicoBasicModelMachine
        self.ModelMachine.setRefFreq(self.RefFreq,Force=True)
        self.MinorCycleConfig["ModelMachine"] = ModelMachine
        self.ModelMachine.setModelShape(self.SubDirty.shape)
        self.ModelMachine.setListComponants(self.DeconvMachine.ModelMachine.ListScales)
        
        self.DeconvMachine.updateMask(np.logical_not(self.SubMask))
        self.DeconvMachine.Update(self.DicoSubDirty)
        self.DeconvMachine.updateModelMachine(ModelMachine)
        self.DeconvMachine.resetCounter()
        self.DeconvMachine.Deconvolve(UpdateRMS=False)

        ModelImage=self.ModelMachine.GiveModelImage()

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
        SModel=ModelImage[0,0,x,y]
        AModel=self.ModelMachine.GiveSpectralIndexMap()[0,0,x,y]
        return SModel,AModel
