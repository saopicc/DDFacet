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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import numpy as np
from DDFacet.Other import logger
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
log=logger.getLogger("ClassModelMachineSSD3")
from DDFacet.Array import NpParallel
from DDFacet.Array import ModLinAlg
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyPickle
from DDFacet.Other import reformat
from DDFacet.Imager import ClassFrequencyMachine
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.Imager import ClassModelMachine as ClassModelMachinebase
from DDFacet.ToolsDir import ModFFTW
import scipy.ndimage
from SkyModel.Sky import ModRegFile
from pyrap.images import image
from SkyModel.Sky import ClassSM
import os
import copy
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
from collections import deque
import scipy.stats
from . import ClassFitPreviousModels
def fMyScale(x,RMS=1.): return np.arcsinh((x/RMS)/2)/np.log(10)
def inv_fMyScale(x,RMS=1.): return RMS*np.sinh(x*np.log(10))*2

# def fMyScale(x,RMS=1.):
#     x[x<=0]=1e-10
#     return np.log10(x/RMS)
# def inv_fMyScale(x,RMS=1.): return RMS*10**x







class ClassModelMachine(ClassModelMachinebase.ClassModelMachine):
    def __init__(self,*args,**kwargs):
        ClassModelMachinebase.ClassModelMachine.__init__(self, *args, **kwargs)
        # self.GD=GD
        # if Gain is None:
        #     self.Gain=self.GD["Deconv"]["Gain"]
        # else:
        #     self.Gain=Gain
        # self.GainMachine=GainMachine
        # self.DicoSMStacked["Comp"]={}
        if self.GD is not None:
            self.setParams()
        self.RefFreq=None
        self.DicoSMStacked={}
        self.DicoSMStacked["Type"]="SSD3"
        self.DicoSMStacked["Comp"]={}
        self.NDeque=self.GD["SSD3"]["NLookBackModels"]
        self.PastModels=deque([],self.NDeque)
        self.PastModels_Resid=deque([],self.NDeque)
        #self.PastModels_STD=deque([],self.NDeque)
        self.Alpha=1.
        self.AAlpha=None
        self.xrRand=None
        self.LSTD=[]
        self.LResid1D=[]
        
    def setParams(self):
        NOrderPoly=self.GD["SSD3"]["PolyFreqOrder"]
        SolveParamType=self.GD["SSD3"]["SolvePars"]
        
        SolveParam=[]
        if "Poly" in SolveParamType:
            for iOrder in range(NOrderPoly): SolveParam.append("Poly%i"%iOrder)
        if "GSig" in SolveParamType: 
            SolveParam.append("GSig")
            
        #self.PolyOrder=NOrderPoly#np.sum([("Poly" in key) for key in SolveParam])
        
        self.SolveParam=SolveParam
        
        
        print("Solved parameters: %s"%(str(self.SolveParam)), file=log)
        self.NParam=len(self.SolveParam)
        

    def setRefFreq(self,RefFreq,Force=False):#,AllFreqs):
        if self.RefFreq is not None and not Force and RefFreq!=self.RefFreq:
            print(ModColor.Str("Reference frequency already set to %f MHz"%(self.RefFreq/1e6)), file=log)
            return
        self.RefFreq=RefFreq
        self.DicoSMStacked["RefFreq"]=RefFreq
        #self.DicoSMStacked["AllFreqs"]=np.array(AllFreqs)
        # print "ModelMachine:",self.RefFreq, self.DicoSMStacked["RefFreq"], self.DicoSMStacked["AllFreqs"]


    def ToFile(self,FileName,DicoIn=None):
        print("Saving dico model to %s"%FileName, file=log)
        if DicoIn is None:
            D=self.DicoSMStacked
        else:
            D=DicoIn

        #D["PM"]=self.PM
        D["GD"]=self.GD
        D["ModelShape"]=self.ModelShape
        D["Type"]="SSD3"
        D["SolveParam"]=self.SolveParam
        D["PastModels"]=self.PastModels
        D["PastModels_Resid"]=self.PastModels_Resid

        MyPickle.Save(D,FileName)

    def ChangeNPix(self,NPixOut):
        stop
        NPix=self.ModelShape[-1]
        NPixOut, _ = EstimateNpix(float(NPixOut), Padding=1)
        NPix0, _ = EstimateNpix(float(NPix), Padding=1)
        if NPix!=NPix0: stop
        print>>log,"Changing image size: %i -> %i pixels"%(NPix,NPixOut)
        xc0=NPix//2
        xc1=NPixOut//2
        dx=xc0-xc1
        DCompOut={}
        DCompOut["Type"]="SSD3"
        N,M,_,_=self.ModelShape
        self.ModelShape=[N,M,NPixOut,NPixOut]
        for (x0,y0) in self.DicoSMStacked['Comp'].keys():
            x1=x0-dx
            y1=y0-dx
            c0=(x1>=0)&(x1<NPixOut)
            c1=(y1>=0)&(y1<NPixOut)
            if c0&c1:
                #print "(%i,%i)->(%i,%i)"%(x0,y0,x1,y1)
                DCompOut[(x1,y1)]=self.DicoSMStacked['Comp'][(x0,y0)]
        self.DicoSMStacked=DCompOut
        
    def giveDico(self):
        D=self.DicoSMStacked
        D["GD"]=self.GD
        D["ModelShape"]=self.ModelShape
        D["Type"]="SSD3"
        D["SolveParam"]=self.SolveParam
        return D

    def FromFile(self,FileName):
        print("Reading dico model from file %s"%FileName, file=log)
        self.DicoSMStacked=MyPickle.Load(FileName)
        self.FromDico(self.DicoSMStacked)


    def FromDico(self,DicoSMStacked):
        print("Reading dico model from dico ", file=log)
        self.DicoSMStacked=DicoSMStacked
        self.RefFreq=self.DicoSMStacked["RefFreq"]
        self.ModelShape=self.DicoSMStacked["ModelShape"]
        self.SolveParam=self.DicoSMStacked["SolveParam"]
        self.PastModels=self.DicoSMStacked["PastModels"]
        self.PastModels_Resid=self.DicoSMStacked["PastModels_Resid"]
            
        self.NParam=len(self.SolveParam)

    def GiveConvertedSolveParamDico(self,SolveParam1):
        stop
        SolveParam0=self.DicoSMStacked["SolveParam"]
        print("Converting SSD3 model %s into %s..."%(str(SolveParam0),str(SolveParam1)), file=log)
        DicoOut=copy.deepcopy(self.DicoSMStacked)
        DicoOut["SolveParam"]=SolveParam1
        del(DicoOut["Comp"])
        DicoOut["Comp"]={}
        NParam0=len(SolveParam0)
        NParam1=len(SolveParam1)
        indexParm=[]
        for TypeParm in SolveParam1:
            indexParm.append(np.where(np.array(SolveParam0)==TypeParm)[0])

        for xy in self.DicoSMStacked["Comp"].keys():
            Coefs=np.zeros((NParam1,),np.float32)
            Comp=self.DicoSMStacked["Comp"][xy]
            for iTypeParm,iIndex in enumerate(indexParm):
                if iIndex.size==0: continue
                Coefs[iTypeParm]=Comp["Vals"][0][iIndex[0]]
            DicoOut["Comp"][xy]={"Vals":[Coefs]}

        return DicoOut
            
        

        
    def setModelShape(self,ModelShape):
        self.ModelShape=ModelShape

    def setThreshold(self,Th):
        self.Th=Th

        
    def GiveIndividual(self,ListPixParms):
        NParms=self.NParam
        try:
            DicoComp=self.DicoSMStacked["Comp"]
        except:
            self.DicoSMStacked["Comp"]={}
            DicoComp=self.DicoSMStacked["Comp"]

        if "Vals" in DicoComp.keys():
            x,y=np.array(ListPixParms).T
            OutArr=np.array([DicoComp["Vals"][iParam][x,y] for iParam in range(self.NParam)])
        else:
            OutArr=np.zeros((NParms,len(ListPixParms)),np.float32)
            
        # for iPix in range(len(ListPixParms)):

        #     xy=x,y
        #     try:
        #         Vals=DicoComp[xy]["Vals"][0]
        #         OutArr[:,iPix]=Vals[:]
        #         #del(DicoComp[xy])
        #     except:
        #         pass

        return OutArr.flatten()


    def AppendIsland(self,ListPixParms,V,W=None,JonesNorm=None):
        ListPix=ListPixParms
        Vr=V.reshape((self.NParam,V.size//self.NParam))
        NPixListParms=len(ListPixParms)

        DicoComp=self.DicoSMStacked["Comp"]
        
        x,y=np.array(ListPix).T
        for iParam in range(self.NParam):
            #Vr[iParam].fill(1)
            DicoComp["Vals"][iParam][x,y]+=W[:]*Vr[iParam]
        DicoComp["Weights"][x,y]+=W[:]


    def giveMask_nonZeroModel(self):
        Mask=np.zeros(self.ModelShape,np.bool_)
        if "Comp" in self.DicoSMStacked.keys() and len(self.DicoSMStacked["Comp"])>0:
            
            ImTaylor0=self.DicoSMStacked["Comp"]["Vals"][0]
            Mask[0,0].flat[:]=(ImTaylor0!=0).flat[:]
        return Mask
            
    def reinitIslands(self,ListIslands):
        if "Comp" not in self.DicoSMStacked.keys():
            self.DicoSMStacked["Comp"]={}
            
        DicoComp=self.DicoSMStacked["Comp"]

        if DicoComp.get("Vals",None) is None:
            _,_,nx,ny=self.ModelShape
            DicoComp["Vals"]=np.zeros((self.NParam,nx,ny),np.float32)
            DicoComp["Weights"]=np.zeros((nx,ny),np.float32)
            
        
            
        for Island in ListIslands:
            x,y=np.array(Island).T
            for iParam in range(self.NParam):
                DicoComp["Vals"][iParam][x,y]=0
                DicoComp["Weights"][x,y]=0
                
            # for key in Island:
            #     key=tuple(key)
            #     try:
            #         del(DicoComp[key]["Vals"])
            #         del(DicoComp[key]["Weights"])
            #     except:
            #         pass
            #     DicoComp[key]={}
            #     DicoComp[key]["Vals"]=[]
            #     DicoComp[key]["Weights"]=[]

    def AppendComponentToDictStacked(self,key,Vals,W=None):
        stop
        DicoComp=self.DicoSMStacked["Comp"]
        DicoComp[key]["Vals"].append(Vals)
        if W is None:
            DicoComp[key]["Weights"].append(1.)
        else:
            DicoComp[key]["Weights"].append(W)
            
    def updateAlpha(self,MeanDirty):
        if self.GD["SSD3"]["NLookBackModels"]==0: return
        nch,_,nx,ny=MeanDirty.shape
        if nch!=1: stop
        if self.xrRand is None:
            Nr=10000
            xr=np.int64(np.random.rand(Nr)*nx)
            yr=np.int64(np.random.rand(Nr)*nx)
            self.xryr=(xr,yr)
        
        self.PastModels_Resid.append(MeanDirty.copy())
        
        # xr,yr = self.xryr
        # Resid1D=MeanDirty.flat[xr*ny+yr]
        # STD = scipy.stats.median_abs_deviation(Resid1D,axis=None,scale="normal")
        # self.LSTD.append(STD)
        # #self.PastModels_STD.append(STD)
        # if len(self.LSTD)>0:
        #     CSTD=(STD>0.75*self.LSTD[-1])
        #     if CSTD:
        #         STD0,STD1=self.LSTD[-1],STD
        #         Alpha=self.Alpha/2
        #         log.print("Reducing Alpha %.2f -> %.2f [std = %f vs %f Jy]"%(self.Alpha,Alpha,STD0,STD1))
        #         self.Alpha=Alpha
        # self.LResid1D.append( Resid1D.copy() )
        
            
    def RenormaliseMultiEstimatesPerPixel(self):
        DicoComp=self.DicoSMStacked["Comp"]

                
        x,y=np.where(DicoComp["Vals"][0]!=0)

        for iParam in range(self.NParam):
            DicoComp["Vals"][iParam][x,y]/=DicoComp["Weights"][x,y]
        # DicoComp["Weights"][x,y]=1

        # ThisModel=DicoComp["Vals"][0]
        # if len(self.PastModels)>1:
        #     sgn0=np.sign(self.PastModels[1][0]-self.PastModels[0][0])
        #     sgn1=np.sign(ThisModel-self.PastModels[1][0])
        #     C0=True#(self.PastModels[0][0]!=0)
        #     C1=True#(self.PastModels[1][0]!=0)
        #     C2=True#(ThisModel!=0)
        #     Cs=(sgn0!=sgn1)
        #     x,y=np.where(C0 & C1 & C2 & Cs)
        #     # # ThisModel[x,y] = (self.PastModels[-1][x,y]+ThisModel[x,y])/2
        # #     dThisModel = ThisModel[x,y]-self.PastModels[-1][x,y]
        # #     ThisModel[x,y]=self.PastModels[-1][x,y]+dThisModel/4
        # #     log.print("  Have rescaled %.2f%% of oscilating componants"%(100*x.size/ThisModel.size))
        # # self.PastModels.append(ThisModel.copy())
        #     DicoComp["Vals"][:,x,y]=(self.PastModels[-1][:,x,y]+DicoComp["Vals"][:,x,y])/2
        #     log.print("  Have rescaled %.2f%% of oscilating componants"%(100*x.size/ThisModel.size))
        # self.PastModels.append(DicoComp["Vals"].copy())

        nParms,nx,ny=DicoComp["Vals"].shape
        if self.AAlpha is None:
            self.AAlpha=np.ones((1,nx,ny),np.float32)
        ThisModel_mean=DicoComp["Vals"][0]

        
        #if len(self.PastModels)>=self.GD["Deconv"]["MaxMajorIter"]//2:
        if self.GD["SSD3"]["NLookBackModels"]!=0 and len(self.PastModels)>=2:
            log.print("Use %i past models to update..."%len(self.PastModels))
            # sgn0=np.sign(self.PastModels[1][0]-self.PastModels[0][0])
            # sgn1=np.sign(ThisModel_mean-self.PastModels[1][0])
            # C0=True#(self.PastModels[0][0]!=0)
            # C1=True#(self.PastModels[1][0]!=0)
            # C2=True#(ThisModel!=0)
            # Cs=(sgn0!=sgn1)
            # x,y=np.where(C0 & C1 & C2 & Cs)
            # self.AAlpha[0,x,y]=self.AAlpha[0,x,y]/2
            # dThisModel = DicoComp["Vals"][:]-self.PastModels[-1][:]
            # DicoComp["Vals"][:] = self.PastModels[-1][:] + self.AAlpha * dThisModel
            # ##################
            
            # self.PastModels_Resid: NDeque,1,1,nx,ny
            # self.PastModels: NDeque,nParm,nx,ny
            PastModels_Resid=np.array(self.PastModels_Resid).copy()
            PastModels=np.array(self.PastModels)
            NDeque,nParm,nx,ny=PastModels.shape
            
            
            PastModels_Resid=PastModels_Resid[-NDeque:].reshape((NDeque,1,nx,ny))
            PastModels=PastModels.reshape((NDeque,self.NParam,nx,ny))
            #STD=np.array(self.PastModels_STD[-NDeque:]).reshape((NDeque,1,1,1))

            LSTD=[]
            xr,yr = self.xryr
            for ThisResid in PastModels_Resid:
                Resid1D=ThisResid.flat[xr*ny+yr]
                LSTD.append(scipy.stats.median_abs_deviation(Resid1D,axis=None,scale="normal"))
            STD=np.array(LSTD)
            
            aPastModels_Resid=np.abs(PastModels_Resid)
            
            
            Th=0.1*STD
            for iModel in range(NDeque):
                M=(aPastModels_Resid[iModel,0]<Th[iModel])
                indx,indy=np.where(M)
                aPastModels_Resid[iModel,0,indx,indy]=Th[iModel]
            W=1./aPastModels_Resid**2#**2
            
            S0=PastModels[:,0:1]
            W[S0==0]=0
            
            Ws=np.sum(W,axis=0)
            Ws[Ws==0]=1.
            MeanModelPast0=np.sum(W*PastModels,axis=0)/Ws

            # weight parameters others than the flux by the flux
            MeanModelPast=np.zeros((PastModels[0].shape),PastModels.dtype)
            for iTerm in range(self.NParam):
                wt=W
                if iTerm!=0:
                    wt=W*np.abs(S0)
                Sm=np.sum(wt*PastModels[:,iTerm:iTerm+1],axis=0)
                Sw=np.sum(wt,axis=0)
                Sw[Sw==0]=1
                Sm=Sm/Sw
                MeanModelPast[iTerm]=Sm[0]

            # Wa=np.ones((1,nx,ny),np.float32)
            # Wa[MeanModelPast[0:1]==0]=0
            # Wb=np.ones((1,nx,ny),np.float32)
            # #DicoComp["Vals"][:] = MeanModelPast[:]
            # DicoComp["Vals"][:] = (Wa*MeanModelPast + Wb*DicoComp["Vals"][:])/(Wa+Wb)

            Wa=np.ones((self.NParam,nx,ny),np.float32)
            Wb=np.ones((self.NParam,nx,ny),np.float32)
            for iTerm in range(1,self.NParam):
                Wa[iTerm]*=np.abs(MeanModelPast[0])
                Wb[iTerm]*=np.abs(DicoComp["Vals"][0])



            # ####################
            # self.CurrentModel = DicoComp["Vals"][:]
            # CFPM=ClassFitPreviousModels.ClassFitPreviousModels(self)
            # MeanModelPast=CFPM.avgWeighted()
            
            # Wa.fill(1)
            
            # Sw=Wa+Wb
            # Sw[Sw==0]=1
            # #Wa[MeanModelPast[0:1]==0]=0
            # #DicoComp["Vals"][:] = (Wa*MeanModelPast + Wb*DicoComp["Vals"][:])/Sw#(Wa+Wb)
            # DicoComp["Vals"][:] = MeanModelPast[:]
            # ##################

            
            #DicoComp["Vals"][:] = self.PastModels[-1][:] + self.Alpha * dThisModel
            #log.print("  Have rescaled model using Alpha = %.2f"%(self.Alpha))

        if self.GD["SSD3"]["NLookBackModels"]!=0:
            self.PastModels.append(DicoComp["Vals"].copy())
        
        if self.GD["SSD3"]["ForcePositiveModel"]:
            x,y=np.where(DicoComp["Vals"][0]<0)
            for iParam in range(self.NParam):
                DicoComp["Vals"][iParam][x,y]=0
        
        
        
    def GiveModelImage(self,FreqIn=None,out=None,InModelParms=None):
        
        RefFreq=self.DicoSMStacked["RefFreq"]
        if FreqIn is None:
            FreqIn=np.array([RefFreq])
            
        log.print("Compute model image at %s MHz..."%str((FreqIn/1e6).tolist()))

        #if type(FreqIn)==float:
        #    FreqIn=np.array([FreqIn]).flatten()
        #if type(FreqIn)==np.ndarray:

        FreqIn=np.array([FreqIn.ravel()]).flatten()


        # print "ModelMachine GiveModelImage:",FreqIn, RefFreq

        _,npol,nx,ny=self.ModelShape
        nchan=FreqIn.size
        if out is not None:
            if out.shape != (nchan,npol,nx,ny) or out.dtype != np.float32:
                raise RuntimeError("supplied image has incorrect type (%s) or shape (%s)" % (out.dtype, out.shape))
            ModelImage = out
        else:
            ModelImage = np.zeros((nchan,npol,nx,ny),dtype=np.float32)

        if "Comp" not in  self.DicoSMStacked.keys():
            return ModelImage
        if  len(self.DicoSMStacked["Comp"])==0: 
            return ModelImage
        
        if InModelParms is None:
            DicoComp=self.DicoSMStacked["Comp"]
            C=DicoComp["Vals"]
        else:
            C=InModelParms
        
        N0=nx

        DicoSM={}
        SolveParam=np.array(self.SolveParam)
        
        x,y=np.where(C[0]!=0)
        logS=np.zeros((nx,ny),np.float32)
        for ich in range(nchan):
            S0=C[0]#np.zeros((ind.size,),np.float32)
            ThisFreq=FreqIn[ich]
            logS.fill(0)
            for iCoef in range(1,self.NParam):
                logS+=C[iCoef]*(np.log(ThisFreq/RefFreq))**iCoef
            ModelImage[ich,0]=S0*np.exp(logS)
        
        return ModelImage
    
        

        
    def setListComponants(self,ListScales):
        self.ListScales=ListScales

    # def GiveSpectralIndexMap(self, threshold=0.1, save_dict=True):
    #     # Get the model image
    #     IM = self.GiveModelImage(self.FreqMachine.Freqsp)
    #     nchan, npol, Nx, Ny = IM.shape
    #     # Fit the alpha map
    #     self.FreqMachine.FitAlphaMap(IM[:, 0, :, :], threshold=threshold) # should set threshold based on SNR of final residual
    #     if save_dict:
    #         FileName = self.GD['Output']['Name'] + ".Dicoalpha"
    #         print>>log, "Saving componentwise SPI map to %s"%FileName
    #         MyPickle.Save(self.FreqMachine.alpha_dict, FileName)
    #     return self.FreqMachine.weighted_alpha_map.reshape((1, 1, Nx, Ny))


    def GiveSpectralIndexMap(self,CellSizeRad=1.,GaussPars=[(1,1,0)],DoConv=True,MaxSpi=100,MaxDR=1e+6,threshold=None):
    
        dFreq=1e6
        RefFreq=self.DicoSMStacked["RefFreq"]
        f0=RefFreq/1.5#self.DicoSMStacked["AllFreqs"].min()
        f1=RefFreq*1.5#self.DicoSMStacked["AllFreqs"].max()
        M0=self.GiveModelImage(f0)
        M1=self.GiveModelImage(f1)
        if DoConv:
            CellSizeRad_x,CellSizeRad_y=CellSizeRad
            FWHMFact = 2. * np.sqrt(2. * np.log(2.))
            FWHMdeg=GaussPars[0][0]
            FWHMrad=FWHMdeg*np.pi/180
            Sig=FWHMrad/FWHMFact/CellSizeRad_x
            #M0=ModFFTW.ConvolveGaussian(M0,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
            #M1=ModFFTW.ConvolveGaussian(M1,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
            #M0,_=ModFFTW.ConvolveGaussianWrapper(M0,Sig=GaussPars[0][0]/CellSizeRad)
            #M1,_=ModFFTW.ConvolveGaussianWrapper(M1,Sig=GaussPars[0][0]/CellSizeRad)
            M0,_=ModFFTW.ConvolveGaussianScipy(M0,Sig=(Sig,Sig))
            M1,_=ModFFTW.ConvolveGaussianScipy(M1,Sig=(Sig,Sig))

            
        # compute threshold for alpha computation by rounding DR threshold to .1 digits (i.e. 1.65e-6 rounds to 1.7e-6)
        if threshold is not None:
            minmod = threshold
        elif not np.all(M0==0):
            minmod = float("%.1e"%(np.max(np.abs(M0))/MaxDR))
        else:
            minmod=1e-6
    
        # mask out pixels above threshold
        mask=(M1<minmod)|(M0<minmod)
        print("computing alpha map for model pixels above %.1e Jy (based on max DR setting of %g)"%(minmod,MaxDR), file=log)
        M0[mask]=minmod
        M1[mask]=minmod
        alpha = (np.log(M0)-np.log(M1))/(np.log(f0/f1))
        alpha[mask] = 0

        # Np=1000
        # indx,indy=np.int64(np.random.rand(Np)*M0.shape[0]),np.int64(np.random.rand(Np)*M0.shape[1])
        # med=np.median(np.abs(M0[:,:,indx,indy]))
        # Mask=((M1>100*med)&(M0>100*med))
        # alpha=np.zeros_like(M0)
        # alpha[Mask]=(np.log(M0[Mask])-np.log(M1[Mask]))/(np.log(f0/f1))
        return alpha

        
    def RemoveNegComponants(self):
        print("Cleaning model dictionary from negative components", file=log)
        stop
        ModelImage=self.GiveModelImage(self.DicoSMStacked["RefFreq"])[0,0]
        
        Lx,Ly=np.where(ModelImage<0)

        for icomp in range(Lx.size):
            key=Lx[icomp],Ly[icomp]
            try:
                del(self.DicoSMStacked["Comp"][key])
            except:
                print("  Component at (%i, %i) not in dict "%key, file=log)

    def FilterNegComponants(self,box=20,sig=3,RemoveNeg=True):
        stop
        print("Cleaning model dictionary from negative components with (box, sig) = (%i, %i)"%(box,sig), file=log)
        
        print("  Number of components before filtering: %i"%len(self.DicoSMStacked["Comp"]), file=log)
        ModelImage=self.GiveModelImage(self.DicoSMStacked["RefFreq"])[0,0]
        
        Min=scipy.ndimage.filters.minimum_filter(ModelImage,(box,box))
        Min[Min>0]=0
        Min=-Min

        if RemoveNeg==False:
            Lx,Ly=np.where((ModelImage<sig*Min)&(ModelImage!=0))
        else:
            print("  Removing neg components too", file=log)
            Lx,Ly=np.where( ((ModelImage<sig*Min)&(ModelImage!=0)) | (ModelImage<0))

        for icomp in range(Lx.size):
            key=Lx[icomp],Ly[icomp]
            try:
                del(self.DicoSMStacked["Comp"][key])
            except:
                print("  Component at (%i, %i) not in dict "%key, file=log)
        print("  Number of components after filtering: %i"%len(self.DicoSMStacked["Comp"]), file=log)



    def CleanMaskedComponants(self,MaskName,InvertMask=False):
        stop
        print("Cleaning model dictionary from masked components using %s [%i components]"%(MaskName,len(self.DicoSMStacked["Comp"])), file=log)

        im=image(MaskName)
        MaskArray=im.getdata()[0,0].T[::-1]
        if InvertMask:
            print("  Inverting the mask", file=log)
            MaskArray=1-MaskArray
        # copy keys to avoid py3 error
        iterkeys=list(self.DicoSMStacked["Comp"].keys())
        for (x,y) in iterkeys:
            if MaskArray[x,y]==0:
                del(self.DicoSMStacked["Comp"][(x,y)])
        print("  There are %i components left"%len(self.DicoSMStacked["Comp"]), file=log)

                
    def ToNPYModel(self,FitsFile,SkyModel,BeamImage=None):
        #R=ModRegFile.RegToNp(PreCluster)
        #R.Read()
        #R.Cluster()
        #PreClusterCat=R.CatSel
        #ExcludeCat=R.CatExclude


        AlphaMap=self.GiveSpectralIndexMap()
        ModelMap=self.GiveModelImage()
        nch,npol,_,_=ModelMap.shape

        for ch in range(nch):
            for pol in range(npol):
                ModelMap[ch,pol]=ModelMap[ch,pol][::-1]#.T
                AlphaMap[ch,pol]=AlphaMap[ch,pol][::-1]#.T

        if BeamImage is not None:
            ModelMap*=(BeamImage)


        im=image(FitsFile)
        pol,freq,decc,rac=im.toworld((0,0,0,0))

        Lx,Ly=np.where(ModelMap[0,0]!=0)
        
        X=np.array(Lx)
        Y=np.array(Ly)

        #pol,freq,decc1,rac1=im.toworld((0,0,1,0))
        dx=abs(im.coordinates().dict()["direction0"]["cdelt"][0])

        SourceCat=np.zeros((X.size,),dtype=[('Name','|S200'),('ra',float),('dec',float),('Sref',float),('I',float),('Q',float),\
                                           ('U',float),('V',float),('RefFreq',float),('alpha',float),('ESref',float),\
                                           ('Ealpha',float),('kill',int),('Cluster',int),('Type',int),('Gmin',float),\
                                           ('Gmaj',float),('Gangle',float),("Select",int),('l',float),('m',float),("Exclude",bool),
                                           ("X",np.int32),("Y",np.int32)])
        SourceCat=SourceCat.view(np.recarray)

        IndSource=0

        SourceCat.RefFreq[:]=self.DicoSMStacked["RefFreq"]
        _,_,nx,ny=ModelMap.shape
        
        for iSource in range(X.shape[0]):
            x_iSource,y_iSource=X[iSource],Y[iSource]
            _,_,dec_iSource,ra_iSource=im.toworld((0,0,y_iSource,x_iSource))
            SourceCat.ra[iSource]=ra_iSource
            SourceCat.dec[iSource]=dec_iSource
            SourceCat.X[iSource]=(nx-1)-X[iSource]
            SourceCat.Y[iSource]=Y[iSource]
            
            #print self.DicoSMStacked["Comp"][(SourceCat.X[iSource],SourceCat.Y[iSource])]
            # SourceCat.Cluster[IndSource]=iCluster
            Flux=ModelMap[0,0,x_iSource,y_iSource]
            Alpha=AlphaMap[0,0,x_iSource,y_iSource]
            # print iSource,"/",X.shape[0],":",x_iSource,y_iSource,Flux,Alpha
            SourceCat.I[iSource]=Flux
            SourceCat.alpha[iSource]=Alpha


        SourceCat=(SourceCat[SourceCat.ra!=0]).copy()
        np.save(SkyModel,SourceCat)
        self.AnalyticSourceCat=ClassSM.ClassSM(SkyModel)

    def DelAllComp(self):
        stop
        for key in self.DicoSMStacked["Comp"].keys():
            del(self.DicoSMStacked["Comp"][key])


    def PutBackSubsComps(self):
        stop
        #if self.GD["Data"]["RestoreDico"] is None: return

        SolsFile=self.GD["DDESolutions"]["DDSols"]
        if not(".npz" in SolsFile):
            Method=SolsFile
            ThisMSName=reformat.reformat(os.path.abspath(self.GD["Data"]["MS"]),LastSlash=False)
            SolsFile="%s/killMS.%s.sols.npz"%(ThisMSName,Method)
        DicoSolsFile=np.load(SolsFile)
        SourceCat=DicoSolsFile["SourceCatSub"]
        SourceCat=SourceCat.view(np.recarray)
        #RestoreDico=self.GD["Data"]["RestoreDico"]
        RestoreDico=DicoSolsFile["ModelName"][()][0:-4]+".DicoModel"
        
        print("Adding previously subtracted components", file=log)
        ModelMachine0=ClassModelMachine(self.GD)

        
        ModelMachine0.FromFile(RestoreDico)

        

        _,_,nx0,ny0=ModelMachine0.DicoSMStacked["ModelShape"]
        
        _,_,nx1,ny1=self.ModelShape
        dx=nx1-nx0

        

        for iSource in range(SourceCat.shape[0]):
            x0=SourceCat.X[iSource]
            y0=SourceCat.Y[iSource]
            
            x1=x0+dx
            y1=y0+dx
            
            if not((x1,y1) in self.DicoSMStacked["Comp"].keys()):
                self.DicoSMStacked["Comp"][(x1,y1)]=ModelMachine0.DicoSMStacked["Comp"][(x0,y0)]
            else:
                self.DicoSMStacked["Comp"][(x1,y1)]+=ModelMachine0.DicoSMStacked["Comp"][(x0,y0)]
                
