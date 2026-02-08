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
from DDFacet.Array import shared_dict
import pylab

def fMyScale(x,RMS=1.): return np.arcsinh((x/RMS)/2)/np.log(10)
def inv_fMyScale(x,RMS=1.): return RMS*np.sinh(x*np.log(10))*2

import DDFacet.Other.AsyncProcessPool

import DDFacet.Imager.SSD3.ClassModelMachineSSD

def Show(R0,Lxy=[]):
    RMS0=scipy.stats.median_abs_deviation(R0,axis=None,scale="normal")
    v0=-5*RMS0
    v0=R0[0,0].min()
    v1=5*RMS0
    pylab.imshow(R0.T,vmin=v0,vmax=v1,origin='lower')
    for (x,y) in Lxy:
        print(R0[x,y])
        pylab.scatter(x,y,color="red", facecolors='none', edgecolors='r',s=80)
    


def testFit():
    DicoName="SSD3.0Gen_Orieux.03.DicoModel"
    D=MyPickle.Load(DicoName)
    MM=DDFacet.Imager.SSD3.ClassModelMachineSSD.ClassModelMachine(D["GD"])
    MM.FromFile(DicoName)
    
    
    
    R=MM.PastModels_Resid[-1]

    import pyregion
    L=pyregion.open("ds9_fit.reg") # saved in ds9 "Image" coordimate system format
    _,_,nx,ny=R.shape
    Lxy=[(int(nx-r0.coord_list[0]),int(r0.coord_list[1])) for r0 in L]
    
    #R0=R[0,0]
    #Show(R0,Lxy)
    #pylab.show()
    
    
    FPM=ClassFitPreviousModels(MM)
    
    return FPM.avgWeighted(Lxy=Lxy)

SERIAL=False
SERIAL=True

class ClassFitPreviousModels():

    def __init__(self,MM):
        self.MM=MM
        self.GD=MM.GD

    def startWorker(self):
        self.APP=DDFacet.Other.AsyncProcessPool.initNew(Name="APP_FitModel",
                                                           ncpu=self.GD["Parallel"]["NCPU"],
                                                           affinity="disable",#self.GD["Parallel"]["Affinity"],
                                                           #parent_affinity=self.GD["Parallel"]["MainProcessAffinity"],
                                                           #verbose=self.GD["Debug"]["APPVerbose"],
                                                           #pause_on_start=self.GD["Debug"]["PauseWorkers"]
                                                           )
        self.APP.registerJobHandlers(self)
        self.APP.startWorkers()

        
    def avgWeighted(self,Lxy=None):
        MM=self.MM
        
        NTerms=MM.NParam

        freqs=np.linspace(0.5*MM.RefFreq,2.*MM.RefFreq,NTerms+1)
        freqs=self.MM.GridFreqs
        self.freqs=freqs
        M=np.array(MM.PastModels)
        NPastModels,NTerms,nx,ny=M.shape
        self.NPastModels=NPastModels
        self.NTerms=NTerms

        if Lxy is not None:
            indx,indy=np.array(Lxy).T
            self.indxy=(indx[0],indy[0])

        
        NPastModels,NTerms,nx,ny=M.shape
        NFreqs=freqs.size
        LModel=np.zeros((NPastModels,NFreqs,nx,ny),M.dtype)
        Lfreqs=[]
        for iModel in range(NPastModels):
            LModel[iModel]=MM.GiveModelImage(FreqIn=freqs,InModelParms=M[iModel])[:,0]
            Lfreqs.append(freqs.copy())
        Lfreqs=np.array(Lfreqs).flatten()
        
        Lfreqs2=np.array([freqs.copy(),freqs.copy()]).flatten()

        
        LResid=np.array(MM.PastModels_Resid)[-NPastModels:]
        RMS0=scipy.stats.median_abs_deviation(LResid,axis=None,scale="normal")
        
        NBands=LResid.shape[1]
        _,NBands,_,nx,ny=LResid.shape
        
        W=np.zeros((NPastModels,NBands,nx,ny),np.float32)
        for iResid in range(NPastModels):
            for iBand in range(NBands):
                RMS=scipy.stats.median_abs_deviation(LResid[iResid,iBand],axis=None,scale="normal")
                aW=np.abs(LResid[iResid,iBand,0])
                aW[aW<RMS]=RMS
                W[iResid,iBand,:,:]=1./aW
        self.W=W
        
        #s=1.*(freqs/MM.RefFreq)**(-1)
        #Model[:,0,indx[0],indy[0]]=s[:]

        factRMS=1e-3
        LModel=fMyScale(LModel,RMS=RMS0*factRMS)

        
        CurrentModelParms = self.MM.DicoSMStacked["Comp"]["Vals"][:]
        
        self.CurrentModel=fMyScale(MM.GiveModelImage(FreqIn=freqs,InModelParms=CurrentModelParms)[:,0])

        self.LModel=LModel
            
        # Model from initialisation
        
        Lyy=LModel[:,:,indx,indy]
        xx=freqs
        self.xx=xx
        self.Lfreqs=Lfreqs
        self.Lfreqs2=Lfreqs2
    
        V = np.vander(np.log10(Lfreqs/MM.RefFreq), N=NTerms, increasing=True)  # Vandermonde matrix
        self.V=V
        
        V2 = np.vander(np.log10(Lfreqs2/MM.RefFreq), N=NTerms, increasing=True)  # Vandermonde matrix
        self.V2=V2
        
        # M1=np.zeros((NTerms,nx,ny),np.float32)

        self.ShmName="DicoFitModel"
        DicoFitModel  = shared_dict.attach(self.ShmName)
        DicoFitModel.addSharedArray("M1", (NTerms,nx,ny), np.float32)

        self.startWorker()
        APP=self.APP
        self._runFit(indx,indy)
        
        for ii in range(nx): #[indx]:
            APP.runJob("Fit.%i"%(ii),
                       self._runFit,
                       args=(ii,), serial=SERIAL)
        APP.awaitJobResults("Fit.*", progress="Fit past models")
        
        APP.terminate()
        APP.shutdown()
        del(APP,self.APP)
        
        DicoFitModel.reload()
        M1=DicoFitModel["M1"].copy()
        DicoFitModel.delete()

        
        M1s=M1.copy()
        SGN=np.sign(M1[0])
        M1[0,:,:]=inv_fMyScale(M1[0,:,:],RMS=RMS0*factRMS)
        M1[1:,:,:]*=SGN

        ###########################
        # print(M[:,indx[0],indy[0]])
        # print(M1[:,indx[0],indy[0]])
        # return

        ii,jj=indx,indy
        V=self.V
        # y=LModel.reshape((Lfreqs.size,nx,ny))[:,ii,jj]#.reshape((-1,1))
        # #VTV = V.T @ (W @ V)
        # #VTy = V.T @ (W @ y)
        # VTV = V.T @ ( V)
        # VTy = V.T @ ( y)
        # yyp = np.linalg.solve(VTV, VTy)  # Closed-form solution


        
        Model1=MM.GiveModelImage(FreqIn=freqs,InModelParms=M1)
        Model1=fMyScale(Model1,RMS=RMS0*factRMS)
        yy1=Model1[:,0,indx,indy]
        
        print(Lyy)
        print(yy1)
        import pylab
        pylab.clf()

        rr=LResid[:,:,0,indx,indy]
        rr0=np.abs(rr).min()
        
        for iModel in range(NPastModels):
            yy=LModel[iModel,:,indx,indy].ravel()
            rr=np.abs(LResid[iModel,:,0,indx,indy]).ravel()/rr0
            
            #yy=fMyScale(yy,RMS=RMS0*factRMS)
            pylab.scatter(np.log10(xx),yy,marker="+",label="previous",s=rr*40)
        pylab.scatter(np.log10(xx),yy1,color="red", facecolors='none', edgecolors='r',s=80,label="fit")

        yy1=M1s[:,ii,jj]
        yy1m=V @ ( yy1)
        # yy1m=fMyScale(yy1m,RMS=RMS0*factRMS)
        xxa=np.array(self.Lfreqs)
        pylab.scatter(np.log10(xxa).ravel(),yy1m,marker="x")
        
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)
        stop
        ##################
        
        return M1

    def _runFit(self,ii,jj=None):
        WAll=self.W
        freqs=self.freqs
        Lfreqs=self.Lfreqs
        Lfreqs2=self.Lfreqs2
        LModel=self.LModel
        CurrentModel=self.CurrentModel
        V=self.V
        V2=self.V2


        def weighted_polyfit(x, y, w, degree):
            x = np.asarray(x).flatten()
            y = np.asarray(y).flatten()
            W = np.asarray(w).flatten()
            V = np.vander(x, degree)
            WW = np.diag(W)
            coeffs = np.linalg.lstsq(WW @ V, WW @ y, rcond=None)[0]
            return coeffs
        
        M1  = shared_dict.attach(self.ShmName)["M1"]
        nx,ny=M1.shape[-2:]
        # LModel=np.zeros((NPastModels,NFreqs,nx,ny),M.dtype)
        if jj is None:
            Lny=range(ny)
        else:
            Lny=[jj]
        for jj in Lny:
            
            x=[self.freqs for iPastModels in range(self.NPastModels)]
            y=[LModel[iPastModels,:,ii,jj] for iPastModels in range(self.NPastModels)]
            w=[WAll[iPastModels,:,ii,jj] for iPastModels in range(self.NPastModels)]
            c=weighted_polyfit(x, y, w, self.NTerms)
            stop
            M1[:,ii,jj] = c[:]
            
            # ##########################

            # W=WAll[...,ii,jj].ravel()
            # #W=np.diag((ww.reshape((-1,1))*np.ones((1,freqs.size))).flatten())

            # y=LModel.reshape((Lfreqs.size,nx,ny))[:,ii,jj]#.reshape((-1,1))
            # VTV = V.T @ (W @ V)
            # VTy = V.T @ (W @ y)
            # #VTV = V.T @ ( V)
            # #VTy = V.T @ ( y)

            
            # M1[:,ii,jj] = np.linalg.solve(VTV, VTy)  # Closed-form solution
            # yy1_0=M1[:,ii,jj].copy()
            
            
            # # Average of previous past solutions and new one
            # yy1=M1[:,ii,jj]
            # yy1m=V2 @ ( yy1)
            
            # #yy2=CurrentModel[:,ii,jj]
            # #yy2m=V2 @ ( yy2)
            # yy2m = CurrentModel[:,ii,jj]
            
            # #yym  = np.concatenate([yy1m,yy2m])
            # yy1m[freqs.size:]=yy2m[:]
            
            # VTV2 = V2.T @ ( V2)
            # VTy2 = V2.T @ ( yy1m)
            
            # M1[:,ii,jj] = np.linalg.solve(VTV2, VTy2)  # Closed-form solution 
            # yy1_1=M1[:,ii,jj].copy()
           

            
            
            # ii0,jj0=self.indxy
            # yy1=M1[:,ii,jj]
            
            # if ii==ii0 and jj==jj0:
            #     if np.random.rand(1)[0]>0.01: continue
            #     print("Coefs",ii,jj,yy1)
            #     import pylab
            #     pylab.clf()
            #     yy1m_0=V @ ( yy1_0)
            #     yy1m_1=V @ ( yy1_0)
            #     #yy=LModel[iModel,:,indx,indy]
            #     xx=np.array(self.Lfreqs)
            #     pylab.scatter(np.log10(xx).ravel(),y,marker="+")
            #     pylab.scatter(np.log10(xx).ravel(),yy1m_0,marker="x")
            #     pylab.scatter(np.log10(xx).ravel(),yy1m_1,marker="o")
            #     pylab.draw()
            #     pylab.show(block=False)
            #     pylab.pause(0.1)
            #     #stop
                
            
            
    # def avgSimple(self):
    #     MM=self.MM
        
    #     NTerms=MM.NParam

    #     freqs=np.linspace(0.5*MM.RefFreq,2.*MM.RefFreq,NTerms)
    #     M=np.array(MM.PastModels)
    #     NPastModels,NTerms,nx,ny=M.shape
    
    #     _,indx,indy=np.where(M[:,0]==M[:,0].max())
        
    #     #M[:,indx[0],indy[0]]=np.array([1.,-1.])
    
        
    #     NPastModels,NTerms,nx,ny=M.shape
    #     NFreqs=freqs.size
    #     LModel=np.zeros((NPastModels,NFreqs,nx,ny),M.dtype)
    #     Lfreqs=[]
    #     for iModel in range(NPastModels):
    #         LModel[iModel]=MM.GiveModelImage(FreqIn=freqs,InModelParms=M[iModel])[:,0]
    #         Lfreqs.append(freqs.copy())
            
    #     Lfreqs=np.array(Lfreqs).flatten()
        
    #     Resid=MM.PastModels_Resid[-1]
    #     RMS=scipy.stats.median_abs_deviation(Resid,axis=None,scale="normal")
    
    #     #s=1.*(freqs/MM.RefFreq)**(-1)
    #     #Model[:,0,indx[0],indy[0]]=s[:]
        
    #     LModel=fMyScale(LModel,RMS=RMS)
        
        
        
        
    #     Lyy=LModel[:,:,indx,indy]
    #     xx=freqs
    
        
    #     V = np.vander(np.log10(Lfreqs/MM.RefFreq), N=NTerms, increasing=True)  # Vandermonde matrix
    
    #     VTV = V.T @ V
    #     y=LModel.reshape((Lfreqs.size,nx*ny))
    #     VTy = V.T @ y
    #     coeffs = np.linalg.solve(VTV, VTy)  # Closed-form solution
    #     M1=coeffs.reshape((NTerms,nx,ny))
    #     M1[0,:,:]=inv_fMyScale(M1[0,:,:],RMS)
    
    
        
    #     # print(M[:,indx[0],indy[0]])
    #     # print(M1[:,indx[0],indy[0]])
    #     # return
        
    #     # Model1=MM.GiveModelImage(FreqIn=freqs,InModelParms=M1)
    #     # Model1=fMyScale(Model1,RMS=RMS)
    #     # yy1=Model1[:,0,indx,indy]
    #     # print(Lyy)
    #     # print(yy1)
    #     # import pylab
    #     # pylab.clf()
    #     # for iModel in range(NPastModels):
    #     #     yy=LModel[iModel,:,indx,indy]
    #     #     pylab.scatter(np.log10(xx),yy,marker="+")
    #     # pylab.scatter(np.log10(xx),yy1)
    #     # pylab.draw()
    #     # pylab.show(block=False)
    
    #     return M1
    
