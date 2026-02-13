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
import numpy as np
from scipy.optimize import minimize

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
    DicoName="Major2.DicoModel"
    D=MyPickle.Load(DicoName)
    MM=DDFacet.Imager.SSD3.ClassModelMachineSSD.ClassModelMachine(D["GD"])
    MM.FromFile(DicoName)
    
    
    
    R=MM.PastModels_Resid[-1]

    import pyregion
    L=pyregion.open("ds9_fit.reg") # saved in ds9 "Image" coordimate system format
    L=pyregion.open("ds9_fitNeg.reg") # saved in ds9 "Image" coordimate system format
    L=pyregion.open("ds9_fitRem.reg") # saved in ds9 "Image" coordimate system format
    L=pyregion.open("ds9_fitMeanNeg.reg") # saved in ds9 "Image" coordimate system format
    
    _,_,nx,ny=R.shape
    Lxy=[(int(nx-r0.coord_list[0]),int(r0.coord_list[1])) for r0 in L]
    #Lxy=Lxy[0:1]
    #R0=R[0,0]
    #Show(R0,Lxy)
    #pylab.show()
    
    FPM=ClassFitPreviousModels(MM)
    FPM.avgWeighted(Lxy=Lxy)
    FPM.PlotRegions()


# ###########################################

SERIAL=False
#SERIAL=True




def nnls_fit(x,y,weights,nterms,x0,LymMin,LymMax,beta0=None):
    
    # inputs
    x = np.asarray(x)
    y = np.asarray(y)
    w = np.asarray(weights).copy()
    nterms = nterms  # number of polynomial terms (increasing powers)
    # xc = np.asarray(xc)  # constraint x_i
    # yc = np.asarray(yc)  # constraint y_i
    
    # design matrices
    A = np.vander(x, nterms, increasing=True)
    Ac = np.vander(x0, nterms, increasing=True)



    
    # weighted least squares objective
    def obj(beta):
        ym=A @ beta
        r = ym - y
        chi2=np.sum(w * r * r)
        
        ym0=Ac @ beta
        if np.count_nonzero(ym0>LymMax):
            chi2*=10
        if np.count_nonzero(ym0<LymMin):
            chi2*=10
        return chi2

    # constraints: Ac @ beta <= yc
    # cons = [{'type': 'ineq',
    #          'fun': lambda b, Ac=Ac[i], yc=yc[i]: yc - Ac @ b}
    #         for i in range(len(yc))]

    # solve
    if beta0 is None: beta0 = np.zeros(nterms)
    res = minimize(obj, beta0)#, constraints=cons)
    beta = res.x
    
    return beta



class ClassFitPreviousModels():

    def __init__(self,MM):
        self.MM=MM
        self.GD=MM.GD

    def startWorker(self):
        logger.setSilent(["AsyncProcessPool"])
        
        self.APP=DDFacet.Other.AsyncProcessPool.initNew(Name="APP_FitModel",
                                                           ncpu=self.GD["Parallel"]["NCPU"],
                                                           affinity="disable",#self.GD["Parallel"]["Affinity"],
                                                           #parent_affinity=self.GD["Parallel"]["MainProcessAffinity"],
                                                           #verbose=self.GD["Debug"]["APPVerbose"],
                                                           #pause_on_start=self.GD["Debug"]["PauseWorkers"]
                                                           )
        self.APP.registerJobHandlers(self)
        self.APP.startWorkers()
        self.APP.awaitWorkerStart()
        logger.setLoud(["AsyncProcessPool"])

    def stopWorker(self):
        logger.setSilent(["AsyncProcessPool"])
        self.APP.terminate()
        self.APP.shutdown()
        del(self.APP)
        logger.setLoud(["AsyncProcessPool"])
        
    def avgWeighted(self,Lxy=None):
        MM=self.MM
        
        NTerms=MM.NParam

        freqs=np.linspace(0.5*MM.RefFreq,2.*MM.RefFreq,NTerms+1)
        freqs=self.MM.GridFreqs
        self.freqs=freqs
        nch=self.freqs.shape
        CurrentModel=self.MM.DicoSMStacked["Comp"]["Vals"]
        _,nx,ny=CurrentModel.shape
        CurrentModel=CurrentModel.reshape((1,self.MM.NParam,nx,ny))
        M=np.concatenate([np.array(MM.PastModels),CurrentModel],axis=0)
        
        NModels,NTerms,nx,ny=M.shape
        self.NModels=NModels
        self.NTerms=NTerms

        self.Lxy=Lxy
        if Lxy is not None:
            indx,indy=np.array(Lxy).T
            #self.indxy=(indx[0],indy[0])

        
        NFreqs=freqs.size
        LModel=np.zeros((NModels,NFreqs,nx,ny),M.dtype)
        for iModel in range(NModels):
            LModel[iModel]=MM.GiveModelImage(FreqIn=freqs,InModelParms=M[iModel])[:,0]


            
        nch,_,nx,ny=self.MM.CurrentResid.shape
        CurrentResid=self.MM.CurrentResid.reshape((1,nch,1,nx,ny))
        
        #Lfreqs=[freqs.copy() for iModel in range(NPastModels)]
        #Lfreqs2=np.array([freqs.copy(),freqs.copy()]).flatten()

        NPastModels=len(MM.PastModels)
        LResid=np.concatenate([np.array(MM.PastModels_Resid)[-NPastModels:],CurrentResid])
        RMS0=scipy.stats.median_abs_deviation(LResid,axis=None,scale="normal")
        
        NBands=LResid.shape[1]
        _,NBands,_,nx,ny=LResid.shape
        
        W=np.zeros((NModels,NBands,nx,ny),np.float32)
        #W=np.abs(np.random.randn(NModels,NBands,nx,ny))
        for iResid in range(NModels):
            for iBand in range(NBands):
                RMS=scipy.stats.median_abs_deviation(LResid[iResid,iBand],axis=None,scale="normal")
                aW=np.abs(LResid[iResid,iBand,0])
                f=.1
                aW[aW<f*RMS]=f*RMS
                W[iResid,iBand,:,:]=(1./aW)
        Wmin=np.min(np.min(W,axis=0),axis=0).reshape((1,1,nx,ny))
        W=W/Wmin
        
        #     Wmin=np.min(W[iResid],axis=0).reshape((1,nx,ny))
        #     W[iResid]=W[iResid]/Wmin
        # # Wmin=np.min(np.min(W,axis=0),axis=0).reshape((1,1,nx,ny))
        # # W=W/Wmin
        # for iBand in range(NBands):
        #     Wmin=np.min(W[:,iBand],axis=0).reshape((1,1,nx,ny))
        #     W[:,iBand]=W[:,iBand]/Wmin
        
        self.W=W
        
        
        
        
        #s=1.*(freqs/MM.RefFreq)**(-1)
        #Model[:,0,indx[0],indy[0]]=s[:]

        factRMS=1e-3
        LModel=fMyScale(LModel,RMS=RMS0*factRMS)

        

        self.LModel=LModel
            
        # Model from initialisation
        
        # xx=freqs
        # self.xx=xx
        #self.Lfreqs=Lfreqs
        #self.Lfreqs2=Lfreqs2
    
        # V = np.vander(np.log10(Lfreqs/MM.RefFreq), N=NTerms, increasing=True)  # Vandermonde matrix
        # self.V=V
        
        # V2 = np.vander(np.log10(Lfreqs2/MM.RefFreq), N=NTerms, increasing=True)  # Vandermonde matrix
        # self.V2=V2

        self.x0=np.log10(freqs/MM.RefFreq)
        self.x=np.array([np.log10(freqs/MM.RefFreq) for iModels in range(self.NModels)]).ravel()
        self.V0 = np.vander(self.x0, N=NTerms, increasing=True)  # Vandermonde matrix
        self.Vx = np.vander(self.x, N=NTerms, increasing=True)  # Vandermonde matrix
            

        
        # M1=np.zeros((NTerms,nx,ny),np.float32)

        self.ShmName="DicoFitModel"
        DicoFitModel  = shared_dict.attach(self.ShmName)
        DicoFitModel.addSharedArray("M1", (NTerms,nx,ny), np.float32)

        #self._runFit(indx[0],indy[0])
        
        self.startWorker()
        APP=self.APP
        runFit=self._runFit_ExceptionFree
        if SERIAL: runFit=self._runFit
        #runFit=self._runFit
        for ii in range(nx): #[indx]:
            APP.runJob("Fit.%i"%(ii),
                       runFit,
                       args=(ii,), serial=SERIAL)
        APP.awaitJobResults("Fit.*", progress="Fit past models")
        self.stopWorker()
        
        DicoFitModel.reload()
        M1=DicoFitModel["M1"].copy()
        DicoFitModel.delete()

        
        self.M1s=M1.copy()
        SGN=np.sign(M1[0])
        M1[0,:,:]=inv_fMyScale(M1[0,:,:],RMS=RMS0*factRMS)
        M1[1:,:,:]*=SGN

        self.M1=M1
        self.RMS0=RMS0
        self.factRMS=factRMS
        self.LModel=LModel
        self.LResid=LResid
        self.W=W
        return M1
    
        ###########################


    def PlotRegions(self):
        for ii,jj in self.Lxy:
            self.PlotSingleRegions(ii,jj)
            
    def PlotSingleRegions(self,ii,jj):
        MM=self.MM
        freqs=self.freqs
        M1=self.M1
        M1s=self.M1s
        RMS0=self.RMS0
        factRMS=self.factRMS
        LModel=self.LModel
        LResid=self.LResid
        W=self.W
        Model1=MM.GiveModelImage(FreqIn=freqs,InModelParms=M1)
        Model1=fMyScale(Model1,RMS=RMS0*factRMS)
        yy1=Model1[:,0,ii,jj]
        
        Lyy=LModel[:,:,ii,jj]
        import pylab
        pylab.figure("images")
        pylab.clf()
        
        NFreqs=freqs.size
        NModels=LModel.shape[0]
        
        xc,yc=ii,jj
        iPlot=1
        dx=30
        DRMS={}
        for iFreq in range(NFreqs):
            DRMS[iFreq]=np.std(LResid[0,iFreq])


            
        for iModel in range(NModels):
            for iFreq in range(NFreqs):
                pylab.subplot(NModels,NFreqs,iPlot); iPlot+=1
                rms=DRMS[iFreq]
                v0,v1=-5*rms,30*rms
                IM=LResid[iModel,iFreq,0,xc-dx:xc+dx+1,yc-dx:yc+dx+1]
                pylab.imshow(IM,interpolation="nearest",vmin=v0,vmax=v1)
                pylab.title("(cycle, freq)\n= (%i, %i)"%(iModel,iFreq))
        pylab.draw()
        
        pylab.figure("Fits")
        pylab.clf()
        x0=self.x0
        rr=LResid[:,:,0,ii,jj]
        rr0=np.abs(rr).min()

        for iModel in range(NModels):
            yy=LModel[iModel,:,ii,jj].ravel()
            rr=np.abs(LResid[iModel,:,0,ii,jj]).ravel()/rr0
            rr=W[iModel,:,ii,jj]
            print("Weights",iModel,W[iModel,:,ii,jj])
            # yy=fMyScale(yy,RMS=RMS0*factRMS)
            pylab.scatter(x0,yy,marker="+",s=rr*40,label="ModelIn#%i"%iModel)
        pylab.scatter(x0,yy1,color="red", facecolors='none', edgecolors='r',s=80,label="ModelOut")

        yy1=M1s[:,ii,jj]
        yy1m=self.V0 @ ( yy1)
        # yy1m=fMyScale(yy1m,RMS=RMS0*factRMS)
        
        x=self.x
        Y=LModel[:,:,ii,jj].ravel()
        LymMax=LModel[:,:,ii,jj].max(axis=0).ravel()
        LymMin=LModel[:,:,ii,jj].min(axis=0).ravel()
        x0=self.x0
        weights=W[:,:,ii,jj].ravel()
        yy1_Constr=nnls_fit(x,Y,weights,self.NTerms,self.x0,LymMin,LymMax)
        yy1m_Constr=self.V0 @ yy1_Constr

        
        pylab.scatter(x0,yy1m,marker="x",label="PolyModel",color="red")
        pylab.scatter(x0,yy1m_Constr,marker="+",label="PolyModel Constr",color="red")
        pylab.legend()
        pylab.draw()
        pylab.show()
        pylab.pause(0.1)
        pylab.close("all")
        ##################
        
        return M1

    def _runFit_ExceptionFree(self,*args,**kwargs):
        try:
            return self._runFit(*args,**kwargs)
        except Exception as e:
            print("Failed to fit %s: %s"%(str(args[0]),str(e)))
            

            
    def _runFit(self,ii,jj=None):
        WAll=self.W
        freqs=self.freqs
        # Lfreqs=self.Lfreqs
        # Lfreqs2=self.Lfreqs2
        LModel=self.LModel
        #V=self.V
        #V2=self.V2
        #indx,indy=self.indxy
        def weighted_polyfit(x, y, W, degree):
            V = self.Vx
            WW = np.diag(W)
            coeffs = np.linalg.lstsq(WW @ V, WW @ y, rcond=None)[0]
            return coeffs#[::-1]
        
        M1  = shared_dict.attach(self.ShmName)["M1"]
        nx,ny=M1.shape[-2:]
        _,NFreqs,nx,ny=LModel.shape
        if jj is None:
            Lny=range(ny)
        else:
            Lny=[jj]
        for jj in Lny:
            #x=[self.freqs for iPastModels in range(self.NPastModels)]
            x=self.x
            y=LModel[:,:,ii,jj].flatten()#[LModel[iModels,:,ii,jj] for iModels in range(self.NModels)]
            w=WAll[:,:,ii,jj].copy()#[WAll[iModels,:,ii,jj] for iModels in range(self.NModels)]
            y=np.array([LModel[iModels,:,ii,jj] for iModels in range(self.NModels)])
            w=np.array([WAll[iModels,:,ii,jj] for iModels in range(self.NModels)])
            
            mask=(np.array(y).reshape((self.NModels,NFreqs))==0).all(axis=1)
            mask=mask.reshape((self.NModels,1))
            Nmask=np.count_nonzero(mask==1)
            NNonMasked=np.count_nonzero(mask==0)
            
            if NNonMasked==0: continue

            
            
            if Nmask>1:
                mask.flat[0]=0 # keep one zero model
            maskf=mask*np.ones((1,NFreqs),bool)
            
            y.reshape((self.NModels,NFreqs))
            
            y=np.array(y).ravel()

            w=np.array(w)
            w[maskf]=0
            
            y=y.ravel()
            w=w.ravel()
            c=weighted_polyfit(x, y, w, self.NTerms)

            M1[:,ii,jj] = c[:]

            ym=self.V0 @ c
            LymMax=LModel[:,:,ii,jj].max(axis=0).ravel()
            LymMin=LModel[:,:,ii,jj].min(axis=0).ravel()
            CondExcess=((np.count_nonzero(ym > LymMax) or np.count_nonzero(ym < LymMin)))
            
            #print("\n\n FDSLOJSDFSDFLJ CondExcess c1a:",CondExcess,c.ravel(),mask.ravel())
            # test is all the non-zero models are the same
            # otherwise make nnls_fitto return 0
            if NNonMasked>1 and not CondExcess:
                indNonZeroModel=np.where(mask.ravel()==0)[0]
                ysel=y.reshape((self.NModels,NFreqs))[indNonZeroModel,:]
                yres=np.abs(ysel-ysel[0:1,:])
                if yres.max()==0: continue
                            
            #print("\n\n FDSLOJSDFSDFLJ c1b:",c)
            if CondExcess:
                #weights=self.W[:,:,ii,jj].ravel()
                w1=w.copy()
                w1.fill(1)
                weights=w
                beta0=weighted_polyfit(x, y, w1, self.NTerms)
                c=nnls_fit(self.x,y,weights,self.NTerms,self.x0,LymMin,LymMax,beta0=beta0)
            #print("\n\n FDSLOJSDFSDFLJ c2:",c)


                
            
            M1[:,ii,jj] = c[:]
            
