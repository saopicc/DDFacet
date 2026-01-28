from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import collections
import random

import numpy as np
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import logger

log=logger.getLogger("ClassArrayMethodSSD")
import multiprocessing
import psutil

from DDFacet.Imager.SSD3 import ClassConvMachine
import time
from scipy.stats import chi2

from deap import tools

log= logger.getLogger("ClassArrayMethodSSD")


from DDFacet.Imager.SSD3.ClassParamMachine import ClassParamMachine
from DDFacet.ToolsDir.GeneDist import ClassDistMachine
from DDFacet.Imager.SSD3 import ClassMutate
import DDFacet.Other.AsyncProcessPool

SERIAL=False

        
class ClassArrayMethodSSD():
    def __init__(self,Dirty,PSF,ListPixParms,ListPixData,FreqsInfo,GD=None,
                 PixVariance=1.e-2,IslandBestIndiv=None,WeightFreqBands=None,iFacet=0,
                 island_dict=None,
                 iIsland=0,
                 ParallelMode=None,
                 NCPU=None):
        self.GD=GD
        self.ParallelFitness= ( (ParallelMode=="PerIsland") and (self.GD["GAClean"]["NMaxGen"]>0) )
        self.iFacet=iFacet
        self.WeightFreqBands=WeightFreqBands
        self.iIsland=iIsland

        self._island_dict = island_dict

        self.PSF = PSF

        self.IslandBestIndiv=IslandBestIndiv
        self.GD=GD
        self.NCPU=NCPU
        if NCPU==None:
            self.NCPU=int(self.GD["Parallel"]["NCPU"] or psutil.cpu_count())

        self.BestChi2=1.
        self.EntropyMinMax=None
        
        _,_,nx,_=Dirty.shape
        
        self.ListPixParms=ListPixParms
        self.ListPixData=ListPixData
        self.NPixListParms=len(ListPixParms)
        self.NPixListData=len(ListPixData)
        
        self.ConvMode="Matrix"
        if self.NPixListData>self.GD["SSDClean"]["ConvFFTSwitch"]:
            self.ConvMode="FFT"

        self.ConvMachine=ClassConvMachine.ClassConvMachine(PSF,ListPixParms,ListPixData,self.ConvMode)
        
        
        
        self.GD=GD
        self.WeightMaxFunc=collections.OrderedDict()
        #self.WeightMaxFunc["Chi2"]=1.

        for key in self.GD["SSDClean"]["SSDCostFunc"]:
            self.WeightMaxFunc[key]=1.


        #self.WeightMaxFunc["BIC"]=1.
        
        # self.WeightMaxFunc["MinFlux"]=100.
        # print("FSDLSDFLJFSDLSDFJ")
        # print("FSDLSDFLJFSDLSDFJ")
        # print("FSDLSDFLJFSDLSDFJ")
        # del(self.WeightMaxFunc["MinFlux"])
        
        #self.WeightMaxFunc["L1"]=1.
        #self.WeightMaxFunc["L0"]=1.
        
        self.MaxFunc=self.WeightMaxFunc.keys()
        
        self.NFuncMin=len(self.MaxFunc)
        self.WeightsEA=[1.]#*len(self.MaxFunc)
        #self.WeightsEA=[1.]*len(self.MaxFunc)
        self.MinVar=np.array([0.01,0.01])
        self.PixVariance=PixVariance
        self.FreqsInfo=FreqsInfo
        
        self.NFreqBands,self.npol,self.NPixPSF_x,self.NPixPSF_y=PSF.shape
        self.PM=ClassParamMachine(ListPixParms,ListPixData,FreqsInfo,
                                  NOrderPoly=GD["SSD3"]["PolyFreqOrder"],
                                  SolveParamType=GD["SSD3"]["SolvePars"])
        self.PM.setFreqs(FreqsInfo)
        self.ConvMachine.setParamMachine(self.PM)
        
        self.NParms=self.NPixListParms*self.PM.NParam
        self.DataTrue=None
        self.MutMachine=ClassMutate.ClassMutate(self.PM)

        self.APP_Fitness=None
        self.SetDirtyArrays(Dirty)
        
    def startWorkers(self):
        if not self.ParallelFitness: return
        logger.setSilent(["AsyncProcessPool"])
        self.APP_Fitness=DDFacet.Other.AsyncProcessPool.initNew(Name="APP_Fitness_%i"%self.iIsland,
                                                                ncpu=self.GD["Parallel"]["NCPU"],
                                                                affinity="disable",
                                                                silent_warning=True,
                                                                )
        self.APP_Fitness.registerJobHandlers(self)
        self.APP_Fitness.startWorkers()
        self.APP_Fitness.awaitWorkerStart()
        logger.setLoud(["AsyncProcessPool"])


    def stopWorkers(self):
        if self.APP_Fitness is None: return
        logger.setSilent(["AsyncProcessPool"])
        self.APP_Fitness.terminate()
        self.APP_Fitness.shutdown()
        del(self.APP_Fitness)
        self.APP_Fitness=None
        logger.setLoud(["AsyncProcessPool"])
        
    def __del__(self):

        if "Population" in self._island_dict:
            self._island_dict.delete_item("Population")
        if self.APP_Fitness is not None:
            self.stopWorker()
            
    def SetDirtyArrays(self,Dirty):
        print("SetConvMatrix", file=log)
        PSF=self.PSF
        NPixPSF_x,NPixPSF_y=PSF.shape[-2:]
        if self.ListPixData is None:
            x,y=np.mgrid[0:NPixPSF_x:1,0:NPixPSF_y:1]
            self.ListPixData=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()
        if self.ListPixParms is None:
            x,y=np.mgrid[0:NPixPSF_x:1,0:NPixPSF_y:1]
            self.ListPixParms=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()

        self.DirtyArray=np.zeros((self.NFreqBands,1,self.NPixListData),np.float32)
        xc=NPixPSF_x//2
        yc=NPixPSF_y//2

        x0,y0=np.array(self.ListPixData).T
        for iBand in range(self.NFreqBands):
            self.DirtyArray[iBand,0,:]=Dirty[iBand,0,x0,y0]
        self.ResidArray=self.DirtyArray.copy()
        ALPHA=1.

        import scipy.special
        d=self.DirtyArray[:,0,:].ravel()
        x=np.linspace(-10,10,1000)
        f=0.5*(1.+scipy.special.erf(x/np.sqrt(2.)))
        n=d.size
        F=1.-(1.-f)**n
        ratio=np.abs(np.interp(0.5,F,x))
        EstimatedStdFromMin=np.abs(np.min(d))/ratio
        EstimatedStdFromMax=np.abs(np.max(d))/ratio
        self.EstimatedStdFromResid=np.max([EstimatedStdFromMin,EstimatedStdFromMax])


        if (self.IslandBestIndiv is not None):
            S=self.PM.ArrayToSubArray(self.IslandBestIndiv,"Poly0")
            if np.max(np.abs(S))>0:
                AddArray=self.ToConvArray(self.IslandBestIndiv,OutMode="Data")
                # if np.max(self.IslandBestIndiv)!=0:
                #     Gain=1.-np.max(np.abs(self.DirtyArray))/np.max(np.abs(AddArray))
                #     Gain=np.min([1.,Gain])
                #     self.Gain=np.max([.3,Gain])
                ind=np.where(AddArray==np.max(np.abs(AddArray)))
                
                # print "ind",ind
                if ind[0].size==0:
                    ALPHA=1.
                else:
                    R=self.DirtyArray[ind].flat[0]
                    D=AddArray[ind].flat[0]
                    # print "R",R
                    # print "D",D
                    ALPHA=(1.-R/D)
                    ALPHA=np.max([1.,ALPHA])
                # ALPHA=1
                self.ALPHA=ALPHA
                # print("ALPHA=",self.ALPHA)
                # print("ALPHA=",self.ALPHA)
                # print("ALPHA=",self.ALPHA)
                
                
                if self.GD["SSDClean"]["ArtifactRobust"]:
                    self.DirtyArray/=self.ALPHA
                    
                self.DirtyArray+=AddArray
        
        self.DirtyArrayMean=np.mean(self.DirtyArray,axis=0).reshape((1,1,self.NPixListData))
        self.DirtyArrayAbsMean=np.mean(np.abs(self.DirtyArray),axis=0).reshape((1,1,self.NPixListData))
        self.DirtyArrayParms=np.zeros((self.NFreqBands,1,self.NPixListParms),np.float32)

        x0,y0=np.array(self.ListPixParms).T
        for iBand in range(self.NFreqBands):
            self.DirtyArrayParms[iBand,0,:]=Dirty[iBand,0,x0,y0]

        if self.IslandBestIndiv is not None:
            self.DirtyArrayParms+=self.ToConvArray(self.IslandBestIndiv.reshape((self.PM.NParam,self.NPixListParms)),OutMode="Parms")

        self.DirtyArrayParmsMean=np.mean(self.DirtyArrayParms,axis=0).reshape((1,1,self.NPixListParms))
        self.DicoData={"DirtyArrayParmsMean":self.DirtyArrayParmsMean}
        self.MutMachine.setData(self.DicoData)


    def ToConvArray(self,V,OutMode="Data",Noise=False):
        A=self.PM.GiveModelArray(V)
        if Noise is not False:
            A+=np.random.randn(*A.shape)*Noise
        A=self.ConvMachine.Convolve(A,OutMode=OutMode)
        return A




    def DeconvCLEAN(self,gain=0.1,StopThFrac=0.01,NMaxIter=20000):

        PSF=self.PSF#/np.max(self.PSF)

        if False:#self.ConvMachine.ConvMode=="Matrix" or  self.ConvMachine.ConvMode=="Vector":
            A=self.DirtyArrayParmsMean.copy()#.reshape((1,1,self.NPixListParms))
            SModelArray=np.zeros_like(A.flatten())
            ArrayMode="Array"
        else:
            Asq=self.PM.ModelToSquareArray(self.DirtyArrayParms.copy(),TypeInOut=("Parms","Parms"))
            _,npol,NPix_x,NPix_y=Asq.shape
            A=np.mean(Asq,axis=0).reshape((NPix_x,NPix_y))
            Mask=(A==0)
            _,_,NPixPSF_x,NPixPSF_y=PSF.shape
            PSFMean=np.mean(PSF,axis=0).reshape((NPixPSF_x,NPixPSF_y))
            ArrayMode="Image"
            xcPSF=NPixPSF_x//2
            xcDirty=NPix_x//2
            ycPSF=NPixPSF_y//2
            ycDirty=NPix_y//2
            SModelArray=np.zeros_like(A)

        MaxA=np.max(A)

        Alpha=0
        # if self.NFreqBands>1:
        #     FluxFreq=np.sum(self.DirtyArrayParms,axis=-1).reshape((self.NFreqBands,))
        #     Freqs=self.FreqsInfo["freqs"]
        #     nf=len(Freqs)
        #     FreqsMean=np.array([np.mean(Freqs[i]) for i in range(nf)])
        #     Alpha=np.log(FluxFreq[-1]/FluxFreq[0])/np.log(FreqsMean[-1]/FreqsMean[0])
        #     # stop

        # iMax=np.argmax(A)
        # SModelArray[iMax]=A[iMax]
        # return SModelArray

        Th=StopThFrac*MaxA
        MaxResid=np.max(np.abs(A))
        if ArrayMode=="Array":
            for iIter in range(NMaxIter): 
                iPix=np.argmax(np.abs(A))
                f=A[0,0,iPix]
                if (np.abs(f)<Th): break
                if self.ConvMachine.ConvMode=="Matrix":
                    CM=self.ConvMachine.CMParmsMean[0,0]
                    A-=CM[iPix]*gain*f
                elif self.ConvMachine.ConvMode=="Vector":
                    V=self.ConvMachine.GiveConvVector(iPix,TypeOut="Parms")
                    Vm=np.mean(np.array(V),axis=0).reshape((1,1,self.NPixListParms))
                    A-=Vm*gain*f
                SModelArray[iPix]+=gain*f
                ThisMaxResid=np.max(np.abs(A))
                if ThisMaxResid<MaxResid:
                    MaxResid=ThisMaxResid
                else:
                    break

                # import pylab
                # pylab.clf()
                # pylab.plot(A.ravel())
                # pylab.plot(SModelArray.ravel())
                # pylab.draw()
                # pylab.show(False)
                # pylab.pause(0.1)

#            stop
        elif ArrayMode=="Image":
            for iIter in range(NMaxIter): 
                #print iIter,"/", NMaxIter
                #import pylab
                #pylab.clf()
                #pylab.subplot(1,3,1); pylab.imshow(A,interpolation="nearest")

                AbsA=np.abs(A)
                iPix,jPix=np.where(AbsA==np.max(AbsA))
                iPix,jPix=iPix[0],jPix[0]
                f=A[iPix,jPix]
                if np.abs(f)<Th: break
                dx=iPix-xcDirty
                dy=jPix-ycDirty
                ThisPSF=np.roll(np.roll(PSFMean,dx,axis=0),dy,axis=1)
                ThisPSFCut=ThisPSF[xcPSF-NPixPSF_x//2:xcPSF+NPixPSF_x//2+1,ycPSF-NPixPSF_y//2:ycPSF+NPixPSF_y//2+1]

                NMin_x=np.min([A.shape[-2],ThisPSFCut.shape[-2]])
                NMin_y=np.min([A.shape[-1],ThisPSFCut.shape[-1]])
                xc0=A.shape[-2]//2
                xc1=ThisPSFCut.shape[-2]//2
                yc0=A.shape[-1]//2
                yc1=ThisPSFCut.shape[-1]//2
                #pylab.subplot(1,3,2); pylab.imshow(ThisPSFCut,interpolation="nearest")
                A[xc0-NMin_x//2:xc0+NMin_x//2+1,yc0-NMin_y//2:yc0+NMin_y//2+1]-=gain*f*ThisPSFCut[xc1-NMin_x//2:xc1+NMin_x//2+1,yc1-NMin_y//2:yc1+NMin_y//2+1]
                A[Mask]=0
                #pylab.subplot(1,3,3); pylab.imshow(A,interpolation="nearest")
                #pylab.draw()
                #pylab.show(False)
                #pylab.pause(0.1)
                SModelArray[iPix,jPix]+=gain*f

                ThisMaxResid=np.max(np.abs(A))
                if ThisMaxResid<MaxResid:
                    MaxResid=ThisMaxResid
                else:
                    break

            SModelArraySq=np.zeros_like(Asq)
            for iFreq in range(self.NFreqBands):
                for iPol in range(self.npol):
                    SModelArraySq[iFreq,iPol]=SModelArray[:,:]
            SModelArray=self.PM.SquareArrayToModel(SModelArraySq,TypeInOut=("Parms","Parms")).reshape((self.NFreqBands,self.npol,self.NPixListParms))
            SModelArray=SModelArray[0,0]


        return SModelArray,Alpha

        # stop




    def setBestIndiv(self,BestIndiv):
        self.BestContinuousFitNess=BestIndiv.ContinuousFitNess




    def giveDistanceIndiv(self,pop):
        N=len(pop)
        D=np.zeros((N,N),np.float32)
        for i,iIndiv in enumerate(pop):
            for j,jIndiv in enumerate(pop):
                D[i,j]=D[j,i]=np.count_nonzero(iIndiv-jIndiv)
                
        print(D)
        from tsp_solver.greedy import solve_tsp
        path = solve_tsp( D )
        for i in path[1::]:
            print(D[i,path[i-1]]/float(len(iIndiv)))

    def _fill_pop_array(self, pop):
        """Creates "Population" cube inside the island dict, iof not already created, or if the wrong shape."""
        if len(pop) > 0:
            pop_shape = tuple([len(pop)] + list(pop[0].shape))
            if "Population" not in self._island_dict:
                pop_array = self._island_dict.addSharedArray("Population", pop_shape, pop[0].dtype)
            else:
                pop_array = self._island_dict["Population"]
                if pop_array.shape != pop_shape:
                    self._island_dict.delete_item("Population")
                    pop_array = self._island_dict.addSharedArray("Population", pop_shape, pop[0].dtype)
            for i,individual in enumerate(pop):
                pop_array[i,...] = pop[i]
            return pop_array

    def GiveFitnessPop(self,pop):

        Parallel=self.ParallelFitness
        self._fill_pop_array(pop)

        #print("LSFDSDLFK self.ParallelFitness",self.ParallelFitness)
        if self.ParallelFitness:
            for iIndividual,individual in enumerate(pop):
                DicoJob={"iIndividual":iIndividual,
                         "BestChi2":self.BestChi2,
                         "EntropyMinMax":self.EntropyMinMax,
                         "OperationType":"Fitness"}
                self.APP_Fitness.runJob("Fitness:%i:%i"%(self.iIsland,iIndividual),
                                        self._runTask,
                                        args=(DicoJob,),
                                        serial=SERIAL)
            LDicoResults=self.APP_Fitness.awaitJobResults("Fitness:%i:*"%(self.iIsland),
                                                          #progress="Fitness",
                                                          )
        else:
            LDicoResults=[]
            for iIndividual,individual in enumerate(pop):
                DicoJob={"iIndividual":iIndividual,
                         "BestChi2":self.BestChi2,
                         "EntropyMinMax":self.EntropyMinMax,
                         "OperationType":"Fitness"}
                res=self._runTask(DicoJob)
                LDicoResults.append(res)
            
        DicoFitnesses={}
        DicoChi2={}
        for DicoResult in LDicoResults:
            if DicoResult["Success"]:
                iIndividual=DicoResult["iIndividual"]
                DicoFitnesses[iIndividual]=DicoResult["fitness"]
                DicoChi2[iIndividual]=DicoResult["Chi2"]

        fitnesses=[]
        Chi2=[]
        for iIndividual in range(len(pop)):
            fitnesses.append(DicoFitnesses[iIndividual])
            Chi2.append(DicoChi2[iIndividual])
        #print "finished"

        # self.BestChi2=np.min(Chi2)
        # iBestChi2=np.argmin(Chi2)
        # BestInidividual=pop[iBestChi2]
        # S=self.PM.ArrayToSubArray(BestInidividual,"Poly0")
        # St=np.sum(np.abs(S))[()].copy()
        # if St==0: St=1e-10
        # MaxEntropy=-St*np.log(St/self.NPixListParms)
        # MinEntropy=-St*np.log(St)
        # self.EntropyMinMax=MinEntropy,MaxEntropy
        
        #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        #print "Best chi2 %f"%self.BestChi2
        # print "=============================="
        # print fitnesses
        # print Chi2
        # print "=============================="
        return fitnesses,Chi2

    def mutatePop(self,pop,mutpb,MutConfig):

        Parallel=self.ParallelFitness
        DicoFitnesses={}
        DicoChi2={}
        pop_array = self._island_dict["Population"]
        if self.ParallelFitness:
            for iIndividual,individual in enumerate(pop):
                if random.random() < mutpb:
                    pop_array[iIndividual,...] = individual
                    DicoJob={"iIndividual":iIndividual,
                             "mutConfig":MutConfig,
                             "OperationType":"Mutate"}
                    self.APP_Fitness.runJob("Mutate:%i:%i"%(self.iIsland,iIndividual),
                                            self._runTask,
                                            args=(DicoJob,),serial=SERIAL)
            LDicoResults=self.APP_Fitness.awaitJobResults("Mutate:%i:*"%(self.iIsland),
                                                          #progress="Mutate",
                                                          )
        else:
            LDicoResults=[]
            for iIndividual,individual in enumerate(pop):
                if random.random() < mutpb:
                    pop_array[iIndividual,...] = individual
                    DicoJob={"iIndividual":iIndividual,
                             "mutConfig":MutConfig,
                             "OperationType":"Mutate"}
                    res=self._runTask(DicoJob)
                    LDicoResults.append(res)


        for DicoResult in LDicoResults:
            if DicoResult["Success"]:
                iIndividual=DicoResult["iIndividual"]
                mutant = pop_array[iIndividual]
                pop[iIndividual][:] = mutant[:]

        return pop



    
    def mutGaussian(self,*args,**kwargs):
        return self.MutMachine.mutGaussian(*args,**kwargs)
    
    def Plot(self,pop,iGen):

        V = tools.selBest(pop, 1)[0]

        S=self.PM.ArrayToSubArray(V,"Poly0")
        Al=self.PM.ArrayToSubArray(V,"Poly1")
        # Al.fill(0)
        # Al[11]=0.
        # S[0:11]=0
        # S[12:]=0
        # S[11]=10000.

        for iChannel in range(self.NFreqBands):
            self.PlotChannel(pop,iGen,iChannel=iChannel)

        # import pylab
        # pylab.figure(30,figsize=(5,3))
        # #pylab.clf()
        # S=self.PM.ArrayToSubArray(V,"S")
        # Al=self.PM.ArrayToSubArray(V,"Alpha")

        # pylab.subplot(1,2,1)
        # pylab.plot(S)
        # pylab.subplot(1,2,2)
        # pylab.plot(Al)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        



    def PlotChannel(self,pop,iGen,iChannel=0):

        import pylab
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        best_ind = tools.selBest(pop, 1)[0]
        V=best_ind

        #print self.PM.ArrayToSubArray(V,"Alpha")

        # A=self.PM.GiveModelArray(V)
        # A=self.Convolve(A)

        ConvModelArray=self.ToConvArray(V)
        IM=self.PM.ModelToSquareArray(ConvModelArray,TypeInOut=("Data","Data"))
        Dirty=self.PM.ModelToSquareArray(self.DirtyArray,TypeInOut=("Data","Data"))
        Resid=self.PM.ModelToSquareArray(self.ResidArray,TypeInOut=("Data","Data"))


        vmin,vmax=np.min([Dirty[iChannel].min(),0]),Dirty[iChannel].max()
    
    
        ax0=pylab.subplot(2,3,1)
        AA=Dirty[iChannel,0]
        im0=pylab.imshow(AA,interpolation="nearest",vmin=vmin,vmax=vmax)
        pylab.title("Data\n(mm= %f %f)"%(AA.min(),AA.max()))
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        divider0 = make_axes_locatable(ax0)
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im0, cax=cax0)
    
        ax1=pylab.subplot(2,3,2,sharex=ax0,sharey=ax0)
        AA=IM[iChannel,0]
        im1=pylab.imshow(AA,interpolation="nearest")#,vmin=vmin,vmax=vmax)
        pylab.title("Convolved Model\n%f %f"%(AA.min(),AA.max()))
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im1, cax=cax1)
    
        ax2=pylab.subplot(2,3,3,sharex=ax0,sharey=ax0)
        R=Dirty[iChannel,0]-IM[iChannel,0]
        im2=pylab.imshow(R,interpolation="nearest")#,vmin=vmin,vmax=vmax)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)
        pylab.title("Residual Data\n %f %f"%(R.min(),R.max()))
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im2, cax=cax2)
        
    
        #pylab.colorbar()
        if self.DataTrue is not None:
            DataTrue=self.DataTrue
            vmin,vmax=DataTrue.min(),DataTrue.max()
            ax3=pylab.subplot(2,3,4)
            AA=DataTrue[iChannel,0]
            im3=pylab.imshow(AA,interpolation="nearest",vmin=vmin,vmax=vmax)
            ax3.axes.get_xaxis().set_visible(False)
            ax3.axes.get_yaxis().set_visible(False)
            pylab.title("True Sky %f"%AA.max())
            divider3 = make_axes_locatable(ax3)
            cax3 = divider3.append_axes("right", size="5%", pad=0.05)
            pylab.colorbar(im3, cax=cax3)
    
        ax0=pylab.subplot(2,3,4)
        AA=Resid[iChannel,0]
        im0=pylab.imshow(AA,interpolation="nearest",vmin=vmin,vmax=vmax)
        pylab.title("Resid\n(mm= %f %f)"%(AA.min(),AA.max()))
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        divider0 = make_axes_locatable(ax0)
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im0, cax=cax0)

            
        ax4=pylab.subplot(2,3,5,sharex=ax0,sharey=ax0)
        ModelArray=self.PM.GiveModelArray(V)
        IM=self.PM.ModelToSquareArray(ModelArray)


        #im4=pylab.imshow(IM[iChannel,0],interpolation="nearest",vmin=vmin-0.1,vmax=vmax)
        AA=IM[iChannel,0]
        dx=AA.shape[0]
        im4=pylab.imshow(AA,interpolation="nearest")#,vmin=vmin-0.1,vmax=1.5)
        ax4.axes.get_xaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)
        pylab.title("Model \n%f %f \n %s"%(AA.min(),AA.max(),np.sum(AA)))
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im4, cax=cax4)

        PSF=self.PSF
        vmin,vmax=PSF[iChannel,0].min(),PSF[iChannel,0].max()
        nx,ny=PSF.shape[-2:]
        ax5=pylab.subplot(2,3,6)
        AA=PSF[iChannel,0]
        AA=AA[nx//2-dx//2:nx//2+dx//2+1,nx//2-dx//2:nx//2+dx//2+1]
        im5=pylab.imshow(AA,interpolation="nearest",vmin=vmin,vmax=vmax)
        ax5.axes.get_xaxis().set_visible(False)
        ax5.axes.get_yaxis().set_visible(False)
        pylab.title("PSF %f"%AA.max())
        divider5 = make_axes_locatable(ax5)
        cax5 = divider5.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im5, cax=cax5)
    
    
        pylab.suptitle('Population generation %i [%f]'%(iGen,best_ind.fitness.values[0]),size=16)
        #pylab.tight_layout()
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)
        # fig.savefig("png/fig%2.2i_%4.4i.png"%(iChannel,iGen))

# #################################################################"    
# #################################################################"    
# #################################################################"    
# #################################################################"    

    def ToConvArray(self,V,OutMode="Data"):
        self.ModelA=self.PM.GiveModelArray(V)
        A=self.ConvMachine.Convolve(self.ModelA,OutMode=OutMode)
        return A


    

    def _runTask(self,DicoJob):
        if DicoJob["OperationType"]=="Fitness":
            return self.GiveFitnessWorker(DicoJob)
        elif DicoJob["OperationType"]=="Mutate":
            return self.runSingleMutation(DicoJob)

    def runSingleMutation(self,DicoJob):
        pid=str(multiprocessing.current_process())
        iIndividual=DicoJob["iIndividual"]

        self._island_dict.reload()
        individual = self._island_dict["Population"][iIndividual]
        Mut_pFlux, Mut_p0, Mut_pMove, Mut_pScale, Mut_pOffset=DicoJob["mutConfig"]
        self.MutMachine.mutGaussian(individual,Mut_pFlux, Mut_p0, Mut_pMove, Mut_pScale, Mut_pOffset)
        ## mutation above is in-place, so no need to copy
        #individual[:]=individualOut[:]

        return {"Success": True, 
                "iIndividual": iIndividual}


    def GiveFitnessWorker(self,DicoJob):
        pid=str(multiprocessing.current_process())
        iIndividual=DicoJob["iIndividual"]
        self.BestChi2=DicoJob["BestChi2"]
        if "EntropyMinMax" in DicoJob.keys():
            self.EntropyMinMax=DicoJob["EntropyMinMax"]
        self._island_dict.reload()
        individual = self._island_dict["Population"][iIndividual]
        fitness,Chi2=self.GiveFitness(individual)
        return {"Success": True, 
                "iIndividual": iIndividual,
                "fitness":fitness,
                "Chi2":Chi2}


    def GiveCompacity(self,S):
        DM=ClassDistMachine()
        #S.fill(1)
        #S[0]=100
        DM.setRefSample(np.arange(S.size),W=np.sort(S),Ns=100,xmm=[0,S.size-1])#,W=sAround,Ns=10)
        xs,ys=DM.xyCumulD
        dx=xs[1]-xs[0]
        #I=2.*(S.size-np.sum(ys)*dx)/S.size-1.
        # number between 1 (compact) and homogeneous 0
        I=1.-np.sum(ys)*dx/(S.size/2)
        
        # import pylab
        # pylab.figure(4,figsize=(5,3))
        # pylab.plot(xs,ys)
        # pylab.title("%f"%I)
        # pylab.draw()
        # pylab.show(block=False)
        # pylab.pause(0.1)

        return I


    def GiveFitness(self,individual,DoPlot=False):
        
        A=self.ToConvArray(individual)
        fitness=0.
        Resid=self.DirtyArray-A
        
        # print self.MaxFunc
        # print self.DirtyArray
        # print np.max(A)
        # print 


        # if True:#DoPlot:
        #     import pylab
        #     pylab.clf()
        #     pylab.plot(self.DirtyArray.flatten())
        #     pylab.plot(A.flatten())
        #     #pylab.plot(Resid.flatten())
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        #     #stop

        nFreqBands,_,_=Resid.shape
        
        #ResidShape=(self.NFreqBands,1,self.NPixListData)
        #WeightFreqBands=self.WeightFreqBands.reshape((nFreqBands,1,1))
        #Weight=WeightFreqBands/np.sum(WeightFreqBands)
        S=self.PM.ArrayToSubArray(individual,"Poly0")
        chi2=0.
        ContinuousFitNess=[]
        PixVariance=self.PixVariance.reshape((nFreqBands,1,1))
        Verbose=0
        for FuncType in self.MaxFunc:
            if FuncType=="Chi2":
                # chi2=-np.sum(Weight*(Resid)**2)/(self.PixVariance*Resid.size)
                chi2=np.sqrt(np.sum(((Resid)**2/PixVariance).reshape((nFreqBands,-1)),axis=-1))
                if Verbose: print("chi=",np.sum(chi2),chi2.flatten())
                chi2_norm=np.sum(chi2)
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(-chi2_norm*W)
            if FuncType=="L1":
                A=self.PM.GiveModelArray(individual)
                l1=np.sum(A/np.sqrt(self.PixVariance.reshape((nFreqBands,1))),axis=1)
                W=self.WeightMaxFunc[FuncType]
                if Verbose: print("l1=",np.sum(l1),l1.flatten())
                l1=np.sum(l1)
                ContinuousFitNess.append(-l1*W)
            if FuncType=="Sum2":
                # chi2=-np.sum(Weight*(Resid)**2)/(self.PixVariance*Resid.size)
                chi2=np.sum((Resid)**2)
                chi2_norm=chi2#/np.abs(self.BestChi2)
                #print chi2_norm
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(-chi2_norm*W)
            if FuncType=="Chi2Th":
                chi2=np.sum((Resid)**2/PixVariance)

                f=chi2/self.BestChi2
                f=(chi2-self.BestChi2)/self.BestChi2
                if np.abs(f)<0.5:
                    chi2Th=0.
                else:
                    chi2Th=1e6
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(-chi2Th*W)
            if FuncType=="BIC":
                chi2=np.sum((Resid)**2/PixVariance)
                #chi2/=self.BestChi2
                n=Resid.size
                k=np.count_nonzero(S)
                #BIC=chi2+100*k*np.log(n)
                BIC=chi2+self.GD["SSDClean"]["BICFactor"]*k*np.log(n)
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(-BIC*W)
            if FuncType=="MEM":
                chi2=np.sum((Resid)**2/PixVariance)
                if self.EntropyMinMax is None:
                    if Verbose: print("Not computing entropy")
                    ContinuousFitNess.append(-chi2)
                    continue
                aS=np.abs(self.ModelA)
                aS[aS==0]=1e-10
                #aS/=np.sum(aS)
                E=-np.sum(aS*np.log10(aS))
                e0,e1=self.EntropyMinMax
                ENorm=(E-e0)/(e1-e0)
                logChi2n=-np.log10(chi2/np.abs(self.BestChi2))
                
                E+=logChi2n/0.1
                W=self.WeightMaxFunc[FuncType]
                #print chi2,chi2/self.BestChi2,self.BestChi2,E
                ContinuousFitNess.append(E)
            if FuncType=="MaxFlux":
                FMax=-np.max(np.abs(Resid)/np.sqrt(PixVariance))
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(FMax*W)
            if FuncType=="L0":
                # ResidNonZero=S[S!=0]
                # W=self.WeightMaxFunc[FuncType]
                # l0=-(ResidNonZero.size)
                l0=self.GiveCompacity(S)*Resid.size
                if Verbose: print("l0=",l0)
                ContinuousFitNess.append(l0*W)
            if FuncType=="MinFlux":
                SNegArr=np.abs(S[S<0])[()]
                rSNR=Resid/np.sqrt(PixVariance)
                rSNR[rSNR>0]=0
                nn=[np.max([1,np.count_nonzero(rSNR[ii])]) for ii in range(nFreqBands)]
                FNeg=[np.sqrt(np.sum(rSNR[ii]**2))/nn[ii] for ii in range(nFreqBands)]
                if Verbose: print("FNeg",np.sum(FNeg),FNeg)
                sFNeg=np.sum(FNeg)
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(-sFNeg*W)
            if FuncType=="MinFluxNorm":
                SNegArr=np.abs(S[S<0])[()]
                FNeg=-np.sum(SNegArr**2/PixVariance)
                FNeg/=np.abs(self.BestChi2)
                if FNeg==0: continue
                FNeg=np.sign(FNeg)*np.log10(np.abs(FNeg))
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(FNeg*W)
            

        fit=np.sum(ContinuousFitNess)
        if np.isnan(fit): stop
        return (fit,),chi2
        #return (ContinuousFitNess,),chi2



    def PlotIndiv(self,best_ind,iChannel=0):

        import pylab
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        V=best_ind

        ConvModelArray=self.ToConvArray(V)
        IM=self.PM.ModelToSquareArray(ConvModelArray,TypeInOut=("Data","Data"))
        Dirty=self.PM.ModelToSquareArray(self.DirtyArray,TypeInOut=("Data","Data"))


        vmin,vmax=np.min([Dirty.min(),0]),Dirty.max()
    
        fig=pylab.figure(2,figsize=(5,3))
        pylab.clf()
    
        ax0=pylab.subplot(2,3,1)
        im0=pylab.imshow(Dirty[iChannel,0],interpolation="nearest",vmin=vmin,vmax=vmax)
        pylab.title("Data")
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        divider0 = make_axes_locatable(ax0)
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im0, cax=cax0)
    
        ax1=pylab.subplot(2,3,2)
        im1=pylab.imshow(IM[iChannel,0],interpolation="nearest",vmin=vmin,vmax=vmax)
        pylab.title("Convolved Model")
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im1, cax=cax1)
    
        ax2=pylab.subplot(2,3,3)
        R=Dirty[iChannel,0]-IM[iChannel,0]
        im2=pylab.imshow(R,interpolation="nearest")#,vmin=vmin,vmax=vmax)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)
        pylab.title("Residual Data")
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im2, cax=cax2)
    
    
        # #pylab.colorbar()
        # if self.DataTrue is not None:
        #     DataTrue=self.DataTrue
        #     vmin,vmax=DataTrue.min(),DataTrue.max()
        #     ax3=pylab.subplot(2,3,4)
        #     im3=pylab.imshow(DataTrue[iChannel,0],interpolation="nearest",vmin=vmin,vmax=vmax)
        #     ax3.axes.get_xaxis().set_visible(False)
        #     ax3.axes.get_yaxis().set_visible(False)
        #     pylab.title("True Sky")
        #     divider3 = make_axes_locatable(ax3)
        #     cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        #     pylab.colorbar(im3, cax=cax3)
    
    
        ax4=pylab.subplot(2,3,5)
        ModelArray=self.PM.GiveModelArray(V)
        IM=self.PM.ModelToSquareArray(ModelArray)


        #im4=pylab.imshow(IM[iChannel,0],interpolation="nearest",vmin=vmin-0.1,vmax=vmax)
        im4=pylab.imshow(IM[iChannel,0],interpolation="nearest",vmin=vmin-0.1,vmax=1.5)
        ax4.axes.get_xaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)
        pylab.title("Best individual")
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im4, cax=cax4)

        PSF=self.PSF
        vmin,vmax=PSF.min(),PSF.max()
        ax5=pylab.subplot(2,3,6)
        im5=pylab.imshow(PSF[iChannel,0],interpolation="nearest",vmin=vmin,vmax=vmax)
        ax5.axes.get_xaxis().set_visible(False)
        ax5.axes.get_yaxis().set_visible(False)
        pylab.title("PSF")
        divider5 = make_axes_locatable(ax5)
        cax5 = divider5.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im5, cax=cax5)
    
        #pylab.suptitle('Population generation %i [%f]'%(iGen,best_ind.fitness.values[0]),size=16)
        #pylab.tight_layout()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)

        #fig.savefig("png/fig%2.2i_%4.4i.png"%(iChannel,iGen))
        
