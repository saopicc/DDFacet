import collections
import random

import numpy as np
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyLogger

log=MyLogger.getLogger("ClassArrayMethodSSD")
import multiprocessing

import ClassConvMachine
import time
from scipy.stats import chi2

from deap import tools

log= MyLogger.getLogger("ClassArrayMethodSSD")


from ClassParamMachine import ClassParamMachine
from DDFacet.ToolsDir.GeneDist import ClassDistMachine
import ClassMutate

class ClassArrayMethodSSD():
    def __init__(self,Dirty,PSF,ListPixParms,ListPixData,FreqsInfo,GD=None,
                 PixVariance=1.e-2,IslandBestIndiv=None,WeightFreqBands=None,iFacet=0,
                 island_dict=None,
                 iIsland=0,
                 ParallelFitness=False,
                 NCPU=None):
        self.ParallelFitness=ParallelFitness
        self.iFacet=iFacet
        self.WeightFreqBands=WeightFreqBands
        self.iIsland=iIsland

        self._island_dict = island_dict
        self._chain_dict = island_dict.addSubdict("Chains")

        self.PSF = PSF

        self.IslandBestIndiv=IslandBestIndiv
        self.GD=GD
        self.NCPU=NCPU
        if NCPU==None:
            self.NCPU=int(self.GD["Parallel"]["NCPU"])

        self.BestChi2=1.
        self.EntropyMinMax=None
        # IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
        # ListPixData=IncreaseIslandMachine.IncreaseIsland(ListPixData,dx=5)

        #ListPixParms=ListPixData
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
        #self.WeightMaxFunc["MinFlux"]=1.
        #self.WeightMaxFunc["MaxFlux"]=1.

        #self.WeightMaxFunc["L0"]=1.
        self.MaxFunc=self.WeightMaxFunc.keys()
        
        self.NFuncMin=len(self.MaxFunc)
        self.WeightsEA=[1.]#*len(self.MaxFunc)
        #self.WeightsEA=[1.]*len(self.MaxFunc)
        self.MinVar=np.array([0.01,0.01])
        self.PixVariance=PixVariance
        self.FreqsInfo=FreqsInfo
        
        self.NFreqBands,self.npol,self.NPixPSF,_=PSF.shape
        self.PM=ClassParamMachine(ListPixParms,ListPixData,FreqsInfo,SolveParam=GD["SSDClean"]["SSDSolvePars"])
        self.PM.setFreqs(FreqsInfo)
        self.ConvMachine.setParamMachine(self.PM)
        
        self.NParms=self.NPixListParms*self.PM.NParam
        self.DataTrue=None
        self.MutMachine=ClassMutate.ClassMutate(self.PM)


        self.SetDirtyArrays(Dirty)
        self.InitWorkers()


        #pylab.figure(3,figsize=(5,3))
        #pylab.clf()
        # pylab.figure(4,figsize=(5,3))
        # pylab.clf()

    def __del__(self):
        self._chain_dict.delete()
        if "Population" in self._island_dict:
            self._island_dict.delete_item("Population")

    def SetDirtyArrays(self,Dirty):
        print>>log,"SetConvMatrix"
        PSF=self.PSF
        NPixPSF=PSF.shape[-1]
        if self.ListPixData is None:
            x,y=np.mgrid[0:NPixPSF:1,0:NPixPSF:1]
            self.ListPixData=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()
        if self.ListPixParms is None:
            x,y=np.mgrid[0:NPixPSF:1,0:NPixPSF:1]
            self.ListPixParms=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()

        self.DirtyArray=np.zeros((self.NFreqBands,1,self.NPixListData),np.float32)
        xc=yc=NPixPSF/2

        x0,y0=np.array(self.ListPixData).T
        for iBand in range(self.NFreqBands):
            self.DirtyArray[iBand,0,:]=Dirty[iBand,0,x0,y0]

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
            S=self.PM.ArrayToSubArray(self.IslandBestIndiv,"S")
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
                self.ALPHA=ALPHA
                # print "ALPHA=",self.ALPHA
                
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
            _,npol,NPix,_=Asq.shape
            A=np.mean(Asq,axis=0).reshape((NPix,NPix))
            Mask=(A==0)
            _,_,NPixPSF,_=PSF.shape
            PSFMean=np.mean(PSF,axis=0).reshape((NPixPSF,NPixPSF))
            ArrayMode="Image"
            xcPSF=NPixPSF/2
            xcDirty=NPix/2
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
                dy=jPix-xcDirty
                ThisPSF=np.roll(np.roll(PSFMean,dx,axis=0),dy,axis=1)
                ThisPSFCut=ThisPSF[xcPSF-NPixPSF/2:xcPSF+NPixPSF/2+1,xcPSF-NPixPSF/2:xcPSF+NPixPSF/2+1]

                NMin=np.min([A.shape[-1],ThisPSFCut.shape[-1]])
                xc0=A.shape[-1]/2
                xc1=ThisPSFCut.shape[-1]/2
                #pylab.subplot(1,3,2); pylab.imshow(ThisPSFCut,interpolation="nearest")
                A[xc0-NMin/2:xc0+NMin/2+1,xc0-NMin/2:xc0+NMin/2+1]-=gain*f*ThisPSFCut[xc1-NMin/2:xc1+NMin/2+1,xc1-NMin/2:xc1+NMin/2+1]
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



    def InitWorkers(self):
        Parallel=self.ParallelFitness
        if not(Parallel):
            NCPU=1
        else:
            NCPU=self.NCPU
        self.NCPU=NCPU

        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()
        workerlist=[]
        #print "start"
        for ii in range(NCPU):
            W=WorkerFitness(work_queue,
                            result_queue,
                            island_dict=self._island_dict,
                            iIsland=self.iIsland,
                            ListPixParms=self.ListPixParms,
                            ListPixData=self.ListPixData,
                            PSF=self.PSF,
                            GD=self.GD,
                            PauseOnStart=False,
                            PM=self.PM,
                            PixVariance=self.PixVariance,
                            EstimatedStdFromResid=self.EstimatedStdFromResid,
                            MaxFunc=self.MaxFunc,
                            WeightMaxFunc=self.WeightMaxFunc,
                            DirtyArray=self.DirtyArray,
                            ConvMode=self.ConvMode,
                            StopWhenQueueEmpty=not(Parallel),
                            BestChi2=self.BestChi2,
                            DicoData=self.DicoData)

            workerlist.append(W)

        self.work_queue=work_queue
        self.result_queue=result_queue
        self.workerlist=workerlist
        if self.ParallelFitness:
            for ii in range(NCPU):
                #print "launch parallel", ii
                workerlist[ii].start()


    def KillWorkers(self):
        workerlist=self.workerlist
        if self.ParallelFitness:
            #print "turn off"
            for ii in range(self.NCPU):
                workerlist[ii].shutdown()
                workerlist[ii].terminate()
                workerlist[ii].join()

    def giveDistanceIndiv(self,pop):
        N=len(pop)
        D=np.zeros((N,N),np.float32)
        for i,iIndiv in enumerate(pop):
            for j,jIndiv in enumerate(pop):
                D[i,j]=D[j,i]=np.count_nonzero(iIndiv-jIndiv)
                
        print D
        from tsp_solver.greedy import solve_tsp
        path = solve_tsp( D )
        for i in path[1::]:
            print D[i,path[i-1]]/float(len(iIndiv))

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

        work_queue=self.work_queue
        result_queue=self.result_queue
        workerlist=self.workerlist
        Parallel=self.ParallelFitness
        DicoFitnesses={}
        DicoChi2={}
        NJobs = len(pop)
        NCPU=self.NCPU
        #self.giveDistanceIndiv(pop)
        #print "OK"
        self._fill_pop_array(pop)
        for iIndividual,individual in enumerate(pop):
            work_queue.put({"iIndividual":iIndividual,
                            "BestChi2":self.BestChi2,
                            "EntropyMinMax":self.EntropyMinMax,
                            "OperationType":"Fitness"})

        
        if not Parallel:
            for ii in range(NCPU):
                workerlist[ii].run()  # just run until all work is completed

        # for ii in range(NCPU):
        #     print "launch parallel", ii
        #     workerlist[ii].start()


        iResult=0

        while iResult < NJobs:
            DicoResult=None
            #print work_queue.qsize(),result_queue.qsize()
            if result_queue.qsize()!=0:
                try:
                    DicoResult=result_queue.get_nowait()
                except Exception,e:
                    #print "Exception: %s"%(str(e))
                    pass
                

            if DicoResult==None:
                time.sleep(.1)
                continue            

            # try:
            #     DicoResult = result_queue.get(True, 5)
            # except:
            #     time.sleep(0.1)
            #     continue

            if DicoResult["Success"]:
                iIndividual=DicoResult["iIndividual"]
                iResult += 1
                DicoFitnesses[iIndividual]=DicoResult["fitness"]
                DicoChi2[iIndividual]=DicoResult["Chi2"]
            NDone = iResult

        # for ii in range(NCPU):
        #     workerlist[ii].shutdown()
        #     workerlist[ii].terminate()
        #     workerlist[ii].join()

        fitnesses=[]
        Chi2=[]
        for iIndividual in range(len(pop)):
            fitnesses.append(DicoFitnesses[iIndividual])
            Chi2.append(DicoChi2[iIndividual])
        #print "finished"

        self.BestChi2=np.min(Chi2)

        iBestChi2=np.argmin(Chi2)
        BestInidividual=pop[iBestChi2]
        S=self.PM.ArrayToSubArray(BestInidividual,"S")
        St=np.sum(np.abs(S))[()].copy()
        if St==0: St=1e-10
        MaxEntropy=-St*np.log(St/self.NPixListParms)
        MinEntropy=-St*np.log(St)
        self.EntropyMinMax=MinEntropy,MaxEntropy
        
        #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        #print "Best chi2 %f"%self.BestChi2
        # print "=============================="
        # print fitnesses
        # print Chi2
        # print "=============================="
        return fitnesses,Chi2

    def mutatePop(self,pop,mutpb,MutConfig):

        work_queue=self.work_queue
        result_queue=self.result_queue
        workerlist=self.workerlist
        Parallel=self.ParallelFitness
        DicoFitnesses={}
        DicoChi2={}
        NCPU=self.NCPU
        NJobs = 0
        pop_array = self._island_dict["Population"]
        for iIndividual,individual in enumerate(pop):
            if random.random() < mutpb:
                NJobs+=1
                pop_array[iIndividual,...] = individual
                work_queue.put({"iIndividual":iIndividual,
                                "mutConfig":MutConfig,
                                "OperationType":"Mutate"})


        
        if not Parallel:
            for ii in range(NCPU):
                workerlist[ii].run()  # just run until all work is completed

        # for ii in range(NCPU):
        #     print "launch parallel", ii
        #     workerlist[ii].start()


        iResult=0

        while iResult < NJobs:
            DicoResult=None
            #print work_queue.qsize(),result_queue.qsize()
            if result_queue.qsize()!=0:
                try:
                    DicoResult=result_queue.get_nowait()
                except Exception,e:
                    #print "Exception: %s"%(str(e))
                    pass
                

            if DicoResult==None:
                time.sleep(.1)
                continue            

            # try:
            #     DicoResult = result_queue.get(True, 5)
            # except:
            #     time.sleep(0.1)
            #     continue

            if DicoResult["Success"]:
                iIndividual=DicoResult["iIndividual"]
                iResult += 1
                mutant = pop_array[iIndividual]
                pop[iIndividual][:] = mutant[:]
            NDone = iResult

        return pop


    def GiveMetroChains(self,pop,NSteps=1000):

        work_queue=self.work_queue
        result_queue=self.result_queue
        workerlist=self.workerlist
        Parallel=self.ParallelFitness
        DicoFitnesses={}
        DicoChi2={}
        NJobs = len(pop)
        NCPU=self.NCPU

        self._chain_dict.delete()

        self._fill_pop_array(pop)
        for iIndividual,individual in enumerate(pop):
            work_queue.put({"iIndividual":iIndividual,
                            "BestChi2":self.BestChi2,
                            "OperationType":"Metropolis",
                            "NSteps":NSteps})

        
        if not Parallel:
            for ii in range(NCPU):
                workerlist[ii].run()  # just run until all work is completed

        # for ii in range(NCPU):
        #     print "launch parallel", ii
        #     workerlist[ii].start()


        iResult=0
        DicoChains = {}
        while iResult < NJobs:
            DicoResult=None
            #print work_queue.qsize(),result_queue.qsize()
            if result_queue.qsize()!=0:
                try:
                    DicoResult=result_queue.get_nowait()
                except Exception,e:
                    #print "Exception: %s"%(str(e))
                    pass
                

            if DicoResult==None:
                time.sleep(.1)
                continue            

            # try:
            #     DicoResult = result_queue.get(True, 5)
            # except:
            #     time.sleep(0.1)
            #     continue

            if DicoResult["Success"]:
                iIndividual=DicoResult["iIndividual"]
                iResult += 1
                # result already in _chain_dict
        self._chain_dict.reload()

        return self._chain_dict

    
    def mutGaussian(self,*args,**kwargs):
        return self.MutMachine.mutGaussian(*args,**kwargs)
    
    def Plot(self,pop,iGen):

        V = tools.selBest(pop, 1)[0]

        S=self.PM.ArrayToSubArray(V,"S")
        Al=self.PM.ArrayToSubArray(V,"Alpha")
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
        

    def GiveCompacity(self,S):
        DM=ClassDistMachine()
        #S.fill(1)
        #S[0]=100
        DM.setRefSample(np.arange(S.size),W=np.sort(S),Ns=100,xmm=[0,S.size-1])#,W=sAround,Ns=10)
        #DM.setRefSample(S)#,W=sAround,Ns=10)
        xs,ys=DM.xyCumulD
        dx=xs[1]-xs[0]
        I=2.*(S.size-np.sum(ys)*dx)/S.size-1.
        return I
        # pylab.figure(4,figsize=(5,3))
        # pylab.plot(xp,yp)
        # pylab.title("%f"%I)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop


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


        vmin,vmax=np.min([Dirty.min(),0]),Dirty.max()
    
        fig=pylab.figure(iChannel+1,figsize=(5,3))
        pylab.clf()
    
        ax0=pylab.subplot(2,3,1)
        im0=pylab.imshow(Dirty[iChannel,0],interpolation="nearest",vmin=vmin,vmax=vmax)
        pylab.title("Data")
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        divider0 = make_axes_locatable(ax0)
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im0, cax=cax0)
    
        ax1=pylab.subplot(2,3,2,sharex=ax0,sharey=ax0)
        im1=pylab.imshow(IM[iChannel,0],interpolation="nearest")#,vmin=vmin,vmax=vmax)
        pylab.title("Convolved Model")
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
        pylab.title("Residual Data")
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im2, cax=cax2)
    
    
        #pylab.colorbar()
        if self.DataTrue is not None:
            DataTrue=self.DataTrue
            vmin,vmax=DataTrue.min(),DataTrue.max()
            ax3=pylab.subplot(2,3,4)
            im3=pylab.imshow(DataTrue[iChannel,0],interpolation="nearest",vmin=vmin,vmax=vmax)
            ax3.axes.get_xaxis().set_visible(False)
            ax3.axes.get_yaxis().set_visible(False)
            pylab.title("True Sky")
            divider3 = make_axes_locatable(ax3)
            cax3 = divider3.append_axes("right", size="5%", pad=0.05)
            pylab.colorbar(im3, cax=cax3)
    
    
        ax4=pylab.subplot(2,3,5,sharex=ax0,sharey=ax0)
        ModelArray=self.PM.GiveModelArray(V)
        IM=self.PM.ModelToSquareArray(ModelArray)


        #im4=pylab.imshow(IM[iChannel,0],interpolation="nearest",vmin=vmin-0.1,vmax=vmax)
        im4=pylab.imshow(IM[iChannel,0],interpolation="nearest")#,vmin=vmin-0.1,vmax=1.5)
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
    
    
        pylab.suptitle('Population generation %i [%f]'%(iGen,best_ind.fitness.values[0]),size=16)
        #pylab.tight_layout()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        fig.savefig("png/fig%2.2i_%4.4i.png"%(iChannel,iGen))
        stop
# #################################################################"    
# #################################################################"    
# #################################################################"    
# #################################################################"    


class WorkerFitness(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 island_dict=None,
                 iIsland=None,
                 ListPixParms=None,
                 ListPixData=None,
                 GD=None,
                 PSF=None,
                 PauseOnStart=False,
                 PM=None,
                 PixVariance=1e-2,
                 EstimatedStdFromResid=0,
                 MaxFunc=None,
                 WeightMaxFunc=None,
                 DirtyArray=None,
                 ConvMode=None,
                 StopWhenQueueEmpty=False,
                 BestChi2=1.,
                 DicoData=None):
        self.T=ClassTimeIt.ClassTimeIt("WorkerFitness")
        self.T.disable()
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self._pause_on_start = PauseOnStart
        self._island_dict = island_dict
        self._chain_dict = island_dict["Chains"]
        self.GD=GD
        self.PM=PM
        self.EstimatedStdFromResid=EstimatedStdFromResid
        self.ListPixParms=ListPixParms
        self.ListPixData=ListPixData
        self.iIsland=iIsland
        self.PSF = PSF
        self.PixVariance=PixVariance
        self.ConvMachine=ClassConvMachine.ClassConvMachine(self.PSF,self.ListPixParms,self.ListPixData,ConvMode)
        self.ConvMachine.setParamMachine(self.PM)
        self.DicoData=DicoData
        self.MutMachine=ClassMutate.ClassMutate(self.PM)
        self.MutMachine.setData(DicoData)
        self.MaxFunc=MaxFunc
        self.WeightMaxFunc=WeightMaxFunc
        self.DirtyArray=DirtyArray
        self.T.timeit("init")
        self.StopWhenQueueEmpty=StopWhenQueueEmpty
        self.BestChi2=BestChi2



    def shutdown(self):
        self.exit.set()

    def CondContinue(self):
        if self.StopWhenQueueEmpty:
            return not(self.work_queue.qsize()==0)
        else:
            return True

    def ToConvArray(self,V,OutMode="Data"):
        self.ModelA=self.PM.GiveModelArray(V)
        A=self.ConvMachine.Convolve(self.ModelA,OutMode=OutMode)
        return A

    def run(self):
        # # pause self in debugging mode
        # #if self._pause_on_start:
        # #    os.kill(os.getpid(),signal.SIGSTOP)
        # while not self.kill_received and not self.work_queue.empty():
        #     DicoJob = self.work_queue.get()
        #     individual=DicoJob["individual"]
        #     iIndividual=DicoJob["iIndividual"]
        #     fitness=self.GiveFitness(individual)
        #     #print iIndividual
        #     self.result_queue.put({"Success": True, 
        #                            "iIndividual": iIndividual,
        #                            "fitness":fitness})
         
        while not self.kill_received and self.CondContinue():
            #gc.enable()
            try:
                DicoJob = self.work_queue.get()
            except:
                break
            

            if DicoJob["OperationType"]=="Fitness":
                #print "FitNess"
                self.GiveFitnessWorker(DicoJob)
            elif DicoJob["OperationType"]=="Metropolis":
                self.runMetroSingleChainWorker(DicoJob)
            elif DicoJob["OperationType"]=="Mutate":
                #print "Mutate"
                self.runSingleMutation(DicoJob)

    def runSingleMutation(self,DicoJob):
        pid=str(multiprocessing.current_process())
        self.T.reinit()
        iIndividual=DicoJob["iIndividual"]

        self._island_dict.reload()
        individual = self._island_dict["Population"][iIndividual]

        Mut_pFlux, Mut_p0, Mut_pMove, Mut_pScale, Mut_pOffset=DicoJob["mutConfig"]


        self.MutMachine.mutGaussian(individual,
                                                   Mut_pFlux, Mut_p0, Mut_pMove, Mut_pScale, Mut_pOffset)
        ## mutation above is in-place, so no need to copy
        #individual[:]=individualOut[:]

        self.result_queue.put({"Success": True, 
                               "iIndividual": iIndividual})
        self.T.timeit("done job: %s"%pid)








    def GiveFitnessWorker(self,DicoJob):
        pid=str(multiprocessing.current_process())
        self.T.reinit()
        iIndividual=DicoJob["iIndividual"]
        #print "Worker %s processing indiv %i"%(pid,iIndividual)
        self.BestChi2=DicoJob["BestChi2"]
        if "EntropyMinMax" in DicoJob.keys():
            self.EntropyMinMax=DicoJob["EntropyMinMax"]

        #individual=DicoJob["individual"]
        self._island_dict.reload()
        individual = self._island_dict["Population"][iIndividual]

        fitness,Chi2=self.GiveFitness(individual)
        #print iIndividual
        self.result_queue.put({"Success": True, 
                               "iIndividual": iIndividual,
                               "fitness":fitness,
                               "Chi2":Chi2})
        self.T.timeit("done job: %s"%pid)




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
        S=self.PM.ArrayToSubArray(individual,"S")
        chi2=0.
        ContinuousFitNess=[]
        for FuncType in self.MaxFunc:
            if FuncType=="Chi2":
                # chi2=-np.sum(Weight*(Resid)**2)/(self.PixVariance*Resid.size)
                chi2=np.sum((Resid)**2)/(self.PixVariance)
                chi2_norm=chi2#/np.abs(self.BestChi2)
                #print chi2_norm
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(-chi2_norm*W)
            if FuncType=="Sum2":
                # chi2=-np.sum(Weight*(Resid)**2)/(self.PixVariance*Resid.size)
                chi2=np.sum((Resid)**2)
                chi2_norm=chi2#/np.abs(self.BestChi2)
                #print chi2_norm
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(-chi2_norm*W)
            if FuncType=="Chi2Th":
                chi2=np.sum((Resid)**2)/(self.PixVariance)

                f=chi2/self.BestChi2
                f=(chi2-self.BestChi2)/self.BestChi2
                if np.abs(f)<0.5:
                    chi2Th=0.
                else:
                    chi2Th=1e6
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(-chi2Th*W)
            if FuncType=="BIC":
                chi2=np.sum((Resid)**2)/(self.PixVariance)
                #chi2/=self.BestChi2
                n=Resid.size
                k=np.count_nonzero(S)
                #BIC=chi2+100*k*np.log(n)
                BIC=chi2+self.GD["SSDClean"]["BICFactor"]*k*np.log(n)
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(-BIC*W)
            if FuncType=="MEM":
                chi2=np.sum((Resid)**2)/(self.PixVariance)
                if self.EntropyMinMax is None:
                    print "Not computing entropy"
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
                FMax=-np.max(np.abs(Resid))/(np.sqrt(self.PixVariance))
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(FMax*W)
            if FuncType=="L0":
                # ResidNonZero=S[S!=0]
                # W=self.WeightMaxFunc[FuncType]
                # l0=-(ResidNonZero.size)
                l0=self.GiveCompacity(S)
                ContinuousFitNess.append(l0*W)
            if FuncType=="MinFlux":
                SNegArr=np.abs(S[S<0])[()]
                FNeg=-np.sum(SNegArr**2)/((self.PixVariance))
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(FNeg*W)
            if FuncType=="MinFluxNorm":
                SNegArr=np.abs(S[S<0])[()]
                FNeg=-np.sum(SNegArr**2)/((self.PixVariance))
                FNeg/=np.abs(self.BestChi2)
                if FNeg==0: continue
                FNeg=np.sign(FNeg)*np.log10(np.abs(FNeg))
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(FNeg*W)
            

        return (np.sum(ContinuousFitNess),),chi2
        #return (ContinuousFitNess,),chi2


    def runMetroSingleChainWorker(self,DicoJob):
        pid=str(multiprocessing.current_process())
        self.T.reinit()
        iIndividual=DicoJob["iIndividual"]
        #print "Worker %s processing indiv %i"%(pid,iIndividual)
        self.BestChi2=DicoJob["BestChi2"]
        self._island_dict.reload()
        individual = self._island_dict["Population"][iIndividual]
        self._chain_dict.reload()
        chain_dict = self._chain_dict.addSubdict(iIndividual)
        self.runMetroSingleChain(individual,NSteps=DicoJob["NSteps"],chain_dict=chain_dict)
        self.result_queue.put({"Success": True,
                               "iIndividual": iIndividual})
        self.T.timeit("done job: %s"%pid)

    def runMetroSingleChain(self,individual0,NSteps=1000,chain_dict={}):

        df=self.PM.NPixListData
        self.rv = chi2(df)
        _,Chi2=self.GiveFitness(individual0)
        self.MinChi2=Chi2
        logProb=self.rv.logpdf(Chi2)

        x=np.linspace(0,2*self.rv.moment(1),1000)
        lP=self.rv.logpdf(x)
        iMax=np.argmax(lP)
        self.Chi2PMax=x[iMax]

        # #####################
        # # V0
        #self.Var=self.MinChi2/self.Chi2PMax
        #Chi20_n=self.MinChi2/self.Var
        #VarMin=(3e-3)**2
        #ThVar=np.max([self.Var,VarMin])
        #ShrinkFactor=np.min([1.,self.Var/ThVar])
        # # print
        # # print ShrinkFactor
        # # print
        # # stop
        # #####################
        VarMin=(3e-4)**2
        #self.Var=np.max([self.EstimatedStdFromResid**2,VarMin])
        Var=self.MinChi2/self.Chi2PMax
        S=self.PM.ArrayToSubArray(individual0,Type="S")
        B=np.sum(np.abs(S))/float(S.size)
        B0=7e-4
        Sig0=3e-3
        Sig=B*Sig0/B0

        # print 
        # print "%f %f %f -> %f"%(B,B0,Sig0,Sig)
        # print 

        self.Var=np.max([4.*self.EstimatedStdFromResid**2,
                         Sig**2])

        Chi20_n=self.MinChi2/self.Var
        ShrinkFactor=1.
        # #####################


        
        DicoChains={}
        Parms=individual0

        # ##################################
        DoPlot=True
        if DoPlot:
            import pylab
            pylab.figure(1)
            x=np.linspace(0,2*self.rv.MeanChi2,1000)
            P=self.rv.pdf(x)
            pylab.clf()
            pylab.plot(x,P)
            Chi2Red=Chi2_0#/self.Var
            pylab.scatter(Chi2Red,np.mean(P),c="black")
            pylab.draw()
            pylab.show(False)
        # ##################################

        # ##################################
        DoPlot=False
        # DoPlot=True
        if DoPlot:
            import pylab
            x=np.linspace(0,2*self.rv.moment(1),1000)
            P=self.rv.pdf(x)
            pylab.clf()
            pylab.plot(x,P)
            pylab.scatter(Chi20_n,np.mean(P),c="black")
            pylab.draw()
            pylab.show(False)
        # ##################################

        DicoChains["Parms"]=[]
        DicoChains["Chi2"]=[]
        DicoChains["logProb"]=[]
        logProb0=self.rv.logpdf(Chi20_n)

        
        Mut_pFlux, Mut_p0, Mut_pMove=0.2,0.,0.3


        #T.disable()
        FactorAccelerate=1.
        lAccept=[]
        NBurn=self.GD["MetroClean"]["MetroNBurnin"]

        NSteps=NSteps+NBurn
        
        NAccepted=0
        iStep=0
        NMax=NSteps#10000
        
        #for iStep in range(NSteps):
        while NAccepted<NSteps and iStep<NMax:
            iStep+=1
            #print "========================"
            #print iStep
            individual1,=self.MutMachine.mutGaussian(individual0.copy(), 
                                                     Mut_pFlux, Mut_p0, Mut_pMove)#,
                                                     #FactorAccelerate=FactorAccelerate)
            # ds=Noise
            # individual1,=self.MutMachine.mutNormal(individual0.copy(),ds*1e-1*FactorAccelerate)
            # #T.timeit("mutate")

            _,Chi2=self.GiveFitness(individual1)
            # if Chi2<self.MinChi2:
            #     self.Var=Chi2/self.Chi2PMax
            #     #print "           >>>>>>>>>>>>>> %f"%np.min(Chi2)


            Chi2_n=Chi2/self.Var
            

            Chi2_n=Chi20_n+ShrinkFactor*(Chi2_n-Chi20_n)

            logProb=self.rv.logpdf(Chi2_n)
            
            p1=logProb
            p0=logProb0#DicoChains["logProb"][-1]
            if p1-p0>5:
                R=1
            elif p1-p0<-5:
                R=0
            else:
                R=np.min([1.,np.exp(p1-p0)])

            r=np.random.rand(1)[0]
            #print "%5.3f [%f -> %f]"%(R,p0,p1)
            # print "MaxDiff ",np.max(np.abs(self.pop[iChain]-DicoChains[iChain]["Parms"][-1]))
            lAccept.append((r<R))
            if r<R: # accept
                individual0=individual1
                logProb0=logProb
                NAccepted+=1
                if NAccepted>NBurn:
                    DicoChains["logProb"].append(p1)
                    DicoChains["Parms"].append(individual1)
                    DicoChains["Chi2"].append(Chi2_n)
                
                if DoPlot:
                    pylab.scatter(Chi2_n,np.exp(p1),lw=0)
                    pylab.draw()
                    pylab.show(False)
                    pylab.pause(0.1)

                # print "  accept"
                # # Model=self.StackChain()
                
                # # Asq=self.ArrayMethodsMachine.PM.ModelToSquareArray(Model,TypeInOut=("Parms","Parms"))
                # # _,npol,NPix,_=Asq.shape
                # # A=np.mean(Asq,axis=0).reshape((NPix,NPix))
                # # Mask=(A==0)
                # # pylab.clf()
                # # pylab.imshow(A,interpolation="nearest")
                # # pylab.draw()
                # # pylab.show(False)
                # # pylab.pause(0.1)
                
                
            else:
                
                # # #######################
                if DoPlot:
                    pylab.scatter(Chi2_n,np.exp(p1),c="red",lw=0)
                    pylab.draw()
                    pylab.show(False)
                    pylab.pause(0.1)
                # # #######################
                pass

            #T.timeit("Compare")

            AccRate=np.count_nonzero(lAccept)/float(len(lAccept))
            #print "[%i] Acceptance rate %f [%f with ShrinkFactor %f]"%(iStep,AccRate,FactorAccelerate,ShrinkFactor)
            if (iStep%50==0)&(iStep>10):
                if AccRate>0.234:
                    FactorAccelerate*=1.5
                else:
                    FactorAccelerate/=1.5
                FactorAccelerate=np.min([3.,FactorAccelerate])
                FactorAccelerate=np.max([.01,FactorAccelerate])
                lAccept=[]
            #T.timeit("Acceptance")

        T.timeit("Chain")

        chain_dict["logProb"]=np.array(DicoChains["logProb"])
        chain_dict["Parms"]=np.array(DicoChains["Parms"])
        chain_dict["Chi2"]=np.array(DicoChains["Chi2"])

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
        
