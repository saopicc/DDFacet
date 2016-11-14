import collections
import random

import numpy as np
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyLogger

log=MyLogger.getLogger("ClassArrayMethodGA")
import multiprocessing

from ClassParamMachine import ClassParamMachine
from DDFacet.ToolsDir.GeneDist import ClassDistMachine
from SkyModel.PSourceExtract import ClassIncreaseIsland
from DDFacet.Array import NpShared
import ClassConvMachine
import time

from deap import tools

log= MyLogger.getLogger("ClassArrayMethodGA")


from ClassParamMachine import ClassParamMachine
from DDFacet.ToolsDir.GeneDist import ClassDistMachine


class ClassArrayMethodGA():
    def __init__(self,Dirty,PSF,ListPixParms,ListPixData,FreqsInfo,GD=None,
                 PixVariance=1.e-2,IslandBestIndiv=None,WeightFreqBands=None,iFacet=0,
                 IdSharedMem="",
                 iIsland=0,
                 ParallelFitness=False):
        self.ParallelFitness=ParallelFitness
        self.iFacet=iFacet
        self.WeightFreqBands=WeightFreqBands
        self.IdSharedMem=IdSharedMem
        self.iIsland=iIsland
        NpShared.DelArray("%sPSF_Island_%4.4i"%(IdSharedMem,iIsland))
        self.PSF=NpShared.ToShared("%sPSF_Island_%4.4i"%(IdSharedMem,iIsland),PSF)
        self.IslandBestIndiv=IslandBestIndiv
        self.GD=GD
        self.NCPU=int(self.GD["Parallel"]["NCPU"])
        self.BestChi2=1.
        # IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
        # ListPixData=IncreaseIslandMachine.IncreaseIsland(ListPixData,dx=5)

        #ListPixParms=ListPixData
        _,_,nx,_=Dirty.shape
        
        self.ListPixParms=ListPixParms
        self.ListPixData=ListPixData
        self.NPixListParms=len(ListPixParms)
        self.NPixListData=len(ListPixData)
        
        self.ConvMode="Matrix"
        if self.NPixListData>self.GD["GAClean"]["ConvFFTSwitch"]:
            self.ConvMode="FFT"

        self.ConvMachine=ClassConvMachine.ClassConvMachine(PSF,ListPixParms,ListPixData,self.ConvMode)
        
        
        
        self.GD=GD
        self.WeightMaxFunc=collections.OrderedDict()
        #self.WeightMaxFunc["Chi2"]=1.

        for key in self.GD["GAClean"]["GACostFunc"]:
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
        self.PM=ClassParamMachine(ListPixParms,ListPixData,FreqsInfo,SolveParam=GD["GAClean"]["GASolvePars"])

        self.PM.setFreqs(FreqsInfo)
        self.ConvMachine.setParamMachine(self.PM)
        
        self.NParms=self.NPixListParms*self.PM.NParam
        self.DataTrue=None


        self.SetDirtyArrays(Dirty)
        self.InitWorkers()


        #pylab.figure(3,figsize=(5,3))
        #pylab.clf()
        # pylab.figure(4,figsize=(5,3))
        # pylab.clf()

    


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
                
                if self.GD["GAClean"]["ArtifactRobust"]:
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

    

    def ToConvArray(self,V,OutMode="Data"):
        A=self.PM.GiveModelArray(V)
        A=self.ConvMachine.Convolve(A,OutMode=OutMode)
        return A

    def DeconvCLEAN(self,gain=0.1,StopThFrac=0.01,NMaxIter=20000):

        PSF=self.PSF/np.max(self.PSF)

        if self.ConvMachine.ConvMode=="Matrix" or  self.ConvMachine.ConvMode=="Vector":
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

    def GiveDecreteFitNess(self,ContinuousFitNess):
        # M=np.concatenate([0.01*np.abs(self.BestContinuousFitNess),self.MinVar]).reshape((2,self.NFuncMin))
        # Sig=np.max(M,axis=0)
        # Sig.fill(.01)
        Sig=self.MinVar

        sh=ContinuousFitNess.shape
        #d=(ContinuousFitNess-self.BestContinuousFitNess)/Sig
        d=(ContinuousFitNess)/Sig
        DecreteFitNess=np.array(np.int64(np.round(d))).reshape(sh)
        #DecreteFitNess=np.array(d).reshape(sh)

        # print "=============================="
        # print "Best",self.BestContinuousFitNess
        # print "In  ",ContinuousFitNess
        # print "Sig ",Sig
        # print "Out ",DecreteFitNess
        return DecreteFitNess




    def InitWorkers(self):
        import Queue
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
                            iIsland=self.iIsland,
                            ListPixParms=self.ListPixParms,
                            ListPixData=self.ListPixData,
                            GD=self.GD,
                            IdSharedMem=self.IdSharedMem,
                            PauseOnStart=False,
                            PM=self.PM,
                            PixVariance=self.PixVariance,
                            MaxFunc=self.MaxFunc,
                            WeightMaxFunc=self.WeightMaxFunc,
                            DirtyArray=self.DirtyArray,
                            ConvMode=self.ConvMode,
                            StopWhenQueueEmpty=not(Parallel),
                            BestChi2=self.BestChi2)

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


    def GiveFitnessPop(self,pop):

        work_queue=self.work_queue
        result_queue=self.result_queue
        workerlist=self.workerlist
        Parallel=self.ParallelFitness
        DicoFitnesses={}
        DicoChi2={}
        NJobs = len(pop)
        NCPU=self.NCPU

        for iIndividual,individual in enumerate(pop):
            NpShared.ToShared("%sIsland_%5.5i_Individual_%4.4i"%(self.IdSharedMem,self.iIsland,iIndividual),individual)
            work_queue.put({"iIndividual":iIndividual,"BestChi2":self.BestChi2})

        
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

        for iIndividual,individual in enumerate(pop):
            NpShared.DelArray("%sIsland_%5.5i_Individual_%4.4i"%(self.IdSharedMem,self.iIsland,iIndividual))

        self.BestChi2=np.min(Chi2)

        #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        #print "Best chi2 %f"%self.BestChi2

        return fitnesses,Chi2



    
    def testMovePix(self):
        A=np.random.randn(self.PM.NParam,self.NPixListParms)
        A.fill(0.)
        A[:,10]=1.
        print A.shape

        ArrayModel=self.PM.GiveModelArray(A)
        A0=self.PM.ModelToSquareArray(ArrayModel,TypeInOut=("Parms","Parms"),DomainOut="Parms").copy()

        import pylab
        for reg in np.linspace(0,0.99,8):

            pylab.clf()
            pylab.imshow(A0[0,0],interpolation="nearest",vmax=1.)
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)


            
            A1=self.MovePix(A.copy().ravel(),10,InReg=reg)

            ArrayModel=self.PM.GiveModelArray(A1)
            A1=self.PM.ModelToSquareArray(ArrayModel,TypeInOut=("Parms","Parms"),DomainOut="Parms").copy()

            #pylab.subplot(1,2,1)
            #pylab.subplot(1,2,2)
            pylab.imshow(A1[0,0],interpolation="nearest",vmax=1.)
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)
    
    def MovePix(self,indiv,iPix,Flux,FluxWeighted=True,InReg=None):
    
        
        dx,dy=np.mgrid[-1:1:3*1j,-1:1:3*1j]
        Dx=np.int32(np.concatenate((dx.flatten()[0:4],dx.flatten()[5::])))
        Dy=np.int32(np.concatenate((dy.flatten()[0:4],dy.flatten()[5::])))

        
        
        # ArrayModel=self.PM.GiveModelArray(indiv)
        # ArrayModel_S=self.PM.ArrayToSubArray(indiv,Type="S")
        ArrayModel_S=indiv # ArrayModel_S.reshape((1,ArrayModel_S.size))*np.ones((2,1))
        A=self.PM.ModelToSquareArray(ArrayModel_S,TypeInOut=("Parms","Parms"),DomainOut="Parms")
        
        nf,npol,nx,nx=A.shape
        #A=np.mean(A,axis=0).reshape(1,npol,nx,nx)

        mapi,mapj=self.PM.SquareGrids["Parms"]["MappingIndexToXYPix"]
        i0,j0=mapi[iPix],mapj[iPix]
        FluxWeighted=False
        if FluxWeighted:
            iAround=i0+Dx
            jAround=j0+Dy
            cx=((iAround>=0)&(iAround<nx))
            cy=((jAround>=0)&(jAround<nx))
            indIN=(cx&cy)
            iAround=iAround[indIN]
            jAround=jAround[indIN]
            sAround=A[0,0,iAround,jAround].copy()
            #sInt=np.sum(sAround)
            #sAround[sAround==0]=sInt*0.05
            X=np.arange(iAround.size)
            DM=ClassDistMachine()
            DM.setRefSample(X,W=sAround,Ns=10)
            ind=int(round(DM.GiveSample(1)[0]))
            ind=np.max([0,ind])
            ind=np.min([ind,iAround.size-1])
            ind=indIN[ind]

        # else:
        #     if InReg is None:
        #         reg=random.random()
        #     else:
        #         reg=InReg
        #     ind=int(reg*8)

        
        
        i1=i0+Dx[InReg]
        j1=j0+Dy[InReg]

            

        f0=Flux#alpha*A[0,0,i0,j0]
        
        _,_,nx,ny=A.shape
        condx=((i1>0)&(i1<nx))
        condy=((j1>0)&(j1<ny))
        if condx&condy:
            A[0,0,i1,j1]+=f0
            A[0,0,i0,j0]-=f0
            AParm=self.PM.SquareArrayToModel(A,TypeInOut=("Parms","Parms"))

            ArrayModel_S.flat[:]=AParm.flat[:]
        return indiv
    
    
    def mutGaussian(self,individual, pFlux, p0, pMove):
        #return individual,
        T= ClassTimeIt.ClassTimeIt()
        T.disable()
        size = len(individual)
        #mu = repeat(mu, size)
        #sigma = repeat(sigma, size)

        T.timeit("start0")
        #A0=IndToArray(individual).copy()
        Ps=np.array([pFlux, p0, pMove])
        _p0=p0/np.sum(Ps)
        _pMove=pMove/np.sum(Ps)
        _pFlux=pFlux/np.sum(Ps)
    
        T.timeit("start1")
        
        Af=self.PM.ArrayToSubArray(individual,"S")
        index=np.arange(Af.size)
        ind=np.where(Af!=0.)[0]
        NNonZero=(ind.size)
        if NNonZero==0: return individual,

        T.timeit("start2")
    
        RType=random.random()
        T.timeit("start3")
        if RType < _pFlux:
            Type=0
            N=1
            N=int(random.uniform(0, 3.))
        elif RType < _pFlux+_pMove:
            Type=1
            N=np.max([(NNonZero/10),1])
        else:
            Type=2
            N=1
            N=int(random.uniform(0, 3.))
            
            # InReg=random.uniform(-1,1)
            # if InReg<0:
            #     InReg=-1


    
        indR=sorted(list(set(np.int32(np.random.rand(N)*NNonZero).tolist())))
        indSel=ind[indR]
        #Type=0

        #print "Type:",Type

        for iPix in indSel:
            #print iPix,Type

            # randomly change value of parameter
            if Type==0:
                iTypeParm=int(np.random.rand(1)[0]*len(self.PM.SolveParam))
                
                for TypeParm in self.PM.SolveParam:
#                for TypeParm in [self.PM.SolveParam[iTypeParm]]:
                    A=self.PM.ArrayToSubArray(individual,TypeParm)
                    #if TypeParm=="GSig": continue
                    if TypeParm=="S":
                        ds=0.1*np.abs(self.DirtyArrayAbsMean.ravel()[iPix]-np.abs(A[iPix]))
                    else:
                        if "Sigma" in self.PM.DicoIParm[TypeParm]["Default"].keys():
                            ds=self.PM.DicoIParm[TypeParm]["Default"]["Sigma"]["Value"]
                        else:
                            ds=A[iPix]
                    #print "Mutating %f"%A[iPix],TypeParm
                    A[iPix] += random.gauss(0, 1.)*ds
                    #print "      ---> %f"%A[iPix]

            # zero a pixel
            if Type==1:
                for TypeParm in self.PM.SolveParam:
                    A=self.PM.ArrayToSubArray(individual,TypeParm)
                    A[iPix] = 0.#1e-3
    
            # move a pixel
            if Type==2:
                Flux=random.random()*Af[iPix]
                InReg=random.random()*8
                individual=self.MovePix(individual,iPix,Flux,InReg=InReg)

                # if random.random()<0.5:
                #     Flux=random.random()*Af[iPix]
                #     InReg=random.random()*8
                #     individual=self.MovePix(individual,iPix,Flux,InReg=InReg)
                # else:
                #     Flux=random.random()*0.3*Af[iPix]
                #     for iReg in [1,3,5,7]:
                #         individual=self.MovePix(individual,iPix,Flux,InReg=iReg)
                    
        if "GSig" in self.PM.SolveParam:
            GSig=self.PM.ArrayToSubArray(individual,"GSig")
            GSig[GSig<0]=0
            #GSig.fill(0)
            # GSig[49]=1
            # #print GSig
            # # if individual[i]==0:
            # #     if random.random() < indpb/100.:
            # #         individual[i] += random.gauss(m, s)
    
        T.timeit("for")
    
        # if Type==2:
        #     A1=IndToArray(individual).copy()
        #     v0,v1=A0.min(),A0.max()
        #     import pylab
        #     pylab.clf()
        #     #pylab.subplot(1,2,1)
        #     pylab.imshow(A0[0,0],interpolation="nearest",vmin=v0,vmax=v1)
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        #     #pylab.subplot(1,2,2)
        #     pylab.imshow(A1[0,0],interpolation="nearest",vmin=v0,vmax=v1)
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
    
    
        return individual,
    
    
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

        pylab.figure(30,figsize=(5,3))
        #pylab.clf()
        S=self.PM.ArrayToSubArray(V,"S")
        Al=self.PM.ArrayToSubArray(V,"Alpha")

        pylab.subplot(1,2,1)
        pylab.plot(S)
        pylab.subplot(1,2,2)
        pylab.plot(Al)
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        

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
    
    
        pylab.suptitle('Population generation %i [%f]'%(iGen,best_ind.fitness.values[0]),size=16)
        #pylab.tight_layout()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        fig.savefig("png/fig%2.2i_%4.4i.png"%(iChannel,iGen))

# #################################################################"    
# #################################################################"    
# #################################################################"    
# #################################################################"    


class WorkerFitness(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 iIsland=None,
                 ListPixParms=None,
                 ListPixData=None,
                 GD=None,
                 DicoImager=None,
                 IdSharedMem=None,
                 PauseOnStart=False,
                 PM=None,
                 PixVariance=1e-2,
                 MaxFunc=None,WeightMaxFunc=None,DirtyArray=None,
                 ConvMode=None,
                 StopWhenQueueEmpty=False,
                 BestChi2=1.):
        self.T=ClassTimeIt.ClassTimeIt("WorkerFitness")
        self.T.disable()
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.IdSharedMem=IdSharedMem
        self._pause_on_start = PauseOnStart
        self.GD=GD
        self.PM=PM
        self.ListPixParms=ListPixParms
        self.ListPixData=ListPixData
        self.iIsland=iIsland
        self.PSF=NpShared.GiveArray("%sPSF_Island_%4.4i"%(IdSharedMem,iIsland))
        self.PixVariance=PixVariance
        self.ConvMachine=ClassConvMachine.ClassConvMachine(self.PSF,self.ListPixParms,self.ListPixData,ConvMode)
        self.ConvMachine.setParamMachine(self.PM)
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

            self.T.reinit()
            iIndividual=DicoJob["iIndividual"]
            self.BestChi2=DicoJob["BestChi2"]
            #individual=DicoJob["individual"]
            Name="%sIsland_%5.5i_Individual_%4.4i"%(self.IdSharedMem,self.iIsland,iIndividual)
            individual=NpShared.GiveArray(Name)
            fitness,Chi2=self.GiveFitness(individual)
            #print iIndividual
            self.result_queue.put({"Success": True, 
                                   "iIndividual": iIndividual,
                                   "fitness":fitness,
                                   "Chi2":Chi2})
            pid=str(multiprocessing.current_process())
            self.T.timeit("done job: %s"%pid)

    def ToConvArray(self,V,OutMode="Data"):
        self.ModelA=self.PM.GiveModelArray(V)
        A=self.ConvMachine.Convolve(self.ModelA,OutMode=OutMode)
        return A



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
                BIC=chi2+self.GD["GAClean"]["BICFactor"]*k*np.log(n)
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(-BIC*W)
            if FuncType=="MEM":
                aS=np.abs(self.ModelA)
                
                aS[aS==0]=1e-10
                #aS/=np.sum(aS)
                E=-np.sum(aS*np.log10(aS))
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
                FNeg=-np.sum(SNegArr)/(np.sqrt(self.PixVariance))
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(FNeg*W)

        return (np.sum(ContinuousFitNess),),chi2
        #return (ContinuousFitNess,),chi2


