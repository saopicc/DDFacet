import numpy as np
from DDFacet.Imager.SSD import ClassArrayMethodSSD


def FilterIslandsPix(ListIn,Npix):
    ListOut=[]
    for x,y in ListIn:
        Cx=((x>=0)&(x<Npix))
        Cy=((y>=0)&(y<Npix))
        if (Cx&Cy):
            ListOut.append([x,y])
    return ListOut

from scipy.stats import chi2

class ClassMetropolis():
    def __init__(self,Dirty,PSF,FreqsInfo,ListPixData=None,ListPixParms=None,IslandBestIndiv=None,GD=None,
                 WeightFreqBands=None,PixVariance=1e-2,iFacet=0,iIsland=None,IdSharedMem="",
                 ParallelFitness=False,NChains=1,NIter=1000):
        _,_,NPixPSF,_=PSF.shape
        if ListPixData is None:
            x,y=np.mgrid[0:NPixPSF:1,0:NPixPSF:1]
            ListPixData=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()
        if ListPixParms is None:
            x,y=np.mgrid[0:NPixPSF:1,0:NPixPSF:1]
            ListPixParms=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()
        self.IslandBestIndiv=IslandBestIndiv

        _,_,Npix,_=Dirty.shape
        ListPixData=FilterIslandsPix(ListPixData,Npix)
        ListPixParms=FilterIslandsPix(ListPixParms,Npix)
        self.GD=GD
        self.NChains=NChains
        self.NIter=NIter
        
        self.PixVariance=PixVariance
        self.IdSharedMem=IdSharedMem
        self.iIsland=iIsland


        #print "PixVariance",PixVariance

        GD["SSDClean"]["SSDCostFunc"]=["Sum2"]
        self.ArrayMethodsMachine=ClassArrayMethodSSD.ClassArrayMethodSSD(Dirty,PSF,ListPixParms,ListPixData,FreqsInfo,
                                                                         PixVariance=PixVariance,
                                                                         #PixVariance=1.,
                                                                         iFacet=iFacet,
                                                                         IslandBestIndiv=IslandBestIndiv,
                                                                         GD=GD,
                                                                         WeightFreqBands=WeightFreqBands,
                                                                         iIsland=iIsland,
                                                                         IdSharedMem=IdSharedMem,
                                                                         ParallelFitness=ParallelFitness,
                                                                         NCPU=NChains)
        self.InitChain()

    def InitChain(self):

        df=self.ArrayMethodsMachine.PM.NPixListData
        self.rv = chi2(df)

        self.pop=[]
        
        Point0=self.ArrayMethodsMachine.PM.GiveIndivZero()
        self.pop=[Point0]*self.NChains # is the starting chain
        #print self.IslandBestIndiv
        #print
        if self.IslandBestIndiv is not None:
            if np.max(np.abs(self.IslandBestIndiv))==0:
                #print "from clean"
                SModelArray,Alpha=self.ArrayMethodsMachine.DeconvCLEAN()
                self.ArrayMethodsMachine.PM.ReinitPop(self.pop,SModelArray,PutNoise=False)
            else:
                #print "from previous"
                SModelArrayBest=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.IslandBestIndiv,"S")
                AlphaModel=None
                if "Alpha" in self.ArrayMethodsMachine.PM.SolveParam:
                    AlphaModel=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.IslandBestIndiv,"Alpha")
                GSigModel=None
                if "GSig" in self.ArrayMethodsMachine.PM.SolveParam:
                    GSigModel=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.IslandBestIndiv,"GSig")
                self.ArrayMethodsMachine.PM.ReinitPop(self.pop,SModelArrayBest,AlphaModel=AlphaModel,GSigModel=GSigModel,PutNoise=False)

            SModelArray=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.pop[0],"S")
            self.TotFlux0=np.sum(SModelArray)
        #print
        _,Chi2=self.ArrayMethodsMachine.GiveFitnessPop(self.pop)
        logProb=[self.rv.logpdf(x) for x in Chi2]

        # MaxP=



        # Zero=self.ArrayMethodsMachine.PM.GiveIndivZero()
        # RealNoise=np.array([self.ArrayMethodsMachine.ToConvArray(Zero,OutMode="Data",Noise=self.PixVariance).reshape((-1,)) for i in range(10)])
        
        # stop

        #stop
        #for 
        #ClassConvMachine():
        #def __init__(self,PSF,ListPixParms,ListPixData,ConvMode):



        x=np.linspace(0,2*self.rv.moment(1),1000)
        lP=self.rv.logpdf(x)
        iMax=np.argmax(lP)
        self.Chi2PMax=x[iMax]
        self.MinChi2=np.min(Chi2)
        self.Var=self.MinChi2/self.Chi2PMax
        DicoChains={}
        for iChain in range(self.NChains):
            DicoChains[iChain]={}
            Parms=self.pop[iChain]
            DicoChains[iChain]["Parms"]=[Parms]
            DicoChains[iChain]["Chi2"]=[Chi2[iChain]]
            DicoChains[iChain]["logProb"]=[self.rv.logpdf(Chi2[iChain]/self.Var)]
        self.DicoChains=DicoChains

        # self.PlotPDF()

    def PlotPDF(self):
        import pylab
        x=np.linspace(0,2*self.rv.moment(1),1000)
        P=self.rv.pdf(x)

        pylab.clf()
        pylab.plot(x,P)
        #pylab.scatter(self.Chi2PMax,P[iMax])
        Chi2Red=self.DicoChains[0]["Chi2"][0]/self.Var
        pylab.scatter(Chi2Red,np.mean(P),c="black")
        pylab.draw()
        pylab.show(False)
        # stop
        
    def StackChain(self):
        P0=self.ArrayMethodsMachine.PM.GiveIndivZero()
        P=P0.copy()
        Model=self.ArrayMethodsMachine.PM.GiveModelArray(P0)
        N=0
        Chi2Min=1e10
        for iChain in range(self.NChains):
            ChainParms=self.DicoChains[iChain]["Parms"]
            ChainChi2=self.DicoChains[iChain]["Chi2"]
            #print len(ChainParms),len(ChainChi2)
            for iPoint in range(len(ChainParms)):
                P+=ChainParms[iPoint]
                #Model+=self.ArrayMethodsMachine.PM.GiveModelArray(ChainParms[iPoint])
                Chi2=ChainChi2[iPoint]
                if Chi2<Chi2Min:
                    Chi2Min=Chi2
                    Pmin=ChainParms[iPoint]
                N+=1
        Model/=N
        P/=N

        sP=P0.copy()
        for iChain in range(self.NChains):
            ChainParms=self.DicoChains[iChain]["Parms"]
            for iPoint in range(len(ChainParms)):
                sP+=(ChainParms[iPoint]-P)**2
                N+=1
        sP=np.sqrt(sP)/N
        
        return Model,P,Pmin,sP


    def main(self,NSteps=1000):
        self.DicoChains=self.ArrayMethodsMachine.GiveMetroChains(self.pop,NSteps=NSteps)
        self.ArrayMethodsMachine.KillWorkers()
        Model,V,Vmin,sV=self.StackChain()

        SModelArray1=self.ArrayMethodsMachine.PM.ArrayToSubArray(V,"S")
        TotFlux1=np.sum(SModelArray1)
        SModelArray1*=TotFlux1/self.TotFlux0

        
        if "Alpha" in self.ArrayMethodsMachine.PM.SolveParam:
            sAlpha=self.ArrayMethodsMachine.PM.ArrayToSubArray(sV,"Alpha")
            sAlpha.fill(0)

        return V,sV



    def mainSerial(self,NSteps=1000):


        Mut_pFlux, Mut_p0, Mut_pMove=0.,0.,0.3
        DicoChains=self.DicoChains
        
        FactorAccelerate=1.
        lAccept=[]
        for iStep in range(NSteps):
            #print "========================"

            for iChain in range(self.NChains):
                self.pop[iChain],=self.ArrayMethodsMachine.mutGaussian(self.pop[iChain].copy(), 
                                                                       Mut_pFlux, Mut_p0, Mut_pMove,
                                                                       FactorAccelerate=FactorAccelerate)

            _,Chi2=self.ArrayMethodsMachine.GiveFitnessPop(self.pop)
            if np.min(Chi2)<self.MinChi2:
                self.Var=np.min(Chi2)/self.Chi2PMax
                #print "           >>>>>>>>>>>>>> %f"%np.min(Chi2)


            Chi2Norm=[]
            for iChain in range(self.NChains):
                Chi2Norm.append(Chi2[iChain]/self.Var)
                
            logProb=[self.rv.logpdf(x) for x in Chi2Norm]
            
            for iChain in range(self.NChains):
                p1=logProb[iChain]
                p0=DicoChains[iChain]["logProb"][-1]
                if p1-p0<5:
                    R=np.min([1.,np.exp(p1-p0)])
                else:
                    R=1
                r=np.random.rand(1)[0]
                # print "%5.3f [%f -> %f]"%(R,p0,p1)
                # print "MaxDiff ",np.max(np.abs(self.pop[iChain]-DicoChains[iChain]["Parms"][-1]))
                lAccept.append((r<R))
                if r<R: # accept
                    DicoChains[iChain]["logProb"].append(p1)
                    DicoChains[iChain]["Parms"].append(self.pop[iChain])
                    DicoChains[iChain]["Chi2"].append(Chi2[iChain])
                    

                    #pylab.scatter([Chi2Norm[iChain]],[np.exp(p1)],lw=0)
                    #pylab.draw()
                    #pylab.show(False)
                    #pylab.pause(0.1)

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
                    self.pop[iChain]=DicoChains[iChain]["Parms"][-1]
                    # pylab.scatter([Chi2Norm[iChain]],[np.exp(p1)],c="red",lw=0)
                    # pylab.draw()
                    # pylab.show(False)
                    # pylab.pause(0.1)


            AccRate=np.count_nonzero(lAccept)/float(len(lAccept))
            # print "[%i] Acceptance rate %f [%f]"%(iStep,AccRate,FactorAccelerate)
            if (iStep%50==0)&(iStep>10):
                if AccRate>0.5:
                    FactorAccelerate*=1.5
                else:
                    FactorAccelerate/=1.5
                FactorAccelerate=np.min([3.,FactorAccelerate])
                lAccept=[]



                    
            
