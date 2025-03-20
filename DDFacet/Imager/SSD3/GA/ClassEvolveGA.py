
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from DDFacet.compatibility import range

from deap import base
from deap import creator
from deap import tools
import numpy
from DDFacet.Imager.SSD3.GA import algorithms
import numpy as np
import random
import psutil
from DDFacet.Imager.SSD3 import ClassArrayMethodSSD
from DDFacet.Array import shared_dict

def FilterIslandsPix(ListIn,Npix_x,Npix_y):
    ListOut=[]
    for x,y in ListIn:
        Cx=((x>=0)&(x<Npix_x))
        Cy=((y>=0)&(y<Npix_y))
        if (Cx&Cy):
            ListOut.append([x,y])
    return ListOut


class ClassEvolveGA():
    def __init__(self,Dirty,PSF,FreqsInfo,ListPixData=None,ListPixParms=None,IslandBestIndiv=None,GD=None,
                 WeightFreqBands=None,PixVariance=1e-2,iFacet=0,iIsland=None,island_dict=None,
                 ParallelFitness=False):
        if GD["Misc"]["RandomSeed"] is not None:
            random.seed(int(GD["Misc"]["RandomSeed"]))
            np.random.seed(int(GD["Misc"]["RandomSeed"]))
            
        _,_,NPixPSF,_ = PSF.shape
        if ListPixData is None:
            x,y=np.mgrid[0:NPixPSF:1,0:NPixPSF:1]
            ListPixData=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()
        if ListPixParms is None:
            x,y=np.mgrid[0:NPixPSF:1,0:NPixPSF:1]
            ListPixParms=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()
        self.IslandBestIndiv=IslandBestIndiv

        _,_,Npix_x,Npix_y=Dirty.shape
        ListPixData=FilterIslandsPix(ListPixData,Npix_x,Npix_y)
        ListPixParms=FilterIslandsPix(ListPixParms,Npix_x,Npix_y)
        

        self.iIsland=iIsland
        
        NCPU=(GD["GAClean"]["NCPU"] or None)
        if NCPU==0:
            NCPU=int(GD["Parallel"]["NCPU"] or psutil.cpu_count())
        
        self.ArrayMethodsMachine=ClassArrayMethodSSD.ClassArrayMethodSSD(Dirty,PSF,ListPixParms,ListPixData,FreqsInfo,
                                                                         PixVariance=PixVariance,
                                                                         iFacet=iFacet,
                                                                         IslandBestIndiv=IslandBestIndiv,
                                                                         GD=GD,
                                                                         WeightFreqBands=WeightFreqBands,
                                                                         iIsland=iIsland,
                                                                         island_dict=island_dict,
                                                                         ParallelFitness=ParallelFitness,
                                                                         NCPU=NCPU)

        self.setDEAP()
        

    def setDEAP(self):
        if "FitnessMax" not in dir(creator):
            creator.create("FitnessMax", base.Fitness, weights=self.ArrayMethodsMachine.WeightsEA)
        if "Individual" not in dir(creator):
            creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        Obj=self.ArrayMethodsMachine.PM.GiveInitList(toolbox)
        toolbox.register("individual",
                         tools.initCycle,
                         creator.Individual,
                         Obj, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.MutConfig=pFlux,p0,pMove,pScale,pOffset=0.5,0.5,0.5,0.5,0.5
        toolbox.register("mutate", self.ArrayMethodsMachine.mutGaussian, pFlux=0.2, p0=0.5, pMove=0.2, pScale=0.2, pOffset=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox=toolbox

    def main(self,NGen=1000,NIndiv=100,DoPlot=True):
        toolbox=self.toolbox
        self.pop = toolbox.population(n=NIndiv)
        self.hof = tools.HallOfFame(1, similar=numpy.array_equal)
        for indiv in self.pop:
            indiv.fill(0)
            
        # #########################################################
        def GiveListPolyArrayMP(N,iTypeInit=None):
            return [GivePolyArrayMP(iTypeInit=iTypeInit) for iIndiv in range(N)]
                
        def GivePolyArrayMP(iTypeInit=None):
            
            NTypeInit=len(self.DicoDicoInitIndiv.keys())
            
            if iTypeInit is None:
                iTypeInit=int(np.random.rand(1)[0]*NTypeInit)
                
            DicoModelMP=None
            DicoInitIndiv=self.DicoDicoInitIndiv.get(iTypeInit,None)
            if DicoInitIndiv is not None:
                DicoModelMP=DicoInitIndiv.get(self.iIsland,None)
            
            if DicoModelMP is not None:
                PolyModelArrayMP=DicoModelMP
            else:
                SModelArrayMP,_=self.ArrayMethodsMachine.DeconvCLEAN()
                AModelArrayMP=np.zeros_like(SModelArrayMP)
                PolyModelArrayMP=np.zeros((self.ArrayMethodsMachine.PM.NOrderPoly,self.ArrayMethodsMachine.PM.NPixListParms),np.float32)
                PolyModelArrayMP[0,:]=SModelArrayMP
            return PolyModelArrayMP

        def GiveListPolyArrayMP_LinComb(N):
            L=[GivePolyArrayMP_LinComb() for iIndiv in range(N)]
            NTypeInit=len(self.DicoDicoInitIndiv.keys())
            for iTypeInit in range(NTypeInit):
                L[iTypeInit]=GivePolyArrayMP(iTypeInit=iTypeInit)
            return L
        
        def GivePolyArrayMP_LinComb():
            NTypeInit=len(self.DicoDicoInitIndiv.keys())
            LInit=[]
            Nrand=np.max([1,NTypeInit])
            w=np.random.rand(Nrand)
            w/=np.sum(w)
            #print(w)
            PolyModelArrayMP=w[0]*GivePolyArrayMP(iTypeInit=0)
            for iTypeInit in range(1,Nrand):
                PolyModelArrayMP+=w[iTypeInit]*GivePolyArrayMP(iTypeInit=iTypeInit)
            return PolyModelArrayMP
        # #########################################################

            
        self.DicoDicoInitIndiv  = shared_dict.attach("DicoDicoInitIndiv")
        self.DicoDicoInitIndiv.reload()
        
        if self.IslandBestIndiv is not None:
            if NGen==0:
                self.ArrayMethodsMachine.PM.ReinitPop(self.pop,GiveListPolyArrayMP_LinComb(len(self.pop)),PutNoise=False)
                self.ArrayMethodsMachine.KillWorkers()
                return self.pop[0]


            PutNoise=True#False
            if np.max(np.abs(self.IslandBestIndiv))==0:
                #print("NEW")
                ListPolyModelArrayMP=GiveListPolyArrayMP_LinComb(len(self.pop))
                self.ArrayMethodsMachine.PM.ReinitPop(self.pop,ListPolyModelArrayMP,PutNoise=PutNoise)
            else:
                #print("MIX")
                NIndiv=len(self.pop)//10
                pop0=self.pop[0:NIndiv]
                pop1=self.pop[NIndiv::]

                pop1=self.pop
                pop0=[]

                pop1=self.pop[0:1]
                pop0=self.pop[1::]
                
                pop1=self.pop[0:NIndiv//2]
                pop0=self.pop[NIndiv//2::]
                
                BestIndiv=self.IslandBestIndiv.copy()
                
                # self.ArrayMethodsMachine.PM.ReinitPop(pop0,SModelArray)

                # print("Best!!!",BestIndiv)
                # print("Best!!!",BestIndiv)
                # print("Best!!!",BestIndiv)
                # print("Best!!!",BestIndiv)
                # print("Best!!!",BestIndiv)
                # print("Best!!!",BestIndiv)
                # print("Best!!!",BestIndiv)


                ##################"
                # BEST
                # half with the best indiv
                PolyModelArray=None
                if True:#"Poly1" in self.ArrayMethodsMachine.PM.SolveParam:
                    PolyModelArray=np.zeros((self.ArrayMethodsMachine.PM.NOrderPoly,self.ArrayMethodsMachine.PM.NPixListParms),np.float32)
                    for iOrder in range(self.ArrayMethodsMachine.PM.NOrderPoly):
                        PolyModelArray[iOrder]=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.IslandBestIndiv,"Poly%i"%iOrder)

                        
                GSigModel=None
                if "GSig" in self.ArrayMethodsMachine.PM.SolveParam:
                    GSigModel=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.IslandBestIndiv,"GSig")

                self.ArrayMethodsMachine.PM.ReinitPop(pop1,[PolyModelArray]*len(pop1),GSigModel=GSigModel,PutNoise=PutNoise)
                pop1[0].flat[:]=BestIndiv.flat[:]
                
                ##################"
                # From Minor Cycle estimate
                
                # half of the pop with the MP model
                self.ArrayMethodsMachine.PM.ReinitPop(pop0,GiveListPolyArrayMP_LinComb( len(pop0) ),PutNoise=PutNoise)

                # NTypeInit=len(self.DicoDicoInitIndiv.keys())
                # for iTypeInit in range(NTypeInit):
                #     pop0a=pop0[iTypeInit:iTypeInit+1]
                #     self.ArrayMethodsMachine.PM.ReinitPop(pop0a,GiveListPolyArrayMP( len(pop0a) , iTypeInit=iTypeInit),PutNoise=False)
                
                    
                # _,Chi20=self.ArrayMethodsMachine.GiveFitnessPop(pop0)
                # _,Chi21=self.ArrayMethodsMachine.GiveFitnessPop(pop1)
                # print
                # print Chi20
                # print Chi21
                # stop



                self.pop=pop1+pop0
                #print(self.pop)
                #stop
        #print



        _=self.ArrayMethodsMachine.GiveFitnessPop(self.pop)



        self.pop, log= algorithms.eaSimple(self.pop, toolbox, cxpb=0.3, mutpb=0.5, ngen=NGen, 
                                           halloffame=self.hof, 
                                           #stats=stats,
                                           verbose=False, 
                                           ArrayMethodsMachine=self.ArrayMethodsMachine,
                                           DoPlot=DoPlot,
                                           MutConfig=self.MutConfig)

        self.ArrayMethodsMachine.KillWorkers()

        # #:param mu: The number of individuals to select for the next generation.
        # #:param lambda\_: The number of children to produce at each generation.
        # #:param cxpb: The probability that an offspring is produced by crossover.
        # #:param mutpb: The probability that an offspring is produced by mutation.

        # mu=70
        # lambda_=50
        # cxpb=0.3
        # mutpb=0.5
        # ngen=1000

        # self.pop, log= algorithms.eaMuPlusLambda(self.pop, toolbox, mu, lambda_, cxpb, mutpb, ngen,
        #                               stats=None, halloffame=None, verbose=__debug__,
        #                               ArrayMethodsMachine=self.ArrayMethodsMachine)

        V = tools.selBest(self.pop, 1)[0]

        #print "Best indiv end"
        #self.ArrayMethodsMachine.PM.PrintIndiv(V)
        
        # V.fill(0)
        # S=self.ArrayMethodsMachine.PM.ArrayToSubArray(V,"S")
        # G=self.ArrayMethodsMachine.PM.ArrayToSubArray(V,"GSig")
        
        # S[0]=1.
        # #S[1]=2.
        # G[0]=1.
        # #G[1]=2.

        # MA=self.ArrayMethodsMachine.PM.GiveModelArray(V)

        # # print "Sum best indiv",MA.sum(axis=1)
        # # print "Size indiv",V.size
        # # print "indiv",V
        # # print self.ArrayMethodsMachine.ListPixData
        # # print MA[0,:]

        return V
