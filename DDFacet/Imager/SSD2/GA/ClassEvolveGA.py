
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from DDFacet.compatibility import range

from deap import base
from deap import creator
from deap import tools
import numpy
from DDFacet.Imager.SSD2.GA import algorithms
import numpy as np
import random
import psutil
from DDFacet.Imager.SSD2 import ClassArrayMethodSSD
from DDFacet.Array import shared_dict
from DDFacet.Other.ClassTimeIt import ClassTimeIt as CTI 
import os
import pylab
from DDFacet.Other import ClassTimeIt

def FilterIslandsPix(ListIn,Npix_x,Npix_y):
    ListOut=[]
    for x,y in ListIn:
        Cx=((x>=0)&(x<Npix_x))
        Cy=((y>=0)&(y<Npix_y))
        if (Cx&Cy):
            ListOut.append([x,y])
    return ListOut

DOPLOT=False

class ClassEvolveGA():
    def __init__(self,Dirty,PSF,FreqsInfo,ListPixData=None,ListPixParms=None,IslandBestIndiv=None,GD=None,
                 WeightFreqBands=None,PixVariance=1e-2,iFacet=0,iIsland=None,island_dict=None,
                 ParallelFitness=False,iIslandInit=None):
        if GD["Misc"]["RandomSeed"] is not None:
            random.seed(int(GD["Misc"]["RandomSeed"]))
            np.random.seed(int(GD["Misc"]["RandomSeed"]))
            
        self.iIslandInit=iIslandInit
        
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
        # self.ArrayMethodsMachine.testMovePix()
        # stop

    def setDEAP(self):
        if "FitnessMax" not in dir(creator):
            creator.create("FitnessMax", base.Fitness, weights=self.ArrayMethodsMachine.WeightsEA)
        if "Individual" not in dir(creator):
            creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Obj=[toolbox.attr_float_unif]*self.ArrayMethodsMachine.NParms
        # Obj=[toolbox.attr_float_unif]*self.ArrayMethodsMachine.NParms
        # Obj=[toolbox.attr_float_normal]*self.ArrayMethodsMachine.NParms

        Obj=self.ArrayMethodsMachine.PM.GiveInitList(toolbox)

        toolbox.register("individual",
                         tools.initCycle,
                         creator.Individual,
                         Obj, n=1)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        
        
        #toolbox.register("evaluate", self.ArrayMethodsMachine.GiveFitness)
        # toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        # toolbox.register("mate", tools.cxOrdered)
        # toolbox.register("mutate", tools.mutGaussian, indpb=0.3,  mu=0.0, sigma=.1)
        #toolbox.register("mutate", self.ArrayMethodsMachine.mutGaussian, pFlux=0.3, p0=0.3, pMove=0.3)
        self.MutConfig=pFlux,p0,pMove,pScale,pOffset=0.5,0.5,0.5,0.5,0.5
        toolbox.register("mutate", self.ArrayMethodsMachine.mutGaussian, pFlux=0.2, p0=0.5, pMove=0.2, pScale=0.2, pOffset=0.2)

        toolbox.register("select", tools.selTournament, tournsize=3)
        # toolbox.register("select", Select.selTolTournament, tournsize=3, Tol=4)
        # toolbox.register("select", tools.selRoulette)

        self.toolbox=toolbox



    def main(self,NGen=1000,NIndiv=100,DoPlot=True):
        T=ClassTimeIt.ClassTimeIt("   GA: Main")
        T.disable()
        self.setDEAP()
        #os.system("rm png/*.png")
        #random.seed(64)
        #np.random.seed(64)
        toolbox=self.toolbox
        
        # pool = multiprocessing.Pool(processes=6)
        # toolbox.register("map", pool.map)
        self.pop = toolbox.population(n=NIndiv)
        self.hof = tools.HallOfFame(1, similar=numpy.array_equal)
        #self.hof = tools.ParetoFront(1, similar=numpy.array_equal)

        # stats = tools.Statistics(lambda ind: ind.fitness.values)
        # stats.register("avg", numpy.mean)
        # stats.register("std", numpy.std)
        # stats.register("min", numpy.min)
        # stats.register("max", numpy.max)


        for indiv in self.pop:
            indiv.fill(0)
        T.timeit("Init")

        #print "Best indiv start",
        #self.ArrayMethodsMachine.PM.PrintIndiv(self.IslandBestIndiv)
        #print
        
        def GiveListPolyArrayMP(N,iTypeInit=None):
            return [GivePolyArrayMP(iTypeInit=iTypeInit) for iIndiv in range(N)]
                
        def GivePolyArrayMP(iTypeInit=None):
            T=CTI("GivePolyArrayMP")
            T.disable()
            NTypeInit=len(self.DicoDicoInitIndiv.keys())
            
            if iTypeInit is None:
                iTypeInit=int(np.random.rand(1)[0]*NTypeInit)
            DicoModelMP=None
            DicoInitIndiv=self.DicoDicoInitIndiv.get(iTypeInit,None)
            if DicoInitIndiv is not None and self.iIslandInit is not None:
                #print(self.ArrayMethodsMachine.PM.NPixListData,self.iIsland,self.iIslandInit)
                DicoModelMP=DicoInitIndiv.get(self.iIslandInit,None)
            T.timeit("Init")
            
            if DicoModelMP is not None:
                PolyModelArrayMP=DicoModelMP["PolyModel"]
                T.timeit("get")
            else:
                SModelArrayMP,_=self.ArrayMethodsMachine.DeconvCLEAN()
                AModelArrayMP=np.zeros_like(SModelArrayMP)
                PolyModelArrayMP=np.zeros((self.ArrayMethodsMachine.PM.NOrderPoly,self.ArrayMethodsMachine.PM.NPixListParms),np.float32)
                PolyModelArrayMP[0,:]=SModelArrayMP
                T.timeit("CLEAN")
            return PolyModelArrayMP

        def GiveListPolyArrayMP_LinComb(N):
            T.reinit()
            
            L=[GivePolyArrayMP_LinComb() for iIndiv in range(N)]
            DoPutNoise=np.ones((N,),bool)
            # T.timeit("GiveListPolyArrayMP_LinComb: L")
            NTypeInit=len(self.DicoDicoInitIndiv.keys())
            for iTypeInit in range(NTypeInit):
                L[iTypeInit]=GivePolyArrayMP(iTypeInit=iTypeInit)
                DoPutNoise[iTypeInit]=False
                
            # # print("VLKFSDLKSFDL")
            # # iDone=0
            # L=[]
            # DoPutNoise=np.ones((N,),bool)
            # NTypeInit=len(self.DicoDicoInitIndiv.keys())
            # LiTypeInit=np.int16(np.arange(0,NTypeInit,NTypeInit/N))
            # iCurrent=None
            # for ii,iTypeInit in enumerate(LiTypeInit):
            #     if iTypeInit!=iCurrent:
            #         iCurrent=iTypeInit
            #         DoPutNoise[ii]=0
            #     L.append(GivePolyArrayMP(iTypeInit=iTypeInit))
            
            DoPutNoise=np.min(np.concatenate([DoPutNoise,
                                              np.int16(np.random.rand(N)*2)]).reshape((2,N)),
                              axis=0)
            #DoPutNoise=np.ones((len(L),),bool)
            #DoPutNoise.fill(0)
            T.timeit("GiveListPolyArrayMP_LinComb: for")
            #DoPutNoise=True
            return L,DoPutNoise

        def GiveInitPop():
            NTypeInit=len(self.DicoDicoInitIndiv.keys())
            L=[]
            if NTypeInit==0:
                # run simplistic clean
                L.append(GivePolyArrayMP())
            else:
                for iTypeInit in range(NTypeInit):
                    L.append(GivePolyArrayMP(iTypeInit=iTypeInit))
                    
            if np.max(np.abs(self.IslandBestIndiv))!=0.:
                L.append(self.IslandBestIndiv_PolyModelArray)
                
            pop = toolbox.population(n=len(L))
            self.ArrayMethodsMachine.PM.ReinitPop(pop,L,PutNoise=False)
            fitnesses,_=self.ArrayMethodsMachine.GiveFitnessPop(pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            pop_init=pop
            if DOPLOT:
                os.system("mkdir PNG")
                for iChannel in range(1):
                    for iType in range(len(pop_init)):
                        iIter=0
                        fig=pylab.figure("Plot indiv",figsize=(10,6))
                        pylab.clf()
                        self.ArrayMethodsMachine.PlotChannel(pop_init[iType:iType+1],0,iChannel=iChannel)
                        while True:
                            FName="PNG/Fig_Ch%i_Type%i_Iter%i.png"%(iChannel,iType,iIter)
                            if not os.path.isfile(FName):
                                break
                            iIter+=1
                        fig.savefig(FName)
                #stop
                
            return pop
        
        def GivePolyArrayMP_LinComb():

            T=CTI("GivePolyArrayMP_LinComb")
            T.disable()
            NTypeInit=len(self.DicoDicoInitIndiv.keys())
            LInit=[]
            Nrand=np.max([1,NTypeInit])
            w=np.random.rand(Nrand)
            w/=np.sum(w)
            T.timeit("Init")
            #print(w)
            #print("Nrand=",Nrand)
            PolyModelArrayMP=w[0]*GivePolyArrayMP(iTypeInit=0)
            T.timeit("Init1")
            for iTypeInit in range(1,Nrand):
                PolyModelArrayMP+=w[iTypeInit]*GivePolyArrayMP(iTypeInit=iTypeInit)
            T.timeit("Init2")
            return PolyModelArrayMP

            
        self.DicoDicoInitIndiv  = shared_dict.attach("DicoDicoInitIndiv")
        self.DicoDicoInitIndiv.reload()
        
        if self.IslandBestIndiv is not None:

            self.IslandBestIndiv_PolyModelArray=np.zeros((self.ArrayMethodsMachine.PM.NOrderPoly,self.ArrayMethodsMachine.PM.NPixListParms),np.float32)
            for iOrder in range(self.ArrayMethodsMachine.PM.NOrderPoly):
                self.IslandBestIndiv_PolyModelArray[iOrder]=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.IslandBestIndiv,"Poly%i"%iOrder)
            
            # SModelArrayMP,Alpha=self.ArrayMethodsMachine.DeconvCLEAN()
            # AModelArrayMP=None
            
            
            if NGen==0:
                #self.ArrayMethodsMachine.PM.ReinitPop(self.pop,GiveListPolyArrayMP(1)*len(self.pop),PutNoise=False)
                #pop,PutNoise=GiveListPolyArrayMP_LinComb(len(self.pop))
                #self.ArrayMethodsMachine.PM.ReinitPop(self.pop,pop,PutNoise=PutNoise)

                pop_init=GiveInitPop()
                # print([ind.fitness.values for ind in pop_init])
                # print([ind.fitness.values for ind in pop_init])
                # print([ind.fitness.values for ind in pop_init])
                V = tools.selBest(pop_init, 1)[0]

                
                self.ArrayMethodsMachine.KillWorkers()
                return V
            T.timeit("N=0")


            if np.max(np.abs(self.IslandBestIndiv))==0:
                #print("NEW")
                pop,PutNoise=GiveListPolyArrayMP_LinComb(len(self.pop))
                #T.timeit("New0")
                self.ArrayMethodsMachine.PM.ReinitPop(self.pop,pop,PutNoise=PutNoise)
                #T.timeit("New")
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
                T.timeit("Mix: split")
                
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
                    T.timeit("Mix: build PolyModelArray")

                        
                GSigModel=None
                if "GSig" in self.ArrayMethodsMachine.PM.SolveParam:
                    GSigModel=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.IslandBestIndiv,"GSig")
                    T.timeit("Mix: GSigModel")

                PutNoise=True
                self.ArrayMethodsMachine.PM.ReinitPop(pop1,[PolyModelArray]*len(pop1),GSigModel=GSigModel,PutNoise=PutNoise)
                pop1[0].flat[:]=BestIndiv.flat[:]
                T.timeit("Mix: ReinitPop pop1")
                
                ##################"
                # From Minor Cycle estimate
                
                # half of the pop with the MP model
                pop,PutNoise=GiveListPolyArrayMP_LinComb( len(pop0) )
                self.ArrayMethodsMachine.PM.ReinitPop(pop0,pop,PutNoise=PutNoise)
                T.timeit("Mix: ReinitPop pop0")

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


        T.timeit("Init pop")


        # if self.IslandBestIndiv is not None:

        #     if np.max(np.abs(self.IslandBestIndiv))==0:
        #         #print "deconv"
        #         SModelArray,Alpha=self.ArrayMethodsMachine.DeconvCLEAN()

        #         #print "Estimated alpha",Alpha
        #         AlphaModel=np.zeros_like(SModelArray)+Alpha
        #         #AlphaModel[SModelArray==np.max(SModelArray)]=0

        #         self.ArrayMethodsMachine.PM.ReinitPop(self.pop,SModelArray)#,AlphaModel=AlphaModel)

        #         #print self.ArrayMethodsMachine.GiveFitness(self.pop[0],DoPlot=True)
        #         #stop
        #         #print self.pop
        #     else:
        #         SModelArray=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.IslandBestIndiv,"S")
        #         AlphaModel=None
        #         if "Alpha" in self.ArrayMethodsMachine.PM.SolveParam:
        #             AlphaModel=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.IslandBestIndiv,"Alpha")
                
        #         GSigModel=None
        #         if "GSig" in self.ArrayMethodsMachine.PM.SolveParam:
        #             GSigModel=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.IslandBestIndiv,"GSig")
                
        #         self.ArrayMethodsMachine.PM.ReinitPop(self.pop,SModelArray,AlphaModel=AlphaModel,GSigModel=GSigModel)

        # set best Chi2
        # _=self.ArrayMethodsMachine.GiveFitnessPop([self.IslandBestIndiv])


        F0,_=self.ArrayMethodsMachine.GiveFitnessPop(self.pop)
        T.timeit("Init Givefitness")
        
        self.pop, log= algorithms.eaSimple(self.pop, toolbox, cxpb=0.3, mutpb=0.5, ngen=NGen, 
                                           halloffame=self.hof, 
                                           #stats=stats,
                                           verbose=False, 
                                           ArrayMethodsMachine=self.ArrayMethodsMachine,
                                           DoPlot=DoPlot,
                                           MutConfig=self.MutConfig)

        pop_init=GiveInitPop()
        
        # # ###############################
        # V = tools.selBest(pop_init, 1)[0]
        # F=[ind.fitness.values for ind in pop_init]
        # iind=np.argmax(F)
        # #for ii in range(len(pop_init)):
        # #    self.ArrayMethodsMachine.PlotChannel(pop_init[ii:ii+1],0,iChannel=0)
        # return V
    
        pop_merge=pop_init+self.pop
        #F1,_=self.ArrayMethodsMachine.GiveFitnessPop(pop_merge)
        V = tools.selBest(pop_merge, 1)[0]
        
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

        
        # import pylab
        # pylab.clf()
        # pylab.plot(F0,ls="-")
        # pylab.plot(F1,ls="-")
        # pylab.draw()
        # pylab.show(block=False)
        # pylab.pause(0.1)
        

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
