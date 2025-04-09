
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from DDFacet.compatibility import range

from DDFacet.Other import ClassTimeIt
from DDFacet.Other.ClassTimeIt import ClassTimeIt as CTI 

from deap import base
from deap import creator
from deap import tools
import numpy
from DDFacet.Imager.SSD3.GA import algorithms
import numpy as np
import random
import psutil
from DDFacet.Imager.SSD3 import ClassArrayMethodSSD
#from DDFacet.Imager.SSD3 import ClassImageArrayMethodSSD
from DDFacet.Array import shared_dict
from DDFacet.Imager.SSD3 import ClassImageDeconvMachineSSD
import DDFacet.Other.AsyncProcessPool
from SkyModel.PSourceExtract import ClassIncreaseIsland

def FilterIslandsPix(ListIn,Npix_x,Npix_y):
    ListOut=[]
    for x,y in ListIn:
        Cx=((x>=0)&(x<Npix_x))
        Cy=((y>=0)&(y<Npix_y))
        if (Cx&Cy):
            ListOut.append([x,y])
    return ListOut

SERIAL=True
SERIAL=False


class ClassEvolveGA():
    def __init__(self,ImageDeconvMachine):
        self.__dict__ = ImageDeconvMachine.__dict__
        self.ImageDeconvMachine=ImageDeconvMachine
        
        self.APP_GA=DDFacet.Other.AsyncProcessPool.initNew(Name="APP_GA",
                                                           ncpu=self.GD["Parallel"]["NCPU"],
                                                           affinity="disable",#self.GD["Parallel"]["Affinity"],
                                                           #parent_affinity=self.GD["Parallel"]["MainProcessAffinity"],
                                                           #verbose=self.GD["Debug"]["APPVerbose"],
                                                           #pause_on_start=self.GD["Debug"]["PauseWorkers"]
                                                           )
        self.APP_GA.registerJobHandlers(self)
        self.APP_GA.startWorkers()
        #print("LKLKFLSKFD")
        
    def runGA_AllIslands(self):
        APP=self.APP_GA
        
        T=ClassTimeIt.ClassTimeIt("runGA_AllIslands")
        T.disable()
        for iIsland,Island in enumerate(self.ListIslands):
            #IslandBestIndiv=self.ModelMachine.GiveIndividual(self.ListIslands[iIsland])
            APP.runJob("runGA.%i"%(iIsland),
                             self._runGA,
                             args=(#self.ListIslands,
                                   iIsland,
                                   #IslandBestIndiv,
                                   self.DicoDirty.path,self.GridFreqs,self.DegridFreqs), serial=SERIAL)
        LDicoResults=APP.awaitJobResults("runGA.*", progress="Genetic Alg.")
        T.timeit("runGA")

        allIslandModelDict  = shared_dict.attach("DeconvListIslands%s"%self.StrField)
        allIslandModelDict.reload()
        
        self.ModelMachine.reinitIslands(self.ListIslands)
        
        for iRes,DicoResult in enumerate(LDicoResults):
            iIsland=DicoResult["iIsland"]
            ThisIslandModelDict = allIslandModelDict[iIsland]
            ThisIslandModelDict.reload()
            self.ModelMachine.AppendIsland(self.ListIslands[iIsland], ThisIslandModelDict["Model"].copy(),W=self.ListSpacialWeight[iIsland])
            if DicoResult["HasError"]:
                self.ErrorModelMachine.AppendIsland(ListIslands[iIsland], ThisIslandModelDict["sModel"].copy())
        
        APP.terminate()
        APP.shutdown()
        del(self.APP_GA)
        
        self.ModelMachine.RenormaliseMultiEstimatesPerPixel()
        
    def _runGA(self,#ListIslands,
               iIsland,
               #IslandBestIndiv,
               DicoDirty_path,GridFreqs,DegridFreqs):


        #from pympler import tracker
        #tr=tracker.SummaryTracker()

        
        IslandBestIndiv=self.ModelMachine.GiveIndividual(self.ListIslands[iIsland])
        #del(self.ModelMachine)
        
        #tr.print_diff()
        
        #NIslands=len(ListIslands)
        #if NIslands==0: return
        T=ClassTimeIt.ClassTimeIt("  ----  _runGA #%i"%iIsland)
        T.disable()
        self.ImageDeconvMachine._updateWorkerInternals(DicoDirty_path,GridFreqs,DegridFreqs)
        T.timeit("updateWorkerInternals")
        ##tr.print_diff()
        
        ListInitIslands=None
        allIslandModelDict  = shared_dict.attach("DeconvListIslands%s"%self.StrField)
        ThisIslandModelDict=allIslandModelDict[iIsland]
        ThisPixList=ThisIslandModelDict["Island"]
        
        

        XY=np.array(ThisPixList,dtype=np.float32)
        xm,ym=np.mean(np.float32(XY),axis=0).astype(int)
        T.timeit("xm,ym")
        nchan,npol,_,_=self._Dirty.shape
        JonesNorm=(self.DicoDirty["JonesNorm"][:,:,xm,ym]).reshape((nchan,npol,1,1))
        W=self.DicoDirty["WeightChansImages"]
        JonesNorm=np.sum(JonesNorm*W.reshape((nchan,1,1,1)),axis=0).reshape((1,npol,1,1))
        T.timeit("JonesNorm")
        
        ##tr.print_diff()
        
        FacetID=self.PSFServer.giveFacetID2(xm,ym)
        T.timeit("FacetID")

        ThisIslandModelDict["BestIndiv"] = IslandBestIndiv
        
        # ListOrder=[iIsland,FacetID,JonesNorm.flat[0],self.RMS**2,island_dict.path,iIslandInit]
        # ##############################################
        # self.MultiFreqMode=MultiFreqMode
        self.FreqsInfo=self.PSFServer.DicoMappingDesc
        # self._Dirty = Dirty
        self.CubeVariablePSF = self.DicoVariablePSF["CubeVariablePSF"]
        
        ThisPixList = ThisIslandModelDict["Island"]#.tolist()
        IslandBestIndiv = ThisIslandModelDict["BestIndiv"]

        PSF=self.CubeVariablePSF[FacetID]
        NGen=self.GD["GAClean"]["NMaxGen"]
        NIndiv=self.GD["GAClean"]["NSourceKin"]

        ListPixParms=ThisPixList
        ListPixData=ThisPixList
        dx=self.GD["SSDClean"]["NEnlargeData"]
        if dx>0:
            IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland(self.MaskMachine.CurrentNegMask)
            ListPixData,_=IncreaseIslandMachine.IncreaseIsland(ListPixData,AllowMasked=True,dx=dx)

        
        ParmDict = shared_dict.attach("ParmDict%s"%self.StrField) # ParmDict
        PixVariance=ParmDict["RMS"]**2

        ##tr.print_diff()

        # # ##############################
        # np_island_dict={}
        # for k in ThisIslandModelDict.keys():
        #     np_island_dict[k]=ThisIslandModelDict[k].copy()
        # def giveCopy(D):
        #     d={}
        #     import copy
        #     for k in D.keys():
        #         if "SharedDict" in str(type(D[k])):
        #             for kk in D[k].keys():
        #                 d[k]=giveCopy(D[k])
        #         elif "array" in str(type(D[k])):
        #             d[k]=D[k].copy()
        #         else:
        #             d[k]=copy.deepcopy(D[k])
        #     return d
        # np_FreqsInfo=giveCopy(self.FreqsInfo)
        # np.savez("SingleIsland_input_%i.npz"%iIsland,
        #          Dirty=self._Dirty.copy(),
        #          PSF=PSF.copy(),
        #          FreqsInfo=np_FreqsInfo,
        #          ListPixParms=ListPixParms,
        #          ListPixData=ListPixData,
        #          iFacet=FacetID,
        #          PixVariance=PixVariance,
        #          IslandBestIndiv=IslandBestIndiv,
        #          GD=self.GD,
        #          iIsland=iIsland,
        #          island_dict=np_island_dict,
        #          ModelMachine=self.ModelMachine)
        # # ##############################

        
        self.CEv=ClassEvolveGA_SingleIsland(self._Dirty,
                                       PSF,
                                       self.FreqsInfo,
                                       ListPixParms=ListPixParms,
                                       ListPixData=ListPixData,
                                       iFacet=FacetID,PixVariance=PixVariance,
                                       IslandBestIndiv=IslandBestIndiv,#*np.sqrt(JonesNorm),
                                       GD=self.GD,
                                       iIsland=iIsland,
                                       island_dict=ThisIslandModelDict,
                                       ParallelFitness=False)
        Model=self.CEv.main(NGen=NGen,NIndiv=NIndiv,DoPlot=False)
        
        #tr.print_diff()
        ThisIslandModelDict["Model"] = np.array(Model)

        # from pympler import muppy, summary
        # def print_object_summary():
        #     all_objects = muppy.get_objects()
        #     sum_obj = summary.summarize(all_objects)
        #     summary.print_(sum_obj[:10])  # Top 10 object types
        # print_object_summary()
        
        del(self.CEv)
        return {"Success":True,"iIsland":iIsland,"HasError":False}
    

class ClassEvolveGA_SingleIsland():
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
                                                                         NCPU=NCPU,
                                                                         ScaleS0="linear")
        self.ArrayMethodsMachine.InitWorkers()

        

    def setDEAP(self):
        T=ClassTimeIt.ClassTimeIt("   SET DEAP")
        T.disable()
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
        T.timeit("set DEAP")

    def main(self,NGen=1000,NIndiv=100,DoPlot=True):
        T=ClassTimeIt.ClassTimeIt("   GA: Main")
        T.disable()
        self.setDEAP()
        toolbox=self.toolbox
        self.pop = toolbox.population(n=NIndiv)
        self.hof = tools.HallOfFame(1, similar=numpy.array_equal)
        for indiv in self.pop:
            indiv.fill(0)
        T.timeit("Init")
        # #########################################################
        self.DicoDicoInitIndiv  = shared_dict.attach("DicoDicoInitIndiv")
        self.DicoDicoInitIndiv.reload()
        NTypeInitAllIslands=len(self.DicoDicoInitIndiv.keys())
        LModel=[]
        NTypeInit=0
        for iTypeInit in range(NTypeInitAllIslands):
            DModels=self.DicoDicoInitIndiv.get(iTypeInit)
            Model=DModels.get(self.iIsland,None)
            #print(Model)
            if Model is not None: NTypeInit+=1
        # eirther has no model init (islands too small) or has all of them (no crash)
        if NTypeInit!=0 and NTypeInit!=NTypeInitAllIslands: stop

        
        def GiveListPolyArrayMP(N,iTypeInit=None):
            return [GivePolyArrayMP(iTypeInit=iTypeInit) for iIndiv in range(N)]
                
        def GivePolyArrayMP(iTypeInit=None):
            T=CTI("GivePolyArrayMP")
            T.disable()

            
            
            if iTypeInit is None:
                iTypeInit=int(np.random.rand(1)[0]*NTypeInit)
                
            DicoModelMP=None
            DicoInitIndiv=self.DicoDicoInitIndiv.get(iTypeInit,None)
            if DicoInitIndiv is not None:
                DicoModelMP=DicoInitIndiv.get(self.iIsland,None)
            T.timeit("Init")

            if DicoModelMP is not None:
                PolyModelArrayMP=DicoModelMP
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
            T.timeit("GiveListPolyArrayMP_LinComb: L")
            for iTypeInit in range(NTypeInit):
                L[iTypeInit]=GivePolyArrayMP(iTypeInit=iTypeInit)
            T.timeit("GiveListPolyArrayMP_LinComb: for")
            return L
        
        def GivePolyArrayMP_LinComb():
            T=CTI("GivePolyArrayMP_LinComb")
            T.disable()
            LInit=[]
            Nrand=np.max([1,NTypeInit])
            w=np.random.rand(Nrand)
            w/=np.sum(w)
            T.timeit("Init")
            #print(w)
            PolyModelArrayMP=w[0]*GivePolyArrayMP(iTypeInit=0)
            T.timeit("Init1")
            # print("Nrand=",Nrand)
            for iTypeInit in range(1,Nrand):
                PolyModelArrayMP+=w[iTypeInit]*GivePolyArrayMP(iTypeInit=iTypeInit)
            T.timeit("Init2")
            # stop
            return PolyModelArrayMP
        # #########################################################

            
        
        if self.IslandBestIndiv is not None:
            if NGen==0:
                self.ArrayMethodsMachine.PM.ReinitPop(self.pop,GiveListPolyArrayMP_LinComb(len(self.pop)),PutNoise=False)
                self.ArrayMethodsMachine.KillWorkers()
                return self.pop[0]
            T.timeit("N=0")


            PutNoise=True#False
            if np.max(np.abs(self.IslandBestIndiv))==0:
                #print("NEW")
                ListPolyModelArrayMP=GiveListPolyArrayMP_LinComb(len(self.pop))
                T.timeit("New0")
                self.ArrayMethodsMachine.PM.ReinitPop(self.pop,ListPolyModelArrayMP,PutNoise=PutNoise)
                T.timeit("New")
            else:
                #print("MIX")
                # pop0=self.pop[0:NIndiv]
                # pop1=self.pop[NIndiv::]

                # pop1=self.pop
                # pop0=[]

                # pop1=self.pop[0:1]
                # pop0=self.pop[1::]
                
                
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

                NoiseType="Mutate"
                if NoiseType=="Normal":
                    NIndiv=len(self.pop)//10
                    pop1=self.pop[0:NIndiv//2]
                    pop0=self.pop[NIndiv//2::]
                    # #########################
                    # # just noise
                    self.ArrayMethodsMachine.PM.ReinitPop(pop1,[PolyModelArray]*len(pop1),GSigModel=GSigModel,PutNoise=PutNoise)
                    T.timeit("Mix: ReinitPop pop1")
                    pop1[0].flat[:]=BestIndiv.flat[:]
                    # From Minor Cycle estimate
                    # half of the pop with the MP model
                    self.ArrayMethodsMachine.PM.ReinitPop(pop0,GiveListPolyArrayMP_LinComb( len(pop0) ),PutNoise=PutNoise)
                    T.timeit("Mix: ReinitPop pop0")
                elif NoiseType=="Mutate":
                    NIndiv=len(self.pop)//2
                    pop0=self.pop[0:NIndiv//2]
                    pop1=self.pop[NIndiv//2::]
                    self.ArrayMethodsMachine._fill_pop_array(self.pop)
                    # #########################
                    # Mutation
                    self.ArrayMethodsMachine.PM.ReinitPop(pop0,[PolyModelArray]*len(pop0),GSigModel=GSigModel,PutNoise=False)
                    self.ArrayMethodsMachine.mutatePop(pop0,1.,self.MutConfig)
                    pop0[0].flat[:]=BestIndiv.flat[:]
                    # #########################
                    self.ArrayMethodsMachine.PM.ReinitPop(pop1,GiveListPolyArrayMP_LinComb( len(pop1) ),PutNoise=False)
                    T.timeit("Mix: ReinitPop pop1")
                    self.ArrayMethodsMachine.mutatePop(pop1,1.,self.MutConfig)
                    pop1[0].flat[:]=BestIndiv.flat[:]
                
                    


                self.pop=pop1+pop0

        T.timeit("Init pop")

        _=self.ArrayMethodsMachine.GiveFitnessPop(self.pop)
        T.timeit("Init Givefitness")

        self.pop, log= algorithms.eaSimple(self.pop, toolbox, cxpb=0.3, mutpb=0.5, ngen=NGen, 
                                           halloffame=self.hof, 
                                           #stats=stats,
                                           verbose=False, 
                                           ArrayMethodsMachine=self.ArrayMethodsMachine,
                                           DoPlot=DoPlot,
                                           MutConfig=self.MutConfig)
        T.timeit("eaSimple")
        #print(self.pop[0])
        #stop
        self.ArrayMethodsMachine.KillWorkers()

        V = tools.selBest(self.pop, 1)[0]
        

        return V
