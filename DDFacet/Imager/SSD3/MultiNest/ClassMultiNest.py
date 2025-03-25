
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
from DDFacet.Imager.SSD3.MultiNest.svgd import SVGD

def FilterIslandsPix(ListIn,Npix_x,Npix_y):
    ListOut=[]
    for x,y in ListIn:
        Cx=((x>=0)&(x<Npix_x))
        Cy=((y>=0)&(y<Npix_y))
        if (Cx&Cy):
            ListOut.append([x,y])
    return ListOut

SERIAL=True
#SERIAL=False

def test():
    iIsland=0
    S=np.load("SingleIsland_input_%i.npz"%iIsland,allow_pickle=True)
    Dirty=S["Dirty"]
    PSF=S["PSF"]
    FreqsInfo=S["FreqsInfo"][()]
    ListPixParms=S["ListPixParms"]
    ListPixData=S["ListPixData"]
    iFacet=S["iFacet"]
    PixVariance=S["PixVariance"]
    IslandBestIndiv=S["IslandBestIndiv"]
    GD=S["GD"][()]
    ModelMachine=S["ModelMachine"][()]
    iIsland=S["iIsland"]
    island_dict=S["island_dict"]
    
    CEv=ClassEvolveGA_SingleIsland(Dirty,
                                   PSF,
                                   FreqsInfo,
                                   ListPixParms=ListPixParms,
                                   ListPixData=ListPixData,
                                   iFacet=iFacet,
                                   PixVariance=PixVariance,
                                   IslandBestIndiv=IslandBestIndiv,#*np.sqrt(JonesNorm),
                                   GD=GD,
                                   iIsland=iIsland,
                                   island_dict=island_dict,
                                   ParallelFitness=False,
                                   ModelMachine=ModelMachine)
    
    Model=CEv.doStein(NIter=1000)


class ClassEvolveGA():
    def __init__(self,ImageDeconvMachine):
        self.__dict__ = ImageDeconvMachine.__dict__
        self.ImageDeconvMachine=ImageDeconvMachine
        
        self.APP=DDFacet.Other.AsyncProcessPool.initNew(Name="APP_GA",
                                                         ncpu=self.GD["Parallel"]["NCPU"],
                                                         affinity=self.GD["Parallel"]["Affinity"],
                                                         parent_affinity=self.GD["Parallel"]["MainProcessAffinity"],
                                                         verbose=self.GD["Debug"]["APPVerbose"],
                                                         pause_on_start=self.GD["Debug"]["PauseWorkers"])
        self.APP.registerJobHandlers(self)
        self.APP.startWorkers()
        
    def runGA_AllIslands(self):
        APP=self.APP
        
        T=ClassTimeIt.ClassTimeIt("runGA_AllIslands")
        T.disable()
        for iIsland,Island in enumerate(self.ListIslands):
            IslandBestIndiv=self.ModelMachine.GiveIndividual(self.ListIslands[iIsland])
            self.APP.runJob("runGA.%i"%(iIsland),
                             self._runGA,
                             args=(self.ListIslands,iIsland,IslandBestIndiv,self.DicoDirty.path,self.GridFreqs,self.DegridFreqs), serial=SERIAL)
        LDicoResults=self.APP.awaitJobResults("runGA.*", progress="Genetic Alg.")
        T.timeit("runGA")

        allIslandModelDict  = shared_dict.attach("DeconvListIslands%s"%self.StrField)
        allIslandModelDict.reload()
        for iRes,DicoResult in enumerate(LDicoResults):
            iIsland=DicoResult["iIsland"]
            ThisIslandModelDict = allIslandModelDict[iIsland]
            ThisIslandModelDict.reload()
            self.ModelMachine.AppendIsland(self.ListIslands[iIsland], ThisIslandModelDict["Model"].copy())
            if DicoResult["HasError"]:
                self.ErrorModelMachine.AppendIsland(ListIslands[iIsland], ThisIslandModelDict["sModel"].copy())
        self.APP.shutdown()
        del(self.APP)
        
    def _runGA(self,ListIslands,iIsland,IslandBestIndiv,DicoDirty_path,GridFreqs,DegridFreqs):
        NIslands=len(ListIslands)
        if NIslands==0: return
        T=ClassTimeIt.ClassTimeIt("  ----  _runGA #%i"%iIsland)
        T.disable()
        self.ImageDeconvMachine._updateWorkerInternals(DicoDirty_path,GridFreqs,DegridFreqs)
        T.timeit("updateWorkerInternals")
        
        ListInitIslands=None
        ThisPixList=ListIslands[iIsland]
        allIslandModelDict  = shared_dict.attach("DeconvListIslands%s"%self.StrField)
        ThisIslandModelDict = allIslandModelDict.addSubdict(iIsland)
        ThisIslandModelDict["Island"] = np.array(ThisPixList)

        XY=np.array(ThisPixList,dtype=np.float32)
        xm,ym=np.mean(np.float32(XY),axis=0).astype(int)
        T.timeit("xm,ym")
        nchan,npol,_,_=self._Dirty.shape
        JonesNorm=(self.DicoDirty["JonesNorm"][:,:,xm,ym]).reshape((nchan,npol,1,1))
        W=self.DicoDirty["WeightChansImages"]
        JonesNorm=np.sum(JonesNorm*W.reshape((nchan,1,1,1)),axis=0).reshape((1,npol,1,1))
        T.timeit("JonesNorm")
        
        
        FacetID=self.PSFServer.giveFacetID2(xm,ym)
        T.timeit("FacetID")

        ThisIslandModelDict["BestIndiv"] = IslandBestIndiv
        
        # ListOrder=[iIsland,FacetID,JonesNorm.flat[0],self.RMS**2,island_dict.path,iIslandInit]
        # ##############################################
        # self.MultiFreqMode=MultiFreqMode
        self.FreqsInfo=self.PSFServer.DicoMappingDesc
        # self._Dirty = Dirty
        self.CubeVariablePSF = self.DicoVariablePSF["CubeVariablePSF"]
        
        ThisPixList = ThisIslandModelDict["Island"].tolist()
        IslandBestIndiv = ThisIslandModelDict["BestIndiv"]

        PSF=self.CubeVariablePSF[FacetID]
        NGen=self.GD["GAClean"]["NMaxGen"]
        NIndiv=self.GD["GAClean"]["NSourceKin"]

        ListPixParms=ThisPixList
        ListPixData=ThisPixList
        dx=self.GD["SSDClean"]["NEnlargeData"]
        if dx>0:
            IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
            ListPixData=IncreaseIslandMachine.IncreaseIsland(ListPixData,dx=dx)


        ParmDict = shared_dict.attach("ParmDict%s"%self.StrField) # ParmDict
        PixVariance=ParmDict["RMS"]**2

        np_island_dict={}
        for k in ThisIslandModelDict.keys():
            np_island_dict[k]=ThisIslandModelDict[k].copy()

        def giveCopy(D):
            d={}
            import copy
            for k in D.keys():
                if "SharedDict" in str(type(D[k])):
                    for kk in D[k].keys():
                        d[k]=giveCopy(D[k])
                elif "array" in str(type(D[k])):
                    d[k]=D[k].copy()
                else:
                    d[k]=copy.deepcopy(D[k])
        
        np_FreqsInfo=giveCopy(self.FreqsInfo)
        
        np.savez("SingleIsland_input_%i.npz"%iIsland,
                 Dirty=self._Dirty.copy(),
                 PSF=PSF.copy(),
                 FreqsInfo=np_FreqsInfo,
                 ListPixParms=ListPixParms,
                 ListPixData=ListPixData,
                 iFacet=FacetID,
                 PixVariance=PixVariance,
                 IslandBestIndiv=IslandBestIndiv,
                 GD=self.GD,
                 iIsland=iIsland,
                 island_dict=np_island_dict)
        stop
        
        CEv=ClassEvolveGA_SingleIsland(self._Dirty,
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
        Model=CEv.doStein(NGen=NGen,NIndiv=NIndiv,DoPlot=False)
        
        ThisIslandModelDict["Model"] = np.array(Model)
        
        del(CEv)
        return {"Success":True,"iIsland":iIsland,"HasError":False}
    

class ClassEvolveGA_SingleIsland():
    def __init__(self,Dirty,PSF,FreqsInfo,ListPixData=None,ListPixParms=None,IslandBestIndiv=None,GD=None,
                 WeightFreqBands=None,PixVariance=1e-2,iFacet=0,iIsland=None,island_dict=None,
                 ParallelFitness=False,ModelMachine=None):

                 
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
        self.ModelMachine=ModelMachine
        self.IslandBestIndiv=self.ModelMachine.GiveIndividual(ListPixParms)
                    

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

        self.PM=self.ArrayMethodsMachine.PM
        self.ConvMachine=self.ArrayMethodsMachine.ConvMachine
        
    def doStein(self,NIter=1000):
        
        class MVN:
            def __init__(self, mu, A):
                self.mu = mu
                self.A = A
            def dlnprob(self, theta):
                r=-1*np.matmul(theta-nm.repmat(self.mu, theta.shape[0], 1), self.A)
                return r

        NPoints=100

        class MODEL:
            def __init__(self,ArrayMethodsMachine):
                self.ArrayMethodsMachine=ArrayMethodsMachine
                pass
                
            def dlnprob(self, Lindividual):
                # individual=
                # toConvArray(V)
                # A=self.PM.GiveModelArray(V)
                # A=self.ConvMachine.Convolve(A,OutMode=OutMode)
                L=[]
                for individual in Lindividual:
                    ym=self.ArrayMethodsMachine.ToConvArray(individual)
                    y=self.ArrayMethodsMachine.DirtyArray
                    yr=y-ym
                    yr/=self.ArrayMethodsMachine.PixVariance
                    Cyr=self.ArrayMethodsMachine.ConvMachine.Convolve(yr,InMode="Data",OutMode="Parms")
                    L.append(Cyr.reshape((1,-1)))
                r=np.array(L)
                r=r.reshape((NPoints,self.ArrayMethodsMachine.NParms))
                return r

        ndims=self.IslandBestIndiv.size
        x0=self.IslandBestIndiv.reshape((1,ndims))
        x0=x0+np.random.randn(NPoints,ndims)*np.sqrt(self.ArrayMethodsMachine.PixVariance)

        
        M=MODEL(self.ArrayMethodsMachine)
        SVGD().update(x0, M.dlnprob, n_iter=10000, stepsize=0.1)
        stop
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
