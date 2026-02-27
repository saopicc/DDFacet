
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
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("ClassEvolveGA")
import pylab
from scipy.optimize import minimize

DOPLOT=0
DISABLE_TIMEIT=True

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


class ClassEvolveGA():
    def __init__(self,ImageDeconvMachine,ParallelMode="OverIslands"):
        self.__dict__ = ImageDeconvMachine.__dict__
        self.ImageDeconvMachine=ImageDeconvMachine
        self.ParallelMode=ParallelMode
        
    def setDicoInitModel(self,DicoInitModel):
        self.DicoInitIndiv=DicoInitModel

    
        
    def _runGA(self,#ListIslands,
               iIsland,
               #IslandBestIndiv,
               DicoDirty_path,DicoPSF_path,GridFreqs,DegridFreqs):


        #from pympler import tracker
        #tr=tracker.SummaryTracker()

        # if iIsland<2700:
        #     return {"Success":False,"iIsland":iIsland,"HasError":False}
        
        IslandBestIndiv=self.ModelMachine.GiveIndividual(self.ListAllIslands[iIsland])
        #print("FLFJDLFJ",iIsland,np.array(IslandBestIndiv).size)
        #del(self.ModelMachine)
        
        #tr.print_diff()
        
        #NIslands=len(ListIslands)
        #if NIslands==0: return
        T=ClassTimeIt.ClassTimeIt("[%i]  _runGA"%iIsland)
        if DISABLE_TIMEIT: T.disable()        
        #T.disable()
        T.timeit("start")
        T.timeit("start")
        T.timeit("start")
        self.ImageDeconvMachine._updateWorkerInternals(DicoDirty_path,DicoPSF_path,GridFreqs,DegridFreqs)
        T.timeit("updateWorkerInternals")
        ##tr.print_diff()
        
        ListInitIslands=None
        allIslandModelDict  = shared_dict.attach("DeconvListIslands%s"%self.StrField)
        ThisIslandModelDict=allIslandModelDict[iIsland]
        ThisPixList=ThisIslandModelDict["IslandXY"]
        
        

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


        if "BestIndiv" not in list(ThisIslandModelDict.keys()):
            ThisIslandModelDict.addSharedArray("BestIndiv", IslandBestIndiv.shape, IslandBestIndiv.dtype)

        ThisIslandModelDict["BestIndiv"][:] = IslandBestIndiv[:]
        
        # ListOrder=[iIsland,FacetID,JonesNorm.flat[0],self.RMS**2,island_dict.path,iIslandInit]
        # ##############################################
        # self.MultiFreqMode=MultiFreqMode
        self.FreqsInfo=self.PSFServer.DicoMappingDesc
        # self._Dirty = Dirty
        self.CubeVariablePSF = self.DicoVariablePSF["CubeVariablePSF"]
        
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

            
        T.timeit("Increase")
        
        ParmDict = shared_dict.attach("ParmDict%s"%self.StrField) # ParmDict
        #PixVariance=ParmDict["RMS"]**2
        PixVariance=np.array(self.DicoDirty["LRMS"])**2        

        ##tr.print_diff()



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
                                            ParallelMode=self.ParallelMode,
                                            DicoInitIndiv=self.DicoInitIndiv
                                            )
        T.timeit("Declare class")

        # ###################################
        # NParms=self.CEv.ArrayMethodsMachine.PM.NParam
        # DicoInitModel=self.DicoInitIndiv
        # if len(DicoInitModel)==0:
        #     S,Alpha=self.CEv.ArrayMethodsMachine.DeconvCLEAN()
        #     if NParms==1:
        #         AModel=np.array([S]).reshape((1,1,S.size))
        #     else:
        #         AModel=np.zeros((NParms,S.size),np.float32)
        #         AModel[0,:]=S
        #         AModel[1,:]=Alpha
        #         AModel=AModel.reshape((1,NParms,S.size))
        # else:
        #     AModel=np.array([DicoInitModel[iMachine] for iMachine in DicoInitModel.keys()])
        # NModel=AModel.shape[0]
        # NFreqBands=self.CEv.ArrayMethodsMachine.NFreqBands

        # LConvModel=[]
        # Lx0=[]
        # for iModel,Model in enumerate(AModel):
        #     V=Model.copy().ravel()
        #     Model1=Model.copy()
        #     ind=np.where(Model1[0,:]!=0)[0]
        #     Model1[0,ind]=1
        #     V1=Model1.ravel()
        #     ConvModelArray=self.CEv.ArrayMethodsMachine.ToConvArray(V,OutMode="Parms")
        #     ConvModelArray1=self.CEv.ArrayMethodsMachine.ToConvArray(V1,OutMode="Parms")
        #     LConvModel.append([ConvModelArray1,ConvModelArray])
        #     Lx0+=[0.,.5]

        # Lx0=np.array(Lx0)
        # SpacialWeight=ThisIslandModelDict["SpacialWeight"]
        # SpacialWeight=SpacialWeight.reshape((1,1,-1))
        # ARMS=np.array(self.DicoDirty["LRMS"]).reshape((-1,1,1))

        # def combineModels(x):
        #     x=x.reshape((NModel,2))
        #     IM=np.zeros(LConvModel[0][0].shape,np.float64)
        #     for iModel in range(NModel):
        #         b,a=x[iModel]
        #         B,A=LConvModel[iModel]
        #         IM+=b*np.float64(B)+a*np.float64(A)
        #         #IM+=a*A
        #     return IM
        
        # def giveChi2(x):
        #     x=x.reshape((NModel,2))
        #     IM=combineModels(x)
        #     R=np.float64(self.CEv.ArrayMethodsMachine.DirtyArrayParms)-IM
        #     R=R/ARMS
        #     R=R*SpacialWeight
        #     Chi2=(np.sqrt(np.sum(R**2)))
        #     return Chi2
            
        # # Dirty2D=self.CEv.ArrayMethodsMachine.PM.ModelToSquareArray(self.CEv.ArrayMethodsMachine.DirtyArray,TypeInOut=("Data","Data"))

        # res = minimize(giveChi2, Lx0)#, constraints=cons)
        # x = res.x
        
        # Vm=np.zeros_like(AModel[0])
        # x=x.reshape((NModel,2))
        # for iModel in range(NModel):
        #     b,a=x[iModel]
        #     V=AModel[iModel].copy()
        #     V=V.reshape((NParms,V.size//NParms)) 
        #     V[0,:]=(b+a*V[0,:])
        #     Vm+=V

        # if "Model" not in  list(ThisIslandModelDict.keys()):
        #     ThisIslandModelDict.addSharedArray("Model", Model.shape, np.float32)
        # ThisIslandModelDict["Model"][:] = np.array(Vm)[:]
        # V=Vm.ravel()
        # ConvModelArray=self.CEv.ArrayMethodsMachine.ToConvArray(V,OutMode="Parms")
        # Resid1D=self.CEv.ArrayMethodsMachine.DirtyArrayParms-ConvModelArray
        # ThisIslandModelDict.addSharedArray("Resid", Resid1D.shape, Resid1D.dtype)
        # ThisIslandModelDict["Resid"] = np.array(Resid1D)[:]

        # return {"Success":True,"iIsland":iIsland,"HasError":False}
    
        # # def plotModel(V,Name=""):
        # #     ConvModelArray=self.CEv.ArrayMethodsMachine.ToConvArray(V,OutMode="Data")
        # #     Resid1D=self.CEv.ArrayMethodsMachine.DirtyArray-ConvModelArray
        # #     Resid2D=self.CEv.ArrayMethodsMachine.PM.ModelToSquareArray(Resid1D,TypeInOut=("Data","Data"))
        # #     Dirty2D=self.CEv.ArrayMethodsMachine.PM.ModelToSquareArray(self.CEv.ArrayMethodsMachine.DirtyArray,TypeInOut=("Data","Data"))
        # #     Model2D=self.CEv.ArrayMethodsMachine.PM.ModelToSquareArray(V,TypeInOut=("Parms","Parms"))
        # #     iPlot=1
        # #     pylab.figure("Model %s"%Name)
        # #     for ich in range(NFreqBands):
        # #         ax=pylab.subplot(3,NFreqBands,iPlot); iPlot+=1
        # #         ax.imshow(Dirty2D[ich,0])
        # #     for ich in range(NFreqBands):
        # #         ax=pylab.subplot(3,NFreqBands,iPlot); iPlot+=1
        # #         ax.imshow(Model2D[ich,0])
        # #     for ich in range(NFreqBands):
        # #         v0,v1=Dirty2D[ich,0].min(),Dirty2D[ich,0].max()
        # #         ax=pylab.subplot(3,NFreqBands,iPlot); iPlot+=1
        # #         #ax.imshow(Resid2D[ich,0],vmin=v0,vmax=v1)
        # #         ax.imshow(Resid2D[ich,0])
        # #         ax.set_title("%f %f"%(Resid2D[ich,0].min(),Resid2D[ich,0].max()))
        # # V0=combineModels(Lx0)
        # # plotModel(V0,"Init")
        # # for iModel,Model in enumerate(AModel):
        # #     V=Model.ravel()
        # #     plotModel(V,"Model %i"%iModel)
        # # plotModel(Vm,"Fit")
        # # pylab.show()
        # # stop
        # ##################################


        
        Model=self.CEv.main(NGen=NGen,NIndiv=NIndiv,DoPlot=False)
        T.timeit("HasRun")

        
        #return {"Success":False,"iIsland":iIsland,"HasError":False}
        
        #tr.print_diff()
        if "Model" not in  list(ThisIslandModelDict.keys()):
            ThisIslandModelDict.addSharedArray("Model", Model.shape, np.float32)
        ThisIslandModelDict["Model"][:] = np.array(Model)[:]
        
        V=Model.ravel()
        ConvModelArray=self.CEv.ArrayMethodsMachine.ToConvArray(V,OutMode="Parms")
        Resid1D=self.CEv.ArrayMethodsMachine.DirtyArrayParms-ConvModelArray
        ThisIslandModelDict.addSharedArray("Resid", Resid1D.shape, Resid1D.dtype)
        ThisIslandModelDict["Resid"] = np.array(Resid1D)[:]
        
        
        # from sys import getsizeof
        # print("DLKSDLSDFLSDF [%i]"%iIsland,np.array(Model).shape,getsizeof(np.array(Model)),Model.max())
        # print("DLKSDLSDFLSDF [%i]"%iIsland,np.array(Model).shape,getsizeof(np.array(Model)),Model.max())
        # print("DLKSDLSDFLSDF [%i]"%iIsland,np.array(Model).shape,getsizeof(np.array(Model)),Model.max())
        # print("DLKSDLSDFLSDF [%i]"%iIsland,np.array(Model).shape,getsizeof(np.array(Model)),Model.max())
        
        
        # del(self.CEv)
        # import gc
        # gc.collect()
        # from pympler import muppy, summary
        # print("==================================")
        # print("========= %i"%iIsland)
        # def print_object_summary():
        #     all_objects = muppy.get_objects()
        #     sum_obj = summary.summarize(all_objects)
        #     summary.print_(sum_obj[:10])  # Top 10 object types
        # print_object_summary()
        # print("==================================")




        
        return {"Success":True,"iIsland":iIsland,"HasError":False}
    

class ClassEvolveGA_SingleIsland():
    def __init__(self,Dirty,PSF,FreqsInfo,ListPixData=None,ListPixParms=None,IslandBestIndiv=None,GD=None,
                 WeightFreqBands=None,PixVariance=1e-2,iFacet=0,iIsland=None,island_dict=None,
                 ParallelMode=None,
                 DicoInitIndiv=None):
        self.DicoInitIndiv=DicoInitIndiv
        self.ParallelMode=ParallelMode
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

        self.GD=GD
            
        self.ArrayMethodsMachine=ClassArrayMethodSSD.ClassArrayMethodSSD(Dirty,PSF,ListPixParms,ListPixData,FreqsInfo,
                                                                         PixVariance=PixVariance,
                                                                         iFacet=iFacet,
                                                                         IslandBestIndiv=IslandBestIndiv,
                                                                         GD=GD,
                                                                         WeightFreqBands=WeightFreqBands,
                                                                         iIsland=iIsland,
                                                                         island_dict=island_dict,
                                                                         ParallelMode=self.ParallelMode,
                                                                         NCPU=NCPU)

        

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
        T=ClassTimeIt.ClassTimeIt("   GA: Main [%i]"%self.iIsland)
        if DISABLE_TIMEIT: T.disable()        
        #T.disable()
        self.SModelArrayMP_CLEAN=None
        self.ArrayMethodsMachine.startWorkers()
        self.setDEAP()
        T.timeit("self.setDEAP")
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
            T=CTI("[%i] GivePolyArrayMP"%self.iIsland)
            if DISABLE_TIMEIT: T.disable()        
            #T.disable()
            NTypeInit=len(self.DicoInitIndiv.keys())
            
            if iTypeInit is None:
                iTypeInit=int(np.random.rand(1)[0]*NTypeInit)
            DicoModelMP=self.DicoInitIndiv.get(iTypeInit,None)
            T.timeit("Init")
            
            if DicoModelMP is not None:
                PolyModelArrayMP=DicoModelMP
                T.timeit("get")
            else:
                if self.SModelArrayMP_CLEAN is None:
                    self.SModelArrayMP_CLEAN,_=self.ArrayMethodsMachine.DeconvCLEAN()
                PolyModelArrayMP=np.zeros((self.ArrayMethodsMachine.PM.NOrderPoly,self.ArrayMethodsMachine.PM.NPixListParms),np.float32)
                PolyModelArrayMP[0,:]=self.SModelArrayMP_CLEAN
                T.timeit("CLEAN")
            return PolyModelArrayMP

        def GiveListPolyArrayMP_LinComb(N):
            T.reinit()
            
            L=[GivePolyArrayMP_LinComb() for iIndiv in range(N)]
            DoPutNoise=np.ones((N,),bool)
            # T.timeit("GiveListPolyArrayMP_LinComb: L")
            NTypeInit=len(self.DicoInitIndiv.keys())
            for iTypeInit in range(NTypeInit):
                L[iTypeInit]=GivePolyArrayMP(iTypeInit=iTypeInit)
                DoPutNoise[iTypeInit]=False
                
            # # print("VLKFSDLKSFDL")
            # # iDone=0
            # L=[]
            # DoPutNoise=np.ones((N,),bool)
            # NTypeInit=len(self.DicoInitIndiv.keys())
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
            NTypeInit=len(self.DicoInitIndiv.keys())
            L=[]
            if NTypeInit==0:
                # run simplistic clean
                L.append(GivePolyArrayMP())
            else:
                for iTypeInit in range(NTypeInit):
                    L.append(GivePolyArrayMP(iTypeInit=iTypeInit))

            if np.max(np.abs(self.IslandBestIndiv))!=0. and self.GD["GAClean"]["NSourceKin"]>0:
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

            T=CTI("[%i] GivePolyArrayMP_LinComb"%self.iIsland)
            if DISABLE_TIMEIT: T.disable()        
            #T.disable()
            NTypeInit=len(self.DicoInitIndiv.keys())
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

            
        #self.DicoInitIndiv  = shared_dict.attach("DicoInitIndiv")
        #self.DicoInitIndiv.reload()
        
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

                
                self.ArrayMethodsMachine.stopWorkers()
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

                # NTypeInit=len(self.DicoInitIndiv.keys())
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
        
        self.ArrayMethodsMachine.stopWorkers()
        

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


