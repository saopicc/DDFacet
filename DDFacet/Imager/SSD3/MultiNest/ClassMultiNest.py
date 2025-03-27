
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
import numpy as np
import random
import psutil
from DDFacet.Imager.SSD3 import ClassArrayMethodSSD
#from DDFacet.Imager.SSD3 import ClassImageArrayMethodSSD
from DDFacet.Array import shared_dict
#from DDFacet.Imager.SSD3 import ClassImageDeconvMachineSSD
from SkyModel.PSourceExtract import ClassIncreaseIsland
from DDFacet.Imager.SSD3.MultiNest.svgd import SVGD
from DDFacet.Imager.SSD3.ClassParamMachine import ClassParamMachine
from DDFacet.Other import ModColor

from DDFacet.Other import logger
log=logger.getLogger("ClassEvolveStein")

def FilterIslandsPix(ListIn,Npix_x,Npix_y):
    ListOut=[]
    for x,y in ListIn:
        Cx=((x>=0)&(x<Npix_x))
        Cy=((y>=0)&(y<Npix_y))
        if (Cx&Cy):
            ListOut.append([x,y])
    return ListOut

global SERIAL
# ################
SERIAL=True
DOPLOT=True
# ################
SERIAL=False
DOPLOT=False
# ################

def debug(FName="SingleIsland_input_0.npz"):
    global SERIAL
    SERIAL=False
    iIsland=1
    S=np.load(FName,allow_pickle=True)
    Dirty=S["Dirty"]
    PSF=S["PSF"]
    FreqsInfo=S["FreqsInfo"][()]
    ListPixParms=S["ListPixParms"]
    ListPixData=S["ListPixData"]
    iFacet=S["iFacet"]
    PixVariance=S["PixVariance"]
    IslandBestIndiv=S["IslandBestIndiv"]
    GD=S["GD"][()]
    #ModelMachine=S["ModelMachine"][()]
    iIsland=S["iIsland"]
    island_dict=S["island_dict"]

    # I=ModelMachine.GiveModelImage()
    # import pylab
    # pylab.clf()
    # pylab.imshow(I[0,0],interpolation="nearest")
    # pylab.show()
    # return
    CEv=ClassEvolveStein_SingleIsland(Dirty,
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
                                      DoPlot=1,
                                      #ModelMachine=ModelMachine
    )
    
    Model=CEv.doStein()


class ClassEvolveStein():
    def __init__(self,ImageDeconvMachine):
        self.__dict__ = ImageDeconvMachine.__dict__
        self.ImageDeconvMachine=ImageDeconvMachine
        import DDFacet.Other.AsyncProcessPool
        self.APP_Stein=DDFacet.Other.AsyncProcessPool.initNew(Name="APP_Stein",
                                                         ncpu=self.GD["Parallel"]["NCPU"],
                                                         affinity=self.GD["Parallel"]["Affinity"],
                                                         parent_affinity=self.GD["Parallel"]["MainProcessAffinity"],
                                                         verbose=self.GD["Debug"]["APPVerbose"],
                                                         pause_on_start=self.GD["Debug"]["PauseWorkers"])
        self.APP_Stein.registerJobHandlers(self)
        self.APP_Stein.startWorkers()
        
    def runStein_AllIslands(self):
        APP=self.APP_Stein
        
        T=ClassTimeIt.ClassTimeIt("runStein_AllIslands")
        T.disable()
        for iIsland,Island in enumerate(self.ListIslands):
            IslandBestIndiv=self.ModelMachine.GiveIndividual(self.ListIslands[iIsland]).copy()
            APP.runJob("runStein.%i"%(iIsland),
                             self._runStein,
                             args=(self.ListIslands,iIsland,IslandBestIndiv,self.DicoDirty.path,self.GridFreqs,self.DegridFreqs), serial=SERIAL)
        LDicoResults=APP.awaitJobResults("runStein.*", progress="Stein VGD")
        T.timeit("run")

        allIslandModelDict  = shared_dict.attach("DeconvListIslands%s"%self.StrField)
        allIslandModelDict.reload()
        for iRes,DicoResult in enumerate(LDicoResults):
            if not DicoResult["Success"]:
                log.print(ModColor.Str("Island #%i, error: %s"%(iIsland,DicoResult["ErrorMessage"])))
                log.print(ModColor.Str("Island #%i, error: %s"%(iIsland,DicoResult["ErrorMessage"])))
                continue
                #self.ErrorModelMachine.AppendIsland(ListIslands[iIsland], ThisIslandModelDict["sModel"].copy())
            iIsland=DicoResult["iIsland"]
            ThisIslandModelDict = allIslandModelDict[iIsland]
            ThisIslandModelDict.reload()
            Model,StdModel=ThisIslandModelDict["SteinMedianModel"],ThisIslandModelDict["SteinStdModel"]
            self.SteinModelMachine.AppendIsland(self.ListIslands[iIsland], Model.copy())

        APP.terminate()
        APP.shutdown()
        del(self.APP_Stein)
        
    def _runStein(self,ListIslands,iIsland,IslandBestIndiv,DicoDirty_path,GridFreqs,DegridFreqs):
        NIslands=len(ListIslands)
        if NIslands==0: return
        T=ClassTimeIt.ClassTimeIt("  ----  _runStep #%i"%iIsland)
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

        #ThisIslandModelDict["BestIndiv"] = IslandBestIndiv
        
        # ListOrder=[iIsland,FacetID,JonesNorm.flat[0],self.RMS**2,island_dict.path,iIslandInit]
        # ##############################################
        # self.MultiFreqMode=MultiFreqMode
        self.FreqsInfo=self.PSFServer.DicoMappingDesc
        # self._Dirty = Dirty
        self.CubeVariablePSF = self.DicoVariablePSF["CubeVariablePSF"]
        
        ThisPixList = ThisIslandModelDict["Island"].tolist()
        #IslandBestIndiv = ThisIslandModelDict["BestIndiv"]

        PSF=self.CubeVariablePSF[FacetID]

        ListPixParms=ThisPixList
        ListPixData=ThisPixList
        dx=self.GD["SSDClean"]["NEnlargeData"]
        if dx>0:
            IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
            ListPixData=IncreaseIslandMachine.IncreaseIsland(ListPixData,dx=dx)


        ParmDict = shared_dict.attach("ParmDict%s"%self.StrField) # ParmDict
        PixVariance=ParmDict["RMS"]**2

        
        CEv=ClassEvolveStein_SingleIsland(self._Dirty,
                                          PSF,
                                          self.FreqsInfo,
                                          ListPixParms=ListPixParms,
                                          ListPixData=ListPixData,
                                          iFacet=FacetID,PixVariance=PixVariance,
                                          IslandBestIndiv=IslandBestIndiv,#*np.sqrt(JonesNorm),
                                          iIsland=iIsland,
                                          GD=self.GD,
                                          island_dict=ThisIslandModelDict,
                                          ParallelFitness=False)
        #Model,StdModel=CEv.doStein()

        if SERIAL:
            Model,StdModel=CEv.doStein()
        else:
        
            try:
                Model,StdModel=CEv.doStein()
                # if iIsland==394: stop
                ###########################################
            except Exception as e:
#            if True:
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
                    return d
                np_FreqsInfo=giveCopy(self.FreqsInfo)
                np.savez("SingleIsland_exception_input_%i.npz"%iIsland,
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
                         island_dict=np_island_dict,
                         ModelMachine=self.ModelMachine)
                
                print("================================")
                print("Island #%i, error is: "%iIsland)
                print(e)
                print()
                return {"Success":False,"iIsland":iIsland,"ErrorMessage":str(e)}
                ###########################################
        
        ThisIslandModelDict["SteinMedianModel"] = np.array(Model)
        ThisIslandModelDict["SteinStdModel"] = np.array(StdModel)
        
        del(CEv)
        return {"Success":True,"iIsland":iIsland}
    

class ClassEvolveStein_SingleIsland():
    def __init__(self,Dirty,PSF,FreqsInfo,ListPixData=None,ListPixParms=None,IslandBestIndiv=None,
                 WeightFreqBands=None,PixVariance=1e-2,iFacet=0,iIsland=None,island_dict=None,
                 ParallelFitness=False,GD=None,DoPlot=None):

        self.GD=GD
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
        
        NCPU=int(GD["Parallel"]["NCPU"] or psutil.cpu_count())
        if DoPlot is None:
            self.DoPlot=DOPLOT
        else:
            self.DoPlot=DoPlot


        self.ScaleS0="linear"
        #self.ScaleS0="log"

        # # #######################################
        # ScaleS0="linear"
        # PM=ClassParamMachine(ListPixParms,ListPixData,FreqsInfo,
        #                      NOrderPoly=GD["SSD3"]["PolyFreqOrder"],
        #                      SolveParamType=GD["SSD3"]["SolvePars"],
        #                      ScaleS0=ScaleS0)
        # ArrayMethodsMachine=ClassArrayMethodSSD.ClassArrayMethodSSD(Dirty,PSF,ListPixParms,ListPixData,FreqsInfo,
        #                                                             PixVariance=PixVariance,
        #                                                             iFacet=iFacet,
        #                                                             IslandBestIndiv=IslandBestIndiv,
        #                                                             GD=GD,
        #                                                             WeightFreqBands=WeightFreqBands,
        #                                                             iIsland=iIsland,
        #                                                             island_dict=island_dict,
        #                                                             ParallelFitness=ParallelFitness,
        #                                                             NCPU=NCPU,
        #                                                             ScaleS0=ScaleS0)
        # S=PM.ArrayToSubArray(IslandBestIndiv,"Poly0")
        # if np.abs(S).max()==0:
        #     print("LFJKDLSDFLKSDFKLK !!!!!!!!!!!!!!!!!!!!!")
        #     print("LFJKDLSDFLKSDFKLK !!!!!!!!!!!!!!!!!!!!!")
        #     print("LFJKDLSDFLKSDFKLK !!!!!!!!!!!!!!!!!!!!!")
        #     SModelArrayMP,_=self.ArrayMethodsMachine.DeconvCLEAN()
        #     S[:]=SModelArrayMP
        #     T.timeit("CLEAN")
        # # #######################################


        if self.ScaleS0=="log":
            ScaleS0=self.ScaleS0
            PM=ClassParamMachine(ListPixParms,ListPixData,FreqsInfo,
                                 NOrderPoly=GD["SSD3"]["PolyFreqOrder"],
                                 SolveParamType=GD["SSD3"]["SolvePars"],
                                 ScaleS0=ScaleS0)
            S=PM.ArrayToSubArray(IslandBestIndiv,"Poly0")
            Sm=np.max([np.abs(S).max()*1e-6,np.sqrt(PixVariance)])
            S[S<Sm]=Sm
            S[:]=np.log10(S[:])
        
        self.IslandBestIndiv=IslandBestIndiv
        #print("LFLJGLDFJ",self.IslandBestIndiv)
        
        
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
                                                                         ScaleS0=self.ScaleS0)

        self.ConvMachine=self.ArrayMethodsMachine.ConvMachine


        
    def doStein(self):
        
        class MVN:
            def __init__(self, mu, A):
                self.mu = mu
                self.A = A
            def dlnprob(self, theta):
                r=-1*np.matmul(theta-nm.repmat(self.mu, theta.shape[0], 1), self.A)
                return r

        NPoints=self.GD["SSD3"]["PosteriorNPoints"]
        ScaleS0=self.ScaleS0
        class MODEL:
            def __init__(self,ArrayMethodsMachine,DoPos=False):
                self.ArrayMethodsMachine=ArrayMethodsMachine
                
                
            def lnprob(self, Lindividual):
                LChi2=[]
                for individual in Lindividual:
                    ym=self.ArrayMethodsMachine.ToConvArray(individual)
                    y=self.ArrayMethodsMachine.DirtyArray
                    yr=y-ym
                    LChi2.append(np.sum((y-ym)**2/self.ArrayMethodsMachine.PixVariance))
                return np.median(LChi2),np.std(LChi2)
            
            def dlnprob(self, Lindividual):
                L=[]
                for individual in Lindividual:
                    ym=self.ArrayMethodsMachine.ToConvArray(individual)
                    y=self.ArrayMethodsMachine.DirtyArray
                    yr=y-ym
                    yr/=self.ArrayMethodsMachine.PixVariance
                    Cyr=self.ArrayMethodsMachine.ConvMachine.Convolve(yr,InMode="Data",OutMode="Parms")
                    
                    if ScaleS0=="log":
                        S=self.ArrayMethodsMachine.PM.ArrayToSubArray(individual,Type="Poly0")
                        S=S.reshape((1,1,-1))
                        Cyr = Cyr * 10**(S)
                        
                    
                    #Cyr*=np.sign(S)
                    
                    L.append(Cyr.reshape((1,-1)))
                r=np.array(L)
                r=r.reshape((NPoints,self.ArrayMethodsMachine.NParms))
                return r


        ndims=self.IslandBestIndiv.size
        x0=self.IslandBestIndiv.copy()
        x00=x0.flatten().copy()

        if self.ScaleS0=="log":
            S=self.ArrayMethodsMachine.PM.ArrayToSubArray(x0,Type="Poly0")
            Slin0=(10**S).copy()
            Lx0=[]
            for iPoint in range(NPoints):
                Slin=Slin0+np.random.randn(Slin0.size)*np.sqrt(self.ArrayMethodsMachine.PixVariance)
                ssmax=np.abs(Slin).max()/1e10
                Slin[Slin<ssmax]=ssmax
                S[:]=np.log10(Slin[:])
                Lx0.append(x0.copy())
        elif self.ScaleS0=="linear":
            S=self.ArrayMethodsMachine.PM.ArrayToSubArray(x0,Type="Poly0")
            Slin0=S.copy()
            Lx0=[]
            for iPoint in range(NPoints):
                Slin=Slin0+np.random.randn(Slin0.size)*np.sqrt(self.ArrayMethodsMachine.PixVariance)
                S[:]=Slin[:]
                Lx0.append(x0.copy())

        x0=np.array(Lx0)
        #x0=np.random.rand(NPoints,ndims)#*np.sqrt(self.ArrayMethodsMachine.PixVariance)#*0.01
        

        x0[0]=x00[:]
        #ym=self.ArrayMethodsMachine.ToConvArray(self.IslandBestIndiv)


        # ################################
        # IslandBestIndiv (gen code) -> sq model image
        if self.DoPlot:
            A=self.ArrayMethodsMachine.PM.GiveModelArray(self.IslandBestIndiv)
            Im=self.ArrayMethodsMachine.PM.ModelToSquareArray(A,TypeInOut=("Parms","Data"))
            import pylab
            v0=Im.min()
            v1=Im.max()
            fig=pylab.figure("Init")
            pylab.clf()
            ax=pylab.subplot(121)
            im=ax.imshow(Im[0,0],interpolation="nearest",vmin=v0,vmax=v1)
            pylab.colorbar(im)
            ax=pylab.subplot(122)
            im=ax.imshow(Im[1,0],interpolation="nearest",vmin=v0,vmax=v1)
            pylab.colorbar(im)
            pylab.draw()
            pylab.show(block=False)
            pylab.pause(0.1)
        # stop
        # # ################################
        #import warnings
        #warnings.filterwarnings("error")
        M=MODEL(self.ArrayMethodsMachine)
        
        theta=SVGD(M,self.ArrayMethodsMachine).update(x0,
                                                      n_iter=self.GD["SSD3"]["PosteriorNIter"],
                                                      stepsize=self.GD["SSD3"]["PosteriorAlpha"],
                                                      alpha = 0.9,
                                                      DoPlot=self.DoPlot)
            
        # LIM,LDirty=[],[]
        # for V in theta:
        #     ConvModelArray=self.ToConvArray(V)
        #     IM=self.PM.ModelToSquareArray(ConvModelArray,TypeInOut=("Data","Parms"))
        #     LIM.append(IM)
        # IM=np.mean(np.array(LIM),axis=0)
        # Dirty=np.mean(np.array(LDirty),axis=0)

        m0,m1=np.median(theta,axis=0),np.std(theta,axis=0)
        S=self.ArrayMethodsMachine.PM.ArrayToSubArray(m0,"Poly0")
        S[:]=10**(S[:])
        
        return m0,m1
    
