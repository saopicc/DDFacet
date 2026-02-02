'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import os
import numpy as np
import multiprocessing
import time
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("ClassImageDeconvMachineSSD3")
from DDFacet.Other import ClassTimeIt
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Imager import ClassGainMachine
from SkyModel.PSourceExtract import ClassIncreaseIsland
from DDFacet.Imager.SSD3.GA.ClassEvolveGA import ClassEvolveGA
from DDFacet.Imager.SSD3.MCMC.ClassMetropolis import ClassMetropolis
from DDFacet.Imager.SSD3.MultiNest.ClassMultiNest import ClassEvolveStein
from DDFacet.Array import NpParallel
from DDFacet.Imager.SSD3 import ClassIslandDistanceMachine
from DDFacet.Array import shared_dict
import psutil
import copy
import DDFacet.Other.AsyncProcessPool
from DDFacet.Other import MPIManager

#from DDFacet.Imager.ModModelMachine import ClassModModelMachine
from DDFacet.Imager.SSD3 import ClassModelMachineSSD

logger.setSilent("ClassArrayMethodSSD")
logger.setSilent("ClassIsland")

DO_INIT=True
SERIAL=True
# SERIAL=False

class ClassImageDeconvMachine():
    def __init__(self,Gain=0.3,
                 MaxMinorIter=100,NCPU=6,
                 CycleFactor=2.5,FluxThreshold=None,RMSFactor=3,PeakFactor=0,
                 GD=None,SearchMaxAbs=1,IdSharedMem=None,
                 ModelMachine=None,
                 NFreqBands=1,
                 RefFreq=None,
                 MainCache=None,
                 APP=None,
                 **kw    # absorb any unknown keywords arguments into this
                 ):
        #self.im=CasaImage
        self.APP=APP
        self.maincache = MainCache
        self.SearchMaxAbs=SearchMaxAbs
        self.ModelImage=None
        self.MaxMinorIter=MaxMinorIter
        self.NCPU=NCPU
        self.GD=copy.deepcopy(GD)
        self.iMajorCycle=None
        
        from DDFacet.Imager.MultiFields.AppendSubFieldInfo import AppendSubFieldInfo
        AppendSubFieldInfo(self)
        
        self.DicoDicoInitIndiv=None
        if NCPU==0:
            self.NCPU=int(GD["Parallel"]["NCPU"] or psutil.cpu_count())
        self.Chi2Thr=10000
        if IdSharedMem is None:
            self.IdSharedMem=str(os.getpid())
        else:
            self.IdSharedMem=IdSharedMem
        self.IdSharedMem="%s%s"%(self.IdSharedMem,self.StrField)
        self.SubPSF=None
        self.MultiFreqMode=(self.GD["Freq"]["NBand"]>1)
        self.FluxThreshold = FluxThreshold 
        self.CycleFactor = CycleFactor
        self.RMSFactor = RMSFactor
        self.PeakFactor = PeakFactor
        #self.GainMachine=ClassGainMachine.get_instance()
        # if ModelMachine is None:
        #     from DDFacet.Imager.SSD import ClassModelMachineSSD
        #     self.ModelMachine=ClassModelMachineSSD.ClassModelMachine(self.GD,GainMachine=self.GainMachine)
        # else:
        self.ModelMachine = ModelMachine
        self.SteinModelMachine = None
        if self.ModelMachine.DicoSMStacked["Type"]!="SSD3":
            raise ValueError("ModelMachine Type should be SSD3")
        self._CurrentMajorIter=0
        ## If the Model machine was already initialised, it will ignore it in the setRefFreq method
        ## and we need to set the reference freq in PSFServer
        #self.ModelMachine.setRefFreq(self.RefFreq)#,self.PSFServer.AllFreqs)

        # reset overall iteration counter
        self._niter = 0
        self.NChains=self.NCPU

        self.DeconvMode="GAClean"
        
        if self.GD["SSD3"]["PolyFreqOrder"]>NFreqBands:
            raise ValueError("NFreqBands should be greater that SSD3-PolyFreqOrder")
            
        if self.GD["SSD3"]["PolyFreqOrder"]==0:
            # that works but I prefer to forbid it since it's a bit dangerous
            stop
            self.GD["SSD3"]["PolyFreqOrder"]=NFreqBands
        
        self.GD["MultiSliceDeconv"]["PolyFitOrder"]=self.GD["SSD3"]["PolyFreqOrder"]

        # In case PolyFreqOrder=0 in the parset, it has to be set to NFreqBands
        self.ModelMachine.GD["SSD3"]["PolyFreqOrder"]=self.GD["SSD3"]["PolyFreqOrder"]
        self.ModelMachine.setParams()

        
        ListInitType=self.GD["SSD3"]["InitType"]
        if isinstance(ListInitType,str):
            ListInitType=[self.GD["SSD3"]["InitType"]]
        if ListInitType is None:
            ListInitType=[]
        self.ListInitMachine=[]

        for InitType in ListInitType:
            if InitType == "HMP":
                from . import ClassInitSSDModelHMP
                print(ModColor.Str("Initialisation of sourcekins using HMP",col="blue"),file=log)
                self.ListInitMachine.append( ClassInitSSDModelHMP.ClassInitSSDModelParallel(self.GD,
                                                                                            NFreqBands,RefFreq,
                                                                                            MainCache=self.maincache,
                                                                                            IdSharedMem=self.IdSharedMem,
                                                                                            APP=self.APP) )
            elif InitType == "MORESANE":
                from . import ClassInitSSDModelMoresane
                print(ModColor.Str("Initialisation of sourcekins using MORESANE",col="blue"),file=log)
                self.ListInitMachine.append( ClassInitSSDModelMoresane.ClassInitSSDModelParallel(self.GD,
                                                                                                 NFreqBands, RefFreq,
                                                                                                 NCPU=self.NCPU,
                                                                                                 MainCache=self.maincache,
                                                                                                 IdSharedMem=self.IdSharedMem,
                                                                                                 APP=self.APP) )
            elif "MultiSlice" in InitType:
                GD=copy.deepcopy(GD)
                _,SubType=InitType.split(":")
                GD["MultiSliceDeconv"]["Type"]=SubType
                from . import ClassInitSSDModelMultiSlice
                print(ModColor.Str("Initialisation of sourcekins using MultiSlice/%s"%GD["MultiSliceDeconv"]["Type"],col="blue"),file=log)
                ThisMachine=ClassInitSSDModelMultiSlice.ClassInitSSDModelParallel(GD,
                                                                                  NFreqBands, RefFreq,
                                                                                  NCPU=self.NCPU,
                                                                                  MainCache=self.maincache,
                                                                                  IdSharedMem=self.IdSharedMem,
                                                                                  APP=self.APP)
                
                self.ListInitMachine.append( ThisMachine)
            else:
                raise ValueError("InitType should be HMP or MultiSlice or MORESANE")
            
        # if len(self.ListInitMachine)>1 and GD["GAClean"]["NMaxGen"]==0:
        #     stop
            
        # for ThisMachine in self.ListInitMachine:
        #     print("ID",ThisMachine.Type,id(ThisMachine),id(ThisMachine.InitMachine),id(ThisMachine.InitMachine.DeconvMachine))
            
        self.APP.registerJobHandlers(self)
        self._init_machine_initialized = False

    def setMaxMajorIter(self,MaxMajorIter):
        self.MaxMajorIter=MaxMajorIter

    def setMaskMachine(self,MaskMachine):
        self.MaskMachine=MaskMachine

    def setDeconvMode(self,Mode="MetroClean"):
        self.DeconvMode=Mode

    def Reset(self):
        # clear anything we have left lying around in shared memory ## OMS how can this be right, what about others?
        # NpShared.DelAll()
        self._reset_InitMachine()
        facetcache=shared_dict.attach("HMP_InitSSD")
        facetcache.delete()

        
    def GiveModelImage(self,*args):
        return self.ModelMachine.GiveModelImage(*args)

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe
        

    def SetPSF(self,DicoVariablePSF):

        self.PSFServer=ClassPSFServer(self.GD)
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF)
        self.PSFServer.setRefFreq(self.ModelMachine.RefFreq)
        self.DicoVariablePSF=DicoVariablePSF
        #self.NChannels=self.DicoDirty["NChannels"]

        #self.PSFServer.RefFreq=self.ModelMachine.RefFreq

    def _init_InitMachine(self,useCachedHMP=False):
        if not self._init_machine_initialized:
            for InitMachine in self.ListInitMachine:
                kwargs={}
                if InitMachine.Type=="HMP" and useCachedHMP:
                    facetcache=shared_dict.attach("HMP_InitSSD")
                    # print("SDKSFKNSDFKS",facetcache.keys())
                    kwargs={"facetcache":facetcache}
                # print("_init_InitMachine ID",InitMachine.Type,id(InitMachine.InitMachine.DeconvMachine))
                InitMachine.Init(self.DicoVariablePSF, self.GridFreqs, self.DegridFreqs,**kwargs)

                if InitMachine.Type.startswith("MultiSlice"):
                    # cache Taylor term grid
                    InitMachine.InitMachine.DeconvMachine.SetPSF(self.DicoVariablePSF)
                    
                #print(self.DicoVariablePSF.keys())
                
            self._init_machine_initialized = True

    def _reset_InitMachine(self):
        if self._init_machine_initialized:
            for InitMachine in self.ListInitMachine:
                InitMachine.Reset()
            self._init_machine_initialized = False

    def Init(self,**kwargs):
        self.SetPSF(kwargs["PSFVar"])
        self.DicoVariablePSF["PSFSideLobes"]=kwargs["PSFAve"]
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
        self.ModelMachine.setRefFreq(kwargs["RefFreq"])
        # store grid and degrid freqs for ease of passing to MSMF
        #print kwargs["GridFreqs"],kwargs["DegridFreqs"]
        self.GridFreqs=kwargs["GridFreqs"]
        AllDegridFreqs = []
        for i in kwargs["DegridFreqs"].keys():
            AllDegridFreqs.append(kwargs["DegridFreqs"][i])
        self.DegridFreqs = np.unique(np.asarray(AllDegridFreqs).flatten())


    def AdaptArrayShape(self,A,Nout):
        nch,npol,Nin,_=A.shape
        if Nin==Nout: 
            return A
        elif Nin>Nout:
            dx=Nout//2
            B=np.zeros((nch,npol,Nout,Nout),A.dtype)
            print>>log,"  Adapt shapes: %s -> %s"%(str(A.shape),str(B.shape))
            B[:]=A[...,Nin//2-dx:Nin//2+dx+1,Nin//2-dx:Nin//2+dx+1]
            return B
        else:
            stop
            return None

    def SetDirty(self,DicoDirty):
        self.DicoDirty=DicoDirty
        
        self._Dirty=self.DicoDirty["ImageCube"]
        self._MeanDirty=self.DicoDirty["MeanImage"]

        NPSF_x,NPSF_y=self.PSFServer.NPSF
        _,_,NDirty_x,NDirty_y=self._Dirty.shape

        off_x=(NPSF_x-NDirty_x)//2
        off_y=(NPSF_y-NDirty_y)//2

        self.DirtyExtent=(off_x,off_x+NDirty_x,off_y,off_y+NDirty_y)

        if self.ModelImage is None:
            self._ModelImage=np.zeros_like(self._Dirty)
        self.ModelMachine.setModelShape(self._Dirty.shape)

    def SearchIslands(self,Threshold):

        # if self.MaskMachine.CurrentNegMask is None:
        #     raise RuntimeError("SSD requires either a user supplied FITS mask or automasking to be enabled. Check your options.")

        if self.GD["SSD3"]["UniqueIsland"]:
            NegMask=np.zeros((self.DicoDirty["MeanImage"].shape),bool)
        else:
            #Mask = (self.MaskMachine.CurrentMask | ModelMask)
            #NegMask = (Mask==0)
            NegMask=self.MaskMachine.CurrentNegMask
        ModelMask=self.ModelMachine.giveMask_nonZeroModel()
            
        
        IslandDistanceMachine=ClassIslandDistanceMachine.ClassIslandDistanceMachine(self.GD,
                                                                                    NegMask,#self.MaskMachine.CurrentNegMask,
                                                                                    self.PSFServer,
                                                                                    self.DicoDirty,
                                                                                    IdSharedMem=self.IdSharedMem)
        ListIslands=IslandDistanceMachine.SearchIslands(Threshold)
        # FluxIslands=[]
        # for iIsland in range(len(ListIslands)):
        #     x,y=np.array(ListIslands[iIsland]).T
        #     FluxIslands.append(np.sum(Dirty[0,0,x,y]))
        # ind=np.argsort(np.array(FluxIslands))[::-1]

        # _,_,nx,ny=self._Dirty.shape
        # self.iIslandImage=np.zeros_like(self._Dirty)
        # for i in range(0,nx,5):
        #     print("%i/%i"%(i,nx))
        #     for j in range(0,ny,5):
        #         FacetID=self.PSFServer.giveFacetID2(i,j)
        #         self.iIslandImage[:,:,i,j]=FacetID
                
        # ListIslandsSort=[ListIslands[i] for i in ind]
        

        # ListIslands=self.CalcCrossIslandFlux(ListIslandsSort)

        
        #ListIslands=[np.load("errIsland_000524.npy").tolist()]

        if not self.GD["SSD3"]["UniqueIsland"]:
            ListIslands=IslandDistanceMachine.CalcCrossIslandFlux(ListIslands)
            if self.GD["SSD3"]["ConvexifyIslands"]:
                ListIslands=IslandDistanceMachine.ConvexifyIsland(ListIslands)
            
        # ListIslands=IslandDistanceMachine.MergeIslands(ListIslands)
        ListIslands=IslandDistanceMachine.BreakLargeIslands(ListIslands)

        # #############################
        # Filter by peak flux 
        ListIslandsFiltered=[]
        Dirty=self.DicoDirty["MeanImage"]
        log.print("Filter islands...")
        N_PixelInCurrentModel,N_Th=0,0
        
        for iIsland in range(len(ListIslands)):
            x,y=np.array(ListIslands[iIsland]).T
            PixVals=Dirty[0,0,x,y]
            ModelMaskVals=ModelMask[0,0,x,y]
            DoThisOne=False
            MaxIsland=np.max(np.abs(PixVals))
            C_PixelInCurrentModel=ModelMaskVals.any()
            C_Th=(MaxIsland>Threshold)
            N_PixelInCurrentModel+=C_PixelInCurrentModel
            N_Th+=C_Th
            if C_Th or C_PixelInCurrentModel:
                ListIslandsFiltered.append(ListIslands[iIsland])
        print("  selected %i islands [out of %i] with peak flux > %.3g Jy [%i], or with pixel in previous ModelMachine [%i]"%(len(ListIslandsFiltered),len(ListIslands),Threshold, N_Th, N_PixelInCurrentModel), file=log)
        if len(ListIslandsFiltered)==0:
            return "NoIslands"
        ListIslands=ListIslandsFiltered
        # #############################
        
        ListIslands,ListSpacialWeight=IslandDistanceMachine.IncreaseIslands(ListIslands)

        #ListSpacialWeight_App=[]
        for Island,W in zip(ListIslands,ListSpacialWeight):
            x,y=Island.T
            xc=int(np.mean(x))
            yc=int(np.mean(y))
            W[:]*=self.MeanJonesNorm[0,x,y].flat[:]
            # W[:]*=self.MeanJonesNorm[0,xc,yc]
            
        
        self.LabelIslandsImage=None#IslandDistanceMachine.CalcLabelImage(ListIslands)

        self.ListAllIslands=ListIslands
        self.NIslands=len(self.ListAllIslands)

        print("Sorting islands by size", file=log)
        Sz=np.array([len(self.ListAllIslands[iIsland]) for iIsland in range(self.NIslands)])
        #print ":::::::::::::::::"
        ind=np.argsort(Sz)[::-1]

        # sorted_indices = sorted(range(len(data)), key=lambda i: (data[i][1], data[i][0]))
        # stop
        
        ListIslandsOut=[self.ListAllIslands[i] for i in ind]
        self.ListAllIslands=ListIslandsOut#[100::10][0:1]
        self.ListAllSpacialWeight=[ListSpacialWeight[i] for i in ind]
        self.NIslands=len(self.ListAllIslands)
        




    def setChannel(self,ch=0):
        self.Dirty=self._MeanDirty[ch]
        self.ModelImage=self._ModelImage[ch]

    def GiveThreshold(self,Max):
        return ((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*Max if self.CycleFactor else 0

    def Deconvolve(self,ch=0):
        # for ThisMachine in self.ListInitMachine:
        #     print("ID",ThisMachine.Type,id(ThisMachine),id(ThisMachine.InitMachine),id(ThisMachine.InitMachine.DeconvMachine))

            
        if self._niter >= self.MaxMinorIter:
            return "MaxIter", False, False

        self.setChannel(ch)

        _,npix,_=self.Dirty.shape
        xc=(npix)//2

        npol,_,_=self.Dirty.shape

        m0,m1=self.Dirty[0].min(),self.Dirty[0].max()
        
        DoAbs=int(self.GD["Deconv"]["AllowNegative"])
        print("  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter,self.MaxMinorIter,DoAbs), file=log)

        NPixStats=1000
        #RandomInd=np.int64(np.random.rand(NPixStats)*npix**2)
        RandomInd=np.int64(np.linspace(0,self.Dirty.size-1,NPixStats))
        RMS=np.std(np.real(self.Dirty.ravel()[RandomInd]))
        #print "::::::::::::::::::::::"
        self.RMS=RMS
        
        self.DicoDirty["RMS"]=RMS

        LRMS=[]
        nch=self.DicoDirty["ImageCube"].shape[0]
        for ich in range(nch):
            ThisRMS=np.std(np.real(self.DicoDirty["ImageCube"][ich].flat[RandomInd]))
            LRMS.append(ThisRMS)
        self.DicoDirty["LRMS"]=LRMS
        #self.GainMachine.SetRMS(RMS)
        
        Fluxlimit_RMS = self.RMSFactor*RMS

        x,y,MaxDirty=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskMachine.CurrentNegMask)
        #MaxDirty=np.max(np.abs(self.Dirty))
        #Fluxlimit_SideLobe=MaxDirty*(1.-self.SideLobeLevel)
        #Fluxlimit_Sidelobe=self.CycleFactor*MaxDirty*(self.SideLobeLevel)
        Fluxlimit_Peak = MaxDirty*self.PeakFactor
        Fluxlimit_Sidelobe = self.GiveThreshold(MaxDirty)

        mm0,mm1=self.Dirty.min(),self.Dirty.max()

        # work out uper threshold
        StopFlux = max(Fluxlimit_Peak, Fluxlimit_RMS, Fluxlimit_Sidelobe, Fluxlimit_Peak, self.FluxThreshold)

        self._CurrentMajorIter+=1
        NLastCyclesDeconvAll=self.GD["SSD3"]["NLastCyclesDeconvAll"]
        print("SSD3 Cycle %i/%i, NLastCyclesDeconvAll=%i)"%(self._CurrentMajorIter,self.MaxMajorIter,NLastCyclesDeconvAll), file=log)
        print("    Dirty image peak flux      = %10.6f Jy [(min, max) = (%.3g, %.3g) Jy]"%(MaxDirty,mm0,mm1), file=log)
        print("      RMS-based threshold      = %10.6f Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor), file=log)
        print("      Sidelobe-based threshold = %10.6f Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor), file=log)
        print("      Peak-based threshold     = %10.6f Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor), file=log)
        print("      Absolute threshold       = %10.6f Jy"%(self.FluxThreshold), file=log)
        self.ThSpectralFit=None
        #self.ThSpectralFit=1.
        if self.GD["SSD3"]["NLastCyclesDeconvAll"]==-1 or (self._CurrentMajorIter > (self.MaxMajorIter-NLastCyclesDeconvAll)) :
            print(ModColor.Str("    ... overwriting these values with zero",col="green"), file=log)
            StopFlux=0.
            
        self.ThSpectralFit=3
        print("    Stopping flux              = %10.6f Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty), file=log)

        self.ModelMachine.updateAlpha(self._MeanDirty)
        
        # MaxModelInit=np.max(np.abs(self.ModelImage))
        # Fact=4
        # self.BookKeepShape=(npix/Fact,npix/Fact)
        # BookKeep=np.zeros(self.BookKeepShape,np.float32)
        # NPixBook,_=self.BookKeepShape
        # FactorBook=float(NPixBook)/npix
        
        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskMachine.CurrentNegMask)

        if ThisFlux < StopFlux:
            print(ModColor.Str("    Initial maximum peak %g Jy below threshold, we're done here" % (ThisFlux),col="green" ), file=log)
            return "FluxThreshold", False, False
        
        FreqsModel=np.array([np.mean(self.DicoVariablePSF["freqs"][iBand]) for iBand in range(len(self.DicoVariablePSF["freqs"]))])
        ModelImage=self.ModelMachine.GiveModelImage(FreqsModel)
        if "SmoothJonesNorm" in self.DicoDirty.keys():
            MeanJonesNorm=np.mean(self.DicoDirty["SmoothJonesNorm"],axis=0)
            ModelImageApp=ModelImage*np.sqrt(self.DicoDirty["SmoothJonesNorm"])
        else:
            ModelImageApp=ModelImage*np.sqrt(self.DicoDirty["JonesNorm"])
            MeanJonesNorm=np.mean(self.DicoDirty["JonesNorm"],axis=0)
        self.ModelImageApp=ModelImageApp
        self.MeanJonesNorm=MeanJonesNorm
        
        
        ###########################
        self.ListAllIslands=[]
        self.ListAllSpacialWeight=[]
        if MPIManager.rank==0:
            rep=self.SearchIslands(StopFlux)
            if rep=="NoIslands":
                return "FluxThreshold", False, False
        ###########################

            

        if not MPIManager.useMPI:
            DicoJobIslands={0:np.arange(len(self.ListAllIslands)).tolist()}
        else:
            DicoJobIslands={}
            if MPIManager.rank==0:
                irank=0
                nrank=MPIManager.size
                for iIsland in range(len(self.ListAllIslands)):
                    L=DicoJobIslands.get(irank,[])
                    L.append(iIsland)
                    DicoJobIslands[irank]=L
                    irank+=1
                    if irank==nrank: irank=0

            DicoJobIslands=MPIManager.COMM_WORLD.bcast(DicoJobIslands, root=0)
            log.print("Broadcast islands")
            self.ListAllIslands=MPIManager.COMM_WORLD.bcast(self.ListAllIslands, root=0)
            self.ListAllSpacialWeight=MPIManager.COMM_WORLD.bcast(self.ListAllSpacialWeight, root=0)
            # self.ListIslands=[self.ListAllIslands[iIsland] for iIsland in DicoJobIslands.get(MPIManager.rank,[])]
            # self.ListSpacialWeight=[self.ListAllSpacialWeight[iIsland] for iIsland in DicoJobIslands.get(MPIManager.rank,[])]

        self.ListJobIslands=DicoJobIslands[MPIManager.rank]
        self.NIslands=len(self.ListJobIslands)
        log.print("Number of islands to deconvolve: %i [/%i Total]"%(len(self.ListJobIslands),len(self.ListAllIslands)))
        
        

        
            
        allIslandModelDict  = shared_dict.create("DeconvListIslands%s"%self.StrField)
        for iIsland in self.ListJobIslands:
            IslandXY=self.ListAllIslands[iIsland]
            IslandXY=np.array(IslandXY)
            allIslandModelDict.addSubdict(iIsland)
            allIslandModelDict[iIsland].addSharedArray("IslandXY", IslandXY.shape, np.int32)
            allIslandModelDict[iIsland]["IslandXY"][:]=IslandXY[:]

        
        ParmDict = shared_dict.create("ParmDict%s"%self.StrField) # ParmDict
        ParmDict["ModelImageInt"] = ModelImage
        ParmDict["ModelImageApp"] = ModelImageApp
        ParmDict["GridFreqs"] = self.GridFreqs
        ParmDict["DegridFreqs"] = self.DegridFreqs
        ParmDict["RMS"] = RMS
        ParmDict["iMajor"] = self._CurrentMajorIter
        
        self._init_InitMachine()
        
        log.print("Deconvolving %i islands"%(self.NIslands))
        
        DoAbs=int(self.GD["Deconv"]["AllowNegative"])

        logger.setSilent(["AsyncProcessPool"])
        self.APP_GA=DDFacet.Other.AsyncProcessPool.initNew(Name="APP_GA",
                                                           ncpu=self.GD["Parallel"]["NCPU"],
                                                           affinity="disable",
                                                           )
        self.APP_GA.registerJobHandlers(self)
        self.APP_GA.startWorkers()
        self.APP_GA.awaitWorkerStart()
        logger.setLoud(["AsyncProcessPool"])
        
        # ############################################
        ParallelMode="OverIslands"
        NDeconv=0
        NDeconvBig=0
        for iIsland in self.ListJobIslands:
            Island=self.ListAllIslands[iIsland]
            if len(Island)>self.GD["SSDClean"]["ConvFFTSwitch"]:
                ParallelMode="PerIsland"
            else:
                ParallelMode="OverIslands"

            self.APP_GA.runJob("initIsland.%i"%(iIsland),
                               self._initIsland,
                               args=(iIsland,self.DicoDirty.path,self.DicoVariablePSF.path,self.GridFreqs,self.DegridFreqs,self.ThSpectralFit,ParallelMode), serial=SERIAL) 
               
        LDicoResults=self.APP_GA.awaitJobResults("initIsland.*", progress="Deconv Islands")
        if MPIManager.useMPI: MPIManager.COMM_WORLD.Barrier()
        # ############################################
        # Collect results
        DTime={}
        LNSpectralFit=[]
        for DicoResult in LDicoResults:
            if DicoResult is None: continue
            DInfo=DicoResult.get("DInfo",None)
            if DInfo is None: continue
            
            for InitType in DInfo.keys():
                L=DTime.get(InitType,[])
                L.append(DInfo[InitType]["Time"])
                DTime[InitType]=L
                NSpectralFit=DInfo[InitType].get("NSpectralFit",None)
                # print("[%s] %s"%(InitType,str(NSpectralFit)))
                if NSpectralFit is not None:
                    LNSpectralFit.append(NSpectralFit)
        Lkey=list(DTime.keys())
        # #######################
        # Total times
        Ttot=[np.sum(DTime[InitType]) for InitType in Lkey]
        TtotTot=np.sum(Ttot)
        TFrac=[100*np.sum(DTime[InitType])/TtotTot for InitType in Lkey]
        Lss=[]
        for iInitType,InitType in enumerate(Lkey):
            Lss.append("[%s] %.1f min. (%.1f%%)"%(InitType,Ttot[iInitType]/60,TFrac[iInitType]))
        ss="Initialisation times [total]: %s"%(", ".join(Lss))
        log.print(ss)
        # #######################
        # Max times
        Ttot=[np.max(DTime[InitType]) for InitType in Lkey]
        TtotTot=np.sum(Ttot)
        TFrac=[100*np.max(DTime[InitType])/TtotTot for InitType in Lkey]
        Lss=[]
        for iInitType,InitType in enumerate(Lkey):
            Lss.append("[%s] %.1f min. (%.1f%%)"%(InitType,Ttot[iInitType]/60,TFrac[iInitType]))
        ss="                       [max]: %s"%(", ".join(Lss))
        log.print(ss)
        # #######################

        
        NSpectralFit=np.array(LNSpectralFit)
        #print("DLFKFDLKDF NSpectralFit",NSpectralFit)
        if NSpectralFit.size>0:
            nPixFit,nPixTot=np.sum(NSpectralFit,axis=0)
            #print("DLFKFDLKDF nPixFit,nPixTot",nPixFit,nPixTot)
            if nPixTot>0:
                fracFit=100*nPixFit/nPixTot
                sss="[%i/%i, Th=%.1f]"%(nPixFit,nPixTot,self.ThSpectralFit)
                log.print("[MultiSlice] precise spectral fit done for %.2f%% of the pixels %s"%(fracFit,sss))
            else:
                fracFit=0.
                sss=""
                log.print("[MultiSlice] precise spectral fit likely skipped for all pixels [Th=%.1f]"%self.ThSpectralFit)
                

        
        self.APP_GA.terminate()
        self.APP_GA.shutdown()

        
        # GA final estimate    
        allIslandModelDict  = shared_dict.attach("DeconvListIslands%s"%self.StrField)
        allIslandModelDict.reload()
        
        DicoIslandsOut={}
        for iRes,DicoResult in enumerate(LDicoResults):
            if not DicoResult["Success"]:
                continue
            iIsland=DicoResult["iIsland"]
            ThisIslandModelDict = allIslandModelDict[iIsland]
            ThisIslandModelDict.reload()
            DicoIslandsOut[iIsland]=ThisIslandModelDict["Model"]
        LDicoIslandsOut=[DicoIslandsOut]
        
        if MPIManager.useMPI: MPIManager.COMM_WORLD.Barrier()
        
        log.print("Have fitted %i islands"%len(DicoIslandsOut))
        
        if MPIManager.useMPI:
            log.print("  Gather islands from ranks...")
            LDicoIslandsOut = MPIManager.COMM_WORLD.gather(DicoIslandsOut,root=0)
            log.print("      got them...")
            
        if MPIManager.rank==0:
            DicoIslandsOut1={}
            for D in LDicoIslandsOut:
                for iIsland in D.keys():
                    DicoIslandsOut1[iIsland]=D[iIsland]
            DicoIslandsOut=DicoIslandsOut1

        if MPIManager.useMPI: MPIManager.COMM_WORLD.Barrier()
        
        if MPIManager.rank==0:
            log.print("  Reinit islands in ModelMachine...")
            self.ModelMachine.reinitIslands(self.ListAllIslands)
            
            log.print("  Update islands...")
            for iIsland in sorted(list(DicoIslandsOut.keys())):
                Model=DicoIslandsOut[iIsland]
                self.ModelMachine.AppendIsland(self.ListAllIslands[iIsland],
                                            Model,
                                            W=self.ListAllSpacialWeight[iIsland])
                




            log.print("  Renormalise...")
            self.ModelMachine.RenormaliseMultiEstimatesPerPixel()
            log.print("  Done SSD3...")


        if self.GD["Misc"]["ConserveMemory"]:
            self._reset_InitMachine()
        self.Reset()#_reset_InitMachine()
        
        allIslandModelDict.delete()
        
        return "MaxIter", True, True   # stop deconvolution but do update model

    def _updateWorkerInternals(self,DicoDirty_path,DicoPSF_path,GridFreqs,DegridFreqs):
        
        DicoDirty  = shared_dict.attach(DicoDirty_path)
        if self.iMajorCycle == DicoDirty["iMajorCycle"]: return 
        self.DicoDirty=DicoDirty
        self.iMajorCycle=self.DicoDirty["iMajorCycle"]
        
        self.ModelImageApp = shared_dict.attach("ParmDict%s"%self.StrField)["ModelImageApp"]
        self.ModelImageInt = shared_dict.attach("ParmDict%s"%self.StrField)["ModelImageInt"]
        
        self.DicoVariablePSF = shared_dict.attach(DicoPSF_path)
        self.SetPSF(self.DicoVariablePSF)

        self.SetDirty(self.DicoDirty)
        self.GridFreqs = GridFreqs
        self.DegridFreqs = DegridFreqs

    
    def _initIsland(self,
                    iIsland,DicoDirty_path,DicoPSF_path,GridFreqs,DegridFreqs,ThSpectralFit,ParallelMode):

        self.DicoInitIndiv={}


        self.ThSpectralFit=ThSpectralFit
        
        LSilent=["ClassInitSSDModelHMP", "ClassMultiScaleMachine", "ClassInitSSDModelMultiSlice", "ClassImageDeconvMachineMSMF"]
        logger.setSilent(LSilent)
        
        self.ListSizeIslands=[]
        #ThisPixList=ListIslands[iIsland]
        
        allIslandModelDict  = shared_dict.attach("DeconvListIslands%s"%self.StrField)
        
        ThisPixList=allIslandModelDict[iIsland]["IslandXY"]
        
        x,y=np.array(ThisPixList,dtype=np.float32).T
        dx,dy=x.max()-x.min(),y.max()-y.min()
        dd=np.max([dx,dy])+1
        DoIslandsInit = (dd>=self.GD["GAClean"]["MinSizeInit"])
        
        self._updateWorkerInternals(DicoDirty_path,DicoPSF_path,GridFreqs,DegridFreqs)
        
        self._init_InitMachine(useCachedHMP=True)
        
        # print(self.GridFreqs,self.DegridFreqs)

        # Initialise the model using various deconv machines
        DicoInitModel  = {} # shared_dict.attach("DicoDicoInitIndiv%s"%self.StrField)
        t0=time.time()
        DInfo={}
        for iMachine,InitMachine in enumerate(self.ListInitMachine):
            kwargs={}
            if InitMachine.Type=="MultiSlice":
                kwargs["ThSpectralFit"]=self.ThSpectralFit
                
            if self.GD["GAClean"]["MinSizeInit"]==-1:
                continue
            if not DoIslandsInit or not DO_INIT:
                continue
        
            rep = InitMachine.giveDicoInitIndiv(Island=ThisPixList,
                                                #ListIslands=ListIslands,
                                                iIsland=iIsland,
                                                #ModelImage=ModelImage,
                                                DicoDirty=self.DicoDirty,
                                                **kwargs)
            #InitMachine.Reset()
            #print("SDKSDFKJSFKLJSF",rep,type(rep))
            #self.DicoDicoInitIndiv[iMachine].addSubdict(iIsland)
            t1=time.time()
            
            DInfo[InitMachine.Type]={}
            DInfo[InitMachine.Type]["Time"]=t1-t0
            if InitMachine.Type=="MultiSlice":
                rep,NSpectralFit=rep
                DInfo[InitMachine.Type]["NSpectralFit"]=NSpectralFit
                
            t0=t1
            DicoInitModel[iMachine] = rep
            #print("FLSKSFDLDKLSDK Init Type,Max",InitMachine.Type,rep.max())

        logger.setLoud(LSilent)

        # logger.setSilent(["AsyncProcessPool"])
        
        t0=time.time()
        GAMachine=ClassEvolveGA(self,ParallelMode)
        GAMachine.setDicoInitModel(DicoInitModel)
        DicoResult=GAMachine._runGA(iIsland,self.DicoDirty.path,self.DicoVariablePSF.path,self.GridFreqs,self.DegridFreqs)
        t1=time.time()
        DInfo["GA"]={}
        DInfo["GA"]["Time"]=t1-t0
        DicoResult["DInfo"]=DInfo
        del(GAMachine)

        if  self.GD["SSD3"]["Posterior"] and self._CurrentMajorIter==self.MaxMajorIter:
            log.print("Doing Stein VGD to estimate the posterior.")
            self.SteinModelMachine = ClassModelMachineSSD.ClassModelMachine(self.GD)
            self.SteinModelMachine.setRefFreq(self.ModelMachine.RefFreq)
            self.SteinModelMachine.setModelShape(self.ModelMachine.ModelShape)
            self.SteinMachine=ClassEvolveStein(self)
            self.SteinMachine.runStein_AllIslands()
            del(self.SteinMachine)

        
        
        return DicoResult




        


    # ########################################################
    

    def Update(self,DicoDirty,**kwargs):
        """
        Method to update attributes from ClassDeconvMachine
        """
        #Update image dict
        self.SetDirty(DicoDirty)

    def ToFile(self, fname):
        """
        Write model dict to file
        """
        self.ModelMachine.ToFile(fname)

    def FromFile(self, fname):
        """
        Read model dict from file SubtractModel
        """
        self.ModelMachine.FromFile(fname)

    def FromDico(self, DicoName):
        """
        Read in model dict
        """
        self.ModelMachine.FromDico(DicoName)

        
