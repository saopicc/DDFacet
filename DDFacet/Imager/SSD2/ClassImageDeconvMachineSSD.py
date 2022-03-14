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
log=logger.getLogger("ClassImageDeconvMachineSSD2")
from DDFacet.Other import ClassTimeIt
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Imager import ClassGainMachine
from SkyModel.PSourceExtract import ClassIncreaseIsland
from DDFacet.Imager.SSD2.GA.ClassEvolveGA import ClassEvolveGA
from DDFacet.Imager.SSD2.MCMC.ClassMetropolis import ClassMetropolis
from DDFacet.Array import NpParallel
from DDFacet.Imager.SSD2 import ClassIslandDistanceMachine
from DDFacet.Array import shared_dict
import psutil
import copy

logger.setSilent("ClassArrayMethodSSD")
logger.setSilent("ClassIsland")


class ClassImageDeconvMachine():
    def __init__(self,Gain=0.3,
                 MaxMinorIter=100,NCPU=6,
                 CycleFactor=2.5,FluxThreshold=None,RMSFactor=3,PeakFactor=0,
                 GD=None,SearchMaxAbs=1,IdSharedMem=None,
                 ModelMachine=None,
                 NFreqBands=1,
                 RefFreq=None,
                 MainCache=None,
                 **kw    # absorb any unknown keywords arguments into this
                 ):
        #self.im=CasaImage
        self.maincache = MainCache
        self.SearchMaxAbs=SearchMaxAbs
        self.ModelImage=None
        self.MaxMinorIter=MaxMinorIter
        self.NCPU=NCPU
        self.GD=copy.deepcopy(GD)
        self.DicoDicoInitIndiv=None
        if NCPU==0:
            self.NCPU=int(GD["Parallel"]["NCPU"] or psutil.cpu_count())
        self.Chi2Thr=10000
        if IdSharedMem is None:
            self.IdSharedMem=str(os.getpid())
        else:
            self.IdSharedMem=IdSharedMem
        self.SubPSF=None
        self.MultiFreqMode=(self.GD["Freq"]["NBand"]>1)
        self.FluxThreshold = FluxThreshold 
        self.CycleFactor = CycleFactor
        self.RMSFactor = RMSFactor
        self.PeakFactor = PeakFactor
        self.GainMachine=ClassGainMachine.get_instance()
        # if ModelMachine is None:
        #     from DDFacet.Imager.SSD import ClassModelMachineSSD
        #     self.ModelMachine=ClassModelMachineSSD.ClassModelMachine(self.GD,GainMachine=self.GainMachine)
        # else:
        self.ModelMachine = ModelMachine
        if self.ModelMachine.DicoSMStacked["Type"]!="SSD2":
            raise ValueError("ModelMachine Type should be SSD2")
        self._CurrentMajorIter=0
        ## If the Model machine was already initialised, it will ignore it in the setRefFreq method
        ## and we need to set the reference freq in PSFServer
        #self.ModelMachine.setRefFreq(self.RefFreq)#,self.PSFServer.AllFreqs)

        # reset overall iteration counter
        self._niter = 0
        self.NChains=self.NCPU

        self.DeconvMode="GAClean"
        
        if self.GD["SSD2"]["PolyFreqOrder"]>NFreqBands:
            stop
            
        if self.GD["SSD2"]["PolyFreqOrder"]==0:
            # that works but I prefer to forbid it since it's a bit dangerous
            stop
            self.GD["SSD2"]["PolyFreqOrder"]=NFreqBands
        
        self.GD["MultiSliceDeconv"]["PolyFitOrder"]=self.GD["SSD2"]["PolyFreqOrder"]

        # In case PolyFreqOrder=0 in the parset, it has to be set to NFreqBands
        self.ModelMachine.GD["SSD2"]["PolyFreqOrder"]=self.GD["SSD2"]["PolyFreqOrder"]
        self.ModelMachine.setParams()

        
        ListInitType=self.GD["SSD2"]["InitType"]
        if isinstance(ListInitType,str):
            ListInitType=[self.GD["SSD2"]["InitType"]]
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
                                                                                            IdSharedMem=self.IdSharedMem) )
            elif InitType == "MORESANE":
                from . import ClassInitSSDModelMoresane
                print(ModColor.Str("Initialisation of sourcekins using MORESANE",col="blue"),file=log)
                self.ListInitMachine.append( ClassInitSSDModelMoresane.ClassInitSSDModelParallel(self.GD,
                                                                                                 NFreqBands, RefFreq,
                                                                                                 NCPU=self.NCPU,
                                                                                                 MainCache=self.maincache,
                                                                                                 IdSharedMem=self.IdSharedMem) )
            elif "MultiSlice" in InitType:
                GD=copy.deepcopy(GD)
                _,SubType=InitType.split(":")
                GD["MultiSliceDeconv"]["Type"]=SubType
                from . import ClassInitSSDModelMultiSlice
                print(ModColor.Str("Initialisation of sourcekins using MultiSlice/%s"%GD["MultiSliceDeconv"]["Type"],col="blue"),file=log)
                self.ListInitMachine.append( ClassInitSSDModelMultiSlice.ClassInitSSDModelParallel(GD,
                                                                                                   NFreqBands, RefFreq,
                                                                                                   NCPU=self.NCPU,
                                                                                                   MainCache=self.maincache,
                                                                                                   IdSharedMem=self.IdSharedMem) )
                
            else:
                raise ValueError("InitType should be HMP or MultiSlice or MORESANE")

        if len(self.ListInitMachine)>1 and GD["GAClean"]["NMaxGen"]==0:
            stop
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

    def _init_InitMachine(self):
        if not self._init_machine_initialized:
            for InitMachine in self.ListInitMachine:
                InitMachine.Init(self.DicoVariablePSF, self.GridFreqs, self.DegridFreqs)
                
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

        NPSF=self.PSFServer.NPSF
        _,_,NDirty,_=self._Dirty.shape

        off=(NPSF-NDirty)//2

        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)

        if self.ModelImage is None:
            self._ModelImage=np.zeros_like(self._Dirty)
        self.ModelMachine.setModelShape(self._Dirty.shape)

    def SearchIslands(self,Threshold):

        if self.MaskMachine.CurrentNegMask is None:
            raise RuntimeError("SSD requires either a user supplied FITS mask or automasking to be enabled. Check your options.")

        IslandDistanceMachine=ClassIslandDistanceMachine.ClassIslandDistanceMachine(self.GD,
                                                                                    self.MaskMachine.CurrentNegMask,
                                                                                    self.PSFServer,
                                                                                    self.DicoDirty,
                                                                                    IdSharedMem=self.IdSharedMem)
        ListIslands=IslandDistanceMachine.SearchIslands(Threshold)
        # FluxIslands=[]
        # for iIsland in range(len(ListIslands)):
        #     x,y=np.array(ListIslands[iIsland]).T
        #     FluxIslands.append(np.sum(Dirty[0,0,x,y]))
        # ind=np.argsort(np.array(FluxIslands))[::-1]

        # ListIslandsSort=[ListIslands[i] for i in ind]
        

        # ListIslands=self.CalcCrossIslandFlux(ListIslandsSort)

        # #############################
        # Filter by peak flux 
        ListIslandsFiltered=[]
        Dirty=self.DicoDirty["MeanImage"]
        for iIsland in range(len(ListIslands)):
            x,y=np.array(ListIslands[iIsland]).T
            PixVals=Dirty[0,0,x,y]
            DoThisOne=False
            
            MaxIsland=np.max(np.abs(PixVals))

           # print "island %i [%i]: %f"%(iIsland,x.size,MaxIsland)

#            if (MaxIsland>(3.*self.RMS))|(MaxIsland>Threshold):
            if (MaxIsland>Threshold):
                ListIslandsFiltered.append(ListIslands[iIsland])
            # else:
            #     self.MaskMachine.CurrentNegMask[:,:,x,y]=1
            #     self.MaskMachine.CurrentMask[:,:,x,y]=0
            # ###############################
            # if np.max(np.abs(PixVals))>Threshold:
            #     DoThisOne=True
            #     self.IslandHasBeenDone[0,0,x,y]=1
            # if ((DoThisOne)|self.IslandHasBeenDone[0,0,x[0],y[0]]):
            #     self.ListIslands.append(ListIslands[iIsland])
            # ###############################
        # #############################
        print("  selected %i islands [out of %i] with peak flux > %.3g Jy"%(len(ListIslandsFiltered),len(ListIslands),Threshold), file=log)
        ListIslands=ListIslandsFiltered
        #ListIslands=[np.load("errIsland_000524.npy").tolist()]
        
        ListIslands=IslandDistanceMachine.CalcCrossIslandFlux(ListIslands)
        ListIslands=IslandDistanceMachine.ConvexifyIsland(ListIslands)
        ListIslands=IslandDistanceMachine.MergeIslands(ListIslands)
        ListIslands=IslandDistanceMachine.BreakLargeIslands(ListIslands)
        
        self.LabelIslandsImage=IslandDistanceMachine.CalcLabelImage(ListIslands)

        self.ListIslands=ListIslands
        
        self.NIslands=len(self.ListIslands)

        print("Sorting islands by size", file=log)
        Sz=np.array([len(self.ListIslands[iIsland]) for iIsland in range(self.NIslands)])
        #print ":::::::::::::::::"
        ind=np.argsort(Sz)[::-1]

        ListIslandsOut=[self.ListIslands[i] for i in ind]
        self.ListIslands=ListIslandsOut#[100::10][0:1]
        self.NIslands=len(self.ListIslands)
        



    def InitIslands(self):
        self.DicoInitIndiv={}
        if self.GD["GAClean"]["MinSizeInit"]==-1: return

        DoAbs=int(self.GD["Deconv"]["AllowNegative"])
        print("  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter,self.MaxMinorIter,DoAbs), file=log)

        # ##########################################################################
        # # Init SSD model using MSMF

        FreqsModel=np.array([np.mean(self.DicoVariablePSF["freqs"][iBand]) for iBand in range(len(self.DicoVariablePSF["freqs"]))])
        ModelImage=self.ModelMachine.GiveModelImage(FreqsModel)
        ModelImage*=np.sqrt(self.DicoDirty["JonesNorm"])
        # ######################
        # SERIAL
        # InitMachine=ClassInitSSDModel.ClassInitSSDModel(self.GD,
        #                                                      self.DicoVariablePSF,
        #                                                      self.DicoDirty,
        #                                                      self.ModelMachine.RefFreq,
        #                                                      MainCache=self.maincache)
        # InitMachine.setSSDModelImage(ModelImage)
        # DicoInitIndiv={}
        # for iIsland,Island in enumerate(self.ListIslands):
        #     SModel,AModel=InitMachine.giveModel(Island)
        #     DicoInitIndiv[iIsland]={"S":SModel,"Alpha":AModel}
        # self.DicoInitIndiv=DicoInitIndiv
        # ######################
        # Parallel
        self.ListSizeIslands=[]
        for ThisPixList in self.ListIslands:
            x,y=np.array(ThisPixList,dtype=np.float32).T
            dx,dy=x.max()-x.min(),y.max()-y.min()
            dd=np.max([dx,dy])+1
            self.ListSizeIslands.append(dd)


        ListDoIslandsInit=[True if self.ListSizeIslands[iIsland]>=self.GD["GAClean"]["MinSizeInit"] else False for iIsland in range(len(self.ListIslands))]

        #ListDoMSMFIslandsInit=[True if iIsland==16 else False for iIsland in range(len(self.ListIslands))]



        print("  selected %i islands larger than %i pixels for initialisation"%(np.count_nonzero(ListDoIslandsInit),self.GD["GAClean"]["MinSizeInit"]), file=log)

        self._init_InitMachine()
        if self.DicoDicoInitIndiv is not None:
            self.DicoDicoInitIndiv.delete()
            
        self.DicoDicoInitIndiv  = shared_dict.create("DicoDicoInitIndiv")
        if np.count_nonzero(ListDoIslandsInit)>0:
            self.ListDicoInitIndiv=[]
            for iMachine,InitMachine in enumerate(self.ListInitMachine):
                self.DicoDicoInitIndiv[iMachine] = InitMachine.giveDicoInitIndiv(self.ListIslands,
                                                                                 ListDoIsland=ListDoIslandsInit,
                                                                                 ModelImage=ModelImage,
                                                                                 DicoDirty=self.DicoDirty)
                
                # print(iMachine,(self.DicoDicoInitIndiv[iMachine][0]["PolyModel"]))
                
        if self.GD["Misc"]["ConserveMemory"]:
            self._reset_InitMachine()

    def setChannel(self,ch=0):
        self.Dirty=self._MeanDirty[ch]
        self.ModelImage=self._ModelImage[ch]

    def GiveThreshold(self,Max):
        return ((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*Max if self.CycleFactor else 0

    def Deconvolve(self,ch=0):
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

        self.GainMachine.SetRMS(RMS)
        
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
        NLastCyclesDeconvAll=self.GD["SSD2"]["NLastCyclesDeconvAll"]
        print("SSD2 Cycle %i/%i, NLastCyclesDeconvAll=%i)"%(self._CurrentMajorIter,self.MaxMajorIter,NLastCyclesDeconvAll), file=log)
        print("    Dirty image peak flux      = %10.6f Jy [(min, max) = (%.3g, %.3g) Jy]"%(MaxDirty,mm0,mm1), file=log)
        print("      RMS-based threshold      = %10.6f Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor), file=log)
        print("      Sidelobe-based threshold = %10.6f Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor), file=log)
        print("      Peak-based threshold     = %10.6f Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor), file=log)
        print("      Absolute threshold       = %10.6f Jy"%(self.FluxThreshold), file=log)


        DoZeroTh=False
        if self.GD["SSD2"]["NLastCyclesDeconvAll"]==-1:
            DoZeroTh=True
        
        if self._CurrentMajorIter>(self.MaxMajorIter-NLastCyclesDeconvAll):
            DoZeroTh=True

        if DoZeroTh:
            print(ModColor.Str("    ... overwriting these values with zero",col="green"), file=log)
            StopFlux=0.
            
        print("    Stopping flux              = %10.6f Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty), file=log)

        
        MaxModelInit=np.max(np.abs(self.ModelImage))

        
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

        self.SearchIslands(StopFlux)
        #return None,None,None
        self.InitIslands()


        if self.DeconvMode=="GAClean":
            print("Evolving %i generations of %i sourcekin"%(self.GD["GAClean"]["NMaxGen"],self.GD["GAClean"]["NSourceKin"]), file=log)
            ListBigIslands=[]
            ListSmallIslands=[]
            
            ListInitBigIslands=[]
            ListInitSmallIslands=[]
            
            for iIsland,Island in enumerate(self.ListIslands):
                if len(Island)>self.GD["SSDClean"]["ConvFFTSwitch"]:
                    ListBigIslands.append(Island)
                    ListInitBigIslands.append(iIsland)
                else:
                    ListSmallIslands.append(Island)
                    ListInitSmallIslands.append(iIsland)


            if len(ListSmallIslands)>0:
                print("Deconvolve small islands (<=%i pixels) (parallelised over island)"%(self.GD["SSDClean"]["ConvFFTSwitch"]), file=log)
                self.DeconvListIsland(ListSmallIslands,ParallelMode="OverIslands",ListInitIslands=ListInitSmallIslands)
            else:
                print("No small islands", file=log)

            if len(ListBigIslands)>0:
                print("Deconvolve large islands (>%i pixels) (parallelised per island)"%(self.GD["SSDClean"]["ConvFFTSwitch"]), file=log)
                self.DeconvListIsland(ListBigIslands,ParallelMode="PerIsland",ListInitIslands=ListInitBigIslands)
            else:
                print("No large islands", file=log)


        elif self.DeconvMode=="MetroClean":
            if self.GD["MetroClean"]["MetroNChains"]!="NCPU":
                self.NChains=self.GD["MetroClean"]["MetroNChains"]
            else:
                self.NChains=self.NCPU
            print("Evolving %i chains of %i iterations"%(self.NChains,self.GD["MetroClean"]["MetroNIter"]), file=log)
            
            ListBigIslands=[]
            for ThisPixList in self.ListIslands:
                x,y=np.array(ThisPixList,dtype=np.float32).T
                dx,dy=x.max()-x.min(),y.max()-y.min()
                dd=np.max([dx,dy])+1
                if dd>self.GD["SSDClean"]["RestoreMetroSwitch"]:
                    ListBigIslands.append(ThisPixList)

            # ListBigIslands=ListBigIslands[1::]
            # ListBigIslands=[Island for Island in self.ListIslands if len(Island)>=self.GD["SSDClean"]["RestoreMetroSwitch"]]
            print("Deconvolve %i large islands (>=%i pixels) (parallelised per island)"%(len(ListBigIslands),self.GD["SSDClean"]["RestoreMetroSwitch"]), file=log)
            self.SelectedIslandsMask=np.zeros_like(self.DicoDirty["MeanImage"])
            for ThisIsland in ListBigIslands:
                x,y=np.array(ThisIsland).T
                self.SelectedIslandsMask[0,0,x,y]=1
                
            self.DeconvListIsland(ListBigIslands,ParallelMode="PerIsland")
            
        self.DicoDicoInitIndiv.delete()
        return "MaxIter", True, True   # stop deconvolution but do update model




    def DeconvListIsland(self,ListIslands,ParallelMode="OverIslands",ListInitIslands=None):
        # ================== Parallel part
        
        NIslands=len(ListIslands)
        if NIslands==0: return
        if ParallelMode=="OverIslands":
            NCPU=self.NCPU
            NCPU=np.min([NCPU,NIslands])
            Parallel=True
            ParallelPerIsland=False
            StopWhenQueueEmpty=True
        elif ParallelMode=="PerIsland":
            NCPU=1#self.NCPU
            Parallel=True
            ParallelPerIsland=True
            StopWhenQueueEmpty=True
        
        # ######### Debug
        # ParallelPerIsland=False
        # Parallel=False
        # NCPU=1
        # StopWhenQueueEmpty=True
        # # ##################
        

        work_queue = multiprocessing.Queue()


        # shared dict to hold inputs and outputs to workers (each island number is a key)
        deconv_dict  = shared_dict.create("DeconvListIslands")


        NJobs=NIslands
        T=ClassTimeIt.ClassTimeIt("    ")
        T.disable()
        for iIsland, ThisPixList in enumerate(ListIslands):
            island_dict = deconv_dict.addSubdict(iIsland)

            # print "%i/%i"%(iIsland,self.NIslands)
            island_dict["Island"] = np.array(ThisPixList)

            XY=np.array(ThisPixList,dtype=np.float32)
            xm,ym=np.mean(np.float32(XY),axis=0).astype(int)
            T.timeit("xm,ym")
            nchan,npol,_,_=self._Dirty.shape
            JonesNorm=(self.DicoDirty["JonesNorm"][:,:,xm,ym]).reshape((nchan,npol,1,1))
            W=self.DicoDirty["WeightChansImages"]
            JonesNorm=np.sum(JonesNorm*W.reshape((nchan,1,1,1)),axis=0).reshape((1,npol,1,1))
            T.timeit("JonesNorm")
            
            IslandBestIndiv=self.ModelMachine.GiveIndividual(ThisPixList)
            T.timeit("GiveIndividual")
            FacetID=self.PSFServer.giveFacetID2(xm,ym)
            T.timeit("FacetID")

            island_dict["BestIndiv"] = IslandBestIndiv
            
            iIslandInit=ListInitIslands[iIsland]
            
            ListOrder=[iIsland,FacetID,JonesNorm.flat[0],self.RMS**2,island_dict.path,iIslandInit]

            work_queue.put(ListOrder)
            T.timeit("Put")


        # ListArrayIslands=[np.array(ListIslands[iIsland]) for iIsland in range(NIslands)]
        # NpShared.PackListArray(SharedListIsland,ListArrayIslands)
        # T.timeit("Pack0")
        # SharedBestIndiv="%s.ListBestIndiv"%(self.IdSharedMem)
        # NpShared.PackListArray(SharedBestIndiv,ListBestIndiv)
        # T.timeit("Pack1")
        

        workerlist=[]

        # List_Result_queue=[]
        # for ii in range(NCPU):
        #     List_Result_queue.append(multiprocessing.JoinableQueue())


        result_queue=multiprocessing.Queue()
        Title=" Evolve pop."
        if self.DeconvMode=="MetroClean":
            Title=" Running chain"
            
        pBAR= ProgressBar(Title=Title)
        #pBAR.disable()
        pBAR.render(0, NJobs)
        for ii in range(NCPU):
            W=WorkerDeconvIsland(work_queue, 
                                 result_queue,
                                 self.GD,
                                 self._Dirty,
                                 self.DicoVariablePSF["CubeVariablePSF"],
                                 IdSharedMem=self.IdSharedMem,
                                 FreqsInfo=self.PSFServer.DicoMappingDesc,
                                 ParallelPerIsland=ParallelPerIsland,
                                 StopWhenQueueEmpty=StopWhenQueueEmpty,
                                 DeconvMode=self.DeconvMode,
                                 NChains=self.NChains)
            
            workerlist.append(W)
            
            if Parallel: 
                workerlist[ii].start()
            else:
                workerlist[ii].run()
            
        iResult=0
        #print "!!!!!!!!!!!!!!!!!!!!!!!!",iResult,NJobs
        while iResult < NJobs:
            DicoResult=None
            # for result_queue in List_Result_queue:
            #     if result_queue.qsize()!=0:
            #         try:
            #             DicoResult=result_queue.get_nowait()
                        
            #             break
            #         except:
                        
            #             pass
            #         #DicoResult=result_queue.get()
            #print "!!!!!!!!!!!!!!!!!!!!!!!!! Qsize",result_queue.qsize()
            if result_queue.qsize()!=0:
                try:
                    DicoResult=result_queue.get_nowait()
                except:
                    pass
                    #DicoResult=result_queue.get()


            if DicoResult is None:
                time.sleep(0.05)
                continue

            iResult+=1
            NDone=iResult
            intPercent=int(100*  NDone / float(NJobs))
            pBAR.render(NDone,NJobs)

            if DicoResult["Success"]:
                iIsland=DicoResult["iIsland"]
                island_dict = deconv_dict[iIsland]
                island_dict.reload()

                self.ModelMachine.AppendIsland(ListIslands[iIsland], island_dict["Model"].copy())

                if DicoResult["HasError"]:
                    self.ErrorModelMachine.AppendIsland(ThisPixList, ListIslands[iIsland], island_dict["sModel"].copy())

        deconv_dict.delete()

        for ii in range(NCPU):
            try:
                workerlist[ii].shutdown()
                workerlist[ii].terminate()
                workerlist[ii].join()
            except:
                pass
        




    ###################################################################################
    ###################################################################################
    
    def GiveEdges(self,xc0,yc0,N0,xc1,yc1,N1):
        M_xc=xc0
        M_yc=yc0
        NpixMain=N0
        F_xc=xc1
        F_yc=yc1
        NpixFacet=N1
                
        ## X
        M_x0=M_xc-NpixFacet//2
        x0main=np.max([0,M_x0])
        dx0=x0main-M_x0
        x0facet=dx0
                
        M_x1=M_xc+NpixFacet//2
        x1main=np.min([NpixMain-1,M_x1])
        dx1=M_x1-x1main
        x1facet=NpixFacet-dx1
        x1main+=1
        ## Y
        M_y0=M_yc-NpixFacet//2
        y0main=np.max([0,M_y0])
        dy0=y0main-M_y0
        y0facet=dy0
        
        M_y1=M_yc+NpixFacet//2
        y1main=np.min([NpixMain-1,M_y1])
        dy1=M_y1-y1main
        y1facet=NpixFacet-dy1
        y1main+=1

        Aedge=[x0main,x1main,y0main,y1main]
        Bedge=[x0facet,x1facet,y0facet,y1facet]
        return Aedge,Bedge


    def SubStep(self,dx,dy,LocalSM):
        npol,_,_=self.Dirty.shape
        x0,x1,y0,y1=self.DirtyExtent
        xc,yc=dx,dy
        N0=self.Dirty.shape[-1]
        N1=LocalSM.shape[-1]
        Aedge,Bedge=self.GiveEdges(xc,yc,N0,N1//2,N1//2,N1)
        factor=-1.
        nch,npol,nx,ny=LocalSM.shape
        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge
        self._Dirty[:,:,x0d:x1d,y0d:y1d]-=LocalSM[:,:,x0p:x1p,y0p:y1p]
        W=np.float32(self.DicoDirty["WeightChansImages"])
        self._MeanDirty[0,:,x0d:x1d,y0d:y1d]-=np.sum(LocalSM[:,:,x0p:x1p,y0p:y1p]*W.reshape((W.size,1,1,1)),axis=0)

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

#===============================================
#===============================================
#===============================================
#===============================================

class WorkerDeconvIsland(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 GD,
                 Dirty,
                 CubeVariablePSF,
                 IdSharedMem=None,
                 FreqsInfo=None,
                 MultiFreqMode=False,
                 ParallelPerIsland=False,
                 StopWhenQueueEmpty=False,
                 DeconvMode="GAClean",
                 NChains=1):
        multiprocessing.Process.__init__(self)
        self.MultiFreqMode=MultiFreqMode
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.GD=GD
        self.IdSharedMem=IdSharedMem
        self.FreqsInfo=FreqsInfo

        self._Dirty = Dirty
        self.CubeVariablePSF = CubeVariablePSF

        #self.WeightFreqBands=WeightFreqBands
        self.ParallelPerIsland=ParallelPerIsland
        self.StopWhenQueueEmpty=StopWhenQueueEmpty
        self.DeconvMode=DeconvMode
        self.NChains=NChains

    def shutdown(self):
        self.exit.set()

    def CondContinue(self):
        if self.StopWhenQueueEmpty:
            return not(self.work_queue.qsize()==0)
        else:
            return True

 
    def run(self):


        while not self.kill_received and self.CondContinue():

            #gc.enable()
            try:
                iIsland,FacetID,JonesNorm,PixVariance,shdict_path,iIslandInit = self.work_queue.get(True,2)
            except Exception as e:
                #print "Exception worker: %s"%str(e)
                break

            # iIsland=DicoOrder["iIsland"]
            # FacetID=DicoOrder["FacetID"]
            
            # JonesNorm=DicoOrder["JonesNorm"]

            island_dict = shared_dict.attach(shdict_path)

            ThisPixList = island_dict["Island"].tolist()
            IslandBestIndiv = island_dict["BestIndiv"]

            PSF=self.CubeVariablePSF[FacetID]
            NGen=self.GD["GAClean"]["NMaxGen"]
            NIndiv=self.GD["GAClean"]["NSourceKin"]

            ListPixParms=ThisPixList
            ListPixData=ThisPixList
            dx=self.GD["SSDClean"]["NEnlargeData"]
            if dx>0:
                IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
                ListPixData=IncreaseIslandMachine.IncreaseIsland(ListPixData,dx=dx)


            # ################################
            # DicoSave={"Dirty":self._Dirty,
            #           "PSF":PSF,
            #           "FreqsInfo":self.FreqsInfo,
            #           #"DicoMappingDesc":self.PSFServer.DicoMappingDesc,
            #           "ListPixData":ListPixData,
            #           "ListPixParms":ListPixParms,
            #           "IslandBestIndiv":IslandBestIndiv,
            #           "GD":self.GD,
            #           "FacetID":FacetID,
            #           "iIsland":iIsland,"IdSharedMem":self.IdSharedMem}
            # print "saving"
            # MyPickle.Save(DicoSave, "SaveTest")
            # print "saving ok"
            # ################################

            
            if self.DeconvMode=="GAClean":
                CEv=ClassEvolveGA(self._Dirty,
                                  PSF,
                                  self.FreqsInfo,
                                  ListPixParms=ListPixParms,
                                  ListPixData=ListPixData,
                                  iFacet=FacetID,PixVariance=PixVariance,
                                  IslandBestIndiv=IslandBestIndiv,#*np.sqrt(JonesNorm),
                                  GD=self.GD,
                                  iIsland=iIsland,
                                  island_dict=island_dict,
                                  ParallelFitness=self.ParallelPerIsland,
                                  iIslandInit=iIslandInit)
                Model=CEv.main(NGen=NGen,NIndiv=NIndiv,DoPlot=False)
                island_dict["Model"] = np.array(Model)
                del(CEv)
                self.result_queue.put({"Success":True,"iIsland":iIsland,"HasError":False})

            elif self.DeconvMode=="MetroClean":
                CEv=ClassMetropolis(self._Dirty,
                                    PSF,
                                    self.FreqsInfo,
                                    ListPixParms=ListPixParms,
                                    ListPixData=ListPixData,
                                    iFacet=FacetID,PixVariance=PixVariance,
                                    IslandBestIndiv=IslandBestIndiv,#*np.sqrt(JonesNorm),
                                    GD=self.GD,
                                    iIsland=iIsland,
                                    island_dict=island_dict,
                                    ParallelFitness=self.ParallelPerIsland,
                                    NChains=self.NChains)
                Model,sModel=CEv.main(NSteps=self.GD["MetroClean"]["MetroNIter"])
            
                island_dict["Model"] = np.array(Model)
                island_dict["sModel"] = np.array(sModel)

                del(CEv)
                self.result_queue.put({"Success":True,"iIsland":iIsland,"HasError":True})


            # # if island lies inside image
            # try:
            #     nch=self.FreqsInfo["MeanJonesBand"][FacetID].size
            #     #WeightMeanJonesBand=self.FreqsInfo["MeanJonesBand"][FacetID].reshape((nch,1,1,1))
            #     #WeightMueller=WeightMeanJonesBand.ravel()
            #     #WeightMuellerSignal=np.sqrt(WeightMueller*self.FreqsInfo["WeightChansImages"].ravel())

            #     CEv=ClassEvolveGA(self._Dirty,
            #                       PSF,
            #                       self.FreqsInfo,
            #                       ListPixParms=ListPixParms,
            #                       ListPixData=ListPixData,
            #                       iFacet=FacetID,PixVariance=PixVariance,
            #                       IslandBestIndiv=IslandBestIndiv,#*np.sqrt(JonesNorm),
            #                       GD=self.GD)
            #     #,
            #      #                 WeightFreqBands=WeightMuellerSignal)
            #     Model=CEv.main(NGen=NGen,NIndiv=NIndiv,DoPlot=False)
            
            #     Model=np.array(Model).copy()#/np.sqrt(JonesNorm)
            #     #Model*=CEv.ArrayMethodsMachine.Gain
                
            #     del(CEv)
                
            #     NpShared.ToShared("%s.FitIsland_%5.5i"%(self.IdSharedMem,iIsland),Model)
                
            #     #print "Current process: %s [%s left]"%(str(multiprocessing.current_process()),str(self.work_queue.qsize()))
                
            #     self.result_queue.put({"Success":True,"iIsland":iIsland})
            # except Exception,e:
            #     print "Exception on island %i: %s"%(iIsland,str(e))

            #     self.result_queue.put({"Success":False})

        #print "WORKER DONE"
