
import numpy as np
import pylab
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.Array import NpShared
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
import ClassMultiScaleMachine
from pyrap.images import image
from ClassPSFServer import ClassPSFServer
import ClassModelMachineGA
from DDFacet.Other.progressbar import ProgressBar
import ClassGainMachine
from SkyModel.PSourceExtract import ClassIslands
from SkyModel.PSourceExtract import ClassIncreaseIsland
from GA.ClassEvolveGA import ClassEvolveGA 
from DDFacet.Other import MyPickle
import multiprocessing
import time

MyLogger.setSilent("ClassArrayMethodGA")
MyLogger.setSilent("ClassIsland")

class ClassImageDeconvMachine():
    def __init__(self,Gain=0.3,
                 MaxMinorIter=100,NCPU=6,
                 CycleFactor=2.5,FluxThreshold=None,RMSFactor=3,PeakFactor=0,
                 GD=None,SearchMaxAbs=1,CleanMaskImage=None,IdSharedMem="",
                 **kw    # absorb any unknown keywords arguments into this
                 ):
        #self.im=CasaImage
        self.SearchMaxAbs=SearchMaxAbs
        self.ModelImage=None
        self.MaxMinorIter=MaxMinorIter
        self.NCPU=NCPU
        self.Chi2Thr=10000
        self.MaskArray=None
        self.GD=GD
        self.IdSharedMem=IdSharedMem
        self.SubPSF=None
        self.MultiFreqMode=(self.GD["MultiFreqs"]["NFreqBands"]>1)
        self.FluxThreshold = FluxThreshold 
        self.CycleFactor = CycleFactor
        self.RMSFactor = RMSFactor
        self.PeakFactor = PeakFactor
        self.GainMachine=ClassGainMachine.ClassGainMachine(GainMin=Gain)
        self.ModelMachine=ClassModelMachineGA.ClassModelMachine(self.GD,GainMachine=self.GainMachine)
        # reset overall iteration counter
        self._niter = 0
        
        if CleanMaskImage!=None:
            print>>log, "Reading mask image: %s"%CleanMaskImage
            MaskArray=image(CleanMaskImage).getdata()
            nch,npol,_,_=MaskArray.shape
            self._MaskArray=np.zeros(MaskArray.shape,np.bool8)
            for ch in range(nch):
                for pol in range(npol):
                    self._MaskArray[ch,pol,:,:]=np.bool8(1-MaskArray[ch,pol].T[::-1].copy())[:,:]
            self.MaskArray=self._MaskArray[0]
            self.IslandArray=np.zeros_like(self._MaskArray)
            self.IslandHasBeenDone=np.zeros_like(self._MaskArray)

    def GiveModelImage(self,*args): return self.ModelMachine.GiveModelImage(*args)

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe
        

    def SetPSF(self,DicoVariablePSF):
        self.PSFServer=ClassPSFServer(self.GD)
        DicoVariablePSF["CubeVariablePSF"]=NpShared.ToShared("%s.CubeVariablePSF"%self.IdSharedMem,DicoVariablePSF["CubeVariablePSF"])
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF)
        #self.DicoPSF=DicoPSF
        self.DicoVariablePSF=DicoVariablePSF
        #self.NChannels=self.DicoDirty["NChannels"]
        self.ModelMachine.setRefFreq(self.PSFServer.RefFreq,self.PSFServer.AllFreqs)
        


    def InitMSMF(self):
        pass

        


    def SetDirty(self,DicoDirty):


        DicoDirty["ImagData"]=NpShared.ToShared("%s.Dirty.ImagData"%self.IdSharedMem,DicoDirty["ImagData"])
        DicoDirty["MeanImage"]=NpShared.ToShared("%s.Dirty.MeanImage"%self.IdSharedMem,DicoDirty["MeanImage"])
        self.DicoDirty=DicoDirty
        self._Dirty=self.DicoDirty["ImagData"]
        self._MeanDirty=self.DicoDirty["MeanImage"]
        NPSF=self.PSFServer.NPSF
        _,_,NDirty,_=self._Dirty.shape

        off=(NPSF-NDirty)/2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)

        if self.ModelImage==None:
            self._ModelImage=np.zeros_like(self._Dirty)
        self.ModelMachine.setModelShape(self._Dirty.shape)
        if self.MaskArray==None:
            self._MaskArray=np.zeros(self._Dirty.shape,dtype=np.bool8)
            self.IslandArray=np.zeros_like(self._MaskArray)
            self.IslandHasBeenDone=np.zeros_like(self._MaskArray)


    def SearchIslands(self,Threshold):
        print>>log,"Searching Islands"
        Dirty=self.DicoDirty["MeanImage"]
        self.IslandArray[0,0]=(Dirty[0,0]>Threshold)|(self.IslandArray[0,0])
        MaskImage=(self.IslandArray[0,0])&(np.logical_not(self._MaskArray[0,0]))
        #MaskImage=(np.logical_not(self._MaskArray[0,0]))
        MaskImage=(np.logical_not(self._MaskArray[0,0]))
        Islands=ClassIslands.ClassIslands(Dirty[0,0],MaskImage=MaskImage,
                                          MinPerIsland=0,DeltaXYMin=0)
        Islands.FindAllIslands()
        

        ListIslands=Islands.LIslands
        self.ListIslands=[]

        for iIsland in range(len(ListIslands)):
            x,y=np.array(ListIslands[iIsland]).T
            PixVals=Dirty[0,0,x,y]
            DoThisOne=False
            if np.max(np.abs(PixVals))>Threshold:
                DoThisOne=True
                self.IslandHasBeenDone[0,0,x,y]=1

            if ((DoThisOne)|self.IslandHasBeenDone[0,0,x[0],y[0]]):
                self.ListIslands.append(ListIslands[iIsland])


        self.NIslands=len(self.ListIslands)
        print>>log,"Selected %i islands [out of %i] with peak flux > %.3g Jy"%(self.NIslands,len(ListIslands),Threshold)

        dx=self.GD["GAClean"]["NEnlargePars"]
        if dx>0:
            IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
            for iIsland in range(self.NIslands):
                self.ListIslands[iIsland]=IncreaseIslandMachine.IncreaseIsland(self.ListIslands[iIsland],dx=dx)

        Sz=np.array([len(self.ListIslands[iIsland]) for iIsland in range(self.NIslands)])
        ind=np.argsort(Sz)[::-1]

        ListIslandsOut=[self.ListIslands[ind[i]] for i in ind]
        self.ListIslands=ListIslandsOut

                

    def setChannel(self,ch=0):
        self.Dirty=self._MeanDirty[ch]
        self.ModelImage=self._ModelImage[ch]
        self.MaskArray=self._MaskArray[ch]


    def GiveThreshold(self,Max):
        return ((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*Max if self.CycleFactor else 0

    def Clean(self,*args,**kwargs):
        #return self.CleanSerial(*args,**kwargs)
        return self.CleanParallel(*args,**kwargs)

    def CleanSerial(self,ch=0):
        """
        Runs minor cycle over image channel 'ch'.
        initMinor is number of minor iteration (keeps continuous count through major iterations)
        Nminor is max number of minor iteration

        Returns tuple of: return_code,continue,updated
        where return_code is a status string;
        continue is True if another cycle should be executed;
        update is True if model has been updated (note that update=False implies continue=False)
        """
        if self._niter >= self.MaxMinorIter:
            return "MaxIter", False, False

        self.setChannel(ch)

        _,npix,_=self.Dirty.shape
        xc=(npix)/2

        npol,_,_=self.Dirty.shape

        m0,m1=self.Dirty[0].min(),self.Dirty[0].max()
        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.imshow(self.Dirty[0],interpolation="nearest",vmin=m0,vmax=m1)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        DoAbs=int(self.GD["ImagerDeconv"]["SearchMaxAbs"])
        print>>log, "  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter,self.MaxMinorIter,DoAbs)

        NPixStats=1000
        RandomInd=np.int64(np.random.rand(NPixStats)*npix**2)
        RMS=np.std(np.real(self.Dirty.ravel()[RandomInd]))
        self.RMS=RMS

        self.GainMachine.SetRMS(RMS)
        
        Fluxlimit_RMS = self.RMSFactor*RMS

        x,y,MaxDirty=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)
        #MaxDirty=np.max(np.abs(self.Dirty))
        #Fluxlimit_SideLobe=MaxDirty*(1.-self.SideLobeLevel)
        #Fluxlimit_Sidelobe=self.CycleFactor*MaxDirty*(self.SideLobeLevel)
        Fluxlimit_Peak = MaxDirty*self.PeakFactor
        Fluxlimit_Sidelobe = self.GiveThreshold(MaxDirty)

        mm0,mm1=self.Dirty.min(),self.Dirty.max()

        # work out uper threshold
        StopFlux = max(Fluxlimit_Peak, Fluxlimit_RMS, Fluxlimit_Sidelobe, Fluxlimit_Peak, self.FluxThreshold)

        print>>log, "    Dirty image peak flux      = %10.6f Jy [(min, max) = (%.3g, %.3g) Jy]"%(MaxDirty,mm0,mm1)
        print>>log, "      RMS-based threshold      = %10.6f Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor)
        print>>log, "      Sidelobe-based threshold = %10.6f Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor)
        print>>log, "      Peak-based threshold     = %10.6f Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor)
        print>>log, "      Absolute threshold       = %10.6f Jy"%(self.FluxThreshold)
        print>>log, "    Stopping flux              = %10.6f Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty)


        MaxModelInit=np.max(np.abs(self.ModelImage))

        
        # Fact=4
        # self.BookKeepShape=(npix/Fact,npix/Fact)
        # BookKeep=np.zeros(self.BookKeepShape,np.float32)
        # NPixBook,_=self.BookKeepShape
        # FactorBook=float(NPixBook)/npix
        
        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)

        if ThisFlux < StopFlux:
            print>>log, ModColor.Str("    Initial maximum peak %g Jy below threshold, we're done here" % (ThisFlux),col="green" )
            return "FluxThreshold", False, False

        self.SearchIslands(StopFlux)

        for iIsland in range(self.NIslands):
            ThisPixList=self.ListIslands[iIsland]
            print>>log,"  Fitting island #%4.4i with %i pixels"%(iIsland,len(ThisPixList))

            XY=np.array(ThisPixList,dtype=np.float32)
            xm,ym=np.int64(np.mean(np.float32(XY),axis=0))

            FacetID=self.PSFServer.giveFacetID2(xm,ym)
            PSF=self.DicoVariablePSF["CubeVariablePSF"][FacetID]
            # self.DicoVariablePSF["CubeMeanVariablePSF"][FacetID]
            
            # FreqsInfo={"freqs":self.DicoVariablePSF["freqs"],
            #            "WeightChansImages":self.DicoVariablePSF["WeightChansImages"]}

            FreqsInfo=self.PSFServer.DicoMappingDesc


            nchan,npol,_,_=self._Dirty.shape
            JonesNorm=(self.DicoDirty["NormData"][:,:,xm,ym]).reshape((nchan,npol,1,1))
            W=self.DicoDirty["WeightChansImages"]
            JonesNorm=np.sum(JonesNorm*W.reshape((nchan,1,1,1)),axis=0).reshape((1,npol,1,1))
            


            IslandBestIndiv=self.ModelMachine.GiveIndividual(ThisPixList)

            ################################
            DicoSave={"Dirty":self._Dirty,
                      "PSF":PSF,
                      "FreqsInfo":FreqsInfo,
                      #"DicoMappingDesc":self.PSFServer.DicoMappingDesc,
                      "ListPixData":ThisPixList,
                      "ListPixParms":ThisPixList,
                      "IslandBestIndiv":IslandBestIndiv,
                      "GD":self.GD}
            
            print "saving"
            MyPickle.Save(DicoSave, "SaveTest")
            print "saving ok"
            ################################

            
            CEv=ClassEvolveGA(self._Dirty,PSF,FreqsInfo,ListPixParms=ThisPixList,
                              ListPixData=ThisPixList,IslandBestIndiv=IslandBestIndiv,
                              GD=self.GD)
            Model=CEv.main(NGen=100,DoPlot=True)#False)
            

            #self.ModelMachine.setParamMachine(CEv.ArrayMethodsMachine.PM)
            #Threshold=self.GiveThreshold(np.max(np.abs(Model)))
            #self.ModelMachine.setThreshold(Threshold)
            self.ModelMachine.AppendIsland(ThisPixList,Model)
            


        return "MaxIter", True, True   # stop deconvolution but do update model





    def CleanParallel(self,ch=0):
        if self._niter >= self.MaxMinorIter:
            return "MaxIter", False, False

        self.setChannel(ch)

        _,npix,_=self.Dirty.shape
        xc=(npix)/2

        npol,_,_=self.Dirty.shape

        m0,m1=self.Dirty[0].min(),self.Dirty[0].max()

        DoAbs=int(self.GD["ImagerDeconv"]["SearchMaxAbs"])
        print>>log, "  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter,self.MaxMinorIter,DoAbs)

        NPixStats=1000
        RandomInd=np.int64(np.random.rand(NPixStats)*npix**2)
        RMS=np.std(np.real(self.Dirty.ravel()[RandomInd]))
        self.RMS=RMS

        self.GainMachine.SetRMS(RMS)
        
        Fluxlimit_RMS = self.RMSFactor*RMS

        x,y,MaxDirty=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)
        #MaxDirty=np.max(np.abs(self.Dirty))
        #Fluxlimit_SideLobe=MaxDirty*(1.-self.SideLobeLevel)
        #Fluxlimit_Sidelobe=self.CycleFactor*MaxDirty*(self.SideLobeLevel)
        Fluxlimit_Peak = MaxDirty*self.PeakFactor
        Fluxlimit_Sidelobe = self.GiveThreshold(MaxDirty)

        mm0,mm1=self.Dirty.min(),self.Dirty.max()

        # work out uper threshold
        StopFlux = max(Fluxlimit_Peak, Fluxlimit_RMS, Fluxlimit_Sidelobe, Fluxlimit_Peak, self.FluxThreshold)

        print>>log, "    Dirty image peak flux      = %10.6f Jy [(min, max) = (%.3g, %.3g) Jy]"%(MaxDirty,mm0,mm1)
        print>>log, "      RMS-based threshold      = %10.6f Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor)
        print>>log, "      Sidelobe-based threshold = %10.6f Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor)
        print>>log, "      Peak-based threshold     = %10.6f Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor)
        print>>log, "      Absolute threshold       = %10.6f Jy"%(self.FluxThreshold)
        print>>log, "    Stopping flux              = %10.6f Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty)


        MaxModelInit=np.max(np.abs(self.ModelImage))

        
        # Fact=4
        # self.BookKeepShape=(npix/Fact,npix/Fact)
        # BookKeep=np.zeros(self.BookKeepShape,np.float32)
        # NPixBook,_=self.BookKeepShape
        # FactorBook=float(NPixBook)/npix
        
        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)

        if ThisFlux < StopFlux:
            print>>log, ModColor.Str("    Initial maximum peak %g Jy below threshold, we're done here" % (ThisFlux),col="green" )
            return "FluxThreshold", False, False

        self.SearchIslands(StopFlux)
        


        # ================== Parallel part
        NCPU=self.NCPU
        work_queue = multiprocessing.Queue()


        ListBestIndiv=[]

        NJobs=self.NIslands
        T=ClassTimeIt.ClassTimeIt("    ")
        T.disable()
        for iIsland in range(self.NIslands):
            # print "%i/%i"%(iIsland,self.NIslands)
            ThisPixList=self.ListIslands[iIsland]
            XY=np.array(ThisPixList,dtype=np.float32)
            xm,ym=np.mean(np.float32(XY),axis=0)
            T.timeit("xm,ym")
            nchan,npol,_,_=self._Dirty.shape
            JonesNorm=(self.DicoDirty["NormData"][:,:,xm,ym]).reshape((nchan,npol,1,1))
            W=self.DicoDirty["WeightChansImages"]
            JonesNorm=np.sum(JonesNorm*W.reshape((nchan,1,1,1)),axis=0).reshape((1,npol,1,1))
            T.timeit("JonesNorm")

            IslandBestIndiv=self.ModelMachine.GiveIndividual(ThisPixList)
            T.timeit("GiveIndividual")
            ListBestIndiv.append(IslandBestIndiv)
            FacetID=self.PSFServer.giveFacetID2(xm,ym)
            T.timeit("FacetID")

            DicoOrder={"iIsland":iIsland,
                       "FacetID":FacetID,
                       "JonesNorm":JonesNorm}
            
            ListOrder=[iIsland,FacetID,JonesNorm.flat[0]]


            work_queue.put(ListOrder)
            T.timeit("Put")
            
        SharedListIsland="%s.ListIslands"%(self.IdSharedMem)
        ListArrayIslands=[np.array(self.ListIslands[iIsland]) for iIsland in range(self.NIslands)]
        NpShared.PackListArray(SharedListIsland,ListArrayIslands)
        T.timeit("Pack0")
        SharedBestIndiv="%s.ListBestIndiv"%(self.IdSharedMem)
        NpShared.PackListArray(SharedBestIndiv,ListBestIndiv)
        T.timeit("Pack1")
        

        workerlist=[]

        # List_Result_queue=[]
        # for ii in range(NCPU):
        #     List_Result_queue.append(multiprocessing.JoinableQueue())


        result_queue=multiprocessing.Queue()


        for ii in range(NCPU):
            W=WorkerDeconvIsland(work_queue, 
                                 result_queue,
                                 # List_Result_queue[ii],
                                 self.GD,
                                 IdSharedMem=self.IdSharedMem,
                                 FreqsInfo=self.PSFServer.DicoMappingDesc)
            workerlist.append(W)
            workerlist[ii].start()


        print>>log, "  Evolving %i generations of %i sourcekin"%(self.GD["GAClean"]["NMaxGen"],self.GD["GAClean"]["NSourceKin"])
        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title=" Evolve pop.", HeaderSize=10,TitleSize=13)
        #pBAR.disable()
        pBAR.render(0, '%4i/%i' % (0,NJobs))

        iResult=0
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
            if result_queue.qsize()!=0:
                try:
                    DicoResult=result_queue.get_nowait()
                except:
                    pass
                    #DicoResult=result_queue.get()


            if DicoResult==None:
                time.sleep(0.05)
                continue

            if DicoResult["Success"]:
                iResult+=1
                NDone=iResult
                intPercent=int(100*  NDone / float(NJobs))
                pBAR.render(intPercent, '%4i/%i' % (NDone,NJobs))


            iIsland=DicoResult["iIsland"]
            ThisPixList=self.ListIslands[iIsland]
            SharedIslandName="%s.FitIsland_%5.5i"%(self.IdSharedMem,iIsland)
            Model=NpShared.GiveArray(SharedIslandName)

            self.ModelMachine.AppendIsland(ThisPixList,Model)


            NpShared.DelArray(SharedIslandName)



        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()

        


        return "MaxIter", True, True   # stop deconvolution but do update model


    ###################################################################################
    ###################################################################################
    
    def GiveEdges(self,(xc0,yc0),N0,(xc1,yc1),N1):
        M_xc=xc0
        M_yc=yc0
        NpixMain=N0
        F_xc=xc1
        F_yc=yc1
        NpixFacet=N1
                
        ## X
        M_x0=M_xc-NpixFacet/2
        x0main=np.max([0,M_x0])
        dx0=x0main-M_x0
        x0facet=dx0
                
        M_x1=M_xc+NpixFacet/2
        x1main=np.min([NpixMain-1,M_x1])
        dx1=M_x1-x1main
        x1facet=NpixFacet-dx1
        x1main+=1
        ## Y
        M_y0=M_yc-NpixFacet/2
        y0main=np.max([0,M_y0])
        dy0=y0main-M_y0
        y0facet=dy0
        
        M_y1=M_yc+NpixFacet/2
        y1main=np.min([NpixMain-1,M_y1])
        dy1=M_y1-y1main
        y1facet=NpixFacet-dy1
        y1main+=1

        Aedge=[x0main,x1main,y0main,y1main]
        Bedge=[x0facet,x1facet,y0facet,y1facet]
        return Aedge,Bedge


    def SubStep(self,(dx,dy),LocalSM):
        npol,_,_=self.Dirty.shape
        x0,x1,y0,y1=self.DirtyExtent
        xc,yc=dx,dy
        N0=self.Dirty.shape[-1]
        N1=LocalSM.shape[-1]
        Aedge,Bedge=self.GiveEdges((xc,yc),N0,(N1/2,N1/2),N1)
        factor=-1.
        nch,npol,nx,ny=LocalSM.shape
        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge
        self._Dirty[:,:,x0d:x1d,y0d:y1d]-=LocalSM[:,:,x0p:x1p,y0p:y1p]
        W=np.float32(self.DicoDirty["WeightChansImages"])
        self._MeanDirty[0,:,x0d:x1d,y0d:y1d]-=np.sum(LocalSM[:,:,x0p:x1p,y0p:y1p]*W.reshape((W.size,1,1,1)),axis=0)


#===============================================
#===============================================
#===============================================
#===============================================

class WorkerDeconvIsland(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 GD,
                 IdSharedMem=None,
                 FreqsInfo=None,
                 MultiFreqMode=False):
        multiprocessing.Process.__init__(self)
        self.MultiFreqMode=MultiFreqMode
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.GD=GD
        self.IdSharedMem=IdSharedMem
        self.FreqsInfo=FreqsInfo
        self.CubeVariablePSF=NpShared.GiveArray("%s.CubeVariablePSF"%self.IdSharedMem)
        self._Dirty=NpShared.GiveArray("%s.Dirty.ImagData"%self.IdSharedMem)

    def shutdown(self):
        self.exit.set()

 
    def run(self):
        while not self.kill_received:
            #gc.enable()
            try:
                iIsland,FacetID,JonesNorm = self.work_queue.get()
            except:
                break


            # iIsland=DicoOrder["iIsland"]
            # FacetID=DicoOrder["FacetID"]
            
            # JonesNorm=DicoOrder["JonesNorm"]

            SharedListIsland="%s.ListIslands"%(self.IdSharedMem)
            ThisPixList=NpShared.UnPackListArray(SharedListIsland)[iIsland].tolist()

            SharedBestIndiv="%s.ListBestIndiv"%(self.IdSharedMem)
            IslandBestIndiv=NpShared.UnPackListArray(SharedBestIndiv)[iIsland]
            
            PSF=self.CubeVariablePSF[FacetID]
            NGen=self.GD["GAClean"]["NMaxGen"]
            NIndiv=self.GD["GAClean"]["NSourceKin"]

            ListPixParms=ThisPixList
            ListPixData=ThisPixList
            dx=self.GD["GAClean"]["NEnlargeData"]
            if dx>0:
                IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
                ListPixData=IncreaseIslandMachine.IncreaseIsland(ListPixData,dx=dx)


            CEv=ClassEvolveGA(self._Dirty,
                              PSF,
                              self.FreqsInfo,
                              ListPixParms=ListPixParms,
                              ListPixData=ListPixData,
                              IslandBestIndiv=IslandBestIndiv,#*np.sqrt(JonesNorm),
                              GD=self.GD)
            Model=CEv.main(NGen=NGen,NIndiv=NIndiv,DoPlot=False)
            
            Model=np.array(Model).copy()#/np.sqrt(JonesNorm)
            #Model*=CEv.ArrayMethodsMachine.Gain

            del(CEv)

            NpShared.ToShared("%s.FitIsland_%5.5i"%(self.IdSharedMem,iIsland),Model)

            #print "Current process: %s [%s left]"%(str(multiprocessing.current_process()),str(self.work_queue.qsize()))

            self.result_queue.put({"Success":True,"iIsland":iIsland})
                
