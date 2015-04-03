import numpy as np
import ClassMS
from pyrap.tables import table
from Other import MyLogger
log=MyLogger.getLogger("ClassVisServer")
# import MyPickle
from Array import NpShared
from Other import ClassTimeIt
from Other import ModColor
from Array import ModLinAlg
MyLogger.setSilent(["NpShared"])
import ClassWeighting
from Other import reformat
import ClassSmearMapping

def test():
    MSName="/media/tasse/data/killMS_Pack/killMS2/Test/0000.MS"
    VS=ClassVisServer(MSName,TVisSizeMin=1e8,Weighting="Natural")
    VS.CalcWeigths((1,1,1000,1000),20.*np.pi/180)
    VS.LoadNextVisChunk()

class ClassVisServer():
    def __init__(self,MSName,GD=None,
                 ColName="DATA",
                 TChunkSize=1,
                 TVisSizeMin=1,
                 DicoSelectOptions={},
                 LofarBeam=None,
                 AddNoiseJy=None,IdSharedMem="",
                 Robust=2,Weighting="Briggs",NCPU=6):
        self.IdSharedMem=IdSharedMem
        self.Robust=Robust
        PrefixShared="%sSharedVis"%self.IdSharedMem
        self.AddNoiseJy=AddNoiseJy
        self.ReInitChunkCount()
        self.TMemChunkSize=TChunkSize
        self.TVisSizeMin=TVisSizeMin
        self.ReadOnce=False
        self.ReadOnce_AlreadyRead=False
        if TChunkSize<=TVisSizeMin*60:
            self.ReadOnce=True
        self.Weighting=Weighting
        self.NCPU=NCPU
        self.MSName=MSName
        self.VisWeights=None
        self.CountPickle=0
        self.ColName=ColName
        self.DicoSelectOptions=DicoSelectOptions
        self.SharedNames=[]
        self.PrefixShared=PrefixShared
        self.VisInSharedMem = (PrefixShared!=None)
        self.LofarBeam=LofarBeam
        self.ApplyBeam=False
        self.GD=GD
        self.Init()
        
        self.dTimesVisMin=self.TVisSizeMin
        self.CurrentVisTimes_SinceStart_Sec=0.,0.
        self.iCurrentVisTime=0

        # self.LoadNextVisChunk()

        #self.TEST_TLIST=[]

    def SetBeam(self,LofarBeam):
        self.BeamMode,self.DtBeamMin,self.BeamRAs,self.BeamDECs = LofarBeam
        useArrayFactor=("A" in self.BeamMode)
        useElementBeam=("E" in self.BeamMode)
        self.MS.LoadSR(useElementBeam=useElementBeam,useArrayFactor=useArrayFactor)
        self.ApplyBeam=True

    def Init(self,PointingID=0):
        #MSName=self.MDC.giveMS(PointingID).MSName
        MS=ClassMS.ClassMS(self.MSName,Col=self.ColName,DoReadData=False)
        print MS
        TimesInt=np.arange(0,MS.DTh,self.TMemChunkSize).tolist()
        if not(MS.DTh in TimesInt): TimesInt.append(MS.DTh)
        self.TimesInt=TimesInt
        self.NTChunk=len(self.TimesInt)-1
        self.MS=MS

        #TimesVisMin=np.arange(0,MS.DTh*60.,self.TVisSizeMin).tolist()
        #if not(MS.DTh*60. in TimesVisMin): TimesVisMin.append(MS.DTh*60.)
        #self.TimesVisMin=np.array(TimesVisMin)


    def ReInitChunkCount(self):
        self.CurrentMemTimeChunk=0
        self.CurrentVisTimes_SinceStart_Sec=0.,0.
        self.iCurrentVisTime=0

    def SetImagingPars(self,OutImShape,CellSizeRad):
        self.OutImShape=OutImShape
        self.CellSizeRad=CellSizeRad

    def CalcWeigths(self):
        if self.VisWeights!=None: return
        ImShape=self.PaddedFacetShape
        CellSizeRad=self.CellSizeRad
        WeightMachine=ClassWeighting.ClassWeighting(ImShape,CellSizeRad)
        uvw,WEIGHT,flags=self.GiveAllUVW()
        #uvw=DATA["uvw"]
        #WEIGHT=DATA["Weights"]
        VisWeights=WEIGHT#[:,0]#np.ones((uvw.shape[0],),dtype=np.float32)
        if np.max(VisWeights)==0.:
            print>>log,"All imaging weights are 0, setting them to ones"
            VisWeights.fill(1)
        #VisWeights=np.ones((uvw.shape[0],),dtype=np.float32)
        Robust=self.Robust

        #self.VisWeights=np.ones((uvw.shape[0],self.MS.ChanFreq.size),dtype=np.float64)

        self.VisWeights=WeightMachine.CalcWeights(uvw,VisWeights,flags,self.MS.ChanFreq,
                                                  Robust=Robust,
                                                  Weighting=self.Weighting)

    def GiveNextVis(self):

        #print>>log, "GiveNextVis"

        t0_bef,t1_bef=self.CurrentVisTimes_SinceStart_Sec
        t0_sec,t1_sec=t1_bef,t1_bef+60.*self.dTimesVisMin

        its_t0,its_t1=self.MS.CurrentChunkTimeRange_SinceT0_sec
        t1_sec=np.min([its_t1,t1_sec])

        self.iCurrentVisTime+=1
        self.CurrentVisTimes_SinceStart_Minutes = t0_sec/60.,t1_sec/60.
        self.CurrentVisTimes_SinceStart_Sec     = t0_sec,t1_sec

        #print>>log,("(t0_sec,t1_sec,t1_bef,t1_bef+60.*self.dTimesVisMin)",t0_sec,t1_sec,t1_bef,t1_bef+60.*self.dTimesVisMin)
        #print>>log,("(its_t0,its_t1)",its_t0,its_t1)
        #print>>log,("self.CurrentVisTimes_SinceStart_Minutes",self.CurrentVisTimes_SinceStart_Minutes)

        if (t0_sec>=its_t1):
            return "EndChunk"

        MS=self.MS
        

        
        t0_MS=self.MS.F_tstart
        t0_sec+=t0_MS
        t1_sec+=t0_MS
        self.CurrentVisTimes_MS_Sec=t0_sec,t1_sec

        D=self.ThisDataChunk
        # time selection
        ind=np.where((self.ThisDataChunk["times"]>=t0_sec)&(self.ThisDataChunk["times"]<t1_sec))[0]
        if ind.shape[0]==0:
            return "EndChunk"
        DATA={}
        for key in D.keys():
            if type(D[key])!=np.ndarray: continue
            if not(key in ['times', 'A1', 'A0', 'flags', 'uvw', 'data',"Weights"]):             
                DATA[key]=D[key]
            else:
                DATA[key]=D[key][ind]


        #DATA["Weights"].fill(1)



        # ROW0=self.ThisDataChunk["ROW0"]
        # ROW1=self.ThisDataChunk["ROW1"]
        # W=self.VisWeights[ROW0:ROW1]
        # DATA["Weights"]=W
        # DATA["flags"]=flags
        # DATA["uvw"]=uvw
        # DATA["data"]=data
        # DATA["A0"]=A0
        # DATA["A1"]=A1
        # DATA["times"]=times
        # DATA["Weights"]=Weights


        #NVisChan=data.shape[1]
        #DATA["Weights"]=DATA["Weights"].reshape((uvw.shape[0],1))*np.ones((1,NVisChan))

        # if self.VisInSharedMem:
        #     self.ClearSharedMemory()
        #     DATA=self.PutInShared(DATA)
        #     DATA["A0A1"]=(DATA["A0"],DATA["A1"])

        if "DicoBeam" in D.keys():
            DATA["DicoBeam"]=D["DicoBeam"]


        print>>log, "Putting data in shared memory"
        DATA=NpShared.DicoToShared("%sDicoData"%self.IdSharedMem,DATA)







        #print
        #print self.MS.ROW0,self.MS.ROW1
        #t0=np.min(DATA["times"])-self.MS.F_tstart
        #t1=np.max(DATA["times"])-self.MS.F_tstart
        #self.TEST_TLIST+=sorted(list(set(DATA["times"].tolist())))
        

        return DATA


    def UpdateFlag(self,DATA):
        print>>log, "Updating flags ..."

        flags=DATA["flags"]
        uvw=DATA["uvw"]
        data=DATA["data"]
        A0=DATA["A0"]
        A1=DATA["A1"]
        times=DATA["times"]
        #Weights=DATA["Weights"]

        MS=self.MS
        self.ThresholdFlag=0.9
        self.FlagAntNumber=[]

        for A in range(MS.na):
            ind=np.where((A0==A)|(A1==A))[0]
            fA=flags[ind].ravel()
            nf=np.count_nonzero(fA)
            Frac=nf/float(fA.size)
            if Frac>self.ThresholdFlag:
                print>>log, "  Flagging antenna %i has ~%4.1f%s of flagged data (more than %4.1f%s)"%\
                    (A,Frac*100,"%",self.ThresholdFlag*100,"%")
                self.FlagAntNumber.append(A)
        

        if self.DicoSelectOptions["UVRangeKm"]!=None:
            d0,d1=self.DicoSelectOptions["UVRangeKm"]
            print>>log, "  Flagging uv data outside uv distance of [%5.1f~%5.1f] km"%(d0,d1)
            d0*=1e3
            d1*=1e3
            u,v,w=uvw.T
            duv=np.sqrt(u**2+v**2)
            ind=np.where(((duv>d0)&(duv<d1))!=True)[0]
            flags[ind,:,:]=True

        
        if self.DicoSelectOptions["TimeRange"]!=None:
            t0=times[0]
            tt=(times-t0)/3600.
            st0,st1=self.DicoSelectOptions["TimeRange"]
            print>>log, "  Selecting uv data in time range [%.4f~%5.4f] hours"%(st0,st1)
            indt=np.where((tt>=st0)&(tt<st1))[0]
            flags[ind,:,:]=True

        if self.DicoSelectOptions["FlagAnts"]!=None:
            FlagAnts=self.DicoSelectOptions["FlagAnts"]
            if not((FlagAnts==None)|(FlagAnts=="")|(FlagAnts==[])): 
                if type(FlagAnts)==str: FlagAnts=[FlagAnts] 
                for Name in FlagAnts:
                    for iAnt in range(MS.na):
                        if Name in MS.StationNames[iAnt]:
                            print>>log, "  Flagging antenna #%2.2i[%s]"%(iAnt,MS.StationNames[iAnt])
                            self.FlagAntNumber.append(iAnt)

        if self.DicoSelectOptions["DistMaxToCore"]!=None:
            DMax=self.DicoSelectOptions["DistMaxToCore"]*1e3
            X,Y,Z=MS.StationPos.T
            Xm,Ym,Zm=np.median(MS.StationPos,axis=0).flatten().tolist()
            Dist=np.sqrt((X-Xm)**2+(Y-Ym)**2+(Z-Zm)**2)
            ind=np.where(Dist>DMax)[0]
            for iAnt in ind.tolist():
                print>>log,"  Flagging antenna #%2.2i[%s] (distance to core: %.1f km)"%(iAnt,MS.StationNames[iAnt],Dist[iAnt]/1e3)
                self.FlagAntNumber.append(iAnt)

            # if Field=="Antenna":
            #     if self.DicoSelectOptions[Field]==None: break
            #     AntList=self.DicoSelectOptions[Field].split(",")
            #     for self.DicoSelectOptions[Field]


        for A in self.FlagAntNumber:
            ind=np.where((A0==A)|(A1==A))[0]
            flags[ind,:,:]=True

        ind=np.where(A0==A1)[0]
        flags[ind,:,:]=True
        # flags.fill(0)
        # ind=np.where(A0!=A1)[0]
        # flags[ind,:,:]=True

        ind=np.where(np.isnan(data))
        flags[ind]=1

        DATA["flags"]=flags

    def InitDDESols(self,DATA):
        GD=self.GD
        SolsFile=GD["DDESolutions"]["DDSols"]
        self.ApplyCal=False
        if (SolsFile!=""):#&(False):
            print>>log, "Loading solution file: %s"%SolsFile
            self.ApplyCal=True
            DicoSolsFile=np.load(SolsFile)
            DicoSols={}
            DicoSols["t0"]=DicoSolsFile["Sols"]["t0"]
            DicoSols["t1"]=DicoSolsFile["Sols"]["t1"]
            nt,na,nd,_,_=DicoSolsFile["Sols"]["G"].shape
            G=np.swapaxes(DicoSolsFile["Sols"]["G"],1,2).reshape((nt,nd,na,1,2,2))
            DicoSols["Jones"]=G

            if GD["DDESolutions"]["GlobalNorm"]=="MeanAbs":
                print>>log, "  Normalising by the mean of the amplitude"
                gmean_abs=np.mean(np.abs(G[:,:,:,:,0,0]),axis=0)
                gmean_abs=gmean_abs.reshape((1,nd,na,1))
                DicoSols["Jones"][:,:,:,:,0,0]/=gmean_abs
                DicoSols["Jones"][:,:,:,:,1,1]/=gmean_abs


            # if not("A" in self.GD["DDESolutions"]["ApplyMode"]):
            #     print>>log, "  Amplitude normalisation"
            #     gabs=np.abs(G)
            #     gabs[gabs==0]=1.
            #     G/=gabs


            NpShared.DicoToShared("%skillMSSolutionFile"%self.IdSharedMem,DicoSols)
            #D=NpShared.SharedToDico("killMSSolutionFile")
            #ClusterCat=DicoSolsFile["ClusterCat"]
            ClusterCat=DicoSolsFile["SkyModel"]
            ClusterCat=ClusterCat.view(np.recarray)
            DicoClusterDirs={}
            DicoClusterDirs["l"]=ClusterCat.l
            DicoClusterDirs["m"]=ClusterCat.m
            DicoClusterDirs["I"]=ClusterCat.SumI
            DicoClusterDirs["Cluster"]=ClusterCat.Cluster
            
            _D=NpShared.DicoToShared("%sDicoClusterDirs"%self.IdSharedMem,DicoClusterDirs)

            print>>log, "  Built time-mapping"
            DicoJonesMatrices=DicoSols

            # times=DATA["times"]
            # ind=np.array([],np.int32)
            # nt,na,nd,_,_,_=DicoJonesMatrices["Jones"].shape
            # for it in range(nt):
            #     t0=DicoJonesMatrices["t0"][it]
            #     t1=DicoJonesMatrices["t1"][it]
            #     indMStime=np.where((times>=t0)&(times<t1))[0]
            #     indMStime=np.ones((indMStime.size,),np.int32)*it
            #     ind=np.concatenate((ind,indMStime))

            times=DATA["times"]
            ind=np.zeros((times.size,),np.int32)
            nt,na,nd,_,_,_=DicoJonesMatrices["Jones"].shape
            ii=0
            for it in range(nt):
                t0=DicoJonesMatrices["t0"][it]
                t1=DicoJonesMatrices["t1"][it]
                indMStime=np.where((times>=t0)&(times<t1))[0]
                indMStime=np.ones((indMStime.size,),np.int32)*it
                ind=ind[ii:indMStime.size]=indMStime[:]
                ii+=indMStime.size

            NpShared.ToShared("%sMapJones"%self.IdSharedMem,ind)
            print>>log, "Done"



    def setFOV(self,FullImShape,PaddedFacetShape,FacetShape,CellSizeRad):
        self.FullImShape=FullImShape
        self.PaddedFacetShape=PaddedFacetShape
        self.FacetShape=FacetShape
        self.CellSizeRad=CellSizeRad
        
        
    def LoadNextVisChunk(self):
        if self.CurrentMemTimeChunk==self.NTChunk:
            #print>>log, ModColor.Str("Reached end of observations")
            self.ReInitChunkCount()
            return "EndOfObservation"
        MS=self.MS
        iT0,iT1=self.CurrentMemTimeChunk,self.CurrentMemTimeChunk+1
        self.CurrentMemTimeChunk+=1

        if (self.ReadOnce)&(self.ReadOnce_AlreadyRead):
            return "LoadOK"
        self.ReadOnce_AlreadyRead=True

        print>>log, "Reading next data chunk in [%5.2f, %5.2f] hours"%(self.TimesInt[iT0],self.TimesInt[iT1])
        MS.ReadData(t0=self.TimesInt[iT0],t1=self.TimesInt[iT1])




        
        #print>>log, "    Rows= [%i, %i]"%(MS.ROW0,MS.ROW1)
        #print float(MS.ROW0)/MS.nbl,float(MS.ROW1)/MS.nbl

        ###############################
        MS=self.MS

        self.TimeMemChunkRange_sec=MS.times_all[0],MS.times_all[-1]

        times=MS.times_all
        data=MS.data
        A0=MS.A0
        A1=MS.A1
        uvw=MS.uvw
        flags=MS.flag_all
        freqs=MS.ChanFreq.flatten()
        nbl=MS.nbl

        DATA={}
        DATA["flags"]=flags
        DATA["data"]=data
        DATA["uvw"]=uvw
        DATA["A0"]=A0
        DATA["A1"]=A1
        DATA["times"]=times
        DATA["Weights"]=self.VisWeights[MS.ROW0:MS.ROW1]

        self.UpdateFlag(DATA)
        if self.GD["Compression"]["CompGridMode"]:
            if self.GD["Compression"]["CompGridFOV"]:
                _,_,nx,ny=self.FacetShape
            elif self.GD["Compression"]["CompGridFOV"]=="Full":
                _,_,nx,ny=self.FullImShape
            FOV=self.CellSizeRad*nx*(np.sqrt(2.)/2.)*180./np.pi
            SmearMapMachine=ClassSmearMapping.ClassSmearMapping(self.MS,radiusDeg=FOV,Decorr=(1.-self.GD["Compression"]["CompGridDecorr"]),IdSharedMem=self.IdSharedMem,NCPU=self.NCPU)
            #SmearMapMachine.BuildSmearMapping(DATA)
            FinalMapping,fact=SmearMapMachine.BuildSmearMappingParallel(DATA)
            Map=NpShared.ToShared("%sMappingSmearing.Grid"%(self.IdSharedMem),FinalMapping)
            print>>log, ModColor.Str("  Effective compression [Grid]  :   %.2f%%"%fact,col="green")

        if self.GD["Compression"]["CompDeGridMode"]:
            if self.GD["Compression"]["CompDeGridFOV"]=="Facet":
                _,_,nx,ny=self.FacetShape
            elif self.GD["Compression"]["CompDeGridFOV"]=="Full":
                _,_,nx,ny=self.FullImShape
            FOV=self.CellSizeRad*nx*(np.sqrt(2.)/2.)*180./np.pi
            SmearMapMachine=ClassSmearMapping.ClassSmearMapping(self.MS,radiusDeg=FOV,Decorr=(1.-self.GD["Compression"]["CompDeGridDecorr"]),IdSharedMem=self.IdSharedMem,NCPU=self.NCPU)
            #SmearMapMachine.BuildSmearMapping(DATA)
            FinalMapping,fact=SmearMapMachine.BuildSmearMappingParallel(DATA)
            Map=NpShared.ToShared("%sMappingSmearing.DeGrid"%(self.IdSharedMem),FinalMapping)
            print>>log, ModColor.Str("  Effective compression [DeGrid]:   %.2f%%"%fact,col="green")

        self.InitDDESols(DATA)

        #############################
        #############################

        
        # ## debug
        # ind=np.where((A0==0)&(A1==1))[0]
        # flags=flags[ind]
        # data=data[ind]
        # A0=A0[ind]
        # A1=A1[ind]
        # uvw=uvw[ind]
        # times=times[ind]
        # ##





        if self.AddNoiseJy!=None:
            data+=(self.AddNoiseJy/np.sqrt(2.))*(np.random.randn(*data.shape)+1j*np.random.randn(*data.shape))
        
        DicoDataOut={"times":times,
                     "freqs":freqs,
                     #"A0A1":(A0[ind],A1[ind]),
                     #"A0A1":(A0,A1),
                     "A0":A0,
                     "A1":A1,
                     "uvw":uvw,
                     "flags":flags,
                     "nbl":nbl,
                     "na":MS.na,
                     "data":data,
                     "ROW0":MS.ROW0,
                     "ROW1":MS.ROW1,
                     "infos":np.array([MS.na]),
                     "Weights":self.VisWeights[MS.ROW0:MS.ROW1]
                     }
        


        if self.ApplyBeam:
            print>>log, "Update LOFAR beam .... "
            DtBeamSec=self.DtBeamMin*60
            tmin,tmax=np.min(times),np.max(times)
            TimesBeam=np.arange(np.min(times),np.max(times),DtBeamSec).tolist()
            if not(tmax in TimesBeam): TimesBeam.append(tmax)
            TimesBeam=np.array(TimesBeam)
            T0s=TimesBeam[:-1]
            T1s=TimesBeam[1:]
            Tm=(T0s+T1s)/2.
            RA,DEC=self.BeamRAs,self.BeamDECs
            NDir=RA.size
            Beam=np.zeros((Tm.size,NDir,self.MS.na,self.MS.NSPWChan,2,2),np.complex64)
            for itime in range(Tm.size):
                ThisTime=Tm[itime]
                Beam[itime]=self.MS.GiveBeam(ThisTime,self.BeamRAs,self.BeamDECs)
            BeamH=ModLinAlg.BatchH(Beam)

            DicoBeam={}
            DicoBeam["t0"]=T0s
            DicoBeam["t1"]=T1s
            DicoBeam["tm"]=Tm
            DicoBeam["Beam"]=Beam
            DicoBeam["BeamH"]=BeamH
            DicoDataOut["DicoBeam"]=DicoBeam
            
            print>>log, "       .... done Update LOFAR beam "

        #MyPickle.Save(DicoDataOut,"Pickle_All_%2.2i"%self.CountPickle)
        #self.CountPickle+=1

        DATA=DicoDataOut

        #A0,A1=DATA["A0A1"]
        #DATA["A0"]=A0
        #DATA["A1"]=A1

        ##############################################
        
        
        # DATA["data"].fill(1)

        # DATA.keys()
        # ['uvw', 'MapBLSel', 'Weights', 'nbl', 'data', 'ROW_01', 'itimes', 'freqs', 'nf', 'times', 'A1', 'A0', 'flags', 'nt', 'A0A1']

        self.ThisDataChunk = DATA
        return "LoadOK"


    def GiveAllUVW(self):
        t=table(self.MS.MSName,ack=False)
        uvw=t.getcol("UVW")
        WEIGHT=t.getcol("IMAGING_WEIGHT")
        flags=t.getcol("FLAG")
        t.close()
        return uvw,WEIGHT,flags


    # def ClearSharedMemory(self):
    #     #NpShared.DelAll(self.PrefixShared)
    #     #NpShared.DelAll("%sDicoData"%self.IdSharedMem)
    #     #NpShared.DelAll("%sKernelMat"%self.IdSharedMem)

    #     # for Name in self.SharedNames:
    #     #     NpShared.DelArray(Name)
    #     self.SharedNames=[]

    # def PutInShared(self,Dico):
    #     DicoOut={}
    #     for key in Dico.keys():
    #         if type(Dico[key])!=np.ndarray: continue
    #         #print "%s.%s"%(self.PrefixShared,key)
    #         Shared=NpShared.ToShared("%s.%s"%(self.PrefixShared,key),Dico[key])
    #         DicoOut[key]=Shared
    #         self.SharedNames.append("%s.%s"%(self.PrefixShared,key))
            
    #     return DicoOut

