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

class ClassVisServer():
    def __init__(self,MSName,
                 ColName="DATA",
                 TChunkSize=1,
                 TVisSizeMin=1,
                 DicoSelectOptions={},
                 LofarBeam=None,
                 AddNoiseJy=None,IdSharedMem="",
                 Robust=2,Weighting="Briggs"):
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

    def CalcWeigths(self,ImShape,CellSizeRad):
        if self.VisWeights!=None: return

        WeightMachine=ClassWeighting.ClassWeighting(ImShape,CellSizeRad)
        uvw,WEIGHT=self.GiveAllUVW()
        VisWeights=WEIGHT[:,0]#np.ones((uvw.shape[0],),dtype=np.float32)
        #VisWeights=np.ones((uvw.shape[0],),dtype=np.float32)
        Robust=self.Robust
        self.VisWeights=WeightMachine.CalcWeights(uvw,VisWeights,Robust=Robust,
                                                  Weighting=self.Weighting)
        
        #self.VisWeights.fill(1)

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
            if not(key in ['times', 'A1', 'A0', 'flags', 'uvw', 'data']):             
                DATA[key]=D[key]
            else:
                DATA[key]=D[key][ind]


        ROW0=self.ThisDataChunk["ROW0"]
        ROW1=self.ThisDataChunk["ROW1"]
        W=self.VisWeights[ROW0:ROW1]
        DATA["Weights"]=W
        #############################
        ### data selection
        #############################
        flags=DATA["flags"]
        uvw=DATA["uvw"]
        data=DATA["data"].copy()
        A0=DATA["A0"]
        A1=DATA["A1"]
        times=DATA["times"]
        Weights=DATA["Weights"]
        

        for Field in self.DicoSelectOptions.keys():
            if self.DicoSelectOptions[Field]==None: break
            if Field=="UVRangeKm":
                d0,d1=self.DicoSelectOptions[Field]
                print>>log, "Flagging uv data outside uv distance of [%5.1f~%5.1f] km"%(d0,d1)
                d0*=1e3
                d1*=1e3
                u,v,w=uvw.T
                duv=np.sqrt(u**2+v**2)
                #ind=np.where((duv<d0)|(duv>d1))[0]
                ind=np.where((duv>d0)&(duv<d1))[0]
                
                flags=flags[ind]
                data=data[ind]
                A0=A0[ind]
                A1=A1[ind]
                uvw=uvw[ind]
                times=times[ind]
                Weights=Weights[ind]

        if "FlagAnts" in self.DicoSelectOptions.keys():
            FlagAnts=self.DicoSelectOptions["FlagAnts"]
            for Name in FlagAnts:
                for iAnt in range(MS.na):
                    if Name in MS.StationNames[iAnt]:
                        print>>log, "Flagging antenna #%i [%s]"%(iAnt,MS.StationNames[iAnt])
                        self.FlagAntNumber.append(iAnt)

            # if Field=="Antenna":
            #     if self.DicoSelectOptions[Field]==None: break
            #     AntList=self.DicoSelectOptions[Field].split(",")
            #     for self.DicoSelectOptions[Field]


        for A in self.FlagAntNumber:
            ind=np.where((A0!=A)&(A1!=A))[0]
            flags=flags[ind]
            data=data[ind]
            A0=A0[ind]
            A1=A1[ind]
            uvw=uvw[ind]
            times=times[ind]
            Weights=Weights[ind]
        
        

        ind=np.where(A0!=A1)[0]
        flags=flags[ind,:,:]
        data=data[ind,:,:]
        A0=A0[ind]
        A1=A1[ind]
        uvw=uvw[ind,:]
        times=times[ind]
        Weights=Weights[ind]

        # ###########
        # ind=np.where((A0==32)&(A1==33))[0]
        # flags=flags[ind,:,:]
        # data=data[ind,:,:]
        # A0=A0[ind]
        # A1=A1[ind]
        # uvw=uvw[ind,:]
        # times=times[ind]
        # Weights=Weights[ind]
        # ############

        DATA["flags"]=flags
        DATA["uvw"]=uvw
        DATA["data"]=data
        DATA["A0"]=A0
        DATA["A1"]=A1
        DATA["times"]=times
        DATA["Weights"]=Weights

        NVisChan=data.shape[1]
        DATA["Weights"]=DATA["Weights"].reshape((uvw.shape[0],1))*np.ones((1,NVisChan))

        # if self.VisInSharedMem:
        #     self.ClearSharedMemory()
        #     DATA=self.PutInShared(DATA)
        #     DATA["A0A1"]=(DATA["A0"],DATA["A1"])

        if "DicoBeam" in D.keys():
            DATA["DicoBeam"]=D["DicoBeam"]

        DATA=NpShared.DicoToShared("%sDicoData"%self.IdSharedMem,DATA)


        DicoJonesMatrices=NpShared.SharedToDico("%skillMSSolutionFile"%self.IdSharedMem)
        if DicoJonesMatrices!=None:
            JonesMatrices=DicoJonesMatrices["Jones"]
            t0=DicoJonesMatrices["t0"]
            t0=t0.reshape((1,t0.size))
            t1=DicoJonesMatrices["t1"]
            t1=t1.reshape((1,t1.size))
            tMS=times.reshape(times.size,1)
            cond0=(tMS>t0)
            cond1=(tMS<=t1)
            cond=(cond0&cond1)
            MapJones=np.int32(np.argmax(cond,axis=1))
            NpShared.ToShared("%sMapJones"%self.IdSharedMem,MapJones)




        #print
        #print self.MS.ROW0,self.MS.ROW1
        #t0=np.min(DATA["times"])-self.MS.F_tstart
        #t1=np.max(DATA["times"])-self.MS.F_tstart
        #self.TEST_TLIST+=sorted(list(set(DATA["times"].tolist())))
        

        return DATA




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

        #flags.fill(0)

        # f=(np.random.rand(*flags.shape)>0.5)
        # flags[f]=1
        # data[flags]=1e6

        # iAFlag=12
        # ind=np.where((A0==iAFlag)|(A1==iAFlag))[0]
        # flags[ind,:,:]=1


        MS=self.MS
        self.ThresholdFlag=0.9
        self.FlagAntNumber=[]
        for A in range(MS.na):
            ind=np.where((MS.A0==A)|(MS.A1==A))[0]
            fA=MS.flag_all[ind].ravel()
            nf=np.count_nonzero(fA)
            Frac=nf/float(fA.size)
            if Frac>self.ThresholdFlag:
                print>>log, "I found that antenna %i has ~%4.1f%s of flagged data (more than %4.1f%s)"%\
                    (A,Frac*100,"%",self.ThresholdFlag*100,"%")
                self.FlagAntNumber.append(A)
                
            

        #############################
        #############################

        ind=np.where(np.isnan(data))
        flags[ind]=1
        
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
                     "infos":np.array([MS.na])
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
        WEIGHT=t.getcol("WEIGHT")
        t.close()
        return uvw,WEIGHT


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

