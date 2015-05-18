import numpy as np
import ClassMS
from pyrap.tables import table
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassVisServer")
# import MyPickle
from DDFacet.Array import NpShared
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
from DDFacet.Array import ModLinAlg
MyLogger.setSilent(["NpShared"])
from DDFacet.Imager import ClassWeighting
from DDFacet.Other import reformat
import ClassSmearMapping
import os
import ClassJones

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

        self.ReadOnce=False
        self.ReadOnce_AlreadyRead=False
        if TChunkSize<=TVisSizeMin*60:
            self.ReadOnce=True

        if type(MSName)==list:
            self.MultiMSMode=True
            self.ListMSName=MSName
            self.ReadOnce=False
        else:
            self.MultiMSMode=False
            self.ListMSName=[MSName]
        self.ListMSName=[MSName for MSName in self.ListMSName if MSName!=""]

        self.IdSharedMem=IdSharedMem
        self.Robust=Robust
        PrefixShared="%sSharedVis"%self.IdSharedMem
        self.AddNoiseJy=AddNoiseJy
        self.TMemChunkSize=TChunkSize
        self.TVisSizeMin=TVisSizeMin

        self.Weighting=Weighting
        self.NCPU=NCPU
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


    def Init(self,PointingID=0):
        #MSName=self.MDC.giveMS(PointingID).MSName
        if self.MultiMSMode:
            print>>log, "Multiple MS mode"

        self.ListMS=[]
        self.ListGlobalFreqs=[]
        for MSName in self.ListMSName:
            MS=ClassMS.ClassMS(MSName,Col=self.ColName,DoReadData=False) 
            self.ListMS.append(MS)
            self.ListGlobalFreqs+=MS.ChanFreq.flatten().tolist()
            if self.GD["Stores"]["DeleteDDFProducts"]:
                ThisMSName=reformat.reformat(os.path.abspath(MS.MSName),LastSlash=False)

                MapName="%s/Flagging.npy"%ThisMSName
                os.system("rm %s"%MapName)

                MapName="%s/Mapping.CompGrid.npy"%ThisMSName
                os.system("rm %s"%MapName)

                MapName="%s/Mapping.CompDeGrid.npy"%ThisMSName
                os.system("rm %s"%MapName)

                JonesName="%s/JonesNorm.npz"%ThisMSName
                os.system("rm %s"%JonesName)

        self.nMS=len(self.ListMS)
        self.GlobalFreqs=np.array(self.ListGlobalFreqs)
        self.NFreqBands=np.min([self.GD["MultiFreqs"]["NFreqBands"],self.nMS])
        self.CurrentMS=self.ListMS[0]
        self.iCurrentMS=0

        self.MultiFreqMode=False
        NFreqBands=self.NFreqBands
        if self.NFreqBands>1: 
            self.MultiFreqMode=True
            print>>log, ModColor.Str("MultiFrequency Mode: ON")
        else:
            self.GD["MultiFreqs"]["NFreqBands"] = 1
            self.GD["MultiFreqs"]["Alpha"] = [0.,0.,1.]
            print>>log, ModColor.Str("MultiFrequency Mode: OFF")
            
        FreqBands=np.linspace(self.GlobalFreqs.min(),self.GlobalFreqs.max(),NFreqBands+1)
        self.FreqBandsMean=(FreqBands[0:-1]+FreqBands[1::])/2.
        self.FreqBandsMin=FreqBands[0:-1].copy()
        self.FreqBandsMax=FreqBands[1::].copy()
        self.FreqBandsInfos={}
        for iBand in range(self.NFreqBands):
            self.FreqBandsInfos[iBand]=[]


        self.ListFreqs=[]
        for MS in self.ListMS:
            FreqBand = np.where((self.FreqBandsMin <= np.mean(MS.ChanFreq))&(self.FreqBandsMax > np.mean(MS.ChanFreq)))[0][0]
            self.FreqBandsInfos[FreqBand]+=MS.ChanFreq.tolist()
            self.ListFreqs+=MS.ChanFreq.tolist()
            print MS
            
        self.RefFreq=np.mean(self.ListFreqs)

        MS=self.ListMS[0]
        TimesInt=np.arange(0,MS.DTh,self.TMemChunkSize).tolist()
        if not(MS.DTh in TimesInt): TimesInt.append(MS.DTh)
        self.TimesInt=TimesInt
        self.NTChunk=len(self.TimesInt)-1
        self.MS=MS
        self.ReInitChunkCount()


        

        #TimesVisMin=np.arange(0,MS.DTh*60.,self.TVisSizeMin).tolist()
        #if not(MS.DTh*60. in TimesVisMin): TimesVisMin.append(MS.DTh*60.)
        #self.TimesVisMin=np.array(TimesVisMin)



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

    def VisChunkToShared(self):

        # t0_bef,t1_bef=self.CurrentVisTimes_SinceStart_Sec
        # t0_sec,t1_sec=t1_bef,t1_bef+60.*self.dTimesVisMin
        # its_t0,its_t1=self.MS.CurrentChunkTimeRange_SinceT0_sec
        # t1_sec=np.min([its_t1,t1_sec])
        # self.iCurrentVisTime+=1
        # self.CurrentVisTimes_SinceStart_Minutes = t0_sec/60.,t1_sec/60.
        # self.CurrentVisTimes_SinceStart_Sec     = t0_sec,t1_sec
        # #print>>log,("(t0_sec,t1_sec,t1_bef,t1_bef+60.*self.dTimesVisMin)",t0_sec,t1_sec,t1_bef,t1_bef+60.*self.dTimesVisMin)
        # #print>>log,("(its_t0,its_t1)",its_t0,its_t1)
        # #print>>log,("self.CurrentVisTimes_SinceStart_Minutes",self.CurrentVisTimes_SinceStart_Minutes)
        # if (t0_sec>=its_t1):
        #     return "EndChunk"
        # MS=self.MS
        # t0_MS=self.MS.F_tstart
        # t0_sec+=t0_MS
        # t1_sec+=t0_MS
        # self.CurrentVisTimes_MS_Sec=t0_sec,t1_sec
        # # time selection
        # ind=np.where((self.ThisDataChunk["times"]>=t0_sec)&(self.ThisDataChunk["times"]<t1_sec))[0]
        # if ind.shape[0]==0:
        #     return "EndChunk"


        D=self.ThisDataChunk
        DATA={}
        for key in D.keys():
            if type(D[key])!=np.ndarray: continue
            if not(key in ['times', 'A1', 'A0', 'flags', 'uvw', 'data',"Weights"]):             
                DATA[key]=D[key]
            else:
                DATA[key]=D[key][:]

        if "DicoBeam" in D.keys():
            DATA["DicoBeam"]=D["DicoBeam"]

        #print>>log, "!!!!!!!!!"
        #DATA["flags"].fill(0)


        print>>log, "Putting data in shared memory"
        DATA=NpShared.DicoToShared("%sDicoData"%self.IdSharedMem,DATA)

        return DATA

    def ReInitChunkCount(self):
        self.CurrentMemTimeChunk=0
        self.CurrentVisTimes_SinceStart_Sec=0.,0.
        self.iCurrentVisTime=0
        self.iCurrentMS=0
        self.CurrentFreqBand=0
        for MS in self.ListMS:
            MS.ReinitChunkIter(self.TMemChunkSize)
        self.CurrentMS=self.ListMS[0]

        print>>log, (ModColor.Str("NextMS %s"%(self.CurrentMS.MSName),col="green") + (" --> freq. band %i"%self.CurrentFreqBand))

    def setNextMS(self):
        if (self.iCurrentMS+1)==self.nMS:
            print>>log, ModColor.Str("Reached end of MSList")
            return "EndListMS"
        else:
            self.iCurrentMS+=1
            self.CurrentMS=self.ListMS[self.iCurrentMS]
            self.CurrentFreqBand=0
            if self.MultiFreqMode:
                self.CurrentFreqBand = np.where((self.FreqBandsMin <= np.mean(self.CurrentMS.ChanFreq))&(self.FreqBandsMax > np.mean(self.CurrentMS.ChanFreq)))[0][0]
            print>>log, (ModColor.Str("NextMS %s"%(self.CurrentMS.MSName),col="green") + (" --> freq. band %i"%(self.CurrentFreqBand)))
            return "OK"
        

    def LoadNextVisChunk(self):

        while True:
            MS=self.CurrentMS
            repLoadChunk=MS.GiveNextChunk()
            if repLoadChunk=="EndMS":
                repNextMS=self.setNextMS()
                if repNextMS=="EndListMS":
                    print>>log, ModColor.Str("Reached end of Observation")
                    self.ReInitChunkCount()
                    return "EndOfObservation"
                elif repNextMS=="OK":
                    continue
            DATA=repLoadChunk
            break
        
        
        self.TimeMemChunkRange_sec=DATA["times"][0],DATA["times"][-1]

        times=DATA["times"]
        data=DATA["data"]
        A0=DATA["A0"]
        A1=DATA["A1"]
        uvw=DATA["uvw"]

        flags=DATA["flag"]
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

        ThisMSName=reformat.reformat(os.path.abspath(self.CurrentMS.MSName),LastSlash=False)
        TimeMapName="%s/Flagging.npy"%ThisMSName
        try:
            DATA["flags"]=np.load(TimeMapName)
        except:
            self.UpdateFlag(DATA)


        


        self.UpdateCompression(DATA)

        JonesMachine=ClassJones.ClassJones(self.GD,self.FacetMachine,self.CurrentMS,IdSharedMem=self.IdSharedMem)
        JonesMachine.InitDDESols(DATA)

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

        if freqs.size>1:
            freqs=np.float64(freqs)
        else:
            freqs=np.array([freqs[0]],dtype=np.float64)
        
        DicoDataOut={"times":DATA["times"],
                     "freqs":freqs,
                     "A0":DATA["A0"],
                     "A1":DATA["A1"],
                     "uvw":DATA["uvw"],
                     "flags":DATA["flags"],
                     "nbl":nbl,
                     "na":MS.na,
                     "data":DATA["data"],
                     "ROW0":MS.ROW0,
                     "ROW1":MS.ROW1,
                     "infos":np.array([MS.na]),
                     "Weights":self.VisWeights[MS.ROW0:MS.ROW1]
                     }
        




        DATA=DicoDataOut

        self.ThisDataChunk = DATA
        return "LoadOK"




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


        for A in self.FlagAntNumber:
            ind=np.where((A0==A)|(A1==A))[0]
            flags[ind,:,:]=True

        ind=np.where(A0==A1)[0]
        flags[ind,:,:]=True

        ind=np.any(flags,axis=2)
        flags[ind]=True
        
        # flags.fill(0)
        # ind=np.where(A0!=A1)[0]
        # flags[ind,:,:]=True

        ind=np.where(np.isnan(data))
        flags[ind]=1

        ThisMSName=reformat.reformat(os.path.abspath(self.CurrentMS.MSName),LastSlash=False)
        TimeMapName="%s/Flagging.npy"%ThisMSName
        np.save(TimeMapName,flags)

        DATA["flags"]=flags




        # if dt0<dt1:
        #     JonesBeam=np.zeros((Tm.size,),dtype=[("t0",np.float32),("t1",np.float32),("tm",np.float32),("Jones",(NDir,self.MS.na,self.MS.NSPWChan,2,2),np.complex64)])


    def setFacetMachine(self,FacetMachine):
        self.FacetMachine=FacetMachine
        self.FullImShape=self.FacetMachine.OutImShape
        self.PaddedFacetShape=self.FacetMachine.PaddedGridShape
        self.FacetShape=self.FacetMachine.FacetShape
        self.CellSizeRad=self.FacetMachine.CellSizeRad

    def UpdateCompression(self,DATA):
        ThisMSName=reformat.reformat(os.path.abspath(self.CurrentMS.MSName),LastSlash=False)
        if self.GD["Compression"]["CompGridMode"]:
            MapName="%s/Mapping.CompGrid.npy"%ThisMSName
            try:
                FinalMapping=np.load(MapName)
            except:
                if self.GD["Compression"]["CompGridFOV"]=="Facet":
                    _,_,nx,ny=self.FacetShape
                elif self.GD["Compression"]["CompGridFOV"]=="Full":
                    _,_,nx,ny=self.FullImShape
                FOV=self.CellSizeRad*nx*(np.sqrt(2.)/2.)*180./np.pi
                SmearMapMachine=ClassSmearMapping.ClassSmearMapping(self.MS,radiusDeg=FOV,Decorr=(1.-self.GD["Compression"]["CompGridDecorr"]),IdSharedMem=self.IdSharedMem,NCPU=self.NCPU)
                #SmearMapMachine.BuildSmearMapping(DATA)
                FinalMapping,fact=SmearMapMachine.BuildSmearMappingParallel(DATA)
                np.save(MapName,FinalMapping)
                print>>log, ModColor.Str("  Effective compression [Grid]  :   %.2f%%"%fact,col="green")

            Map=NpShared.ToShared("%sMappingSmearing.Grid"%(self.IdSharedMem),FinalMapping)

        if self.GD["Compression"]["CompDeGridMode"]:
            MapName="%s/Mapping.CompDeGrid.npy"%ThisMSName
            try:
                FinalMapping=np.load(MapName)
            except:
                if self.GD["Compression"]["CompDeGridFOV"]=="Facet":
                    _,_,nx,ny=self.FacetShape
                elif self.GD["Compression"]["CompDeGridFOV"]=="Full":
                    _,_,nx,ny=self.FullImShape
                FOV=self.CellSizeRad*nx*(np.sqrt(2.)/2.)*180./np.pi
                SmearMapMachine=ClassSmearMapping.ClassSmearMapping(self.MS,radiusDeg=FOV,Decorr=(1.-self.GD["Compression"]["CompDeGridDecorr"]),IdSharedMem=self.IdSharedMem,NCPU=self.NCPU)
                #SmearMapMachine.BuildSmearMapping(DATA)
                FinalMapping,fact=SmearMapMachine.BuildSmearMappingParallel(DATA)
                np.save(MapName,FinalMapping)
                print>>log, ModColor.Str("  Effective compression [DeGrid]:   %.2f%%"%fact,col="green")

            Map=NpShared.ToShared("%sMappingSmearing.DeGrid"%(self.IdSharedMem),FinalMapping)




    def GiveAllUVW(self):
        t=table(self.MS.MSName,ack=False)
        uvw=t.getcol("UVW")
        WEIGHT=t.getcol("IMAGING_WEIGHT")
        flags=t.getcol("FLAG")
        t.close()
        return uvw,WEIGHT,flags


