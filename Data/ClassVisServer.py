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
import ClassBeamMean

def test():
    MSName="/media/tasse/data/killMS_Pack/killMS2/Test/0000.MS"
    VS=ClassVisServer(MSName,TVisSizeMin=1e8,Weighting="Natural")
    VS.CalcWeights((1,1,1000,1000),20.*np.pi/180)
    VS.LoadNextVisChunk()

class ClassVisServer():
    def __init__(self,MSName,GD=None,
                 ColName="DATA",
                 TChunkSize=1,
                 TVisSizeMin=1,
                 DicoSelectOptions={},
                 LofarBeam=None,
                 AddNoiseJy=None,IdSharedMem="",
                 Robust=2,Weighting="Briggs",Super=1,NCPU=6):

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
        self.FacetMachine=None
        self.IdSharedMem=IdSharedMem
        self.Robust=Robust
        PrefixShared="%sSharedVis"%self.IdSharedMem
        self.AddNoiseJy=AddNoiseJy
        self.TMemChunkSize=TChunkSize
        self.TVisSizeMin=TVisSizeMin

        self.Weighting=Weighting
        self.Super=Super
        self.NCPU=NCPU
        self.VisWeights=None
        self.CountPickle=0
        self.ColName=ColName
        self.Field = DicoSelectOptions.get("Field",0)
        self.DDID = DicoSelectOptions.get("DDID",0)
        self.TaQL = "FIELD_ID==%d&&DATA_DESC_ID==%d" % (self.Field, self.DDID)
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
        NChanMax=0
        ChanStart = self.DicoSelectOptions.get("ChanStart",0)
        ChanEnd   = self.DicoSelectOptions.get("ChanEnd",-1)
        ChanStep  = self.DicoSelectOptions.get("ChanStep",1)
        if (ChanStart,ChanEnd,ChanStep) == (0,-1,1):
            chanslice = None
        else:
            chanslice = slice(ChanStart, ChanEnd if ChanEnd != -1 else None, ChanStep) 
        for MSName in self.ListMSName:
            MS=ClassMS.ClassMS(MSName,Col=self.ColName,DoReadData=False,AverageTimeFreq=(1,3),
                               Field=self.Field,DDID=self.DDID,
                               ChanSlice=chanslice,
                               ToRADEC=self.GD["ImagerGlobal"]["PhaseCenterRADEC"])
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

                JonesName="%s/JonesNorm_Beam.npz"%ThisMSName
                os.system("rm %s"%JonesName)

                JonesName="%s/JonesNorm_killMS.npz"%ThisMSName
                os.system("rm %s"%JonesName)

        self.nMS=len(self.ListMS)
        self.GlobalFreqs=np.array(self.ListGlobalFreqs)
        self.NFreqBands=np.min([self.GD["MultiFreqs"]["NFreqBands"],len(self.GlobalFreqs)])#self.nMS])
        self.CurrentMS=self.ListMS[0]
        self.iCurrentMS=0

        self.MultiFreqMode=False
        NFreqBands=self.NFreqBands
        if self.NFreqBands>1: 
            print>>log, ModColor.Str("MultiFrequency Mode: ON")
            if not("Alpha" in self.GD["GAClean"]["GASolvePars"]):
                self.GD["GAClean"]["GASolvePars"].append("Alpha")
            
            self.MultiFreqMode=True
        else:
            self.GD["MultiFreqs"]["NFreqBands"] = 1
            self.GD["MultiFreqs"]["Alpha"] = [0.,0.,1.]
            if "Alpha" in self.GD["GAClean"]["GASolvePars"]:
                self.GD["GAClean"]["GASolvePars"].remove("Alpha")
            print>>log, ModColor.Str("MultiFrequency Mode: OFF")
            

        FreqBands=np.linspace(self.GlobalFreqs.min(),self.GlobalFreqs.max(),NFreqBands+1)
        self.FreqBandsMean=(FreqBands[0:-1]+FreqBands[1::])/2.
        self.FreqBandsMin=FreqBands[0:-1].copy()
        self.FreqBandsMax=FreqBands[1::].copy()
        self.FreqBandsInfos={}
        for iBand in range(self.NFreqBands):
            self.FreqBandsInfos[iBand]=[]

        self.FreqBandsInfosDegrid={}

        print
        self.ListFreqs=[]
        self.DicoMSChanMapping={}
        self.DicoMSChanMappingDegridding={}
        for MS,iMS in zip(self.ListMS,range(self.nMS)):
            FreqBand = np.where((self.FreqBandsMin <= np.mean(MS.ChanFreq))&(self.FreqBandsMax >= np.mean(MS.ChanFreq)))[0][0]
            self.ListFreqs+=MS.ChanFreq.tolist()

            nch=MS.ChanFreq.size
            FreqBandsMin=self.FreqBandsMin.reshape((1,NFreqBands))
            FreqBandsMax=self.FreqBandsMax.reshape((1,NFreqBands))
            ThisChanFreq=MS.ChanFreq.reshape((nch,1))
            Mask=((ThisChanFreq>=FreqBandsMin)&(ThisChanFreq<=FreqBandsMax))
            ThisMapping=np.argmax(Mask,axis=1)
            self.DicoMSChanMapping[iMS]=ThisMapping
            
            ThisFreqs=MS.ChanFreq.ravel()
            for iFreqBand in list(set(ThisMapping.tolist())):
                indChan=np.where(ThisMapping==iFreqBand)[0]
                self.FreqBandsInfos[iFreqBand]+=(ThisFreqs[indChan]).tolist()
            
            NChanDegrid = self.GD["MultiFreqs"]["NChanDegridPerMS"] or ThisFreqs.size
            NChanDegrid = min(NChanDegrid, ThisFreqs.size)
            ChanDegridding=np.linspace(ThisFreqs.min(),ThisFreqs.max(),NChanDegrid+1)
            FreqChanDegridding=(ChanDegridding[1::]+ChanDegridding[0:-1])/2.
            NChanDegrid=FreqChanDegridding.size
            NChanMS=MS.ChanFreq.size
            DChan=np.abs(MS.ChanFreq.reshape((NChanMS,1))-FreqChanDegridding.reshape((1,NChanDegrid)))
            ThisMappingDegrid=np.argmin(DChan,axis=1)
            self.DicoMSChanMappingDegridding[iMS]=ThisMappingDegrid

            
            
            MeanFreqDegrid=np.zeros((NChanDegrid),np.float32)
            for iFreqBand in range(NChanDegrid):
                ind=np.where(ThisMappingDegrid==iFreqBand)[0]
                MeanFreqDegrid[iFreqBand]=np.mean(ThisFreqs[ind])
            self.FreqBandsInfosDegrid[iMS]=MeanFreqDegrid
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

    def CalcWeights(self):
        if self.VisWeights!=None: return
        #ImShape=self.PaddedFacetShape
        ImShape=self.FullImShape#self.FacetShape
        CellSizeRad=self.CellSizeRad
        WeightMachine=ClassWeighting.ClassWeighting(ImShape,CellSizeRad)
        uvw,WEIGHT,flags,nrows = self.GiveAllUVW()
        # uvw=DATA["uvw"]
        # WEIGHT=DATA["Weights"]
        VisWeights=WEIGHT#[:,0]#np.ones((uvw.shape[0],),dtype=np.float32)
        #VisWeights=np.ones((uvw.shape[0],),dtype=np.float32)
        Robust=self.Robust

        #self.VisWeights=np.ones((uvw.shape[0],self.MS.ChanFreq.size),dtype=np.float64)

        allweights = WeightMachine.CalcWeights(uvw,VisWeights,flags,self.MS.ChanFreq,
                                              Robust=Robust,
                                              Weighting=self.Weighting,
                                              Super=self.Super)




        # allweights = WeightMachine.CalcWeightsOld(uvw,VisWeights,flags,self.MS.ChanFreq,
        #                                       Robust=Robust,
        #                                       Weighting=self.Weighting,
        #                                       Super=self.Super)

        # self.WisWeights is a list of weight arrays, one per each MS in self.ListMS
        self.VisWeights = []
        row0 = 0
        for nr in nrows:
            self.VisWeights.append(allweights[row0:(row0+nr)])
            row0 += nr
        self.CurrentVisWeights = self.VisWeights[0]

        # DDF_WEIGHTS="DDF_WEIGHTS"
        # print>>log, "Writing weights in column %s"%(DDF_WEIGHTS)
        # for iMS,ThisMS in zip(range(len(self.ListMS)),self.ListMS):
        #     ThisMS.AddCol(DDF_WEIGHTS,LikeCol="IMAGING_WEIGHT")
        #     t=table(ThisMS.MSName,readonly=False,ack=False)
        #     t.putcol(DDF_WEIGHTS,self.VisWeights[iMS])
        #     t.close()


        # self.CalcMeanBeam()

    def CalcMeanBeam(self):
        AverageBeamMachine=ClassBeamMean.ClassBeamMean(self)
        AverageBeamMachine.LoadData()
        AverageBeamMachine.CalcMeanBeam()

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
            if not(key in ['times', 'A1', 'A0', 'flags', 'uvw', 'data',"Weights","uvw_dt", "MSInfos","ChanMapping","ChanMappingDegrid"]):             
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
        print>>log,"Reinit ms iterator [%i / %i]"%(self.iCurrentMS+1,self.nMS)
        self.CurrentFreqBand=0
        self.CurrentVisWeights = self.VisWeights and self.VisWeights[0]   # first time VisWeights might still be unset -- but then CurrentVisWeights will be set later in CalcWeights
        for MS in self.ListMS:
            MS.ReinitChunkIter(self.TMemChunkSize)
        self.CurrentMS=self.ListMS[0]
        self.CurrentChanMapping=self.DicoMSChanMapping[0]
        self.CurrentChanMappingDegrid=self.FreqBandsInfosDegrid[0]
        #print>>log, (ModColor.Str("NextMS %s"%(self.CurrentMS.MSName),col="green") + (" --> freq. band %i"%self.CurrentFreqBand))

    def setNextMS(self):
        if (self.iCurrentMS+1)==self.nMS:
            print>>log, ModColor.Str("Reached end of MSList")
            return "EndListMS"
        else:
            self.iCurrentMS+=1
            print>>log,"Setting next ms [%i / %i]"%(self.iCurrentMS+1,self.nMS)
            self.CurrentMS=self.ListMS[self.iCurrentMS]
            self.CurrentFreqBand=0
            self.CurrentVisWeights = self.VisWeights[self.iCurrentMS]
            if self.MultiFreqMode:
                self.CurrentFreqBand = np.where((self.FreqBandsMin <= np.mean(self.CurrentMS.ChanFreq))&(self.FreqBandsMax > np.mean(self.CurrentMS.ChanFreq)))[0][0]

            self.CurrentChanMapping=self.DicoMSChanMapping[self.iCurrentMS]
            self.CurrentChanMappingDegrid=self.FreqBandsInfosDegrid[self.iCurrentMS]
            #print>>log, (ModColor.Str("NextMS %s"%(self.CurrentMS.MSName),col="green") + (" --> freq. band %i"%(self.CurrentFreqBand)))
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

        # ## debug
        # ind=np.where((A0==14)&(A1==31))[0]
        # flags=flags[ind]
        # data=data[ind]
        # A0=A0[ind]
        # A1=A1[ind]
        # uvw=uvw[ind]
        # times=times[ind]
        # ##

        DATA={}
        DATA["flags"]=flags
        DATA["data"]=data
        DATA["uvw"]=uvw
        DATA["A0"]=A0
        DATA["A1"]=A1
        DATA["times"]=times


        DATA["Weights"]=self.CurrentVisWeights[MS.ROW0:MS.ROW1]
        DecorrMode=self.GD["DDESolutions"]["DecorrMode"]

        
        if ('F' in DecorrMode)|("T" in DecorrMode):
            DATA["uvw_dt"]=np.float64(self.CurrentMS.Give_dUVW_dt(times,A0,A1))
            DATA["MSInfos"]=np.array([repLoadChunk["dt"],repLoadChunk["dnu"].ravel()[0]],np.float32)
            #DATA["MSInfos"][1]=20000.*30
            #DATA["MSInfos"][0]=500.


        ThisMSName=reformat.reformat(os.path.abspath(self.CurrentMS.MSName),LastSlash=False)
        TimeMapName="%s/Flagging.npy"%ThisMSName
        try:
            DATA["flags"]=np.load(TimeMapName)
        except:
            self.UpdateFlag(DATA)


        

        DATA["ChanMapping"]=self.CurrentChanMapping
        DATA["ChanMappingDegrid"]=self.DicoMSChanMappingDegridding[self.iCurrentMS]
        
        print>>log, "  Channel Mapping Gridding  : %s"%(str(self.CurrentChanMapping))
        print>>log, "  Channel Mapping DeGridding: %s"%(str(DATA["ChanMappingDegrid"]))

        self.UpdateCompression(DATA,
                               ChanMappingGridding=DATA["ChanMapping"],
                               ChanMappingDeGridding=self.DicoMSChanMappingDegridding[self.iCurrentMS])


        JonesMachine=ClassJones.ClassJones(self.GD,self.CurrentMS,self.FacetMachine,IdSharedMem=self.IdSharedMem)
        JonesMachine.InitDDESols(DATA)

        #############################
        #############################

        




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
                     "Weights":self.CurrentVisWeights[MS.ROW0:MS.ROW1],
                     "ChanMapping":DATA["ChanMapping"],
                     "ChanMappingDegrid":DATA["ChanMappingDegrid"]
                     }
        
        DecorrMode=self.GD["DDESolutions"]["DecorrMode"]
        if ('F' in DecorrMode)|("T" in DecorrMode):
            DicoDataOut["uvw_dt"]=DATA["uvw_dt"]
            DicoDataOut["MSInfos"]=DATA["MSInfos"]


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
            if ind.size==0: continue
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


        DATA["data"][flags]=1e9

        # if dt0<dt1:
        #     JonesBeam=np.zeros((Tm.size,),dtype=[("t0",np.float32),("t1",np.float32),("tm",np.float32),("Jones",(NDir,self.MS.na,self.MS.NSPWChan,2,2),np.complex64)])


    def setFacetMachine(self,FacetMachine):
        self.FacetMachine=FacetMachine
        self.FullImShape=self.FacetMachine.OutImShape
        self.PaddedFacetShape=self.FacetMachine.PaddedGridShape
        self.FacetShape=self.FacetMachine.FacetShape
        self.CellSizeRad=self.FacetMachine.CellSizeRad

    def setFOV(self,sh0,sh1,sh2,cell):
        self.FullImShape=sh0
        self.PaddedFacetShape=sh1
        self.FacetShape=sh2
        self.CellSizeRad=cell

    def UpdateCompression(self,DATA,ChanMappingGridding=None,ChanMappingDeGridding=None):
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

                #FinalMapping,fact=SmearMapMachine.BuildSmearMapping(DATA)
                #stop
                FinalMapping,fact=SmearMapMachine.BuildSmearMappingParallel(DATA,ChanMappingGridding)

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
                FinalMapping,fact=SmearMapMachine.BuildSmearMappingParallel(DATA,ChanMappingDeGridding)
                np.save(MapName,FinalMapping)
                print>>log, ModColor.Str("  Effective compression [DeGrid]:   %.2f%%"%fact,col="green")

            Map=NpShared.ToShared("%sMappingSmearing.DeGrid"%(self.IdSharedMem),FinalMapping)




    def GiveAllUVW(self):
        """Reads UVWs, weights and flags from all MSs in the list and concatenates them into 
        consolidated arrays.
        Returns uvw,weight,flags,nrows, where shapes are 
        (nrow,3), (nrow,nchan) and (nrow,nchan,ncorr) respectively.
        nrows is a vector showing the number of rows in each MS in the list.
        """
        WeightCol=self.GD["VisData"]["WeightCol"]
        # make lists of tables and row counts (one per MS)
        tabs = [ ms.GiveMainTable() for ms in self.ListMS ]
        chanslices = [ ms.ChanSlice for ms in self.ListMS ]
        nrows = [ tab.nrows() for tab in tabs ]
        nr = sum(nrows)
        # preallocate arrays
        # NB: this assumes nchan and ncorr is the same across all MSs in self.ListMS. Tough luck if it isn't!
        uvws = np.zeros((nr,3),np.float64)
        weights = np.zeros((nr,self.MS.Nchan),np.float64)
        flags = np.zeros((nr,self.MS.Nchan,len(self.MS.CorrelationNames)),bool)

        # now loop over MSs and read data
        row0 = 0
        for num_ms, (nrow, tab, chanslice) in enumerate(zip(nrows, tabs, chanslices)):
            uvws[row0:(row0+nrow),...]   = tab.getcol("UVW")
            flags[row0:(row0+nrow),...] = tab.getcol("FLAG")[:,chanslice,:]

            if WeightCol == "WEIGHT_SPECTRUM":
                WEIGHT=tab.getcol(WeightCol)[:,chanslice]
                print>>log, "  Reading column %s for the weights, shape is %s"%(WeightCol,WEIGHT.shape)
                WEIGHT = (WEIGHT[:,:,0]+WEIGHT[:,:,3])/2.
            elif WeightCol == "WEIGHT":
                WEIGHT=tab.getcol(WeightCol)
                print>>log, "  Reading column %s for the weights, shape is %s"%(WeightCol,WEIGHT.shape)
                WEIGHT = (WEIGHT[:,0]+WEIGHT[:,3])/2.
                # expand to have frequency axis
                WEIGHT = WEIGHT[:,np.newaxis] + np.zeros(self.MS.Nchan,np.float32)[np.newaxis,:]
            elif WeightCol == "WEIGHT+WEIGHT_SPECTRUM" or WeightCol == "WEIGHT_SPECTRUM+WEIGHT":
                w = tab.getcol("WEIGHT")
                ws = tab.getcol("WEIGHT_SPECTRUM")[:,chanslice]
                print>>log, "  Reading column %s for the weights, shape is %s and %s"%(WeightCol,w.shape,ws.shape)
                WEIGHT = w[:,np.newaxis,:] * ws
                WEIGHT = (WEIGHT[:,:,0]+WEIGHT[:,:,3])/2.
            else:
                ## in all other cases (i.e. IMAGING_WEIGHT) assume a column of shape NRow,NFreq to begin with, check for this:
                WEIGHT=tab.getcol(WeightCol)[:,chanslice]
                print>>log, "  Reading column %s for the weights, shape is %s"%(WeightCol,WEIGHT.shape)

            if WEIGHT.shape != (nrow, self.MS.Nchan):
                raise TypeError,"weights expected to have shape of %s"%((nrow, self.MS.Nchan),)

            if np.max(WEIGHT)==0:
                print>>log,"    All imaging weights are 0, setting them to ones"
                WEIGHT.fill(1)


            weights[row0:(row0+nrow),...] = WEIGHT

            tab.close()
            row0 += nrow

        MeanW=np.mean(weights)
        if MeanW!=0.:
            weights/=MeanW


        return uvws,weights,flags,nrows


