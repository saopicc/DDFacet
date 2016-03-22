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
        global_freqs = set()
        NChanMax=0
        ChanStart = self.DicoSelectOptions.get("ChanStart",0)
        ChanEnd   = self.DicoSelectOptions.get("ChanEnd",-1)
        ChanStep  = self.DicoSelectOptions.get("ChanStep",1)
        if (ChanStart,ChanEnd,ChanStep) == (0,-1,1):
            chanslice = None
        else:
            chanslice = slice(ChanStart, ChanEnd if ChanEnd != -1 else None, ChanStep) 

        min_freq = 1e+999
        max_freq = 0

        for MSName in self.ListMSName:
            MS=ClassMS.ClassMS(MSName,Col=self.ColName,DoReadData=False,AverageTimeFreq=(1,3),
                Field=self.Field,DDID=self.DDID,
                ChanSlice=chanslice) 
            self.ListMS.append(MS)
            # accumulate global set of frequencies, and min/max frequency
            global_freqs.update(MS.ChanFreq)
            min_freq = min(min_freq,(MS.ChanFreq-MS.ChanWidth/2).min())
            max_freq = max(max_freq,(MS.ChanFreq+MS.ChanWidth/2).max())
            
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
        # make list of unique frequencies
        self.GlobalFreqs = np.array(sorted(global_freqs))
        self.NFreqBands  = np.min([self.GD["MultiFreqs"]["NFreqBands"],len(self.GlobalFreqs)])#self.nMS])
        self.CurrentMS   = self.ListMS[0]
        self.iCurrentMS=0

        bandwidth = max_freq - min_freq
        print>>log,"Total bandwidth is %g MHz (%g to %g MHz), with %d channels"%(bandwidth*1e-6, min_freq*1e-6, max_freq*1e-6, len(global_freqs))

        # print>>log,"GlobalFreqs: %d: %s"%(len(self.GlobalFreqs),repr(self.GlobalFreqs))

        self.MultiFreqMode=False
        NFreqBands=self.NFreqBands
        if self.NFreqBands>1: 
            self.MultiFreqMode=True
            print>>log, ModColor.Str("MultiFrequency Mode: ON")
        else:
            self.GD["MultiFreqs"]["NFreqBands"] = 1
            self.GD["MultiFreqs"]["Alpha"] = [0.,0.,1.]
            print>>log, ModColor.Str("MultiFrequency Mode: OFF")
            
        # Divide the global frequencies into frequency bands.
        # Somewhat of an open question how best to do it (equal bandwidth, or equal number of channels?), this is where
        # we can play various games with mapping the GlobalFreqs into a number of image bands.
        # For now, let's do it by equal bandwith as Cyril used to do: 

        ## these aren't used anywhere except the loop to construct the mapping below, so I'll remove them
        # FreqBands = np.linspace(self.GlobalFreqs.min(),self.GlobalFreqs.max(),NFreqBands+1)
        # self.FreqBandsMin = FreqBands[0:-1].copy()
        # self.FreqBandsMax = FreqBands[1::].copy()
        # self.FreqBandsMean = (self.FreqBandsMin + self.FreqBandsMax)/2
        # now make mapping from global frequency into band number
        grid_bw = bandwidth/NFreqBands

        # grid_band: array of ints, same size as self.GlobalFreqs, giving the grid band number of each frequency
        grid_band = np.floor((self.GlobalFreqs - min_freq)/grid_bw).astype(int)
        # freq_to_grid_band: mapping from frequency to grid band number
        freq_to_grid_band = dict(zip(self.GlobalFreqs, grid_band))
        # print>>log,sorted(freq_to_grid_band.items())

        self.FreqBandsInfos = {}
        # freq_to_grid_band_chan: mapping from frequency to channel number within its grid band 
        freq_to_grid_band_chan = {}
        for iBand in range(self.NFreqBands):
            freqlist = sorted([ freq for freq,band in freq_to_grid_band.iteritems() if band == iBand ])
            self.FreqBandsInfos[iBand] = freqlist
            freq_to_grid_band_chan.update(dict([ (freq,chan) for chan,freq in enumerate(freqlist)]))
            print>>log,"Band %d: %d channels centred on %g...%g MHz"%(iBand, len(freqlist), freqlist[0]*1e-6, freqlist[-1]*1e-6)

        self.FreqBandsInfosDegrid={}
        self.DicoMSChanMapping={}
        self.DicoMSChanMappingChan={}
        self.DicoMSChanMappingDegridding={}
        # structures initialized here:
        # self.FreqBandsInfosDegrid: a dict, indexed by MS number
        #       [iMS] = float32 array of NChanDegridPerMS frequencies at which the degridding will proceed for this MS
        # self.FreqBandsInfos: a list, indexed by freq band number (NFreqBands items)
        #       [iband] = list of frequencies within that frequency band 
        # self.DicoMSChanMappingDegridding: a dict, indexed by MS number
        #       [iMS] = int array of band numbers, as many as there are channels in the MS. For each channel, gives the degridding band number
        #               (from 0 to NChanDegridPerMS-1)
        # self.DicoMSChanMapping: a dict, indexed by MS number
        #       [iMS] = int array of band numbers, as many as there are channels in the MS. For each channel, gives the gridding band number
        #               (from 0 to NFreqBands-1)
        # self.DicoMSChanMappingChan: a dict, indexed by MS number
        #       [iMS] = int array of channel numbers, as many as there are channels in the MS. 
        #               For each channel, gives its number in the gridding band

        for iMS, MS in enumerate(self.ListMS):
            min_freq = (MS.ChanFreq - MS.ChanWidth/2).min()
            max_freq = (MS.ChanFreq + MS.ChanWidth/2).max()
            bw = max_freq - min_freq
            # print>>log,bw,min_freq,max_freq
            # map each channel to a gridding band
            bands = [ freq_to_grid_band[freq] for freq in MS.ChanFreq ]
            self.DicoMSChanMapping[iMS] = np.array(bands)
            self.DicoMSChanMappingChan[iMS] = np.array([ freq_to_grid_band_chan[freq] for freq in MS.ChanFreq ])

            # now split the bandwidth into NChanDegridPerMS band, and map each channel to a degridding band
            NChanDegrid = self.GD["MultiFreqs"]["NChanDegridPerMS"] or MS.ChanFreq.size
            degrid_bw = bw/NChanDegrid
            self.DicoMSChanMappingDegridding[iMS] = np.floor((MS.ChanFreq - min_freq)/degrid_bw).astype(int)

            # calculate center frequency of each degridding band
            edges = np.linspace(min_freq, max_freq, NChanDegrid+1)
            self.FreqBandsInfosDegrid[iMS] = (edges[:-1] + edges[1:])/2

            print>>log,"%s   Bandwidth is %g MHz (%g to %g MHz), gridding bands are %s"%(MS, bw*1e-6, min_freq*1e-6, max_freq*1e-6, ", ".join(map(str,set(bands))))
            print>>log,"Band mapping: %s"%(" ".join(map(str,bands)))
            print>>log,"Chan mapping: %s"%(" ".join(map(str,self.DicoMSChanMappingChan[iMS])))

#            print>>log,MS

            # print>>log,"FreqBandsInfosDegrid %s"%repr(self.FreqBandsInfosDegrid[iMS])
            # print>>log,"self.DicoMSChanMappingDegriding %s"%repr(self.DicoMSChanMappingDegridding[iMS])
            # print>>log,"self.DicoMSChanMapping %s"%repr(self.DicoMSChanMapping[iMS])

        # print>>log,"FreqBandsInfos %s"%repr(self.FreqBandsInfos)

#        self.RefFreq=np.mean(self.ListFreqs)
        self.RefFreq=np.mean(self.GlobalFreqs)

        

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
        ImShape = self.FullImShape#self.FacetShape
        CellSizeRad = self.CellSizeRad
        WeightMachine = ClassWeighting.ClassWeighting(ImShape,CellSizeRad)
        uv_weights_flags_freqs = self.GiveUvWeightsFlagsFreqs()

        VisWeights = [ weights for uv,weights,flags,freqs in uv_weights_flags_freqs ] 
        
        if all([ w.max() == 0 for w in VisWeights]):
            print>>log,"All imaging weights are 0, setting them to ones"
            for w in VisWeights:
                w.fill(1)
        #VisWeights=np.ones((uvw.shape[0],),dtype=np.float32)
        
        Robust = self.Robust

        #self.VisWeights=np.ones((uvw.shape[0],self.MS.ChanFreq.size),dtype=np.float64)

        self.VisWeights = WeightMachine.CalcWeights(uv_weights_flags_freqs,
                                              Robust=Robust,
                                              Weighting=self.Weighting,
                                              Super=self.Super)

        self.CurrentVisWeights = self.VisWeights[0]
        print>>log,self.CurrentVisWeights.mean(),self.CurrentVisWeights
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
        # self.CurrentFreqBand=0
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
            print>>log,"next ms (%d/%d)"%(self.iCurrentMS+1,self.nMS)
            self.CurrentMS=self.ListMS[self.iCurrentMS]
            # self.CurrentFreqBand=0
            self.CurrentVisWeights = self.VisWeights[self.iCurrentMS]
            ## OMS: CurrentFreqBand no longer used anywhere, so I remove it. An MS can correspond to multiple (gridding) bands anyway.
            # if self.MultiFreqMode:
            #     self.CurrentFreqBand = np.where((self.FreqBandsMin <= np.mean(self.CurrentMS.ChanFreq))&(self.FreqBandsMax > np.mean(self.CurrentMS.ChanFreq)))[0][0]

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




    def GiveUvWeightsFlagsFreqs (self):
        """Reads UVs, weights, flags, freqs from all MSs in the list.
        Returns list of (uv,weights,flags,freqs) tuples, one per each MS in self.ListMS, where shapes are 
        (nrow,2), (nrow,nchan), (nrow) and (nchan) respectively. 
        Note that flags are "row flags" i.e. only True if all channels are flagged. 
        Per-channel flagging is taken care of in here, by setting that channel's weight to 0.
        """
        WeightCol=self.GD["VisData"]["WeightCol"]
        # now loop over MSs and read data
        weightsum = 0
        nweights = 0
        output_list = []
        for num_ms, ms in enumerate(self.ListMS):
            tab = ms.GiveMainTable()
            chanslice = ms.ChanSlice
            if not tab.nrows():
                # if no data in this MS, make single, flagged entry
                output_list.append((np.zeros((1,2)), np.zeros((1,len(ms.ChanFreq))), np.array([True]), ms.ChanFreq))
                continue
            uvs = tab.getcol("UVW")[:,:2]
            flags = tab.getcol("FLAG")[:,chanslice,:]
            # if any polarization is flagged, flag all 4 correlations. Shape of flags becomes nrow,nchan
            flags = flags.max(axis=2)
            # valid: array of Nrow,Nchan, with meaning inverse to flags 
            valid = ~flags
            # if all channels are flagged, flag whole row. Shape of flags becomes nrow
            flags = flags.min(axis=1)

            if WeightCol == "WEIGHT_SPECTRUM":
                WEIGHT=tab.getcol(WeightCol)[:,chanslice]
                print>>log, "  Reading column %s for the weights, shape is %s"%(WeightCol,WEIGHT.shape)
                # take the mean XX/YY weight
                WEIGHT = (WEIGHT[:,:,0]+WEIGHT[:,:,3])/2 * valid
                
            elif WeightCol == "WEIGHT":
                WEIGHT=tab.getcol(WeightCol)
                print>>log, "  Reading column %s for the weights, shape is %s, will expand frequency axis"%(WeightCol,WEIGHT.shape)
                WEIGHT = (WEIGHT[:,0]+WEIGHT[:,3])/2.
                # expand to have frequency axis
                WEIGHT = WEIGHT[:,np.newaxis] * valid
            else:
                ## in all other cases (i.e. IMAGING_WEIGHT) assume a column of shape NRow,NFreq to begin with, check for this:
                WEIGHT = tab.getcol(WeightCol)[:,chanslice]
                print>>log, "  Reading column %s for the weights, shape is %s"%(WeightCol,WEIGHT.shape)
                if WEIGHT.shape != valid.shape:
                    raise TypeError,"weights column expected to have shape of %s"%(valid.shape,)
                WEIGHT *= valid

            WEIGHT = WEIGHT.astype(np.float64)

            output_list.append((uvs, WEIGHT, flags, ms.ChanFreq))
            tab.close()

            weightsum = weightsum + WEIGHT.sum(dtype=np.float64)
            nweights += valid.sum()

        # normalize weights
        print>>log,"normalizing weights (sum %g from %d valid visibility points)"%(weightsum, nweights)
        mw = nweights and weightsum / nweights
        if mw:
            for uvw, weights, flags, freqs in output_list:
                weights /= mw
        print>>log,"normalization done, mean weight was %g"%mw

        return output_list

