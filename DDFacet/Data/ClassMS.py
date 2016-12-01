import os
import pyrap.measures as pm
import pyrap.quanta as qa
from pyrap.tables import table

import ephem
import numpy as np
from DDFacet.Other import ModColor
from DDFacet.Other import MyLogger
from DDFacet.Other import reformat
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms

log= MyLogger.getLogger("ClassMS")
from DDFacet.Other import ClassTimeIt
from DDFacet.Other.CacheManager import CacheManager
from DDFacet.Array import NpShared
import sidereal
import time

try:
    import lofar.stationresponse as lsr
except:
    print>>log, ModColor.Str("Could not import lofar.stationresponse")

class ClassMS():
    def __init__(self,MSname,Col="DATA",zero_flag=True,ReOrder=False,EqualizeFlag=False,DoPrint=True,DoReadData=True,
                 TimeChunkSize=None,GetBeam=False,RejectAutoCorr=False,SelectSPW=None,DelStationList=None,
                 AverageTimeFreq=None,
                 Field=0,DDID=0,TaQL=None,ChanSlice=None,GD=None,
                 DicoSelectOptions={},
                 ResetCache=False):
        """
        Args:
            MSname:
            Col:
            zero_flag:
            ReOrder:
            EqualizeFlag:
            DoPrint:
            DoReadData:
            TimeChunkSize:
            GetBeam:
            RejectAutoCorr:
            SelectSPW:
            DelStationList:
            AverageTimeFreq:
            Field:
            DDID:
            ChanSlice:
            DicoSelectOptions: dict of data selection options applied to this MS
            ResetCache: if True, cached products will be reset
        """

        if MSname=="": exit()
        self.AverageSteps=AverageTimeFreq
        MSname= reformat.reformat(os.path.abspath(MSname), LastSlash=False)
        self.MSName=MSname
        self.ColName=Col
        self.ChanSlice = ChanSlice or slice(None)
        self.zero_flag=zero_flag
        self.ReOrder=ReOrder
        self.EqualizeFlag=EqualizeFlag
        self.DoPrint=DoPrint
        self.TimeChunkSize=TimeChunkSize
        self.RejectAutoCorr=RejectAutoCorr
        self.SelectSPW=SelectSPW
        self.DelStationList=DelStationList
        self.Field = Field
        self.DDID = DDID
        self.TaQL = "FIELD_ID==%d && DATA_DESC_ID==%d" % (Field, DDID)
        if TaQL:
            self.TaQL += " && (%s)"%TaQL
        self.ReadMSInfo(DoPrint=DoPrint)
        self.LFlaggedStations=[]
        self.GD = GD
        self.DicoSelectOptions = DicoSelectOptions
        self._datapath = self._flagpath = None
        self._start_time = time.time()

        try:
            self.LoadLOFAR_ANTENNA_FIELD()
        except:
            self.LOFAR_ANTENNA_FIELD=None
            pass

        #self.LoadLOFAR_ANTENNA_FIELD()

        if DoReadData: self.ReadData()
        #self.RemoveStation()


        self.SR=None
        if GetBeam:
            self.LoadSR()

        # the MS has two caches associated with it. self.maincache stores DDF-related data that is not related to
        # iterating through the MS. self.cache is created in GiveNextChunk, and is thus different from chunk
        # to chunk. It is stored in self._chunk_caches, so that the per-chunk cache manager is initialized only
        # once.
        self._reset_cache = ResetCache
        self._chunk_caches = {}
        self.maincache = CacheManager(MSname+".ddfcache", reset=ResetCache)

    def GiveMainTable (self,**kw):
        """Returns main MS table, applying TaQL selection if any"""
        ack = kw.pop("ack", False)
        t = table(self.MSName,ack=ack,**kw)

        if self.TaQL:
            t = t.query(self.TaQL)
        return t

    def GiveDate(self,tt):
        time_start = qa.quantity(tt, 's')
        me = pm.measures()
        dict_time_start_MDJ = me.epoch('utc', time_start)
        time_start_MDJ=dict_time_start_MDJ['m0']['value']
        JD=time_start_MDJ+2400000.5-2415020
        d=ephem.Date(JD)

        #return d.datetime().isoformat().replace("T","/")
        return d.datetime()#.isoformat().replace("T","/")

    def GiveDataChunk(self,it0,it1):
        MapSelBLs=self.MapSelBLs
        nbl=self.nbl
        row0,row1=it0*nbl,it1*nbl
        NtimeBlocks=nt=it1-it0
        nrow=row1-row0
        _,nch,_=self.data.shape

        DataOut=self.data[row0:row1,:,:].copy()
        DataOut=DataOut.reshape((NtimeBlocks,nbl,nch,4))
        DataOut=DataOut[:,self.MapSelBLs,:,:]
        DataOut=DataOut.reshape((DataOut.shape[1]*NtimeBlocks,nch,4))

        flags=self.flag_all[row0:row1,:,:].copy()
        flags=flags.reshape((NtimeBlocks,nbl,nch,4))
        flags=flags[:,self.MapSelBLs,:,:]
        flags=flags.reshape((flags.shape[1]*NtimeBlocks,nch,4))

        uvw=self.uvw[row0:row1,:].copy()
        uvw=uvw.reshape((NtimeBlocks,self.nbl,3))
        uvw=uvw[:,self.MapSelBLs,:]
        uvw=uvw.reshape((uvw.shape[1]*NtimeBlocks,3))

        A0=self.A0[self.MapSelBLs].copy()
        A1=self.A1[self.MapSelBLs].copy()

        times=self.times_all[row0:row1].copy()
        times=times.reshape((NtimeBlocks,self.nbl))
        times=times[:,self.MapSelBLs]
        times=times.reshape((times.shape[1]*NtimeBlocks))
        
        DicoOut={"data":DataOut,
                 "flags":flags,
                 "A0A1":(A0,A1),
                 "times":times,
                 "uvw":uvw}
        return DicoOut

        


    def PutLOFARKeys(self):
        keys=["LOFAR_ELEMENT_FAILURE", "LOFAR_STATION", "LOFAR_ANTENNA_FIELD"]
        t=table(self.MSName,ack=False)
        for key in keys:
            t.putkeyword(key,'Table: %s/%s'%(self.MSName,key))
        t.close()

    def DelData(self):
        try:
            del(self.Weights)
        except:
            pass

        try:
            del(self.data,self.flag_all)
        except:
            pass


    def LoadSR(self,useElementBeam=True,useArrayFactor=True):
        if self.SR is not None: return
        # t=table(self.MSName,ack=False,readonly=False)
        # if not("LOFAR_ANTENNA_FIELD" in t.getkeywords().keys()):
        #     self.PutLOFARKeys()
        # t.close()
        
        #print>>log, "Import"
        #print>>log, "  Done"
        
        # f=self.ChanFreq.flatten()
        # if f.shape[0]>1:
        #     t=table(self.MSName+"/SPECTRAL_WINDOW/",ack=False)
        #     c=t.getcol("CHAN_WIDTH")
        #     c.fill(np.abs((f[0:-1]-f[1::])[0]))
        #     t.putcol("CHAN_WIDTH",c)
        #     t.close()

        #print>>log, "Declare %s"%self.MSName
        self.SR = lsr.stationresponse(self.MSName,
                                      useElementResponse=useElementBeam,
                                      #useElementBeam=useElementBeam,
                                      useArrayFactor=useArrayFactor)#,useChanFreq=True)
        #print>>log, "  Done"
        #print>>log, "Set direction %f, %f"%(self.rarad,self.decrad)
        self.SR.setDirection(self.rarad,self.decrad)
        #print>>log, "  Done"

        
    def CopyNonSPWDependent(self,MSnodata):
        MSnodata.A0=self.A0
        MSnodata.A1=self.A1
        MSnodata.uvw=self.uvw
        MSnodata.ntimes=self.ntimes
        MSnodata.times=self.times
        MSnodata.times_all=self.times_all
        MSnodata.LOFAR_ANTENNA_FIELD=self.LOFAR_ANTENNA_FIELD
        return MSnodata

    def LoadLOFAR_ANTENNA_FIELD(self):
        t=table("%s/LOFAR_ANTENNA_FIELD"%self.MSName,ack=False)
        #print>>log, ModColor.Str(" ... Loading LOFAR_ANTENNA_FIELD table...")
        na,NTiles,dummy=t.getcol("ELEMENT_OFFSET").shape

        try:
            dummy,nAntPerTiles,dummy=t.getcol("TILE_ELEMENT_OFFSET").shape
            TileOffXYZ=t.getcol("TILE_ELEMENT_OFFSET").reshape(na,1,nAntPerTiles,3)
            RCU=t.getcol("ELEMENT_RCU")
            RCUMask=(RCU!=-1)[:,:,0]
            Flagged=t.getcol("ELEMENT_FLAG")[:,:,0]
        except:
            nAntPerTiles=1
            # RCUMask=(RCU!=-1)[:,:,:]
            # RCUMask=RCUMask.reshape(na,96)
            RCUMask=np.ones((na,96),bool)
            TileOffXYZ=np.zeros((na,1,nAntPerTiles,3),float)
            Flagged=t.getcol("ELEMENT_FLAG")[:,:,0]
            #Flagged=Flagged.reshape(Flagged.shape[0],Flagged.shape[1],1,1)*np.ones((1,1,1,3),bool)
            
        StationXYZ=t.getcol("POSITION").reshape(na,1,1,3)
        ElementOffXYZ=t.getcol("ELEMENT_OFFSET")
        ElementOffXYZ=ElementOffXYZ.reshape(na,NTiles,1,3)
        
        Dico={}
        Dico["FLAGED"]=Flagged
        Dico["StationXYZ"]=StationXYZ
        Dico["ElementOffXYZ"]=ElementOffXYZ
        Dico["TileOffXYZ"]=TileOffXYZ
        #Dico["RCU"]=RCU
        Dico["RCUMask"]=RCUMask
        Dico["nAntPerTiles"]=nAntPerTiles

        t.close()
        self.LOFAR_ANTENNA_FIELD=Dico
        
        
    def GiveBeam(self,time,ra,dec):
        self.LoadSR()
        Beam=np.zeros((ra.shape[0],self.na,self.NSPWChan,2,2),dtype=np.complex)
        for i in range(ra.shape[0]):
            self.SR.setDirection(ra[i],dec[i])
            Beam[i]=self.SR.evaluate(time)
        #Beam=np.swapaxes(Beam,1,2)
        return Beam


    def GiveMappingAnt(self,ListStrSel,(row0,row1)=(None,None),FlagAutoCorr=True,WriteAttribute=True):

        if type(ListStrSel)!=list:
            assert(False)

        #ListStrSel=["RT9-RTA", "RTA-RTB", "RTC-RTD", "RT6-RT7", "RT5"]

        print>>log, ModColor.Str("  ... Building BL-mapping for %s" % str(ListStrSel))

        if row1 is None:
            row0=0
            row1=self.nbl
        A0=self.F_A0[row0:row1]
        A1=self.F_A1[row0:row1]
        MapOut=np.ones((self.nbl,),dtype=np.bool)
        if FlagAutoCorr:
            ind=np.where(A0==A1)[0]
            MapOut[ind]=False

        def GiveStrAntToNum(self,StrAnt):
            ind=[]
            for i in range(len(self.StationNames)):
                if StrAnt in self.StationNames[i]:
                    ind.append(i)
            #print ind
            return ind
        #MapOut=np.ones(A0.shape,bool)###)

        LFlaggedStations=[]
        LNumFlaggedStations=[]

        for blsel in ListStrSel:
            if blsel=="": continue
            if "-" in blsel:
                StrA0,StrA1=blsel.split("-")
                LNumStrA0=GiveStrAntToNum(self,StrA0)
                LNumStrA1=GiveStrAntToNum(self,StrA1)
                for NumStrA0 in LNumStrA0:
                    for NumStrA1 in LNumStrA1:
                        NumA0=np.where(np.array(self.StationNames)==NumStrA0)[0]
                        NumA1=np.where(np.array(self.StationNames)==NumStrA1)[0]
                        C0=((A0==NumA0)&(A1==NumA1))
                        C1=((A1==NumA0)&(A0==NumA1))
                        ind=np.where(C1|C0)[0]
                        MapOut[ind]=False
            else:
                #NumA0=np.where(np.array(self.StationNames)==blsel)[0]
                StrA0=blsel
                LNumStrA0=GiveStrAntToNum(self,StrA0)

                LNumFlaggedStations.append(LNumStrA0)

                for NumStrA0 in LNumStrA0:
                    LFlaggedStations.append(self.StationNames[NumStrA0])
                    # NumA0=np.where(np.array(self.StationNames)==NumStrA0)[0]
                    # stop
                    # print NumStrA0,NumA0
                    C0=(A0==NumStrA0)
                    C1=(A1==NumStrA0)
                    ind=np.where(C1|C0)[0]

                    MapOut[ind]=False

        if WriteAttribute:
            self.MapSelBLs=MapOut
            self.LFlaggedStations=list(set(LFlaggedStations))
            return self.MapSelBLs
        else:
            LNumFlaggedStations=sorted(list(set(range(self.na))-set(np.array(LNumFlaggedStations).flatten().tolist())))
            return LNumFlaggedStations
        


    # def GiveMappingAntOld(self,ListStrSel,(row0,row1)=(None,None),FlagAutoCorr=True):
    #     #ListStrSel=["RT9-RTA", "RTA-RTB", "RTC-RTD", "RT6-RT7", "RT5-RT*"]

    #     print ModColor.Str("  ... Building BL-mapping for %s"%str(ListStrSel))

    #     if row1 is None:
    #         row0=0
    #         row1=self.nbl
    #     A0=self.A0[row0:row1]
    #     A1=self.A1[row0:row1]
    #     MapOut=np.ones((self.nbl,),dtype=np.bool)
    #     if FlagAutoCorr:
    #         ind=np.where(A0==A1)[0]
    #         MapOut[ind]=False


    #     for blsel in ListStrSel:
    #         if "-" in blsel:
    #             StrA0,StrA1=blsel.split("-")
    #             NumA0=np.where(np.array(self.StationNames)==StrA0)[0]
    #             NumA1=np.where(np.array(self.StationNames)==StrA1)[0]
    #             C0=((A0==NumA0)&(A1==NumA1))
    #             C1=((A1==NumA0)&(A0==NumA1))
    #         else:
    #             NumA0=np.where(np.array(self.StationNames)==blsel)[0]
    #             C0=(A0==NumA0)
    #             C1=(A1==NumA0)
    #         ind=np.where(C1|C0)[0]
    #         MapOut[ind]=False
    #     self.MapSelBLs=MapOut
    #     return self.MapSelBLs
                



    # def SelChannel(self,(start,end,step)=(None,None,None),Revert=False):
    #     if start is not None:
    #         if Revert==False:
    #             ind=np.arange(self.Nchan)[start:end:step]
    #         else:
    #             ind=np.array(sorted(list(set(np.arange(self.Nchan).tolist())-set(np.arange(self.Nchan)[start:end:step].tolist()))))
    #         self.data=self.data[:,ind,:]
    #         self.flag_all=self.flag_all[:,ind,:]
    #         shape=self.ChanFreq.shape
    #         self.ChanFreq=self.ChanFreq[ind]
                
    def Give_dUVW_dt(self,ttVec,A0,A1,LongitudeDeg=6.8689,R="UVW_dt"):

        # tt=self.times_all[0]
        # A0=self.A0[self.times_all==tt]
        # A1=self.A1[self.times_all==tt]
        # uvw=self.uvw[self.times_all==tt]


        tt=np.mean(ttVec)
        ra,d=self.radec
        D=self.GiveDate(tt)
        
        #Lon=LongitudeDeg*np.pi/180
        Lon=np.arctan2(self.StationPos[:,1],self.StationPos[:,0]).mean()
        h= sidereal.raToHourAngle(ra, D, Lon)
        

        c=np.cos
        s=np.sin
        L=self.StationPos[A1]-self.StationPos[A0]

        if R=="UVW":
            R=np.array([[ s(h)      ,  c(h)      , 0.  ],
                        [-s(d)*c(h) ,  s(d)*s(h) , c(d)],
                        [ c(d)*c(h) , -c(d)*s(h) , s(d)]])
            UVW=np.dot(R,L.T).T
            import pylab
            pylab.clf()
            # pylab.subplot(1,2,1)
            # pylab.scatter(uvw[:,0],uvw[:,1],marker='.')
            #pylab.subplot(1,2,2)
            pylab.scatter(UVW[:,0],UVW[:,1],marker='.')
            pylab.draw()
            pylab.show(False)
            return UVW
        else:
        # stop
            K=2.*np.pi/(24.*3600)
            R_dt=np.array([[K*c(h)      , -K*s(h)     , 0.  ],
                           [K*s(d)*s(h) , K*s(d)*c(h) , 0.  ],
                           [-K*c(d)*s(h), -K*c(d)*c(h), 0.  ]])

            UVW_dt=np.dot(R_dt,L.T).T
            return np.float32(UVW_dt)

    def ReinitChunkIter(self):
        self.current_chunk = -1

    def getChunkCache (self, row0, row1):
        if (row0, row1) not in self._chunk_caches:
            self._chunk_caches[row0, row1] = CacheManager(
                self.MSName + ".ddfcache/F%d:D%d:%d:%d" % (self.Field, self.DDID, row0, row1), self._reset_cache)
        return self._chunk_caches[row0, row1]

    def GiveNextChunk(self,databuf=None,flagbuf=None,use_cache=None,read_data=True):
        # release data/flag arrays, if holding them, and mark cache as valid
        if self._datapath:
            self.cache.saveCache("Data")
        if self._flagpath:
            self.cache.saveCache("Flags")
        self._datapath = self._flagpath = None
        # get row0:row1 of next chunk. If row1==row0, chunk is empty and we must skip it
        while self.current_chunk < self.Nchunk-1:
            self.current_chunk += 1
            row0, row1 = self._chunk_r0r1[self.current_chunk]
            if row1 > row0:
                self.cache = self.getChunkCache(row0,row1)
                return self.ReadData(row0,row1,databuf=databuf,flagbuf=flagbuf,use_cache=use_cache,read_data=read_data)
        return "EndMS"

    def getChunkRow0Row1 (self):
        return self._chunk_r0r1
        
    def ReadData(self,row0,row1,DoPrint=False,ReadWeight=False,databuf=None,flagbuf=None,use_cache=False,read_data=True):
        """
        Args:
            row0:
            row1:
            DoPrint:
            ReadWeight:
            use_cache: if True, reads data and flags from the chunk cache, if available
            databuf: a buffer to read data into. If None, a new array is created.
            flagbuf: a buffer to read flags into. If None, a new array is created.
            read_data: if False, visibilities will not be read, only flags and other data

        Returns:

        """

        if row0>=self.F_nrows:
            return "EndMS"
        if row1>(self.F_nrows):
            row1=self.F_nrows
        
        self.ROW0 = row0
        self.ROW1 = row1
        self.nRowRead = nRowRead = row1-row0
        # expected data column shape
        datashape = (nRowRead, len(self.ChanFreq), len(self.CorrelationNames))

        strMS="%s"%(ModColor.Str(self.MSName, col="green"))
        print>>log, "%s: Reading next data chunk in [%i, %i] rows"%(strMS,row0,row1)

        table_all = self.GiveMainTable()
        #SPW=table_all.getcol('DATA_DESC_ID',row0,nRowRead)
        A0=table_all.getcol('ANTENNA1',row0,nRowRead)#[SPW==self.ListSPW[0]]
        A1=table_all.getcol('ANTENNA2',row0,nRowRead)#[SPW==self.ListSPW[0]]
        #print self.ListSPW[0]
        time_all=table_all.getcol("TIME",row0,nRowRead)#[SPW==self.ListSPW[0]]
        #print np.max(time_all)-np.min(time_all)
        #time_slots_all=np.array(sorted(list(set(time_all))))
        ntimes=time_all.shape[0]/self.nbl
        uvw = table_all.getcol('UVW', row0, nRowRead)

        # create data array (if databuf is None, array uses memory of buffer)
        visdata = np.ndarray(shape=datashape, dtype=np.complex64, buffer=databuf)
        if read_data:
            # check cache for visibilities
            if use_cache:
                datapath, datavalid = self.cache.checkCache("Data.npy", dict(time=self._start_time))
            else:
                datavalid = False
            # read from cache if available, else from MS
            if datavalid:
                print>> log, "reading cached visibilities from %s" % datapath
                visdata[...] = np.load(datapath)
            else:
                print>> log, "reading MS visibilities from column %s" % self.ColName
                table_all.getcolslicenp(self.ColName, visdata, self.cs_tlc, self.cs_brc, self.cs_inc, row0, nRowRead)
                if use_cache:
                    print>> log, "caching visibilities to %s" % datapath
                    np.save(datapath, visdata)
                    self.cache.saveCache("Data.npy")
        else:
            visdata.fill(0)
        # create flag array (if flagbuf is None, array uses memory of buffer)
        flags = np.ndarray(shape=datashape, dtype=np.bool, buffer=flagbuf)
        # check cache for flags
        if use_cache:
            flagpath, flagvalid = self.cache.checkCache("Flags.npy", dict(time=self._start_time))
        else:
            flagvalid = False
        # read from cache if available, else from MS
        if flagvalid:
            print>> log, "reading cached flags from %s" % flagpath
            flags[...] = np.load(flagpath)
        else:
            print>> log, "reading MS flags from column FLAG"
            table_all.getcolslicenp("FLAG", flags, self.cs_tlc, self.cs_brc, self.cs_inc, row0, nRowRead)
            self.UpdateFlags(flags, uvw, visdata, A0, A1, time_all)
            if use_cache:
                print>> log, "caching flags to %s" % flagpath
                np.save(flagpath, flags)
                self.cache.saveCache("Flags.npy")

        DATA={}

        DATA["data"] = visdata
        DATA["flags"] = flags
        # store paths to shared arrays (if any) in DATA dict
        DATA["flagpath"] = self._flagpath
        DATA["datapath"] = self._datapath

        if ReadWeight:
            self.Weights = table_all.getcol("WEIGHT",row0,nRowRead)

        table_all.close()

        DATA["uvw"]=uvw
        DATA["times"]=time_all
        DATA["nrows"]=time_all.shape[0]
        DATA["A0"]=A0
        DATA["A1"]=A1
        DATA["dt"]=self.dt
        DATA["dnu"]=self.ChanWidth

        if visdata is not None:
            if self.zero_flag: 
                visdata[flags] = 1e10
        # print "count",np.count_nonzero(flag_all),np.count_nonzero(np.isnan(vis_all))
            visdata[np.isnan(visdata)] = 0.
        # print "visMS",vis_all.min(),vis_all.max()

        # if self.AverageSteps is not None:
        #     StepTime,StepFreq=self.AverageSteps
        #     DATA=self.GiveAverageTimeFreq(DATA,StepTime=StepTime,StepFreq=StepFreq)

        return DATA
            

    def GiveAverageTimeFreq(self,DicoData,StepTime=None,StepFreq=None):
        DicoDataOut={}
        DicoDataOut["A0"]=DicoData["A0"]
        DicoDataOut["A1"]=DicoData["A1"]
        #DicoDataOut["nrows"]=DicoData["nrows"]
        DicoDataOut["uvw"]=DicoData["uvw"]

        if StepFreq is None:
            StepFreq=1

        if StepTime is None:
            StepTime=1
            
        NTimesIn=(DicoData["times"][-1]-DicoData["times"][0])/DicoData["dt"]

        NTimesOut=int(NTimesIn/StepTime)+1

        NChanMS=self.ChanFreqOrig.size
        NChanOut=NChanMS/StepFreq
        
        VisOut=np.zeros((NTimesOut,self.nbl,NChanOut,4),dtype=DicoData["data"].dtype)
        FlagOut=np.ones((NTimesOut,self.nbl,NChanOut,4),dtype=DicoData["flag"].dtype)
        NPointsOutChan=np.zeros((NTimesOut,self.nbl,NChanOut,4),dtype=np.int32)

        UVWOut=np.zeros((NTimesOut,self.nbl,3),dtype=DicoData["uvw"].dtype)
        TimeOut=np.zeros((NTimesOut,self.nbl),dtype=DicoData["times"].dtype)
        A0Out=np.zeros((NTimesOut,self.nbl),dtype=DicoData["A0"].dtype)
        A1Out=np.zeros((NTimesOut,self.nbl),dtype=DicoData["A0"].dtype)
        NPointsOut=np.zeros((NTimesOut,self.nbl),dtype=np.int32)

        blNum=np.zeros((self.na,self.na),dtype=np.int32)
        for itime in range(NTimesOut):
            ibl=0
            for iA0 in range(self.na):
                for iA1 in range(iA0,self.na):
                    A0Out[itime,ibl]=iA0
                    A1Out[itime,ibl]=iA1
                    blNum[iA0,iA1]=ibl
                    ibl+=1

        T0=DicoData["times"][0]
        
        dtOut=self.dt*StepTime
        nrow=DicoData["times"].size
        for irow in range(nrow):
            print irow,"/",nrow
            ThisTime=DicoData["times"][irow]
            iTimeOut=int((ThisTime-DicoData["times"][0])/dtOut)
            iA0=DicoData["A0"][irow]
            iA1=DicoData["A1"][irow]
            ibl=blNum[iA0,iA1]

            A0Out[iTimeOut,ibl]=iA0
            A1Out[iTimeOut,ibl]=iA1
            TimeOut[iTimeOut,ibl]+=ThisTime
            UVWOut[iTimeOut,ibl,:]+=DicoData["uvw"][irow,:]
            NPointsOut[iTimeOut,ibl]+=1

            for ichan in range(NChanMS):
                ichanOut=int(ichan/StepFreq)
                V4=DicoData["data"][irow,ichan,:].copy()
                F4=DicoData["flag"][irow,ichan,:].copy()
                AllFlagged=(np.count_nonzero(F4)==F4.size)
                if not(AllFlagged):
                    V4[F4==1]=0.
                    VisOut[iTimeOut,ibl,ichanOut,:]+=V4[:]
                    FlagOut[iTimeOut,ibl,ichanOut,:]=0
                    NPointsOutChan[iTimeOut,ibl,ichanOut,:]+=1

        VisOut/=NPointsOutChan
        
        TimeOut/=NPointsOut
        UVWOut/=NPointsOut.reshape((NTimesOut,self.nbl,1))


        NRowOut=NTimesOut*self.nbl
        DATA={}
        ind=np.where(TimeOut!=0.)[0]

        
        DATA["uvw"]=(UVWOut.reshape((NRowOut,3))[ind]).copy()
        DATA["times"]=(TimeOut.reshape((NRowOut,))[ind]).copy()
        DATA["nrows"]=TimeOut.shape[0]
        DATA["A0"]=(A0Out.reshape((NRowOut,))[ind]).copy()
        DATA["A1"]=(A1Out.reshape((NRowOut,))[ind]).copy()
        DATA["dt"]=self.dt*StepTime
        DATA["dnu"]=np.ones((NChanOut,),np.float32)*(self.ChanWidth[0])
        
        
        DATA["data"]=(VisOut.reshape((NRowOut,NChanOut,4))[ind]).copy()
        DATA["flags"]=(FlagOut.reshape((NRowOut,NChanOut,4))[ind]).copy()
        return DATA

    def SaveAllDataStruct(self):
        t=self.GiveMainTable(readonly=False)

        t.putcol('ANTENNA1',self.A0)
        t.putcol('ANTENNA2',self.A1)
        t.putcol("TIME",self.times_all)
        t.putcol("TIME_CENTROID",self.times_all)
        t.putcol("UVW",self.uvw)
        t.putcol("FLAG",self.flag_all)
        for icol in range(len(self.ColName)):
            t.putcol(self.ColName[icol],self.data[icol])
        t.close()

    def RemoveStation(self):
        
        DelStationList=self.DelStationList
        if DelStationList is None: return

        StationNames=self.StationNames
        self.MapStationsKeep=np.arange(len(StationNames))
        DelNumStationList=[]
        for Station in DelStationList:
            ind=np.where(Station==np.array(StationNames))[0]
            self.MapStationsKeep[ind]=-1
            DelNumStationList.append(ind)
            indRemove=np.where((self.A0!=ind)&(self.A1!=ind))[0]
            self.A0=self.A0[indRemove]
            self.A1=self.A1[indRemove]
            self.data=self.data[indRemove,:,:]
            self.flag_all=self.flag_all[indRemove,:,:]
            self.times_all=self.times_all[indRemove,:,:]
        self.MapStationsKeep=self.MapStationsKeep[self.MapStationsKeep!=-1]
        StationNames=(np.array(StationNames)[self.MapStationsKeep]).tolist()

        na=self.MapStationsKeep.shape[0]
        self.na=na
        self.StationPos=self.StationPos[self.MapStationsKeep,:]
        self.nbl=(na*(na-1))/2+na
        

    def ReadMSInfo(self,DoPrint=True):
        T= ClassTimeIt.ClassTimeIt()
        T.enableIncr()
        T.disable()

        # open main table
        table_all = self.GiveMainTable()
        if not table_all.nrows():
            raise RuntimeError,"no rows in MS %s, check your Field/DDID/TaQL settings"%(self.MSName)

        #print MSname+'/ANTENNA'
        ta=table(table_all.getkeyword('ANTENNA'),ack=False)

        StationNames=ta.getcol('NAME')

        na=ta.getcol('POSITION').shape[0]
        self.StationPos=ta.getcol('POSITION')
        nbl=(na*(na-1))/2+na
        #nbl=(na*(na-1))/2
        ta.close()
        T.timeit()

        # get spectral window and polarization id
        ta_ddid = table(table_all.getkeyword('DATA_DESCRIPTION'),ack=False)
        self._spwid = ta_ddid.getcol("SPECTRAL_WINDOW_ID")[self.DDID]
        self._polid = ta_ddid.getcol("POLARIZATION_ID")[self.DDID]
        ta_ddid.close()


        # get polarizations
        # This a list of the Stokes enums (as defined in casacore header measures/Stokes.h)
        # These are referenced by the CORR_TYPE column of the MS POLARIZATION subtable.
        # E.g. 5,6,7,8 corresponds to RR,RL,LR,LL
        MS_STOKES_ENUMS = [
            "Undefined", "I", "Q", "U", "V", "RR", "RL", "LR", "LL", "XX", "XY", "YX", "YY", "RX", "RY", "LX", "LY", "XR", "XL", "YR", "YL", "PP", "PQ", "QP", "QQ", "RCircular", "LCircular", "Linear", "Ptotal", "Plinear", "PFtotal", "PFlinear", "Pangle"
          ]
        tp = table(table_all.getkeyword('POLARIZATION'),ack=False)
        # get list of corrype enums for first row of polarization table, and convert to strings via MS_STOKES_ENUMS. 
        # self.CorrelationNames will be a list of strings
        self.CorrelationIds = tp.getcol('CORR_TYPE',0,1)[self._polid]
        self.CorrelationNames = [ (ctype >= 0 and ctype < len(MS_STOKES_ENUMS) and MS_STOKES_ENUMS[ctype]) or
                None for ctype in self.CorrelationIds ]
        self.Ncorr = len(self.CorrelationNames)
        # NB: it is possible for the MS to have different polarization

        self.ColNames=table_all.colnames()
        self.F_nrows=table_all.nrows()#-nbl

        # make mapping into chunks
        if not self.TimeChunkSize:
            T0=table_all.getcol('TIME',0,1)[0]
            T1=table_all.getcol('TIME',self.F_nrows-1,1)[0]
            print>>log,"ChunkHours is not set; MS %s (%d rows) will be processed as a single chunk"%(self.MSName, self.F_nrows)
            chunk_rows = [0]
        else:
            all_times = table_all.getcol("TIME")
            if (all_times[1:] - all_times[:-1]).min() < 0:
                raise RuntimeError,"MS %s: the TIME column must be in increasing order"%self.MSName
            T0, T1 = all_times[0], all_times[-1]
            chunk_t0 = np.arange(T0, T1, self.TimeChunkSize*3600)
            # chunk_t0 now gives starting time of each chunk
            chunk_row0 = [ np.argmax(all_times>=ch_t0) for ch_t0 in chunk_t0 ]
            # chunk_row0 gives the starting row of each chunk
            if len(chunk_row0) == 1:
                print>>log,"MS %s (%d rows) will be processed as a single chunk"%(self.MSName, self.F_nrows)
            else:
                print>>log,"MS %s (%d rows) will be split into %d chunks, at rows %s"%(self.MSName, self.F_nrows,
                                                                                       len(chunk_row0), " ".join(map(str,chunk_row0)))
        self.Nchunk = len(chunk_row0)
        chunk_row0.append(self.F_nrows)
        self._chunk_r0r1 = [ chunk_row0[i:i+2] for i in range(self.Nchunk) ]

        #SPW=table_all.getcol('DATA_DESC_ID')
        # if self.SelectSPW is not None:
        #     self.ListSPW=self.SelectSPW
        #     #print "dosel"
        # else:
        #     self.ListSPW=sorted(list(set(SPW.tolist())))
        # T.timeit()

        #self.F_nrows=table_all.getcol("TIME").shape[0]
        #F_time_all=table_all.getcol("TIME")[SPW==self.ListSPW[0]]

        #self.F_A0=table_all.getcol("ANTENNA1")[SPW==self.ListSPW[0]]
        #self.F_A1=table_all.getcol("ANTENNA2")[SPW==self.ListSPW[0]]

        #nbl=(np.where(F_time_all==F_time_all[0])[0]).shape[0]
        T.timeit()

        #F_time_slots_all=np.array(sorted(list(set(F_time_all.tolist()))))
        #F_ntimes=F_time_slots_all.shape[0]
        dt=table_all.getcol('INTERVAL',0,1)[0]

        T.timeit()

        ta_spectral=table(table_all.getkeyword('SPECTRAL_WINDOW'),ack=False)
        NSPW = ta_spectral.nrows()
        reffreq=ta_spectral.getcol('REF_FREQUENCY')[self._spwid]
        orig_freq = ta_spectral.getcol('CHAN_FREQ')[self._spwid,:] 
        chan_freq = orig_freq[self.ChanSlice]


        self.dFreq=ta_spectral.getcol("CHAN_WIDTH")[self._spwid,self.ChanSlice].flatten()[0]
        self.ChanWidth=ta_spectral.getcol('CHAN_WIDTH')[self._spwid,self.ChanSlice]

        # if chan_freq.shape[0]>len(self.ListSPW):
        #     print ModColor.Str("  ====================== >> More SPW in headers, modifying that error....")
        #     chan_freq=chan_freq[np.array(self.ListSPW),:]
        #     reffreq=reffreq[np.array(self.ListSPW)]
            

        T.timeit()

        wavelength=299792458./reffreq
        self.ChanFreq=chan_freq
        self.ChanFreqOrig=self.ChanFreq.copy()
        self.Freq_Mean=np.mean(chan_freq)
        wavelength_chan=299792458./chan_freq

        if NSPW>1:
            print "Don't deal with multiple SPW yet"

        self.Nchan = Nchan = len(wavelength_chan)
        NSPWChan=NSPW*Nchan

        # set up cs_tlc,cd_brc,cs_inc: these are pyrap-style slice selection arguments
        # to select the [subset] of the column
        if self.ChanSlice is not None:
            chan_nums = range(len(orig_freq))[self.ChanSlice]
            self.cs_tlc = (chan_nums[0], 0)
            self.cs_brc = (chan_nums[-1], self.Ncorr - 1)
            self.cs_inc = (self.ChanSlice.step or 1, 1)
        else:
            self.cs_tlc = (0, 0)
            self.cs_brc = (self.Nchan - 1, self.Ncorr - 1)
            self.cs_inc = (1, 1)

        ta=table(table_all.getkeyword('FIELD'),ack=False)
        rarad,decrad=ta.getcol('PHASE_DIR')[self.Field][0]
        if rarad<0.: rarad+=2.*np.pi

        T.timeit()

        self.radeg=rarad*180./np.pi
        self.decdeg=decrad*180./np.pi
        ta.close()
         
        self.DoRevertChans=False
        if Nchan>1:
            self.DoRevertChans=(self.ChanFreq.flatten()[0]>self.ChanFreq.flatten()[-1])
        if self.DoRevertChans:
            print ModColor.Str("  ====================== >> Revert Channel order!")
            wavelength_chan=wavelength_chan[0,::-1]
            self.ChanFreq=self.ChanFreq[0,::-1]
            self.dFreq=np.abs(self.dFreq)

        # if self.AverageSteps is not None:
        #     _,StepFreq=self.AverageSteps
        #     NChanOut=self.ChanFreq.size/StepFreq
        #     NChanMS=self.ChanFreq.size
        #     if (self.ChanFreq.size%StepFreq!=0):
        #         raise NameError('Number of channels should be a multiple of d_channels')
        #     ChanFreq=np.zeros((NChanOut,),self.ChanFreq.dtype)
        #     ChanFreqPoint=np.zeros((NChanOut,),self.ChanFreq.dtype)
        #     for ichan in range(NChanMS):
        #         ichanOut=int(ichan/StepFreq)
        #         ChanFreq[ichanOut]+=self.ChanFreq.ravel()[ichan]
        #         ChanFreqPoint+=1
        #     ChanFreq/=ChanFreqPoint
        #     self.ChanFreq=ChanFreq.reshape((1,NChanOut))
        #     self.dFreq*=StepFreq
        #     chan_freq=self.ChanFreq


        T.timeit()

        self.na=na
        self.Nchan=Nchan
        self.NSPW=NSPW
        self.NSPWChan=NSPWChan
        self.F_tstart=T0
        #self.F_times_all=T1
        #self.F_times=F_time_slots_all
        #self.F_ntimes=F_time_slots_all.shape[0]
        self.dt=dt
        self.DTs=T1-T0
        self.DTh=self.DTs/3600.
        self.radec=(rarad,decrad)
        self.rarad=rarad
        self.decrad=decrad
        self.reffreq=reffreq
        self.StationNames=StationNames
        self.wavelength_chan=wavelength_chan
        self.rac=rarad
        self.decc=decrad
        self.nbl=nbl
        self.StrRA  = rad2hmsdms(self.rarad,Type="ra").replace(" ",":")
        self.StrDEC = rad2hmsdms(self.decrad,Type="dec").replace(" ",".")

        table_all.close()
        T.timeit()
        # self.StrRADEC=(rad2hmsdms(self.rarad,Type="ra").replace(" ",":")\
        #                ,rad2hmsdms(self.decrad,Type="dec").replace(" ","."))


    def UpdateFlags(self, flags, uvw, data, A0, A1, times):
        """Updates a flag array in place (flagging stuff)
        by applying various selection criteria from self.DicoSelectOptions.
        Also sets flagged data to 1e9.
        """
        print>> log, "Updating flags"

        ThresholdFlag = 0.9 # flag antennas with % of flags over threshold
        FlagAntNumber = []  # accumulates list of flagged antennas

        for A in range(self.na):
            ind = np.where((A0 == A) | (A1 == A))[0]
            fA = flags[ind].ravel()
            if ind.size == 0: continue
            nf = np.count_nonzero(fA)
            Frac = nf / float(fA.size)
            if Frac > ThresholdFlag:
                print>> log, "  Flagging antenna %i has ~%4.1f%s of flagged data (more than %4.1f%s)" % \
                             (A, Frac * 100, "%", ThresholdFlag * 100, "%")
                FlagAntNumber.append(A)

        if self.DicoSelectOptions["UVRangeKm"] != None:
            d0, d1 = self.DicoSelectOptions["UVRangeKm"]
            print>> log, "  Flagging uv data outside uv distance of [%5.1f~%5.1f] km" % (d0, d1)
            d0 *= 1e3
            d1 *= 1e3
            u, v, w = uvw.T
            duv = np.sqrt(u ** 2 + v ** 2)
            ind = np.where(((duv > d0) & (duv < d1)) != True)[0]
            flags[ind, :, :] = True

        if self.DicoSelectOptions["TimeRange"] != None:
            t0 = times[0]
            tt = (times - t0) / 3600.
            st0, st1 = self.DicoSelectOptions["TimeRange"]
            print>> log, "  Selecting uv data in time range [%.4f~%5.4f] hours" % (st0, st1)
            ind = np.where((tt >= st0) & (tt < st1))[0]
            flags[ind, :, :] = True

        if self.DicoSelectOptions["FlagAnts"] != None:
            FlagAnts = self.DicoSelectOptions["FlagAnts"]
            if not ((FlagAnts == None) | (FlagAnts == "") | (FlagAnts == [])):
                if type(FlagAnts) == str: FlagAnts = [FlagAnts]
                for Name in FlagAnts:
                    for iAnt in range(self.na):
                        if Name in self.StationNames[iAnt]:
                            print>> log, "  Flagging antenna #%2.2i[%s]" % (iAnt, self.StationNames[iAnt])
                            FlagAntNumber.append(iAnt)

        if self.DicoSelectOptions["DistMaxToCore"] != None:
            DMax = self.DicoSelectOptions["DistMaxToCore"] * 1e3
            X, Y, Z = self.StationPos.T
            Xm, Ym, Zm = np.median(self.StationPos, axis=0).flatten().tolist()
            Dist = np.sqrt((X - Xm) ** 2 + (Y - Ym) ** 2 + (Z - Zm) ** 2)
            ind = np.where(Dist > DMax)[0]
            for iAnt in ind.tolist():
                print>> log, "  Flagging antenna #%2.2i[%s] (distance to core: %.1f km)" % (
                iAnt, self.StationNames[iAnt], Dist[iAnt] / 1e3)
                FlagAntNumber.append(iAnt)

        for A in FlagAntNumber:
            ind = np.where((A0 == A) | (A1 == A))[0]
            flags[ind, :, :] = True
        # flag autocorrelations
        ind = np.where(A0 == A1)[0]
        flags[ind, :, :] = True
        # if one of 4 correlations is flagged, flag all 4
        ind = np.any(flags, axis=2)
        flags[ind] = True
        # flag NaNs
        if data is not None:
            ind = np.where(np.isnan(data))
            flags[ind] = 1
            data[flags] = 1e9

    def __str__(self):
        ll=[]
        ll.append(ModColor.Str(" MS PROPERTIES: "))
        ll.append("   - File Name: %s" % ModColor.Str(self.MSName, col="green"))
        ll.append("   - Column Name: %s" % ModColor.Str(str(self.ColName), col="green"))
        ll.append("   - Selection: %s, channels: %s" % (ModColor.Str(str(self.TaQL), col="green"), self.ChanSlice))
        ll.append("   - Phase centre (field %d): (ra, dec)=(%s, %s) "%(self.Field, rad2hmsdms(self.rarad,Type="ra").replace(" ",":")\
                                                                       ,rad2hmsdms(self.decrad,Type="dec").replace(" ",".")))
        ll.append("   - Frequency = %s MHz"%str(self.reffreq/1e6))
        ll.append("   - Wavelength = %5.2f meters"%(np.mean(self.wavelength_chan)))
        ll.append("   - Time bin = %4.1f seconds"%(self.dt))
        ll.append("   - Total Integration time = %6.2f hours"%self.DTh)
        ll.append("   - Number of antenna  = %i"%self.na)
        ll.append("   - Number of baseline = %i"%self.nbl)
        ll.append("   - Number of SPW = %i"%self.NSPW)
        ll.append("   - Number of channels = %i"%self.Nchan)
        ll.append("   - Number of time chunks = %i"%self.Nchunk)

        ss="\n".join(ll)+"\n"
        return ss

    def radec2lm_scalar(self,ra,dec):
        l = np.cos(dec) * np.sin(ra - self.rarad)
        m = np.sin(dec) * np.cos(self.decrad) - np.cos(dec) * np.sin(self.decrad) * np.cos(ra - self.rarad)
        return l,m

    def PutVisColumn (self,colname,vis):
        self.AddCol(colname,quiet=True)
        print>>log,"writing column %s rows %d:%d"%(colname,self.ROW0,self.ROW1-1)
        t = self.GiveMainTable(readonly=False,ack=False)
        if self.ChanSlice and self.ChanSlice != slice(None):
            # if getcol fails, maybe because this is a new col which hasn't been filled
            # in this case read DATA instead
            try:
                vis0 = t.getcol(colname,self.ROW0,self.nRowRead)
            except RuntimeError:
                vis0 = t.getcol("DATA",self.ROW0,self.nRowRead)
            vis0[:,self.ChanSlice,:] = vis
            t.putcol(colname,vis0,self.ROW0,self.nRowRead)
        else:
            t.putcol(colname,vis,self.ROW0,self.nRowRead)
        t.close()

    def SaveVis(self,vis=None,Col="CORRECTED_DATA",spw=0,DoPrint=True):
        if vis is None:
            vis=self.data
        if DoPrint: print>>log, "Writting data in column %s" % ModColor.Str(Col, col="green")
        table_all=self.GiveMainTable(readonly=False)

        if self.swapped:
            visout=np.swapaxes(vis[spw*self.Nchan:(spw+1)*self.Nchan],0,1)
            flag_all=np.swapaxes(self.flag_all[spw*self.Nchan:(spw+1)*self.Nchan],0,1)
        else:
            visout=vis
            flag_all=self.flag_all

        table_all.putcol(Col,visout.astype(self.data.dtype),self.ROW0,self.nRowRead)
        table_all.putcol("FLAG",flag_all,self.ROW0,self.nRowRead)
        table_all.close()
        
    def GiveUvwBL(self,a0,a1):
        vecout=self.uvw[(self.A0==a0)&(self.A1==a1),:]
        return vecout

    def GiveVisBL(self,a0,a1,col=0,pol=None):
        if self.multidata:
            vecout=self.data[col][(self.A0==a0)&(self.A1==a1),:,:]
        else:
            vecout=self.data[(self.A0==a0)&(self.A1==a1),:,:]
        if pol is not None:
            vecout=vecout[:,:,pol]
        return vecout

    def GiveVisBLChan(self,a0,a1,chan,pol=None):
        if pol is None:
            vecout=(self.data[(self.A0==a0)&(self.A1==a1),chan,0]+self.data[(self.A0==a0)&(self.A1==a1),chan,3])/2.
        else:
            vecout=self.data[(self.A0==a0)&(self.A1==a1),chan,pol]
        return vecout

    def plotBL(self,a0,a1,pol=0):
        
        import pylab
        if self.multidata:
            vis0=self.GiveVisBL(a0,a1,col=0,pol=pol)
            vis1=self.GiveVisBL(a0,a1,col=1,pol=pol)
            pylab.clf()
            pylab.subplot(2,1,1)
            #pylab.plot(vis0.real)
            pylab.plot(np.abs(vis0))
            #pylab.subplot(2,1,2)
            #pylab.plot(vis1.real)
            pylab.plot(np.abs(vis1),ls=":")
            pylab.title("%i-%i"%(a0,a1))
            #pylab.plot(vis1.real-vis0.real)
            pylab.subplot(2,1,2)
            pylab.plot(np.angle(vis0))
            pylab.plot(np.angle(vis1),ls=":")
            pylab.draw()
            pylab.show()
        else:
            pylab.clf()
            vis=self.GiveVisBL(a0,a1,col=0,pol=pol)
            pylab.subplot(2,1,1)
            pylab.plot(np.abs(vis))
            #pylab.plot(np.real(vis))
            pylab.subplot(2,1,2)
            pylab.plot(np.angle(vis))
            #pylab.plot(np.imag(vis))
            pylab.draw()
            pylab.show()

    def GiveCol(self,ColName):
        t=self.GiveMainTable(readonly=False)
        col=t.getcol(ColName)
        t.close()
        return col

    def PutColInData(self,SpwChan,pol,data):
        if self.swapped:
            self.data[SpwChan,:,pol]=data
        else:
            self.data[:,SpwChan,pol]=data

    def Restore(self):
        backname="CORRECTED_DATA_BACKUP"
        backnameFlag="FLAG_BACKUP"
        t=table(self.MSName,readonly=False,ack=False)
        if backname in t.colnames():
            print>>log, "  Copying ",backname," to CORRECTED_DATA"
            #t.putcol("CORRECTED_DATA",t.getcol(backname))
            self.CopyCol(backname,"CORRECTED_DATA")
            print>>log, "  Copying ",backnameFlag," to FLAG"
            self.CopyCol(backnameFlag,"FLAG")
            #t.putcol(,t.getcol(backnameFlag))
        t.close()

    def ZeroFlagSave(self,spw=0):
        self.flag_all.fill(0)
        if self.swapped:
            flagout=np.swapaxes(self.flag_all[spw*self.Nchan:(spw+1)*self.Nchan],0,1)
        else:
            flagout=self.flag_all
        t=self.GiveMainTable(readonly=False)
        t.putcol("FLAG",flagout)
        
        t.close()

    def CopyCol(self,Colin,Colout):
        t=table(self.MSName,readonly=False,ack=False)
        if self.TimeChunkSize is None:
            print>>log, "  ... Copying column %s to %s"%(Colin,Colout)
            t.putcol(Colout,t.getcol(Colin))
        else:
            print>>log, "  ... Copying column %s to %s"%(Colin,Colout)
            TimesInt=np.arange(0,self.DTh,self.TimeChunkSize).tolist()
            if not(self.DTh in TimesInt): TimesInt.append(self.DTh)
            for i in range(len(TimesInt)-1):
                t0,t1=TimesInt[i],TimesInt[i+1]
                print>>log, "      ... Copy in [%5.2f,%5.2f] hours"%( t0,t1)
                t0=t0*3600.+self.F_tstart
                t1=t1*3600.+self.F_tstart
                ind0=np.argmin(np.abs(t0-self.F_times))
                ind1=np.argmin(np.abs(t1-self.F_times))
                row0=ind0*self.nbl
                row1=ind1*self.nbl
                NRow=row1-row0
                t.putcol(Colout,t.getcol(Colin,row0,NRow),row0,NRow)
        t.close()

    def AddCol(self,ColName,LikeCol="DATA",quiet=False):
        t=table(self.MSName,readonly=False,ack=False)
        if (ColName in t.colnames() and not self.GD["Images"]["AllowColumnOverwrite"]):
            if not quiet:
                print>>log, "  Column %s already in %s"%(ColName,self.MSName)
            t.close()
            return
        elif (ColName in t.colnames() and self.GD["Images"]["AllowColumnOverwrite"]):
            t.removecols(ColName)
        print>>log, "  Putting column %s in %s"%(ColName,self.MSName)
        desc=t.getcoldesc(LikeCol)
        desc["name"]=ColName
        desc['comment']=desc['comment'].replace(" ","_")
        t.addcols(desc)
        t.close()
        
    def PutBackupCol(self,incol="CORRECTED_DATA"):
        backname="%s_BACKUP"%incol
        backnameFlag="FLAG_BACKUP"
        self.PutCasaCols()
        t=table(self.MSName,readonly=False,ack=False)
        JustAdded=False
        if not(backname in t.colnames()):
            print>>log, "  Putting column ",backname," in MS"
            desc=t.getcoldesc("CORRECTED_DATA")
            desc["name"]=backname
            desc['comment']=desc['comment'].replace(" ","_")
            t.addcols(desc)
            print>>log, "  Copying %s in %s"%(incol,backname)
            self.CopyCol(incol,backname)
        else:
            print>>log, "  Column %s already there"%(backname)

        if not(backnameFlag in t.colnames()):
            desc=t.getcoldesc("FLAG")
            desc["name"]=backnameFlag
            desc['comment']=desc['comment'].replace(" ","_")
            t.addcols(desc)
            self.CopyCol("FLAG",backnameFlag)

            JustAdded=True

        t.close()
        return JustAdded

    def PutNewCol(self,Name,LikeCol="CORRECTED_DATA"):
        if not(Name in self.ColNames):
            print>>log, "  Putting column %s in MS, with format of %s"%(Name,LikeCol)
            t=table(self.MSName,readonly=False,ack=False)
            desc=t.getcoldesc(LikeCol)
            desc["name"]=Name
            t.addcols(desc) 
            t.close()
    
    def RotateMS(self,radec):
        import ModRotate
        ModRotate.Rotate(self,radec)
        ta=table(self.MSName+'/FIELD/',ack=False,readonly=False)
        ra,dec=radec
        radec=np.array([[[ra,dec]]])
        ta.putcol("DELAY_DIR",radec)
        ta.putcol("PHASE_DIR",radec)
        ta.putcol("REFERENCE_DIR",radec)
        ta.close()
        t=self.GiveMainTable(readonly=False)
        t.putcol(self.ColName,self.data)
        # t.putcol("UVW",self.uvw) ??? self.uvw not defined
        t.close()
    
    def PutCasaCols(self):
        import pyrap.tables
        pyrap.tables.addImagingColumns(self.MSName,ack=False)
        #self.PutNewCol("CORRECTED_DATA")
        #self.PutNewCol("MODEL_DATA")
