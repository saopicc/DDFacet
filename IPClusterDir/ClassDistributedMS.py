import numpy as np
import PrintRecArray
from CheckJob import LaunchAndCheck
import ModColor
import MyLogger
log=MyLogger.getLogger("ClassDistributedMS")


class ClassDistributedSinglePointingMS():
    def __init__(self,E,PointingID):
        self.E=E
        self.GD=self.E.GD
        self.PointingID=PointingID
        self.MaskFreqName="MaskFreqMS_P%s"%self.PointingID
        self.MDC=None
        self.MaskHDF5=None
        self.updateClusterInfo()
        self.isInitBeam=False
        self.ReplaceMissingFreqs=self.GD.DicoConfig["Files"]["ReplaceMissingFreqs"]

    def getLocalMS(self):
        irc=self.ids[0]
        #r=self.E.rc[irc].execute("import copy; MSnodata=copy.deepcopy(MS); MSnodata.DelData()"); r.wait(); print r.get()
        LaunchAndCheck(self.E.rc[irc],"import copy; MSnodata=copy.deepcopy(MS); MSnodata.DelData()")
        self.MS=self.E.rc[irc].get("MSnodata")
        DicoConfig=self.E.GD.DicoConfig
        if DicoConfig["Select"]["FlagAntBL"]!=None:
            BLSel=DicoConfig["Select"]["FlagAntBL"]#.replace(" ","").split(',')
        else:
            BLSel=[]

        self.MapSelBLs=self.MS.GiveMappingAnt(BLSel)



    def updateClusterInfo(self,MSMounted=True):
        self.SubCluster=self.E.GiveSubCluster("Data",self.PointingID,MSMounted=MSMounted)
        self.Cat=self.SubCluster["Cat"]
        self.ids=self.SubCluster["ids"]
        self.V=self.SubCluster["V"]
        self.MaskFreqMS=self.E.getData(self.MaskFreqName)
        if not(self.MDC is None):
            self.MaskHDF5=self.MDC.getCurrentData("MaskHDF5_Pointing%i"%self.PointingID)


    def SendSM(self,SM):
        self.updateClusterInfo()
        self.V["SM"]=SM

    def updateMaskFreqs(self):
        PointingID=self.PointingID
        freqs=self.freqs
        MaskFreqMS=np.ones((freqs.shape[0],),bool)
        SubCluster=self.E.GiveSubCluster("Data",PointingID,MSMounted=True,available=True)
        Cat=SubCluster["Cat"]


        freqMask=Cat.Freq
        # for i in range(freqMask.shape[0]):
        #     ind=np.where(np.abs(np.mean(freqs,axis=1)-freqMask[i])<.01e6)[0]
        #     MaskFreqMS[ind]=False
        # self.E.setData(self.MaskFreqName,MaskFreqMS)

        freqsDispo=Cat.Freq

        FreqMean=np.mean(freqs,axis=1)
        DF=FreqMean.reshape(FreqMean.size,1)-freqsDispo.reshape(1,freqsDispo.size)
        MaskFreqMS=(np.min(np.abs(DF),axis=1)>0.01)


        self.E.setData(self.MaskFreqName,MaskFreqMS)
        


        
        print>>log, ("MS available:",freqMask.size)
        #print "MaskFreq size:",MaskFreq.shape
        #print "MaskFreq number falses:",np.where(MaskFreq==False)[0].size
        print>>log, ("MaskFreqMS size:",MaskFreqMS.shape)
        print>>log, ("MaskFreqMS number falses:",np.where(MaskFreqMS==False)[0].size)
        #print ""

    # def ReadData(self,t0=0,t1=-1):
    #     self.updateClusterInfo(MSMounted=False)
    #     for iCat in range(self.Cat.shape[0]):
    #         irc=self.ids[iCat]
    #         LaunchAndCheck(self.E.rc[irc],"t0=%f; t1=%f"%(t0,t1))
    #     strExec="MS.ReadData(t0=self.TimesInt[iT0],t1=self.TimesInt[iT1])"
    #     LaunchAndCheck(self.V,strExec,Progress=True)


    def LoadMS(self):
        self.updateClusterInfo(MSMounted=False)

        # declare MSNames
        #self.V.execute("import IPClusterDir.ClassVisServer")
        self.V["VS"]=True
        self.V["MS"]=True
        self.V["GD"]=self.GD
        self.V["Fail"]=False
        for iCat in range(self.Cat.shape[0]):
            irc=self.ids[iCat]
            #LaunchAndCheck(self.E.rc[irc],"NameMS='%s'; t0=%f; t1=%f"%(self.Cat.dirMSName[iCat],t0,t1))
            LaunchAndCheck(self.E.rc[irc],"NameMS='%s'"%(self.Cat.dirMSName[iCat]))
            
            

            


        #self.V["ColName"]=self.E.GD.DicoConfig["Files"]["ColName"]
        #strExec="%s/HyperCal/Scripts/ScriptLoadData.py"%self.E.GD.HYPERCAL_DIR
        strExec="%s/Scripts/ScriptDeclareVisServer.py"%self.E.GD.HYPERCAL_DIR
        #r=self.V.execute('execfile("%s")'%strExec); r.wait(); print r.get()
        LaunchAndCheck(self.V,'execfile("%s")'%strExec,Progress=True)
        nMS=self.Cat.shape[0]

        Fails=self.V["Fail"]
        freq=np.array(self.V["freq"])
        self.freqOrderEngines=freq.copy()
        ChanFreq=np.array(self.V["ChanFreq"])
        for (iband,DidFail) in zip(range(nMS),Fails):
            irc=self.ids[iband]
            if DidFail:
                print>>log, ModColor.Str("   --------> could not find %s"%self.E.GD.CatEngines.dirMSName[irc])
                continue
            
            self.nbl=self.V.get("nbl")[iband]
            self.na=self.V.get("na")[iband]
            self.E.GD.CatEngines.Freq[irc]=freq[iband]
            self.E.GD.CatEngines.NChan[irc]=ChanFreq[iband].size
            self.NChan=ChanFreq[iband].size
            ThisMSChanFreqs=ChanFreq[iband].flatten()
            self.E.GD.CatEngines.ChanFreq[irc][0:ThisMSChanFreqs.size]=ThisMSChanFreqs
            self.E.GD.CatEngines.MSMounted[irc]=True
            

        self.updateClusterInfo()

        #rr=self.V.execute("f=MS.ChanFreq.tolist()"); rr.wait(); print "rr.get()",rr.get()
        LaunchAndCheck(self.V,"f=MS.ChanFreq.tolist()")
        freqsDispo=np.array(sorted(np.array(self.V.get("f")).flatten().tolist()))
        #freqsDispo=ChanFreq.flatten()

        if freqsDispo.size>1:
            dfreq=np.median(freqsDispo[1::]-freqsDispo[0:-1])
            ind=np.where(freqsDispo>0)[0]
            ifStart=ind[0]
            fStart=freqsDispo[ifStart]-ifStart*dfreq
            nFreq=freqsDispo.size
            nFreq=np.round((np.max(freqsDispo)-fStart)/dfreq)+1
            freqs=np.arange(fStart,fStart+(nFreq-1)*dfreq+0.01,dfreq)
            
            Nchan=np.max(self.NChan)

            Nbands=freqs.size/self.NChan
            
            self.freqs=freqs
            self.dfreq=dfreq
            self.freqsDispo=freqsDispo

            if self.ReplaceMissingFreqs:
                self.freqs=self.freqs.reshape((Nbands,self.NChan))
                self.updateMaskFreqs()
                self.DoMaskFreq=True
            else:
                self.freqs=self.freqsDispo
                self.DoMaskFreq=False
                



            #MaskFreqMS=np.zeros(self.freqs.size,bool)
            #MaskFreqMS[np.array(Fails)]=True
            #self.indMSAvailable=np.where(self.MaskFreqMS==False)[0].tolist()

            #DF=self.freqs.reshape(self.freqs.size,1)-freqsDispo.reshape(1,self.freqsDispo)
            #MaskFreqMS=np.min(np.abs(DF),axis=1)<0.01
            #self.E.setData(self.MaskFreqName,MaskFreqMS)
            self.updateClusterInfo()
        else:
            self.freqs=freqsDispo
            self.DoMaskFreq=False

        self.getLocalMS()

        # ChanSel
        # if ChanSel!=None:
        #     rr=self.rc[engsel].execute("MSdata.SelChannel(%s)"%str(self.ChanSel)); rr.wait()
                     






    def GiveViSTchunk(self,it0=0,it1=-1,GiveMapBLSel=False):
        import ClassTimeIt
        self.updateClusterInfo()
        T=ClassTimeIt.ClassTimeIt("IPCluster.GiveViSTchunk")

        MS=self.MS

        r=self.V.execute("row0=%i*MS.nbl; row1=%i*MS.nbl; "%(it0,it1)); r.wait()
        r=self.V.execute("if row1<0: row1=None"); r.wait()
        strExec="%s/Scripts/ScriptGiveData.py"%self.E.GD.HYPERCAL_DIR
        #r=self.V.execute('execfile("%s")'%strExec); r.wait(); print r.get()
        LaunchAndCheck(self.V,'execfile("%s")'%strExec)
        T.timeit("ScriptGiveData")
        lout=self.V.get("VisC")
        T.timeit("get")
        null=np.zeros_like(lout[0])
        onesBool=np.ones(lout[0].shape,dtype=bool)

        freqOrderEngines=self.V["freq"]
        indf=np.argsort(freqOrderEngines)
        lout=[lout[i] for i in indf]


        if self.DoMaskFreq:
            for i in range(self.MaskFreqMS.shape[0]):
                if self.MaskFreqMS[i]:
                    lout.insert(i,null)

        T.timeit("insert zeros")
        GiveAnt=True
        GiveFlag=True

        out=np.array(lout)
        del lout
        out=np.swapaxes(out,1,2)
        T.timeit("swap axes0")

        NtimeBlocks=out.shape[2]/self.nbl
        
        out=out.reshape((self.freqs.size,out.shape[2]/self.nbl,self.nbl,4))
        T.timeit("reshape0")
        ROW0=self.V["ROW0"][0]
        ROW1=self.V["ROW1"][0]
        BLSel=None#self.MapSelBLs

        if BLSel!=None: 
            out=out[:,:,BLSel,:]
        T.timeit("blsel")
        nblSel=out.shape[2]
        data=out.reshape((self.freqs.size,NtimeBlocks*nblSel,4))
        T.timeit("reshape1")
        data=np.swapaxes(data,0,1)

        T.timeit("swap axes1")

        flagList=None
        flags=None
        if GiveFlag:
            flagList=self.V["flag"]

            flagList=[flagList[i] for i in indf]

            null=np.zeros_like(flagList[0])

            if self.DoMaskFreq:
                for i in range(self.MaskFreqMS.shape[0]):
                    #print i,self.MaskFreqMS[i]
                    if self.MaskFreqMS[i]:
                        #flagList.insert(i,null)
                        flagList.insert(i,onesBool)

            flagList=np.swapaxes(flagList,1,2)

            flagList=flagList.reshape((self.freqs.size,flagList.shape[2]/self.nbl,self.nbl,4))
            if BLSel!=None:
                flagList=flagList[:,:,BLSel,:]

            flags=flagList.reshape((self.freqs.size,NtimeBlocks*nblSel,4))
            flags=np.swapaxes(flags,0,1)
            

            indFlag=np.any(flags,axis=2)
            nbl,nf=indFlag.shape
            indFlag=indFlag.reshape((nbl,nf,1))*np.ones((1,1,4),bool)
            flags[indFlag]=True
            if (self.MaskHDF5!=None)&(self.MaskHDF5!=False):
                flags[:,self.MaskHDF5,:]=True

            data[flags]=0


        T.timeit("flag")


        irc=self.ids[0]
        if GiveAnt:
            r.wait()
            A0=np.array(self.E.rc[irc].get("A0"))
            A1=np.array(self.E.rc[irc].get("A1"))
            if BLSel!=None:
                A0=A0[BLSel]
                A1=A1[BLSel]
        else:
            A0,A1=None,None

        uvw=self.GiveUVW(it0,it1)
        if BLSel!=None:
            uvw=uvw.reshape((NtimeBlocks,self.nbl,3))
            uvw=uvw[:,BLSel,:]
            uvw=uvw.reshape((NtimeBlocks*uvw.shape[1],3))

        times=self.GiveTimes(it0,it1)#MS.times_all[row0:row1]
        if BLSel!=None:
            times=times.reshape((NtimeBlocks,MS.nbl))
            times=times[:,BLSel]
            times=times.reshape((times.size,))

        
            


        DicoDataOut={"nt":NtimeBlocks,
                     "itimes":(it0,it1),
                     "times":times,
                     "freqs":self.freqs.flatten(),
                     "A0A1":(A0,A1),
                     #"uvw":None,#uvw,
                     "uvw":uvw,
                     "flags":flags,
                     "nf":self.freqs.size,
                     "nbl":data.shape[0]/NtimeBlocks,
                     "data":data,
                     "ROW_01":(ROW0,ROW1)
                     }
        if GiveMapBLSel:
            Map=np.arange(NtimeBlocks*MS.nbl).reshape((NtimeBlocks,MS.nbl))
            Map=Map[:,BLSel]
            Map=Map.reshape((Map.size,))
            DicoDataOut["MapBLSel"]=Map


        return DicoDataOut

    
    def GiveUVW(self,it0=0,it1=-1):
        self.updateClusterInfo()
        irc=self.ids[0]
        r=self.E.rc[irc].execute("row0=%i*MS.nbl; row1=%i*MS.nbl"%(it0,it1)); r.wait()
        r=self.E.rc[irc].execute("if row1<0: row1=None"); r.wait()
        r=self.E.rc[irc].execute("uvw=MS.uvw[row0:row1,:]"); r.wait()
        uvw=self.E.rc[irc].get("uvw")
        return uvw

    def GiveTimes(self,it0=0,it1=-1):
        self.updateClusterInfo()
        irc=self.ids[0]
        r=self.E.rc[irc].execute("row0=%i*MS.nbl; row1=%i*MS.nbl"%(it0,it1)); r.wait()
        r=self.E.rc[irc].execute("if row1<0: row1=None"); r.wait()
        r=self.E.rc[irc].execute("times_all=MS.times_all[row0:row1]"); r.wait()
        times_all=self.E.rc[irc].get("times_all")
        return times_all

        
    def GiveBeam(self,time,RefBeam=False):
        self.updateClusterInfo()
        if self.isInitBeam==False:
            Mode=self.GD.DicoConfig["Jones"]["LOFARBeam"]["Mode"]
            useElementBeam=("E" in Mode)
            useArrayFactor=("A" in Mode)
            LaunchAndCheck(self.V,"MS.LoadSR(useElementBeam=%s,useArrayFactor=%s)"%(useElementBeam,useArrayFactor))
            self.isInitBeam=True

        #log=MyLogger.getLogger("IPCluster.GiveBeam")
        if not(RefBeam):
            #rr=self.V.execute("BeamMS=MS.GiveBeam(%f,SM.ClusterCat.ra,SM.ClusterCat.dec)"%time); rr.wait(); print rr.get()
            LaunchAndCheck(self.V,"BeamMS=MS.GiveBeam(%f,SM.ClusterCat.ra,SM.ClusterCat.dec)"%time)
            #print>>log, ("loadbeam",rr.get())
        else:
            stop
            rr=self.V.execute("BeamMS=MS.GiveBeam(%f,np.array([Xp.MSdata.rac]),np.array([Xp.MSdata.decc]))"%time); rr.wait()
            
            print>>log, ("loadbeam",rr.get())
            
        lout=self.V["BeamMS"]
        freqOrderEngines=self.V["freq"]
        
        indf=np.argsort(freqOrderEngines)
        lout=[lout[i] for i in indf]

        null=np.zeros_like(lout[0])

        # pdb> null.shape
        # (1, 1, 53, 2, 2)
        null[:,:,:,0,0]=1
        null[:,:,:,1,1]=1

        if self.DoMaskFreq:
            for i in range(self.MaskFreqMS.shape[0]):
                if self.MaskFreqMS[i]:
                    #print>>log, "slot %i is empty"%i
                    lout.insert(i,null)

        Beam=np.array(lout)[:,:,:,0,:,:]
        # (121, 2, 53, 2, 2)
        Beam=np.swapaxes(Beam,0,1)
        nd,nf,na,_,_=Beam.shape
        Beam=Beam.reshape((nd,nf,na,4))
        # # shape (2, 5, 50, 36, 2, 2)
        # print "0",Beam.shape
        # #(121, 2, 1, 53, 2, 2)
        # nf,nd, nb, na, _,_=Beam.shape
        # Beam=np.swapaxes(Beam,1,2)
        # print "1",Beam.shape
        # Beam=Beam.reshape((nb*nf,nd,na,2,2))
        # print "2",Beam.shape
        # Beam=np.swapaxes(Beam,0,1)
        # print "3",Beam.shape
        # #Beam=np.swapaxes(Beam,1,2)
        # print "4",Beam.shape

        return Beam
    
