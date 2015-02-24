from progressbar import ProgressBar
#ProgressBar.silent=1
import multiprocessing
import ClassGridMachine
import ClassDDEGridMachine
import numpy as np
import ClassMS
import pylab
import ClassCasaImage
import MyImshow
import pyfftw
import ModCoord
import ToolsDir
import MyPickle
import MyLogger
import ModSharedArray
import time
import ModColor
from IPClusterDir.CheckJob import LaunchAndCheck, SendAndCheck
import NpShared

log=MyLogger.getLogger("ClassFacetImager")
MyLogger.setSilent("MyLogger")
from ModToolBox import EstimateNpix
#import ClassJonesContainer


class ClassFacetMachine():
    def __init__(self,
                 MDC,GD,
                 #ParsetFile="ParsetNew.txt",
                 Precision="S",
                 PolMode="I",Sols=None,PointingID=0,
                 Parallel=False,#True,
                 DoPSF=False,
                 NCPU=6):

        self.NCPU=int(GD.DicoConfig["Cluster"]["NImagEngine"])
        if Precision=="S":
            self.dtype=np.complex64
        elif Precision=="D":
            self.dtype=np.complex128
        self.DoDDE=False
        if Sols!=None:
            self.setSols(Sols)
        self.PolMode=PolMode
        if PolMode=="I":
            self.npol=1
        elif PolMode=="IQUV":
            self.npol=4
        self.PointingID=PointingID
        self.MDC,self.GD=MDC,GD#ToolsDir.GiveMDC.GiveMDC(ParsetFile)
        self.Parallel=Parallel
        ChanFreq=self.MDC.giveFreqs(self.PointingID).flatten()
        DicoConfigGM={}
        self.DicoConfigGM=DicoConfigGM
        self.DoPSF=DoPSF
        #self.MDC.setFreqs(ChanFreq)
        self.CasaImage=None
        self.IsDirtyInit=False
        self.IsDDEGridMachineInit=False
        self.SharedNames=[]

    def SetLogModeSubModules(self,Mode="Silent"):
        SubMods=["ModelBeamSVD","ClassParam","ModToolBox","ModelIonSVD2","ClassPierce"]

        if Mode=="Silent":
            MyLogger.setSilent(SubMods)
        if Mode=="Loud":
            MyLogger.setLoud(SubMods)





    def setSols(self,SolsClass):
        self.DoDDE=True
        self.Sols=SolsClass


    def appendMainField(self,Npix=512,Cell=10.,NFacets=5,
                        Support=11,OverS=5,Padding=1.2,wmax=10000,Nw=11,RaDecRad=(0.,0.),
                        ImageName="Facet.image"):
        


        #print "Append0"; self.IM.CI.E.clear()
        self.ImageName=ImageName
        if self.DoPSF:
            #Npix*=2
            Npix*=1
        NpixFacet,_=EstimateNpix(float(Npix)/NFacets,Padding=1)
        Npix=NpixFacet*NFacets
        self.Npix=Npix

        MS=self.MDC.giveMS(0)
        rac,decc=MS.radec
        self.MainRaDec=(rac,decc)

        self.CoordMachine=ModCoord.ClassCoordConv(rac,decc)

        _,NpixPaddedGrid=EstimateNpix(NpixFacet,Padding=Padding)
        self.NChanGrid=1
        self.PaddedGridShape=(self.NChanGrid,self.npol,NpixPaddedGrid,NpixPaddedGrid)
        self.setWisdom()
        self.SumWeights=np.zeros((self.NChanGrid,self.npol),float)

        self.nch=1

        self.NFacets=NFacets
        lrad=Npix*(Cell/3600.)*0.5*np.pi/180.
        self.ImageExtent=[-lrad,lrad,-lrad,lrad]
        lfacet=NpixFacet*(Cell/3600.)*0.5*np.pi/180.
        self.NpixFacet=NpixFacet
        self.FacetShape=(self.nch,self.npol,NpixFacet,NpixFacet)
        lcenter_max=lrad-lfacet
        lFacet,mFacet,=np.mgrid[-lcenter_max:lcenter_max:(NFacets)*1j,-lcenter_max:lcenter_max:(NFacets)*1j]
        lFacet=lFacet.flatten()
        mFacet=mFacet.flatten()
        x0facet,y0facet=np.mgrid[0:Npix:NpixFacet,0:Npix:NpixFacet]
        x0facet=x0facet.flatten()
        y0facet=y0facet.flatten()
        self.Cell=Cell
        self.CellSizeRad=(Cell/3600.)*np.pi/180.

        #print "Append1"; self.IM.CI.E.clear()
        
        self.OutImShape=(self.nch,self.npol,self.Npix,self.Npix)    
        
        self.DicoImager={}

        ChanFreq=self.MDC.giveFreqs(self.PointingID)

        DicoConfigGM={"Npix":NpixFacet,
                      "Cell":Cell,
                      "ChanFreq":ChanFreq,
                      "DoPSF":False,
                      "Support":Support,
                      "OverS":OverS,
                      "wmax":wmax,
                      "Nw":Nw,
                      "WProj":True,
                      "DoDDE":self.DoDDE}

        #print "Append2"; self.IM.CI.E.clear()

        self.LraFacet=[]
        self.LdecFacet=[]
        for iFacet in range(lFacet.size):
            self.DicoImager[iFacet]={}
            lmShift=(lFacet[iFacet],mFacet[iFacet])
            self.DicoImager[iFacet]["lmShift"]=lmShift
            lfacet=NpixFacet*(Cell/3600.)*0.5*np.pi/180.
            
            self.DicoImager[iFacet]["lmDiam"]=lfacet
            raFacet,decFacet=self.CoordMachine.lm2radec(np.array([lmShift[0]]),np.array([lmShift[1]]))
            self.DicoImager[iFacet]["RaDec"]=raFacet[0],decFacet[0]
            self.LraFacet.append(raFacet[0])
            self.LdecFacet.append(decFacet[0])
            x0,y0=x0facet[iFacet],y0facet[iFacet]
            self.DicoImager[iFacet]["pixExtent"]=x0,x0+NpixFacet,y0,y0+NpixFacet
            self.DicoImager[iFacet]["DicoConfigGM"]=DicoConfigGM

        #print "Append3"; self.IM.CI.E.clear()

        # NPraFacet=np.array(self.LraFacet).flatten()
        # NPdecFacet=np.array(self.LdecFacet).flatten()
        # self.JC=ClassJonesContainer.ClassJonesContainer(self.GD,self.MDC)
        # self.JC.InitAJM(NPraFacet,NPdecFacet)
        # MS=self.MDC.giveMS(0)
        # MS.ReadData()
        # self.JC.CalcJones(MS.times_all,(MS.A0,MS.A1))
        self.SetLogModeSubModules("Silent")


    def Init(self):
        if self.IsDDEGridMachineInit: return
        if self.Parallel:
            self.InitParallel()
        else:
            self.InitSerial()
        self.IsDDEGridMachineInit=True
        self.SetLogModeSubModules("Loud")

    def setCasaImage(self,ImageName=None):
        if ImageName==None:
            ImageName=self.ImageName
        self.CasaImage=ClassCasaImage.ClassCasaimage(ImageName,self.OutImShape,self.Cell,self.MainRaDec)

    def ToCasaImage(self,ImageIn=None,Fits=True,ImageName=None):
        if ImageIn==None:
            Image=self.FacetsToIm()
        else:
            Image=ImageIn
        if self.CasaImage==None:
            self.setCasaImage(ImageName=ImageName)
        self.CasaImage.setdata(Image,CorrT=True)
        if Fits:
            self.CasaImage.ToFits()
        self.CasaImage.close()
        self.CasaImage=None


    def GiveEmptyMainField(self):
        return np.zeros(self.OutImShape,dtype=np.float32)

    def setWisdom(self):
        return
        # FFTW wisdom
        import ModFFTW

        a=np.random.randn(*(self.PaddedGridShape))+1j*np.random.randn(*(self.PaddedGridShape))
        FM=ModFFTW.FFTW_2Donly(a.astype(np.complex128))
        b=FM.fft(a)
        import pyfftw
        self.FFTW_Wisdom=pyfftw.export_wisdom()
        
        

    def InitSerial(self):
        for iFacet in sorted(self.DicoImager.keys()):
            TransfRaDec=None
            GridMachine=ClassDDEGridMachine.ClassDDEGridMachine(self.GD,self.MDC,RaDec=self.DicoImager[iFacet]["RaDec"],
                                                                lmShift=self.DicoImager[iFacet]["lmShift"],
                                                                **self.DicoImager[iFacet]["DicoConfigGM"])

            if GridMachine.DoDDE:
                GridMachine.setSols(self.Sols.SolsCat.time,self.Sols.SolsCat.xi)
                GridMachine.CalcAterm()
                Xp=GridMachine.MME.Xp
                for Term in Xp.KeyOrderKeep:
                    T=Xp.giveModelTerm(Term)
                    if hasattr(T,"DelForPickle"): T.DelForPickle()
                
            self.DicoImager[iFacet]["GridMachine"]=GridMachine


    # def setModelIm(self,ModelIm):
    #     nch,npol,_,_=self.Image.shape
    #     # for ch in range(nch):
    #     #     for pol in range(npol):
    #     #         self.Image[ch,pol]=ModelIm[ch,pol].T[::-1]

    #     self.Image=ModelIm

    def putChunk(self,*args,**kwargs):
        self.SetLogModeSubModules("Silent")
        if not(self.IsDDEGridMachineInit):
            self.Init()
        if not(self.IsDirtyInit):
            self.ReinitDirty()
        if self.Parallel:
            return self.CalcDirtyImagesParallel(*args,**kwargs)
        else:
            return self.GiveDirtyimage(*args,**kwargs)
        self.SetLogModeSubModules("Loud")

    def getChunk(self,*args,**kwargs):
        self.SetLogModeSubModules("Silent")
        if self.Parallel:
            return self.GiveVisParallel(*args,**kwargs)
        else:
            return self.GiveVis(*args,**kwargs)
        self.SetLogModeSubModules("Loud")

    def GiveDirtyimage(self,times,uvwIn,visIn,flag,A0A1,W=None,doStack=False):
        Npix=self.Npix
        

        for iFacet in self.DicoImager.keys():
            uvw=uvwIn.copy()
            vis=visIn.copy()
            if self.DoPSF: vis.fill(1)
            GridMachine=self.DicoImager[iFacet]["GridMachine"]
            #self.DicoImager[iFacet]["Dirty"]=GridMachine.put(times,uvw,vis,flag,A0A1,W,doStack=False)
            #self.DicoImager[iFacet]["Dirty"]=GridMachine.getDirtyIm()
            Dirty=GridMachine.put(times,uvw,vis,flag,A0A1,W,DoNormWeights=False)
            if (doStack==True)&("Dirty" in self.DicoImager[iFacet].keys()):
                self.DicoImager[iFacet]["Dirty"]+=Dirty.copy()
            else:
                self.DicoImager[iFacet]["Dirty"]=Dirty.copy()
                
            self.DicoImager[iFacet]["Weights"]=GridMachine.SumWeigths

        ThisSumWeights=self.DicoImager[0]["Weights"]
        self.SumWeights+=ThisSumWeights
        print self.SumWeights

    def FacetsToIm(self):
        nch,npol=self.nch,self.npol
        Image=self.GiveEmptyMainField()
        for iFacet in self.DicoImager.keys():
            x0,x1,y0,y1=self.DicoImager[iFacet]["pixExtent"]
            for ch in range(nch):
                for pol in range(npol):
                    Image[ch,pol,x0:x1,y0:y1]=self.DicoImager[iFacet]["Dirty"][ch,pol][::-1,:].T.real
        # for ch in range(nch):
        #     for pol in range(npol):
        #         self.Image[ch,pol]=self.Image[ch,pol].T[::-1,:]
        Image/=self.SumWeights.reshape((nch,npol,1,1))
        return Image



    def ImToFacets(self,Image):
        nch,npol=self.nch,self.npol
        for iFacet in self.DicoImager.keys():
            x0,x1,y0,y1=self.DicoImager[iFacet]["pixExtent"]
            #GGridMachine=self.DicoImager[iFacet]["GridMachine"]
            ModelIm=np.zeros((nch,npol,self.NpixFacet,self.NpixFacet),dtype=np.float32)
            for ch in range(nch):
                for pol in range(npol):
                    ModelIm[ch,pol]=Image[ch,pol,x0:x1,y0:y1].T[::-1,:].real

            self.DicoImager[iFacet]["ModelFacet"]=ModelIm
            #GridMachine.setModelIm(ModelIm)
            
    def GiveVis(self,times,uvwIn,visIn,flags,A0A1,ModelImage):
        Npix=self.Npix
        visOut=np.zeros_like(visIn)
        self.ImToFacets(ModelImage)
        for iFacet in self.DicoImager.keys():
            uvw=uvwIn.copy()
            vis=visIn.copy()
            GridMachine=self.DicoImager[iFacet]["GridMachine"]
            ModelIm=self.DicoImager[iFacet]["ModelFacet"]
            vis=GridMachine.get(times,uvw,vis,flags,A0A1,ModelIm)
            #self.DicoImager[iFacet]["Predict"]=vis
            visOut+=vis
        return visOut

    def ReinitDirty(self):
        self.SumWeights.fill(0)
        self.IsDirtyInit=True
        for iFacet in self.DicoImager.keys():
            if "Dirty" in self.DicoImager[iFacet].keys():
                self.DicoImager[iFacet]["Dirty"].fill(0)
            if "GridMachine" in self.DicoImager[iFacet].keys():
                self.DicoImager[iFacet]["GridMachine"].reinitGrid() # reinitialise sumWeights
        if self.Parallel:
            V=self.IM.CI.E.GiveSubCluster("Imag")["V"]
            LaunchAndCheck(V,'execfile("%s/Scripts/ScriptReinitGrids.py")'%self.GD.HYPERCAL_DIR)

    def CalcDirtyImagesParallel(self,times,uvwIn,visIn,flag,A0A1,W=None,doStack=True):
        
        
        NCPU=self.NCPU

        #print "CalcDirtyImagesParallel 0"; self.IM.CI.E.clear()
        NFacets=len(self.DicoImager.keys())
        irc=self.IM.CI.E.GiveSubCluster("Imag")["ids"]
        V=self.IM.CI.E.GiveSubCluster("Imag")["V"]
        E=self.IM.CI.E

        if self.DoPSF: visIn.fill(1)
        vis=visIn.copy()
        LaunchAndCheck(V,'execfile("%s/Scripts/ScriptClearDicoImager.py")'%self.GD.HYPERCAL_DIR,Progress=True,TitlePBAR="Clear DicoImager")
        #LaunchAndCheck(V,'execfile("%s/DDFacet/Scripts/ScriptInspectSizes.py")'%self.GD.HYPERCAL_DIR,Progress=True,TitlePBAR="Inspect object")
        #print V["DicoSizes"]

        if not(self.GD.DicoConfig["Files"]["VisInSharedMem"]):
            ##### sending data
            SendAndCheck(V,"times",times,TitlePBAR="Send times",Progress=True)
            SendAndCheck(V,"uvw",uvwIn,TitlePBAR="Send uvw",Progress=True)
            SendAndCheck(V,"vis",vis,TitlePBAR="Send vis",Progress=True)
            SendAndCheck(V,"flags",flag,TitlePBAR="Send flag",Progress=True)
            SendAndCheck(V,"A0A1",A0A1,TitlePBAR="Send A0A1",Progress=True)
            SendAndCheck(V,"W",W,TitlePBAR="Send Weights",Progress=True)
            V["UseShared"]=False

        ##### grid

        LaunchAndCheck(V,'execfile("%s/Scripts/ScriptGrid.py")'%self.GD.HYPERCAL_DIR,Progress=True,TitlePBAR="Grid data")

        V.execute("LFacets=DicoImager.keys()")
        LFacets=V["LFacets"]


        iFacet=LFacets[0][0]
        E.rc[irc[0]].execute('Wtot=DicoImager[%i]["Weights"]'%iFacet)
        
        ThisSumWeights=E.rc[irc[0]].get("Wtot")
        self.SumWeights+=ThisSumWeights
        
        
        for iEngine in range(len(irc)):
            ThisRC=E.rc[irc[iEngine]]
            L_iFacets=LFacets[iEngine]
            for iFacet in L_iFacets:
                ThisRC.push({"iFacet":iFacet})
                r=ThisRC.execute('Dirty=DicoImager[iFacet]["Dirty"]')#; r.wait(); print r.get()
                ThisDirty=ThisRC.get("Dirty")
                ThisDirty.setflags(write=True)
                if (doStack==True)&("Dirty" in self.DicoImager[iFacet].keys()):
                    self.DicoImager[iFacet]["Dirty"]+=ThisDirty
                else:
                    self.DicoImager[iFacet]["Dirty"]=ThisDirty
                r=ThisRC.execute('del(DicoImager[iFacet]["Dirty"])')#; r.wait(); print r.get()
                #r=ThisRC.execute('del(D)')#; r.wait(); print r.get()
        LaunchAndCheck(V,'del(Dirty)')
        #LaunchAndCheck(V,'del(DicoImager)')
        
                
        #print "CalcDirtyImagesParallel 9"; self.IM.CI.E.clear()
        E.clear()
        
    def reset(self):
        irc=self.IM.CI.E.GiveSubCluster("Imag")["ids"]
        V=self.IM.CI.E.GiveSubCluster("Imag")["V"]
        E=self.IM.CI.E
        V.execute("ll=%who_ls")
        ll=V["ll"]
        for var in ll[0]:
            print var
            V.execute("%reset_selective -f "+var)
            time.sleep(1)
            
        


    def GiveVisParallel(self,times,uvwIn,visIn,flag,A0A1,ModelImage):
        NCPU=self.NCPU
        visOut=np.zeros_like(visIn)

        irc=self.IM.CI.E.GiveSubCluster("Imag")["ids"]
        V=self.IM.CI.E.GiveSubCluster("Imag")["V"]
        E=self.IM.CI.E
        self.ImToFacets(ModelImage)
        
        V.execute("LFacets=DicoImager.keys()")
        LFacets=V["LFacets"]

        if not(self.GD.DicoConfig["Files"]["VisInSharedMem"]):
        #     ##### sharing data
        #     self.ClearSharedMemory()
        #     print>>log, ModColor.Str("Sharing data: start")
        #     PrefixShared="SharedVis"
        #     times=NpShared.ToShared("%s.times"%PrefixShared,times); self.SharedNames.append("%s.times"%PrefixShared)
        #     uvw=NpShared.ToShared("%s.uvw"%PrefixShared,uvwIn); self.SharedNames.append("%s.uvw"%PrefixShared)
        #     visOut=NpShared.ToShared("%s.vis"%PrefixShared,np.complex128(visOut)); self.SharedNames.append("%s.vis"%PrefixShared)
        #     flag=NpShared.ToShared("%s.flag"%PrefixShared,flag); self.SharedNames.append("%s.flag"%PrefixShared)
        #     A0,A1=A0A1
        #     A0=NpShared.ToShared("%s.A0"%PrefixShared,A0); self.SharedNames.append("%s.A0"%PrefixShared)
        #     A1=NpShared.ToShared("%s.A1"%PrefixShared,A1); self.SharedNames.append("%s.A1"%PrefixShared)
        #     V["UseShared"]=True
        #     V["PrefixShared"]=PrefixShared
        #     print>>log, ModColor.Str("Sharing data: done")
        # else:
            ##### sending data
            SendAndCheck(V,"times",times,TitlePBAR="Send times",Progress=True)
            SendAndCheck(V,"uvw",uvwIn,TitlePBAR="Send uvw",Progress=True)
            SendAndCheck(V,"vis",visIn,TitlePBAR="Send vis",Progress=True)
            SendAndCheck(V,"flags",flag,TitlePBAR="Send flag",Progress=True)
            SendAndCheck(V,"A0A1",A0A1,TitlePBAR="Send A0A1",Progress=True)
            V["UseShared"]=False

        
        ##### send facets
        for iEngine in range(len(irc)):
            ThisRC=E.rc[irc[iEngine]]
            L_iFacets=LFacets[iEngine]
            for iFacet in L_iFacets:
                #ThisModelName="ModelFacet%3.3i"%iFacet
                ThisModelName="ModelFacet"
                ThisRC.push({ThisModelName:self.DicoImager[iFacet]["ModelFacet"]})
                print iFacet,self.DicoImager[iFacet]["ModelFacet"].max()
                r=ThisRC.execute('DicoImager[%i]["ModelFacet"]=%s'%(iFacet,ThisModelName))#; r.wait(); print r.get()


        nRows=uvwIn.shape[0]
        V["Row0"]=0
        V["Row1"]=nRows
        LaunchAndCheck(V,'execfile("%s/Scripts/ScriptDeGrid.py")'%self.GD.HYPERCAL_DIR,Progress=True,TitlePBAR="DeGrid data")

        # UseShared=True
        # PrefixShared=self.PrefixShared        
        # execfile("%s/DDFacet/Scripts/ScriptDeGrid.py")%self.GD.HYPERCAL_DIR


        # ##### MapRows
        # nRows=uvwIn.shape[0]
        # nEng=nCycle=len(irc)
        # MapRows0=np.linspace(0,nRows,nEng+1)
        # V["vis"]=None
        # V["UseShared"]=True

        # for iCycle in range(nCycle):
        #     MapRows=np.roll(MapRows0,iCycle)
        #     for iEngine in range(len(irc)):
        #         ThisRC=E.rc[irc[iEngine]]
        #         L_iFacets=LFacets[iEngine]
        #         Rows=MapRows[iEngine:iEngine+2]
        #         Row0,Row1=Rows.min(),Rows.max()
        #         for iFacet in L_iFacets:
        #             #ThisModelName="ModelFacet%3.3i"%iFacet
        #             ThisModelName="ModelFacet"
        #             ThisRC.push({"Row0":Row0,"Row1":Row1})

        #         # r=ThisRC.execute('execfile("%s/DDFacet/Scripts/ScriptDeGrid.py"'); r.wait()
        #         # visThis=NpShared.GiveArray("%s.predict_data"%self.PrefixShared)
        #         # print L_iFacets, np.max(visThis)

        # #         if L_iFacets[0]==4:
        # #             r=ThisRC.execute('execfile("%s/DDFacet/Scripts/ScriptDeGrid.py")'%self.GD.HYPERCAL_DIR); r.wait(); print r.get()
        # #             visThis=NpShared.GiveArray("%s.predict_data"%self.PrefixShared)
        # # #             print L_iFacets, np.max(visThis)
        # # #             print ThisRC.get("vis")
        # # # stop

        #     LaunchAndCheck(V,'execfile("%s/DDFacet/Scripts/ScriptDeGrid.py")'%self.GD.HYPERCAL_DIR,Progress=True,TitlePBAR="DeGrid data")
            
    
        # ##### degrid
        # for iEngine in range(len(irc)):
        #     ThisRC=E.rc[irc[iEngine]]
        #     L_iFacets=LFacets[iEngine]
        #     for iFacet in L_iFacets:
        #         ThisRC.execute('Vis=DicoImager[%i]["PredictVis"]'%(iFacet))
        #         visOut+=ThisRC.get("Vis")
        #         ThisRC.execute("del(Vis)")
        # E.clear()
        
        return visOut

    def setInitMachine(self,IM):
        self.IM=IM
        

    def InitParallel(self):

        #print "Init00"; self.IM.CI.E.clear()
        NCPU=self.NCPU

        NFacets=len(self.DicoImager.keys())
        irc=self.IM.CI.E.GiveSubCluster("Imag")["ids"]
        V=self.IM.CI.E.GiveSubCluster("Imag")["V"]

        #print "Init01"; self.IM.CI.E.clear()

        LaunchAndCheck(V,'DicoImager={}')

        #print "Init0"; self.IM.CI.E.clear()
        V["FFTW_Wisdom"]=None#self.FFTW_Wisdom
        E=self.IM.CI.E
        iFacet=0
        IDDico=0
        iEngine=0
        while iFacet<NFacets:
            ThisRC=E.rc[irc[iEngine]]
            ThisRC.push({"Dico%3.3i"%IDDico:self.DicoImager[iFacet]})
            #print "DicoImager[%i]=Dico%3.3i"%(iFacet,IDDico)
            r=ThisRC.execute("DicoImager[%i]=Dico%3.3i"%(iFacet,IDDico))#; r.wait()#; print r.get()
            iFacet+=1
            iEngine+=1
            IDDico+=1
            #print iFacet,iEngine,IDDico
            if iEngine%NCPU==0:
                iEngine=0
        
        #print "Init1"; self.IM.CI.E.clear()
        SendAndCheck(V,"GD",self.GD,TitlePBAR="Send GD",Progress=True)
        SendAndCheck(V,"MDC",self.GD,TitlePBAR="Send MDC",Progress=True)
        LaunchAndCheck(V,'execfile("%s/Scripts/ScriptInitFacets.py")'%self.GD.HYPERCAL_DIR,Progress=True,TitlePBAR="Init GridMachine")
        #        print "Init2"
        #self.IM.CI.E.clear()

            # if self.DoDDE:
            #     ThisWorker.setSols(self.Sols.SolsCat.time,self.Sols.SolsCat.xi)

        

##########################################
####### Workers
##########################################
           
class WorkerImager(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,DicoImager,FacetShape,FFTW_Wisdom,GD,MDC):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.DicoImager=DicoImager
        self.Mode="Grid"
        self.FacetShape=FacetShape
        self.FFTW_Wisdom=FFTW_Wisdom
        self.GD=GD
        self.MDC=MDC
        self.SolsXi=None

    def setMode(self,Mode):
        self.Mode=Mode

    def setVis(self,times,uvw,data,flags,A0A1,W):
        #times,uvw,data,flags,A0A1,W=ModSharedArray.SharedToNumpy([times,uvw,data,flags,A0A1,W])
        self.uvw=uvw
        self.data=data
        self.flags=flags
        self.A0A1=A0A1
        self.times=times
        self.W=W

        

    def setSols(self,times,xi):
        self.SolsXi=xi
        self.SolsTimes=times

    def setModelImage(self,ModelIm):
        self.ModelIm=ModelIm

    def shutdown(self):
        self.exit.set()

    def run(self):
        print multiprocessing.current_process()
        while not self.kill_received:
            try:
                iFacet = self.work_queue.get()
            except:
                break
            #print "Do %i"%iFacet
            if self.FFTW_Wisdom!=None:
                pyfftw.import_wisdom(self.FFTW_Wisdom)



            if self.Mode=="Init":
                TransfRaDec=None
                # GridMachine=ClassGridMachine.ClassGridMachine(ImageName="Image_%i"%iFacet,
                #                                               TransfRaDec=TransfRaDec,
                #                                               lmShift=self.DicoImager[iFacet]["lmShift"],
                #                                               **self.DicoImager[iFacet]["DicoConfigGM"])

                
                GridMachine=ClassDDEGridMachine.ClassDDEGridMachine(self.GD,self.MDC,
                                                                    RaDec=self.DicoImager[iFacet]["RaDec"],
                                                                    lmShift=self.DicoImager[iFacet]["lmShift"],
                                                                    **self.DicoImager[iFacet]["DicoConfigGM"])
                #print " %i: declare"%iFacet


                #self.DicoImager[iFacet]["GridMachine"]=GridMachine
                # if self.SolsXi!=None:
                #     A0A1=self.A0A1
                #     times=self.times
                #     GridMachine.CalcAterm()

                if GridMachine.DoDDE:
                    GridMachine.setSols(self.SolsTimes,self.SolsXi)
                    GridMachine.CalcAterm()
                    Xp=GridMachine.MME.Xp
                    for Term in Xp.KeyOrderKeep:
                        T=Xp.giveModelTerm(Term)
                        if hasattr(T,"DelForPickle"): T.DelForPickle()


                self.result_queue.put({"iFacet":iFacet,"GridMachine":GridMachine})
                
            elif self.Mode=="CalcAterm":
                GridMachine=self.DicoImager[iFacet]["GridMachine"]
            elif self.Mode=="Grid":
                
                GridMachine=self.DicoImager[iFacet]["GridMachine"]

                uvw=self.uvw.copy()
                vis=self.data.copy()
                flags=self.flags
                A0A1=self.A0A1
                times=self.times
                W=self.W
                Dirty=GridMachine.put(times,uvw,vis,flags,A0A1,W,doStack=False)
                #Dirty=GridMachine.getDirtyIm()
                self.result_queue.put({"iFacet":iFacet,"Dirty":Dirty})

            elif self.Mode=="Predict":
                
                GridMachine=self.DicoImager[iFacet]["GridMachine"]
                uvw=self.uvw
                vis=self.data.copy()
                flags=self.flags
                A0A1=self.A0A1
                times=self.times

                # x0,x1,y0,y1=self.DicoImager[iFacet]["pixExtent"]
                # nch,npol,NpixFacet,NpixFacet=self.FacetShape
                # ModelIm=np.zeros((nch,npol,NpixFacet,NpixFacet),dtype=np.float32)
                # for ch in range(nch):
                #     for pol in range(npol):
                #         ModelIm[ch,pol]=self.ModelIm[ch,pol,x0:x1,y0:y1].T[::-1,:].real

                #GridMachine.setModelIm(ModelIm)
                ModelIm=self.DicoImager[iFacet]["ModelFacet"]
                vis=GridMachine.get(times,uvw,vis,flags,A0A1,ModelIm)
                self.result_queue.put({"iFacet":iFacet,"Vis":vis})
#            print "Done %i"%iFacet
