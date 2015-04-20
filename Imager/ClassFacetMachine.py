from DDFacet.Other.progressbar import ProgressBar
#ProgressBar.silent=1
import multiprocessing
#import ClassGridMachine
import ClassDDEGridMachine
import numpy as np
#import ClassMS
import pylab
import ClassCasaImage
#import MyImshow
import pyfftw
from DDFacet.ToolsDir import ModCoord
#import ToolsDir
from DDFacet.Other import MyPickle
from DDFacet.Other import MyLogger
#import ModSharedArray
import time
from DDFacet.Other import ModColor
from DDFacet.Array import NpShared
from DDFacet.ToolsDir import ModFFTW
import pyfftw

log=MyLogger.getLogger("ClassFacetImager")
MyLogger.setSilent("MyLogger")
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
#import ClassJonesContainer
from DDFacet.ToolsDir.GiveEdges import GiveEdges

class ClassFacetMachine():
    def __init__(self,
                 VS,
                 GD,
                 #ParsetFile="ParsetNew.txt",
                 Precision="S",
                 PolMode="I",Sols=None,PointingID=0,
                 Parallel=False,#True,
                 DoPSF=False,
                 NCPU=6,
                 IdSharedMem="",
                 ApplyCal=False):
        self.IdSharedMem=IdSharedMem
        self.NCPU=int(GD["Parallel"]["NCPU"])
        self.ApplyCal=ApplyCal
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
        self.VS,self.GD=VS,GD
        self.Parallel=Parallel
        ChanFreq=self.VS.MS.ChanFreq.flatten()
        DicoConfigGM={}
        self.DicoConfigGM=DicoConfigGM
        self.DoPSF=DoPSF
        #self.MDC.setFreqs(ChanFreq)
        self.CasaImage=None
        self.IsDirtyInit=False
        self.IsDDEGridMachineInit=False
        self.SharedNames=[]
        self.ConstructMode= GD["ImagerMainFacet"]["ConstructMode"]


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

        MS=self.VS.MS
        

        rac,decc=MS.radec
        self.MainRaDec=(rac,decc)

        self.CoordMachine=ModCoord.ClassCoordConv(rac,decc)

        _,NpixPaddedGrid=EstimateNpix(NpixFacet,Padding=Padding)
        self.NChanGrid=1
        self.NpixPaddedFacet=NpixPaddedGrid
        self.PaddedGridShape=(self.NChanGrid,self.npol,NpixPaddedGrid,NpixPaddedGrid)
        print>>log,"Sizes (%i x %i facets):"%(NFacets,NFacets)
        print>>log,"   - Main field :   [%i x %i] pix"%(self.Npix,self.Npix)
        print>>log,"   - Each facet :   [%i x %i] pix"%(NpixFacet,NpixFacet)
        print>>log,"   - Padded-facet : [%i x %i] pix"%(NpixPaddedGrid,NpixPaddedGrid)
        
        #self.setWisdom()
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

        ChanFreq=self.VS.MS.ChanFreq.flatten()

        DicoConfigGM={"Npix":NpixFacet,
                      "Cell":Cell,
                      "ChanFreq":ChanFreq,
                      "DoPSF":False,
                      "Support":Support,
                      "OverS":OverS,
                      "wmax":wmax,
                      "Nw":Nw,
                      "WProj":True,
                      "DoDDE":self.DoDDE,
                      "Padding":Padding}




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
            self.DicoImager[iFacet]["l0m0"]=self.CoordMachine.radec2lm(raFacet,decFacet)
            self.DicoImager[iFacet]["RaDec"]=raFacet[0],decFacet[0]
            self.LraFacet.append(raFacet[0])
            self.LdecFacet.append(decFacet[0])
            x0,y0=x0facet[iFacet],y0facet[iFacet]
            self.DicoImager[iFacet]["pixExtent"]=x0,x0+NpixFacet,y0,y0+NpixFacet
            self.DicoImager[iFacet]["pixCentral"]=x0+NpixFacet/2,y0+NpixFacet/2
            self.DicoImager[iFacet]["NpixFacet"]=NpixFacet
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


    ############################################################################################
    ################################ Initialisation ############################################
    ############################################################################################


    def PlotFacetSols(self):

        DicoClusterDirs=NpShared.SharedToDico("%sDicoClusterDirs"%self.IdSharedMem)
        lc=DicoClusterDirs["l"]
        mc=DicoClusterDirs["m"]
        sI=DicoClusterDirs["I"]
        x0,x1=lc.min()-np.pi/180,lc.max()+np.pi/180
        y0,y1=mc.min()-np.pi/180,mc.max()+np.pi/180
        InterpMode=self.GD["DDESolutions"]["Type"]
        if InterpMode=="Krigging":
            for iFacet in sorted(self.DicoImager.keys()):
                l0,m0=self.DicoImager[iFacet]["lmShift"]
                d0=self.GD["DDESolutions"]["Scale"]*np.pi/180
                gamma=self.GD["DDESolutions"]["gamma"]
        
                d=np.sqrt((l0-lc)**2+(m0-mc)**2)
                idir=np.argmin(d)
                w=sI/(1.+d/d0)**gamma
                w/=np.sum(w)
                w[w<(0.2*w.max())]=0
                ind=np.argsort(w)[::-1]
                w[ind[4::]]=0

                ind=np.where(w!=0)[0]
                pylab.clf()
                pylab.scatter(lc[ind],mc[ind],c=w[ind],vmin=0,vmax=w.max())
                pylab.scatter([l0],[m0],marker="+")
                pylab.xlim(x0,x1)
                pylab.ylim(y0,y1)
                pylab.draw()
                pylab.show(False)
                pylab.pause(0.1)

            


    def Init(self):
        if self.IsDDEGridMachineInit: return
        self.DicoGridMachine={}
        for iFacet in self.DicoImager.keys():
            self.DicoGridMachine[iFacet]={}
        self.setWisdom()
        if self.Parallel:
            self.InitParallel()
        else:
            self.InitSerial()
        self.IsDDEGridMachineInit=True
        self.SetLogModeSubModules("Loud")


    def InitSerial(self):
        if self.ConstructMode=="Fader":
            SpheNorm=False
        for iFacet in sorted(self.DicoImager.keys()):
            GridMachine=ClassDDEGridMachine.ClassDDEGridMachine(self.GD,#RaDec=self.DicoImager[iFacet]["RaDec"],
                                                                self.DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
                                                                self.DicoImager[iFacet]["DicoConfigGM"]["Npix"],
                                                                lmShift=self.DicoImager[iFacet]["lmShift"],
                                                                IdSharedMem=self.IdSharedMem,IDFacet=iFacet,SpheNorm=SpheNorm)
            #,
             #                                                   **self.DicoImager[iFacet]["DicoConfigGM"])

            self.DicoGridMachine[iFacet]["GM"]=GridMachine

    def setWisdom(self):
        self.FFTW_Wisdom=None
        return
        print>>log, "Set fftw widsdom for shape = %s"%str(self.PaddedGridShape)
        a=np.random.randn(*(self.PaddedGridShape))+1j*np.random.randn(*(self.PaddedGridShape))
        FM=ModFFTW.FFTW_2Donly(self.PaddedGridShape, np.complex64)
        b=FM.fft(a)
        self.FFTW_Wisdom=None#pyfftw.export_wisdom()
        for iFacet in sorted(self.DicoImager.keys()):
            A=ModFFTW.GiveFFTW_aligned(self.PaddedGridShape, np.complex64)
            NpShared.ToShared("%sFFTW.%i"%(self.IdSharedMem,iFacet),A)
            


    def InitParallel(self):

        NCPU=self.NCPU

        NFacets=len(self.DicoImager.keys())

        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        NJobs=NFacets
        for iFacet in range(NFacets):
            work_queue.put(iFacet)

        workerlist=[]
        for ii in range(NCPU):
            W=WorkerImager(work_queue, result_queue,
                           self.GD,
                           Mode="Init",
                           FFTW_Wisdom=self.FFTW_Wisdom,
                           DicoImager=self.DicoImager,
                           IdSharedMem=self.IdSharedMem,
                           ApplyCal=self.ApplyCal)
            workerlist.append(W)
            workerlist[ii].start()

        #print>>log, ModColor.Str("  --- Initialising DDEGridMachines ---",col="green")
        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="      Init ", HeaderSize=10,TitleSize=13)
        pBAR.render(0, '%4i/%i' % (0,NFacets))
        iResult=0

        while iResult < NJobs:
            DicoResult=result_queue.get()
            if DicoResult["Success"]:
                iResult+=1
            NDone=iResult
            intPercent=int(100*  NDone / float(NFacets))
            pBAR.render(intPercent, '%4i/%i' % (NDone,NFacets))


        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()

            
        return True


    ############################################################################################
    ############################################################################################
    ############################################################################################

    def setCasaImage(self,ImageName=None):
        if ImageName==None:
            ImageName=self.ImageName
        self.CasaImage=ClassCasaImage.ClassCasaimage(ImageName,self.OutImShape,self.Cell,self.MainRaDec)

    def ToCasaImage(self,ImageIn,Fits=True,ImageName=None,beam=None):
        # if ImageIn==None:
        #     Image=self.FacetsToIm()
        # else:

        # Image=np.ones(self.OutImShape,np.float32)#np.float32(ImageIn)
        #ClassCasaImage.test()
        # print ClassCasaImage.pyfits.__file__
        # print ClassCasaImage.pyrap.images.__file__
        # name,imShape,Cell,radec="lala2.psf", self.OutImShape, 20, (3.7146787856873478, 0.91111035090915093)
        # im=ClassCasaImage.ClassCasaimage(name,imShape,Cell,radec)
        # im.setdata(np.random.randn(*(self.OutImShape)),CorrT=True)
        # im.ToFits()
        # im.setBeam((0.,0.,0.))
        # im.close()

        if self.CasaImage==None:
            self.setCasaImage(ImageName=ImageName)

        self.CasaImage.setdata(ImageIn,CorrT=True)

        if Fits:
            self.CasaImage.ToFits()
            if beam!=None:
                self.CasaImage.setBeam(beam)
        self.CasaImage.close()
        self.CasaImage=None


    def GiveEmptyMainField(self):
        return np.zeros(self.OutImShape,dtype=np.float32)

        
        



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
            print>>log, "Gridding facet #%i"%iFacet
            uvw=uvwIn.copy()
            vis=visIn.copy()
            if self.DoPSF: vis.fill(1)
            GridMachine=self.DicoGridMachine[iFacet]["GM"]
            #self.DicoImager[iFacet]["Dirty"]=GridMachine.put(times,uvw,vis,flag,A0A1,W,doStack=False)
            #self.DicoImager[iFacet]["Dirty"]=GridMachine.getDirtyIm()
            Dirty=GridMachine.put(times,uvw,vis,flag,A0A1,W,DoNormWeights=False)
            if (doStack==True)&("Dirty" in self.DicoImager[iFacet].keys()):
                self.DicoGridMachine[iFacet]["Dirty"]+=Dirty.copy()
            else:
                self.DicoGridMachine[iFacet]["Dirty"]=Dirty.copy()
                
            self.DicoGridMachine[iFacet]["Weights"]=GridMachine.SumWeigths
            print>>log, "Gridding facet #%i: done"%iFacet

        ThisSumWeights=self.DicoGridMachine[0]["Weights"]
        self.SumWeights+=ThisSumWeights
        print self.SumWeights


    def FacetsToIm(self):
        Image=self.GiveEmptyMainField()
        nch,npol=self.nch,self.npol
        _,_,NPixOut,NPixOut=self.OutImShape
        print>>log, "Combining facets using %s mode..."%self.ConstructMode
        if self.ConstructMode=="Fader": 
            SharedMemName="%sSpheroidal"%(self.IdSharedMem)#"%sWTerm.Facet_%3.3i"%(self.IdSharedMem,0)
            NormImage=np.zeros((NPixOut,NPixOut),dtype=Image.dtype)
            #SPhe=NpShared.UnPackListSquareMatrix(SharedMemName)[0]
            SPhe=NpShared.GiveArray(SharedMemName)
            
        for iFacet in self.DicoImager.keys():
            if self.ConstructMode=="Sharp":
                x0,x1,y0,y1=self.DicoImager[iFacet]["pixExtent"]
                for ch in range(nch):
                    for pol in range(npol):
                        Image[ch,pol,x0:x1,y0:y1]=self.DicoGridMachine[iFacet]["Dirty"][ch,pol][::-1,:].T.real
            elif self.ConstructMode=="Fader":
                
                xc,yc=self.DicoImager[iFacet]["pixCentral"]
                NpixFacet=self.DicoGridMachine[iFacet]["Dirty"].shape[2]

                M_xc=xc
                M_yc=yc
                NpixMain=NPixOut
                F_xc=NpixFacet/2
                F_yc=NpixFacet/2
                
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


                # print "======================="
                # print "Facet %i %s"%(iFacet,str(self.DicoGridMachine[iFacet]["Dirty"].shape))
                # print "Facet %i:%i (%i)"%(x0facet,x1facet,x1facet-x0facet)
                # print "Main  %i:%i (%i)"%(x0main,x1main,x1main-x0main)
                for ch in range(nch):
                    for pol in range(npol):
                        #Image[ch,pol,x0main:x1main,y0main:y1main]+=self.DicoGridMachine[iFacet]["Dirty"][ch,pol][::-1,:].T.real[x0facet:x1facet,y0facet:y1facet]
                        sumweight=self.SumWeights.reshape((nch,npol,1,1))[ch,pol,0,0]
                        Image[ch,pol,x0main:x1main,y0main:y1main]+=(self.DicoGridMachine[iFacet]["Dirty"][ch,pol][::-1,:]\
                                                                        .T.real[x0facet:x1facet,y0facet:y1facet]/sumweight)
                NormImage[x0main:x1main,y0main:y1main]+=SPhe[::-1,:].T.real[x0facet:x1facet,y0facet:y1facet]

        if self.ConstructMode=="Fader": 
            for ch in range(nch):
                for pol in range(npol):
                    Image[ch,pol]/=NormImage
 


        for iFacet in self.DicoImager.keys():
            del(self.DicoGridMachine[iFacet]["Dirty"])
            DirtyName="%sImageFacet.%3.3i"%(self.IdSharedMem,iFacet)
            _=NpShared.DelArray(DirtyName)

        # for ch in range(nch):
        #     for pol in range(npol):
        #         self.Image[ch,pol]=self.Image[ch,pol].T[::-1,:]
        

        # Image/=self.SumWeights.reshape((nch,npol,1,1))

        return Image

    def GiveNormImage(self):
        Image=self.GiveEmptyMainField()
        nch,npol=self.nch,self.npol
        _,_,NPixOut,NPixOut=self.OutImShape
        SharedMemName="%sSpheroidal"%(self.IdSharedMem)
        NormImage=np.zeros((NPixOut,NPixOut),dtype=Image.dtype)
        SPhe=NpShared.GiveArray(SharedMemName)
        N1=self.NpixPaddedFacet
            
        for iFacet in self.DicoImager.keys():
                
            xc,yc=self.DicoImager[iFacet]["pixCentral"]
            Aedge,Bedge=GiveEdges((xc,yc),NPixOut,(N1/2,N1/2),N1)
            x0d,x1d,y0d,y1d=Aedge
            x0p,x1p,y0p,y1p=Bedge
            
            for ch in range(nch):
                for pol in range(npol):
                    NormImage[x0d:x1d,y0d:y1d]+=SPhe[::-1,:].T.real[x0p:x1p,y0p:y1p]


        return NormImage



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
            uvw=uvwIn#.copy()
            vis=visIn#.copy()
            GridMachine=self.DicoGridMachine[iFacet]["GM"]
            ModelIm=self.DicoImager[iFacet]["ModelFacet"]
            vis=GridMachine.get(times,uvw,vis,flags,A0A1,ModelIm)
            #self.DicoImager[iFacet]["Predict"]=vis
            visOut+=vis
        return visOut

    def ReinitDirty(self):
        self.SumWeights.fill(0)
        self.IsDirtyInit=True
        for iFacet in self.DicoGridMachine.keys():
            if "Dirty" in self.DicoGridMachine[iFacet].keys():
                self.DicoGridMachine[iFacet]["Dirty"].fill(0)
            if "GM" in self.DicoGridMachine[iFacet].keys():
                self.DicoGridMachine[iFacet]["GM"].reinitGrid() # reinitialise sumWeights
        # if self.Parallel:
        #     V=self.IM.CI.E.GiveSubCluster("Imag")["V"]
        #     LaunchAndCheck(V,'execfile("%s/Scripts/ScriptReinitGrids.py")'%self.GD.HYPERCAL_DIR)

    def CalcDirtyImagesParallel(self,times,uvwIn,visIn,flag,A0A1,W=None,doStack=True):
        
        
        NCPU=self.NCPU

        NFacets=len(self.DicoImager.keys())

        work_queue = multiprocessing.JoinableQueue()


        PSFMode=False
        if self.DoPSF:
            visIn.fill(1)
            PSFMode=True

        NJobs=NFacets
        for iFacet in range(NFacets):
            work_queue.put(iFacet)

        workerlist=[]
        SpheNorm=True
        if self.ConstructMode=="Fader":
            SpheNorm=False

        List_Result_queue=[]
        for ii in range(NCPU):
            List_Result_queue.append(multiprocessing.JoinableQueue())


        for ii in range(NCPU):
            W=WorkerImager(work_queue, List_Result_queue[ii],
                           self.GD,
                           Mode="Grid",
                           FFTW_Wisdom=self.FFTW_Wisdom,
                           DicoImager=self.DicoImager,
                           IdSharedMem=self.IdSharedMem,
                           ApplyCal=self.ApplyCal,
                           SpheNorm=SpheNorm,
                           PSFMode=PSFMode)
            workerlist.append(W)
            workerlist[ii].start()

        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="  Gridding ", HeaderSize=10,TitleSize=13)
        pBAR.render(0, '%4i/%i' % (0,NFacets))
        iResult=0
        while iResult < NJobs:
            DicoResult=None
            for result_queue in List_Result_queue:
                if result_queue.qsize()!=0:
                    try:
                        DicoResult=result_queue.get_nowait()
                        break
                    except:
                        pass
                
            if DicoResult==None:
                time.sleep(1)
                continue


            if DicoResult["Success"]:
                iResult+=1
                NDone=iResult
                intPercent=int(100*  NDone / float(NFacets))
                pBAR.render(intPercent, '%4i/%i' % (NDone,NFacets))
            iFacet=DicoResult["iFacet"]

            if iFacet==0:
                ThisSumWeights=DicoResult["Weights"]
                self.SumWeights+=ThisSumWeights

            DirtyName=DicoResult["DirtyName"]
            ThisDirty=NpShared.GiveArray(DirtyName)
            #print "minmax facet = %f %f"%(ThisDirty.min(),ThisDirty.max())
            if (doStack==True)&("Dirty" in self.DicoGridMachine[iFacet].keys()):
                self.DicoGridMachine[iFacet]["Dirty"]+=ThisDirty
                #print "minmax stack = %f %f"%(self.DicoGridMachine[iFacet]["Dirty"].min(),self.DicoGridMachine[iFacet]["Dirty"].max())
            else:
                self.DicoGridMachine[iFacet]["Dirty"]=ThisDirty

        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()

            
        return True



        
   


    def GiveVisParallel(self,times,uvwIn,visIn,flag,A0A1,ModelImage):
        NCPU=self.NCPU
        #visOut=np.zeros_like(visIn)


        print>>log, "Model image to facets ..."
        self.ImToFacets(ModelImage)
        NFacets=len(self.DicoImager.keys())
        ListModelImage=[]
        for iFacet in self.DicoImager.keys():
            ListModelImage.append(self.DicoImager[iFacet]["ModelFacet"])
        
        NpShared.PackListArray("%sModelImage"%self.IdSharedMem,ListModelImage)
        del(ListModelImage)
        for iFacet in self.DicoImager.keys():
            del(self.DicoImager[iFacet]["ModelFacet"])
        print>>log, "    ... done"



        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        NJobs=NFacets
        for iFacet in range(NFacets):
            work_queue.put(iFacet)

        workerlist=[]
        for ii in range(NCPU):
            W=WorkerImager(work_queue, result_queue,
                           self.GD,
                           Mode="DeGrid",
                           FFTW_Wisdom=self.FFTW_Wisdom,
                           DicoImager=self.DicoImager,
                           IdSharedMem=self.IdSharedMem,
                           ApplyCal=self.ApplyCal)
            workerlist.append(W)
            workerlist[ii].start()

        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="DeGridding ", HeaderSize=10,TitleSize=13)
        pBAR.render(0, '%4i/%i' % (0,NFacets))
        iResult=0
        while iResult < NJobs:
            DicoResult=result_queue.get()
            if DicoResult["Success"]:
                iResult+=1
            NDone=iResult
            intPercent=int(100*  NDone / float(NFacets))
            pBAR.render(intPercent, '%4i/%i' % (NDone,NFacets))



        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()

        NpShared.DelArray("%sModelImage"%self.IdSharedMem)
            
        return True

        
    def GiveGM(self,iFacet):
        GridMachine=ClassDDEGridMachine.ClassDDEGridMachine(self.GD,#RaDec=self.DicoImager[iFacet]["RaDec"],
                                                            self.DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
                                                            self.DicoImager[iFacet]["DicoConfigGM"]["Npix"],
                                                            lmShift=self.DicoImager[iFacet]["lmShift"],
                                                            IdSharedMem=self.IdSharedMem,IDFacet=iFacet,
                                                            SpheNorm=True)#,
        return GridMachine


        

##########################################
####### Workers
##########################################
#import gc
           
class WorkerImager(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 GD,
                 Mode="Init",
                 FFTW_Wisdom=None,
                 DicoImager=None,
                 IdSharedMem=None,
                 ApplyCal=False,
                 SpheNorm=True,
                 PSFMode=False):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.Mode=Mode
        self.FFTW_Wisdom=FFTW_Wisdom
        self.GD=GD
        self.DicoImager=DicoImager
        self.IdSharedMem=IdSharedMem
        self.ApplyCal=ApplyCal
        self.SpheNorm=SpheNorm
        self.PSFMode=PSFMode


    def shutdown(self):
        self.exit.set()

    def GiveGM(self,iFacet):
        GridMachine=ClassDDEGridMachine.ClassDDEGridMachine(self.GD,#RaDec=self.DicoImager[iFacet]["RaDec"],
                                                            self.DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
                                                            self.DicoImager[iFacet]["DicoConfigGM"]["Npix"],
                                                            lmShift=self.DicoImager[iFacet]["lmShift"],
                                                            IdSharedMem=self.IdSharedMem,IDFacet=iFacet,
                                                            SpheNorm=self.SpheNorm)#,
        #**self.DicoImager[iFacet]["DicoConfigGM"])
        return GridMachine
        
    def GiveDicoJonesMatrices(self):
        DicoJonesMatrices=None
        if self.PSFMode: return None
        if self.ApplyCal:
            DicoJonesMatrices=NpShared.SharedToDico("%skillMSSolutionFile"%self.IdSharedMem)
            DicoClusterDirs=NpShared.SharedToDico("%sDicoClusterDirs"%self.IdSharedMem)
            DicoJonesMatrices["DicoClusterDirs"]=DicoClusterDirs
            DicoJonesMatrices["MapJones"]=NpShared.GiveArray("%sMapJones"%self.IdSharedMem)
            
        return DicoJonesMatrices

    def run(self):
        #print multiprocessing.current_process()
        while not self.kill_received:
            #gc.enable()
            try:
                iFacet = self.work_queue.get()
            except:
                break

            if self.FFTW_Wisdom!=None:
                pyfftw.import_wisdom(self.FFTW_Wisdom)



            if self.Mode=="Init":
                self.GiveGM(iFacet)
                self.result_queue.put({"Success":True,"iFacet":iFacet})
                
            elif self.Mode=="Grid":
                #import gc
                #gc.enable()
                GridMachine=self.GiveGM(iFacet)
                DATA=NpShared.SharedToDico("%sDicoData"%self.IdSharedMem)
                uvwThis=DATA["uvw"]
                visThis=DATA["data"]
                flagsThis=DATA["flags"]
                times=DATA["times"]
                A0=DATA["A0"]
                A1=DATA["A1"]
                A0A1=A0,A1
                W=DATA["Weights"]
                freqs=DATA["freqs"]

                DicoJonesMatrices=self.GiveDicoJonesMatrices()
                Dirty=GridMachine.put(times,uvwThis,visThis,flagsThis,A0A1,W,DoNormWeights=False, DicoJonesMatrices=DicoJonesMatrices,freqs=freqs)#,doStack=False)

                DirtyName="%sImageFacet.%3.3i"%(self.IdSharedMem,iFacet)
                _=NpShared.ToShared(DirtyName,Dirty)
                del(Dirty)
                Sw=GridMachine.SumWeigths
                del(GridMachine)

                self.result_queue.put({"Success":True,"iFacet":iFacet,"DirtyName":DirtyName,"Weights":Sw})
                

                # gc.collect()
                # print "sleeping"
                # time.sleep(10)

                # self.result_queue.put({"Success":True})

            elif self.Mode=="DeGrid":
                
                GridMachine=self.GiveGM(iFacet)
                DATA=NpShared.SharedToDico("%sDicoData"%self.IdSharedMem)
                uvwThis=DATA["uvw"]
                visThis=DATA["data"]
                #PredictedDataName="%s%s"%(self.IdSharedMem,"predicted_data")
                #visThis=NpShared.GiveArray(PredictedDataName)
                flagsThis=DATA["flags"]
                times=DATA["times"]
                A0=DATA["A0"]
                A1=DATA["A1"]
                A0A1=A0,A1
                W=DATA["Weights"]
                freqs=DATA["freqs"]
                DicoJonesMatrices=self.GiveDicoJonesMatrices()
                ModelIm = NpShared.UnPackListArray("%sModelImage"%self.IdSharedMem)[iFacet]
                vis=GridMachine.get(times,uvwThis,visThis,flagsThis,A0A1,ModelIm,DicoJonesMatrices=DicoJonesMatrices,freqs=freqs)
                
                self.result_queue.put({"Success":True,"iFacet":iFacet})
#            print "Done %i"%iFacet







