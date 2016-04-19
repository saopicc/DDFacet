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
#import ModSharedArray
import time
from DDFacet.Other import ModColor
from DDFacet.Array import NpShared
from DDFacet.ToolsDir import ModFFTW
import pyfftw
from DDFacet.Other import ClassTimeIt

from DDFacet.Other import MyLogger
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
                 Oversize=1,   # factor my which image is oversized
                 NCPU=6,
                 IdSharedMem="",
                 IdSharedMemData=None,       # == IdSharedMem if None
                 ApplyCal=False):
        # IdSharedMem is used to identify structures in shared memory used by this FacetMachine
        self.IdSharedMem = IdSharedMem
        # IdSharedMemData is used to identify "global" structures in shared memory such as DicoData
        self.IdSharedMemData = IdSharedMemData or IdSharedMem
        self.NCPU=int(GD["Parallel"]["NCPU"])
        self.ApplyCal=ApplyCal
        if Precision=="S":
            self.dtype=np.complex64
            self.CType=np.complex64
            self.FType=np.float32
        elif Precision=="D":
            self.dtype=np.complex128
            self.CType=np.complex128
            self.FType=np.float64
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
        DicoConfigGM={}
        self.DicoConfigGM=DicoConfigGM
        self.DoPSF=DoPSF
        #self.MDC.setFreqs(ChanFreq)
        self.CasaImage=None
        self.IsDirtyInit=False
        self.IsDDEGridMachineInit=False
        self.SharedNames=[]
        self.ConstructMode= GD["ImagerMainFacet"]["ConstructMode"]
        self.SpheNorm=True
        if self.ConstructMode=="Fader":
            self.SpheNorm=False
        self.Oversize = Oversize

        self.NormData=None
        self.NormImage=None

    def __del__ (self):
        #print>>log,"Deleting shared memory"
        NpShared.DelAll(self.IdSharedMem)

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
                        Support=11,OverS=5,Padding=1.2,
                        wmax=10000,Nw=11,RaDecRad=(0.,0.),
                        ImageName="Facet.image"):
        


        #print "Append0"; self.IM.CI.E.clear()
        self.ImageName=ImageName
        if self.DoPSF:
            #Npix*=2
            Npix*=1
        NpixFacet,_=EstimateNpix(float(Npix)/NFacets,Padding=1)
        Npix=NpixFacet*NFacets
        self.Npix=Npix


        rac,decc = self.VS.CurrentMS.radec

        self.MainRaDec=(rac,decc)

        self.CoordMachine=ModCoord.ClassCoordConv(rac,decc)

        _,NpixPaddedGrid=EstimateNpix(NpixFacet,Padding=Padding)

        self.NpixPaddedFacet=NpixPaddedGrid
        self.PaddedGridShape=(self.VS.NFreqBands,self.npol,NpixPaddedGrid,NpixPaddedGrid)
        print>>log,"Sizes (%i x %i facets):"%(NFacets,NFacets)
        print>>log,"   - Main field :   [%i x %i] pix"%(self.Npix,self.Npix)
        print>>log,"   - Each facet :   [%i x %i] pix"%(NpixFacet,NpixFacet)
        print>>log,"   - Padded-facet : [%i x %i] pix"%(NpixPaddedGrid,NpixPaddedGrid)
        
        #self.setWisdom()
        self.SumWeights=np.zeros((self.VS.NFreqBands,self.npol),float)

        self.NFacets=NFacets
        lrad=Npix*(Cell/3600.)*0.5*np.pi/180.
        self.ImageExtent=[-lrad,lrad,-lrad,lrad]
        lfacet=NpixFacet*(Cell/3600.)*0.5*np.pi/180.
        self.NpixFacet=NpixFacet
        self.FacetShape=(self.VS.NFreqBands,self.npol,NpixFacet,NpixFacet)
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
        
        self.OutImShape=(self.VS.NFreqBands,self.npol,self.Npix,self.Npix)    
        stop
        self.DicoImager={}

        ChanFreq=self.VS.GlobalFreqs

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

        self.FacetCat=np.zeros((lFacet.size,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('SumI',np.float),
                                                     ("Cluster",int),
                                                     ("l",np.float),("m",np.float),
                                                     ("I",np.float)])
        self.FacetCat=self.FacetCat.view(np.recarray)
        self.FacetCat.I=1
        self.FacetCat.SumI=1
        

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
            self.DicoImager[iFacet]["IDFacet"]=iFacet

            self.FacetCat.ra[iFacet]=raFacet[0]
            self.FacetCat.dec[iFacet]=decFacet[0]
            l,m=self.DicoImager[iFacet]["l0m0"]
            self.FacetCat.l[iFacet]=l
            self.FacetCat.m[iFacet]=m
            self.FacetCat.Cluster[iFacet]=iFacet

        self.DicoImagerCentralFacet=self.DicoImager[lFacet.size/2]



        #print "Append3"; self.IM.CI.E.clear()

        # NPraFacet=np.array(self.LraFacet).flatten()
        # NPdecFacet=np.array(self.LdecFacet).flatten()
        # self.JC=ClassJonesContainer.ClassJonesContainer(self.GD,self.MDC)
        # self.JC.InitAJM(NPraFacet,NPdecFacet)
        # MS=self.MDC.giveMS(0)
        # MS.ReadData()
        # self.JC.CalcJones(MS.times_all,(MS.A0,MS.A1))


        self.SetLogModeSubModules("Silent")
        self.MakeREG()

    def MakeREG(self):

        regFile="%s.Facets.reg"%self.ImageName
        print>>log, "Writing facets locations in %s"%regFile
        f=open(regFile,"w")
        f.write("# Region file format: DS9 version 4.1\n")
        ss0='global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0'
        ss1=' fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
        
        f.write(ss0+ss1)
        f.write("fk5\n")
 
        for iFacet in self.DicoImager.keys():
            #rac,decc=self.DicoImager[iFacet]["RaDec"]
            l0,m0=self.DicoImager[iFacet]["l0m0"]
            diam=self.DicoImager[iFacet]["lmDiam"]
            dl=np.array([-1,1,1,-1,-1])*diam
            dm=np.array([-1,-1,1,1,-1])*diam
            l=((dl.flatten()+l0)).tolist()
            m=((dm.flatten()+m0)).tolist()
            x=[]; y=[]
            
            for iPoint in range(len(l)):
                xp,yp=self.CoordMachine.lm2radec(np.array([l[iPoint]]),np.array([m[iPoint]]))
                x.append(xp)
                y.append(yp)

            x=np.array(x)#+[x[2]])
            y=np.array(y)#+[y[2]])

            x*=180/np.pi
            y*=180/np.pi


            for iline in range(x.shape[0]-1):
                x0=x[iline]
                y0=y[iline]
                x1=x[iline+1]
                y1=y[iline+1]
                f.write("line(%f,%f,%f,%f) # line=0 0\n"%(x0,y0,x1,y1))
            
        f.close()



    ############################################################################################
    ################################ Initialisation ############################################
    ############################################################################################


    def PlotFacetSols(self):

        DicoClusterDirs=NpShared.SharedToDico("%sDicoClusterDirs"%self.IdSharedMemData)
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
                                                                IdSharedMem=self.IdSharedMem,
                                                                IdSharedMemData=self.IdSharedMemData,
                                                                IDFacet=iFacet,SpheNorm=SpheNorm)
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

        self.SpacialWeigth={}

        workerlist=[]
        for ii in range(NCPU):
            W=WorkerImager(work_queue, result_queue,
                           self.GD,
                           Mode="Init",
                           FFTW_Wisdom=self.FFTW_Wisdom,
                           DicoImager=self.DicoImager,
                           IdSharedMem=self.IdSharedMem,
                           IdSharedMemData=self.IdSharedMemData,
                           ApplyCal=self.ApplyCal)
            workerlist.append(W)
            workerlist[ii].start()

        #print>>log, ModColor.Str("  --- Initialising DDEGridMachines ---",col="green")
        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="      Init W ", HeaderSize=10,TitleSize=13)
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

    def setCasaImage(self,ImageName=None,Shape=None,Freqs=None):
        if ImageName==None:
            ImageName=self.ImageName

        if Shape==None:
            Shape=self.OutImShape
        self.CasaImage=ClassCasaImage.ClassCasaimage(ImageName,Shape,self.Cell,self.MainRaDec,Freqs=Freqs)

    def ToCasaImage(self,ImageIn,Fits=True,ImageName=None,beam=None,beamcube=None,Freqs=None):
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

        #if self.CasaImage==None:
        self.setCasaImage(ImageName=ImageName,Shape=ImageIn.shape,Freqs=Freqs)

        self.CasaImage.setdata(ImageIn,CorrT=True)

        if Fits:
            self.CasaImage.ToFits()
            if beam is not None:
                self.CasaImage.setBeam(beam,beamcube=beamcube)
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
            print>>log, "Gridding facet #%i: done [%s]"%(iFacet,str(GridMachine.SumWeigths))

        ThisSumWeights=self.DicoGridMachine[0]["Weights"]
        self.SumWeights+=ThisSumWeights
        print self.SumWeights

    def FacetsToIm(self,NormJones=False):
        T=ClassTimeIt.ClassTimeIt("FacetsToIm")
        T.disable()
        _,npol,Npix,Npix=self.OutImShape
        DicoImages={}
        DicoImages["freqs"]={}
        ImagData=np.zeros((self.VS.NFreqBands,npol,Npix,Npix),dtype=np.float32)

        T.timeit("0")
        
        DoCalcNormData=False
        if (NormJones)&(self.NormData==None): 
            DoCalcNormData=True
            self.NormData=np.ones((self.VS.NFreqBands,npol,Npix,Npix),dtype=np.float32)

        T.timeit("1")


        DicoImages["SumWeights"]=np.zeros((self.VS.NFreqBands,),np.float64)
        for band,channels in enumerate(self.VS.FreqBandChannels):
            DicoImages["freqs"][band] = channels
            DicoImages["SumWeights"][band] = self.DicoImager[0]["SumWeights"][band]

        ImagData=self.FacetsToIm_Channel()

        if DoCalcNormData:
            self.NormData=self.FacetsToIm_Channel(BeamWeightImage=True)

        T.timeit("2")
                
        # if NormJones: 
        #     ImagData/=np.sqrt(self.NormData)

        T.timeit("3")

        DicoImages["ImagData"]=ImagData
        DicoImages["NormData"]=self.NormData
        DicoImages["WeightChansImages"]=DicoImages["SumWeights"]/np.sum(DicoImages["SumWeights"])
        if self.VS.MultiFreqMode:
            ImMean=np.zeros_like(ImagData)
            W=np.array([DicoImages["SumWeights"][Channel] for Channel in range(self.VS.NFreqBands)])
            W/=np.sum(W)
            W=np.float32(W.reshape((self.VS.NFreqBands,1,1,1)))
            DicoImages["MeanImage"]=np.sum(ImagData*W,axis=0).reshape((1,npol,Npix,Npix))

        else:
            DicoImages["MeanImage"]=ImagData

        T.timeit("4")

        if self.DoPSF:
            print>>log, "  Build PSF facet-slices "
            self.DicoPSF={}
            for iFacet in self.DicoGridMachine.keys():
                self.DicoPSF[iFacet]={}
                self.DicoPSF[iFacet]["PSF"]=(self.DicoGridMachine[iFacet]["Dirty"]).copy()
                self.DicoPSF[iFacet]["l0m0"]=self.DicoImager[iFacet]["l0m0"]
                self.DicoPSF[iFacet]["pixCentral"]=self.DicoImager[iFacet]["pixCentral"]

                nch,npol,n,n=self.DicoPSF[iFacet]["PSF"].shape
                PSFChannel=np.zeros((nch,npol,n,n),np.float32)
                for ch in range(nch):
                    self.DicoPSF[iFacet]["PSF"][ch][0]=self.DicoPSF[iFacet]["PSF"][ch][0].T[::-1,:]
                    self.DicoPSF[iFacet]["PSF"][ch]/=np.max(self.DicoPSF[iFacet]["PSF"][ch])
                    PSFChannel[ch,:,:,:]=self.DicoPSF[iFacet]["PSF"][ch][:,:,:]

                W=DicoImages["WeightChansImages"]
                W=np.float32(W.reshape((self.VS.NFreqBands,1,1,1)))
                
                MeanPSF=np.sum(PSFChannel*W,axis=0).reshape((1,npol,n,n))
                self.DicoPSF[iFacet]["MeanPSF"]=MeanPSF



            DicoVariablePSF=self.DicoPSF
            NFacets=len(DicoVariablePSF.keys())
            NPixMin=1e6
            for iFacet in sorted(DicoVariablePSF.keys()):
                _,npol,n,n=DicoVariablePSF[iFacet]["PSF"].shape
                if n<NPixMin: NPixMin=n

            nch = self.VS.NFreqBands
            CubeVariablePSF=np.zeros((NFacets,nch,npol,NPixMin,NPixMin),np.float32)
            CubeMeanVariablePSF=np.zeros((NFacets,1,npol,NPixMin,NPixMin),np.float32)

            print>>log, "  Cutting PSFs facet-slices "
            for iFacet in sorted(DicoVariablePSF.keys()):
                _,npol,n,n=DicoVariablePSF[iFacet]["PSF"].shape
                for ch in range(nch):
                    i=n/2-NPixMin/2
                    j=n/2+NPixMin/2+1
                    CubeVariablePSF[iFacet,ch,:,:,:]=DicoVariablePSF[iFacet]["PSF"][ch][:,i:j,i:j]
                CubeMeanVariablePSF[iFacet,0,:,:,:]=DicoVariablePSF[iFacet]["MeanPSF"][0,:,i:j,i:j]

            self.DicoPSF["CubeVariablePSF"]=CubeVariablePSF
            self.DicoPSF["CubeMeanVariablePSF"]=CubeMeanVariablePSF
            self.DicoPSF["MeanFacetPSF"]=np.mean(CubeMeanVariablePSF,axis=0).reshape((1,npol,NPixMin,NPixMin))


            self.DicoPSF["MeanJonesBand"]=[]
            for iFacet in sorted(self.DicoImager.keys()):
                #print "==============="
                MeanJonesBand=np.zeros((self.VS.NFreqBands,),np.float64)
                for Channel in range(self.VS.NFreqBands):
                    ThisSumSqWeights=self.DicoImager[iFacet]["SumJones"][1][Channel]
                    if ThisSumSqWeights==0: ThisSumSqWeights=1.
                    ThisSumJones=(self.DicoImager[iFacet]["SumJones"][0][Channel]/ThisSumSqWeights)
                    if ThisSumJones==0:
                        ThisSumJones=1.
                    #print "0",iFacet,Channel,np.sqrt(ThisSumJones)
                    MeanJonesBand[Channel]=ThisSumJones
                self.DicoPSF["MeanJonesBand"].append(MeanJonesBand)


            self.DicoPSF["SumJonesChan"]=[]
            self.DicoPSF["SumJonesChanWeightSq"]=[]
            for iFacet in sorted(self.DicoImager.keys()):

                ThisFacetSumJonesChan=[]
                ThisFacetSumJonesChanWeightSq=[]
                #print "==============="
                for iMS in range(self.VS.nMS):
                    A=self.DicoImager[iFacet]["SumJonesChan"][iMS][1,:]
                    A[A==0]=1.
                    A=self.DicoImager[iFacet]["SumJonesChan"][iMS][0,:]
                    A[A==0]=1.
                    SumJonesChan=self.DicoImager[iFacet]["SumJonesChan"][iMS][0,:]
                    SumJonesChanWeightSq=self.DicoImager[iFacet]["SumJonesChan"][iMS][1,:]
                    #print "1",iFacet,iMS,MeanJonesChan
                    ThisFacetSumJonesChan.append(SumJonesChan)
                    ThisFacetSumJonesChanWeightSq.append(SumJonesChanWeightSq)


                    # # ###############################
                    # l0,m0=self.DicoImager[iFacet]["lmShift"]
                    # d=np.sqrt(l0**2+m0**2)
                    # if d<0.0001:
                    #     print "========================="
                    #     print iFacet
                    #     print self.DicoImager[iFacet]["lmShift"]
                    #     print self.DicoImager[iFacet]["SumJonesChan"][iMS][1,:]
                    #     print self.DicoImager[iFacet]["SumJonesChan"][iMS][0,:]
                    #     print MeanJonesChan

                
                self.DicoPSF["SumJonesChan"].append(ThisFacetSumJonesChan)
                self.DicoPSF["SumJonesChanWeightSq"].append(ThisFacetSumJonesChanWeightSq)
            self.DicoPSF["ChanMappingGrid"]=self.VS.DicoMSChanMapping
            self.DicoPSF["ChanMappingGridChan"]=self.VS.DicoMSChanMappingChan
            self.DicoPSF["freqs"]=DicoImages["freqs"]
            self.DicoPSF["WeightChansImages"]=DicoImages["WeightChansImages"]
        T.timeit("5")
        # for iFacet in self.DicoImager.keys():
        #     del(self.DicoGridMachine[iFacet]["Dirty"])
        #     DirtyName="%sImageFacet.%3.3i"%(self.IdSharedMem,iFacet)
        #     _=NpShared.DelArray(DirtyName)



        return DicoImages


        
        
    def BuildFacetNormImage(self):
        if self.NormImage!=None: return
        print>>log,"  Building Facet-normalisation image"
        nch,npol=self.nch,self.npol
        _,_,NPixOut,NPixOut=self.OutImShape
        NormImage=np.zeros((NPixOut,NPixOut),dtype=np.float32)
        for iFacet in self.DicoImager.keys():
            SharedMemName="%sSpheroidal"%(self.IdSharedMem)#"%sWTerm.Facet_%3.3i"%(self.IdSharedMem,0)
            SharedMemName="%sSpheroidal.Facet_%3.3i"%(self.IdSharedMem,iFacet)
            #SPhe=NpShared.UnPackListSquareMatrix(SharedMemName)[0]
            SPhe=NpShared.GiveArray(SharedMemName)
            
            xc,yc=self.DicoImager[iFacet]["pixCentral"]
            #NpixFacet=self.DicoGridMachine[iFacet]["Dirty"][Channel].shape[2]
            NpixFacet=self.DicoImager[iFacet]["NpixFacetPadded"]

            Aedge,Bedge=GiveEdges((xc,yc),NPixOut,(NpixFacet/2,NpixFacet/2),NpixFacet)
            x0d,x1d,y0d,y1d=Aedge
            x0p,x1p,y0p,y1p=Bedge

            SpacialWeigth=self.SpacialWeigth[iFacet].T[::-1,:]
            SW=SpacialWeigth[::-1,:].T[x0p:x1p,y0p:y1p]
            NormImage[x0d:x1d,y0d:y1d]+=SW#Sphe



        nx,nx=NormImage.shape
        self.NormImage=NormImage
        self.NormImageReShape=self.NormImage.reshape((1,1,nx,nx))
        

    def FacetsToIm_Channel(self,BeamWeightImage=False):
        T=ClassTimeIt.ClassTimeIt("FacetsToIm_Channel")
        T.disable()
        Image=self.GiveEmptyMainField()

        nch,npol,NPixOut,NPixOut=self.OutImShape


        self.BuildFacetNormImage()

        if BeamWeightImage:
            print>>log, "Combining facets to average Jones-amplitude image"
        else:
            print>>log, "Combining facets to residual image"
            

        NormImage=self.NormImage

        for iFacet in self.DicoImager.keys():
                
            SharedMemName="%sSpheroidal.Facet_%3.3i"%(self.IdSharedMem,iFacet)
            SPhe=NpShared.GiveArray(SharedMemName)
            

            xc,yc=self.DicoImager[iFacet]["pixCentral"]
            NpixFacet=self.DicoGridMachine[iFacet]["Dirty"][0].shape[2]
            
            Aedge,Bedge=GiveEdges((xc,yc),NPixOut,(NpixFacet/2,NpixFacet/2),NpixFacet)
            x0main,x1main,y0main,y1main=Aedge
            x0facet,x1facet,y0facet,y1facet=Bedge
            
            #print "#%3.3i %s"%(iFacet,str(self.DicoImager[iFacet]["SumJones"][0]/self.DicoImager[iFacet]["SumJones"][1]))

            for Channel in range(self.VS.NFreqBands):
            
            
                ThisSumWeights=self.DicoImager[iFacet]["SumWeights"][Channel]
                ThisSumJones=1.
            
                # if BeamWeightImage:
                #     ThisSumSqWeights=self.DicoImager[iFacet]["SumJones"][1][Channel]
                #     if ThisSumSqWeights==0: ThisSumSqWeights=1.
                #     ThisSumJones=self.DicoImager[iFacet]["SumJones"][0][Channel]/ThisSumSqWeights
                #     if ThisSumJones==0:
                #         ThisSumJones=1.

                ThisSumSqWeights=self.DicoImager[iFacet]["SumJones"][1][Channel]
                if ThisSumSqWeights==0: ThisSumSqWeights=1.
                ThisSumJones=self.DicoImager[iFacet]["SumJones"][0][Channel]/ThisSumSqWeights
                if ThisSumJones==0:
                    ThisSumJones=1.

            
                SpacialWeigth=self.SpacialWeigth[iFacet].T[::-1,:]
                T.timeit("3")
                for pol in range(npol):
                    sumweight=ThisSumWeights[pol]#ThisSumWeights.reshape((nch,npol,1,1))[Channel, pol, 0, 0]
                    
                    if BeamWeightImage:
                        Im=SpacialWeigth[::-1,:].T[x0facet:x1facet,y0facet:y1facet]*ThisSumJones
                    else:
                    
                        Im=self.DicoGridMachine[iFacet]["Dirty"][Channel][pol].copy()
                        Im/=SPhe.real
                        Im[SPhe<1e-3]=0
                        Im=(Im[::-1,:].T.real/sumweight)
                        SW=SpacialWeigth[::-1,:].T
                        Im*=SW

                        Im/=np.sqrt(ThisSumJones)
                        #Im/=(ThisSumJones)

                        Im=Im[x0facet:x1facet,y0facet:y1facet]
                
                
                    Image[Channel,pol,x0main:x1main,y0main:y1main]+=Im


        for Channel in range(self.VS.NFreqBands):
            for pol in range(npol):
                Image[Channel,pol]/=NormImage
 


        #self.ToCasaImage(self.NormImage.reshape((1,1,nx,nx)),Fits=True,ImageName="NormImage")
        # stop

        # for ch in range(nch):
        #     for pol in range(npol):
        #         self.Image[ch,pol]=self.Image[ch,pol].T[::-1,:]
        

        # Image/=self.SumWeights.reshape((nch,npol,1,1))

        return Image

    def GiveNormImage(self):
        Image=self.GiveEmptyMainField()
        nch,npol=self.nch,self.npol
        _,_,NPixOut,NPixOut=self.OutImShape
        SharedMemName="%sSpheroidal"%(self.IdSharedMemData)
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
        for iFacet in sorted(self.DicoImager.keys()):
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

    def InitGrids(self):
        
        for iFacet in self.DicoGridMachine.keys():
            NX=self.DicoImager[iFacet]["NpixFacetPadded"]
            GridName="%sGridFacet.%3.3i"%(self.IdSharedMem,iFacet)
            self.DicoGridMachine[iFacet]["Dirty"]=NpShared.zeros(GridName,(self.VS.NFreqBands,self.npol,NX,NX),self.CType)
            self.DicoGridMachine[iFacet]["Dirty"]+=1
            self.DicoGridMachine[iFacet]["Dirty"].fill(0)

    def ReinitDirty(self):
        self.SumWeights.fill(0)
        self.IsDirtyInit=True
        for iFacet in self.DicoGridMachine.keys():
            NX=self.DicoImager[iFacet]["NpixFacetPadded"]
            GridName="%sGridFacet.%3.3i"%(self.IdSharedMem,iFacet)
            #self.DicoGridMachine[iFacet]["Dirty"]=NpShared.zeros(GridName,(self.VS.NFreqBands,self.npol,NX,NX),self.CType)
            self.DicoGridMachine[iFacet]["Dirty"]=np.ones((self.VS.NFreqBands,self.npol,NX,NX),self.FType)
            self.DicoGridMachine[iFacet]["Dirty"].fill(0)
            #self.DicoGridMachine[iFacet]["Dirty"]+=1
            #self.DicoGridMachine[iFacet]["Dirty"].fill(0)

            #     if "Dirty" in self.DicoGridMachine[iFacet].keys():
            #         for Channel in self.DicoGridMachine[iFacet]["Dirty"].keys():
            #             self.DicoGridMachine[iFacet]["Dirty"][Channel].fill(0)
            #     if "GM" in self.DicoGridMachine[iFacet].keys():
            #         self.DicoGridMachine[iFacet]["GM"].reinitGrid() # reinitialise sumWeights

            self.DicoImager[iFacet]["SumWeights"] = np.zeros((self.VS.NFreqBands,self.npol),np.float64)
            self.DicoImager[iFacet]["SumJones"]   = np.zeros((2,self.VS.NFreqBands),np.float64)
            self.DicoImager[iFacet]["SumJonesChan"]=[]
            for iMS in range(self.VS.nMS):
                MS=self.VS.ListMS[iMS]
                nVisChan=MS.ChanFreq.size
                self.DicoImager[iFacet]["SumJonesChan"].append(np.zeros((2,nVisChan),np.float64))
            
        # if self.Parallel:
        #     V=self.IM.CI.E.GiveSubCluster("Imag")["V"]
        #     LaunchAndCheck(V,'execfile("%s/Scripts/ScriptReinitGrids.py")'%self.GD.HYPERCAL_DIR)

    def CalcDirtyImagesParallel(self,times,uvwIn,visIn,flag,A0A1,W=None,doStack=True,Channel=0):
        # the input parameters are not actually used, see
        ## https://github.com/cyriltasse/DDFacet/issues/32#issuecomment-176072113

        NCPU=self.NCPU

        NFacets=len(self.DicoImager.keys())

        work_queue = multiprocessing.JoinableQueue()


        PSFMode=False
        if self.DoPSF:
            #visIn.fill(1)
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
                           IdSharedMemData=self.IdSharedMemData,
                           ApplyCal=self.ApplyCal,
                           SpheNorm=SpheNorm,
                           PSFMode=PSFMode)
            workerlist.append(W)
            workerlist[ii].start()

        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="  Gridding ", HeaderSize=10,TitleSize=13)
        pBAR.disable()
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

            self.DicoImager[iFacet]["SumWeights"][Channel]+=DicoResult["Weights"]
            self.DicoImager[iFacet]["SumJones"][Channel]+=DicoResult["SumJones"]

            # if iFacet==0:
            #     ThisSumWeights=DicoResult["Weights"]
            #     self.SumWeights+=ThisSumWeights

            DirtyName=DicoResult["DirtyName"]
            ThisDirty=NpShared.GiveArray(DirtyName)
            #print "minmax facet = %f %f"%(ThisDirty.min(),ThisDirty.max())

            if (doStack==True)&("Dirty" in self.DicoGridMachine[iFacet].keys()):
                #print>>log, (iFacet,Channel)
                if Channel in self.DicoGridMachine[iFacet]["Dirty"].keys():
                    self.DicoGridMachine[iFacet]["Dirty"][Channel]+=ThisDirty
                else:
                    self.DicoGridMachine[iFacet]["Dirty"][Channel]=ThisDirty
                #print "minmax stack = %f %f"%(self.DicoGridMachine[iFacet]["Dirty"].min(),self.DicoGridMachine[iFacet]["Dirty"].max())
            else:
                self.DicoGridMachine[iFacet]["Dirty"]={}
                self.DicoGridMachine[iFacet]["Dirty"][Channel]=ThisDirty

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
        
        NpShared.PackListArray("%sModelImage"%self.IdSharedMemData,ListModelImage)
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
                           IdSharedMemData=self.IdSharedMemData,
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

        NpShared.DelArray("%sModelImage"%self.IdSharedMemData)
            
        return True

        
    def GiveGM(self,iFacet):
        
        GridMachine=ClassDDEGridMachine.ClassDDEGridMachine(self.GD,#RaDec=self.DicoImager[iFacet]["RaDec"],
                                                            self.DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
                                                            self.DicoImager[iFacet]["DicoConfigGM"]["Npix"],
                                                            lmShift=self.DicoImager[iFacet]["lmShift"],
                                                            IdSharedMem=self.IdSharedMem,
                                                            IdSharedMemData=self.IdSharedMemData,
                                                            IDFacet=iFacet,
                                                            SpheNorm=self.SpheNorm)#,
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
                 IdSharedMemData=None,
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
        self.IdSharedMemData=IdSharedMemData
        self.Apply_killMS=(GD["DDESolutions"]["DDSols"]!="")&(GD["DDESolutions"]["DDSols"]!=None)
        self.Apply_Beam=(GD["Beam"]["BeamModel"]!=None)
        self.ApplyCal=(self.Apply_killMS)|(self.Apply_Beam)
        self.SpheNorm=SpheNorm
        self.PSFMode=PSFMode


    def shutdown(self):
        self.exit.set()

    def GiveGM(self,iFacet):
        GridMachine=ClassDDEGridMachine.ClassDDEGridMachine(self.GD,#RaDec=self.DicoImager[iFacet]["RaDec"],
                                                            self.DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
                                                            self.DicoImager[iFacet]["DicoConfigGM"]["Npix"],
                                                            lmShift=self.DicoImager[iFacet]["lmShift"],
                                                            IdSharedMem=self.IdSharedMem,
                                                            IdSharedMemData=self.IdSharedMemData,
                                                            IDFacet=iFacet,
                                                            SpheNorm=self.SpheNorm)#,
        return GridMachine
        
    def GiveDicoJonesMatrices(self):
        DicoJonesMatrices=None
        # if self.PSFMode:
        #     return None

        if self.ApplyCal:
            DicoJonesMatrices={}

        if self.Apply_killMS:
            DicoJones_killMS=NpShared.SharedToDico("%sJonesFile_killMS"%self.IdSharedMemData)
            DicoJonesMatrices["DicoJones_killMS"]=DicoJones_killMS
            DicoJonesMatrices["DicoJones_killMS"]["MapJones"]=NpShared.GiveArray("%sMapJones_killMS"%self.IdSharedMemData)
            DicoClusterDirs_killMS=NpShared.SharedToDico("%sDicoClusterDirs_killMS"%self.IdSharedMemData)
            DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"]=DicoClusterDirs_killMS

        if self.Apply_Beam:
            DicoJones_Beam=NpShared.SharedToDico("%sJonesFile_Beam"%self.IdSharedMemData)
            DicoJonesMatrices["DicoJones_Beam"]=DicoJones_Beam
            DicoJonesMatrices["DicoJones_Beam"]["MapJones"]=NpShared.GiveArray("%sMapJones_Beam"%self.IdSharedMemata)
            DicoClusterDirs_Beam=NpShared.SharedToDico("%sDicoClusterDirs_Beam"%self.IdSharedMemData)
            DicoJonesMatrices["DicoJones_Beam"]["DicoClusterDirs"]=DicoClusterDirs_Beam

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
                DATA=NpShared.SharedToDico("%sDicoData"%self.IdSharedMemData)
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
                Dirty=GridMachine.put(times,uvwThis,visThis,flagsThis,A0A1,W,DoNormWeights=False, DicoJonesMatrices=DicoJonesMatrices,freqs=freqs,DoPSF=self.PSFMode)#,doStack=False)

                DirtyName="%sImageFacet.%3.3i"%(self.IdSharedMem,iFacet)
                _=NpShared.ToShared(DirtyName,Dirty)
                del(Dirty)
                Sw=GridMachine.SumWeigths.copy()
                SumJones=GridMachine.SumJones.copy()
                del(GridMachine)

                self.result_queue.put({"Success":True,"iFacet":iFacet,"DirtyName":DirtyName,"Weights":Sw,"SumJones":SumJones})
                

                # gc.collect()
                # print "sleeping"
                # time.sleep(10)

                # self.result_queue.put({"Success":True})

            elif self.Mode=="DeGrid":
                
                GridMachine=self.GiveGM(iFacet)
                DATA=NpShared.SharedToDico("%sDicoData"%self.IdSharedMemData)
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
                ModelIm = NpShared.UnPackListArray("%sModelImage"%self.IdSharedMemData)[iFacet]
                vis=GridMachine.get(times,uvwThis,visThis,flagsThis,A0A1,ModelIm,DicoJonesMatrices=DicoJonesMatrices,freqs=freqs)
                
                self.result_queue.put({"Success":True,"iFacet":iFacet})
#            print "Done %i"%iFacet







