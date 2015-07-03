import numpy as np
from DDFacet.Imager import ClassFacetMachine
from DDFacet.Other.progressbar import ProgressBar
import multiprocessing
import ClassDDEGridMachine
import numpy as np
import pylab
import ClassCasaImage
import pyfftw
from DDFacet.ToolsDir import ModCoord
from DDFacet.Other import MyPickle
from DDFacet.Other import MyLogger
import time
from DDFacet.Other import ModColor
from DDFacet.Array import NpShared
from DDFacet.ToolsDir import ModFFTW
import pyfftw
from scipy.spatial import Voronoi
from SkyModel.Sky import ModVoronoi

log=MyLogger.getLogger("ClassFacetMachineTessel")
MyLogger.setSilent("MyLogger")
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.Imager.ClassImToGrid import ClassImToGrid
from matplotlib.path import Path




class ClassFacetMachineTessel(ClassFacetMachine.ClassFacetMachine):

    def __init__(self,*args,**kwargs):
        ClassFacetMachine.ClassFacetMachine.__init__(self,*args,**kwargs)
        
    def appendMainField(self,Npix=512,Cell=10.,NFacets=5,
                        Support=11,OverS=5,Padding=1.2,
                        wmax=10000,Nw=11,RaDecRad=(0.,0.),
                        ImageName="Facet.image"):
        

        Cell=self.GD["ImagerMainFacet"]["Cell"]

        self.ImageName=ImageName
        if self.DoPSF:
            Npix*=1

        MS=self.VS.MS
        self.LraFacet=[]
        self.LdecFacet=[]
        
        
        ChanFreq=self.VS.MS.ChanFreq.flatten()
        self.ChanFreq=ChanFreq
        
        self.Cell=Cell
        self.CellSizeRad=(Cell/3600.)*np.pi/180.
        rac,decc=MS.radec
        self.MainRaDec=(rac,decc)
        self.nch=1
        self.NChanGrid=1
        self.SumWeights=np.zeros((self.NChanGrid,self.npol),float)

        self.CoordMachine=ModCoord.ClassCoordConv(rac,decc)
        # self.setFacetsLocsSquare()
        self.setFacetsLocsTessel()


    def setFacetsLocsTessel(self):

        Npix=self.GD["ImagerMainFacet"]["Npix"]
        NFacets=self.GD["ImagerMainFacet"]["NFacets"]
        Padding=self.GD["ImagerMainFacet"]["Padding"]
        self.Padding=Padding
        Npix,_=EstimateNpix(float(Npix),Padding=1)
        self.Npix=Npix
        self.OutImShape=(self.nch,self.npol,self.Npix,self.Npix)    

        
        RadiusTot=self.CellSizeRad*self.Npix/2
        self.RadiusTot=RadiusTot
        
        SolsFile=self.GD["DDESolutions"]["DDSols"]
        if not(".npz" in SolsFile):
            Method=SolsFile
            ThisMSName=reformat.reformat(os.path.abspath(self.MS.MSName),LastSlash=False)
            SolsFile="%s/killMS.%s.sols.npz"%(ThisMSName,Method)
            
        ClusterNodes=np.load(SolsFile)["ClusterCat"]



        ClusterNodes=ClusterNodes.view(np.recarray)
        raNode=ClusterNodes.ra
        decNode=ClusterNodes.dec
        lFacet,mFacet=self.CoordMachine.radec2lm(raNode,decNode)

        self.FacetCat=np.zeros((lFacet.size,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('SumI',np.float),
                                                     ("Cluster",int),
                                                     ("l",np.float),("m",np.float),
                                                     ("I",np.float)])
        self.FacetCat=self.FacetCat.view(np.recarray)
        self.FacetCat.I=1
        self.FacetCat.SumI=1

        print>>log,"Sizes (%i facets):"%(self.FacetCat.shape[0])
        print>>log,"   - Main field :   [%i x %i] pix"%(self.Npix,self.Npix)

        self.DicoImager={}
        
        xy=np.zeros((lFacet.size,2),np.float32)
        xy[:,0]=lFacet
        xy[:,1]=mFacet
        vor = Voronoi(xy)
        regions, vertices = ModVoronoi.voronoi_finite_polygons_2d(vor,radius=2*RadiusTot)
        
        
        l_m_Diam=np.zeros((len(regions),4),np.float32)
        l_m_Diam[:,3]=np.arange(len(regions))
        #X,Y=np.mgrid[-RadiusTot:RadiusTot:5000*1j,-RadiusTot:RadiusTot:5000*1j]
        Np=100000
        X=(np.random.rand(Np)*2-1.)*RadiusTot
        Y=(np.random.rand(Np)*2-1.)*RadiusTot

        self.CornersImageTot=np.array([[-RadiusTot,-RadiusTot],
                                       [RadiusTot,-RadiusTot],
                                       [RadiusTot,RadiusTot],
                                       [-RadiusTot,RadiusTot]])

        # for iCorner in Corners:
        #     IN=False
        #     for iFacet in range(len(regions)):
        #         region=regions[iFacet]
        #         polygon0 = vertices[region]
        #         P=polygon0.tolist()
        #         polygon=np.array(P+[P[0]])
        #         mpath = Path( polygon )
        #         if 

        from SkyModel.Sky import ModVoronoiToReg
        rac,decc=self.MainRaDec
        VM=ModVoronoiToReg.VoronoiToReg(rac,decc,lFacet,mFacet)
        regFile="%s.FacetMachine.tessel.reg"%self.ImageName
        VM.ToReg(regFile,radius=2*RadiusTot)


        D={}
        for iFacet in range(len(regions)):
            print iFacet,"/",len(regions)
            region=regions[iFacet]
            D[iFacet]={}
            polygon0 = vertices[region]
            P=polygon0.tolist()
            polygon=np.array(P+[P[0]])
            D[iFacet]["Polygon"]=polygon
            lPoly,mPoly=polygon.T

            l0=np.max([-RadiusTot,lPoly.min()])
            l1=np.min([RadiusTot,lPoly.max()])
            m0=np.max([-RadiusTot,mPoly.min()])
            m1=np.min([RadiusTot,mPoly.max()])
            dl=l1-l0
            dm=m1-m0
            diam=np.max([dl,dm])
            
            # rect=np.zeros((4,2),np.float32)
            # rect[:,0]=np.array([-RadiusTot,RadiusTot,RadiusTot,-RadiusTot])
            # rect[:,1]=np.array([-RadiusTot,-RadiusTot,RadiusTot,RadiusTot])
            
            mpath = Path( polygon )
            XY = np.dstack((X, Y))
            XY_flat = XY.reshape((-1, 2))
            mask_flat = mpath.contains_points(XY_flat)
            mask=mask_flat.reshape(X.shape)
            
            lc=np.sum(X*mask)/np.sum(mask)
            mc=np.sum(Y*mask)/np.sum(mask)
            dl=np.max(np.abs(X[mask==1]-lc))
            dm=np.max(np.abs(Y[mask==1]-mc))
            diam=2*np.max([dl,dm])
            
        
            l_m_Diam[iFacet,0]=lc
            l_m_Diam[iFacet,1]=mc
            l_m_Diam[iFacet,2]=diam

        self.SpacialWeigth={}
        self.DicoImager={} 
        indDiam=np.argsort(l_m_Diam[:,2])[::-1]
        l_m_Diam=l_m_Diam[indDiam]
        for iFacet in range(l_m_Diam.shape[0]):
            self.DicoImager[iFacet]={}
            self.DicoImager[iFacet]["Polygon"]=D[l_m_Diam[iFacet,3]]["Polygon"]
            l0=round(l_m_Diam[iFacet,0]/self.CellSizeRad)*self.CellSizeRad
            m0=round(l_m_Diam[iFacet,1]/self.CellSizeRad)*self.CellSizeRad
            diam=round(l_m_Diam[iFacet,2]/self.CellSizeRad)*self.CellSizeRad
            self.AppendFacet(iFacet,l0,m0,diam)
            self.MakeMasksTessel(iFacet)

        NpixMax=np.max([self.DicoImager[iFacet]["NpixFacet"] for iFacet in self.DicoImager.keys()])
        NpixMaxPadded=np.max([self.DicoImager[iFacet]["NpixFacetPadded"] for iFacet in self.DicoImager.keys()])
        self.PaddedGridShape=(1,1,NpixMaxPadded,NpixMaxPadded)
        self.FacetShape=(1,1,NpixMax,NpixMax)

        
        
        

    def MakeMasksTessel(self,iFacet):
        print>>log, "Making mask for facet %i"%iFacet
        Npix=self.DicoImager[iFacet]["NpixFacetPadded"]
        l0,l1,m0,m1=self.DicoImager[iFacet]["lmExtentPadded"]
        X, Y = np.mgrid[l0:l1:Npix*1j,m0:m1:Npix*1j]
        XY = np.dstack((X, Y))
        XY_flat = XY.reshape((-1, 2))
        vertices=self.DicoImager[iFacet]["Polygon"]
        mpath = Path( vertices ) # the vertices of the polygon
        mask_flat = mpath.contains_points(XY_flat)

        mask = mask_flat.reshape(X.shape)

        mpath = Path(self.CornersImageTot)
        mask_flat2 = mpath.contains_points(XY_flat)
        mask2 = mask_flat2.reshape(X.shape)
        mask[mask2==0]=0

        GaussPars=(10,10,0)

        SpacialWeigth=np.float32(mask.reshape((1,1,Npix,Npix)))

        # import pylab
        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.imshow(SpacialWeigth.reshape((Npix,Npix)),vmin=0,vmax=1.1,cmap="gray")
        SpacialWeigth=ModFFTW.ConvolveGaussian(SpacialWeigth,CellSizeRad=1,GaussPars=[GaussPars])
        SpacialWeigth=SpacialWeigth.reshape((Npix,Npix))
        SpacialWeigth/=np.max(SpacialWeigth)
        # pylab.subplot(1,2,2)
        # pylab.imshow(SpacialWeigth,vmin=0,vmax=1.1,cmap="gray")
        # pylab.draw()
        # pylab.show()


        #SpacialWeigth[np.abs(SpacialWeigth)<1e-2]=0.
        self.SpacialWeigth[iFacet]=SpacialWeigth
        



    def AppendFacet(self,iFacet,l0,m0,diam):
    

        DicoConfigGM=None
        lmShift=(l0,m0)
        self.DicoImager[iFacet]["lmShift"]=lmShift
        #CellRad=(Cell/3600.)*np.pi/180.
        
        
        raFacet,decFacet=self.CoordMachine.lm2radec(np.array([lmShift[0]]),np.array([lmShift[1]]))

        NpixFacet,_=EstimateNpix(diam/self.CellSizeRad,Padding=1)
        _,NpixPaddedGrid=EstimateNpix(NpixFacet,Padding=self.Padding)

        diam=NpixFacet*self.CellSizeRad
        diamPadded=NpixPaddedGrid*self.CellSizeRad
        RadiusFacet=diam*0.5
        RadiusFacetPadded=diamPadded*0.5
        self.DicoImager[iFacet]["lmDiam"]=RadiusFacet
        self.DicoImager[iFacet]["lmDiamPadded"]=RadiusFacetPadded
        self.DicoImager[iFacet]["RadiusFacet"]=RadiusFacet
        self.DicoImager[iFacet]["RadiusFacetPadded"]=RadiusFacetPadded
        self.DicoImager[iFacet]["lmExtent"]=l0-RadiusFacet,l0+RadiusFacet,m0-RadiusFacet,m0+RadiusFacet
        self.DicoImager[iFacet]["lmExtentPadded"]=l0-RadiusFacetPadded,l0+RadiusFacetPadded,m0-RadiusFacetPadded,m0+RadiusFacetPadded


        DicoConfigGM={"Npix":NpixFacet,
                      "Cell":self.GD["ImagerMainFacet"]["Cell"],
                      "ChanFreq":self.ChanFreq,
                      "DoPSF":False,
                      "Support":self.GD["ImagerCF"]["Support"],
                      "OverS":self.GD["ImagerCF"]["OverS"],
                      "wmax":self.GD["ImagerCF"]["wmax"],
                      "Nw":self.GD["ImagerCF"]["Nw"],
                      "WProj":True,
                      "DoDDE":self.DoDDE,
                      "Padding":self.GD["ImagerMainFacet"]["Padding"]}



        _,_,NpixOutIm,NpixOutIm=self.OutImShape

        self.DicoImager[iFacet]["l0m0"]=lmShift#self.CoordMachine.radec2lm(raFacet,decFacet)
        self.DicoImager[iFacet]["RaDec"]=raFacet[0],decFacet[0]
        self.LraFacet.append(raFacet[0])
        self.LdecFacet.append(decFacet[0])
        xc,yc=l0/self.CellSizeRad+NpixOutIm/2,m0/self.CellSizeRad+NpixOutIm/2
        print xc,yc
        self.DicoImager[iFacet]["pixCentral"]=xc,yc
        self.DicoImager[iFacet]["pixExtent"]=round(xc-NpixFacet/2),round(xc+NpixFacet/2+1),round(yc-NpixFacet/2),round(yc+NpixFacet/2+1)
        self.DicoImager[iFacet]["NpixFacet"]=NpixFacet
        self.DicoImager[iFacet]["NpixFacetPadded"]=NpixPaddedGrid
        self.DicoImager[iFacet]["DicoConfigGM"]=DicoConfigGM
        self.DicoImager[iFacet]["IDFacet"]=iFacet
        #print self.DicoImager[iFacet]
        
        self.FacetCat.ra[iFacet]=raFacet[0]
        self.FacetCat.dec[iFacet]=decFacet[0]
        l,m=self.DicoImager[iFacet]["l0m0"]
        self.FacetCat.l[iFacet]=l
        self.FacetCat.m[iFacet]=m
        self.FacetCat.Cluster[iFacet]=iFacet



    def setFacetsLocsSquare(self):
        Npix=self.GD["ImagerMainFacet"]["Npix"]
        NFacets=self.GD["ImagerMainFacet"]["NFacets"]
        Padding=self.GD["ImagerMainFacet"]["Padding"]
        self.Padding=Padding
        NpixFacet,_=EstimateNpix(float(Npix)/NFacets,Padding=1)
        Npix=NpixFacet*NFacets
        self.Npix=Npix
        self.OutImShape=(self.nch,self.npol,self.Npix,self.Npix)    
        _,NpixPaddedGrid=EstimateNpix(NpixFacet,Padding=Padding)
        self.NpixPaddedFacet=NpixPaddedGrid
        self.NpixFacet=NpixFacet
        self.FacetShape=(self.nch,self.npol,NpixFacet,NpixFacet)
        self.PaddedGridShape=(self.NChanGrid,self.npol,NpixPaddedGrid,NpixPaddedGrid)
        

        print>>log,"Sizes (%i x %i facets):"%(NFacets,NFacets)
        print>>log,"   - Main field :   [%i x %i] pix"%(self.Npix,self.Npix)
        print>>log,"   - Each facet :   [%i x %i] pix"%(NpixFacet,NpixFacet)
        print>>log,"   - Padded-facet : [%i x %i] pix"%(NpixPaddedGrid,NpixPaddedGrid)

        

        ############################

        self.NFacets=NFacets
        lrad=Npix*self.CellSizeRad*0.5
        self.ImageExtent=[-lrad,lrad,-lrad,lrad]

        lfacet=NpixFacet*self.CellSizeRad*0.5
        lcenter_max=lrad-lfacet
        lFacet,mFacet,=np.mgrid[-lcenter_max:lcenter_max:(NFacets)*1j,-lcenter_max:lcenter_max:(NFacets)*1j]
        lFacet=lFacet.flatten()
        mFacet=mFacet.flatten()
        x0facet,y0facet=np.mgrid[0:Npix:NpixFacet,0:Npix:NpixFacet]
        x0facet=x0facet.flatten()
        y0facet=y0facet.flatten()

        #print "Append1"; self.IM.CI.E.clear()
        
        
        self.DicoImager={}
        for iFacet in range(lFacet.size):
            self.DicoImager[iFacet]={}





        #print "Append2"; self.IM.CI.E.clear()


        self.FacetCat=np.zeros((lFacet.size,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('SumI',np.float),
                                                     ("Cluster",int),
                                                     ("l",np.float),("m",np.float),
                                                     ("I",np.float)])
        self.FacetCat=self.FacetCat.view(np.recarray)
        self.FacetCat.I=1
        self.FacetCat.SumI=1

        
        for iFacet in range(lFacet.size):
            l0=x0facet[iFacet]*self.CellSizeRad
            m0=y0facet[iFacet]*self.CellSizeRad
            l0=lFacet[iFacet]
            m0=mFacet[iFacet]


            #print x0facet[iFacet],y0facet[iFacet],l0,m0
            self.AppendFacet(iFacet,l0,m0,NpixFacet*self.CellSizeRad)

        self.DicoImagerCentralFacet=self.DicoImager[lFacet.size/2]

        self.SetLogModeSubModules("Silent")
        self.MakeREG()

    #===========================================

    def ImToGrids(self,Image):
        Im2Grid=ClassImToGrid(OverS=self.GD["ImagerCF"]["OverS"],GD=self.GD)
        nch,npol=self.nch,self.npol
        for iFacet in self.DicoImager.keys():
            
            SharedMemName="%sSpheroidal.Facet_%3.3i"%(self.IdSharedMem,iFacet)
            SPhe=NpShared.GiveArray(SharedMemName)
            SpacialWeight=self.SpacialWeigth[iFacet]
            Grid,_=Im2Grid.GiveGridTessel(Image,self.DicoImager,iFacet,self.NormImage,SPhe,SpacialWeight)

            GridSharedMemName="%sModelGrid.Facet_%3.3i"%(self.IdSharedMem,iFacet)
            NpShared.ToShared(GridSharedMemName,Grid)

    def GiveVisParallel(self,times,uvwIn,visIn,flag,A0A1,ModelImage):
        NCPU=self.NCPU
        #visOut=np.zeros_like(visIn)


        print>>log, "Model image to facets ..."
        self.ImToGrids(ModelImage)

        NFacets=len(self.DicoImager.keys())
        # ListModelImage=[]
        # for iFacet in self.DicoImager.keys():
        #     ListModelImage.append(self.DicoImager[iFacet]["ModelFacet"])
        # NpShared.PackListArray("%sModelImage"%self.IdSharedMem,ListModelImage)
        # del(ListModelImage)
        # for iFacet in self.DicoImager.keys():
        #     del(self.DicoImager[iFacet]["ModelFacet"])

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

        NpShared.DelAll("%sModelGrid"%(self.IdSharedMem))
            
        return True


#===============================================
#===============================================
#===============================================
#===============================================

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
                                                            IdSharedMem=self.IdSharedMem,IDFacet=iFacet,
                                                            SpheNorm=self.SpheNorm)#,
        return GridMachine
        
    def GiveDicoJonesMatrices(self):
        DicoJonesMatrices=None
        if self.PSFMode:
            return None

        if self.ApplyCal:
            DicoJonesMatrices={}

        if self.Apply_killMS:
            DicoJones_killMS=NpShared.SharedToDico("%sJonesFile_killMS"%self.IdSharedMem)
            DicoJonesMatrices["DicoJones_killMS"]=DicoJones_killMS
            DicoJonesMatrices["DicoJones_killMS"]["MapJones"]=NpShared.GiveArray("%sMapJones_killMS"%self.IdSharedMem)
            DicoClusterDirs_killMS=NpShared.SharedToDico("%sDicoClusterDirs_killMS"%self.IdSharedMem)
            DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"]=DicoClusterDirs_killMS

        if self.Apply_Beam:
            DicoJones_Beam=NpShared.SharedToDico("%sJonesFile_Beam"%self.IdSharedMem)
            DicoJonesMatrices["DicoJones_Beam"]=DicoJones_Beam
            DicoJonesMatrices["DicoJones_Beam"]["MapJones"]=NpShared.GiveArray("%sMapJones_Beam"%self.IdSharedMem)
            DicoClusterDirs_Beam=NpShared.SharedToDico("%sDicoClusterDirs_Beam"%self.IdSharedMem)
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
                GridSharedMemName="%sModelGrid.Facet_%3.3i"%(self.IdSharedMem,iFacet)
                ModelGrid = NpShared.GiveArray(GridSharedMemName)
                vis=GridMachine.get(times,uvwThis,visThis,flagsThis,A0A1,ModelGrid,ImToGrid=False,DicoJonesMatrices=DicoJonesMatrices,freqs=freqs)
                
                self.result_queue.put({"Success":True,"iFacet":iFacet})
#            print "Done %i"%iFacet







