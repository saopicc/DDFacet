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
        
        ClusterNodes=np.load("BOOTES24_SB100-109.2ch8s.ms/killMS.KAFCA.sols.npz")["ClusterCat"]
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

        self.DicoImager={}
        
        xy=np.zeros((lFacet.size,2),np.float32)
        xy[:,0]=lFacet
        xy[:,1]=mFacet
        vor = Voronoi(xy)
        regions, vertices = ModVoronoi.voronoi_finite_polygons_2d(vor)

        
        X,Y=np.mgrid[-RadiusTot:RadiusTot:5000*1j,-RadiusTot:RadiusTot:5000*1j]
        for iFacet in range(len(regions)):
            print iFacet,"/",len(regions)
            region=regions[iFacet]
            self.DicoImager[iFacet]={}
            polygon0 = vertices[region]
            P=polygon0.tolist()
            polygon=np.array(P+[P[0]])
            self.DicoImager[iFacet]["Polygon"]=polygon
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
            
            self.AppendFacet(iFacet,lc,mc,diam)

        NpixMax=np.max([self.DicoImager[iFacet]["NpixFacet"] for iFacet in self.DicoImager.keys()])
        NpixMaxPadded=np.max([self.DicoImager[iFacet]["NpixFacetPadded"] for iFacet in self.DicoImager.keys()])
        self.PaddedGridShape=(1,1,NpixMaxPadded,NpixMaxPadded)
        self.FacetShape=(1,1,NpixMax,NpixMax)

        self.MakeMasksTessel()
        
        

    def MakeMasksTessel(self):
        self.SpacialWeigth={}
        for iFacet in self.DicoImager.keys():
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
            GaussPars=(10,10,0)
            
            SpacialWeigth=ModFFTW.ConvolveGaussian(np.float32(mask.reshape((1,1,Npix,Npix))),CellSizeRad=1,GaussPars=[GaussPars])
            SpacialWeigth=SpacialWeigth.reshape((Npix,Npix))
            SpacialWeigth[np.abs(SpacialWeigth)<1e-3]=0.
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

        self.DicoImager[iFacet]["l0m0"]=self.CoordMachine.radec2lm(raFacet,decFacet)
        self.DicoImager[iFacet]["RaDec"]=raFacet[0],decFacet[0]
        self.LraFacet.append(raFacet[0])
        self.LdecFacet.append(decFacet[0])
        xc,yc=round(l0/self.CellSizeRad+NpixOutIm/2),round(m0/self.CellSizeRad+NpixOutIm/2)
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



