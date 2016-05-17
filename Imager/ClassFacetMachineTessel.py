from DDFacet.Imager import ClassFacetMachine
from DDFacet.Other.progressbar import ProgressBar
import multiprocessing
import numpy as np
import pylab
from DDFacet.ToolsDir import ModCoord
from DDFacet.Other import MyPickle
from DDFacet.Array import NpShared
from DDFacet.ToolsDir import ModFFTW
from scipy.spatial import Voronoi
from SkyModel.Sky import ModVoronoi
from DDFacet.Other import reformat
import os
from DDFacet.ToolsDir.ModToolBox import EstimateNpix

from DDFacet.Imager.ClassImToGrid import ClassImToGrid
from matplotlib.path import Path
from SkyModel.Sky import ModVoronoiToReg
import Polygon
from DDFacet.ToolsDir import rad2hmsdms
from DDFacet.Other.ClassTimeIt import ClassTimeIt
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassFacetMachineTessel")
MyLogger.setSilent("MyLogger")


class ClassFacetMachineTessel(ClassFacetMachine.ClassFacetMachine):
    """
    This class extends the basic infrastructure set out in ClassFacetMachine to
    split the sky into a Voronoi tesselated sky. The projection, unprojection and resampling logic
    remains the same.
    """
    def __init__(self,*args,**kwargs):
        ClassFacetMachine.ClassFacetMachine.__init__(self,*args,**kwargs)
        self.FacetParallelEngine = WorkerImager

    def setFacetsLocs(self):
        NFacets = self.NFacets
        MSName=self.GD["VisData"]["MSName"]
        if ".txt" in MSName:
            f=open(MSName)
            Ls=f.readlines()
            f.close()
            MSName=[]
            for l in Ls:
                ll=l.replace("\n","")
                MSName.append(ll)
            MSName=MSName[0]

        
        SolsFile=self.GD["DDESolutions"]["DDSols"]
        if type(SolsFile)==list:
            SolsFile=self.GD["DDESolutions"]["DDSols"][0]

        if (SolsFile!="")&(not(".npz" in SolsFile)):
            Method=SolsFile
            ThisMSName=reformat.reformat(os.path.abspath(MSName),LastSlash=False)
            SolsFile="%s/killMS.%s.sols.npz"%(ThisMSName,Method)


        if SolsFile!="":
            ClusterNodes=np.load(SolsFile)["ClusterCat"]
            ClusterNodes=ClusterNodes.view(np.recarray)
            raNode=ClusterNodes.ra
            decNode=ClusterNodes.dec
            lFacet,mFacet=self.CoordMachine.radec2lm(raNode,decNode)
        else:

            CellSizeRad=(self.GD["ImagerMainFacet"]["Cell"]/3600.)*np.pi/180
            lrad=self.Npix*CellSizeRad*0.5
            NpixFacet=self.Npix/NFacets
            lfacet=NpixFacet*CellSizeRad*0.5
            lcenter_max=lrad-lfacet
            
            lFacet,mFacet,=np.mgrid[-lcenter_max:lcenter_max:(NFacets)*1j,-lcenter_max:lcenter_max:(NFacets)*1j]
            lFacet=lFacet.flatten()
            mFacet=mFacet.flatten()

        self.DicoImager={}
        
        xy=np.zeros((lFacet.size,2),np.float32)
        xy[:,0]=lFacet
        xy[:,1]=mFacet
        vor = Voronoi(xy,furthest_site=False)
        regions, vertices = ModVoronoi.voronoi_finite_polygons_2d(vor,radius=1.)
        LPolygon=[np.array(vertices[region]) for region in regions]
        #
        rac,decc=self.MainRaDec
        VM=ModVoronoiToReg.VoronoiToReg(rac,decc)
        regFile="%s.tessel.reg"%self.ImageName

        Np=100000
        X=(np.random.rand(Np)*2-1.)*self.RadiusTot
        Y=(np.random.rand(Np)*2-1.)*self.RadiusTot
        XY = np.dstack((X, Y))
        XY_flat = XY.reshape((-1, 2))

        ###########################################
        # SubDivide
        def GiveDiam(polygon):
            lPoly,mPoly=polygon.T
            l0=np.max([-self.RadiusTot,lPoly.min()])
            l1=np.min([self.RadiusTot,lPoly.max()])
            m0=np.max([-self.RadiusTot,mPoly.min()])
            m1=np.min([self.RadiusTot,mPoly.max()])
            dl=l1-l0
            dm=m1-m0
            diam=np.max([dl,dm])
            return diam,(l0,l1,m0,m1)

        DiamMax=1.5*np.pi/180
        DiamMin=0.1*np.pi/180
        
        def ClosePolygon(polygon):
            P=polygon.tolist()
            polygon=np.array(P+[P[0]])
            return polygon

        def GiveSubDivideRegions(polygonFacet,DMax):

            polygonFOV=self.CornersImageTot
            #polygonFOV=ClosePolygon(polygonFOV)
            PFOV=Polygon.Polygon(polygonFOV)

            #polygonFacet=ClosePolygon(polygonFacet)
            P0=Polygon.Polygon(polygonFacet)
            P0Cut=Polygon.Polygon(P0&PFOV)

            if P0Cut.nPoints()==0: return []

            polygonFacetCut=np.array(P0Cut[0])
            #polygonFacetCut=ClosePolygon(polygonFacetCut)

            diam,(l0,l1,m0,m1)=GiveDiam(polygonFacetCut)
            if diam<DMax: return [polygonFacetCut]

            Nl=int((l1-l0)/DMax)+1
            Nm=int((m1-m0)/DMax)+1
            dl=(l1-l0)/Nl
            dm=(m1-m0)/Nm
            lEdge=np.linspace(l0,l1,Nl+1)
            mEdge=np.linspace(m0,m1,Nm+1)
            lc=(lEdge[0:-1]+lEdge[1::])/2
            mc=(mEdge[0:-1]+mEdge[1::])/2
            LPoly=[]
            Lc,Mc=np.meshgrid(lc,mc)
            Lc=Lc.ravel().tolist()
            Mc=Mc.ravel().tolist()
            

            DpolySquare=np.array([[-dl,-dm],[dl,-dm],[dl,dm],[-dl,dm]])*0.5
            for lc,mc in zip(Lc,Mc):
                polySquare=DpolySquare.copy()#ClosePolygon(DpolySquare.copy())
                polySquare[:,0]+=lc
                polySquare[:,1]+=mc
                #polySquare=ClosePolygon(polySquare)
                P1=Polygon.Polygon(polySquare)

                POut=(P0Cut&P1)
                if POut.nPoints()==0: continue

                polyOut=np.array(POut[0])
                #polyOut=ClosePolygon(polyOut)
                LPoly.append(polyOut)

                # pylab.clf()
                # x,y=polygonFacetCut.T
                # pylab.plot(x,y,color="blue")
                # x,y=polygonFacet.T
                # pylab.plot(x,y,color="blue",ls=":",lw=3)
                # x,y=np.array(PFOV[0]).T
                # pylab.plot(x,y,color="black")
                # x,y=polySquare.T
                # pylab.plot(x,y,color="green",ls=":",lw=3)
                # x,y=polyOut.T
                # pylab.plot(x,y,color="red",ls="--",lw=3)
                # pylab.xlim(-0.03,0.03)
                # pylab.ylim(-0.03,0.03)
                # pylab.draw()
                # pylab.show(False)
                # pylab.pause(0.5)

            
            return LPoly
                
        def PlotPolygon(P,*args,**kwargs):
            for poly in P:
                x,y=ClosePolygon(np.array(poly)).T
                pylab.plot(x,y,*args,**kwargs)

        LPolygonNew=[]
        
        for iFacet in range(len(regions)):
            polygon=LPolygon[iFacet]
            ThisDiamMax=DiamMax
            SubReg=GiveSubDivideRegions(polygon,ThisDiamMax)

            LPolygonNew+=SubReg

        regFile="%s.FacetMachine.tessel.ReCut.reg"%self.ImageName
        #VM.PolygonToReg(regFile,LPolygonNew,radius=0.1,Col="green",labels=[str(i) for i in range(len(LPolygonNew))])


        DicoPolygon={}
        for iFacet in range(len(LPolygonNew)): 
            DicoPolygon[iFacet]={}
            poly=LPolygonNew[iFacet]
            DicoPolygon[iFacet]["poly"]=poly
            diam,(l0,l1,m0,m1)=GiveDiam(poly)
            DicoPolygon[iFacet]["diam"]=diam
            DicoPolygon[iFacet]["diamMin"]=np.min([(l1-l0),(m1-m0)])
            xc,yc=np.mean(poly[:,0]),np.mean(poly[:,1])
            DicoPolygon[iFacet]["xyc"]=xc,yc
            dSol=np.sqrt((xc-lFacet)**2+(yc-mFacet)**2)
            DicoPolygon[iFacet]["iSol"]=np.where(dSol==np.min(dSol))[0]



        from scipy.spatial import ConvexHull
        for iFacet in sorted(DicoPolygon.keys()):
            diam=DicoPolygon[iFacet]["diamMin"]
            # print iFacet,diam,DiamMin
            if diam<DiamMin:
                dmin=1e6
                xc0,yc0=DicoPolygon[iFacet]["xyc"]
                HasClosest=False
                for iFacetOther in sorted(DicoPolygon.keys()):
                    if iFacetOther==iFacet: continue
                    iSolOther=DicoPolygon[iFacetOther]["iSol"]
                    # print "  ",iSolOther,DicoPolygon[iFacet]["iSol"]
                    if iSolOther!=DicoPolygon[iFacet]["iSol"]:
                        continue
                    xc,yc=DicoPolygon[iFacetOther]["xyc"]
                    d=np.sqrt((xc-xc0)**2+(yc-yc0)**2)
                    if d<dmin:
                        dmin=d
                        iFacetClosest=iFacetOther
                        HasClosest=True
                if (HasClosest):
                    print>>log, "Merging facet #%i to #%i"%(iFacet,iFacetClosest)
                    P0=Polygon.Polygon(DicoPolygon[iFacet]["poly"])
                    P1=Polygon.Polygon(DicoPolygon[iFacetClosest]["poly"])
                    P2=(P0|P1)
                    POut=[]
                    for iP in range(len(P2)):
                        POut+=P2[iP]
                
                    poly=np.array(POut)
                    hull = ConvexHull(poly)
                    Contour=np.array([hull.points[hull.vertices,0],hull. points[hull.vertices,1]])
                    poly2=Contour.T
                    # poly2=hull.points
                    # pylab.clf()
                    # x,y=poly.T
                    # pylab.plot(x,y)
                    # x,y=poly2.T
                    # pylab.plot(x,y)
                    # # PlotPolygon(P0)
                    # # PlotPolygon(P1)
                    # # #PlotPolygon(P2,color="black")
                    # # x,y=poly2.T
                    # # PlotPolygon(x,y,color="black")
                    # pylab.draw()
                    # pylab.show()
                    # time.sleep(0.5)
                    
                    #poly2=np.array(P2[0])
                    del(DicoPolygon[iFacet])
                    DicoPolygon[iFacetClosest]["poly"]=poly2
                    DicoPolygon[iFacetClosest]["diam"]=GiveDiam(poly2)[0]
                    DicoPolygon[iFacetClosest]["xyc"]=np.mean(poly2[:,0]),np.mean(poly2[:,1])
        #stop
        LPolygonNew=[]
        for iFacet in sorted(DicoPolygon.keys()):
            LPolygonNew.append(DicoPolygon[iFacet]["poly"])

        
        # for iFacet in range(len(regions)):
        #     polygon=LPolygon[iFacet]
        #     ThisDiamMax=DiamMax
        #     while True:
        #         SubReg=GiveSubDivideRegions(polygon,ThisDiamMax)
        #         if SubReg==[]:
        #             break
        #         Diams=[GiveDiam(poly)[0] for poly in SubReg]
                
        #         if np.min(Diams)>DiamMin: break
        #         ThisDiamMax*=1.1
        #     LPolygonNew+=SubReg
        #     print 

        

        regFile="%s.tessel.reg"%self.GD["Images"]["ImageName"]
        #labels=["[F%i.C%i]"%(i,DicoPolygon[i]["iSol"]) for i in range(len(LPolygonNew))]
        #VM.PolygonToReg(regFile,LPolygonNew,radius=0.1,Col="green",labels=labels)

        VM.PolygonToReg(regFile,LPolygonNew,radius=0.1,Col="green")

        # pylab.clf()
        # x,y=LPolygonNew[11].T
        # pylab.plot(x,y)
        # pylab.draw()
        # pylab.show()
        # stop
        ###########################################

        NFacets=len(LPolygonNew)

        self.FacetCat=np.zeros((NFacets,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('SumI',np.float),
                                                     ("Cluster",int),
                                                     ("l",np.float),("m",np.float),
                                                     ("I",np.float)])
        self.FacetCat=self.FacetCat.view(np.recarray)
        self.FacetCat.I=1
        self.FacetCat.SumI=1
        print>>log,"Sizes (%i facets):"%(self.FacetCat.shape[0])
        print>>log,"   - Main field :   [%i x %i] pix"%(self.Npix,self.Npix)


        l_m_Diam=np.zeros((NFacets,4),np.float32)
        l_m_Diam[:,3]=np.arange(NFacets)

        D={}

        # Debug phase shift:
        l_phasor_rad = np.deg2rad(self.GD["Debugging"]["FacetPhaseShift"][0]/3600.0)
        m_phasor_rad = np.deg2rad(self.GD["Debugging"]["FacetPhaseShift"][1] / 3600.0)
        shift_px_l = l_phasor_rad / CellSizeRad
        shift_px_m = m_phasor_rad / CellSizeRad
        print>> log, "Shifting the facet centres to new reference centres " \
                     "(%.3f,%.3f) arcsec away [%.3f,%.3f]px" % (float(self.GD["Debugging"]["FacetPhaseShift"][0]),
                                                                float(self.GD["Debugging"]["FacetPhaseShift"][1]),
                                                                shift_px_l,shift_px_m)
        for iFacet in range(NFacets):
            D[iFacet]={}
            polygon=LPolygonNew[iFacet]
            D[iFacet]["Polygon"]=polygon
            lPoly,mPoly=polygon.T

            #This fits a bounding box around the polygon (note not axis aligned):
            ThisDiam,(l0,l1,m0,m1)=GiveDiam(polygon)
            #Get the centre of the bounding box:

            lc=(l1+l0)/2.
            mc=(m1+m0)/2.
            dl=l1-l0
            dm=m1-m0
            diam=np.max([dl,dm]) #Create a square grid
            #note: if the shape is aligned to an axis other than RA,DEC then this simple strategy does not minimize the facet grid area
            #A more complicated solution may include fitting an axis-aligned box and passing the facet rotation around the n-axis to the gridder.

            #(Mostly for debugging rotate the sky to a new phase centre (now the phase centre will differ from the projection pole of the image):
            lc += l_phasor_rad
            mc += m_phasor_rad

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
            x0=round(l_m_Diam[iFacet,0]/self.CellSizeRad)
            y0=round(l_m_Diam[iFacet,1]/self.CellSizeRad)
            if x0%2==0: x0+=1
            if y0%2==0: y0+=1
            l0=x0*self.CellSizeRad
            m0=y0*self.CellSizeRad
            diam=round(l_m_Diam[iFacet,2]/self.CellSizeRad)*self.CellSizeRad
            #self.AppendFacet(iFacet,l0,m0,diam)
            self.AppendFacet(iFacet,l0,m0,diam)





        #self.MakeMasksTessel()

        NpixMax=np.max([self.DicoImager[iFacet]["NpixFacet"] for iFacet in sorted(self.DicoImager.keys())])
        NpixMaxPadded=np.max([self.DicoImager[iFacet]["NpixFacetPadded"] for iFacet in sorted(self.DicoImager.keys())])
        self.PaddedGridShape=(1,1,NpixMaxPadded,NpixMaxPadded)
        self.FacetShape=(1,1,NpixMax,NpixMax)

        dmin=1
        for iFacet in range(len(self.DicoImager)):
            l,m=self.DicoImager[iFacet]["l0m0"]
            d=np.sqrt(l**2+m**2)
            if d<dmin:
                dmin=d
                iCentralFacet=iFacet
        self.iCentralFacet=iCentralFacet
        
        #regFile="%s.tessel.reg"%self.GD["Images"]["ImageName"]
        #labels=["[F%i]"%(i,self.DicoImager[i]["Polygon"]["iSol"]) for i in range(len(LPolygonNew))]
        #VM.PolygonToReg(regFile,LPolygonNew,radius=0.1,Col="green",labels=labels)

        self.WriteCoordFacetFile()

        DicoName="%s.DicoFacet"%self.GD["Images"]["ImageName"]
        print>>log, "Saving DicoImager in %s"%DicoName
        MyPickle.Save(self.DicoImager,DicoName)

    def WriteCoordFacetFile(self):
        FacetCoordFile="%s.facetCoord.txt"%self.GD["Images"]["ImageName"]
        print>>log, "Writing facet coordinates in %s"%FacetCoordFile
        f = open(FacetCoordFile, 'w')
        ss="# (Name, Type, Ra, Dec, I, Q, U, V, ReferenceFrequency='7.38000e+07', SpectralIndex='[]', MajorAxis, MinorAxis, Orientation) = format"
        for iFacet in range(len(self.DicoImager)):
            ra,dec=self.DicoImager[iFacet]["RaDec"]
            sra =rad2hmsdms.rad2hmsdms(ra,Type="ra").replace(" ",":")
            sdec=rad2hmsdms.rad2hmsdms(dec).replace(" ",".")
            ss="%s, %s"%(sra,sdec)
            f.write(ss+'\n')
        f.close()

#===============================================
#===============================================
#===============================================
#===============================================


class WorkerImager(ClassFacetMachine.WorkerImager):
    def init(self, iFacet):
        #Create smoothned facet tessel mask:
        Npix = self.DicoImager[iFacet]["NpixFacetPadded"]
        l0, l1, m0, m1 = self.DicoImager[iFacet]["lmExtentPadded"]
        X, Y = np.mgrid[l0:l1:Npix * 1j, m0:m1:Npix * 1j]
        XY = np.dstack((X, Y))
        XY_flat = XY.reshape((-1, 2))
        vertices = self.DicoImager[iFacet]["Polygon"]
        mpath = Path(vertices)  # the vertices of the polygon
        mask_flat = mpath.contains_points(XY_flat)

        mask = mask_flat.reshape(X.shape)

        mpath = Path(self.CornersImageTot)
        mask_flat2 = mpath.contains_points(XY_flat)
        mask2 = mask_flat2.reshape(X.shape)
        mask[mask2 == 0] = 0

        GaussPars = (10, 10, 0)

        SpacialWeigth = np.float32(mask.reshape((1, 1, Npix, Npix)))
        SpacialWeigth = ModFFTW.ConvolveGaussian(SpacialWeigth, CellSizeRad=1, GaussPars=[GaussPars])
        SpacialWeigth = SpacialWeigth.reshape((Npix, Npix))
        SpacialWeigth /= np.max(SpacialWeigth)
        NameSpacialWeigth = "%sSpacialWeigth.Facet_%3.3i" % (self.IdSharedMem, iFacet)
        NpShared.ToShared(NameSpacialWeigth, SpacialWeigth)
        #Initialize a grid machine per facet:
        self.GiveGM(iFacet)
        self.result_queue.put({"Success": True, "iFacet": iFacet})






