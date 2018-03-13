import numpy as np
import Polygon
from scipy.spatial import Voronoi
from deap import algorithms
import DeapAlgo as algorithms
from deap import base
from deap import creator
from deap import tools
import pylab
from scipy.spatial import Voronoi
import ModVoronoi
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClusterDEAP")
from DDFacet.Other import ClassTimeIt
#from scoop import futures
import multiprocessing
import scipy.stats
import Polygon

Theta=np.linspace(-np.pi,np.pi/2,21)

def hasStraightEdge(P):
    p=np.array(P)[0]
    dx=((p[1:,0]-p[:-1,0])==0)
    dy=((p[1:,1]-p[:-1,1])==0)
    if np.count_nonzero(dx) or np.count_nonzero(dy):
        return True
    return False

def giveSizeRatio(P):
    if P.size==0: return False
    P0=Polygon.Polygon(P.copy())
    if P0.area()==0: return False
    #if hasStraightEdge(P0): return False
    a=np.zeros_like(Theta)
    for iTh,Th in enumerate(Theta):
        P0.rotate(Th)
        a[iTh]=P0.aspectRatio()
    return a.max()/a.min()

def IndivToPolygon(indiv,PolyCut):
    N=indiv.size/2
    xc,yc=indiv.reshape((2,N))
    
    xc,yc=indiv.reshape((2,N))
    xy=np.zeros((xc.size,2),np.float32)
    xy[:,0]=xc
    xy[:,1]=yc
    vor = Voronoi(xy)#incremental=True)
    LPolygon=[]
    regions, vertices = ModVoronoi.voronoi_finite_polygons_2d(vor)
    for iR,region in enumerate(regions):
        polygon = vertices[region]
        PP=(Polygon.Polygon(polygon) & Polygon.Polygon(PolyCut))
        if PP.area()>0:
            LPolygon.append(np.array(PP[0]))
        else:
            LPolygon.append(np.array(PP))
            
    return LPolygon
        
def doOverlap(npP0,npP1):
    T=ClassTimeIt.ClassTimeIt("Overlap")
    T.disable()
    if npP0.size==0: return False
    if npP1.size==0: return False
    P0 = Polygon.Polygon(npP0)
    P1 = Polygon.Polygon(npP1)
    T.timeit("declare")
    P1Cut = (P0 & P1)
    T.timeit("Cut")
    aP1=P1.area()
    aP1Cut=P1Cut.area()
    T.timeit("Area")
    if np.abs(aP1Cut-aP1)<1e-10:
        return "Contains"
    elif aP1Cut==0:
        return "Outside"
    else:
        return "Cut"

def giveMeanDistanceToNode(xc,yc,x,y,S,Poly):
    d0=np.sqrt(Polygon.Polygon(Poly).area())
    w=S/np.sum(S)
    dmean=np.sum(w*np.sqrt((xc-x)**2+(yc-y)**2)/d0)
    return dmean

class ClassMetricDEAP():
    def __init__(self,
                 Indiv,x=None,y=None,S=None,
                 Polygons=None,PolyCut=None,BigPolygon=None):
        
        nNode=Indiv.size/2
        xc,yc=Indiv.reshape((2,nNode))
        self.xc=xc
        self.yc=yc
        self.S=S
        self.x=x
        self.y=y
        
        dx=xc.reshape((-1,1))-x.reshape((1,-1))
        dy=yc.reshape((-1,1))-y.reshape((1,-1))
        self.d_NodeSource=d=np.sqrt(dx**2+dy**2)
        self.indSourceToNode=np.argmin(d,axis=0)
        self.setNodes=np.unique(self.indSourceToNode)
        self.ListPolygons=IndivToPolygon(Indiv,PolyCut)
        self.Polygons=Polygons
        self.BigPolygon=BigPolygon

    def fluxPerFacet(self):
        xc=self.xc
        S=self.S
        SPerNode=np.zeros((xc.size,),np.float32)
        SMeanPerFacet=np.sum(S)/xc.size
        for iC in self.setNodes:
            #NPerNode[iC]=np.count_nonzero(ind==iC)
            #if ind.size==0: continue
            setSourcesThisNode=(self.indSourceToNode==iC)
            SPerNode[iC]=np.sum(S[setSourcesThisNode])/SMeanPerFacet

        
        return SPerNode


    def bigFlux(self,SPerNode):
        for Poly in self.BigPolygon:
            x,y=Poly.T
            x0,y0=np.mean(x),np.mean(y)
            d=np.sqrt((x0-self.xc)**2+(y0-self.yc)**2)
            iNodeBig=np.argmin(d)
            
            

    
    def NPerFacet(self):
        xc=self.xc
        S=self.S
        NPerNode=np.zeros((xc.size,),np.float32)
        NMeanPerFacet=self.x.size/float(xc.size)
        for iC in self.setNodes:
            #NPerNode[iC]=np.count_nonzero(ind==iC)
            #if ind.size==0: continue
            setSourcesThisNode=(self.indSourceToNode==iC)
            NPerNode[iC]=np.count_nonzero(setSourcesThisNode)/NMeanPerFacet
        return NPerNode
    
    def aspectRatioPerFacet(self):
        AspectRatio=np.zeros((self.xc.size,),np.float32)
        for iC,Poly in enumerate(self.ListPolygons):
            a=giveSizeRatio(Poly)
            if a: AspectRatio[iC]=a

        return AspectRatio

    def meanDistancePerFacet(self):
        xc=self.xc
        MeanDistanceToNode=np.zeros((xc.size,),np.float32)
        for iC in self.setNodes:
            Poly=self.ListPolygons[iC]
            setSourcesThisNode=(self.indSourceToNode==iC)
            x=self.x[setSourcesThisNode]
            y=self.y[setSourcesThisNode]
            S=self.S[setSourcesThisNode]
            MeanDistanceToNode[iC]=giveMeanDistanceToNode(self.xc[iC],self.yc[iC],x,y,S,Poly)
        return MeanDistanceToNode
    
    def overlapPerFacet(self):
        if self.Polygons is None:
            return False
        Overlap=np.zeros((self.xc.size,),np.float32)
        for iC,PolyNode in enumerate(self.ListPolygons):
            for P in self.Polygons:
                if doOverlap(PolyNode,P)=="Cut":
                    Overlap[iC]+=1

        return Overlap
