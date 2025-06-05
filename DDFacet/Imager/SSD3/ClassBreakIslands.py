from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import time
import numpy as np
from DDFacet.Other import logger
from DDFacet.Other import MyPickle
from DDFacet.Other import ModColor
log=logger.getLogger("ClassIslandDistanceMachine")
from DDFacet.Other.progressbar import ProgressBar
from SkyModel.PSourceExtract import ClassIslands
from SkyModel.PSourceExtract import ClassIncreaseIsland
from DDFacet.Array import NpShared
from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import psutil
import DDFacet.Other.AsyncProcessPool
from DDFacet.Array import shared_dict


#self.PSFServer.DicoVariablePSF["DicoImager"][0]['Polygon']




class ClassBreakIslands():
    def __init__(self,GD,MaskArray,PSFServer,DicoDirty,IdSharedMem=""):
        self.GD=GD
        self._MaskArray=MaskArray
        self.PSFServer=PSFServer
        self.PSFCross=None
        self.DicoDirty=DicoDirty
        if self.GD is not None:
            self.NCPU=(self.GD["Parallel"]["NCPU"] or psutil.cpu_count())
        else:
            self.NCPU=psutil.cpu_count()
        self.IdSharedMem=IdSharedMem
        self.DicoAPP={}

    def startWorkers(self,Name=""):
        APP_Islands=DDFacet.Other.AsyncProcessPool.initNew(Name="APP_Islands%s"%Name,
                                                                ncpu=self.GD["Parallel"]["NCPU"],
                                                                affinity="disable",#self.GD["Parallel"]["Affinity"],
                                                                #parent_affinity=self.GD["Parallel"]["MainProcessAffinity"],
                                                                #verbose=self.GD["Debug"]["APPVerbose"],
                                                                #pause_on_start=self.GD["Debug"]["PauseWorkers"]
                                                                )
        APP_Islands.registerJobHandlers(self)
        APP_Islands.startWorkers()

        self.DicoAPP[Name]=APP_Islands
        
        
    def killWorkers(self,Name=""):
        APP=self.DicoAPP[Name]
        APP.terminate()
        APP.shutdown()
        del(self.DicoAPP[Name])

    def BreakLargeIslandsFacetsPolygon(self,ListIslands):
        if not self.GD["SSDClean"]["MaxIslandSize"]: return ListIslands
        
        LOut=[]
        
        self.DicoVariablePSF=self.PSFServer.DicoVariablePSF
        CellSizeRad_x,CellSizeRad_y=self.DicoVariablePSF["CellSizeRad"]
        _,_,nx,ny=self.DicoVariablePSF["OutImShape"]
        
        DicoImager=self.DicoVariablePSF["DicoImager"]
        def inPoly(Polygon,X,Y):
            XY=np.array([X,Y]).T
            XY_flat = XY.reshape((-1, 2))
            vertices = Polygon # DicoImager[iFacet]["Polygon"]
            mpath = Path(vertices)  # the vertices of the polygon
            mask_flat = mpath.contains_points(XY_flat)
            mask = mask_flat.reshape(X.shape)
            #print("ggg",iFacet,np.count_nonzero(mask))
            #if np.count_nonzero(mask)==0: stop
            return mask
        dlmax=CellSizeRad_x*self.GD["SSDClean"]["MaxIslandSize"]
        dmmax=CellSizeRad_y*self.GD["SSDClean"]["MaxIslandSize"]
        for iIsland,ThisIsland in enumerate(ListIslands):
            x,y=np.array(ThisIsland).T
            l=CellSizeRad_x*(x-nx//2)
            m=CellSizeRad_y*(y-ny//2)
            IslandID=np.zeros((x.size,),int)
            for iFacet in sorted(DicoImager.keys()):
                #print(iFacet,len(DicoImager.keys()))
                lp,mp=DicoImager[iFacet]["Polygon"].T
                lp0,lp1=lp.min(),lp.max()
                mp0,mp1=mp.min(),mp.max()
                dl=lp1-lp0
                dm=mp1-mp0
                nl=int(dl//dlmax+1)
                nm=int(dm//dmmax+1)
                lg,mg=np.mgrid[lp0:lp1:(nl+1)*1j,mp0:mp1:(nm+1)*1j]

                M=inPoly(DicoImager[iFacet]["Polygon"],l,m)
                ind=np.where(M)[0]
                xs,ys=x[ind],y[ind]
                ls,ms=l[ind],m[ind]
                for ii in range(nl):
                    for jj in range(nm):
                        #print(ii,jj)
                        l0,m0=lg[ii,jj],mg[ii,jj]
                        l1,m1=lg[ii+1,jj],mg[ii+1,jj]
                        l2,m2=lg[ii+1,jj+1],mg[ii+1,jj+1]
                        l3,m3=lg[ii,jj+1],mg[ii,jj+1]
                        Poly=np.array([[l0,m0],
                                       [l1,m1],
                                       [l2,m2],
                                       [l3,m3],
                                       ])
                        M=inPoly(Poly,ls,ms)
                        ind=np.where(M)[0]
                        if ind.size==0: continue
                        LOut.append(np.array([xs[ind],ys[ind]]).T)
                        
        return LOut



    
    def BreakLargeIslandsFacets(self,ListIslands):
        if not self.GD["SSDClean"]["MaxIslandSize"]: return ListIslands
        
        LOut=[]
        for iIsland,ThisIsland in enumerate(ListIslands):
            x,y=np.array(ThisIsland).T

            FacetsIDs=self.giveFacetIDs(x,y)

            for iFacet in np.unique(FacetsIDs):
                ind=np.where(FacetsIDs==iFacet)[0]
                xs,ys=x[ind],y[ind]
                ThisFacetIsland=np.array([xs,ys]).T
                LOut+=self.BreakLargeIslands([ThisFacetIsland])
                
        return LOut

    def giveFacetIDs(self,x,y):
        # self.PSFServer.DicoVariablePSF["DicoImager"][0].keys()
        self.DicoVariablePSF=self.PSFServer.DicoVariablePSF

        CellSizeRad_x,CellSizeRad_y=self.DicoVariablePSF["CellSizeRad"]
        _,_,nx,ny=self.DicoVariablePSF["OutImShape"]


        DicoImager=self.DicoVariablePSF["DicoImager"]
        def inPoly(iFacet,X,Y):
            XY=np.array([X,Y]).T
            XY_flat = XY.reshape((-1, 2))
            vertices = DicoImager[iFacet]["Polygon"]
            mpath = Path(vertices)  # the vertices of the polygon
            mask_flat = mpath.contains_points(XY_flat)
            mask = mask_flat.reshape(X.shape)
            #print("ggg",iFacet,np.count_nonzero(mask))
            #if np.count_nonzero(mask)==0: stop
            return mask
        
        ClosestFacet=-1

        LlmSol=[]
        
        l=CellSizeRad_x*(x-nx//2)
        m=CellSizeRad_y*(y-ny//2)

        FacetID=np.zeros((x.size,),int)
        self.NFacets=len(DicoImager)
        for iFacet in range(self.NFacets):
            print(iFacet,self.NFacets)
            M=inPoly(iFacet,l,m)
            ind=np.where(M)[0]
            #print("- ",iFacet,ind.size)
            
            #xs=x[ind]
            #ys=y[ind]
            FacetID[ind]=iFacet

        return FacetID#[x,y]
            

    
    def BreakLargeIslands(self,ListIslands):
        # self.PSFServer.DicoVariablePSF["DicoImager"][0].keys()
        if not self.GD["SSDClean"]["MaxIslandSize"]: return ListIslands
        
        LOut=[]
        NN=int(self.GD["SSDClean"]["MaxIslandSize"])
        print("  breaking islands with linear size larger than %i pixels into smaller ones"%self.GD["SSDClean"]["MaxIslandSize"], file=log)
        for iIsland,ThisIsland in enumerate(ListIslands):
            x,y=np.array(ThisIsland).T
            x0,x1=x.min(),x.max()
            y0,y1=y.min(),y.max()
            Lx,Ly=x1-x0+1,y1-y0+1
            nx=Lx//self.GD["SSDClean"]["MaxIslandSize"]+1
            ny=Ly//self.GD["SSDClean"]["MaxIslandSize"]+1
            nn=nx*ny

            if nn==1:
                LOut.append(ThisIsland)
            else:
                print("    breaking islands #%i into %i islands"%(iIsland,nn), file=log)
                xb=np.int64(np.mgrid[x0:x1:(nx+1)*1j])
                yb=np.int64(np.mgrid[y0:y1:(ny+1)*1j])
                self.xIsland=x
                self.yIsland=y
                # xb=xb.flatten()
                # yb=yb.flatten()
                key="_%i"%iIsland
                self.startWorkers(key)
                APP=self.DicoAPP[key]
                for ix in range(nx):
                    for iy in range(ny):
                        APP.runJob("selSubIsland.%i.%i.%i"%(iIsland,ix,iy),
                                   self._selSubIsland,
                                   args=(xb[ix],xb[ix+1],yb[iy],yb[iy+1]))#, serial=True)

                ThisIslandLOut=APP.awaitJobResults("selSubIsland.*", progress="Split island #%i"%iIsland)
                ThisIslandLOut=[l for l in ThisIslandLOut if l is not None]
                self.killWorkers(key)
                LOut+=ThisIslandLOut
        ListIslands=LOut
        
        return ListIslands


    def _selSubIsland(self,x0,x1,y0,y1):
        xs,ys=self.xIsland,self.yIsland
        ind=np.where(xs>x0)[0]
        if ind.size==0: return None
        xs=xs[ind]; ys=ys[ind]
        
        ind=np.where(xs<=x1)[0]
        if ind.size==0: return None
        xs=xs[ind]; ys=ys[ind]
        
        ind=np.where(ys>y0)[0]
        if ind.size==0: return None
        xs=xs[ind]; ys=ys[ind]
        
        ind=np.where(ys<=y1)[0]
        if ind.size==0: return None
        xs=xs[ind]; ys=ys[ind]
        
        II=np.zeros((ind.size,2),xs.dtype)
        II[:,0]=xs[:]
        II[:,1]=ys[:]

        return II
    
