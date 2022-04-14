import numpy as np
import DDFacet.Imager.SSD.ClassIslandDistanceMachine
from DDFacet.Other import logger
log=logger.getLogger("ClassBrightFaint")
from astropy.io import fits
import DDFacet.Other.MyPickle
from matplotlib.path import Path

def PutDataInNewImage(oldfits,newfits,data):
    outim=newfits+'.fits'
    log.print("writting image %s"%outim)
    hdu=fits.open(oldfits)
    hdu[0].data=data
    hdu.writeto(outim,overwrite=True)
    return outim

class ClassBrightFaintPerFacet():
    def __init__(self,options=None,ImMask=None,Restored=None,FitsFile=None,incr_rad=None):
        self.options=options
        self.ImMask=ImMask
        self.Restored=Restored
        self.FitsFile=FitsFile
        self.incr_rad=incr_rad
        
    def giveBrightFaintMask(self):
        print("Build facetted bright/faint mask...", file=log)
        GD=None
        Mask=self.ImMask
        nx=Mask.shape[-1]
        CurrentNegMask=np.logical_not(Mask).reshape((1,1,nx,nx))
        PSFServer=None
        IdSharedMem=None
        DicoDirty=None
        
        IslandDistanceMachine=DDFacet.Imager.SSD.ClassIslandDistanceMachine.ClassIslandDistanceMachine(GD,
                                                                                                       CurrentNegMask,
                                                                                                       PSFServer,
                                                                                                       DicoDirty,
                                                                                                       IdSharedMem=IdSharedMem,
                                                                                                       MinMaxGroupDistance=[0,50])
        ListIslands=IslandDistanceMachine.SearchIslands(None,Image=self.Restored)
        
        IslandDistanceMachine.calcDistanceMatrixMinParallel(ListIslands)
        dx,dy=IslandDistanceMachine.dx,IslandDistanceMachine.dy
        IslandDistanceMachine.DistCross=np.sqrt(dx**2+dy**2)
        ListIslands=IslandDistanceMachine.CalcCrossIslandFlux_noPSFInfo(ListIslands,self.Restored)
        
        ListIslands=IslandDistanceMachine.ConvexifyIsland(ListIslands)#,PolygonFile="%s.pickle"%OutMaskExtended)
        Mask=np.zeros((nx,nx),np.float32)
        for Island in ListIslands:
            x,y=np.array(Island).T
            Mask[x,y]=1

 
        OutTest="%s.convex_mask"%self.FitsFile
        ImWrite=Mask.reshape((1,1,nx,nx))
        PutDataInNewImage(self.FitsFile,OutTest,np.float32(ImWrite))
        

        ListPolygons=IslandDistanceMachine.ListPolygons
        
        BaseImageName=self.FitsFile.split(".app.")[0]
        if self.options.BaseImageName: BaseImageName=self.options.BaseImageName 
        D=DDFacet.Other.MyPickle.Load("%s.DicoFacet"%BaseImageName)

        #LSol=[D[iFacet]["iSol"][0] for iFacet in D.keys()]
        DicoDir={}
        for iFacet in list(D.keys()):
            iSol=D[iFacet]["iSol"][0]
            if not iSol in list(DicoDir.keys()):
                DicoDir[iSol]=[iFacet]
            else:
                DicoDir[iSol].append(iFacet)
            
        MaskBright=np.zeros((nx,nx),np.float32)
        MaskFaint=np.zeros((nx,nx),np.float32)
        for iSol in sorted(list(DicoDir.keys())):#[0:5]:
            print("===================== Processing direction %2.2i/%2.2i ====================="%(iSol,len(DicoDir)), file=log)
            ThisFacetMask=np.zeros_like(Mask)-1
            for iFacet in DicoDir[iSol]:
                PolyGon=D[iFacet]["Polygon"]
                l,m=PolyGon.T
                x,y=((l/self.incr_rad+nx//2)), ((m/self.incr_rad+nx//2))
                poly2=np.array([x,y]).T
                x0,x1=x.min(),x.max()
                y0,y1=y.min(),y.max()
                xx,yy=np.mgrid[x0:x1:(x1-x0+1)*1j,y0:y1:(y1-y0+1)*1j]
                xx=np.int16(xx)
                yy=np.int16(yy)
                
                pp=np.zeros((poly2.shape[0]+1,2),dtype=poly2.dtype)
                pp[0:-1,:]=poly2[:,:]
                pp[-1,:]=poly2[0,:]
                #ListPolygons.append(pp)
                mpath = Path(pp)
                
                p_grid=np.zeros((xx.size,2),np.int16)
                p_grid[:,0]=xx.ravel()
                p_grid[:,1]=yy.ravel()
                mask_flat = mpath.contains_points(p_grid)
                
                IslandOut=np.array([xx.ravel()[mask_flat],yy.ravel()[mask_flat]])
                x,y=IslandOut
                ThisFacetMask[x,y]=1
                #raFacet, decFacet = self.CoordMachine.lm2radec(np.array([lmShift[0]]),
                #                                               np.array([lmShift[1]]))
            ThisFacetMask=ThisFacetMask[::-1,:].T
            ThisFacetMask= (np.abs(Mask - ThisFacetMask)<1e-6)
            
            IslandDistanceMachine=DDFacet.Imager.SSD.ClassIslandDistanceMachine.ClassIslandDistanceMachine(GD,
                                                                                                           1-ThisFacetMask.reshape((1,1,nx,nx)),
                                                                                                           PSFServer,
                                                                                                           DicoDirty,
                                                                                                           IdSharedMem=IdSharedMem)
            ListIslands=IslandDistanceMachine.SearchIslands(None,Image=self.Restored)
            ListIslands=IslandDistanceMachine.ConvexifyIsland(ListIslands)
            DFlux=np.zeros((len(ListIslands),),np.float32)
            for iIsland,Island in enumerate(ListIslands):
                x,y=np.array(Island).T
                s=np.abs(self.Restored[0,0,x,y])
                xc,yc=np.sum(s*x)/np.sum(s),np.sum(s*y)/np.sum(s)
                d=np.sqrt((x-xc)**2+(y-yc)**2)
                Size=np.sum(s*d)/np.sum(s)
                Size=np.max([10,Size])
                #DFlux[iIsland]=(np.max(self.Restored[0,0,x,y])+np.sum(self.Restored[0,0,x,y]))/Size**2
                #DFlux[iIsland]=np.max(self.Restored[0,0,x,y])/Size#**2
                DFlux[iIsland]=np.sum(self.Restored[0,0,x,y])#/Size**2
                #print(iIsland,DFlux[iIsland],x,y,Size)
            iIsland_bright=np.argmax(DFlux)
            #print("!!!!!!!!!!!",iIsland_bright)
            
            for iIsland,Island in enumerate(ListIslands):
                x,y=np.array(Island).T
                if iIsland==iIsland_bright:
                    #print("fffffffffffff %i"%iIsland)
                    MaskBright[x,y]=1
                    #print(MaskBright[x,y],x,y)
                else:
                    MaskFaint[x,y]=1

        OutTest="%s.bright_mask"%self.FitsFile
        ImWrite=MaskBright.reshape((1,1,nx,nx))
        ImBright=PutDataInNewImage(self.FitsFile,"%s"%OutTest,np.float32(ImWrite))
 
        OutTest="%s.faint_mask"%self.FitsFile
        ImWrite=MaskFaint.reshape((1,1,nx,nx))
        PutDataInNewImage(self.FitsFile,"%s"%OutTest,np.float32(ImWrite))
        return ImBright
