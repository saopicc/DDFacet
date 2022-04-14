import numpy as np
import DDFacet.Imager.SSD.ClassIslandDistanceMachine
from DDFacet.Other import logger
log=logger.getLogger("ClassBrightFaint")
from astropy.io import fits
from pyrap.images import image

def PutDataInNewImage(oldfits,newfits,data):
    outim=newfits+'.fits'
    log.print("writting image %s"%outim)
    hdu=fits.open(oldfits)
    hdu[0].data=data
    hdu.writeto(outim,overwrite=True)
    return outim

class ClassBrightFaintOverAll():
    def __init__(self,options=None,ImMask=None,Restored=None,FitsFile=None,incr_rad=None):
        self.options=options
        self.ImMask=ImMask
        self.Restored=Restored
        self.FitsFile=FitsFile
        self.incr_rad=incr_rad
        self.CasaIm=image(self.FitsFile)
        
    def giveBrightFaintMask(self):
        print("Build bright/faint mask...", file=log)
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
                                                                                                       IdSharedMem=IdSharedMem)
        ListIslands=IslandDistanceMachine.SearchIslands(None,Image=self.Restored)
        
        IslandDistanceMachine.calcDistanceMatrixMinParallel(ListIslands)
        dx,dy=IslandDistanceMachine.dx,IslandDistanceMachine.dy
        IslandDistanceMachine.DistCross=np.sqrt(dx**2+dy**2)
        ListIslands=IslandDistanceMachine.CalcCrossIslandFlux_noPSFInfo(ListIslands,self.Restored)

        
        ListIslands=IslandDistanceMachine.ConvexifyIsland(ListIslands)#,PolygonFile="%s.pickle"%OutMaskExtended)
        ListIslands=IslandDistanceMachine.MergeIslands(ListIslands)
        Mask=np.zeros((nx,nx),np.float32)

        S=np.zeros((len(ListIslands),),np.float32)
        
        for iIsland,Island in enumerate(ListIslands):
            x,y=np.array(Island).T
            Mask[x,y]=1
            S[iIsland]=np.max(self.Restored[0,0,x,y])
            
        OutTest="%s.convex_mask"%self.FitsFile
        ImWrite=Mask.reshape((1,1,nx,nx))
        PutDataInNewImage(self.FitsFile,OutTest,np.float32(ImWrite))


        indBrightest=np.argsort(S)[::-1]
        NDir=100
        MaskBright=np.zeros((nx,nx),np.float32)
        MaskFaint=np.zeros((nx,nx),np.float32)
        
        ClusterCat=np.zeros((NDir,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('SumI',np.float),("Cluster",int)])
        ClusterCat=ClusterCat.view(np.recarray)
        
        
        
        for iCluster,iIsland in enumerate(indBrightest):
            Island=ListIslands[iIsland]
            x,y=np.array(Island).T
            if iCluster<NDir:
                s=self.Restored[0,0,x,y]
                xcc=np.sum(s*x)/np.sum(s)
                ycc=np.sum(s*y)/np.sum(s)
                ff,pol,dec,ra=self.CasaIm.toworld((0,0,xcc,ycc))
                ClusterCat.ra[iCluster]=ra
                ClusterCat.dec[iCluster]=dec
                ClusterCat.SumI[iCluster]=np.sum(s)
                ClusterCat.Cluster=iCluster
                MaskBright[x,y]=1
            else:
                MaskFaint[x,y]=1
        
        OutTest="%s.bright_mask"%self.FitsFile
        ImWrite=MaskBright.reshape((1,1,nx,nx))
        FitsBright=PutDataInNewImage(self.FitsFile,"%s"%OutTest,np.float32(ImWrite))
 
        OutTest="%s.faint_mask"%self.FitsFile
        ImWrite=MaskFaint.reshape((1,1,nx,nx))
        FitsFaint=PutDataInNewImage(self.FitsFile,"%s"%OutTest,np.float32(ImWrite))

        OutCat="%s.BrightFaint_ClusterCat.npy"%self.FitsFile
        log.print("writting catalog %s"%OutCat)
        np.save(OutCat,ClusterCat)

        return FitsBright
