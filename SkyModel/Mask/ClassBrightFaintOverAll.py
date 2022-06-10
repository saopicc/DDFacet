import numpy as np
import DDFacet.Imager.SSD.ClassIslandDistanceMachine
from DDFacet.Other import logger
log=logger.getLogger("ClassBrightFaint")
from astropy.io import fits
from pyrap.images import image
from DDFacet.ToolsDir import ModFFTW
import SkyModel.Sky.ModRegFile
import SkyModel.Sky.ClassClusterDEAP

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

        ModelImage=image("image_full_ampphase_di_m.NS.app.model.fits").getdata()
        
        for iIsland,Island in enumerate(ListIslands):
            x,y=np.array(Island).T
            if x.size<5: continue
                
            x0,x1=x.min()-10,x.max()+10
            y0,y1=y.min()-10,y.max()+10
            x0=np.max([0,x0])
            y0=np.max([0,y0])
            x1=np.min([x1,ModelImage.shape[-2]])
            y1=np.min([y1,ModelImage.shape[-1]])
            if (x0-x1)%2==0: x1+=1
            if (y0-y1)%2==0: y1+=1
                
            s=ModelImage[0:1,0:1,x0:x1,y0:y1]
            m=np.zeros_like(s)
            m[0,0,x-x0,y-y0]=1
            s=np.complex64(s*m)
                
            sc=s.copy().real
            FM=ModFFTW.FFTW_2Donly_np(s,s.dtype, ncores = 1)
            fs=FM.fft(s)[0,0]
                
            Freq0 = np.fft.fftshift(np.fft.fftfreq(s.shape[-2],d=1))
            dFreq0=1./s.shape[-2]
            Freq1 = np.fft.fftshift(np.fft.fftfreq(s.shape[-1],d=1))
            dFreq1=1./s.shape[-1]
            f0x,f1x=Freq0.min()-dFreq0/2,Freq0.max()+dFreq0/2
            f0y,f1y=Freq1.min()-dFreq1/2,Freq1.max()+dFreq1/2
            # import pylab
            # pylab.clf()
            # pylab.subplot(2,2,1)
            # pylab.imshow(sc[0,0],interpolation="nearest",aspect="auto")
            # pylab.subplot(2,2,2)
            # pylab.imshow(self.Restored[0,0,x0:x1,y0:y1],interpolation="nearest",aspect="auto")
                
            # pylab.subplot(2,2,3)
            # pylab.imshow(np.abs(fs),interpolation="nearest",aspect="auto",extent=(f0x,f1x,f0y,f1y))
            # pylab.subplot(2,2,4)
            # xc,yc=fs.shape[-2:]
            # pylab.scatter(Freq1,np.abs(fs)[xc//2,:])
            # pylab.scatter(Freq0,np.abs(fs)[:,yc//2])
            
            # pylab.ylim(0,fs.max())
            # pylab.draw()
            # pylab.show(block=False)
            # pylab.pause(1)
                
            fx,fy=np.mgrid[Freq0.min():Freq0.max():1j*s.shape[-2],Freq1.min():Freq1.max():1j*s.shape[-1]]
            indx,indy=np.where(np.sqrt(fx**2+fy**2)>0.25)
            fsa=np.abs(fs)
            S[iIsland]=np.sum(fsa[indx,indy])

        

        # #########
        
        # for iIsland,Island in enumerate(ListIslands):
        #     x,y=np.array(Island).T
        #     Mask[x,y]=1
        #     S[iIsland]=np.max(self.Restored[0,0,x,y])
            

        # ######
            
            
        OutTest="%s.convex_mask"%self.FitsFile
        ImWrite=Mask.reshape((1,1,nx,nx))
        PutDataInNewImage(self.FitsFile,OutTest,np.float32(ImWrite))


        indBrightest=np.argsort(S)[::-1]
        NDir=45
        MaskBright=np.zeros((nx,nx),np.float32)
        MaskFaint=np.zeros((nx,nx),np.float32)
        
        ClusterCat=np.zeros((NDir,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('SumI',np.float),("Cluster",int)])
        ClusterCat=ClusterCat.view(np.recarray)

        # # xc=np.zeros((len(ListIslands),),np.float64)
        # # yc=np.zeros((len(ListIslands),),np.float64)
        # # sc=np.zeros((len(ListIslands),),np.float64)
        # # for iIsland,Island in enumerate(ListIslands):
        # #     x,y=np.array(Island).T
        # #     s=self.Restored[0,0,x,y]
        # #     xc[iIsland]=np.sum(s*x)/np.sum(s)
        # #     yc[iIsland]=np.sum(s*y)/np.sum(s)
        # #     sc[iIsland]=np.sum(s)
            
        # CC=SkyModel.Sky.ClassClusterDEAP.ClassCluster(xc,yc,sc,
        #                                               nNode=45,
        #                                               NGen=5,
        #                                               NPop=5,
        #                                               DoPlot=1,
        #                                               PolyCut=None,#self.PolyCut,
        #                                               NCPU=96,
        #                                               FitnessType="BrightCal")
        
        # # CC.setAvoidPolygon(PolyList)
            
        # xyNodes,self.LPolygon=CC.Cluster()
        
        # nNodes=xyNodes.size//2
        # xc,yc=xyNodes.reshape((2,nNodes))
        # self.xcyc=xc,yc
        
        # stop
        
        
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

        
        SkyModel.Sky.ModRegFile.radecRad2Reg("%s.bright_cal.reg"%self.FitsFile,ClusterCat.ra,ClusterCat.dec)

        
        return FitsBright
