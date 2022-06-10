import numpy as np
import DDFacet.Imager.SSD.ClassIslandDistanceMachine
from DDFacet.Other import logger
log=logger.getLogger("ClassBrightFaint")
from astropy.io import fits
import DDFacet.Other.MyPickle
from matplotlib.path import Path
from DDFacet.ToolsDir import ModFFTW
from pyrap.images import image
import SkyModel.Sky.ModRegFile
import scipy.signal

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
        self.CasaIm=image(self.FitsFile)
        
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
        print("!!!!!!!!!!!!!!!!!!!!!")
        PolyFacetFile="image_full_ampphase_di_m.NS_predictFaint.DicoFacet"#"%s.DicoFacet"%BaseImageName
        log.print("Loading %s"%PolyFacetFile)
        D=DDFacet.Other.MyPickle.Load(PolyFacetFile)

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
        
        NRand=1000
        np.random.seed(1)
        def give_in_points(x,y):
            x0,x1=x.min(),x.max()
            y0,y1=y.min(),y.max()
            xx=np.random.rand(NRand)*(x1-x0)+x0
            yy=np.random.rand(NRand)*(y1-y0)+y0
            poly2=np.array([x,y]).T
            pp=np.zeros((poly2.shape[0]+1,2),dtype=poly2.dtype)
            pp[0:-1,:]=poly2[:,:]
            pp[-1,:]=poly2[0,:]
            mpath = Path(pp)
            p_grid=np.zeros((xx.size,2),np.float32)
            p_grid[:,0]=xx.ravel()
            p_grid[:,1]=yy.ravel()
            mask_flat = mpath.contains_points(p_grid)
            x0=xx[mask_flat]
            y0=yy[mask_flat]
            return x0,y0

        DicoDiam={}
        log.print("Computing max distance within tessel...")
        for iSol in sorted(list(DicoDir.keys())):#[27:28]:
            Ll=[]
            Lm=[]
            for iFacet in DicoDir[iSol]:
                PolyGon=D[iFacet]["Polygon"]
                l,m=PolyGon.T
                x0,y0=give_in_points(l,m)
                Ll=np.concatenate([Ll,x0])
                Lm=np.concatenate([Lm,y0])
            # import pylab
            # pylab.clf()
            # pylab.scatter(Ll,Lm)
            # pylab.draw()
            # pylab.show(block=False)
            # pylab.pause(0.1)
            # # stop
            x0,y0=Ll,Lm
            dx=x0.reshape((-1,1))-x0.reshape((1,-1))
            dy=y0.reshape((-1,1))-y0.reshape((1,-1))
            dd=np.sqrt(dx**2+dy**2)
            ThisDMaxDeg=dd.max()*180/np.pi
            DicoDiam[iSol]=ThisDMaxDeg

        DiamMax_deg=2.
            
        L_raCal=[]
        L_decCal=[]
        for iSol in sorted(list(DicoDir.keys())):#[27:28]:
            print("===================== Processing direction %2.2i/%2.2i ====================="%(iSol,len(DicoDir)), file=log)
            ThisFacetMask=np.zeros_like(Mask)-1
            PutAllIslands=False
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

            ThisFacetMask=ThisFacetMask[::-1,:].T
            ThisFacetMaskFacet=ThisFacetMask.copy()
            ThisFacetMask= (np.abs(Mask - ThisFacetMask)<1e-6)


                
                #raFacet, decFacet = self.CoordMachine.lm2radec(np.array([lmShift[0]]),
                #                                               np.array([lmShift[1]]))
            
            IslandDistanceMachine=DDFacet.Imager.SSD.ClassIslandDistanceMachine.ClassIslandDistanceMachine(GD,
                                                                                                           1-ThisFacetMask.reshape((1,1,nx,nx)),
                                                                                                           PSFServer,
                                                                                                           DicoDirty,
                                                                                                           IdSharedMem=IdSharedMem)
            ListIslands=IslandDistanceMachine.SearchIslands(None,Image=self.Restored)
            ListIslands=IslandDistanceMachine.ConvexifyIsland(ListIslands)


            if DicoDiam[iSol]>DiamMax_deg:
                log.print("Tessel is too big [%.2f > %.2f deg] - including all islands as calibrators"%(DicoDiam[iSol],DiamMax_deg))
                PutAllIslands=True
                for iIsland,Island in enumerate(ListIslands):
                    x,y=np.array(Island).T
                    MaskBright[x,y]=1
                continue



            xx,yy=np.where(ThisFacetMaskFacet==1)
            x0,x1=xx.min(),xx.max()
            y0,y1=yy.min(),yy.max()
            xCenter=(x0+x1)/2
            yCenter=(y0+y1)/2

            nSup=500
            SigFact=10
            Sig_x=(x1-x0)/SigFact
            Sig_y=(y1-y0)/SigFact
            nSup_x=Sig_x*3
            nSup_y=Sig_y*3
            xGauss,yGauss=np.mgrid[-nSup_x:nSup_x+1,-nSup_y:nSup_y+1]
            Gauss=np.exp(-(xGauss**2)/(2*Sig_x**2)-(yGauss**2)/(2*Sig_y**2))
            Gauss/=np.sum(Gauss)
            ThisFacetMaskFacet_sel=ThisFacetMaskFacet[x0:x1,y0:y1].copy()
            ThisFacetMaskFacet_sel[ThisFacetMaskFacet_sel==-1]=0
            ThisTaperConv_sel=scipy.signal.fftconvolve(np.float32(ThisFacetMaskFacet_sel), Gauss, mode='same')
            #r0=ThisTaperConv_sel[ThisFacetMaskFacet_sel==0].max()
            ThisTaperConv_sel-=0.5
            ThisTaperConv_sel[ThisTaperConv_sel<0]=0
            ThisTaperConv_sel/=ThisTaperConv_sel.max()
            ThisTaper=np.zeros_like(ThisFacetMaskFacet)
            ThisTaper[x0:x1,y0:y1]=ThisTaperConv_sel[:,:]
            
            # Sig=.5
            # ThisTaperFlat=np.exp(-(xx-xCenter)**2/(2*Sig**2*(x1-x0)**2) + -(yy-yCenter)**2/(2*Sig**2*(y1-y0)**2))
            #
            # ThisTaper=np.zeros_like(ThisFacetMaskFacet)
            # nxTaper,nyTaper=ThisTaper.shape
            # ThisTaper.flat[xx*nyTaper+yy]=ThisTaperFlat.flat[:]

            import pylab
            pylab.clf()
            pylab.subplot(2,1,1)
            pylab.imshow(ThisFacetMaskFacet_sel)
            pylab.subplot(2,1,2)
            pylab.imshow(ThisTaper[x0:x1,y0:y1])
            pylab.draw()
            pylab.show(block=False)
            pylab.pause(0.1)


            
            DFlux=np.zeros((len(ListIslands),),np.float32)
            
            ModelImage=image("image_full_ampphase_di_m.NS.app.model.fits").getdata()
            
            ######################################
            # Take brightest pixel on island
            NMin=101
            for iIsland,Island in enumerate(ListIslands):
                x,y=np.array(Island).T
                xIsland,yIsland=np.mean(x),np.mean(y)
                if x.size<5: continue
                
                x0,x1=x.min()-10,x.max()+10
                y0,y1=y.min()-10,y.max()+10
                #xc,yc=int((x0+x1)/2),int((y0+y1)/2)
                #N=np.max([x0-x1+1,x0-y1+1])
                #N=np.max([N,NMin])
                #x0,x1=xc-N//2-2,xc+N//2+2
                #y0,y1=yc-N//2-2,yc+N//2+2
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
                #print(s.shape[-2:],dFreq0,dFreq1)
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
                w=ThisTaper[int(xIsland),int(yIsland)]
                DFlux[iIsland]=np.sum(fsa[indx,indy])*w
                
            iIsland_bright=np.argmax(DFlux)
            for iIsland,Island in enumerate(ListIslands):
                x,y=np.array(Island).T
                if iIsland==iIsland_bright:
                    #print("fffffffffffff %i"%iIsland)
                    MaskBright[x,y]=1
                    if not PutAllIslands:
                        s=self.Restored[0,0,x,y]
                        xcc=np.sum(s*x)/np.sum(s)
                        ycc=np.sum(s*y)/np.sum(s)
                        ff,pol,dec,ra=self.CasaIm.toworld((0,0,xcc,ycc))
                        L_raCal.append(ra)
                        L_decCal.append(dec)
                    #print(MaskBright[x,y],x,y)
                else:
                    MaskFaint[x,y]=1
            ######################################


            
            # ######################################
            # # Take brightest pixel on island
            # for iIsland,Island in enumerate(ListIslands):
            #     x,y=np.array(Island).T
            #     s=np.abs(self.Restored[0,0,x,y])
            #     xc,yc=np.sum(s*x)/np.sum(s),np.sum(s*y)/np.sum(s)
            #     d=np.sqrt((x-xc)**2+(y-yc)**2)
            #     Size=np.sum(s*d)/np.sum(s)
            #     Size=np.max([10,Size])
            #     #DFlux[iIsland]=(np.max(self.Restored[0,0,x,y])+np.sum(self.Restored[0,0,x,y]))/Size**2
            #     DFlux[iIsland]=np.max(self.Restored[0,0,x,y]) # /Size#**2
            #     #DFlux[iIsland]=np.sum(self.Restored[0,0,x,y])#/Size**2
            #     #print(iIsland,DFlux[iIsland],x,y,Size)
            # iIsland_bright=np.argmax(DFlux)
            # for iIsland,Island in enumerate(ListIslands):
            #     x,y=np.array(Island).T
            #     if iIsland==iIsland_bright:
            #         #print("fffffffffffff %i"%iIsland)
            #         MaskBright[x,y]=1
            #         #print(MaskBright[x,y],x,y)
            #     else:
            #         MaskFaint[x,y]=1
            # ######################################

        SkyModel.Sky.ModRegFile.radecRad2Reg("%s.bright_cal.reg"%self.FitsFile,np.array(L_raCal),np.array(L_decCal))
        OutTest="%s.bright_mask"%self.FitsFile
        ImWrite=MaskBright.reshape((1,1,nx,nx))
        ImBright=PutDataInNewImage(self.FitsFile,"%s"%OutTest,np.float32(ImWrite))
 
        OutTest="%s.faint_mask"%self.FitsFile
        ImWrite=MaskFaint.reshape((1,1,nx,nx))
        PutDataInNewImage(self.FitsFile,"%s"%OutTest,np.float32(ImWrite))
        return ImBright
