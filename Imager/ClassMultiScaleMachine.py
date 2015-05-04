import numpy as np
import pylab
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassMultiScaleMachine")
from DDFacet.Array import NpParallel
from DDFacet.Array import ModLinAlg
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt

from DDFacet.ToolsDir.GiveEdges import GiveEdges

class ClassMultiScaleMachine():

    def __init__(self,GD):
        self.SubPSF=None
        self.CubePSFScales=None
        self.GD=GD
        self.MultiFreqMode=False
        self.Alpha=np.array([-0.8],float)
        if self.GD["MultiFreqs"]["NFreqBands"]:
            self.MultiFreqMode=True
            self.NFreqBand=self.GD["MultiFreqs"]["NFreqBands"]

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe

    def SetDirtyPSF(self,DicoDirty,DicoPSF):

        self.DicoDirty=DicoDirty
        self.DicoPSF=DicoPSF
        #self.NChannels=self.DicoDirty["NChannels"]


        self._PSF=self.DicoPSF["ImagData"]
        self._Dirty=self.DicoDirty["ImagData"]
        self._MeanPSF=self.DicoPSF["MeanImage"]
        self._MeanDirty=self.DicoDirty["MeanImage"]
        _,_,NPSF,_=self._PSF.shape
        _,_,NDirty,_=self._Dirty.shape
        off=(NPSF-NDirty)/2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)
        

        # nch,_,_,_=self._PSF.shape
        # for ich in range(nch):
        #     pylab.clf()
        #     pylab.subplot(1,2,1)
        #     pylab.imshow(self._PSF[ich,0,:,:],interpolation="nearest")
        #     pylab.subplot(1,2,2)
        #     pylab.imshow(self._Dirty[ich,0,:,:],interpolation="nearest")
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        # stop



    def FindPSFExtent(self,Method="FromBox"):
        if self.SubPSF!=None: return
        PSF=self._MeanPSF
        _,_,NPSF,_=PSF.shape
        xtest=np.int64(np.linspace(NPSF/2,NPSF,100))
        box=100
        itest=0

        if Method=="FromBox":
            while True:
                X=xtest[itest]
                psf=PSF[0,0,X-box:X+box,NPSF/2-box:NPSF/2+box]
                std0=np.abs(psf.min()-psf.max())#np.std(psf)
                psf=PSF[0,0,NPSF/2-box:NPSF/2+box,X-box:X+box]
                std1=np.abs(psf.min()-psf.max())#np.std(psf)
                std=np.max([std0,std1])
                if std<1e-2:
                    break
                else:
                    itest+=1
            x0=xtest[itest]
            dx0=(x0-NPSF/2)
            print>>log, "PSF extends to [%i] from center, with rms=%.5f"%(dx0,std)
        elif Method=="FromSideLobe":
            dx0=2*self.OffsetSideLobe
            dx0=np.max([dx0,50])
            print>>log, "PSF extends to [%i] from center"%(dx0)
        
        dx0=np.max([dx0,50])
        npix=2*dx0+1
        npix=ModToolBox.GiveClosestFastSize(npix,Odd=False)

        self.PSFMargin=(NPSF-npix)/2

        dx=npix/2
        self.PSFExtent=(NPSF/2-dx,NPSF/2+dx+1,NPSF/2-dx,NPSF/2+dx+1)
        x0,x1,y0,y1=self.PSFExtent
        self.SubPSF=self._PSF[:,:,x0:x1,y0:y1]



    def MakeMultiScaleCube(self):
        if self.CubePSFScales!=None: return
        print>>log, "Making MultiScale PSFs..."
        LScales=self.GD["MultiScale"]["Scales"]
        ScaleStart=0
        if 0 in LScales: 
            ScaleStart=1
            #LScales.remove(0)
        LRatios=self.GD["MultiScale"]["Ratios"]
        NTheta=self.GD["MultiScale"]["NTheta"]

        NAlpha=1
        if self.MultiFreqMode:
            AlphaMin,AlphaMax,NAlpha=self.GD["MultiFreqs"]["Alpha"]
            Alpha=np.linspace(AlphaMin,AlphaMax,NAlpha)

        _,_,nx,ny=self.SubPSF.shape
        NScales=len(LScales)
        self.NScales=NScales
        NRatios=len(LRatios)

        Scales=np.array(LScales)
        Ratios=np.array(LRatios)


        self.ListScales=[]


        Support=31
        #CubePSFScales=np.zeros((self.NFreqBand,NScales+NRatios*NTheta*(NScales-1),nx,ny))
        ListPSFScales=[]

        # Scale Zero

        FreqBandsFluxRatio=np.zeros((NAlpha,self.NFreqBand),np.float32)

        AllFreqs=[]
        for iChannel in range(self.NFreqBand):
            AllFreqs+=self.DicoPSF["freqs"][iChannel]
        RefFreq=np.mean(AllFreqs)

        for iChannel in range(self.NFreqBand):
            for iAlpha in range(NAlpha):
                ThisAlpha=Alpha[iAlpha]
                ThisFreqs=self.DicoPSF["freqs"][iChannel]
                
                FreqBandsFluxRatio[iAlpha,iChannel]=np.mean((ThisFreqs/RefFreq)**ThisAlpha)
            
        print FreqBandsFluxRatio
        self.Alpha=Alpha
        for iAlpha in range(NAlpha):
            FluxRatios=FreqBandsFluxRatio[iAlpha,:]
            FluxRatios=FluxRatios.reshape((FluxRatios.size,1,1))
            ThisMFPSF=self.SubPSF[:,0,:,:]*FluxRatios
            ThisAlpha=Alpha[iAlpha]

            iSlice=0

            ListPSFScales.append(ThisMFPSF)
            self.ListScales.append({"ModelType":"Delta","Scale":iSlice,"Alpha":ThisAlpha})
            iSlice+=1
            
            for iScales in range(ScaleStart,NScales):
                print "sc"
                Minor=Scales[iScales]/(2.*np.sqrt(2.*np.log(2.)))
                Major=Minor
                PSFGaussPars=(Major,Minor,0.)
                ThisPSF=ModFFTW.ConvolveGaussian(ThisMFPSF,CellSizeRad=1.,GaussPars=[PSFGaussPars])[0,0]
                ListPSFScales.append(ThisPSF)
                Gauss=ModFFTW.GiveGauss(Support,CellSizeRad=1.,GaussPars=PSFGaussPars)
                #fact=np.max(Gauss)/np.sum(Gauss)
                #Gauss*=fact
                self.ListScales.append({"ModelType":"Gaussian",
                                        "Model":Gauss,"Scale":i,"Alpha":ThisAlpha})
            
            iSlice+=1
        
            Theta=np.arange(0.,np.pi-1e-3,np.pi/NTheta)
            
            for iScale in range(ScaleStart,NScales):
                print "sc2"
                for ratio in Ratios:
                    for th in Theta:
                        Minor=Scales[iScale]/(2.*np.sqrt(2.*np.log(2.)))
                        Major=Minor*ratio
                        PSFGaussPars=(Major,Minor,th)
                        ThisPSF=ModFFTW.ConvolveGaussian(ThisMFPSF,CellSizeRad=1.,GaussPars=[PSFGaussPars])[0,0]
                        Max=np.max(ThisPSF)
                        ThisPSF/=Max
                        ListPSFScales.append(ThisPSF)
                        # pylab.clf()
                        # pylab.subplot(1,2,1)
                        # pylab.imshow(CubePSFScales[0,:,:],interpolation="nearest")
                        # pylab.subplot(1,2,2)
                        # pylab.imshow(CubePSFScales[iSlice,:,:],interpolation="nearest")
                        # pylab.title("Scale = %s"%str(PSFGaussPars))
                        # pylab.draw()
                        # pylab.show(False)
                        # pylab.pause(0.1)
                        iSlice+=1
                        Gauss=ModFFTW.GiveGauss(Support,CellSizeRad=1.,GaussPars=PSFGaussPars)/Max
                        #fact=np.max(Gauss)/np.sum(Gauss)
                        #Gauss*=fact
                        self.ListScales.append({"ModelType":"Gaussian",
                                                "Model":Gauss,"Scale":iScale,"Alpha":ThisAlpha})

        # Max=np.max(np.max(CubePSFScales,axis=1),axis=1)
        # Max=Max.reshape((Max.size,1,1))
        # CubePSFScales=CubePSFScales/Max
        self.CubePSFScales=np.array(ListPSFScales)
        self.CubePSFScales=np.float32(self.CubePSFScales)


        self.WeightWidth=6
        CellSizeRad=1.
        PSFGaussPars=(self.WeightWidth,self.WeightWidth,0.)
        self.WeightFunction=ModFFTW.GiveGauss(self.SubPSF.shape[-1],CellSizeRad=1.,GaussPars=PSFGaussPars)
        nch,npol,_,_=self._PSF.shape
        self.WeightFunction=self.WeightFunction.reshape((1,1,self.SubPSF.shape[-1],self.SubPSF.shape[-1]))*np.ones((nch,npol,1,1),np.float32)
        #self.WeightFunction.fill(1)
        print>>log, "   ... Done"

    def MakeBasisMatrix(self):
        self.SupWeightWidth=3.*self.WeightWidth
        nxPSF=self.CubePSFScales.shape[-1]
        x0,x1=nxPSF/2-self.SupWeightWidth,nxPSF/2+self.SupWeightWidth+1
        y0,y1=nxPSF/2-self.SupWeightWidth,nxPSF/2+self.SupWeightWidth+1
        self.x0x1y0y1=x0,x1,y0,y1
        self.CubePSF=self.CubePSFScales[:,:,x0:x1,y0:y1]
        nFunc,nch,nx,ny=self.CubePSF.shape
        BM=(self.CubePSF.reshape((nFunc,nch*nx*ny)).T.copy())
        BMT_BM=np.dot(BM.T,BM)
        
        self.BMT_BM_inv=ModLinAlg.invSVD(BMT_BM)
        self.BM=BM


    def GiveLocalSM(self,(x,y),Fpol):
        x0,y0=x,y
        x,y=x0,y0
        x0,x1,y0,y1=self.x0x1y0y1

        N0=self._Dirty.shape[-1]
        N1=self.SubPSF.shape[-1]
        xc,yc=x,y

        CubePSF=self.CubePSF
        N1=CubePSF.shape[-1]
        
        nchan,npol,_,_=Fpol.shape
        
        Aedge,Bedge=GiveEdges((xc,yc),N0,(N1/2,N1/2),N1)
        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge
        nxp,nyp=x1p-x0p,y1p-y0p

        dirtyNorm=self._Dirty[:,:,x0d:x1d,y0d:y1d]/Fpol
        dirtyVec=dirtyNorm.reshape((dirtyNorm.size,1))

        WCubePSF=self.WeightFunction[:,:,x0:x1,y0:y1][:,:,x0p:x1p,y0p:y1p]
        WVecPSF=WCubePSF.reshape((WCubePSF.size,1))

        BM=self.BM
        BMT_BM=np.dot(BM.T,WVecPSF*BM)
        BMT_BM_inv=ModLinAlg.invSVD(BMT_BM)
        Sol=np.dot(BMT_BM_inv,np.dot(BM.T,WVecPSF*dirtyVec))
        ConvSM=np.dot(BM,Sol)
        print Sol

        nch,npol,_,_=self._Dirty.shape
        ConvSM=ConvSM.reshape((nch,npol,nxp,nyp))

        nFunc,_=self.BM.T.shape
        BBM=self.BM.T.reshape((nFunc,nch,npol,nxp,nyp))
        print np.sum(Sol.flatten()*self.Alpha.flatten())/np.sum(Sol.flatten())


        import pylab

        dv=0.2
        for iFunc in range(nFunc):

            pylab.clf()
            pylab.subplot(3,2,1)
            pylab.imshow(dirtyNorm[0,0],interpolation="nearest")
            pylab.title(self.Alpha[iFunc])
            pylab.subplot(3,2,2)
            pylab.imshow(dirtyNorm[1,0],interpolation="nearest")

            pylab.subplot(3,2,3)
            pylab.imshow(dirtyNorm[0,0]-BBM[iFunc,0,0],interpolation="nearest",vmin=-0.5,vmax=0.5)
            #pylab.colorbar()
            pylab.subplot(3,2,4)
            pylab.imshow(dirtyNorm[1,0]-BBM[iFunc,1,0],interpolation="nearest",vmin=-0.5,vmax=0.5)

            # pylab.subplot(3,2,3)
            # pylab.imshow(BBM[iFunc,0,0],interpolation="nearest",vmin=-0.5,vmax=0.5)
            # #pylab.colorbar()
            # pylab.subplot(3,2,4)
            # pylab.imshow(BBM[iFunc,1,0],interpolation="nearest",vmin=-0.5,vmax=0.5)
            # #pylab.colorbar()

            pylab.subplot(3,2,5)
            pylab.imshow(dirtyNorm[0,0]-ConvSM[0,0],interpolation="nearest",vmin=-0.5,vmax=0.5)
            #pylab.colorbar()
            pylab.subplot(3,2,6)
            pylab.imshow(dirtyNorm[1,0]-ConvSM[1,0],interpolation="nearest",vmin=-0.5,vmax=0.5)
            #pylab.colorbar()
            
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)
            
        stop
            
        
        

    def FindBestScale(self,(x,y),Fpol):
        x0,y0=x,y
        x,y=x0,y0
        
        
        N0=self.Dirty.shape[-1]
        N1=self.SubPSF.shape[-1]
        xc,yc=x,y
        
        N1=CubePSF.shape[-1]
        
        
        
        Aedge,Bedge=GiveEdges((xc,yc),N0,(N1/2,N1/2),N1)
        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge
        #print Aedge
        #print Bedge
        
        #CubePSF=self.CubePSFScales[:,x0p:x1p,y0p:y1p]*Fpol[0,0,0]
        CubePSF=CubePSF[:,x0p:x1p,y0p:y1p]*Fpol[0,0,0]
        
        dirty=self.Dirty[0,x0d:x1d,y0d:y1d]
        nx,ny=dirty.shape
        dirty=dirty.reshape((1,nx,ny))
        
        
        NSlice,nxPSF,_=self.CubePSFScales.shape
        
        WCubePSF=self.WeightFunction[x0:x1,y0:y1][x0p:x1p,y0p:y1p]
        resid=dirty-CubePSF
        WResid=WCubePSF*(dirty-CubePSF)
        resid2=(1./self.RMS**2)*WCubePSF*(resid)**2
        #resid2=(resid)**2

        chi2=np.sum(np.sum(resid2,axis=1),axis=1)/(np.sum(self.WeightFunction))

        iScale=np.argmin(chi2)

        # pylab.clf()
        # vmin=dirty.min()
        # vmax=dirty.max()
        # ax=pylab.subplot(1,3,1)
        # pylab.imshow(dirty[0],vmin=vmin,vmax=vmax,interpolation="nearest")
        # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
        # pylab.imshow(CubePSF[iScale],vmin=vmin,vmax=vmax,interpolation="nearest")
        # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
        # pylab.imshow(WResid[iScale],vmin=vmin,vmax=vmax,interpolation="nearest")
        # pylab.colorbar()
        # pylab.draw()
        # pylab.show(False)

        # # if np.min(chi2)>self.Chi2Thr:
        # #     self._MaskArray[:,:,x,y]=True
        # #     return "BadFit"

            
        # WResid=np.sum(WCubePSF*dirty[0]*CubePSF[iScale])/np.sum(WCubePSF*CubePSF[iScale]*CubePSF[iScale])


        # if WResid<0.9:
        #     self._MaskArray[:,:,x,y]=True
        #     return "BadFit"


        #print WResid
        # stop



        return iScale
