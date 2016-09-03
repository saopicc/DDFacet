import numpy as np
#import pylab
from DDFacet.Other import MyLogger
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassMultiScaleMachine")
from DDFacet.Array import NpParallel
from DDFacet.Array import ModLinAlg
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyPickle

from DDFacet.ToolsDir.GiveEdges import GiveEdges


class ClassMultiScaleMachine():

    def __init__(self,GD,Gain=0.1,GainMachine=None):
        self.SubPSF=None
        self.GainMachine=GainMachine
        self.CubePSFScales=None
        self.GD=GD
        self.MultiFreqMode=False
        self.Alpha=np.array([0.],float)
        if self.GD["MultiFreqs"]["NFreqBands"]:
            self.MultiFreqMode=True
            self.NFreqBand=self.GD["MultiFreqs"]["NFreqBands"]


    def setModelMachine(self,ModelMachine):
        self.ModelMachine=ModelMachine



    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe

    def SetFacet(self,iFacet):
        self.iFacet=iFacet


    def SetPSF(self,PSFServer):#PSF,MeanPSF):
        #self.DicoPSF=DicoPSF
        self.PSFServer=PSFServer
        self.DicoVariablePSF=self.PSFServer.DicoVariablePSF
        PSF,MeanPSF=self.PSFServer.GivePSF()
        self._PSF=PSF#self.DicoPSF["ImagData"]
        self._MeanPSF=MeanPSF
        
        _,_,NPSF,_=self._PSF.shape
        self.NPSF=NPSF


    def SetDirty(self,DicoDirty):

        self.DicoDirty=DicoDirty
        #self.NChannels=self.DicoDirty["NChannels"]


        self._Dirty=self.DicoDirty["ImagData"]
        self._MeanDirty=self.DicoDirty["MeanImage"]
        _,_,NDirty,_=self._Dirty.shape
        NPSF=self.NPSF
        off=(NPSF-NDirty)/2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)
        

#        print>>log, "!!!!!!!!!!!"
#        self._MeanDirtyOrig=self._MeanDirty.copy()
        self.ModelMachine.setModelShape(self._Dirty.shape)

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
            #print>>log, "PSF extends to [%i] from center, with rms=%.5f"%(dx0,std)
        elif Method=="FromSideLobe":
            dx0=2*self.OffsetSideLobe
            dx0=np.max([dx0,50])
            #print>>log, "PSF extends to [%i] from center"%(dx0)
        
        dx0=np.max([dx0,200])
        dx0=np.min([dx0,NPSF/2])
        npix=2*dx0+1
        npix=ModToolBox.GiveClosestFastSize(npix,Odd=False)


        #npix=1
        self.PSFMargin=(NPSF-npix)/2

        dx=npix/2

        dx=np.min([NPSF/2,dx])
        self.PSFExtent=(NPSF/2-dx,NPSF/2+dx+1,NPSF/2-dx,NPSF/2+dx+1)

        #self.PSFExtent=(0,NPSF,0,NPSF)


        x0,x1,y0,y1=self.PSFExtent
        self.SubPSF=self._PSF[:,:,x0:x1,y0:y1]



    def MakeMultiScaleCube(self):
        if self.CubePSFScales!=None: return
        #print>>log, "Making MultiScale PSFs..."
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
            NAlpha=int(NAlpha)
            AlphaL=np.linspace(AlphaMin,AlphaMax,NAlpha)
            Alpha=np.array([0.]+[al for al in AlphaL if not(al==0.)])

        _,_,nx,ny=self.SubPSF.shape
        NScales=len(LScales)
        self.NScales=NScales
        NRatios=len(LRatios)

        Ratios=np.float32(np.array([float(r) for r in LRatios if r!="" and r!="''"]))

        Scales=np.float32(np.array([float(ls) for ls in LScales if ls!="" and ls !="''"]))


        self.ListScales=[]


        Support=31
        #Support=1

        #CubePSFScales=np.zeros((self.NFreqBand,NScales+NRatios*NTheta*(NScales-1),nx,ny))
        ListPSFScales=[]
        ListPSFScalesWeights=[]
        # Scale Zero


        ######################




        self.ModelMachine.setRefFreq(self.PSFServer.RefFreq,self.PSFServer.AllFreqs)

        RefFreq=self.PSFServer.RefFreq
        AllFreqs=self.PSFServer.AllFreqs
        self.RefFreq=RefFreq


        FreqBandsFluxRatio=self.PSFServer.GiveFreqBandsFluxRatio(self.iFacet,Alpha)
        self.FreqBandsFluxRatio=FreqBandsFluxRatio
        print self.iFacet,self.FreqBandsFluxRatio
        # if self.iFacet==96: 
        #     print 96
        #     print FreqBandsFluxRatio
        # if self.iFacet==60: 
        #     print 60
        #     print FreqBandsFluxRatio

        #FreqBandsFluxRatio.fill(1.)

        #####################

        # print "FreqBandsFluxRatio"
        # print FreqBandsFluxRatio
        self.Alpha=Alpha
        nch,_,nx,ny=self.SubPSF.shape
        for iAlpha in range(NAlpha):
            FluxRatios=FreqBandsFluxRatio[iAlpha,:]
            FluxRatios=FluxRatios.reshape((FluxRatios.size,1,1))
            ThisMFPSF=self.SubPSF[:,0,:,:]*FluxRatios
            ThisAlpha=Alpha[iAlpha]
            
            iSlice=0

            ListPSFScales.append(ThisMFPSF)
            
            self.ListScales.append({"ModelType":"Delta","Scale":iSlice,#"fact":1.,
                                    "Alpha":ThisAlpha})
            iSlice+=1
            
            for iScales in range(ScaleStart,NScales):

                Minor=Scales[iScales]/(2.*np.sqrt(2.*np.log(2.)))
                Major=Minor
                PSFGaussPars=(Major,Minor,0.)
                ThisPSF=ModFFTW.ConvolveGaussian(ThisMFPSF.reshape((nch,1,nx,ny)),CellSizeRad=1.,GaussPars=[PSFGaussPars]*self.NFreqBand)[:,0,:,:]#[0,0]
                Max=np.max(ThisPSF)
                ThisPSF/=Max
                ThisSupport=int(np.max([Support,3*Major]))
                Gauss=ModFFTW.GiveGauss(Support,CellSizeRad=1.,GaussPars=PSFGaussPars)
                fact=np.max(Gauss)/np.sum(Gauss)
                #fact=1./np.sum(Gauss)
                Gauss*=fact
                #ThisPSF*=fact
                ListPSFScales.append(ThisPSF)
                self.ListScales.append({"ModelType":"Gaussian",#"fact":fact,
                                        "Model":Gauss,"Scale":iScales,"Alpha":ThisAlpha})
            
            iSlice+=1
        
            Theta=np.arange(0.,np.pi-1e-3,np.pi/NTheta)
            
            for iScale in range(ScaleStart,NScales):

                for ratio in Ratios:
                    for th in Theta:
                        Minor=Scales[iScale]/(2.*np.sqrt(2.*np.log(2.)))
                        Major=Minor*ratio
                        PSFGaussPars=(Major,Minor,th)
                        ThisPSF=ModFFTW.ConvolveGaussian(ThisMFPSF.reshape((nch,1,nx,ny)),CellSizeRad=1.,GaussPars=[PSFGaussPars]*self.NFreqBand)[:,0,:,:]
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
                                                "Model":Gauss,"Scale":iScale,
                                                "Alpha":ThisAlpha})

        # Max=np.max(np.max(CubePSFScales,axis=1),axis=1)
        # Max=Max.reshape((Max.size,1,1))
        # CubePSFScales=CubePSFScales/Max

        # for iChannel in range(self.NFreqBand):
        #     Flat=np.zeros((nch,nx,ny),ThisPSF.dtype)
        #     Flat[iChannel]=1
        #     ListPSFScales.append(Flat)

        self.ModelMachine.setListComponants(self.ListScales)

        self.CubePSFScales=np.array(ListPSFScales)
        self.FFTMachine=ModFFTW.FFTW_2Donly_np(self.CubePSFScales.shape, self.CubePSFScales.dtype)


        self.nFunc=self.CubePSFScales.shape[0]
        self.AlphaVec=np.array([Sc["Alpha"] for Sc in self.ListScales])

        self.WeightWidth=10
        CellSizeRad=1.
        PSFGaussPars=(self.WeightWidth,self.WeightWidth,0.)
        self.GlobalWeightFunction=ModFFTW.GiveGauss(self.SubPSF.shape[-1],CellSizeRad=1.,GaussPars=PSFGaussPars)
        nch,npol,_,_=self._PSF.shape

        # N=self.SubPSF.shape[-1]
        # dW=N/2
        # Wx,Wy=np.mgrid[-dW:dW:1j*N,-dW:dW:1j*N]
        # r=np.sqrt(Wx**2+Wy**2)
        # print r
        # r0=self.WeightWidth
        # weight=(r/r0+1.)**(-1)
        self.GlobalWeightFunction=self.GlobalWeightFunction.reshape((1,1,self.SubPSF.shape[-1],self.SubPSF.shape[-1]))*np.ones((nch,npol,1,1),np.float32)
        #self.GlobalWeightFunction.fill(1)

        ScaleMax=np.max(Scales)
        # self.SupWeightWidth=ScaleMax
        self.SupWeightWidth=3.*self.WeightWidth
        # print "!!!!!!!!!!!!! 0"
        # self.SupWeightWidth=0
       


        #print>>log, "   ... Done"

    def MakeBasisMatrix(self):
        nxPSF=self.CubePSFScales.shape[-1]
        x0,x1=nxPSF/2-self.SupWeightWidth,nxPSF/2+self.SupWeightWidth+1
        y0,y1=nxPSF/2-self.SupWeightWidth,nxPSF/2+self.SupWeightWidth+1
        self.SubSubCoord=(x0,x1,y0,y1)
        self.SubCubePSF=self.CubePSFScales[:,:,x0:x1,y0:y1]
        self.SubWeightFunction=self.GlobalWeightFunction[:,:,x0:x1,y0:y1]
        self.DicoBasisMatrix=self.GiveBasisMatrix()


    def GiveBasisMatrix(self,SubSubSubCoord=None):
#        print>>log,"Calculating basisc function for SubSubSubCoord=%s"%(str(SubSubSubCoord))
        if SubSubSubCoord==None:
            CubePSF=self.SubCubePSF
            WeightFunction=self.SubWeightFunction
        else:
            x0s,x1s,y0s,y1s=SubSubSubCoord
            CubePSF=self.SubCubePSF[:,:,x0s:x1s,y0s:y1s]
            WeightFunction=self.SubWeightFunction[:,:,x0s:x1s,y0s:y1s]

        nFunc,nch,nx,ny=CubePSF.shape
        # Bias=np.zeros((nFunc,),float)
        # for iFunc in range(nFunc):
        #     Bias[iFunc]=np.sum(CubePSF[iFunc]*WeightFunction[:,0,:,:])

        # Bias/=np.sum(Bias)
        # self.Bias=Bias
        # stop
        #BM=(CubePSFNorm.reshape((nFunc,nch*nx*ny)).T.copy())



        BM=(CubePSF.reshape((nFunc,nch*nx*ny)).T.copy())
        WVecPSF=WeightFunction.reshape((WeightFunction.size,1))
        BMT_BM=np.dot(BM.T,WVecPSF*BM)
        BMT_BM_inv=ModLinAlg.invSVD(BMT_BM)

        #fCubePSF=np.float32(self.FFTMachine.fft(np.complex64(CubePSF)).real)
        W=WeightFunction.reshape((1,nch,nx,ny))
        self.OPFT=np.real
        self.OPFT=np.abs
        fCubePSF=np.float32(self.OPFT(self.FFTMachine.fft(np.complex64(CubePSF*W))))
        nch,npol,_,_=self._PSF.shape
        u,v=np.mgrid[-nx/2+1:nx/2:1j*nx,-ny/2+1:ny/2:1j*ny]

        r=np.sqrt(u**2+v**2)
        r0=1.
        UVTaper=1.-np.exp(-(r/r0)**2)
        
        UVTaper=UVTaper.reshape((1,1,nx,ny))*np.ones((nch,npol,1,1),np.float32)


        UVTaper.fill(1)

        self.WeightMeanJonesBand=self.DicoVariablePSF["MeanJonesBand"][self.iFacet].reshape((nch,1,1,1))
        WeightMueller=self.WeightMeanJonesBand.ravel()
        WeightMuellerSignal=WeightMueller*self.DicoVariablePSF["WeightChansImages"].ravel()
        self.WeightMuellerSignal=WeightMuellerSignal
        UVTaper*=WeightMuellerSignal.reshape((nch,1,1,1))

        # fCubePSF[:,:,nx/2,ny/2]=0
        # import pylab
        # for iFunc in range(self.nFunc):
        #     Basis=fCubePSF[iFunc]
        #     pylab.clf()
        #     pylab.subplot(1,3,1)
        #     pylab.imshow(Basis[0]*UVTaper[0,0],interpolation="nearest")
        #     pylab.title(iFunc)
        #     pylab.subplot(1,3,2)
        #     pylab.imshow(Basis[1]*UVTaper[0,0],interpolation="nearest")
        #     pylab.subplot(1,3,3)
        #     pylab.imshow(Basis[2]*UVTaper[0,0],interpolation="nearest")
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)



        fBM=(fCubePSF.reshape((nFunc,nch*nx*ny)).T.copy())
        fBMT_fBM=np.dot(fBM.T,UVTaper.reshape((UVTaper.size,1))*fBM)
        fBMT_fBM_inv=ModLinAlg.invSVD(fBMT_fBM)
        
        # DeltaMatrix=np.zeros((nFunc,),np.float32)
        # #BM_BMT=np.dot(BM,BM.T)
        # #BM_BMT_inv=ModLinAlg.invSVD(BM_BMT)

        # BM_BMT_inv=np.diag(1./np.sum(BM*BM,axis=1))
        # nData,_=BM.shape
        # for iFunc in range(nFunc):
        #     ai=BM[:,iFunc].reshape((nData,1))
        #     DeltaMatrix[iFunc]=1./np.sqrt(np.dot(np.dot(ai.T,BM_BMT_inv),ai))
        # DeltaMatrix=DeltaMatrix.reshape((nFunc,1))
        # print>>log, "Delta Matrix: %s"%str(DeltaMatrix)

        BMnorm=np.sum(BM**2,axis=0)
        BMnorm=1./BMnorm.reshape((nFunc,1))

        DicoBasisMatrix={"BMCube":CubePSF,
                         "BMnorm":BMnorm,
                         #"DeltaMatrix":DeltaMatrix,
                         #"Bias":Bias,
                         "BM":BM,
                         "fBM":fBM,
                         "BMT_BM_inv":BMT_BM_inv,
                         "fBMT_fBM_inv":fBMT_fBM_inv,
                         "CubePSF":CubePSF,
                         "WeightFunction":(WeightFunction),
                         "fWeightFunction":UVTaper,
                         "FreqBandsFluxRatio":self.FreqBandsFluxRatio,
                         "CubePSFScales":self.CubePSFScales}

        return DicoBasisMatrix
        
        



    def GiveLocalSM(self,(x,y),Fpol):
        T=ClassTimeIt.ClassTimeIt("   GiveLocalSM")
        T.disable()
        x0,y0=x,y
        x,y=x0,y0

        N0=self._Dirty.shape[-1]
        N1=self.DicoBasisMatrix["CubePSF"].shape[-1]
        xc,yc=x,y

        #N1=CubePSF.shape[-1]
        

        nchan,npol,_,_=Fpol.shape

        JonesNorm=np.ones((nchan,npol,1,1),Fpol.dtype)
        #print self.DicoDirty.keys()
        #print Fpol
        FpolTrue=Fpol
        if self.DicoDirty["NormData"]!=None:
            JonesNorm=(self.DicoDirty["NormData"][:,:,x,y]).reshape((nchan,npol,1,1))
            
            FpolTrue=Fpol/np.sqrt(JonesNorm)
            #print JonesNorm

        # #print Fpol
        # print "JonesNorm",JonesNorm
        # FpolMean=np.mean(Fpol,axis=0).reshape((1,npol,1,1))

        Aedge,Bedge=GiveEdges((xc,yc),N0,(N1/2,N1/2),N1)
        x0d,x1d,y0d,y1d=Aedge
        x0s,x1s,y0s,y1s=Bedge
        nxs,nys=x1s-x0s,y1s-y0s
        
        if (nxs!=self.DicoBasisMatrix["CubePSF"].shape[-2])|(nys!=self.DicoBasisMatrix["CubePSF"].shape[-1]):
            DicoBasisMatrix=self.GiveBasisMatrix((x0s,x1s,y0s,y1s))
        else:
            DicoBasisMatrix=self.DicoBasisMatrix

        CubePSF=DicoBasisMatrix["CubePSF"]

        nxp,nyp=x1s-x0s,y1s-y0s
        T.timeit("0")
        dirtyNormIm=self._Dirty[:,:,x0d:x1d,y0d:y1d]
        #MeanData=np.sum(np.sum(dirtyNorm*WCubePSF,axis=-1),axis=-1)
        #MeanData=MeanData.reshape(nchan,1,1)
        #dirtyNorm=dirtyNorm-MeanData.reshape((nchan,1,1,1))

        # dirtyNormIm=dirtyNormIm/FpolMean


        #print "0",np.max(dirtyNormIm)
        dirtyNormIm=dirtyNormIm/np.sqrt(JonesNorm)
        #print "1",np.max(dirtyNormIm)

        self.Repr="FT"
        self.Repr="IM"
        


        if self.Repr=="FT":
            BM=DicoBasisMatrix["fBM"]
            WCubePSF=DicoBasisMatrix["fWeightFunction"]#*(JonesNorm)
            WCubePSFIm=DicoBasisMatrix["WeightFunction"]#*(JonesNorm)
            WVecPSF=WCubePSF.reshape((WCubePSF.size,1))
            dirtyNorm=np.float32(self.OPFT(self.FFTMachine.fft(np.complex64(dirtyNormIm*WCubePSFIm))))#.real)
            BMT_BM_inv=DicoBasisMatrix["fBMT_fBM_inv"]
        else:
            #print "0:",DicoBasisMatrix["WeightFunction"].shape,JonesNorm.shape
            WCubePSF=DicoBasisMatrix["WeightFunction"]#*(JonesNorm)
            WVecPSF=WCubePSF.reshape((WCubePSF.size,1))
            dirtyNorm=dirtyNormIm
            BM=DicoBasisMatrix["BM"]
            BMT_BM_inv=DicoBasisMatrix["BMT_BM_inv"]


        dirtyVec=dirtyNorm.reshape((dirtyNorm.size,1))
        T.timeit("1")

        # print "!!!!!!!!!!!!! fill"
        # dirtyVec.fill(10.)
        

        # BMCube=DicoBasisMatrix["BMCube"]
        # nch,_,_=MeanData.shape
        # BMCubeSub=BMCube.copy()
        # nData,nFunc=BM.shape
        # for iFunc in range(nFunc):
        #     BMCubeSub[iFunc]=BMCube[iFunc]-MeanData
        # BM=(BMCubeSub.reshape((nFunc,nData)).T.copy())

        # ModLinAlg.invSVD(BMT_BM)

        # BMT_BM=np.dot(BM.T,WVecPSF*BM)
        # BMT_BM_inv=ModLinAlg.invSVD(BMT_BM)


        self.SolveMode="MatchingPursuit"
        self.SolveMode="PI"
        #self.SolveMode="ComplementaryMatchingPursuit"
        #self.SolveMode="NNLS"

        MeanFluxTrue=np.sum(FpolTrue.ravel()*self.WeightMuellerSignal)/np.sum(self.WeightMuellerSignal)
        
        
        if  self.SolveMode=="MatchingPursuit":
            #Sol=np.dot(BM.T,WVecPSF*dirtyVec)
            Sol=np.dot(BMT_BM_inv,np.dot(BM.T,WVecPSF*dirtyVec))
            #print Sol
            #indMaxSol1=np.where(np.abs(Sol)==np.max(np.abs(Sol)))[0]
            #indMaxSol0=np.where(np.abs(Sol)!=np.max(np.abs(Sol)))[0]
            indMaxSol1=np.where(np.abs(Sol)==np.max(np.abs(Sol)))[0]
            indMaxSol0=np.where(np.abs(Sol)!=np.max(np.abs(Sol)))[0]
            #indMaxSol1=np.where(np.abs(Sol)==np.max((Sol)))[0]
            #indMaxSol0=np.where(np.abs(Sol)!=np.max((Sol)))[0]

            Sol[indMaxSol0]=0
            Max=Sol[indMaxSol1[0]]
            Sol[indMaxSol1]=MeanFluxTrue#np.sign(Max)*MeanFluxTrue

            # D=self.ListScales[indMaxSol1[0]]
            # print "Type %10s (sc, alpha)=(%i, %f)"%(D["ModelType"],D["Scale"],D["Alpha"])
            LocalSM=self.CubePSFScales[indMaxSol1[0]]*MeanFluxTrue#FpolMean.ravel()[0]
            # LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)

        elif  self.SolveMode=="ComplementaryMatchingPursuit":
            #Sol=DicoBasisMatrix["DeltaMatrix"]*np.dot(BM.T,WVecPSF*dirtyVec)
            #Sol=DicoBasisMatrix["BMnorm"]*np.dot(BM.T,WVecPSF*dirtyVec)
            Sol=DicoBasisMatrix["BMnorm"]*np.dot(BM.T,WVecPSF*(dirtyVec/MeanFluxTrue-BM))
            #Sol=np.dot(BM.T,WVecPSF*dirtyVec)
            print x0,y0,Sol
            indMaxSol1=np.where(np.abs(Sol)==np.max(np.abs(Sol)))[0]
            indMaxSol0=np.where(np.abs(Sol)!=np.max(np.abs(Sol)))[0]

            Sol[indMaxSol0]=0
            Max=Sol[indMaxSol1[0]]
            Sol[indMaxSol1]=MeanFluxTrue#np.sign(Max)*MeanFluxTrue

            # D=self.ListScales[indMaxSol1[0]]
            # print "Type %10s (sc, alpha)=(%i, %f)"%(D["ModelType"],D["Scale"],D["Alpha"])
            # LocalSM=self.CubePSFScales[indMaxSol1[0]]*FpolMean.ravel()[0]
            LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)

        elif self.SolveMode=="PI":
            

            
            Sol=np.dot(BMT_BM_inv,np.dot(BM.T,WVecPSF*dirtyVec))
            #Sol.fill(1)

            #LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)*FpolMean.ravel()[0]


            #Sol*=np.sum(FpolTrue.ravel()*self.DicoDirty["WeightChansImages"].ravel())/np.sum(Sol)
            

            coef=np.min([np.abs(np.sum(Sol)/MeanFluxTrue),1.])

            # # ############## debug
            # print
            # print "=====",self.iFacet,x,y
            # print Fpol.ravel()
            # print FpolTrue.ravel()
            # print self.DicoDirty["WeightChansImages"].ravel()
            # print "Data shape",dirtyVec.shape
            # print dirtyVec
            # # #print "BM",BM.shape
            # # #print BM
            # print "Sum, Sol",np.sum(Sol),Sol.ravel()
            # # print "aaa",np.dot(BM,Sol)
            # # stop
            # #print "FpolTrue,WeightChansImages:",FpolTrue.ravel(),self.DicoDirty["WeightChansImages"].ravel()
            # print "MeanFluxTrue",MeanFluxTrue
            # print "coef",coef

            # MyPickle.Save(DicoBasisMatrix,"BM.GAClean")
            # stop
            # # ##########################

            SolReg=np.zeros_like(Sol)
            SolReg[0]=MeanFluxTrue
            #print "SolReg",SolReg.ravel()

            if np.sign(SolReg[0])!=np.sign(np.sum(Sol)):
                Sol=SolReg
            else:
                Sol=Sol*coef+SolReg*(1.-coef)
                # if np.abs(np.sum(Sol))>np.abs(MeanFluxTrue):
                #     Sol=SolReg

            # print "Sum, Sol",np.sum(Sol),Sol.ravel()
            

            
            Sol*=(MeanFluxTrue/np.sum(Sol))
                
            # print "Sum, Sol",np.sum(Sol),Sol.ravel()
            

            LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)

            # print "Min Max dirty",dirtyNormIm.min(),dirtyNormIm.max()

            # #print "Max abs model",np.max(np.abs(LocalSM))
            # print "Min Max model",LocalSM.min(),LocalSM.max()
        elif self.SolveMode=="NNLS":
            import scipy.optimize

            A=BM
            y=dirtyVec
            x,_=scipy.optimize.nnls(A, y.ravel())
            Sol=x
            LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)
            
            # P=set()
            # R=set(range(self.nFunc))
            # x=np.zeros((self.nFunc,1),np.float32)
            # s=np.zeros((self.nFunc,1),np.float32)
            # A=BM
            # y=dirtyVec
            # w=np.dot(A.T,y-np.dot(A,x))
            # print>>log, "init w: %s"%str(w.ravel())
            # while (len(R)>0):
            #     print>>log, "while j (len(R)>0)"
            #     j=np.argmax(w)
            #     print>>log, "selected j: %i"%j
            #     print>>log, "P: %s"%str(P)
            #     print>>log, "R: %s"%str(R)
            #     P.add(j)
            #     R.remove(j)
            #     print>>log, "P: %s"%str(P)
            #     print>>log, "R: %s"%str(R)
            #     LP=sorted(list(P))
            #     LR=sorted(list(R))
            #     Ap=A[:,LP]
            #     ApT_Ap_inv=ModLinAlg.invSVD(np.dot(Ap.T,Ap))
            #     sp=np.dot(ApT_Ap_inv,np.dot(Ap.T,y))
            #     s[LP,0]=sp[:,0]
            #     print>>log, "P: %s, s: %s"%(str(P),str(s.ravel()))
            #     while np.min(sp)<=0.:
            #         alpha=np.min([x[i,0]/(x[i,0]-s[i,0]) for i in LP if s[i,0]<0])
            #         print>>log, "  Alpha= %f"%alpha
            #         x=x+alpha*(s-x)
            #         print>>log, "  x= %s"%str(x)
            #         for j in LP:
            #             if x[j,0]==0: 
            #                 R.add(j)
            #                 P.remove(j)
            #         LP=sorted(list(P))
            #         LR=sorted(list(R))
            #         Ap=A[:,LP]
            #         ApT_Ap_inv=ModLinAlg.invSVD(np.dot(Ap.T,Ap))
            #         sp=np.dot(ApT_Ap_inv,np.dot(Ap.T,y))
            #         print>>log, "  sp= %s"%str(sp)
            #         s[LP,0]=sp[:,0]
            #         s[LR,0]=0.
            #     x=s
            #     w=np.dot(A.T,y-np.dot(A,x))
            #     print>>log, "x: %s, w: %s"%(str(x.ravel()),str(w.ravel()))
                    
                    
            # Sol=x
            # LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)
            
        nch,nx,ny=LocalSM.shape
        LocalSM=LocalSM.reshape((nch,1,nx,ny))
        LocalSM=LocalSM*np.sqrt(JonesNorm)
        

        # print self.AlphaVec,Sol
        # print "alpha",np.sum(self.AlphaVec.ravel()*Sol.ravel())/np.sum(Sol)

        FpolMean=1.
        self.ModelMachine.AppendComponentToDictStacked((xc,yc),FpolMean,Sol)

        BM=DicoBasisMatrix["BM"]
        #print "MaxSM=",np.max(LocalSM)
        ConvSM=((np.dot(BM,Sol)).ravel())#*(WVecPSF.ravel())
        #Sol/=self.Bias

        #Sol[-self.NFreqBand::]=0
        # Sol=np.dot(BM.T,WVecPSF*dirtyVec)
        # Sol[Sol<0]=0

        #print Sol.flatten()

        T.timeit("2")
        
        #print>>log,( "Sol:",Sol)
        #print>>log, ("MaxLSM:",np.max(LocalSM))
        T.timeit("3")

        #print Sol

        T.timeit("4")

        nch,npol,_,_=self._Dirty.shape
        ConvSM=ConvSM.reshape((nch,npol,nxp,nyp))

        nFunc,_=BM.T.shape
        BBM=BM.T.reshape((nFunc,nch,npol,nxp,nyp))
        

        

        #print np.sum(Sol.flatten()*self.Alpha.flatten())/np.sum(Sol.flatten())

        T.timeit("5")



#         ##############################
#         #ConvSM*=FpolMean.ravel()[0]
#         import pylab

#         dv=1
# #        for iFunc in range(nFunc):#[0]:#range(nFunc):
#         for iFunc in [0]:#range(nFunc):

#             pylab.clf()
#             iplot=1
#             nxp,nyp=3,3
            
#             FF=ConvSM[:,0]#BBM[iFunc,:,0]
#             Resid=dirtyNormIm[:,0]-FF[:]
#             #FF=BBM[iFunc,:,0]
#             Resid*=DicoBasisMatrix["WeightFunction"][:,0,:,:]
#             vmin,vmax=np.min([dirtyNormIm[0,0],ConvSM[0,0],dirtyNormIm[0,0]-FF[0]]),np.max([dirtyNormIm[0,0],ConvSM[0,0],dirtyNormIm[0,0]-FF[0]])

#             ax=pylab.subplot(nxp,nyp,iplot); iplot+=1
#             pylab.imshow(dirtyNormIm[0,0],interpolation="nearest",vmin=vmin,vmax=vmax)#)
#             pylab.colorbar()
#             pylab.subplot(nxp,nyp,iplot); iplot+=1
#             #pylab.imshow(dirtyNormIm[1,0],interpolation="nearest",vmin=vmin,vmax=vmax)#)
#             pylab.subplot(nxp,nyp,iplot); iplot+=1
#             #pylab.imshow(dirtyNormIm[2,0],interpolation="nearest",vmin=vmin,vmax=vmax)#)
            
#             pylab.subplot(nxp,nyp,iplot,sharex=ax,sharey=ax); iplot+=1
#             pylab.imshow(ConvSM[0,0],interpolation="nearest",vmin=vmin,vmax=vmax)#)#,vmin=-0.5,vmax=0.5)
#             pylab.colorbar()
#             pylab.subplot(nxp,nyp,iplot); iplot+=1
#             #pylab.imshow(ConvSM[1,0],interpolation="nearest",vmin=vmin,vmax=vmax)#)#,vmin=-0.5,vmax=0.5)
#             pylab.subplot(nxp,nyp,iplot); iplot+=1
#             #pylab.imshow(ConvSM[2,0],interpolation="nearest",vmin=vmin,vmax=vmax)#)#,vmin=-0.5,vmax=0.5)

#             pylab.subplot(nxp,nyp,iplot,sharex=ax,sharey=ax); iplot+=1
#             pylab.imshow(Resid[0],interpolation="nearest")#,vmin=vmin,vmax=vmax)#,vmin=-0.5,vmax=0.5)
#             pylab.colorbar()
#             pylab.subplot(nxp,nyp,iplot); iplot+=1
#             #pylab.imshow(Resid[1],interpolation="nearest")#,vmin=vmin,vmax=vmax)#,vmin=-0.5,vmax=0.5)
#             pylab.colorbar()
#             pylab.subplot(nxp,nyp,iplot); iplot+=1
#             #pylab.imshow(Resid[2],interpolation="nearest")#,vmin=vmin,vmax=vmax)#,vmin=-0.5,vmax=0.5)
#             pylab.colorbar()


#             # pylab.subplot(3,2,iplot)
#             # pylab.imshow(BBM[iFunc,0,0],interpolation="nearest",vmin=-0.5,vmax=0.5)
#             # #pylab.colorbar()
#             # pylab.subplot(3,2,iplot)
#             # pylab.imshow(BBM[iFunc,1,0],interpolation="nearest",vmin=-0.5,vmax=0.5)
#             # #pylab.colorbar()

#             #pylab.colorbar()
            
#             pylab.draw()
#             pylab.show(False)
#             pylab.pause(0.1)

#             # stop

        return LocalSM
        stop
            

#################

