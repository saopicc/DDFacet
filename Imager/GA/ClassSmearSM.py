import numpy as np
import scipy.signal
import ClassConvMachine
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassSmearSM")

class ClassSmearSM():
    def __init__(self,MeanResidual,MeanModelImage,PSFServer,DeltaChi2=4.):
        self.MeanResidual=MeanResidual
        
        NPixStats=10000
        RandomInd=np.int64(np.random.rand(NPixStats)*(MeanResidual.size))
        self.RMS=np.std(np.real(self.MeanResidual.ravel()[RandomInd]))

        self.MeanModelImage=MeanModelImage
        self.PSFServer=PSFServer
        self.DeltaChi2=DeltaChi2
        self.Var=self.RMS**2
        self.NImGauss=31

        self.DicoConvMachine={}

        N=self.NImGauss
        dx,dy=np.mgrid[-(N/2):N/2:1j*N,-(N/2):N/2:1j*N]

        ListPixParms=[(int(dx.ravel()[i]),int(dy.ravel()[i])) for i in range(dx.size)]
        ListPixData=ListPixParms
        ConvMode="Matrix"
        N=self.NImGauss


        #stop
        #for 
        #ClassConvMachine():
        #def __init__(self,PSF,ListPixParms,ListPixData,ConvMode):


        d=np.sqrt(dx**2+dy**2)
        self.dist=d
        self.NGauss=10

        GSig=np.linspace(0.,5,self.NGauss)
        self.GSig=GSig
        ListGauss=[]
        One=np.zeros_like(d)
        One[N/2,N/2]=1.
        ListGauss.append(One)
        for sig in GSig[1::]:
            v=np.exp(-d**2/(2.*sig**2))
            Sv=np.sum(v)
            v/=Sv
            ListGauss.append(v)
        self.ListGauss=ListGauss
        
        print>>log, "Declare convolution machines"
        for iFacet in range(self.PSFServer.NFacets):
            PSF=self.PSFServer.DicoVariablePSF['CubeMeanVariablePSF'][iFacet]#[0,0]
            #sig=1
            #PSF=(np.exp(-self.dist**2/(2.*sig**2))).reshape(1,1,N,N)
            self.DicoConvMachine[iFacet]=ClassConvMachine.ClassConvMachine(PSF,ListPixParms,ListPixData,ConvMode)
            
        PSFMean=np.mean(self.PSFServer.DicoVariablePSF['CubeMeanVariablePSF'],axis=0)
        self.ConvMachineMeanPSF=ClassConvMachine.ClassConvMachine(PSFMean,ListPixParms,ListPixData,ConvMode)
        self.FindSupport()

    def Smear(self):
        self.ModelOut=np.zeros_like(self.MeanModelImage)
        indx,indy=np.where(self.MeanModelImage[0,0]!=0)
        #indx,indy=np.where(self.MeanModelImage==np.max(self.MeanModelImage))
        for iPix in range(indx.size):
            print iPix,"/",indx.size
            xc,yc=indx[iPix],indy[iPix]
            self.SmearThisComp(xc,yc)
        return self.ModelOut

    def GiveChi2(self,Resid):
        #Chi2=np.sum(Resid**2)/self.Var
        #return Chi2
        InvCov=self.CurrentConvMachine.GiveInvertCov(self.Var)
        NPixResid=Resid.size
        return np.dot(np.dot(Resid.reshape((1,NPixResid)),InvCov),Resid.reshape((NPixResid,1))).ravel()[0]
            

    def GiveConv(self,SubModelOrig):
        N=self.NImGauss
        ConvModel=self.CurrentConvMachine.Convolve(SubModelOrig.reshape(1,SubModelOrig.size)).reshape((N,N))
        return ConvModel

    def FindSupport(self):
        ConvMachine=self.CurrentConvMachine=self.ConvMachineMeanPSF
        N=self.NImGauss
        Dirty=np.zeros((N,N),dtype=np.float32)
        Dirty[N/2,N/2]=1.
        Dirty=self.GiveConv(Dirty)
        InvCov=ConvMachine.GiveInvertCov(1.)
        Sol=np.dot(InvCov,Dirty.reshape((Dirty.size,1))).reshape((N,N))
        Profile=(Sol[N/2,:]+Sol[:,N/2])[N/2:]
        
        xp=np.arange(N/2+1)
        Val=np.max(Profile)/2.
        xx=np.linspace(0,N/2,1000)
        a=np.interp(xx,xp,Profile-Val)
        ind=np.where(np.abs(a)==np.min(np.abs(a)))[0]
        FWHM=a[ind[0]]*2
        if FWHM<3: FWHM=3.
        self.SigMin=(FWHM/2.)/np.sqrt(2.*np.log(2.))
        print>>log, "Support for restoring beam: %5.2f pixels (sigma = %5.2f pixels)"%(FWHM,self.SigMin)

        RestoringBeam=(np.exp(-self.dist**2/(2.*self.SigMin**2))).reshape(N,N)
        
        ListRestoredGauss=[]

        for Func in self.ListGauss:
            v=scipy.signal.fftconvolve(Func, RestoringBeam, mode='same')
            ListRestoredGauss.append(v)
        self.ListRestoredGauss=ListRestoredGauss
        
        # import pylab
        # pylab.clf()
        # pylab.plot(xp,Profile)
        # pylab.scatter(xx,a)
        # # pylab.subplot(1,2,1)
        # # pylab.imshow(Dirty,interpolation="nearest")
        # # pylab.colorbar()
        # # vmax=Sol.max()
        # # pylab.subplot(1,2,2)
        # # pylab.imshow(Sol,interpolation="nearest",vmax=vmax,vmin=-0.1*vmax)
        # # pylab.colorbar()
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop


    def SmearThisComp(self,x0,y0):
        FacetID=self.PSFServer.giveFacetID2(x0,y0)
        self.CurrentConvMachine=self.DicoConvMachine[FacetID]
        PSF=self.PSFServer.DicoVariablePSF['CubeMeanVariablePSF'][FacetID][0,0]
        N=self.NImGauss
        SubModelOut=self.ModelOut[0,0][x0-N/2:x0+N/2+1,y0-N/2:y0+N/2+1]
        SubResid=self.MeanResidual[0,0][x0-N/2:x0+N/2+1,y0-N/2:y0+N/2+1]
        SubModelOrig=self.MeanModelImage[0,0][x0-N/2:x0+N/2+1,y0-N/2:y0+N/2+1].copy()
        xc=yc=N/2

        NPSF,_=PSF.shape
        
        xcPSF=ycPSF=NPSF/2
        SubPSF=PSF[xcPSF-N/2:xcPSF+N/2+1,ycPSF-N/2:ycPSF+N/2+1]
        #SubModelOrig.fill(0)
        #SubModelOrig[N/2,N/2]=1
        #SubModelOrig[N/2+10,N/2+10]=10
        #ConvModel=self.CurrentConvMachine.Convolve(SubModelOrig.reshape(1,SubModelOrig.size)).reshape((N,N))
        #ConvModel1=scipy.signal.fftconvolve(SubModelOrig, SubPSF, mode='same')
        ConvModel=self.GiveConv(SubModelOrig)

        Dirty=SubResid+ConvModel
        
        # for i in range(10):
        #     Noise=self.GiveConv(np.random.randn(*(Dirty.shape)))
        #     #Noise*=1e-1/np.max(Noise)
        #     #Dirty+=Noise
        #     Dirty=ConvModel#+Noise
        #     InvCov=self.CurrentConvMachine.GiveInvertCov(1.)#self.Var)
        #     Sol=np.dot(InvCov,Dirty.reshape((Dirty.size,1))).reshape((N,N))
        #     #Sol=np.dot(InvCov,Sol.reshape((Dirty.size,1))).reshape((N,N))
        #     import pylab
        #     pylab.clf()
        #     pylab.subplot(1,2,1)
        #     pylab.imshow(Dirty,interpolation="nearest")
        #     pylab.colorbar()
        #     vmax=Sol.max()
        #     pylab.subplot(1,2,2)

        #     pylab.imshow(Sol,interpolation="nearest",vmax=vmax,vmin=-0.1*vmax)
        #     pylab.colorbar()
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        #     stop


        



        DeltaChi2=self.DeltaChi2
        #Chi2Min=np.sum(SubResid**2)/self.Var
        Chi2Min=self.GiveChi2(SubResid)
        
        SMax=SubModelOrig[xc,yc]#self.MeanModelImage[x0,y0]
        SubModel0=SubModelOrig.copy()
        SubModel0[xc,yc]=0

        iGauss=0
        Chi2=Chi2Min
        
        while True:
            if iGauss==self.NGauss-1:
                #print "max size"
                break
            if self.GSig[iGauss]<self.SigMin:
                iGauss+=1
                continue

            v=self.ListGauss[iGauss]
            Add=v*SMax
            
            ModifiedSubModel=SubModel0+Add
            # import pylab

            # pylab.subplot(1,2,1)
            # pylab.imshow(SubModelOrig,interpolation="nearest")
            # pylab.subplot(1,2,2)
            # pylab.imshow(ModifiedSubModel,interpolation="nearest")
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            # if sig!=0:
            #     stop


            # ModifiedSubModel.fill(0)
            # ModifiedSubModel[xc,xc]=1.
            #ConvModel=scipy.signal.fftconvolve(ModifiedSubModel, SubPSF, mode='same')
            ConvModel=self.GiveConv(ModifiedSubModel)

            ThisDirty=ConvModel
            ThisResid=Dirty-ThisDirty
            #Chi2=np.sum((ThisResid)**2)/self.Var

            #Chi2=np.dot(np.dot(ThisResid.reshape((1,NPixResid)),InvCov),ThisResid.reshape((NPixResid,1)))
            Chi2=self.GiveChi2(ThisResid)#/Chi2Min


            print Chi2,Chi2Min,Chi2Min+DeltaChi2

            # print sig,Chi2,Chi2Min

            # # try:
            # import pylab
            # vmin,vmax=SubResid.min(),SubResid.max()
            # pylab.subplot(1,3,1)
            # pylab.imshow(SubResid,interpolation="nearest",vmin=vmin,vmax=vmax)
            # pylab.subplot(1,3,2)
            # pylab.imshow(ThisDirty,interpolation="nearest")#,vmin=vmin,vmax=vmax)
            # pylab.subplot(1,3,3)
            # pylab.imshow(ThisResid,interpolation="nearest")#,vmin=vmin,vmax=vmax)
            # pylab.title("%f"%Chi2)
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            # # except:
            # #     stop

            if Chi2> Chi2Min+DeltaChi2:
            #if Chi2> DeltaChi2:
                print "reached max chi2 %f"%self.GSig[iGauss]
                break

            iGauss+=1

            #import time
            #time.sleep(0.5)


        # import pylab
        # #vmin,vmax=SubResid.min(),SubResid.max()
        # pylab.imshow(ModifiedSubModel,interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        if self.GSig[iGauss]<self.SigMin:
            iGauss=0

        SubModelOut+=self.ListRestoredGauss[iGauss]*SMax
