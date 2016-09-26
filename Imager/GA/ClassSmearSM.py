import numpy as np
import scipy.signal

class ClassSmearSM():
    def __init__(self,MeanResidual,MeanModelImage,PSFServer,DeltaChi2=40.):
        self.MeanResidual=MeanResidual
        self.MeanModelImage=MeanModelImage
        self.PSFServer=PSFServer
        self.DeltaChi2=DeltaChi2
        self.Var=1e-4

        self.NImGauss=31
        N=self.NImGauss
        dx,dy=np.mgrid[-(N/2):N/2:1j*N,-(N/2):N/2:1j*N]
        d=np.sqrt(dx**2+dy**2)
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

    def Smear(self):
        self.ModelOut=np.zeros_like(self.MeanModelImage)
        indx,indy=np.where(self.MeanModelImage[0,0]!=0)
        #indx,indy=np.where(self.MeanModelImage==np.max(self.MeanModelImage))
        for iPix in range(indx.size):
            print iPix,"/",indx.size
            xc,yc=indx[iPix],indy[iPix]
            self.SmearThisComp(xc,yc)
        return self.ModelOut



    def SmearThisComp(self,x0,y0):
        FacetID=self.PSFServer.giveFacetID2(x0,y0)
        PSF=self.PSFServer.DicoVariablePSF["CubeVariablePSF"][FacetID][0,0]
        N=31
        SubModelOut=self.ModelOut[0,0][x0-N/2:x0+N/2+1,y0-N/2:y0+N/2+1]
        SubResid=self.MeanResidual[0,0][x0-N/2:x0+N/2+1,y0-N/2:y0+N/2+1]
        SubModelOrig=self.MeanModelImage[0,0][x0-N/2:x0+N/2+1,y0-N/2:y0+N/2+1].copy()
        xc=yc=N/2

        NPSF,_=PSF.shape
        
        xcPSF=ycPSF=NPSF/2
        SubPSF=PSF[xcPSF-N/2:xcPSF+N/2+1,ycPSF-N/2:ycPSF+N/2+1]

        ConvModel=scipy.signal.fftconvolve(SubModelOrig, SubPSF, mode='same')
        Dirty=SubResid+ConvModel


        DeltaChi2=self.DeltaChi2
        Chi2Min=np.sum(SubResid**2)/self.Var
        

        
        SMax=SubModelOrig[xc,yc]#self.MeanModelImage[x0,y0]
        SubModel0=SubModelOrig.copy()
        SubModel0[xc,yc]=0

        iGauss=0
        Chi2=Chi2Min
        
        while True:
            if iGauss==self.NGauss-1:
                #print "max size"
                break
            if Chi2> Chi2Min+DeltaChi2:
                print "reached max chi2 %f"%self.GSig[iGauss]
                break

            v=self.ListGauss[iGauss]
            iGauss+=1
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
            ConvModel=scipy.signal.fftconvolve(ModifiedSubModel, SubPSF, mode='same')
            ThisDirty=ConvModel
            ThisResid=Dirty-ThisDirty
            Chi2=np.sum((ThisResid)**2)/self.Var

            # print sig,Chi2,Chi2Min

            # try:
            #     import pylab
            #     vmin,vmax=SubResid.min(),SubResid.max()
            #     pylab.subplot(1,3,1)
            #     pylab.imshow(SubResid,interpolation="nearest",vmin=vmin,vmax=vmax)
            #     pylab.subplot(1,3,2)
            #     pylab.imshow(ThisDirty,interpolation="nearest",vmin=vmin,vmax=vmax)
            #     pylab.subplot(1,3,3)
            #     pylab.imshow(ThisResid,interpolation="nearest",vmin=vmin,vmax=vmax)
            #     pylab.title("%f"%Chi2)
            #     pylab.draw()
            #     pylab.show(False)
            #     pylab.pause(0.1)
            # except:
            #     stop

            

            #print Chi2,Chi2Min+DeltaChi2


        # import pylab
        # #vmin,vmax=SubResid.min(),SubResid.max()
        # pylab.imshow(ModifiedSubModel,interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        SubModelOut+=Add
