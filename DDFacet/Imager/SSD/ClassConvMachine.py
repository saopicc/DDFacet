import numpy as np
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassConvMatrix")
import scipy.signal
from DDFacet.Array import ModLinAlg
from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir.ModToolBox import EstimateNpix

def test():
    import DDFacet.ToolsDir.Gaussian
    _,_,PSF=DDFacet.ToolsDir.Gaussian.Gaussian(10,311,1.)
    #PSF.fill(1.)
    
    #import scipy.signal
    #PP=scipy.signal.fftconvolve(PSF,PSF, mode='same')
    
    #print Fact
    import pylab
    pylab.clf()
    pylab.imshow(PSF,interpolation="nearest") 
    pylab.colorbar()
    pylab.draw()
    pylab.show(False)
    pylab.pause(0.1)

    Dirty=np.zeros_like(PSF)
    nx,_=Dirty.shape
    Dirty[nx/2,nx/2+10]+=2.
    Dirty[nx/2+10,nx/2+10]+=2.
    Dirty=np.random.randn(*(Dirty.shape))
    
    PSF=PSF.reshape((1,1,nx,nx))*np.ones((2,1,1,1))
    Dirty=Dirty.reshape((1,1,nx,nx))*np.ones((2,1,1,1))
    Dirty[1,:,:,:]=Dirty[0,:,:,:]*2
    x,y=np.mgrid[0:nx,0:nx]
    dx=10
    nc=nx/2
    x=x[nc-dx:nc+dx,nc-dx:nc+dx].flatten()
    y=y[nc-dx:nc+dx,nc-dx:nc+dx].flatten()
    ListPixParms=[(x[i],y[i]) for i in range(x.size)]
    x,y=np.mgrid[0:nx,0:nx]

    dx=10
    x=x[nc-dx:nc+dx,nc-dx:nc+dx].flatten()
    y=y[nc-dx:nc+dx,nc-dx:nc+dx].flatten()
    ListPixData=[(x[i],y[i]) for i in range(x.size)]
    CC=ClassConvMachine(PSF,ListPixParms,ListPixData,"Matrix")
    
    NFreqBands,_,_,_=Dirty.shape
    NPixListParms=len(ListPixParms)
    NPixListData=len(ListPixData)
    Array=np.zeros((NFreqBands,1,NPixListParms),np.float32)
    x0,y0=np.array(ListPixParms).T
    for iBand in range(NFreqBands):
        Array[iBand,0,:]=Dirty[iBand,0,x0,y0]


    Array=Array.reshape((NFreqBands,NPixListParms))

    import pylab


    Lchi0=[]
    Lchi1=[]


    NTries=5000
    ArrKeep0=np.zeros((NTries,NPixListParms),Array.dtype)
    ArrKeep1=np.zeros((NTries,NPixListParms),Array.dtype)


    for i in range(NTries):
        Array=np.random.randn(*Array.shape)
        #T=ClassTimeIt.ClassTimeIt()
        chi0=np.sum(Array**2)
        Lchi0.append(chi0)
        ConvArray0=CC.Convolve(Array)
        chi1=np.sum(ConvArray0**2)
        #T.timeit("0")
        #ConvArray1=CC.Convolve(Array,ConvMode="Vector").ravel()
        #T.timeit("1")
        #r=chi1/chi0
        #print "%f -> %f [%r]"%(chi0,chi1,r)
        NChan,_,NN=ConvArray0.shape
        NN=int(np.sqrt(NN))
        ArrKeep0[i]=Array[0].ravel()
        ArrKeep1[i]=ConvArray0[0].ravel()
        # pylab.clf()
        # pylab.imshow(ConvArray0.reshape((2,NN,NN))[0],interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)


        Lchi1.append(chi1)
        #print np.var(Array),np.var(ConvArray0)/Fact

    Fact=CC.NormData[0]
    print np.median(np.std(ArrKeep0,axis=0)**2)
    print np.median(np.std(ArrKeep1,axis=0)**2/Fact)
    return
    
    from scipy.stats import chi2
    from DDFacet.ToolsDir.GeneDist import ClassDistMachine
    DM=ClassDistMachine()



    rv = chi2(Array.size)
    x=np.linspace(0,2*rv.moment(1),1000)
    P=rv.cdf(x)
    pylab.clf()
    pylab.subplot(2,1,1)
    #yd,xe=pylab.histogram(Lchi0,bins=100,normed=True)
    #xd=(xe[1::]+xe[0:-1])/2.
    #yd/=np.sum(yd)
    xd,yd=DM.giveCumulDist(np.array(Lchi0),Ns=100)
    #dx=xd[1]-xd[0]
    #yd/=dx
    pylab.plot(xd,yd)
    pylab.plot(x,P)
    pylab.xlim(0,1600)
    pylab.subplot(2,1,2)
    xd,yd=DM.giveCumulDist(np.array(Lchi1),Ns=20)
    # yd,xe=pylab.histogram(Lchi1,bins=100,normed=True)
    # xd=(xe[1::]+xe[0:-1])/2.
    # dx=xd[1]-xd[0]
    # yd/=np.sum(yd)
    # yd/=dx
    print np.mean(Lchi1)/Fact
    print np.mean(Lchi0)
    # #pylab.xlim(0,800)
    # #pylab.hist(Lchi1,bins=100)

    import scipy.interpolate
    cdf=scipy.interpolate.interp1d(xd, yd,"cubic")
    x=np.linspace(xd.min(),xd.max(),1000)
    #pylab.plot(x,cdf(x),ls="",marker=".")
    #pylab.plot(xd,yd,ls="",marker="s")
    
    y=cdf(x)
    x,y=xd, yd
    y=y[1::]-y[0:-1]
    x=(x[1::]+x[0:-1])/2.
    pylab.plot(x,y,ls="",marker=".")
    
    #pylab.xlim(0,1600)
    pylab.draw()
    pylab.show(False)
    

    

    # import pylab
    # pylab.clf()
    # #pylab.plot(ConvArray0.ravel())
    # pylab.imshow(PSF[0,0])
    # #pylab.plot(ConvArray1)
    # #pylab.plot(ConvArray1-ConvArray0)
    # pylab.draw()
    # pylab.show(False)
    
    stop

from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric

class ClassConvMachineImages():
    def __init__(self,PSF):
        self.PSF=PSF
        
        

    def giveConvModel(self,SubModelImageIn):
        nch,npol,N0x_in,N0y_in=SubModelImageIn.shape
        Nin=np.max([N0y_in,N0x_in])*2
        if Nin%2==0: Nin+=1
        
        SubModelImage=np.zeros((nch,1,Nin,Nin),dtype=SubModelImageIn.dtype)
        Aedge,Bedge=GiveEdgesDissymetric((N0x_in/2,N0y_in/2),(N0x_in,N0y_in),(Nin/2,Nin/2),(Nin,Nin))
        x0d,x1d,y0d,y1d=Aedge
        x0f,x1f,y0f,y1f=Bedge
        SubModelImage[...,x0f:x1f,y0f:y1f]=SubModelImageIn[...,x0d:x1d,y0d:y1d]

        PSF=self.PSF
        nPSF=PSF.shape[-1]
        AedgeP,BedgeP=GiveEdgesDissymetric((Nin/2,Nin/2),(Nin,Nin),(nPSF/2,nPSF/2),(nPSF,nPSF))
        x0dP,x1dP,y0dP,y1dP=AedgeP
        x0fP,x1fP,y0fP,y1fP=BedgeP
        SubPSF=PSF[...,x0fP:x1fP,y0fP:y1fP]

        # import pylab
        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.imshow(PSF[0,0],interpolation="nearest")
        # pylab.subplot(1,2,2)
        # pylab.imshow(SubPSF[0,0],interpolation="nearest")
        # #pylab.colorbar()
        # pylab.draw()
        # pylab.show(False)
        # stop
        
        ConvModel=np.zeros_like(SubModelImage)
        for ich in range(nch):
            for ipol in range(npol):
                ConvModel[ich,ipol]=scipy.signal.fftconvolve(SubModelImage[ich,ipol], SubPSF[ich,ipol], mode='same')

                # import pylab
                # pylab.clf()
                # ax=pylab.subplot(1,3,1)
                # pylab.imshow(SubModelImage[ich,ipol],interpolation="nearest")
                # pylab.subplot(1,3,2)
                # pylab.imshow(SubPSF[ich,ipol],interpolation="nearest")
                # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
                # pylab.imshow(ConvModel[ich,ipol],interpolation="nearest")
                # #pylab.colorbar()
                # pylab.draw()
                # pylab.show(False)
                # stop

        return ConvModel[...,x0f:x1f,y0f:y1f]


class ClassConvMachine():
    def __init__(self,PSF,ListPixParms,ListPixData,ConvMode):

        # _,_,nx,_=PSF.shape
        # dx=5
        # PSF=PSF[...,nx/2-dx:nx/2+dx+1,nx/2-dx:nx/2+dx+1]

        self.PSF=PSF
        self.ListPixParms=ListPixParms
        self.ListPixData=ListPixData
        self.NPixListParms=len(ListPixParms)
        self.NPixListData=len(ListPixData)
        self.ArrayListPixData=np.array(self.ListPixData)
        self.ArrayListPixParms=np.array(self.ListPixParms)
        self.NFreqBands,self.npol,self.NPixPSF,_=PSF.shape
        self.invCM=None
        self.ConvMode=ConvMode
        # if ConvMode==None:
        #     if self.NPixListParms<3000:
        #         self.ConvMode="Matrix"
        #     else:
        #         self.ConvMode="FFT"
        #self.ConvMode="FFT"

        if self.ConvMode=="Matrix":
            self.SetConvMatrix()
        elif self.ConvMode=="FFT":
            pass
        else:
            stop


    def Convolve(self,A,Norm=True,OutMode="Data",ConvMode=None):
        
        if ConvMode==None:
            ConvMode=self.ConvMode

        if ConvMode=="Matrix":
            return self.ConvolveMatrix(A,OutMode=OutMode)
        elif ConvMode=="Vector":
            T=ClassTimeIt.ClassTimeIt("ConvVec")
            T.disable()
            C=self.ConvolveVector(A,OutMode=OutMode)
            T.timeit()
        elif ConvMode=="FFT":
            T=ClassTimeIt.ClassTimeIt("ConvFFT")
            T.disable()
            C=self.ConvolveFFT(A,OutMode=OutMode)
            T.timeit()
            return C

    def setParamMachine(self,PM):
        self.PM=PM

    def ConvolveFFT(self,A,OutMode="Data",AddNoise=None):
        shin=A.shape
        
        T=ClassTimeIt.ClassTimeIt("ConvolveFFT")
        T.disable()
        Asq=self.PM.ModelToSquareArray(A,TypeInOut=("Parms",OutMode))
        T.timeit("0")
        NFreqBand,npol,N,_=Asq.shape
        zN=2*N+1
        zN,_=EstimateNpix(float(zN))
        zAsq=np.zeros((NFreqBand,npol,zN,zN),dtype=Asq.dtype)
        zAsq[:,:,zN/2-N/2:zN/2+N/2+1,zN/2-N/2:zN/2+N/2+1]=Asq[:,:,:,:]
        T.timeit("1")
        if AddNoise is not None:
            zAsq+=np.random.randn(*zAsq.shape)*AddNoise


        N0x=zAsq.shape[-1]
        xc0=N0x/2
        N1=self.PSF.shape[-1]
        Aedge,Bedge=GiveEdgesDissymetric((xc0,xc0),(N0x,N0x),(N1/2,N1/2),(N1,N1))
        x0d,x1d,y0d,y1d=Aedge
        x0s,x1s,y0s,y1s=Bedge
        SubPSF=self.PSF[:,:,x0s:x1s,y0s:y1s]


        #xc=self.PSF.shape[-1]/2
        #SubPSF=self.PSF[:,:,xc-N:xc+N+1,xc-N:xc+N+1]

        Conv=np.zeros_like(zAsq)
        T.timeit("2")
        for ich in range(NFreqBand):
            for ipol in range(npol):
                Conv[ich,ipol]=scipy.signal.fftconvolve(zAsq[ich,ipol], SubPSF[ich,ipol], mode='same')
                #Conv[ich,ipol]=ModFFTW.ConvolveFFTW2D((zAsq[ich,ipol]), (SubPSF[ich,ipol]))
                
                # import pylab
                # pylab.subplot(1,3,1)
                # pylab.imshow(Conv[ich,ipol][zN/2-N/2:zN/2+N/2+1,zN/2-N/2:zN/2+N/2+1],interpolation="nearest")
                # pylab.subplot(1,3,2)
                # pylab.imshow(Conv1[ich,ipol][zN/2-N/2:zN/2+N/2+1,zN/2-N/2:zN/2+N/2+1],interpolation="nearest")
                # pylab.subplot(1,3,3)
                # pylab.imshow((Conv-Conv1)[ich,ipol][zN/2-N/2:zN/2+N/2+1,zN/2-N/2:zN/2+N/2+1],interpolation="nearest")
                # pylab.colorbar()
                # pylab.draw()
                # pylab.show()

        T.timeit("3 [%s]"%str(zAsq.shape))
        A=self.PM.SquareArrayToModel(Conv[:,:,zN/2-N/2:zN/2+N/2+1,zN/2-N/2:zN/2+N/2+1],TypeInOut=(OutMode,OutMode))
        T.timeit("4")





        if OutMode=="Data":
            NPixOut=self.NPixListData
        else:
            NPixOut=self.NPixListParms
            

        # import pylab 
        # pylab.clf()
        # pylab.subplot(1,3,1)
        # pylab.imshow(zAsq[0,0],interpolation="nearest")
        # pylab.subplot(1,3,2)
        # pylab.imshow(SubPSF[0,0],interpolation="nearest")
        # pylab.subplot(1,3,3)
        # pylab.imshow(Conv[0,0],interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # stop

        return A.reshape((NFreqBand,npol,NPixOut))


    def ConvolveVector(self,A,Norm=True,OutMode="Data"):
        sh=A.shape
        if OutMode=="Data":
            OutSize=self.NPixListData
        elif OutMode=="Parms":
            OutSize=self.NPixListParms
        ConvA=np.zeros((self.NFreqBands,1,OutSize),np.float32)
        T=ClassTimeIt.ClassTimeIt("Vec")
        T.disable()
        
        for iPix in range(self.NPixListParms):
            Fch=A[:,iPix]
            if np.abs(Fch).max()==0: continue

            Vec_iPix=self.GiveConvVector(iPix,TypeOut=OutMode)
            T.timeit("GetVec")
            for iBand in range(self.NFreqBands):
                F=Fch[iBand]
                ConvA[iBand]+=F*Vec_iPix[iBand]
            T.timeit("Sum")


        return ConvA


    def ConvolveMatrix(self,A,Norm=True,OutMode="Data"):
        sh=A.shape
        if OutMode=="Data":
            CM=self.CM
            OutSize=self.NPixListData
        elif OutMode=="Parms":
            CM=self.CMParms
            OutSize=self.NPixListParms

        # A.fill(0)
        # A[...,100]=10.

        ConvA=np.zeros((self.NFreqBands,1,OutSize),np.float32)
        for iBand in range(self.NFreqBands):
            AThisBand=A[iBand]
            CF=CM[iBand,0]
            ConvA[iBand,0]=np.dot(CF,AThisBand.reshape((AThisBand.size,1))).reshape((OutSize,))


        # IM0=self.PM.ModelToSquareArray(A,TypeInOut=("Parms",OutMode))
        # IM1=self.PM.ModelToSquareArray(ConvA,TypeInOut=(OutMode,OutMode))
        # import pylab 
        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.imshow(IM0[0,0],interpolation="nearest")
        # pylab.subplot(1,2,2)
        # pylab.imshow(IM1[0,0],interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # stop

        return ConvA

    # #######################################################

    def GiveConvVector(self,iPix,TypeOut="Data"):
        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        PSF=self.PSF
        NPixPSF=PSF.shape[-1]
        xc=yc=NPixPSF/2
        T.timeit("0")
        x1,y1=self.ArrayListPixParms[iPix:iPix+1].T

        if TypeOut=="Data":
            M=np.zeros((self.NFreqBands,1,self.NPixListData,1),np.float32)
            x0,y0=self.ArrayListPixData.T
        else:
            M=np.zeros((self.NFreqBands,1,self.NPixListParms,1),np.float32)
            x0,y0=self.ArrayListPixParms.T
            

        N0=x0.size
        N1=x1.size
        T.timeit("1")
        dx=(x1.reshape((N1,1))-x0.reshape((1,N0))+xc).T
        dy=(y1.reshape((N1,1))-y0.reshape((1,N0))+xc).T
        T.timeit("2")
        Cx=((dx>=0)&(dx<NPixPSF))
        Cy=((dy>=0)&(dy<NPixPSF))
        C=(Cx&Cy)
        T.timeit("3")
        indPSF=np.arange(M.shape[-1]*M.shape[-2])
        indPSF_sel=indPSF[C.ravel()]
        indPixPSF=dx.ravel()[C.ravel()]*NPixPSF+dy.ravel()[C.ravel()]
        T.timeit("4")
        if indPSF_sel.size!=indPSF.size:
            for iBand in range(self.NFreqBands):
                PSF_Chan=PSF[iBand,0]
                M[iBand,0].flat[indPSF_sel] = PSF_Chan.flat[indPixPSF.ravel()]
            return M[:,:,:,0]
        else:
            ListVec=[]
            for iBand in range(self.NFreqBands):
                PSF_Chan=PSF[iBand,0]
                ListVec.append(PSF_Chan.flat[indPixPSF.ravel()])
            return ListVec


    def GiveInvertCov(self,Var):
        if self.invCM is None:

            self.invCM=ModLinAlg.invSVD(np.float64(self.CM[0,0]))
        return self.invCM/Var

    def SetConvMatrix(self):
        #print>>log,"SetConvMatrix"
        PSF=self.PSF
        NPixPSF=PSF.shape[-1]


        M=np.zeros((self.NFreqBands,1,self.NPixListData,self.NPixListParms),np.float32)
        xc=yc=NPixPSF/2

        x0,y0=np.array(self.ListPixData).T
        x1,y1=np.array(self.ListPixParms).T
        N0=x0.size
        N1=x1.size
        dx=(x1.reshape((N1,1))-x0.reshape((1,N0))+xc).T
        dy=(y1.reshape((N1,1))-y0.reshape((1,N0))+xc).T
        Cx=((dx>=0)&(dx<NPixPSF))
        Cy=((dy>=0)&(dy<NPixPSF))
        C=(Cx&Cy)
        indPSF=np.arange(M.shape[-1]*M.shape[-2])
        indPSF_sel=indPSF[C.ravel()]
        indPixPSF=dx.ravel()[C.ravel()]*NPixPSF+dy.ravel()[C.ravel()]
        for iBand in range(self.NFreqBands):
            PSF_Chan=PSF[iBand,0]
            M[iBand,0].flat[indPSF_sel] = PSF_Chan.flat[indPixPSF.ravel()]

        self.CM=M
        self.NormData=np.sum(M**2,axis=2).reshape((self.NFreqBands,self.NPixListParms))
        self.DirtyCMMean=np.mean(M,axis=0).reshape((1,1,self.NPixListData,self.NPixListParms))

        MParms=np.zeros((self.NFreqBands,1,self.NPixListParms,self.NPixListParms),np.float32)

        x0,y0=np.array(self.ListPixParms).T
        x1,y1=np.array(self.ListPixParms).T
        N0=x0.size
        N1=x1.size
        dx=(x1.reshape((N1,1))-x0.reshape((1,N0))+xc).T
        dy=(y1.reshape((N1,1))-y0.reshape((1,N0))+xc).T
        Cx=((dx>=0)&(dx<NPixPSF))
        Cy=((dy>=0)&(dy<NPixPSF))
        C=(Cx&Cy)
        indPSF=np.arange(MParms.shape[-1]*MParms.shape[-2])
        indPSF_sel=indPSF[C.ravel()]
        indPixPSF=dx.ravel()[C.ravel()]*NPixPSF+dy.ravel()[C.ravel()]
        for iBand in range(self.NFreqBands):
            PSF_Chan=PSF[iBand,0]
            MParms[iBand,0].flat[indPSF_sel] = PSF_Chan.flat[indPixPSF.ravel()]




        self.CMParms=MParms
        self.CMParmsMean=np.mean(MParms,axis=0).reshape((1,1,self.NPixListParms,self.NPixListParms))

        
