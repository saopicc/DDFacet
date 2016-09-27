import numpy as np
import pylab
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassConvMatrix")
import scipy.signal
from DDFacet.Array import ModLinAlg

def test():
    import DDFacet.ToolsDir.Gaussian
    _,_,PSF=DDFacet.ToolsDir.Gaussian.Gaussian(10,311,0.5)
    Dirty=np.zeros_like(PSF)
    nx,_=Dirty.shape
    Dirty[nx/2,nx/2+10]+=2.
    Dirty[nx/2+10,nx/2+10]+=2.
    Dirty=np.random.randn(*(Dirty.shape))
    
    PSF=PSF.reshape((1,1,nx,nx))*np.ones((2,1,1,1))
    Dirty=Dirty.reshape((1,1,nx,nx))*np.ones((2,1,1,1))
    Dirty[1,:,:,:]=Dirty[0,:,:,:]*2
    x,y=np.mgrid[0:nx,0:nx]
    dx=100
    nc=nx/2
    x=x[nc-dx:nc+dx,nc-dx:nc+dx].flatten()
    y=y[nc-dx:nc+dx,nc-dx:nc+dx].flatten()
    ListPixParms=[(x[i],y[i]) for i in range(x.size)]
    x,y=np.mgrid[0:nx,0:nx]

    dx=102
    x=x[nc-dx:nc+dx,nc-dx:nc+dx].flatten()
    y=y[nc-dx:nc+dx,nc-dx:nc+dx].flatten()
    ListPixData=[(x[i],y[i]) for i in range(x.size)]
    CC=ClassConvMachine(PSF,ListPixParms,ListPixData)
    
    NFreqBands,_,_,_=Dirty.shape
    NPixListParms=len(ListPixParms)
    NPixListData=len(ListPixData)
    Array=np.zeros((NFreqBands,1,NPixListParms),np.float32)
    x0,y0=np.array(ListPixParms).T
    for iBand in range(NFreqBands):
        Array[iBand,0,:]=Dirty[iBand,0,x0,y0]


    Array=Array.reshape((NFreqBands,NPixListParms))

    T=ClassTimeIt.ClassTimeIt()
    ConvArray0=CC.Convolve(Array).ravel()
    T.timeit("0")
    ConvArray1=CC.Convolve(Array,ConvMode="Vector").ravel()
    T.timeit("1")

    pylab.clf()
    pylab.plot(ConvArray0)
    pylab.plot(ConvArray1)
    pylab.plot(ConvArray1-ConvArray0)
    pylab.draw()
    pylab.show(False)
    
    stop

class ClassConvMachine():
    def __init__(self,PSF,ListPixParms,ListPixData,ConvMode):
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

    def ConvolveFFT(self,A,OutMode="Data"):
        shin=A.shape
        
        T=ClassTimeIt.ClassTimeIt("ConvolveFFT")
        T.disable()
        Asq=self.PM.ModelToSquareArray(A,TypeInOut=("Parms",OutMode))
        T.timeit("0")
        NFreqBand,npol,N,_=Asq.shape
        zN=2*N+1
        zAsq=np.zeros((NFreqBand,npol,zN,zN),dtype=Asq.dtype)
        zAsq[:,:,zN/2-N/2:zN/2+N/2+1,zN/2-N/2:zN/2+N/2+1]=Asq[:,:,:,:]
        T.timeit("1")

        xc=self.PSF.shape[-1]/2
        SubPSF=self.PSF[:,:,xc-N:xc+N+1,xc-N:xc+N+1]
        Conv=np.zeros_like(zAsq)
        T.timeit("2")
        for ich in range(NFreqBand):
            for ipol in range(npol):
                Conv[ich,ipol]=scipy.signal.fftconvolve(zAsq[ich,ipol], SubPSF[ich,ipol], mode='same')

        T.timeit("3 [%s]"%str(zAsq.shape))
        A=self.PM.SquareArrayToModel(Conv[:,:,zN/2-N/2:zN/2+N/2+1,zN/2-N/2:zN/2+N/2+1],TypeInOut=(OutMode,OutMode))
        T.timeit("4")

        if OutMode=="Data":
            NPixOut=self.NPixListData
        else:
            NPixOut=self.NPixListParms
            

        return A.reshape((NFreqBand,npol,NPixOut))

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

        ConvA=np.zeros((self.NFreqBands,1,OutSize),np.float32)
        for iBand in range(self.NFreqBands):
            AThisBand=A[iBand]
            CF=CM[iBand,0]
            ConvA[iBand,0]=np.dot(CF,AThisBand.reshape((AThisBand.size,1))).reshape((OutSize,))

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

        
