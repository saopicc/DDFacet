import numpy as np
import pyfftw
import multiprocessing

import scipy

#Fs=pyfftw.interfaces.numpy_fft.fftshift
#iFs=pyfftw.interfaces.numpy_fft.ifftshift


Fs=scipy.fftpack.fftshift
iFs=scipy.fftpack.ifftshift
 
def test():
    size=20
    dtype=np.complex128
    test_array = np.zeros( (size,size), dtype=dtype)
    
    test_array[11,11]=1
    #test_array.fill(1)
    #test_array[size*3/8:size*5/8, size*3/8:size*5/8] = 1+1j # square aperture oversampling 2...
    A=test_array
    F=FFTWnp(A)
    
    f_A=F.fft(A)
    if_f_A=F.ifft(f_A)

    import pylab
    pylab.clf()
    lA=[A,f_A,if_f_A]
    iplot=0
    for iA in lA:
        pylab.subplot(3,2,iplot+1)
        pylab.imshow(iA.real,interpolation="nearest")
        pylab.colorbar()
        pylab.subplot(3,2,iplot+2)
        pylab.imshow(iA.imag,interpolation="nearest")
        pylab.colorbar()
        iplot+=2
    pylab.draw()
    pylab.show(False)

def test2():
    l=[]
    size=2048
    dtype=np.complex128
    test_array = np.zeros( (size,size), dtype=dtype)
    test_array[size*3/8:size*5/8, size*3/8:size*5/8] = 1+1j # square aperture oversampling 2...
    A=test_array
    for i in range(5):
        print i
        l.append(FFTW(A))





class FFTW():
    def __init__(self, A, ncores = 1):
        dtype=A.dtype
        self.A = pyfftw.n_byte_align_empty( A.shape, 16, dtype=dtype)
 
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30)
        self.ncores=ncores
        #print "plan"
        T=ClassTimeIt.ClassTimeIt("ModFFTW")
        T.disable()

        self.A = pyfftw.interfaces.numpy_fft.fft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE',  threads=self.ncores)
        T.timeit("planF")
        self.A = pyfftw.interfaces.numpy_fft.ifft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE',  threads=self.ncores)
        T.timeit("planB")
        #print "done"
        self.ThisType=dtype

    def fft(self,A):
        axes=(-1,-2)

        T=ClassTimeIt.ClassTimeIt("ModFFTW")
        T.disable()
        self.A[:,:] = iFs(A.astype(self.ThisType),axes=axes)
        T.timeit("shift and copy")
        #print "do fft"
        self.A = pyfftw.interfaces.numpy_fft.fft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE', threads=self.ncores)
        T.timeit("fft")
        #print "done"
        out=Fs(self.A,axes=axes)/(A.shape[-1]*A.shape[-2])
        T.timeit("shift")
        return out
 

    def ifft(self,A):
        axes=(-1,-2)
        #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
        self.A[:,:] = iFs(A.astype(self.ThisType),axes=axes)

        #print "do fft"
        self.A = pyfftw.interfaces.numpy_fft.ifft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE', threads=self.ncores)
        out=Fs(self.A,axes=axes)*(A.shape[-1]*A.shape[-2])
        return out

def GiveFFTW_aligned(shape, dtype):
    return pyfftw.n_byte_align_empty( shape[-2::], 16, dtype=dtype)

#import NpShared

class FFTW_2Donly():
    def __init__(self, shape, dtype, ncores = 1, FromSharedId=None):
        if FromSharedId==None:
            self.A = pyfftw.n_byte_align_empty( shape[-2::], 16, dtype=dtype)
        else:
            self.A = NpShared.GiveArray(FromSharedId)
 
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(3000)
        self.ncores=ncores
        #print "plan"
        T=ClassTimeIt.ClassTimeIt("ModFFTW")
        T.disable()

        self.A = pyfftw.interfaces.numpy_fft.fft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE',  threads=self.ncores)
        T.timeit("planF")
        self.A = pyfftw.interfaces.numpy_fft.ifft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE',  threads=self.ncores)
        T.timeit("planB")
        #print "done"
        self.ThisType=dtype

    def fft(self,Ain):
        axes=(-1,-2)

        T=ClassTimeIt.ClassTimeIt("ModFFTW")
        T.disable()

        sin=Ain.shape
        if len(Ain.shape)==2:
            s=(1,1,Ain.shape[0],Ain.shape[1])
            A=Ain.reshape(s)
        else:
            A=Ain

        nch,npol,_,_=A.shape
        for ich in range(nch):
            for ipol in range(npol):
                self.A[:,:] = iFs(A[ich,ipol].astype(self.ThisType),axes=axes)
                T.timeit("shift and copy")
                self.A = pyfftw.interfaces.numpy_fft.fft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE', threads=self.ncores)
                T.timeit("fft")
                A[ich,ipol]=Fs(self.A,axes=axes)/(A.shape[-1]*A.shape[-2])
                T.timeit("shift")

        return A.reshape(sin)
 

    def ifft(self,A):
        axes=(-1,-2)
        #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
        nch,npol,_,_=A.shape
        for ich in range(nch):
            for ipol in range(npol):
                self.A[:,:] = iFs(A[ich,ipol].astype(self.ThisType),axes=axes)
                self.A = pyfftw.interfaces.numpy_fft.ifft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE', threads=self.ncores)
                A[ich,ipol]=Fs(self.A,axes=axes)*(A.shape[-1]*A.shape[-2])

        return A

class FFTW_2Donly_np():
    def __init__(self, shape, dtype, ncores = 1):

        return

    def fft(self,A):
        axes=(-1,-2)

        T=ClassTimeIt.ClassTimeIt("ModFFTW")
        T.disable()
        
        nch,npol,n,n=A.shape

        for ich in range(nch):
            for ipol in range(npol):
                B = iFs(A[ich,ipol].astype(A.dtype),axes=axes)
                T.timeit("shift and copy")
                B = np.fft.fft2(B,axes=axes)
                T.timeit("fft")
                A[ich,ipol]=Fs(B,axes=axes)/(A.shape[-1]*A.shape[-2])
                T.timeit("shift")

        return A
 

    def ifft(self,A):
        axes=(-1,-2)
        #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
        nch,npol,_,_=A.shape
        for ich in range(nch):
            for ipol in range(npol):
                B = iFs(A[ich,ipol].astype(A.dtype),axes=axes)
                B = np.fft.ifft2(B,axes=axes)
                A[ich,ipol]=Fs(B,axes=axes)*(A.shape[-1]*A.shape[-2])

        return A

def GiveGauss(Npix,CellSizeRad=None,GaussPars=(0.,0.,0.)):
    uvscale=Npix*CellSizeRad/2
    SigMaj,SigMin,ang=GaussPars
    U,V=np.mgrid[-uvscale:uvscale:Npix*1j,-uvscale:uvscale:Npix*1j]
    CT=np.cos(ang)
    ST=np.sin(ang)
    C2T=np.cos(2*ang)
    S2T=np.sin(2*ang)
    sx2=SigMaj**2
    sy2=SigMin**2
    a=(CT**2/(2.*sx2))+(ST**2/(2.*sy2))
    b=-(S2T/(4.*sx2))+(S2T/(4.*sy2))
    c=(ST**2/(2.*sx2))+(CT**2/(2.*sy2))
    x,y=U,V
    k=a*x**2+2.*b*x*y+c*y**2
    Gauss=np.exp(-k)
    #Gauss/=np.sum(Gauss)
    return Gauss

def ConvolveGaussian(Ain0,CellSizeRad=None,GaussPars=[(0.,0.,0.)]):

    nch,npol,_,_=Ain0.shape
    Aout=np.zeros_like(Ain0)

    for ch in range(nch):
        Ain=Ain0[ch]
        ThisGaussPars=GaussPars[ch]
        PSF=GiveGauss(Ain.shape[-1],CellSizeRad,ThisGaussPars)
        FFTM=FFTWnpNonorm(PSF)
        fPSF=np.abs(FFTM.fft(PSF))
        for pol in range(npol):
            A=Ain[pol]
            FFTM=FFTWnpNonorm(A)
            fA=FFTM.fft(A)
            nfA=fA*fPSF#Gauss
            if_fA=FFTM.ifft(nfA)
            Aout[ch,pol,:,:]=if_fA.real

    return Aout

    # import pylab
    # pylab.clf()
    # pylab.subplot(2,2,1)
    # pylab.imshow(np.real(A),interpolation="nearest")
    # pylab.title("Model Image")
    # pylab.colorbar()
    # pylab.subplot(2,2,2)
    # pylab.imshow(np.real(PSF),interpolation="nearest")
    # pylab.title("PSF")
    # pylab.colorbar()
    # pylab.subplot(2,2,3)
    # pylab.imshow(np.real(fPSF),interpolation="nearest")
    # pylab.title("Gaussian")
    # pylab.colorbar()
    # pylab.subplot(2,2,4)
    # pylab.imshow(np.real(if_fA),interpolation="nearest")
    # pylab.title("Convolved Model image")
    # pylab.colorbar()
    # pylab.draw()
    # pylab.show(False)
    # pylab.pause(0.1)

    # print np.sum(if_fA)

    # return if_fA

def testConvolveGaussian():
    A=np.zeros((20,20),np.complex64)
    A[10,10]=1
    SigMaj=2#(20/3600.)*np.pi/180
    SigMin=1#(5/3600.)*np.pi/180
    ang=30.*np.pi/180
    GaussPars=SigMaj,SigMin,ang
    CellSizeRad=1#(5./3600)*np.pi/180
    CA=ConvolveGaussian(A,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
    


class FFTWnp():
    def __init__(self, A, ncores = 1):
        dtype=A.dtype
        self.ThisType=dtype



    def fft(self,A):
        axes=(-2,-1)

        T=ClassTimeIt.ClassTimeIt("ModFFTW")
        T.disable()

        A = iFs(A.astype(self.ThisType),axes=axes)
        T.timeit("shift and copy")
        #print "do fft"
        A = np.fft.fft2(A,axes=axes)
        T.timeit("fft")
        #print "done"
        A=Fs(A,axes=axes)/(A.shape[-1]*A.shape[-2])
        T.timeit("shift")
        return A
 

    def ifft(self,A):
        axes=(-2,-1)
        #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
        A = iFs(A.astype(self.ThisType),axes=axes)

        #print "do fft"
        A = np.fft.ifft2(A,axes=axes)
        out=Fs(A,axes=axes)*(A.shape[-1]*A.shape[-2])
        return out

 
class FFTWnpNonorm():
    def __init__(self, A, ncores = 1):
        #dtype=A.dtype
        self.ThisType=np.complex64



    def fft(self,A):
        axes=(-2,-1)

        #T=ClassTimeIt.ClassTimeIt("ModFFTW")
        #T.disable()

        A = iFs(A.astype(self.ThisType),axes=axes)

        #print "do fft"
        A = np.fft.fft2(A,axes=axes)
        #print "done"
        A=Fs(A,axes=axes)#/(A.shape[-1]*A.shape[-2])

        return A
 

    def ifft(self,A):
        axes=(-2,-1)
        #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
        A = iFs(A.astype(self.ThisType),axes=axes)

        #print "do fft"
        A = np.fft.ifft2(A,axes=axes)
        out=Fs(A,axes=axes)#*(A.shape[-1]*A.shape[-2])
        return out

 

