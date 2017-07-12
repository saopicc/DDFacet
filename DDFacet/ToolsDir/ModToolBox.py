'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

import scipy.fftpack
import scipy.ndimage

import ModTaper
import numpy as np

F=scipy.fftpack.fft
iF=scipy.fftpack.ifft
Fs=scipy.fftpack.fftshift
iFs=scipy.fftpack.ifftshift

import pyfftw
import scipy.signal
import numpy
#import pyfftw
import scipy.signal

from DDFacet.Other import MyLogger
log= MyLogger.getLogger("ModToolBox", disable=True)


# def EstimateNpix(Npix,Padding=1):
#     Npix=int(round(Npix))

#     NpixOrig=Npix
#     if Npix%2!=0: Npix+=1
#     Npix=ModToolBox.GiveClosestFastSize(Npix)
#     NpixOpt=Npix


#     Npix*=Padding
#     Npix=int(round(Npix))
#     if Npix%2!=0: Npix+=1
#     Npix=ModToolBox.GiveClosestFastSize(Npix)
#     print ModColor.Str("(NpixOrig, NpixOpt, NpixOptPadded): %i --> %i --> %i"%(NpixOrig,NpixOpt,Npix))
#     return NpixOpt,Npix

def EstimateNpix(Npix,
                 Padding=1.0,
                 min_size_fft=513):
    """ Picks image size from the list of fast FFT sizes.
        To avoid spectral leakage the number of taps in the FFT
        must not be too small.
    """
    Npix=int(round(Npix))
    Odd=True

    NpixOrig=Npix
    #if Npix%2!=0: Npix+=1
    #if Npix%2==0: Npix+=1
    Npix=GiveClosestFastSize(Npix,Odd=Odd)
    NpixOpt=Npix


    Npix *= Padding
    if Npix < min_size_fft:
        Npix = min_size_fft
    Npix=int(round(Npix))
    #if Npix%2!=0: Npix+=1
    #if Npix%2==0: Npix+=1
    Npix=GiveClosestFastSize(Npix,Odd=Odd)
    return NpixOpt,Npix

class FFTW_Convolve():
    def __init__(self,A,B):
        self.a = pyfftw.n_byte_align_empty(A.shape, 8, dtype='complex64')
        self.b = pyfftw.n_byte_align_empty(B.shape, 8, dtype='complex64')
        scipy.signal.signaltools.fftn = pyfftw.interfaces.scipy_fftpack.fftn
        scipy.signal.signaltools.ifftn = pyfftw.interfaces.scipy_fftpack.ifftn
        pyfftw.interfaces.cache.enable()

    def convolve(self,A,B):

        self.a[:] = (A[:]).astype(np.complex64)
        self.b[:] = (B[:]).astype(np.complex64)

        return scipy.signal.fftconvolve(self.a, self.b)

def testConvolve():
    A=numpy.random.randn(1014, 1014)
    B=numpy.random.randn(11,11)
    F=FFTW_Convolve(A,B)
    print>>log, "first"
    C=F.convolve(A,B)
    print>>log, "sec"
    C=F.convolve(A,B)
    print>>log, "done"
    stop


class FFTM2():
    def __init__(self,A):
        #log=MyLogger.getLogger("ModToolBox.FFTM2.__init__")
        pyfftw.interfaces.cache.enable()
        self.ThisTypeName=A.dtype.type.__name__
        self.ThisType=A.dtype
        #print>>log, "[Size, fftwtype]=[%s,%s]"%(str(A.shape),self.ThisType)
        #print>>log, "define"
        self.a = pyfftw.n_byte_align_empty(A.shape, 16, self.ThisType)#'complex128')
        self.b = pyfftw.n_byte_align_empty(A.shape, 16, self.ThisType)#'complex128')
        #print>>log, "define1"
        self.fft_FORWARD = pyfftw.FFTW(self.a, self.b,direction="FFTW_FORWARD",flags=('FFTW_ESTIMATE', ))#,threads=4)
        self.fft_BACKWARD = pyfftw.FFTW(self.a, self.b,direction="FFTW_BACKWARD",flags=('FFTW_ESTIMATE', ))#,threads=4)
        #print>>log, "done"

    def fft(self,A):
        #log=MyLogger.getLogger("ModToolBox.FFTM2.fft")
        self.a[:] = iFs(A.astype(self.ThisType),axes=-1)
        self.fft_FORWARD()
        out=np.zeros(A.shape,dtype=self.ThisType)
        out[:]=self.b[:]
        out=Fs(out,axes=-1)/A.shape[-1]
        return out

    def ifft(self,A):
        #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
        self.a[:] = iFs(A.astype(self.ThisType),axes=-1)
        self.fft_BACKWARD()
        out=np.zeros(A.shape,dtype=self.ThisType)
        out[:]=self.b[:]
        out=Fs(out,axes=-1)*A.shape[-1]
        return out





class FFTM2D():
    def __init__(self,A):
        #log=MyLogger.getLogger("ModToolBox.FFTM2.__init__")
        pyfftw.interfaces.cache.enable()
        self.ThisTypeName=A.dtype.type.__name__
        self.ThisType=A.dtype
        #print>>log, "[Size, fftwtype]=[%s,%s]"%(str(A.shape),self.ThisType)
        #print>>log, "define"
        self.a = pyfftw.n_byte_align_empty(A.shape, 16, self.ThisType)#'complex128')
        self.b = pyfftw.n_byte_align_empty(A.shape, 16, self.ThisType)#'complex128')
        #print>>log, "define1"
        self.fft_FORWARD = pyfftw.FFTW(self.a, self.b,axes=(-1,-2),direction="FFTW_FORWARD",flags=('FFTW_ESTIMATE', ))#,threads=4)
        self.fft_BACKWARD = pyfftw.FFTW(self.a, self.b,axes=(-1,-2),direction="FFTW_BACKWARD",flags=('FFTW_ESTIMATE', ))#,threads=4)

        # fft = pyfftw.builders.fft2(a, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=multiprocessing.cpu_count())
        # b = fft()

        #print>>log, "done"
        
    def fft(self,A):
        #log=MyLogger.getLogger("ModToolBox.FFTM2.fft")
        axes=(-1,-2)
        self.a[:,:] = iFs(A.astype(self.ThisType),axes=axes)
        self.fft_FORWARD()
        out=np.zeros(A.shape,dtype=self.ThisType)
        out[:,:]=self.b[:,:]
        out=Fs(out,axes=axes)/(A.shape[-1]*A.shape[-2])
        return out

    def ifft(self,A):
        axes=(-1,-2)
        #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
        self.a[:,:] = iFs(A.astype(self.ThisType),axes=axes)
        self.fft_BACKWARD()
        out=np.zeros(A.shape,dtype=self.ThisType)
        out[:,:]=self.b[:,:]
        out=Fs(out,axes=axes)*(A.shape[-1]*A.shape[-2])
        return out

def testFFTW2D():
    n=1024
    A=np.zeros((1,1,n,n),np.complex128)
    
    A[0,0,n/2+1,n/2+1]=1+0.5*1j
    #A[n/2,n/2]=1#+0.5*1j
    #A=np.random.randn(6,6)+1j*np.random.randn(6,6)

    import ModFFTW
    FM1=ModFFTW.FFTW(A)
    FM2=ModFFTW.FFTWnp(A)
    FM0=FFTM2D(A)
    
    import ClassTimeIt
    T=ClassTimeIt.ClassTimeIt()
    ntest=1
    for i in range(ntest):
        f0=FM0.fft(A)
        if0=FM0.ifft(A)
    T.timeit("old")

    for i in range(ntest):
        f1=FM1.fft(A)
        if1=FM1.ifft(A)
    T.timeit("new")

    for i in range(ntest):
        f2=FM2.fft(A)
        if2=FM2.ifft(A)
    T.timeit("newnp")
    print np.std(f0-f1)
    print np.std(if0-if1)
    print np.std(f0-f2)
    print np.std(if0-if2)
    stop

    # f0=FM.fft(A)
    # f1=FM.ifft(f0)
    f0=FM.fft(A)
    f1=FM.ifft(f0)

    pylab.clf()
    pylab.subplot(2,2,1)
    pylab.imshow(A.T.real,interpolation="nearest")
    pylab.imshow(A.T.imag ,interpolation="nearest")
    pylab.colorbar()
    pylab.subplot(2,2,2)
    pylab.imshow(f0.T.real,interpolation="nearest")
    pylab.imshow(f0.T.imag,interpolation="nearest")
    pylab.colorbar()
    pylab.subplot(2,2,3)
    pylab.imshow(f1.T.real,interpolation="nearest")
    pylab.imshow(f1.T.imag,interpolation="nearest")
    pylab.colorbar()
    pylab.draw()
    pylab.show(False)



def testFFTW():
    A=np.zeros((6,6),np.complex128)
    #A[:,1]=1+1j
    A=np.random.randn(6,6)+1j*np.random.randn(6,6)


    FM=FFTM2(A)
    f0=FM.fft(A)
    f1=FM.ifft(f0)

    pylab.clf()
    pylab.subplot(2,2,1)
    pylab.plot(A.T)
    pylab.subplot(2,2,2)
    pylab.plot(f0.T)
    pylab.subplot(2,2,3)
    pylab.plot(f1.T)
    pylab.draw()
    pylab.show()


def GiveFFTFastSizes(Odd=True,NLim=100000):
    """
    Computes list of optimal FFT sizes. From http://www.fftw.org/doc/Real_002ddata-DFTs.html: 
      "FFTW is best at handling sizes of the form 2^a.3^b.5^c.7^d.11^e.13^f,
       where e+f is either 0 or 1, and the other exponents are arbitrary."
    Returns array of such integer numbers, up to NLim.
    If Odd=True, this does not include factors of 2.
    """
    sizes = np.array([1])
    for base, powers in [
             (2,[0] if Odd else xrange(1,20)),
             (3,xrange(15)), (5,xrange(15)), (7,xrange(15)) ]:
        sizes = (sizes[np.newaxis,:] * base**np.array(powers)[:,np.newaxis]).ravel()
    sizes = sizes[np.newaxis,:] * np.array([1,11,13])[:,np.newaxis]

    # no need to take set(), since sizes are unique by construction (from prime factors...)
    return np.array(sorted(sizes[(sizes<NLim)&(sizes>64)]))    
    # return np.array(sorted(set(sizes[(sizes<NLim)&(sizes>64)])))    

FFTOddSizes  = GiveFFTFastSizes(True,200000)
FFTEvenSizes = GiveFFTFastSizes(False,200000)

def GiveClosestFastSize(n,Odd=True):
    #ind=np.argmin(np.abs(n-FFTOddSizes))
    if Odd:
        ind=np.argmin(np.abs(n-FFTOddSizes))
        return FFTOddSizes[ind]
    else:
        ind=np.argmin(np.abs(n-FFTEvenSizes))
        return FFTEvenSizes[ind]


def GiveFFTFreq(A,dt):
    return Fs(scipy.fftpack.fftfreq(A.shape[-1], d=dt))

    


def ZeroPad(A,outshape=1001):
    nx=A.shape[0]
#    B=np.zeros((nx*zp,nx*zp),dtype=A.dtype)
    B=np.zeros((outshape,),dtype=A.dtype)
    if (outshape%2)==0:
        off=(B.shape[0]-A.shape[0])/2+1
    else:
        off=(B.shape[0]-A.shape[0])/2#+1
    B[off:off+nx]=A
    return B   
 
def MakeSphe(Support,NpixIm,factorSup=1):
    #x,y,CF=Gaussian.Gaussian(3,Support,1)
    CF=ModTaper.Sphe1D(Support,factor=factorSup)
    CF=np.complex128(CF)#np.array(np.complex128(CF),order="F")
    zCF=ZeroPad(CF,CF.shape[0])

    fCF=fft(CF)
    fzCF=fft(zCF)
    zfCF=ZeroPad(fCF,NpixIm)
    ifzfCF=ifft(zfCF)



    # pylab.figure(3)
    # pylab.clf()
    # pylab.subplot(2,3,1)
    # pylab.plot(CF)
    # pylab.title("CF")

    # pylab.subplot(2,3,2)
    # pylab.plot(fCF.real)
    # pylab.plot(fCF.imag)
    # pylab.title("fCF")

    # pylab.subplot(2,3,3)
    # pylab.plot(zCF.real)
    # pylab.plot(zCF.imag)
    # pylab.title("zCF")

    # pylab.subplot(2,3,4)
    # pylab.plot(fzCF.real)
    # pylab.plot(fzCF.imag)
    # pylab.title("fzCF")

    # pylab.subplot(2,3,5)
    # pylab.plot(zfCF.real)
    # pylab.plot(zfCF.imag)
    # pylab.title("zfCF")

    # pylab.subplot(2,3,6)
    # pylab.plot(ifzfCF.real)
    # pylab.plot(ifzfCF.imag)
    # pylab.plot(ModTaper.Sphe1D(ifzfCF.shape[0]))

    # pylab.title("ifzfCF")

    # pylab.draw()
    # pylab.show()#False)
    # pylab.figure(1)


    return CF, fCF, ifzfCF

    

def fft(A):
    b=Fs(F(iFs(A.astype(complex))))
    FA= b/A.shape[0]
    return FA

def ifft(A):
    #FFA= iFs( iF( Fs( A.astype(complex) ) )) *A.shape[0]
    FFA= Fs( iF( iFs( A.astype(complex) ))) *A.shape[0]
    return FFA

def ToOdd(a):
    if (a%2!=0):
        a+=1
    return a

def ToEven(a):
    if (a%2==0):
        a+=1
    return a
