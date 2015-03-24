import numpy as np
import Gaussian
import scipy.fftpack
#import pylab
from pyrap.images import image
import scipy.ndimage
import ModTaper

F=scipy.fftpack.fft
iF=scipy.fftpack.ifft
Fs=scipy.fftpack.fftshift
iFs=scipy.fftpack.ifftshift

import pyfftw
import scipy.signal
import numpy
from timeit import Timer
#import pyfftw
import scipy.signal

import MyLogger
log=MyLogger.getLogger("ModToolBox",disable=True)

import ModColor

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

def EstimateNpix(Npix,Padding=1):
    Npix=int(round(Npix))
    Odd=False

    NpixOrig=Npix
    #if Npix%2!=0: Npix+=1
    #if Npix%2==0: Npix+=1
    Npix=GiveClosestFastSize(Npix,Odd=Odd)
    NpixOpt=Npix
    
    
    Npix*=Padding
    Npix=int(round(Npix))
    #if Npix%2!=0: Npix+=1
    #if Npix%2==0: Npix+=1
    Npix=GiveClosestFastSize(Npix,Odd=Odd)
    #print>>log, ModColor.Str("With padding=%f: (NpixOrig, NpixOpt, NpixOptPadded): %i --> %i --> %i"%(Padding,NpixOrig,NpixOpt,Npix))
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


def GiveFFTFastSizes(Odd=True,NLim=20000):

    
    lout=[]
    for i in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:#range(13):
        for j in range(15):
            for k in range(10):
                for l in range(10):
                    for m in range(5):
                        for n in range(2):
                            for nn in range(2):
                                s=(2**i)*(3**j)*(5**k)*(7**l)*(9**m)*(11**n)*(13**nn)
                                if s>NLim: continue
                                if Odd:
                                    if s%2==0:
                                        lout.append(s)
                                else:
                                    if s%2!=0:
                                        lout.append(s)


    lout=np.array(sorted(list(set(lout))))
    
    lout=lout[(lout<NLim)&(lout>64)]
    
    return lout

# FFTOddSizes=GiveFFTFastSizes(Odd=True)
# FFTEvenSizes=GiveFFTFastSizes(Odd=False)

# FFTOddSizes=FFTOddSizes.tolist()
# FFTOddSizes=[str(i) for i in FFTOddSizes]

# FFTEvenSizes=FFTEvenSizes.tolist()
# FFTEvenSizes=[str(i) for i in FFTEvenSizes]

# print ','.join(FFTOddSizes)
# print ','.join(FFTEvenSizes)

FFTOddSizes=np.array([72,80,84,88,96,100,104,108,112,120,128,132,140,144,156,160,168,176,180,192,196,200,208,216,220,224,240,252,256,260,264,280,288,300,308,312,320,324,336,352,360,364,384,392,396,400,416,420,432,440,448,468,480,500,504,512,520,528,540,560,572,576,588,600,616,624,640,648,660,672,700,704,720,728,756,768,780,784,792,800,832,840,864,880,896,900,924,936,960,972,980,1000,1008,1024,1040,1056,1080,1092,1100,1120,1144,1152,1176,1188,1200,1232,1248,1260,1280,1296,1300,1320,1344,1372,1400,1404,1408,1440,1456,1500,1512,1536,1540,1560,1568,1584,1600,1620,1664,1680,1716,1728,1760,1764,1792,1800,1820,1848,1872,1920,1944,1960,1980,2000,2016,2048,2080,2100,2112,2156,2160,2184,2200,2240,2268,2288,2304,2340,2352,2376,2400,2464,2496,2500,2520,2548,2560,2592,2600,2640,2688,2700,2744,2772,2800,2808,2816,2860,2880,2912,2916,2940,3000,3024,3072,3080,3120,3136,3168,3200,3240,3276,3300,3328,3360,3432,3456,3500,3520,3528,3564,3584,3600,3640,3696,3744,3780,3840,3888,3900,3920,3960,4000,4004,4032,4096,4116,4160,4200,4212,4224,4312,4320,4368,4400,4480,4500,4536,4576,4608,4620,4680,4704,4752,4800,4860,4900,4928,4992,5000,5040,5096,5120,5148,5184,5200,5280,5292,5376,5400,5460,5488,5500,5544,5600,5616,5632,5720,5760,5824,5832,5880,5940,6000,6048,6144,6160,6240,6272,6300,6336,6400,6468,6480,6500,6552,6600,6656,6720,6804,6860,6864,6912,7000,7020,7040,7056,7128,7168,7200,7280,7392,7488,7500,7560,7644,7680,7700,7776,7800,7840,7920,8000,8008,8064,8100,8232,8316,8320,8400,8424,8448,8580,8624,8640,8736,8748,8800,8820,8960,9000,9072,9100,9152,9216,9240,9360,9408,9504,9600,9604,9720,9800,9828,9856,9900,9984,10000,10080,10192,10240,10296,10368,10400,10500,10560,10584,10692,10752,10780,10800,10920,10976,11000,11088,11200,11232,11264,11340,11440,11520,11648,11664,11700,11760,11880,12000,12012,12096,12288,12320,12348,12480,12500,12544,12600,12636,12672,12740,12800,12936,12960,13000,13104,13200,13312,13440,13500,13608,13720,13728,13824,13860,14000,14040,14080,14112,14256,14300,14336,14400,14560,14580,14700,14784,14976,15000,15092,15120,15288,15360,15400,15444,15552,15600,15680,15840,15876,16000,16016,16128,16200,16380,16464,16500,16632,16640,16800,16848,16896,17160,17248,17280,17472,17496,17500,17600,17640,17820,17836,17920,18000,18144,18200,18304,18432,18480,18720,18816,18900,19008,19200,19208,19404,19440,19500,19600,19656,19712,19800,19968])

FFTEvenSizes=np.array([65,75,77,81,91,99,105,117,125,135,143,147,165,175,189,195,225,231,243,245,273,275,297,315,325,343,351,375,385,405,429,441,455,495,525,539,567,585,625,637,675,693,715,729,735,819,825,875,891,945,975,1001,1029,1053,1125,1155,1215,1225,1287,1323,1365,1375,1485,1575,1617,1625,1701,1715,1755,1875,1911,1925,2025,2079,2145,2187,2205,2275,2401,2457,2475,2625,2673,2695,2835,2925,3003,3087,3125,3159,3185,3375,3465,3575,3645,3675,3773,3861,3969,4095,4125,4375,4455,4459,4725,4851,4875,5005,5103,5145,5265,5625,5733,5775,6075,6125,6237,6435,6561,6615,6825,6875,7007,7203,7371,7425,7875,8019,8085,8125,8505,8575,8775,9009,9261,9375,9477,9555,9625,10125,10395,10725,10935,11025,11319,11375,11583,11907,12005,12285,12375,13125,13365,13377,13475,14175,14553,14625,15015,15309,15435,15625,15795,15925,16807,16875,17199,17325,17875,18225,18375,18711,18865,19305,19683,19845])


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
