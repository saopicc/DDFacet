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

import scipy

import numpy as np
import pyfftw
from DDFacet.Array import NpShared
from DDFacet.Other import ClassTimeIt
import psutil
import numexpr
from DDFacet.Other.AsyncProcessPool import APP
from DDFacet.Array import shared_dict
from DDFacet.Other import MyLogger
import ModToolBox
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
from DDFacet.ToolsDir import Gaussian

log=MyLogger.getLogger("ModFFTW")

#Fs=pyfftw.interfaces.numpy_fft.fftshift
#iFs=pyfftw.interfaces.numpy_fft.ifftshift


Fs=scipy.fftpack.fftshift
iFs=scipy.fftpack.ifftshift

NCPU_global = 0#psutil.cpu_count()

#def test():
#    size=20
#    dtype=np.complex128
#    test_array = np.zeros( (size,size), dtype=dtype)
#
#    test_array[11,11]=1
#    #test_array.fill(1)
#    #test_array[size*3/8:size*5/8, size*3/8:size*5/8] = 1+1j # square aperture oversampling 2...
#    A=test_array
#    F=FFTWnp(A)
#
#    f_A=F.fft(A)
#    if_f_A=F.ifft(f_A)
#
#    import pylab
#    pylab.clf()
#    lA=[A,f_A,if_f_A]
#    iplot=0
#    for iA in lA:
#        pylab.subplot(3,2,iplot+1)
#        pylab.imshow(iA.real,interpolation="nearest")
#        pylab.colorbar()
#        pylab.subplot(3,2,iplot+2)
#        pylab.imshow(iA.imag,interpolation="nearest")
#        pylab.colorbar()
#        iplot+=2
#    pylab.draw()
#    pylab.show(False)

#def test2():
#    l=[]
#    size=2048
#    dtype=np.complex128
#    test_array = np.zeros( (size,size), dtype=dtype)
#    test_array[size*3/8:size*5/8, size*3/8:size*5/8] = 1+1j # square aperture oversampling 2...
#    A=test_array
#    for i in range(5):
#        print i
#        l.append(FFTW(A))

#class FFTW():
#   def __init__(self, A, ncores = 1):
#       Raise("deprecated: this doesn't work for small ffts", DeprecationWarning)

#       dtype=A.dtype
#       self.A = pyfftw.n_byte_align_empty( A.shape, 16, dtype=dtype)

#       pyfftw.interfaces.cache.enable()
#       pyfftw.interfaces.cache.set_keepalive_time(30)
#       self.ncores=ncores
#       #print "plan"
#       T= ClassTimeIt.ClassTimeIt("ModFFTW")
#       T.disable()

#       self.A = pyfftw.interfaces.numpy_fft.fft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE',  threads=self.ncores)
#       T.timeit("planF")
#       self.A = pyfftw.interfaces.numpy_fft.ifft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE',  threads=self.ncores)
#       T.timeit("planB")
#       #print "done"
#       self.ThisType=dtype

#   def fft(self, A, norm=True):
#       axes=(-1,-2)

#       T= ClassTimeIt.ClassTimeIt("ModFFTW")
#       T.disable()
#       self.A[:,:] = iFs(A.astype(self.ThisType),axes=axes)
#       T.timeit("shift and copy")
#       #print "do fft"
#       self.A = pyfftw.interfaces.numpy_fft.fft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE', threads=self.ncores)
#       T.timeit("fft")
#       #print "done"
#       if norm:
#           out=Fs(self.A,axes=axes)/(A.shape[-1]*A.shape[-2])
#       T.timeit("shift")
#       return out
#

#   def ifft(self,A):
#       axes=(-1,-2)
#       #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
#       self.A[:,:] = iFs(A.astype(self.ThisType),axes=axes)

#       #print "do fft"
#       self.A = out = pyfftw.interfaces.numpy_fft.ifft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE', threads=self.ncores)
#       if norm:
#           out=Fs(self.A,axes=axes)*(A.shape[-1]*A.shape[-2])
#       return out

def GiveFFTW_aligned(shape, dtype):
    return pyfftw.n_byte_align_empty( shape[-2::], 16, dtype=dtype)

# FFTW version of the FFT engine
class FFTW_2Donly():
    def __init__(self, shape, dtype, norm=True, ncores=1, FromSharedId=None):
        # if FromSharedId is None:
        #     self.A = pyfftw.n_byte_align_empty( shape[-2::], 16, dtype=dtype)
        # else:
        #     self.A = NpShared.GiveArray(FromSharedId)

        #pyfftw.interfaces.cache.enable()
        #pyfftw.interfaces.cache.set_keepalive_time(3000)
        self.ncores=ncores or NCPU_global
        #print "plan"
        T= ClassTimeIt.ClassTimeIt("ModFFTW")
        T.disable()

        #self.A = pyfftw.interfaces.numpy_fft.fft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE',  threads=self.ncores)
        T.timeit("planF")
        #self.A = pyfftw.interfaces.numpy_fft.ifft2(self.A, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE',  threads=self.ncores)
        T.timeit("planB")
        #print "done"
        self.ThisType=dtype
        self.norm = norm

    def fft(self, Ain):
        axes=(-1,-2)

        T= ClassTimeIt.ClassTimeIt("ModFFTW")
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
                A_2D = iFs(A[ich,ipol].astype(self.ThisType),axes=axes)
                T.timeit("shift and copy")
                A_2D[...] = pyfftw.interfaces.numpy_fft.fft2(A_2D, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE', threads=self.ncores)
                T.timeit("fft")
                A[ich,ipol]=Fs(A_2D,axes=axes)
                T.timeit("shift")
        if self.norm:
            A /= (A.shape[-1] * A.shape[-2])

        return A.reshape(sin)

    def ifft(self, A, norm=True):
        axes=(-1,-2)
        sin=A.shape
        if len(A.shape)==2:
            s=(1,1,A.shape[0],A.shape[1])
            A=A.reshape(s)
        #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
        nch,npol,_, _ = A.shape
        for ich in range(nch):
            for ipol in range(npol):
                A_2D = iFs(A[ich,ipol].astype(self.ThisType),axes=axes)
                A_2D[...] = pyfftw.interfaces.numpy_fft.ifft2(A_2D, axes=(-1,-2),overwrite_input=True, planner_effort='FFTW_MEASURE', threads=self.ncores)
                A[ich,ipol]=Fs(A_2D,axes=axes)
        if self.norm:
            A *= (A.shape[-1] * A.shape[-2])
        return A.reshape(sin)

# LAPACK (or ATLAS???) version of the FFT engine
class FFTW_2Donly_np():
    def __init__(self, shape=None, dtype=None, ncores = 1):

        return

    def fft(self,A,ChanList=None):
        axes=(-1,-2)

        T= ClassTimeIt.ClassTimeIt("ModFFTW")
        T.disable()

        nch,npol,n,n=A.shape

        if ChanList is not None:
            CSel=ChanList
        else:
            CSel=range(nch)


        for ich in CSel:
            for ipol in range(npol):
                B = iFs(A[ich,ipol].astype(A.dtype),axes=axes)
                T.timeit("shift and copy")
                B = np.fft.fft2(B,axes=axes)
                T.timeit("fft")
                A[ich,ipol]=Fs(B,axes=axes)/(A.shape[-1]*A.shape[-2])
                T.timeit("shift")

        return A

    def ifft(self,A,ChanList=None):
        axes=(-1,-2)
        #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
        nch,npol,_,_=A.shape

        if ChanList is not None:
            CSel=ChanList
        else:
            CSel=range(nch)

        for ich in CSel:
            for ipol in range(npol):
                B = iFs(A[ich,ipol].astype(A.dtype),axes=axes)
                B = np.fft.ifft2(B,axes=axes)
                A[ich,ipol]=Fs(B,axes=axes)*(A.shape[-1]*A.shape[-2])

        return A


_give_gauss_grid_key = None,None
_give_gauss_grid_cache = None,None

def GiveGauss(Npix,CellSizeRad=None,GaussPars=(0.,0.,0.),dtype=np.float32,parallel=True):
    uvscale=Npix*CellSizeRad/2
    SigMaj,SigMin,ang=GaussPars
    ang = 2*np.pi - ang #need counter-clockwise rotation

    # np.mgrid turns out to be *the* major CPU consumer here when GiveGauss() is called repeatedly.
    # Hence, cache and reuse it
    global _give_gauss_grid_key, _give_gauss_grid_cache
    if (uvscale, Npix) == _give_gauss_grid_key:
        U, V = _give_gauss_grid_cache
    else:
        U, V = _give_gauss_grid_cache = np.mgrid[-uvscale:uvscale:Npix*1j,-uvscale:uvscale:Npix*1j]
        _give_gauss_grid_key = uvscale, Npix

    CT=np.cos(ang)
    ST=np.sin(ang)
    C2T=np.cos(2*ang)
    S2T=np.sin(2*ang)
    sx2=SigMaj**2
    sy2=SigMin**2
    a=(CT**2/(2.*sx2))+(ST**2/(2.*sy2))
    b=-(S2T/(4.*sx2))+(S2T/(4.*sy2))
    c=(ST**2/(2.*sx2))+(CT**2/(2.*sy2))
    # print a,b,c
    # print U,V
    # print U.shape,V.shape

    x, y = U, V
    if parallel:
        Gauss = np.zeros((Npix,Npix), dtype)
        numexpr.evaluate("exp(-(a*x**2+2*b*x*y+c*y**2))", out=Gauss, casting="unsafe")
    else:
        Gauss = np.exp(-(a*x**2+2.*b*x*y+c*y**2))
    #print "okx"
    #Gauss/=np.sum(Gauss)
    return Gauss

def ConvolveGaussianScipy(Ain0,Sig=1.,GaussPar=None):
  #warnings.warn("deprecated: this wont work for small ffts...",
  #              DeprecationWarning)
  Npix=int(2*8*Sig)
  if Npix%2==0: Npix+=1
  x0=Npix/2
  x,y=np.mgrid[-x0:x0:Npix*1j,-x0:x0:Npix*1j]
  #in2=np.exp(-(x**2+y**2)/(2.*Sig**2))
  if GaussPar is None:
      GaussPar=(Sig,Sig,0)
  in2=Gaussian.Gaussian2D(x,y,GaussPar=GaussPar)

  nch,npol,_,_=Ain0.shape
  Out=np.zeros_like(Ain0)
  for ch in range(nch):
      in1=Ain0[ch,0]
      Out[ch,0,:,:]=scipy.signal.fftconvolve(in1, in2, mode='same').real
  return Out,in2

def ConvolveGaussianWrapper(Ain0,Sig=1.0,GaussPar=None,Out=None):
    # a drop-in replacement for ConvolveGaussianScipy which uses
    # _convolveSingleGaussianNP . The factor sqrt(2) here in 'pixel
    # size' is a fudge to make the two routines agree: the
    # Gaussian/Gaussian2D code appears to be missing the factor 2 on
    # the denominator of the Gaussian function it computes
    nch,npol,_,_=Ain0.shape
    if Out is None:
        Out = np.zeros_like(Ain0)
    else:
        assert Out.shape == Ain0.shape and Out.dtype == Ain0.dtype, "Out argument should have same shape/type as input"
    dict={'in':Ain0,'out':Out}
    if GaussPar is None:
        GaussPar=(Sig,Sig,0)
    for ch in range(nch):
        # replacing NP->FFTW.  See discussion in https://github.com/cyriltasse/DDFacet/issues/463
        Aout,PSF=_convolveSingleGaussianFFTW(dict,'in','out',ch,np.sqrt(2),GaussPar,return_gaussian=True)
        ### this should not be necessary: _convolveSingleGaussianFFTW() already stores to dict['out'][ch], which is Out[ch]
        # Out[ch,:,:,:]=Aout
    return Out,PSF

def ConvolveGaussianSimpleWrapper(Ain0, CellSizeRad=1.0, Sig=1.0, GaussPars=None):
    nch,npol,_,_=Ain0.shape
    Out=np.zeros_like(Ain0)
    dict={'in':Ain0,'out':Out}
    if GaussPars is None:
        GaussPars=(Sig,Sig,0)
    for ch in range(nch):
        Aout=_convolveSingleGaussianNP(dict,'in','out',ch,CellSizeRad,GaussPars)
        Out[ch,:,:,:]=Aout
    return Out
    

def learnFFTWWisdom(npix,dtype=np.float32):
    """Learns FFTW wisdom for real 2D FFT of npix x npix images"""
    print>>log, "  Computing fftw wisdom FFTs for shape [%i x %i] and dtype %s" % (npix,npix,dtype.__name__)
    test = np.zeros((npix, npix), dtype)
    if "float" in dtype.__name__:
        a = pyfftw.interfaces.numpy_fft.rfft2(test, overwrite_input=True, threads=1)
        b = pyfftw.interfaces.numpy_fft.irfft2(a, overwrite_input=True, threads=1)
    elif "complex" in dtype.__name__:
        a = pyfftw.interfaces.numpy_fft.fft2(test, overwrite_input=True, threads=1)
        b = pyfftw.interfaces.numpy_fft.ifft2(a, overwrite_input=True, threads=1)

# FFTW-based convolution
def _convolveSingleGaussianFFTW(shareddict,
                                field_in,
                                field_out,
                                ch,
                                CellSizeRad,
                                GaussPars_ch,
                                Normalise = False,
                                nthreads = 1,
                                return_gaussian = False):
    """Convolves a single channel in a cube of nchan, npol, Ny, Nx
       @param shareddict: a dictionary containing an input and output array of size
       [nchans, npols, Ny, Nx]
       @param field_in: index of input field in shareddict
       @param field_out: index of the output field in shareddict (can be the
       same as field_in
       @param ch: index of channel to convolve
       @param CellSizeRad: pixel size in radians of the gaussian in image space
       @param nthreads: number of threads to use in FFTW
       @param Normalize: Normalize the gaussian amplitude
       @param return_gaussian: return the convolving Gaussian as well
    """
    # The FFT needs to be big enough to avoid spectral leakage in the
    # transforms, so we pad both sides of the stack of images with the same
    # number of pixels. This preserves the baseband component correctly:
    # Even - 4 pixels becomes 6 for instance:
    #   |   |   |  x  |    | => |    |    |    |  x  |    |    |
    # Odd - 3 pixels becomes 5 for instance:
    #   |   |  x  |   | => |   |   |  x  |   |   |
    # IFFTShift will shift the central location down to 0 (middle + 1 and
    # middle for even and odd respectively). After FFT the baseband is at
    # 0 as expected. FFTShift can then recentre the FFT. Going back the 
    # IFF is again applied so baseband is at 0, ifft taken and FFTShift
    # brings the central location back to middle + 1 and middle for even and
    # odd respectively. The signal can then safely be unpadded

    T = ClassTimeIt.ClassTimeIt()
    T.disable()
    Ain = shareddict[field_in][ch]
    Aout = shareddict[field_out][ch]
    T.timeit("init %d"%ch)
    npol, npix_y, npix_x = Ain.shape
    assert npix_y == npix_x, "Only supports square grids at the moment"
    pad_edge = max(int(np.ceil((ModToolBox.EstimateNpix(npix_x)[1] - npix_x) /
                               2.0) * 2),0)
    PSF = np.pad(GiveGauss(npix_x, CellSizeRad, GaussPars_ch, parallel=True),
                 ((pad_edge//2,pad_edge//2),(pad_edge//2,pad_edge//2)),
                 mode="constant")

    if Normalise:
        PSF /= np.sum(PSF)
    T.timeit("givegauss %d"%ch)
    fPSF = pyfftw.interfaces.numpy_fft.rfft2(iFs(PSF),
                                             overwrite_input=True,
                                             threads=nthreads)
    fPSF = np.abs(fPSF)
    for pol in range(npol):
        A = iFs(np.pad(Ain[pol],
                       ((pad_edge//2,pad_edge//2),(pad_edge//2,pad_edge//2)),
                       mode="constant"))
        fA = pyfftw.interfaces.numpy_fft.rfft2(A, overwrite_input=True,
                                               threads=nthreads)
        nfA = fA*fPSF
        Aout[pol, :, :] = Fs(
            pyfftw.interfaces.numpy_fft.irfft2(nfA,
                                               s=A.shape,
                                               overwrite_input=True,
                                               threads=nthreads))[pad_edge//2:pad_edge//2+npix_y,
                                                                  pad_edge//2:pad_edge//2+npix_x]
    T.timeit("convolve %d" % ch)

    if return_gaussian:
        return Aout,PSF
    else:
        return Aout

# LAPACK / ATLAS-based convolution
def _convolveSingleGaussianNP(shareddict, field_in, field_out, ch,
                              CellSizeRad, GaussPars_ch,
                              Normalise = False, return_gaussian = False):

    """Convolves a single channel in a cube of nchan, npol, Ny, Nx
       @param shareddict: a dictionary containing an input and output array of size
       [nchans, npols, Ny, Nx]
       @param field_in: index of input field in shareddict
       @param field_out: index of the output field in shareddict (can be the
       same as field_in
       @param ch: index of channel to convolve
       @param CellSizeRad: pixel size in radians of the gaussian in image space
       @param Normalize: Normalize the Gaussian amplitude
       @param return_gaussian: return the convolving Gaussian as well
    """
    # The FFT needs to be big enough to avoid spectral leakage in the
    # transforms, so we pad both sides of the stack of images with the same
    # number of pixels. This preserves the baseband component correctly:
    # Even - 4 pixels becomes 6 for instance:
    #   |   |   |  x  |    | => |    |    |    |  x  |    |    |
    # Odd - 3 pixels becomes 5 for instance:
    #   |   |  x  |   | => |   |   |  x  |   |   |
    # IFFTShift will shift the central location down to 0 (middle + 1 and
    # middle for even and odd respectively). After FFT the baseband is at
    # 0 as expected. FFTShift can then recentre the FFT. Going back the 
    # IFF is again applied so baseband is at 0, ifft taken and FFTShift
    # brings the central location back to middle + 1 and middle for even and
    # odd respectively. The signal can then safely be unpadded
    T = ClassTimeIt.ClassTimeIt()
    T.disable()
    Ain = shareddict[field_in][ch]
    Aout = shareddict[field_out][ch]
    T.timeit("init %d"%ch)
    npol, npix_y, npix_x = Ain.shape
    assert npix_y == npix_x, "Only supports square grids at the moment"
    pad_edge = max(int(np.ceil((ModToolBox.EstimateNpix(npix_x)[1] - npix_x) /
                               2.0) * 2),0)
    PSF = np.pad(GiveGauss(npix_x, CellSizeRad, GaussPars_ch, parallel=True),
                 ((pad_edge//2,pad_edge//2),(pad_edge//2,pad_edge//2)),
                 mode="constant")

    # PSF=np.ones((Ain.shape[-1],Ain.shape[-1]),dtype=np.float32)
    if Normalise:
        PSF /= np.sum(PSF)
    T.timeit("givegauss %d"%ch)
    fPSF = np.fft.rfft2(iFs(PSF))
    fPSF = np.abs(fPSF)
    for pol in range(npol):
        A = iFs(np.pad(Ain[pol],
                       ((pad_edge//2,pad_edge//2),(pad_edge//2,pad_edge//2)),
                       mode="constant"))
        fA = np.fft.rfft2(A)
        nfA = fA*fPSF
        Aout[pol, :, :] = Fs(np.fft.irfft2(nfA,
                                           s=A.shape))[pad_edge//2:npix_y+pad_edge//2,
                                                       pad_edge//2:npix_x+pad_edge//2]

    T.timeit("convolve %d" % ch)
    if return_gaussian:
        return Aout,PSF
    else:
        return Aout

ConvolveGaussian = _convolveSingleGaussianFFTW

def ConvolveGaussianParallel(shareddict, field_in, field_out, CellSizeRad=None,GaussPars=[(0.,0.,0.)],Normalise=False):
    """Convolves images held in a dict, using APP.
    """
    Ain0 = shareddict[field_in]
    nch,npol,_,_=Ain0.shape
    Aout = shareddict[field_out]
    # single channel? Handle serially
    if nch == 1:
        return ConvolveGaussian(shareddict, field_in, field_out, 0, CellSizeRad, GaussPars[0], Normalise)

    jobid = "convolve:%s:%s:" % (field_in, field_out)
    for ch in range(nch):
        APP.runJob(jobid+str(ch),_convolveSingleGaussianFFTW_noret, args=(shareddict.readwrite(), field_in, field_out, ch, CellSizeRad, GaussPars[ch], Normalise))
    APP.awaitJobResults(jobid+"*") #, progress="Convolving")

    return Aout


# wrappers that discard return value for use with APP -- avoids wasteful stuffing of images into result queues
def _convolveSingleGaussianFFTW_noret(*args,**kw):
    _convolveSingleGaussianFFTW(*args,**kw)
    return None

def _convolveSingleGaussianNP_noret(*args,**kw):
    _convolveSingleGaussianNP(*args,**kw)
    return None

APP.registerJobHandlers(_convolveSingleGaussianFFTW_noret, _convolveSingleGaussianNP_noret)

## FFTW version
#def ConvolveGaussianFFTW(Ain0,
#                        CellSizeRad=None,
#                        GaussPars=[(0.,0.,0.)],
#                        Normalise=False,
#                        out=None,
#                        nthreads=1,
#                        min_size_fft=2048):
#   warnings.warn("deprecated", DeprecationWarning)

#   assert Ain0.shape == 4, "Expected stack of images: nch, npol, Ny, Nx"
#   nch,npol,Ny,Nx=Ain0.shape
#   pady = max(Ny, min_size_fft)
#   padx = max(Nx, min_size_fft)
#   Aout = np.zeros_like(Ain0) if out is None else out
#   T = ClassTimeIt.ClassTimeIt()
#   T.disable()
#   T.timeit("init")

#   for ch in range(nch):
#       Ain=Ain0[ch]
#       ThisGaussPars=GaussPars[ch]
#       PSF=GiveGauss(Ain.shape[-1],CellSizeRad,ThisGaussPars)
#       T.timeit("givegauss")
#       if Normalise:
#           PSF/=np.sum(PSF)
#       PSF = np.fft.ifftshift(PSF)
#       fPSF = pyfftw.interfaces.numpy_fft.rfft2(PSF, overwrite_input=True, threads=nthreads)
#       for pol in range(npol):
#           A = np.fft.ifftshift(Ain[pol])
#           fA = pyfftw.interfaces.numpy_fft.rfft2(A, overwrite_input=True, threads=nthreads)
#           nfA = fA*fPSF
#           ifA= pyfftw.interfaces.numpy_fft.irfft2(nfA, s=A.shape, overwrite_input=True, threads=nthreads)
#           Aout[ch, pol, :, :] = np.fft.fftshift(ifA)
#       T.timeit("conv")

#   return Aout

#class ZMachine():
#   #Why??: np.pad...
#   def __init__(self,A):
#       self.N=A.shape[-1]
#       zN=2*self.N+1
#       zN,_=EstimateNpix(float(zN))
#       self.zN=zN

#   def toZeroPad(self,A):
#       zN=self.zN
#       N=self.N
#       zA=np.zeros((zN,zN),dtype=A.dtype)
#       if N%2:
#           zA[zN/2-N/2:zN/2+N/2+1,zN/2-N/2:zN/2+N/2+1]=A[:,:]
#           #nx,ny=A.shape
#           #zA[:nx/2+1,0:ny]=A[:nx/2+1,:]
#           #zA[-nx/2+1:,0:ny]=A[-nx/2+1:,:]
#       else:
#           zA[zN/2-N/2:zN/2+N/2,zN/2-N/2:zN/2+N/2]=A[:,:]
#           
#       # import pylab
#       # pylab.subplot(1,2,1)
#       # pylab.imshow(A.real,interpolation="nearest")
#       # pylab.subplot(1,2,2)
#       # pylab.imshow(zA.real,interpolation="nearest")
#       # pylab.draw()
#       # pylab.show(False)
#       # stop

#       return zA

#   def fromZeroPad(self,zA):
#       zN=self.zN
#       N=self.N
#       A=np.zeros((N,N),dtype=zA.dtype)
#       if N%2:
#           A[:,:]=zA[zN/2-N/2:zN/2+N/2+1,zN/2-N/2:zN/2+N/2+1]
#       else:
#           A[:,:]=zA[zN/2-N/2:zN/2+N/2,zN/2-N/2:zN/2+N/2]

#       return A

## FFTW-based convolution version
#def ConvolveFFTW2D(Ain0,Bin0,CellSizeRad=None,GaussPars=[(0.,0.,0.)],Normalise=False,out=None,ZeroPad=False):
#   warnings.warn("deprecated: this doesn't work for small ffts", DeprecationWarning)

#   if Ain0.shape != Bin0.shape:
#       raise NotImplementedError("Arrays should have the same shapes")

#   if ZeroPad:
#       ZM=ZMachine(Ain0)
#       Ain=ZM.toZeroPad(Ain0)
#       PSF=ZM.toZeroPad(Bin0)
#   else:
#       Ain=Ain0
#       PSF=Bin0

#   #Aout = np.zeros_like(Ain) if out is None else out

#   fft_forward=pyfftw.interfaces.numpy_fft.rfft2
#   fft_bakward=pyfftw.interfaces.numpy_fft.irfft2
#   fft_forward=pyfftw.interfaces.numpy_fft.fft2
#   fft_bakward=pyfftw.interfaces.numpy_fft.ifft2


#   if Normalise:
#       PSF/=np.sum(PSF)
#   T = ClassTimeIt.ClassTimeIt()
#   T.disable()
#   T.timeit("init")

#   #print PSF.shape

#   PSF = np.fft.fftshift(PSF)
#   T.timeit("shoft")
#   #print PSF.shape

#   fPSF = fft_forward(PSF, overwrite_input=True, threads=1)#NCPU_global)
#   T.timeit("fft1")
#   #print fPSF.shape

#   #print Ain.shape
#   A = np.fft.fftshift(Ain)
#   #print A.shape
#   T.timeit("shoft")

#   fA = fft_forward(A, overwrite_input=True, threads=1)#NCPU_global), planner_effort='FFTW_MEASURE'
#   T.timeit("fft2")
#   #print fA.shape

#   nfA = fA*fPSF
#   T.timeit("mult")

#    if ZeroPad:
#        nfA=ZM.toZeroPad(nfA)
#   ifA= fft_bakward(nfA, overwrite_input=True, threads=1)#NCPU_global)
#   T.timeit("ifft")

#   Aout = np.fft.ifftshift(ifA)
#   T.timeit("shift")
#   #print Aout.shape
#   if ZeroPad:
#       Aout=ZM.fromZeroPad(Aout)
#   #print Aout.shape
#   return Aout

## numpy.fft version
#def ConvolveGaussianNPclassic(Ain0,CellSizeRad=None,GaussPars=[(0.,0.,0.)],Normalise=False,out=None):
#   warnings.warn("deprecated: this doesn't work for small ffts", DeprecationWarning)
#   nch,npol,_,_=Ain0.shape
#   Aout = np.zeros_like(Ain0) if out is None else out

#   T = ClassTimeIt.ClassTimeIt()
#   T.disable()
#   T.timeit("init %s"%(Ain0.shape,))

#   for ch in range(nch):
#       Ain=Ain0[ch]
#       ThisGaussPars=GaussPars[ch]
#       PSF=GiveGauss(Ain.shape[-1],CellSizeRad,ThisGaussPars)
#       print PSF
#       T.timeit("givegauss")
#       # PSF=np.ones((Ain.shape[-1],Ain.shape[-1]),dtype=np.float32)
#       if Normalise:
#           PSF/=np.sum(PSF)
#       FFTM = FFTWnpNonorm(PSF)
#       fPSF = FFTM.fft(PSF)
#       fPSF = np.abs(fPSF)
#       for pol in range(npol):
#           A = Ain[pol]
#           FFTM = FFTWnpNonorm(A)
#           fA = FFTM.fft(A)
#           nfA = fA*fPSF#Gauss
#           if_fA = FFTM.ifft(nfA)
#           # if_fA=(nfA)
#           Aout[ch,pol,:,:] = if_fA.real
#       T.timeit("conv")

#   return Aout

#def ConvolveGaussianC(Ain0,CellSizeRad=None,GaussPars=[(0.,0.,0.)],Normalise=False,out=None):
#   warnings.warn("deprecated: this doesn't work for small ffts", DeprecationWarning)
#   nch,npol,_,_=Ain0.shape
#   Aout = np.zeros_like(Ain0) if out is None else out

#   T = ClassTimeIt.ClassTimeIt()
#   T.timeit("init")

#   for ch in range(nch):
#       Ain=Ain0[ch]
#       ThisGaussPars=GaussPars[ch]
#       PSF=GiveGauss(Ain.shape[-1],CellSizeRad,ThisGaussPars)
#       T.timeit("givegauss")
#       # PSF=np.ones((Ain.shape[-1],Ain.shape[-1]),dtype=np.float32)
#       if Normalise:
#           PSF/=np.sum(PSF)
#       PSF = np.fft.ifftshift(PSF)
#       fPSF = np.fft.fft2(PSF)
#       fPSF = np.abs(fPSF)
#       for pol in range(npol):
#           A = Ain[pol]
#           fA = np.fft.fft2(A)
#           nfA = fA*fPSF#Gauss
#           if_fA = np.fft.ifft2(nfA)
#           # if_fA=(nfA)
#           Aout[ch,pol,:,:] = np.fft.fftshift(if_fA.real)
#       T.timeit("conv")

#   return Aout


#def ConvolveGaussianR (Ain0, CellSizeRad=None, GaussPars=[(0., 0., 0.)], Normalise=False, out=None):
#   warnings.warn("deprecated: this doesn't work for small ffts", DeprecationWarning)

#   nch, npol, _, _ = Ain0.shape
#   Aout = np.zeros_like(Ain0) if out is None else out

#   T = ClassTimeIt.ClassTimeIt()
#   T.disable()
#   T.timeit("init")

#   for ch in range(nch):
#       Ain = Ain0[ch]
#       ThisGaussPars = GaussPars[ch]
#       PSF = GiveGauss(Ain.shape[-1], CellSizeRad, ThisGaussPars)
#       T.timeit("givegauss")
#       # PSF=np.ones((Ain.shape[-1],Ain.shape[-1]),dtype=np.float32)
#       if Normalise:
#           PSF /= np.sum(PSF)
#       PSF = np.fft.ifftshift(PSF)
#       fPSF = np.fft.rfft2(PSF)
#       fPSF = np.abs(fPSF)
#       for pol in range(npol):
#           fA = np.fft.rfft2(np.fft.ifftshift(Ain[pol]))
#           nfA = fA * fPSF
#           if_fA = np.fft.irfft2(nfA, s=Ain[pol].shape)
#           Aout[ch, pol, :, :] = np.fft.fftshift(if_fA)
#       T.timeit("conv")

#   return Aout



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

#def testConvolveGaussian(parallel=False):
#    nchan = 10
#    npix = 2000
#    T = ClassTimeIt.ClassTimeIt()
#    if parallel:
#        learnFFTWWisdom(npix)
#        T.timeit("learn")
#        APP.registerJobHandlers(_convolveSingleGaussian)
#        APP.startWorkers()
#    sd = shared_dict.attach("test")
#    A = sd.addSharedArray("A", (nchan,1,npix,npix),np.float32)
#    A[0,0,10,10]=1
#    SigMaj=2#(20/3600.)*np.pi/180
#    SigMin=1#(5/3600.)*np.pi/180
#    ang=30.*np.pi/180
#    GaussPars= [(SigMaj,SigMin,ang)]*nchan
#    CellSizeRad=1#(5./3600)*np.pi/180
#    if parallel:
#        sd.addSharedArray("B", (nchan, 1, 2000, 2000), np.float32)
#        ConvolveGaussianInParallel(sd, "A", "B", CellSizeRad=CellSizeRad, GaussPars=GaussPars)
#        T.timeit("********* parallel")
#    ConvolveGaussianFFTW(A,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
#    T.timeit("********* serial-fftw")
#    ConvolveGaussian(A,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
#    T.timeit("********* serial-np")
#    ConvolveGaussianC(A,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
#    T.timeit("********* serial-npc")
#    ConvolveGaussianR(A,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
#    T.timeit("********* serial-npr")
#
#    sd.delete()
#    if parallel:
#        APP.shutdown()

#class FFTWnp():
#   def __init__(self, A, ncores = 1):
#       warnings.warn("deprecated: this doesn't work for small ffts", DeprecationWarning)

#       dtype=A.dtype
#       self.ThisType=dtype



#   def fft(self,A):
#       axes=(-2,-1)

#       T= ClassTimeIt.ClassTimeIt("ModFFTW")
#       T.disable()

#       A = iFs(A.astype(self.ThisType),axes=axes)
#       T.timeit("shift and copy")
#       #print "do fft"
#       A = np.fft.fft2(A,axes=axes)
#       T.timeit("fft")
#       #print "done"
#       A=Fs(A,axes=axes)/(A.shape[-1]*A.shape[-2])
#       T.timeit("shift")
#       return A
#

#   def ifft(self,A):
#       axes=(-2,-1)
#       #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
#       A = iFs(A.astype(self.ThisType),axes=axes)

#       #print "do fft"
#       A = np.fft.ifft2(A,axes=axes)
#       out=Fs(A,axes=axes)*(A.shape[-1]*A.shape[-2])
#       return out

#
#class FFTWnpNonorm():
#   def __init__(self, A, ncores = 1):
#       warnings.warn("deprecated: this doesn't work for small ffts", DeprecationWarning)

#       #dtype=A.dtype
#       self.ThisType=np.complex64



#   def fft(self,A):
#       axes=(-2,-1)

#       T= ClassTimeIt.ClassTimeIt("ModFFTW")
#       T.disable()

#       A = iFs(A.astype(self.ThisType),axes=axes)
#       T.timeit("shift and copy")
#       #print "do fft"
#       A = np.fft.fft2(A,axes=axes)
#       T.timeit("fft")
#       #print "done"
#       A=Fs(A,axes=axes)#/(A.shape[-1]*A.shape[-2])
#       T.timeit("shift")
#       return A
#

#   def ifft(self,A):
#       axes=(-2,-1)
#       #log=MyLogger.getLogger("ModToolBox.FFTM2.ifft")
#       A = iFs(A.astype(self.ThisType),axes=axes)

#       #print "do fft"
#       A = np.fft.ifft2(A,axes=axes)
#       out=Fs(A,axes=axes)#*(A.shape[-1]*A.shape[-2])
#       return out

#def test_gaussian():
#    input = np.zeros((1,1,512,512))
#    input[0,0,250:262,250:262]=1
#    out,gaussian=ConvolveGaussianScipy(input,4.0)
#    np.save('orig-out.npy',out)
#    np.save('orig-gaussian.npy',gaussian)

#def test_new_gaussian():
#    input = np.zeros((1,1,512,512))
#    input[0,0,250:262,250:262]=1
#    out,gaussian=ConvolveGaussianWrapper(input,4.0)
#    np.save('new-out.npy',out)
#    np.save('new-gaussian.npy',gaussian)

#if __name__=='__main__':
#    print 'Running test_gaussian'
#    test_gaussian()
#    print 'Running test_new_gaussian'
#    test_new_gaussian()
