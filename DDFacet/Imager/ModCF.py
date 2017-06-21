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
from DDFacet.ToolsDir import Gaussian
import numpy as np
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
from scipy.interpolate import interp1d as interp
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Array import NpShared
from DDFacet.Other import MyLogger
log = MyLogger.getLogger("WTerm")  # ,disable=True)
from DDFacet.ToolsDir import ModTaper
from DDFacet.ToolsDir import ModFitPoly2D
from DDFacet.ToolsDir import ModFFTW


F2 = scipy.fftpack.fft2
iF2 = scipy.fftpack.ifft2
Fs = scipy.fftpack.fftshift
iFs = scipy.fftpack.ifftshift


def fft2(A):
    # FA= Fs(F2(iFs(A)))#/(np.float64(A.size))
    FA = Fs(F2(iFs(A)))/(np.float64(A.size))
    return FA


def ifft2(A):
    #     a=iFs(A)
    # #    if A.shape[0]==2003: stop
    #     b=iF2(a)
    #     FA= Fs(b)
    # FA=Fs(iF2(iFs(A)))*np.float64(A.size)
    FA = Fs(iF2(iFs(A)))*np.float64(A.size)
    return FA


def ZeroPad(A, outshape=1001):
    nx = A.shape[0]
#    B=np.zeros((nx*zp,nx*zp),dtype=A.dtype)

    if outshape % 2 == 0:
        # PAIR
        B = np.zeros((outshape, outshape), dtype=A.dtype)
        off = (B.shape[0]-A.shape[0])/2+1
        B[off:off+nx, off:off+nx] = A
    else:
        # IMPAIR
        B = np.zeros((outshape, outshape), dtype=A.dtype)
        off = (B.shape[0]-A.shape[0])/2
        B[off:off+nx, off:off+nx] = A
    #print>>log, "!!!!!!!!!! ",outshape,off

    return B


class SpheMachine():

    def __init__(self, Support=11, SupportSpheCalc=111, Type="Sphe"):
        self.Support = Support
        self.SupportSpheCalc = SupportSpheCalc
        self.Type = Type
        self.setSmall_fCF()

    def setSmall_fCF(self):
        Support = self.Support
        SupportSphe = self.SupportSpheCalc
        if self.Type == "Sphe":
            xc = SupportSphe/2
            CF = ModTaper.Sphe2D(SupportSphe)
            # CF.fill(1)
            CF = np.complex128(CF)  # np.array(np.complex128(CF),order="F")
            fCF = fft2(CF)
            fCF = fCF[
                xc-Support/2:xc+Support/2+1,
                xc-Support/2:xc+Support/2+1].copy()
            if_cut_fCF = ifft2(fCF)
        elif self.Type == "Gauss":
            x, y, CF = Gaussian.Gaussian(3, Support, 1)
            CF = np.complex128(CF)  # np.array(np.complex128(CF),order="F")
            fCF = fft2(CF)

        self.Small_fCF = fCF
        self.Small_CF = CF
        self.if_cut_fCF = if_cut_fCF

    def MakeSphe(self, NpixIm):
        fCF = self.Small_fCF
        zfCF = ZeroPad(fCF, NpixIm)

        ifzfCF = ifft2(zfCF)
        CF = self.Small_CF

        # ############"
        # import pylab
        # lpar=list(pylab.imshow.__defaults__)
        # lpar[3]="nearest"
        # pylab.imshow.__defaults__=tuple(lpar)

        # pylab.figure(0)
        # pylab.clf()
        # pylab.subplot(3,2,1)
        # pylab.imshow(CF.real)
        # pylab.colorbar()
        # pylab.subplot(3,2,2)
        # pylab.imshow(CF.imag)
        # pylab.colorbar()
        # pylab.subplot(3,2,3)
        # pylab.imshow(fCF.real)
        # pylab.colorbar()
        # pylab.subplot(3,2,4)
        # pylab.imshow(fCF.imag)
        # pylab.colorbar()
        # pylab.subplot(3,2,5)
        # pylab.imshow(ifzfCF.real)
        # pylab.colorbar()
        # pylab.subplot(3,2,6)
        # pylab.imshow(ifzfCF.imag)
        # pylab.colorbar()
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        # f_inv_ifzfCF=ifft2(1./ifzfCF.real)

        # pylab.figure(1)
        # pylab.clf()
        # pylab.subplot(3,2,1)
        # pylab.imshow(f_inv_ifzfCF.real)
        # pylab.colorbar()
        # pylab.subplot(3,2,2)
        # pylab.imshow(f_inv_ifzfCF.imag)
        # pylab.colorbar()
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)


        # stop
        ifzfCF[ifzfCF < 0] = 1e-10

        return CF, fCF, ifzfCF


def test():
    S = 15
    SpheM0 = SpheMachine(Support=S, Type="Sphe")
    SpheM1 = SpheMachine(Support=S, Type="Gauss")
    Npix = 151
    _, _, ifzfCF0 = SpheM0.MakeSphe(Npix)
    _, _, ifzfCF1 = SpheM1.MakeSphe(Npix)

    import pylab
    pylab.clf()
    lpar = list(pylab.imshow.__defaults__)
    lpar[3] = "nearest"
    pylab.imshow.__defaults__ = tuple(lpar)
    pylab.subplot(2, 2, 1)
    pylab.imshow(ifzfCF0.real)
    pylab.colorbar()
    pylab.subplot(2, 2, 2)
    pylab.imshow(ifzfCF0.imag)
    pylab.colorbar()
    pylab.subplot(2, 2, 3)
    pylab.imshow(ifzfCF1.real/ifzfCF0.real)
    pylab.colorbar()
    pylab.subplot(2, 2, 4)
    pylab.imshow(ifzfCF1.imag)
    pylab.colorbar()
    pylab.draw()
    pylab.show(False)
    pylab.pause(0.1)
    # stop

# test()
# stop


def MakeSphe(Support, NpixIm):
    # x,y,CF=Gaussian.Gaussian(3,Support,1)

    import ClassTimeIt
    T = ClassTimeIt.ClassTimeIt()
    SupportSphe = 111
    xc = SupportSphe/2
    CF = ModTaper.Sphe2D(SupportSphe)
    T.timeit("0")
    CF = np.complex128(CF)  # np.array(np.complex128(CF),order="F")

    fCF = fft2(CF)
    fCF = fCF[xc-Support/2:xc+Support/2+1, xc-Support/2:xc+Support/2+1].copy()
    zfCF = ZeroPad(fCF, NpixIm)
    T.timeit("1")

    ifzfCF = ifft2(zfCF)

    # ############"
    # import pylab
    # pylab.clf()
    # pylab.subplot(3,2,1)
    # lpar=list(pylab.imshow.__defaults__)
    # lpar[3]="nearest"
    # pylab.imshow.__defaults__=tuple(lpar)
    # pylab.imshow(CF.real)
    # pylab.colorbar()
    # pylab.subplot(3,2,2)
    # pylab.imshow(CF.imag)
    # pylab.colorbar()
    # pylab.subplot(3,2,3)
    # pylab.imshow(fCF.real)
    # pylab.colorbar()
    # pylab.subplot(3,2,4)
    # pylab.imshow(fCF.imag)
    # pylab.colorbar()
    # pylab.subplot(3,2,5)
    # pylab.imshow(ifzfCF.real)
    # pylab.colorbar()
    # pylab.subplot(3,2,6)
    # pylab.imshow(ifzfCF.imag)
    # pylab.colorbar()
    # pylab.draw()
    # pylab.show(False)
    # pylab.pause(0.1)
    # stop

    return CF, fCF, ifzfCF


def GiveSupports(FOVrad, w, NSphe):
    fmax = np.pi*w*FOVrad*np.sqrt(2.)/2
    Nw = FOVrad*fmax
    Nw = np.pi*w*FOVrad**2/np.sqrt(2.)
    N = np.sqrt(Nw**2+NSphe**2)
    return N


def Give_dn(l0, m0, rad=1., order=4):

    Np = 100

    l, m = np.mgrid[l0-rad:l0+rad:Np*1j, m0-rad:m0+rad:Np*1j]

    dl = l-l0
    dm = m-m0
    S = dl.shape

    dl = dl.flatten()
    dm = dm.flatten()
    y = np.sqrt(1-(dl+l0)**2-(dm+m0)**2)-np.sqrt(1-l0**2-m0**2)
    coef = ModFitPoly2D.polyfit2d(dl, dm, y, order=order)
    Corig = coef.copy()
    C = coef.reshape((order+1, order+1))
    Cl = C[0, 1]
    Cm = C[1, 0]
    C[0, 1] = 0
    C[1, 0] = 0

    # C=C.T.copy()

    return Cl, Cm, C.flatten()


class ClassWTermModified():
    def __init__(self, Cell=10, Sup=15, Nw=11, wmax=30000, Npix=101, Freqs=np.array([100.e6]), OverS=11, lmShift=None,
                 mode="compute",
                 cf_dict=None, compute_cf=True,
                 IDFacet=None):
        """
        Class for computing/loading/saving w-kernels and spheroidals.

        Args:
            Cell:
            Sup:
            Nw:
            wmax:
            Npix:
            Freqs:
            OverS:
            lmShift:
            mode:       "compute" to compute CFs, save them to store_dict, and write to store_file
                        "load" to load CFs from store_file, and save them to store_dict
                        "dict" to load CFs from store_dict
            IDFacet:
        """

        self.Nw = int(Nw)
        self.Cell = Cell
        self.Sup = Sup
        self.Nw = Nw
        self.Npix = Npix
        self.Freqs = Freqs
        self.OverS = OverS
        self.lmShift = lmShift
        self.IDFacet = IDFacet
        Freqs = self.Freqs
        C = 299792458.
        waveMin = C/Freqs[-1]
        self.RefWave = waveMin

        # recompute?
        if compute_cf:
            cf_dict["wmax"] = self.wmax = wmax
            self.InitSphe()
            self.InitW()
            dS = np.float32
            cf_dict["Sphe"] = dS(self.ifzfCF.real)
            cf_dict["InvSphe"] = dS(1./np.float64(self.ifzfCF.real))
            cf_dict["CuCv"] = np.array([self.Cu, self.Cv])
            NpShared.PackListSquareMatrix(cf_dict, "W", self.Wplanes + self.WplanesConj)
        else:
            self.wmax = cf_dict["wmax"]
            self.ifzfCF = cf_dict["Sphe"]
            self.Cu, self.Cv = cf_dict["CuCv"]
            ww = NpShared.UnPackListSquareMatrix(cf_dict["W"])
            if len(ww) != self.Nw*2:
                raise RuntimeError("mismatch in number of cached w-planes")
            self.Wplanes = ww[:self.Nw]
            self.WplanesConj = ww[self.Nw:]


    def InitSphe(self):
        T = ClassTimeIt.ClassTimeIt("Wterm")
        #self.CF, self.fCF, self.ifzfCF= MakeSphe(self.Sup,self.Npix)

        self.SpheM = SpheMachine(Support=self.Sup)  # ,Type="Gauss")
        self.CF, self.fCF, self.ifzfCF = self.SpheM.MakeSphe(self.Npix)

    def GiveReorgCF(self, A):
        Sup = A.shape[0]/self.OverS
        B = np.zeros((self.OverS, self.OverS, Sup, Sup), dtype=A.dtype)
        for i in xrange(self.OverS):
            for j in xrange(self.OverS):
                B[i, j, :, :] = A[i::self.OverS, j::self.OverS]  # [::-1,:]

        B = B.reshape((A.shape[0], A.shape[0]))
        return B

    def InitW(self):

        #print>>log, "InitW"
        Nw = self.Nw
        Cell = self.Cell
        Sup = self.Sup
        wmax = self.wmax
        Nw = self.Nw
        Npix = self.Npix
        Freqs = self.Freqs
        OverS = self.OverS
        lmShift = self.lmShift

        T = ClassTimeIt.ClassTimeIt()
        T.disable()
        SupMax = 501
        dummy, dummy, self.SpheW = self.SpheM.MakeSphe(SupMax)

        # #print>>log, "MAX Sphe=",np.max(np.abs(self.SpheW))
        # T.timeit("initsphe")

        C = 299792458.

        RadiusDeg = ((Npix)/2.)*Cell/3600.
        lrad = RadiusDeg*np.pi/180.
        # lrad/=1.05

        l, m = np.mgrid[-lrad * np.sqrt(2.): np.sqrt(2.) * lrad: SupMax * 1j, -
                        lrad * np.sqrt(2.): np.sqrt(2.) * lrad: SupMax * 1j]
        n_1 = np.sqrt(1.-l**2-m**2)-1
        waveMin = C/Freqs[-1]
        T.timeit("0")
        W = np.exp(-2.*1j*np.pi*(wmax/waveMin)*n_1)*self.SpheW
        fW = fft2(W)
        fw1d = np.abs(fW[(SupMax-1)/2, :])
        fw1d /= np.max(fw1d)
        fw1d = fw1d[(SupMax-1)/2::]
        ind = np.argsort(fw1d)
        T.timeit("1")

        try:
            Interp = interp(fw1d[ind], np.arange(fw1d.shape[0])[ind])
            T.timeit("2")

            SupMax = np.int64(Interp(np.array([1./1000]))[0])
        except:
            SupMax = Sup

        Sups = np.int64(np.linspace(Sup, np.max([SupMax, Sup]), Nw))

        w = np.linspace(0, wmax, Nw)

        # FOVrad=lrad*2.
        # for i in range(Nw):
        #     Sups[i]=GiveSupports(FOVrad,w[i],Sup)

        # print "Supports=",Sups
        self.Sups = Sups
        # if self.IDFacet==0:
        #    print>>log, "  Support of the %i Wterms: %s"%(self.Sups.size,str(self.Sups))
        # Sups=np.ones((Nw,),int)*Sup
        T.timeit("3")

        Wplanes = []
        WplanesConj = []
        l0, m0 = 0., 0.

        if lmShift is not None:
            l0, m0 = lmShift

        rad = 3*lrad
        # print "do FIT"
        self.Cv, self.Cu, CoefPoly = Give_dn(l0, m0, rad=rad, order=5)
        # print self.IDFacet,l0,m0,self.Cv,self.Cu

        # print "done FIT"

        for i in xrange(Nw):
            #print>>log, "%i/%i"%(i,Nw)
            if not(Sups[i] % 2):
                Sups[i] += 1
            dummy, dymmy, ThisSphe = self.SpheM.MakeSphe(Sups[i])
            wl = w[i]/waveMin

            # ##############
            # l,m=np.mgrid[-lrad:lrad:Npix*1j,-lrad:lrad:Npix*1j]
            # n_1=np.sqrt(1.-l**2-m**2)-1
            # WTrue=np.exp(-2.*1j*np.pi*wl*(n_1))*self.ifzfCF
            # ##############

            DX = 2*lrad/Sups[i]
            l, m = np.mgrid[-lrad+DX/2:lrad-DX/2:Sups[i]
                            * 1j, -lrad+DX/2:lrad-DX/2:Sups[i]*1j]
            # l,m=np.mgrid[-lrad:lrad:Sups[i]*1j,-lrad:lrad:Sups[i]*1j]
            # n_1=np.sqrt(1.-l**2-m**2)-1
            n_1 = ModFitPoly2D.polyval2d(l, m, CoefPoly)
            # n_1=n_1.T[::-1,:]
            T.timeit("3a")

            # stop
            # n_1=np.sqrt(1.-(l-l0)**2-(m-m0)**2)-1
            # n_1=(1./np.sqrt(1.-l0**2-m0**2))*(l0*l+m0*m)
            W = np.exp(-2.*1j*np.pi*wl*(n_1))
            #import pylab

            # pylab.clf()
            # pylab.imshow(np.angle(W),interpolation="nearest",extent=(l.min(),l.max(),m.min(),m.max()),vmin=-np.pi,vmax=np.pi)
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            # T.timeit("3b")

            # ####
            # W.fill(1.)
            # ####
            W *= np.abs(ThisSphe)

            # # ####################
            # # fW=fft2(W)
            # # zfW=ZeroPad(fW,outshape=Npix)
            # # WFull=ifft2(zfW)

            # # # #fW=ifft2(W)
            # # # print W.shape
            # fWTrue=fft2(WTrue)
            # cfWTrue=fft2(np.conj(WTrue))
            # nc,_=fWTrue.shape
            # xc=(nc-1)/2
            # dx=(Sups[i]-1)/2
            # x0,x1=xc-dx,xc+dx+1
            # #ZfWTrue=np.zeros_like(fWTrue)
            # #ZfWTrue[x0:x1,x0:x1]=fWTrue[x0:x1,x0:x1]
            # fzW=fWTrue[x0:x1,x0:x1]
            # fzWconj=cfWTrue[x0:x1,x0:x1]

            # if_fzW=ifft2(fzW)
            # cif_fzW=ifft2(fzWconj)
            # z_if_fzW=ZeroPad(if_fzW,outshape=if_fzW.shape[0]*self.OverS)
            # z_cif_fzW=ZeroPad(cif_fzW,outshape=if_fzW.shape[0]*self.OverS)
            # f_z_if_fzW=fft2(z_if_fzW)
            # f_z_cif_fzW=fft2(z_cif_fzW)
            # # ifZfWTrue=ifft2(ZfWTrue)

            # # iffWTrue=ifft2(fWTrue[x0:x1,x0:x1])
            # # iffW=ifft2(fW)

            # # pylab.clf()
            # # pylab.subplot(1,2,1)
            # # pylab.imshow(np.real(fzW),interpolation="nearest")#,extent=(l.min(),l.max(),m.min(),m.max()),vmin=-np.pi,vmax=np.pi)
            # # pylab.subplot(1,2,2)
            # # pylab.imshow(np.real(f_z_if_fzW),interpolation="nearest")#,extent=(l.min(),l.max(),m.min(),m.max()),vmin=-np.pi,vmax=np.pi)
            # # pylab.draw()
            # # pylab.show(False)
            # # pylab.pause(0.1)
            # # fzW=f_z_if_fzW
            # # fzWconj=f_z_cif_fzW
            # # # ####################

            # T.timeit("3c")

            # # W=ThisSphe

            zW = ZeroPad(W, outshape=W.shape[0]*self.OverS)

            # T.timeit("3d")

            # ####
            # # W=np.abs(W)
            # ####

            zWconj = np.conj(zW)

            # #FFTWMachine=ModFFTW.FFTW_2Donly(W.shape,W.dtype, ncores = 1)
            # #W=FFTWMachine.fft(W)
            # #Wconj=FFTWMachine.fft(Wconj)
            fzW = fft2(zW)
            fzWconj = fft2(zWconj)

            # T.timeit("3e")
            fzW = np.complex64(fzW).copy()
            fzWconj = np.complex64(fzWconj).copy()

            # fzW.fill(2+3*1j)
            # fzWconj.fill(2+3*1j)

            fzW = self.GiveReorgCF(fzW)
            fzWconj = self.GiveReorgCF(fzWconj)

            fzW = np.require(fzW.copy(), requirements=["A", "C"])
            fzWconj = np.require(fzWconj.copy(), requirements=["A", "C"])
            Wplanes.append(fzW)
            WplanesConj.append(fzWconj)
            # T.timeit("3f")

        self.Wplanes = Wplanes
        self.WplanesConj = WplanesConj
        self.Freqs = Freqs
        self.wmap = w
        self.wmax = wmax
        self.Nw = Nw
