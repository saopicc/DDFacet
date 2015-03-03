import scipy.fftpack
import Gaussian
import numpy as np
import ClassTimeIt
from scipy.interpolate import interp1d as interp
import ModToolBox

import MyLogger
log=MyLogger.getLogger("ModCF",disable=True)


F2=scipy.fftpack.fft2
iF2=scipy.fftpack.ifft2
Fs=scipy.fftpack.fftshift
iFs=scipy.fftpack.ifftshift

def fft2(A):
    #FA= Fs(F2(iFs(A)))#/(np.float64(A.size))
    FA= Fs(F2(iFs(A)))/(np.float64(A.size))
    return FA
 
def ifft2(A):
#     a=iFs(A)
# #    if A.shape[0]==2003: stop
#     b=iF2(a)
#     FA= Fs(b)
    #FA=Fs(iF2(iFs(A)))*np.float64(A.size)
    FA=Fs(iF2(iFs(A)))*np.float64(A.size)
    return FA

def ZeroPad(A,outshape=1001):
    nx=A.shape[0]
#    B=np.zeros((nx*zp,nx*zp),dtype=A.dtype)
    
    if outshape%2==0:
        # PAIR
        B=np.zeros((outshape,outshape),dtype=A.dtype)
        off=(B.shape[0]-A.shape[0])/2+1
        B[off:off+nx,off:off+nx]=A
    else:
        # IMPAIR
        B=np.zeros((outshape,outshape),dtype=A.dtype)
        off=(B.shape[0]-A.shape[0])/2
        B[off:off+nx,off:off+nx]=A
    #print>>log, "!!!!!!!!!! ",outshape,off

    return B    

def MakeSphe(Support,NpixIm):
    x,y,CF=Gaussian.Gaussian(3,Support,1)

    #CF=np.roll(np.roll(CF,1,axis=0),1,axis=1)

    CF=np.complex128(CF)#np.array(np.complex128(CF),order="F")

    fCF=fft2(CF)

    zfCF=ZeroPad(fCF,NpixIm)




    ifzfCF=ifft2(zfCF)

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
    # pylab.show()
    # stop
    ###############

    return CF, fCF, ifzfCF

import ToolsDir.ModFitPoly2D 
def Give_dn(l0,m0,rad=1.,order=4):
    
    Np=100

    
    l,m=np.mgrid[l0-rad:l0+rad:Np*1j,m0-rad:m0+rad:Np*1j]

    dl=l-l0
    dm=m-m0
    S=dl.shape

    dl=dl.flatten()
    dm=dm.flatten()
    y=np.sqrt(1-(dl+l0)**2-(dm+m0)**2)-np.sqrt(1-l0**2-m0**2)
    coef=ToolsDir.ModFitPoly2D.polyfit2d(dl,dm,y,order=order)
    Corig=coef.copy()
    C=coef.reshape((order+1,order+1))
    Cl=C[0,1]
    Cm=C[1,0]
    C[0,1]=0
    C[1,0]=0

    #C=C.T.copy()

    return Cl,Cm,C.flatten()


class ClassWTermModified():
    def __init__(self,Cell=10,Sup=15,Nw=11,wmax=30000,Npix=101,Freqs=np.array([100.e6]),OverS=11,lmShift=None):


        Nw=int(Nw)
        T=ClassTimeIt.ClassTimeIt("Wterm")
        self.CF, self.fCF, self.ifzfCF= MakeSphe(Sup,Npix)
        self.OverS=OverS


        SupMax=501
        dummy, dummy, self.SpheW= MakeSphe(Sup,SupMax)

        # #print>>log, "MAX Sphe=",np.max(np.abs(self.SpheW))
        # T.timeit("initsphe")

        C=299792458.
        D=Npix*Cell/3600.
        lrad=D*0.5*np.pi/180.
        l,m=np.mgrid[-lrad*np.sqrt(2.):np.sqrt(2.)*lrad:SupMax*1j,-lrad*np.sqrt(2.):np.sqrt(2.)*lrad:SupMax*1j]
        n_1=np.sqrt(1.-l**2-m**2)-1
        waveMin=C/Freqs[-1]

        W=np.exp(-2.*1j*np.pi*(wmax/waveMin)*n_1)*self.SpheW
        fW=fft2(W)
        fw1d=np.abs(fW[(SupMax-1)/2,:])
        fw1d/=np.max(fw1d)
        fw1d=fw1d[(SupMax-1)/2::]
        ind=np.argsort(fw1d)
        Interp=interp(fw1d[ind],np.arange(fw1d.shape[0])[ind])

        SupMax=np.int64(Interp(np.array([1./10]))[0])
        Sups=np.int64(np.linspace(self.CF.shape[0],np.max([2*SupMax,self.CF.shape[0]]),Nw))
        # print "Supports=",Sups
        self.Sups=Sups
        #Sups=np.ones((Nw,),int)*Sup

        w=np.linspace(0,wmax,Nw)
        Wplanes=[]
        WplanesConj=[]
        l0,m0=0.,0.

        if lmShift!=None:
            l0,m0=lmShift

        rad=3*lrad
        #print "do FIT"
        self.Cv,self.Cu,CoefPoly=Give_dn(l0,m0,rad=rad,order=5)

        #print "done FIT"

        for i in range(Nw):
            if not(Sups[i]%2): Sups[i]+=1
            dummy, dymmy, ThisSphe= MakeSphe(Sup,Sups[i])

            # l,m=np.mgrid[-lrad:lrad:Sups[i]*1j,-lrad:lrad:Sups[i]*1j]
            # #n_1=np.sqrt(1.-l**2-m**2)-1
            # n_1=ToolsDir.ModFitPoly2D.polyval2d(l, m, CoefPoly)
            # # import pylab
            # # pylab.clf()
            # # pylab.imshow(n_1,interpolation="nearest",extent=(l.min(),l.max(),m.min(),m.max()))
            # # pylab.draw()
            # # pylab.show(False)
            # # stop
            # #n_1=np.sqrt(1.-(l-l0)**2-(m-m0)**2)-1
            # #n_1=(1./np.sqrt(1.-l0**2-m0**2))*(l0*l+m0*m)
            # wl=w[i]/waveMin
            # # W=np.exp(-2.*1j*np.pi*wl*(n_1))
            # # ####
            # # #W.fill(1.)
            # # ####
            # # W*=np.abs(ThisSphe)

            W=ThisSphe

            W=ZeroPad(W,outshape=W.shape[0]*self.OverS)
            

            ####
            W=np.abs(W)
            ####

            Wconj=np.conj(W)

            W=fft2(W)
            Wconj=fft2(Wconj)
            Wplanes.append(np.complex64(W).copy())
            WplanesConj.append(np.complex64(Wconj).copy())
        self.Wplanes=Wplanes
        self.WplanesConj=WplanesConj
        self.Freqs=Freqs
        self.wmap=w
        self.wmax=wmax
        self.Nw=Nw
        self.RefWave=waveMin
            

class ClassWTerm():
    def __init__(self,Cell=10,Sup=15,Nw=11,wmax=30000,Npix=101,Freqs=np.array([100.e6]),OverS=11,lmShift=None):

        T=ClassTimeIt.ClassTimeIt("Wterm")
        self.CF, self.fCF, self.ifzfCF= MakeSphe(Sup,Npix)
        self.OverS=OverS
        


        SupMax=501
        dummy, dummy, self.SpheW= MakeSphe(Sup,SupMax)

        # #print>>log, "MAX Sphe=",np.max(np.abs(self.SpheW))
        # T.timeit("initsphe")

        C=299792458.
        D=Npix*Cell/3600.
        lrad=D*0.5*np.pi/180.
        l,m=np.mgrid[-lrad*np.sqrt(2.):np.sqrt(2.)*lrad:SupMax*1j,-lrad*np.sqrt(2.):np.sqrt(2.)*lrad:SupMax*1j]
        n_1=np.sqrt(1.-l**2-m**2)-1
        waveMin=C/Freqs[-1]

        W=np.exp(-2.*1j*np.pi*(wmax/waveMin)*n_1)*self.SpheW
        fW=fft2(W)
        fw1d=np.abs(fW[(SupMax-1)/2,:])
        fw1d/=np.max(fw1d)
        fw1d=fw1d[(SupMax-1)/2::]
        ind=np.argsort(fw1d)
        Interp=interp(fw1d[ind],np.arange(fw1d.shape[0])[ind])

        SupMax=np.int64(Interp(np.array([1./10]))[0])
        Sups=np.int64(np.linspace(self.CF.shape[0],np.max([2*SupMax,self.CF.shape[0]]),Nw))
        #print "Supports=",Sups

        #Sups=np.ones((Nw,),int)*Sup

        w=np.linspace(0,wmax,Nw)
        Wplanes=[]
        WplanesConj=[]
        l0,m0=0.,0.

        if lmShift!=None:
            l0,m0=lmShift

        n0=np.sqrt(1-l0**2-m0**2)

        for i in range(Nw):
            if not(Sups[i]%2): Sups[i]+=1
            l,m=np.mgrid[-lrad:lrad:Sups[i]*1j,-lrad:lrad:Sups[i]*1j]
            n_1=np.sqrt(1.-l**2-m**2)-1
            #n_1=np.sqrt(1.-(l-l0)**2-(m-m0)**2)-1
            #n_1=(1./np.sqrt(1.-l0**2-m0**2))*(l0*l+m0*m)
            wl=w[i]/waveMin
            W=np.exp(-2.*1j*np.pi*wl*(n_1))
            dummy, dymmy, ThisSphe= MakeSphe(Sup,Sups[i])
            #W.fill(1.)
            
            W*=np.abs(ThisSphe)

            W=ZeroPad(W,outshape=W.shape[0]*self.OverS)
            Wconj=np.conj(W)
            #FFTWMachine=ModToolBox.FFTM2D(W)
            W=fft2(W)
            Wconj=fft2(Wconj)
            
            #W=FFTWMachine.fft(W)
            #Wconj=FFTWMachine.fft(Wconj)


            # pylab.clf()
            # pylab.subplot(2,2,1)
            # z=np.abs(ThisSphe)
            # pylab.imshow(z,interpolation="nearest")
            # pylab.title("abs(ThisSphe), max=%f"%np.max(z))
            # pylab.subplot(2,2,2)
            # z=np.angle(W)
            # pylab.imshow(z,interpolation="nearest",vmin=-np.pi,vmax=np.pi)
            # pylab.title("angle(W), max=%f"%np.max(z))
            # pylab.subplot(2,2,3)
            # z=np.real(W)
            # pylab.imshow(z,interpolation="nearest")
            # pylab.title("real(W), max=%f"%np.max(z))
            # pylab.subplot(2,2,4)
            # z=np.imag(W)
            # pylab.imshow(z,interpolation="nearest")
            # pylab.title("imag(W), max=%f"%np.max(z))
            # pylab.draw()
            # pylab.show(False)


            Wplanes.append(W.copy())
            WplanesConj.append(Wconj.copy())
            # if i==3:
            #     stop
        T.timeit("loop")
        #stop

        self.Wplanes=Wplanes
        self.WplanesConj=WplanesConj
        self.Freqs=Freqs
        self.wmap=w
        self.wmax=wmax
        self.Nw=Nw
        self.RefWave=waveMin
        T.timeit("rest")

    def plot(self):
        pylab.ion()
        for i in range(self.Nw):
            pylab.clf()
            pylab.subplot(2,2,1)
            #pylab.imshow(np.angle(self.Wplanes[i]),interpolation="nearest",vmin=-np.pi,vmax=np.pi)
            pylab.imshow(np.real(self.Wplanes[i]),interpolation="nearest")
            pylab.subplot(2,2,2)
            pylab.imshow(np.imag(self.Wplanes[i]),interpolation="nearest")
            pylab.subplot(2,2,3)
            pylab.imshow(np.abs(self.Wplanes[i]),interpolation="nearest")
            pylab.subplot(2,2,4)
            pylab.imshow(np.angle(self.Wplanes[i]),interpolation="nearest",vmin=-np.pi,vmax=np.pi)

            #pylab.imshow(np.abs(self.Wplanes[i]),interpolation="nearest")
            pylab.title("%f"%self.wmap[i])
            pylab.draw()
            pylab.show()


class ClassSTerm():
    def __init__(self,Cell=10,Sup=15,Nw=11,wmax=30000,Npix=101,Freqs=np.array([100.e6]),OverS=11):

        Nw=int(Nw)
        T=ClassTimeIt.ClassTimeIt("Wterm")
        self.CF, self.fCF, self.ifzfCF= MakeSphe(Sup,Npix)
        self.OverS=OverS
        


        SupMax=501
        dummy, dummy, self.SpheW= MakeSphe(Sup,SupMax)

        # #print>>log, "MAX Sphe=",np.max(np.abs(self.SpheW))
        # T.timeit("initsphe")

        C=299792458.
        D=Npix*Cell/3600.
        lrad=D*0.5*np.pi/180.
        self.lrad=lrad
        # l,m=np.mgrid[-lrad*np.sqrt(2.):np.sqrt(2.)*lrad:SupMax*1j,-lrad*np.sqrt(2.):np.sqrt(2.)*lrad:SupMax*1j]
        # n_1=np.sqrt(1.-l**2-m**2)-1
        waveMin=C/Freqs[-1]

        # W=np.exp(-2.*1j*np.pi*(wmax/waveMin)*n_1)*self.SpheW
        # fW=fft2(W)
        # fw1d=np.abs(fW[(SupMax-1)/2,:])
        # fw1d/=np.max(fw1d)
        # fw1d=fw1d[(SupMax-1)/2::]
        # ind=np.argsort(fw1d)
        # Interp=interp(fw1d[ind],np.arange(fw1d.shape[0])[ind])

        # SupMax=np.int64(Interp(np.array([1./10]))[0])
        # Sups=np.int64(np.linspace(self.CF.shape[0],np.max([2*SupMax,self.CF.shape[0]]),Nw))
        # print "Supports=",Sups

        Sups=np.ones((Nw,),int)*Sup
        self.Cv,self.Cu=0,0

        w=np.linspace(0,wmax,Nw)
        Wplanes=[]
        WplanesConj=[]
        for i in range(Nw):
            if not(Sups[i]%2): Sups[i]+=1
            l,m=np.mgrid[-lrad:lrad:Sups[i]*1j,-lrad:lrad:Sups[i]*1j]
            n_1=np.sqrt(1.-l**2-m**2)-1
            wl=w[i]/waveMin
            W=np.exp(-2.*1j*np.pi*wl*n_1)
            dummy, dymmy, ThisSphe= MakeSphe(Sup,Sups[i])
            W.fill(1.)

            W*=np.abs(ThisSphe)
            
            W=ZeroPad(W,outshape=W.shape[0]*self.OverS)
            Wconj=np.conj(W)
            FFTWMachine=ModToolBox.FFTM2D(W)
            #W=fft2(W)
            #Wconj=fft2(Wconj)
            W=FFTWMachine.fft(W)
            Wconj=FFTWMachine.fft(Wconj)


            # pylab.clf()
            # pylab.subplot(2,2,1)
            # z=np.abs(ThisSphe)
            # pylab.imshow(z,interpolation="nearest")
            # pylab.title("abs(ThisSphe), max=%f"%np.max(z))
            # pylab.subplot(2,2,2)
            # z=np.angle(W)
            # pylab.imshow(z,interpolation="nearest",vmin=-np.pi,vmax=np.pi)
            # pylab.title("angle(W), max=%f"%np.max(z))
            # pylab.subplot(2,2,3)
            # z=np.real(W)
            # pylab.imshow(z,interpolation="nearest")
            # pylab.title("real(W), max=%f"%np.max(z))
            # pylab.subplot(2,2,4)
            # z=np.imag(W)
            # pylab.imshow(z,interpolation="nearest")
            # pylab.title("imag(W), max=%f"%np.max(z))
            # pylab.draw()
            # pylab.show(False)


            Wplanes.append(W.copy())
            WplanesConj.append(Wconj.copy())
            # if i==3:
            #     stop
        T.timeit("loop")
        #stop

        self.Wplanes=Wplanes
        self.WplanesConj=WplanesConj
        self.Freqs=Freqs
        self.wmap=w
        self.wmax=wmax
        self.Nw=Nw
        self.RefWave=waveMin
        T.timeit("rest")

    def plot(self):
        pylab.ion()
        for i in range(self.Nw):
            pylab.clf()
            pylab.subplot(2,2,1)
            #pylab.imshow(np.angle(self.Wplanes[i]),interpolation="nearest",vmin=-np.pi,vmax=np.pi)
            pylab.imshow(np.real(self.Wplanes[i]),interpolation="nearest")
            pylab.subplot(2,2,2)
            pylab.imshow(np.imag(self.Wplanes[i]),interpolation="nearest")
            pylab.subplot(2,2,3)
            pylab.imshow(np.abs(self.Wplanes[i]),interpolation="nearest")
            pylab.subplot(2,2,4)
            pylab.imshow(np.angle(self.Wplanes[i]),interpolation="nearest",vmin=-np.pi,vmax=np.pi)

            #pylab.imshow(np.abs(self.Wplanes[i]),interpolation="nearest")
            pylab.title("%f"%self.wmap[i])
            pylab.draw()
            pylab.show()
