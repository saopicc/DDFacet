import scipy.linalg
import numpy as np
import Gaussian
import pylab
import scipy.optimize
import time
import ClassIslands
from SkyModel.Other import ModColor
from SkyModel.Other.progressbar import ProgressBar
#from ClassGaussFit import ClassGaussFit


def test():
    
    nn=101.
    x,y=np.mgrid[0:nn,0:nn]
    xx=sorted(list(set(x.flatten().tolist())))

    GaussPars=3.,15.,190.*np.pi/180
    #GaussPars=0.001,0.001,30.*np.pi/180

    Sm,SM,PA=GaussPars

    #########
    psfPars=1,10.,120.*np.pi/180
    #psfPars=0.001,.001,30.*np.pi/180
    PSm,PSM,PPA=psfPars
    z1=Gaussian.GaussianXY(x,y,1.,off=(50,50),sig=(PSm,PSM),pa=PPA)
    #########

    CPSF=ClassConfPSF(psfPars)
    GaussParsConv=CPSF.GiveConvGaussPars(GaussPars)
    Sm2,SM2,PA2=GaussParsConv

    z0=Gaussian.GaussianXY(x,y,1.,off=(50,50),sig=(Sm,SM),pa=PA)
    z2=Gaussian.GaussianXY(x,y,1.,off=(50,50),sig=(Sm2,SM2),pa=PA2)

    pylab.clf()
    pylab.subplot(1,3,1)
    pylab.imshow(z0,interpolation="nearest")
    pylab.subplot(1,3,2)
    pylab.imshow(z1,interpolation="nearest")
    pylab.subplot(1,3,3)
    pylab.imshow(z2,interpolation="nearest")
    pylab.draw()
    pylab.show(False)
    



class ClassConvPSF():
    def __init__(self,psf):
        self.PMin,self.PMaj,self.PPA=psf
        self.P_a,self.P_b,self.P_c=self.Give_GaussABC(self.PMin,self.PMaj,self.PPA)
        
    def Give_GaussABC(self,m0in,m1in,ang):
        if m0in==0: m0in=1e-6
        if m1in==0: m1in=1e-6
        m0=1./m0in
        m1=1./m1in
        a=0.5*((np.cos(ang)/m0)**2+(np.sin(ang)/m1)**2)
        b=0.25*(-np.sin(2*ang)/(m0**2)+np.sin(2.*ang)/(m1**2))
        c=0.5*((np.sin(ang)/m0)**2+(np.cos(ang)/m1)**2)
        if a==0: a=1e-6
        if b==0: b=1e-6
        if c==0: c=1e-6
        return a,b,c

    def GiveConvGaussPars(self,GaussPars):

        PMin,PMaj,PPA=self.PMin,self.PMaj,self.PPA
        P_a,P_b,P_c=self.P_a,self.P_b,self.P_c

        SigMin,SigMaj,PA=GaussPars

        M_a,M_b,M_c=self.Give_GaussABC(SigMin,SigMaj,PA)
        #da=1./(1./M_a+1./P_a)
        #db=1./(1./M_b+1./P_b)
        #dc=1./(1./M_c+1./P_c)

        da=M_a+P_a
        db=M_b+P_b
        dc=M_c+P_c
        M=np.abs(np.array([[da,db],[db,dc]]))
        #print M
        u,l,d=scipy.linalg.svd(M)

        Theta=np.angle((u[0,0]+1j*u[0,1]))#/np.pi#-np.pi/2
        
        if l[0]!=0:
            a=1./l[0]
            SigMin0=1./np.sqrt(a/2.)
        else:
            SigMin0=0.
        if l[1]!=0:
            b=1./l[1]
            SigMaj0=1./np.sqrt(b/2.)
        else:
            SigMaj0=0.
        if SigMin0>SigMaj0:
            #print "invert"
            c=SigMaj0
            SigMaj0=SigMin0
            SigMin0=c
            Theta+=np.pi/2

        return SigMin0,SigMaj0,Theta
            
