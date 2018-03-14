import numpy as np
import Gaussian
import pylab
import scipy.optimize
import time
import ClassIslands
from SkyModel.Other import ModColor
from SkyModel.Other.progressbar import ProgressBar
from ClassGaussFit import ClassGaussFit
from ModConvPSF import ClassConvPSF

def test():
    nn=101.
    x,y=np.mgrid[0:nn,0:nn]
    xx=sorted(list(set(x.flatten().tolist())))
    dx=xx[1]-xx[0]
    dx=1.
    adp=1.

    psfPars=1,10.,120.*np.pi/180
    PSm,PSM,PPA=psfPars
    CPSF=ClassConvPSF(psfPars)

    GaussPars=3,10,10.*np.pi/180.
    GaussParsConv=CPSF.GiveConvGaussPars(GaussPars)
    Sm2,SM2,PA2=GaussParsConv


    # z=Gaussian.GaussianXY(x,y,1.,off=(50,50),sig=(adp*dx,adp*dx),pa=20.*np.pi/180)
    # z+=Gaussian.GaussianXY(x,y,1.,off=(55,50),sig=(adp*dx,adp*dx),pa=20.*np.pi/180)
    # z+=Gaussian.GaussianXY(x,y,.5,off=(25,25),sig=(adp*dx,adp*dx),pa=20.*np.pi/180)
    # z+=Gaussian.GaussianXY(x,y,.5,off=(75,25),sig=(adp*dx,adp*dx),pa=20.*np.pi/180)

    # #z+=Gaussian.GaussianXY(x,y,.5,off=(75,75),sig=(5*adp*dx,adp*dx),pa=20.*np.pi/180)
    # z+=Gaussian.GaussianXY(x,y,.5,off=(50,50),sig=(5*adp*dx,adp*dx),pa=20.*np.pi/180)

    z=Gaussian.GaussianXY(x,y,.5,off=(75,75),sig=(Sm2,SM2),pa=PA2)


    noise=0.01
    #z+=np.random.randn(nn,nn)*noise
    # z+=Gaussian.GaussianXY(x,y,1.,off=(50,50),sig=(1*dx,1*dx),pa=20.*np.pi/180)
    #pylab.clf()
    #dx*=1.5
    #pylab.ion()

    pylab.clf()
    pylab.imshow(z,interpolation="nearest")
    pylab.draw()
    pylab.show(False)
    #Fit=ClassGaussFit(x,y,z,psf=(dx,dx,0.),noise=noise,FreePars=["l", "m","s","Sm","SM","PA"])
    Fit=ClassGaussFit(x,y,z,psf=psfPars,noise=noise,FreePars=["l", "m","s","Sm","SM","PA"])
    Fit.DoAllFit()
