import numpy as np
from pyrap.images import image
import ClassImageDeconvMachine

def test():
    impsf=image("Continuous.psf")
    psf=impsf.getdata()
    imdirty=image("Continuous.dirty")
    dirty=imdirty.getdata()
    DC=ClassImageDeconvMachine.ClassImageDeconvMachine(Gain=0.051,MaxMinorIter=200,NCPU=30)
    DC.SetDirtyPSF(dirty,psf)
    DC.setSideLobeLevel(0.1)
    DC.Clean()


def test2():
    
    imdirty=np.float32(np.load("imageresid.npy"))

    _,_,N0,_=imdirty.shape
    imdirty=imdirty[:,:,0:1001,0:1001]
    _,_,N1,_=imdirty.shape
    x0,x1=N0/2-N1/2,N0/2+N1/2+1
    impsf=np.float32(np.load("imagepsf.npy"))[:,:,x0:x1,x0:x1]
    print imdirty.shape,impsf.shape

    DC=ClassImageDeconvMachine.ClassImageDeconvMachine(Gain=1.,MaxMinorIter=200,NCPU=1)
    DC.SetDirtyPSF(imdirty.copy(),impsf.copy())
    DC.setSideLobeLevel(0.1,30)
    DC.FindPSFExtent(Method="FromSideLobe")
    LScales=[1,2,4,8,16]
    LRatio=[]
    NTheta=6

    DC.MakeMultiScaleCube(LScales,LRatio,NTheta)
    DC.Clean()
    
def test3():

    psfname="lala2.nocompDeg3.psf.fits"
    dirtyname="lala2.nocompDeg3.dirty.fits"

    
    impsf=image(psfname)
    psf=impsf.getdata()
    imdirty=image(dirtyname)#Test.KAFCA.3SB.dirty.fits")
    dirty=imdirty.getdata()
    
    GD={"MultiScale":{}}
    GD["MultiScale"]["Scales"]=[1,2,4,8,16]
    GD["MultiScale"]["Ratios"]=[1.33,1.66,2]
    GD["MultiScale"]["NTheta"]=6
    DC=ClassImageDeconvMachine.ClassImageDeconvMachine(Gain=.1,MaxMinorIter=1000,NCPU=30,GD=GD)
    DC.SetDirtyPSF(dirty,psf)
    DC.setSideLobeLevel(0.2,308)
    DC.FindPSFExtent(Method="FromSideLobe")

    DC.MakeMultiScaleCube()
    DC.Clean()
    

    c=imdirty.coordinates()
    radec=c.dict()["direction0"]["crval"]

    import ClassCasaImage
    CasaImage=ClassCasaImage.ClassCasaimage("modeltest",DC._ModelImage.shape,2.,radec)
    CasaImage.setdata(DC._ModelImage)#,CorrT=True)
    CasaImage.ToFits()
    CasaImage.close()

