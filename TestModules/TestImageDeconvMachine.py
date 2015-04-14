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
    
    impsf=image("Test.KAFCA.3SB.psf.fits")
    psf=impsf.getdata()
    imdirty=image("Test.KAFCA.3SB.residual6.fits")#Test.KAFCA.3SB.dirty.fits")
    dirty=imdirty.getdata()
    

    DC=ClassImageDeconvMachine.ClassImageDeconvMachine(Gain=.1,MaxMinorIter=1000,NCPU=30)
    DC.SetDirtyPSF(dirty,psf)
    DC.setSideLobeLevel(0.1,50)
    DC.FindPSFExtent(Method="FromSideLobe")
    LScales=[1,2,4,8,16]
    LRatio=[1.33,1.66,2]
    NTheta=6

    DC.MakeMultiScaleCube(LScales,LRatio,NTheta)
    DC.Clean()
    

    c=imdirty.coordinates()
    radec=c.dict()["direction0"]["crval"]

    import ClassCasaImage
    CasaImage=ClassCasaImage.ClassCasaimage("modeltest",DC._ModelImage.shape,2.,radec)
    CasaImage.setdata(DC._ModelImage)#,CorrT=True)
    CasaImage.ToFits()
    CasaImage.close()

