import numpy as np
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassAdaptShape")
from DDFacet.Other import ModColor

class ClassAdaptShape():
    def __init__(self,A):
        self.A=A
    

    def giveOutIm(self,Nout):
        A=self.A
        nch,npol,Nin,_=A.shape

        if Nin==Nout: 
            return A.copy()
        elif Nin>Nout:
            B=np.zeros((nch,npol,Nout,Nout),A.dtype)
            print>>log,ModColor.Str("  Output image smaller than input image")
            print>>log,ModColor.Str("     adapting %s --> %s"%(str(A.shape),str(B.shape)))
            # Input image larger than requested
            N0=A.shape[-1]
            xc0=yc0=N0/2
            x0d,x1d=xc0-Nout/2,xc0-Nout/2+Nout
            s=slice(x0d,x1d)
            B[:,:,:,:]=A[:,:,s,s]
            return B
        else:
            B=np.zeros((nch,npol,Nout,Nout),A.dtype)
            # Input image smaller - has to zero pad
            print>>log,ModColor.Str("  Output image larger than input image")
            print>>log,ModColor.Str("     adapting %s --> %s"%(str(A.shape),str(B.shape)))
            Na=A.shape[-1]
            Nb=B.shape[-1]
            xa=Na/2
            xb=Nb/2
            x0d,x1d=xb-xa,xb-xa+Na
            s=slice(x0d,x1d)
            B[:,:,s,s]=A[:,:,:,:]
            return B
            
