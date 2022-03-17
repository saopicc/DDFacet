from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import numpy as np
from DDFacet.Other import logger
log=logger.getLogger("ClassAdaptShape")
from DDFacet.Other import ModColor

class ClassAdaptShape():
    def __init__(self,A):
        self.A=A
    

    def giveOutIm(self,Nout):
        A=self.A
        try:
            Nout_x,Nout_y=Nout
        except:
            Nout_x=Nout_y=Nout
        nch,npol,Nin_x,Nin_y=A.shape

        if (Nin_x==Nout_x) and (Nin_y==Nout_y): 
            return A.copy()
        
        elif Nin>Nout:
            B=np.zeros((nch,npol,Nout,Nout),A.dtype)
            print(ModColor.Str("  Output image smaller than input image"), file=log)
            print(ModColor.Str("     adapting %s --> %s"%(str(A.shape),str(B.shape))), file=log)
            # Input image larger than requested
            N0=A.shape[-1]
            xc0=yc0=N0//2
            x0d,x1d=xc0-Nout//2,xc0-Nout//2+Nout
            s=slice(x0d,x1d)
            B[:,:,:,:]=A[:,:,s,s]
            return B
        else:
            B=np.zeros((nch,npol,Nout,Nout),A.dtype)
            # Input image smaller - has to zero pad
            print(ModColor.Str("  Output image larger than input image"), file=log)
            print(ModColor.Str("     adapting %s --> %s"%(str(A.shape),str(B.shape))), file=log)
            Na=A.shape[-1]
            Nb=B.shape[-1]
            xa=Na//2
            xb=Nb//2
            x0d,x1d=xb-xa,xb-xa+Na
            s=slice(x0d,x1d)
            B[:,:,s,s]=A[:,:,:,:]
            return B
            
