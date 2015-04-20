import numpy as np
from DDFacet.ToolsDir import ModFFTW


class ClassImToGrid():
    def __init__(self,
                 GridShape,
                 PaddingInnerCoord,
                 OverS,Padding,
                 dtype,
                 ifzfCF=None):
        
        self.GridShape=GridShape
        self.PaddingInnerCoord=PaddingInnerCoord
        self.OverS=OverS
        self.Padding=Padding
        self.dtype=dtype
        self.ifzfCF=ifzfCF
        self.FFTWMachine=ModFFTW.FFTW_2Donly_np(self.GridShape,self.dtype, ncores = 1)

    def setModelIm(self,ModelIm):
        _,_,n,n=ModelIm.shape
        x0,x1=self.PaddingInnerCoord
        ModelImPadded=np.zeros(self.GridShape,dtype=self.dtype)
        ModelImPadded[:,:,x0:x1,x0:x1]=ModelIm
        
        Grid=self.dtype(self.ImToGrid(ModelImPadded)*n**2)
        return Grid

    def ImToGrid(self,ModelIm):
        
        ModelImCorr=ModelIm*(self.OverS*self.Padding)**2

        nchan,npol,_,_=ModelImCorr.shape
        if self.ifzfCF!=None:
            for ichan in range(nchan):
                for ipol in range(npol):
                    ModelImCorr[ichan,ipol][:,:]=ModelImCorr[ichan,ipol][:,:].real/self.ifzfCF


        ModelUVCorr=self.FFTWMachine.fft(ModelImCorr)

        return ModelUVCorr
