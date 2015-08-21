import numpy as np

class ClassPSFServer():
    def __init__(self,GD):
        self.GD=GD

    def setDicoVariablePSF(self,DicoVariablePSF):
        self.DicoVariablePSF=DicoVariablePSF
        NFacets=len(DicoVariablePSF.keys())
        NPixMin=1e6
        for iFacet in sorted(DicoVariablePSF.keys()):
            _,npol,n,n=DicoVariablePSF[iFacet]["PSF"][0].shape
            if n<NPixMin: NPixMin=n

        nch=self.GD["MultiFreqs"]["NFreqBands"]
        CubeVariablePSF=np.zeros((NFacets,nch,npol,NPixMin,NPixMin),np.float32)
        CubeMeanVariablePSF=np.zeros((NFacets,1,npol,NPixMin,NPixMin),np.float32)
        for iFacet in sorted(DicoVariablePSF.keys()):
            _,npol,n,n=DicoVariablePSF[iFacet]["PSF"][0].shape
            for ch in range(nch):
                i=n/2-NPixMin/2
                j=n/2+NPixMin/2+1
                CubeVariablePSF[iFacet,ch,:,:,:]=DicoVariablePSF[iFacet]["PSF"][ch][0,:,i:j,i:j]
            CubeMeanVariablePSF[iFacet,0,:,:,:]=DicoVariablePSF[iFacet]["MeanPSF"][0,:,i:j,i:j]
        self.CubeVariablePSF=CubeVariablePSF
        self.CubeMeanVariablePSF=CubeMeanVariablePSF
        self.ShapePSF=nch,npol,NPixMin,NPixMin
        self.NPSF=NPixMin

    def setLocation(self,xp,yp):
        dmin=1e6
        for iFacet in self.DicoVariablePSF.keys():
            d=np.sqrt((xp-self.DicoVariablePSF[iFacet]["pixCentral"][0])**2+(yp-self.DicoVariablePSF[iFacet]["pixCentral"][1])**2)
            if d<dmin:
                dmin=d
                self.iFacet=iFacet

    def setFacet(self,iFacet):
        self.iFacet=iFacet

    def GivePSF(self):
        return self.CubeVariablePSF[self.iFacet],self.CubeMeanVariablePSF[self.iFacet]



