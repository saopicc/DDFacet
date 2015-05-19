import numpy as np
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir.GiveEdges import GiveEdges


class ClassImToGrid():
    def __init__(self,
                 GridShape,
                 PaddingInnerCoord,
                 OverS,Padding,
                 dtype,
                 ifzfCF=None,Mode="Blender"):
        
        self.GridShape=GridShape
        self.PaddingInnerCoord=PaddingInnerCoord
        self.OverS=OverS
        self.Padding=Padding
        self.dtype=dtype
        self.ifzfCF=ifzfCF
        self.FFTWMachine=ModFFTW.FFTW_2Donly_np(self.GridShape,self.dtype, ncores = 1)
        self.Mode=Mode

    def setModelIm(self,ModelIm):
        _,_,n,n=ModelIm.shape
        x0,x1=self.PaddingInnerCoord
        ModelImPadded=np.zeros(self.GridShape,dtype=self.dtype)
        ModelImPadded[:,:,x0:x1,x0:x1]=ModelIm
        
        Grid=self.dtype(self.ImToGrid(ModelImPadded))
        return Grid

    def ImToGrid(self,ModelIm):
        
        nchan,npol,n,_=ModelIm.shape
        ModelImCorr=ModelIm*(self.OverS*n)**2

        if self.ifzfCF!=None:
            for ichan in range(nchan):
                for ipol in range(npol):
                    ModelImCorr[ichan,ipol][:,:]=ModelImCorr[ichan,ipol][:,:].real/self.ifzfCF


        ModelUVCorr=self.FFTWMachine.fft(ModelImCorr)

        return ModelUVCorr


    def GiveGridSharp(self,Image,DicoImager,iFacet):
        nch,npol,_,_=Image.shape
        _,_,NpixFacet,_=self.GridShape
        
        x0,x1,y0,y1=DicoImager[iFacet]["pixExtent"]
        #ModelIm=np.zeros((nch,npol,NpixFacet,NpixFacet),dtype=np.float32)
        x0p,x1p=self.PaddingInnerCoord
        ModelIm=np.zeros(self.GridShape,dtype=self.dtype)
        for ch in range(nch):
            for pol in range(npol):
                #ModelIm[ch,pol]=Image[ch,pol,x0:x1,y0:y1].T[::-1,:].real
                ModelIm[ch,pol,x0p:x1p,x0p:x1p]=Image[ch,pol,x0:x1,y0:y1].T[::-1,:].real
                ModelIm[ch,pol]/=self.ifzfCF
                SumFlux=np.sum(ModelIm)
                
        ModelIm*=(self.OverS*NpixFacet)**2

        Grid=self.FFTWMachine.fft(ModelIm)

        return Grid,SumFlux

    def GiveGridFader(self,Image,DicoImager,iFacet,NormIm):
        nch,npol,NPixOut,_=Image.shape
        _,_,N1,_=self.GridShape

        xc,yc=DicoImager[iFacet]["pixCentral"]
        Aedge,Bedge=GiveEdges((xc,yc),NPixOut,(N1/2,N1/2),N1)
        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge
        
        ModelIm=np.zeros((nch,npol,N1,N1),dtype=np.float32)
        for ch in range(nch):
            for pol in range(npol):
                ModelIm[ch,pol][x0p:x1p,y0p:y1p]=Image[ch,pol,x0d:x1d,y0d:y1d].T[::-1,:].real
                SumFlux=np.sum(ModelIm)
                ModelIm[ch,pol][x0p:x1p,y0p:y1p]/=NormIm[ch,pol,x0d:x1d,y0d:y1d].T[::-1,:].real
                
        ModelIm*=(self.OverS*N1)**2

        Grid=self.FFTWMachine.fft(ModelIm)

        return Grid,SumFlux

