import numpy as np
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir.GiveEdges import GiveEdges


class ClassImToGrid():
    def __init__(self,
                 GridShape=None,
                 PaddingInnerCoord=None,
                 OverS=None,Padding=None,
                 dtype=None,
                 ifzfCF=None,Mode="Blender",GD=None):
        
        self.GridShape=GridShape
        self.PaddingInnerCoord=PaddingInnerCoord
        self.OverS=OverS
        self.Padding=Padding
        self.dtype=dtype
        self.ifzfCF=ifzfCF
        self.GD=GD
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

        #print "xxA:",x0,x1
        #print "xxB:",x0p,x1p

        for ch in range(nch):
            for pol in range(npol):
                #ModelIm[ch,pol]=Image[ch,pol,x0:x1,y0:y1].T[::-1,:].real
                ModelIm[ch,pol,x0p:x1p,x0p:x1p]=Image[ch,pol,x0:x1,y0:y1].T[::-1,:].real
                ModelIm[ch,pol]/=self.ifzfCF
                SumFlux=np.sum(ModelIm)
                
        #print iFacet,np.max(ModelIm)
        #return ModelIm, None
        ModelIm*=(self.OverS*NpixFacet)**2

        Grid=self.FFTWMachine.fft(ModelIm)

        return Grid,SumFlux

    def GiveGridFader(self,Image,DicoImager,iFacet,NormIm):
        nch,npol,NPixOut,_=Image.shape
        _,_,N1,_=self.GridShape

        xc,yc=DicoImager[iFacet]["pixCentral"]
        #x0,x1,y0,y1=DicoImager[iFacet]["pixExtent"]
        #xc,yc=(x0+x1)/2,(y0+y1)/2

        Aedge,Bedge=GiveEdges((xc,yc),NPixOut,(N1/2,N1/2),N1)
        #Bedge,Aedge=GiveEdges((N1/2,N1/2),N1,(yc,xc),NPixOut)
        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge
        #print "xxA:",x0d,x1d
        #print "xxB:",x0p,x1p
        
        ModelIm=np.zeros((nch,npol,N1,N1),dtype=np.float32)
        for ch in range(nch):
            for pol in range(npol):
                #ModelIm[ch,pol][x0p:x1p,y0p:y1p]=Image[ch,pol].T[::-1,:].real[x0d:x1d,y0d:y1d]
                #ModelIm[ch,pol][x0p:x1p,y0p:y1p]=Image[ch,pol].real[x0d:x1d,y0d:y1d]
                ModelIm[ch,pol][x0p:x1p,y0p:y1p]=Image[ch,pol][x0d:x1d,y0d:y1d].real
                ModelIm[ch,pol][x0p:x1p,y0p:y1p]/=NormIm[x0d:x1d,y0d:y1d].real
                #ModelIm[ch,pol][x0p:x1p,y0p:y1p]/=NormIm[x0d:x1d,y0d:y1d].real
                ModelIm[ch,pol]=ModelIm[ch,pol].T[::-1,:]
                SumFlux=np.sum(ModelIm)

        #print iFacet,np.max(ModelIm)

        #return ModelIm, None

        ModelIm*=(self.OverS*N1)**2
        Grid=np.complex64(self.FFTWMachine.fft(np.complex64(ModelIm)))

        return Grid,SumFlux

    def GiveModelTessel(self,Image,DicoImager,iFacet,NormIm,Sphe,SpacialWeight):
        nch,npol,NPixOut,_=Image.shape
        N1=DicoImager[iFacet]["NpixFacetPadded"]
        N1NonPadded=DicoImager[iFacet]["NpixFacetPadded"]
        dx=(N1-N1NonPadded)/2

        xc,yc=DicoImager[iFacet]["pixCentral"]
        #x0,x1,y0,y1=DicoImager[iFacet]["pixExtent"]
        #xc,yc=(x0+x1)/2,(y0+y1)/2

        Aedge,Bedge=GiveEdges((xc,yc),NPixOut,(N1/2,N1/2),N1)
        #Bedge,Aedge=GiveEdges((N1/2,N1/2),N1,(yc,xc),NPixOut)
        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge
        #print "xxA:",x0d,x1d
        #print "xxB:",x0p,x1p
        
        ModelIm=np.zeros((nch,npol,N1,N1),dtype=np.float32)
        for ch in range(nch):
            for pol in range(npol):
                #ModelIm[ch,pol][x0p:x1p,y0p:y1p]=Image[ch,pol].T[::-1,:].real[x0d:x1d,y0d:y1d]
                #ModelIm[ch,pol][x0p:x1p,y0p:y1p]=Image[ch,pol].real[x0d:x1d,y0d:y1d]
                ModelIm[ch,pol][x0p:x1p,y0p:y1p]=Image[ch,pol][x0d:x1d,y0d:y1d].real
                M=ModelIm[ch,pol][dx:dx+N1NonPadded+1,dx:dx+N1NonPadded+1].copy()
                ModelIm[ch,pol].fill(0)
                ModelIm[ch,pol][dx:dx+N1NonPadded+1,dx:dx+N1NonPadded+1]=M[:,:]
                #ind =np.where(np.abs(ModelIm)==np.max(np.abs(ModelIm)))
                ModelIm[ch,pol][x0p:x1p,y0p:y1p]/=NormIm[x0d:x1d,y0d:y1d].real
                ModelIm[ch,pol][x0p:x1p,y0p:y1p]*=SpacialWeight[x0p:x1p,y0p:y1p]
                SumFlux=np.sum(ModelIm)
                ModelIm[ch,pol][x0p:x1p,y0p:y1p]/=Sphe[x0p:x1p,y0p:y1p].real
                ModelIm[ch,pol][Sphe<1e-3]=0
                ModelIm[ch,pol]=ModelIm[ch,pol].T[::-1,:]

        #print iFacet,DicoImager[iFacet]["l0m0"],DicoImager[iFacet]["NpixFacet"],DicoImager[iFacet]["NpixFacetPadded"],SumFlux
        #if np.max(np.abs(ModelIm))>1: print ind
        
        #if np.abs(SumFlux)>1: stop
        
       # #print iFacet,np.max(ModelIm)

        # #return ModelIm, None
        # #Padding=self.GD["ImagerMainFacet"]["Padding"]

        ModelIm*=(self.OverS*N1)**2
        return ModelIm,SumFlux
        # Grid=np.complex64(self.FFTWMachine.fft(np.complex64(ModelIm)))

        return Grid,SumFlux


