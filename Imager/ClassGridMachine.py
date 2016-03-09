import numpy as np
import DDFacet.cbuild.Gridder._pyGridder as _pyGridder
import pylab
from pyrap.images import image
import MyPickle
import os
import MyLogger
log=MyLogger.getLogger("ClassImager")
import ClassTimeIt
import ModCF
import ModToolBox

import ModColor
import ClassCasaImage


class ClassGridMachine():
    def __init__(self,Npix=512,Cell=10.,Support=11,ChanFreq=np.array([6.23047e7],dtype=np.float64),
                 wmax=10000,Nw=11,DoPSF=True,
                 TransfRaDec=None,ImageName="Image",OverS=5,
                 Padding=1.5,WProj=False,lmShift=None,Precision="S"):
        

        if DoPSF:
            self.DoPSF=True
            Npix=Npix*2
        if TransfRaDec!=None:
            self.radec0,self.radec1=TransfRaDec

        if Precision=="S":
            self.dtype=np.complex64
        elif Precision=="D":
            self.dtype=np.complex128

        self.ImageName=ImageName
        
        self.Padding=Padding
        self.NonPaddedNpix,Npix=EstimateNpix(Npix,Padding)

        
        self.Npix=Npix
        self.NonPaddedShape=(1,4,self.NonPaddedNpix,self.NonPaddedNpix)
        self.GridShape=(1,4,self.Npix,self.Npix)
        x0=(self.Npix-self.NonPaddedNpix)/2
        self.PaddingInnerCoord=(x0,x0+self.NonPaddedNpix)
        #self.ModelIm=np.zeros(self.GridShape,dtype=np.complex128)

        #self.FFTWMachine=ModToolBox.FFTM2D(self.Grid)
        import ModFFTW
        #self.FFTWMachine=ModFFTW.FFTWnp(self.Grid, ncores = 1)
        Grid=np.zeros(self.GridShape,dtype=self.dtype)

        self.FFTWMachine=ModFFTW.FFTW(Grid, ncores = 1)
        self.FFTWMachine=ModFFTW.FFTW_2Donly(Grid, ncores = 1)
        #self.FFTWMachine=ModFFTW.FFTWnp(Grid, ncores = 1)
        
        T=ClassTimeIt.ClassTimeIt("ClassImager")

        self.DoPSF=DoPSF
        self.Cell=Cell
        self.incr=(np.array([-Cell,Cell],dtype=np.float64)/3600.)*(np.pi/180)
        #CF.fill(1.)
        ChanFreq=ChanFreq.flatten()
        self.ChanFreq=ChanFreq
        self.ChanWave=2.99792458e8/self.ChanFreq
        self.UVNorm=2.*1j*np.pi/self.ChanWave
        self.Sup=Support
        
        if WProj:
            self.WTerm=ModCF.ClassWTermModified(Cell=Cell,Sup=Support,Npix=Npix,Freqs=ChanFreq,wmax=wmax,Nw=Nw,OverS=OverS,lmShift=lmShift)
            #self.WTerm=ModCF.ClassWTerm(Cell=Cell,Sup=Support,Npix=Npix,Freqs=ChanFreq,wmax=wmax,Nw=Nw,OverS=OverS,lmShift=lmShift)
        else:
            self.WTerm=ModCF.ClassSTerm(Cell=Cell,Sup=Support,Npix=Npix,Freqs=ChanFreq,wmax=wmax,Nw=Nw,OverS=OverS)
        T.timeit("Wterm")
        #self.WTerm.plot()
        self.CF, self.fCF, self.ifzfCF= self.WTerm.CF, self.WTerm.fCF, self.WTerm.ifzfCF

        T.timeit("Rest")

        self.reinitGrid()
        self.CasaImage=None
        # if TransfRaDec!=None:
        #     self.setCasaImage()
        self.lmShift=lmShift

    def ShiftVis(self,uvw,vis,reverse=False,lmShift=None):
        if self.lmShift==None: return vis
        l0,m0=self.lmShift
        if lmShift!=None:
            l0,m0=lmShift
        u,v,w=uvw.T
        n0=np.sqrt(1-l0**2-m0**2)-1
        if reverse: 
            corr=np.exp(-self.UVNorm*(u*l0+v*m0+w*n0))
            u+=w*self.WTerm.Cu
            v+=w*self.WTerm.Cv
        else:
            corr=np.exp(self.UVNorm*(u*l0+v*m0+w*n0))
            u+=w*self.WTerm.Cu
            v+=w*self.WTerm.Cv

        corr=corr.reshape((corr.size,1,1))
        vis*=corr
        uvw=np.array((u,v,w)).T.copy()
        return uvw,vis


    def setCasaImage(self):
        self.CasaImage=ClassCasaImage.ClassCasaimage(self.ImageName,self.NonPaddedNpix,self.Cell,self.radec1)

    def reinitGrid(self):
        #self.Grid.fill(0)
        self.NChan, self.npol, _,_=self.GridShape
        self.SumWeigths=np.zeros((self.NChan,self.npol),np.float64)

    def put(self,uvw,visIn,flag,doStack=False,lmShift=None):
        log=MyLogger.getLogger("ClassImager.addChunk")
        vis=visIn#.copy()

        uvw,vis=self.ShiftVis(uvw,vis,reverse=True,lmShift=lmShift)
        # uvw,vis=self.RotateVis(uvw,vis,reverse=True)

        dummy=np.abs(vis).astype(np.float32)
        if not(doStack):
            self.reinitGrid()

        npol=self.npol
        NChan=self.NChan
        SumWeigths=self.SumWeigths
        if vis.shape!=flag.shape:
            raise Exception('vis[%s] and flag[%s] should have the same shape'%(str(vis.shape),str(flag.shape)))
        
        u,v,w=uvw.T
        vis[u==0,:,:]=0
        flag[u==0,:,:]=True
        if self.DoPSF:
            vis.fill(0)
            vis[:,:,0]=1
            vis[:,:,3]=1

        Grid=np.zeros(self.GridShape,dtype=self.dtype)

        Grid=_pyGridder.pyGridderWPol(Grid,
                                      vis[:,:,:].astype(np.complex128),
                                      uvw.astype(np.float64),
                                      flag[:,:,:].astype(np.int32),
                                      dummy.astype(np.float64),
                                      SumWeigths,#dummy.astype(np.float64),
                                      0,
                                      self.WTerm.Wplanes,
                                      self.WTerm.WplanesConj,
                                      np.array([self.WTerm.RefWave,self.WTerm.wmax,len(self.WTerm.Wplanes),self.WTerm.OverS],dtype=np.float64),
                                      self.incr.astype(np.float64),
                                      self.ChanFreq.astype(np.float64),
                                      [np.array([0,1,2,3],np.int32)])

        ImPadded= self.GridToIm(Grid)
        Dirty = self.cutImPadded(ImPadded)
        return Dirty

    def setModelIm(self,ModelIm):
        _,_,n,n=ModelIm.shape
        x0,x1=self.PaddingInnerCoord
        # self.ModelIm[:,:,x0:x1,x0:x1]=ModelIm
        ModelImPadded=np.zeros(self.GridShape,dtype=self.dtype)
        ModelImPadded[:,:,x0:x1,x0:x1]=ModelIm
        Grid=self.ImToGrid(ModelImPadded)*n**2
        return Grid

    def cutImPadded(self,Dirty):
        x0,x1=self.PaddingInnerCoord
        Dirty=Dirty[:,:,x0:x1,x0:x1]
        # if self.CasaImage!=None:
        #     self.CasaImage.im.putdata(Dirty[0,0].real)
        return Dirty
        

    def getDirtyIm(self):
        Dirty= self.GridToIm()
        x0,x1=self.PaddingInnerCoord
        Dirty=Dirty[:,:,x0:x1,x0:x1]
        # if self.CasaImage!=None:
        #     self.CasaImage.im.putdata(Dirty[0,0].real)
        return Dirty

    def get(self,uvw,visIn,flag,doStack=False):
        log=MyLogger.getLogger("ClassImager.addChunk")
        vis=visIn#.copy()

        dummy=np.abs(vis).astype(np.float32)
        if not(doStack):
            self.reinitGrid()

        npol=self.npol
        NChan=self.NChan
        SumWeigths=self.SumWeigths
        if vis.shape!=flag.shape:
            raise Exception('vis[%s] and flag[%s] should have the same shape'%(str(vis.shape),str(flag.shape)))

        
        u,v,w=uvw.T
        vis[u==0,:,:]=0
        flag[u==0,:,:]=True
      
        uvwOrig=uvw.copy()
        uvw,vis=self.ShiftVis(uvw,vis,reverse=False)
        vis.fill(0)
        
        vis = _pyGridder.pyDeGridderWPol(self.Grid.astype(np.complex128),
                                        vis[:,:,:].astype(np.complex128),
                                        uvw.astype(np.float64),
                                        flag[:,:,:].astype(np.int32),
                                        dummy.astype(np.float64),
                                        SumWeigths,
                                        0,
                                        self.WTerm.WplanesConj,
                                        self.WTerm.Wplanes,
                                        np.array([self.WTerm.RefWave,self.WTerm.wmax,len(self.WTerm.Wplanes),self.WTerm.OverS],dtype=np.float64),
                                        self.incr.astype(np.float64),
                                        self.ChanFreq.astype(np.float64),
                                        [np.array([0,1,2,3],np.int32)])

        uvw,vis=self.ShiftVis(uvwOrig,vis,reverse=False)
        
        return vis

    def GridToIm(self,Grid):
        log=MyLogger.getLogger("ClassImager.GridToIm")

        npol=self.npol
        import ClassTimeIt
        T=ClassTimeIt.ClassTimeIt()

        GridCorr=Grid/self.SumWeigths.reshape((self.NChan,npol,1,1))
        GridCorr*=(self.WTerm.OverS)**2
        T.timeit("norm")
        Dirty=self.FFTWMachine.ifft(GridCorr)
        T.timeit("fft")
        nchan,npol,_,_=Grid.shape
        for ichan in range(nchan):
            for ipol in range(npol):
                Dirty[ichan,ipol][:,:]=Dirty[ichan,ipol][:,:].real/self.ifzfCF
        T.timeit("sphenorm")

        return Dirty

        # self.GridCorr=self.Grid/self.SumWeigths.reshape((self.NChan,npol,1,1))
        # self.GridCorr*=(self.WTerm.OverS)**2

        # if self.DoPSF:
        #     self.PSF=np.real(fft2(self.PSF_Grid[0,0])/self.PSF_ifzfCF)
        #     self.PSF/=self.NVis
        # #self.PSF*=(self.PSF_WTerm.OverS*self.Sup)**2
        #     self.PSF*=self.PSF_Grid[0,0].size
        #     self.NormPSF=np.max(np.real(self.PSF))
        #     self.PSF/=self.NormPSF
        #     print>>log, "PSF   image (min,max)=(%f, %f)"%(np.min(np.real(self.PSF)),np.max(np.real(self.PSF)))


        # #self.Dirty=np.zeros(self.Grid.shape,float)
        # T=ClassTimeIt.ClassTimeIt("ClassImager.GiveDirty")
        # nchan,npol,_,_=self.Grid.shape
        # #rep=ifft2(self.Grid[ichan,ipol])
        # self.Dirty=self.FFTWMachine.ifft(self.GridCorr)

        # T.timeit("Dirty (fft)")
        # for ichan in range(nchan):
        #     for ipol in range(npol):
        #         self.Dirty[ichan,ipol][:,:]=self.Dirty[ichan,ipol][:,:].real/self.ifzfCF
        # T.timeit("Dirty (norm)")

        # return self.Dirty
        
    def ImToGrid(self,ModelIm):
        log=MyLogger.getLogger("ClassImager.ImToGrid")
        #print>>log, "Compute dirty image from grid..."
        
        npol=self.npol
        ModelImCorr=ModelIm*(self.WTerm.OverS*self.Padding)**2

        nchan,npol,_,_=ModelImCorr.shape
        for ichan in range(nchan):
            for ipol in range(npol):
                ModelImCorr[ichan,ipol][:,:]=ModelImCorr[ichan,ipol][:,:].real/self.ifzfCF


        ModelUVCorr=self.FFTWMachine.fft(ModelImCorr)

        return ModelUVCorr
        

