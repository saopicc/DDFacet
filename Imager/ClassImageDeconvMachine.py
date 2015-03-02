
import numpy as np
import pylab
import MyLogger
log=MyLogger.getLogger(" ClassImageDeconvMachine")

class ClassImageDeconvMachine():
    def __init__(self,Gain=0.3,MaxMinorIter=20):
        #self.im=CasaImage
        self.Gain=Gain
        self.ModelImage=None
        self.MaxMinorIter=MaxMinorIter
        


    def SetDirtyPSF(self,Dirty,PSF):
        # if len(PSF.shape)==4:
        #     self.PSF=PSF[0,0]
        # else:
        #     self.PSF=PSF
        self._PSF=PSF
        self._Dirty=Dirty
        _,_,NPSF,_=PSF.shape
        _,_,NDirty,_=Dirty.shape
        off=(NPSF-NDirty)/2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)

        if self.ModelImage==None:
            self._ModelImage=np.zeros_like(self._Dirty)

        
    def GivePSF(self,(dx,dy)):
        npol,_,_=self.Dirty.shape
        ThisPSF=np.zeros_like(self.Dirty)
        x0,x1,y0,y1=self.DirtyExtent
        for pol in range(npol):
            ThisSlice=np.roll(np.roll(self.PSF[pol].copy(),dx,axis=0),dy,axis=1)
            ThisPSF[pol]=ThisSlice[x0:x1,y0:y1]
            
            # pylab.clf()
            # pylab.subplot(1,2,1)
            # pylab.imshow(self.PSF[pol],interpolation="nearest",vmin=0,vmax=1)
            # pylab.subplot(1,2,2)
            # pylab.imshow(ThisSlice,interpolation="nearest",vmin=0,vmax=1)
            # pylab.draw()
            
            # pylab.show(False)
            # pylab.pause(0.1)
            # stop

        return ThisPSF

    def setChannel(self,ch=0):
        self.PSF=self._PSF[ch]
        self.Dirty=self._Dirty[ch]
        self.ModelImage=self._ModelImage[ch]

    def Clean(self,Nminor=None,ch=0):
        if Nminor==None:
            Nminor=self.MaxMinorIter

        self.setChannel(ch)

        _,npix,_=self.Dirty.shape
        xc=(npix)/2

        npol,_,_=self.Dirty.shape

        m0,m1=self.Dirty[0].min(),self.Dirty[0].max()
        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.imshow(self.Dirty[0],interpolation="nearest",vmin=m0,vmax=m1)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        print>>log, "Running minor cycle with Nminor=%i"%Nminor

        NPixStats=1000
        RandomInd=np.int64(np.random.rand(NPixStats)*npix**2)
        RMS=np.std(np.real(self.Dirty.ravel()[RandomInd]))
        print>>log, "    Estimated RMS = %f Jy"%RMS
        Threshold_RMS=5.
        FluxLimit=Threshold_RMS*RMS

        for i in range(Nminor):
            ThisFlux=np.max(np.abs(self.Dirty))

            if ThisFlux < FluxLimit:
                print>>log, "    Maximum peak lower that limit of %f Jy" % FluxLimit
                return "MinFlux"

            _,x,y=np.where(np.abs(self.Dirty)==ThisFlux)
            Fpol=self.Dirty[:,x,y].reshape(npol,1,1)
            dx=x[0]-xc
            dy=y[0]-xc
            # print dx,dy

            PSF=self.GivePSF((dx,dy))
            self.Dirty-=PSF*(Fpol*self.Gain)

            pylab.clf()
            pylab.subplot(1,2,1)
            pylab.imshow(self.Dirty[0],interpolation="nearest",vmin=m0,vmax=m1)
            pylab.subplot(1,2,2)
            pylab.imshow(PSF[0],interpolation="nearest",vmin=0,vmax=1)
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)

            for pol in range(npol):
                self.ModelImage[pol,x[0],y[0]]+=Fpol[pol,0,0]*self.Gain


        return "MaxIter"
            # corr=np.sqrt(1.-(self.incr*dx)**2-(self.incr*dy)**2)
            # print>>log, corr
            # dec,ra= self.im.toworld((x[0],y[0]))
            # ra*=np.pi/(180.*60.)
            # dec*=np.pi/(180.*60.)
            # from rad2hmsdms import rad2hmsdms
            # strRa=rad2hmsdms(ra,Type="ra").replace(" ",":")
            # strDec=rad2hmsdms(dec,Type="dec").replace(" ",".")
            # print>>log, "(ra, dec, flux)=(%s, %s, %8.1f)"%(strRa,strDec,ThisFlux)

