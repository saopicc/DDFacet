
import numpy as np
import pylab
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger(" ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.Other import ClassTimeIt
import ClassGiveVariablePSF

class ClassImageDeconvMachine():
    def __init__(self,Gain=0.3,MaxMinorIter=100,NCPU=6,CycleFactor=2.5,GD=None):
        #self.im=CasaImage
        self.Gain=Gain
        self.ModelImage=None
        self.MaxMinorIter=MaxMinorIter
        self.NCPU=NCPU
        self.CycleFactor=CycleFactor
        self.GD=GD
        self.PSFNormFactor=1.

    def setPSFMachine(self, dt,dnu,freqs,cellsize, psf, npix):
        #self.PSFMachine=PSFMachine(dt,dnu,freqs)
        self.PSFMachine = ClassGiveVariablePSF.ClassVariablePSF(phasePSF=psf[0,0], freq0=freqs, dtime=dt, dfreq=dnu, cellsize=cellsize,npix=npix)

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

        
    # def GivePSF(self,(dx,dy)):
    #     npol,_,_=self.Dirty.shape
    #     ThisPSF=np.zeros_like(self.Dirty)
    #     x0,x1,y0,y1=self.DirtyExtent
    #     for pol in range(npol):
    #         ThisSlice=np.roll(np.roll(self.PSF[pol].copy(),dx,axis=0),dy,axis=1)
    #         ThisPSF[pol]=ThisSlice[x0:x1,y0:y1]
            
    #         # pylab.clf()
    #         # pylab.subplot(1,2,1)
    #         # pylab.imshow(self.PSF[pol],interpolation="nearest",vmin=0,vmax=1)
    #         # pylab.subplot(1,2,2)
    #         # pylab.imshow(ThisSlice,interpolation="nearest",vmin=0,vmax=1)
    #         # pylab.draw()
            
    #         # pylab.show(False)
    #         # pylab.pause(0.1)
    #         # stop

    #     return ThisPSF


    def SubStep(self,(dx,dy),Fpol):
        npol,_,_=self.Dirty.shape
        x0,x1,y0,y1=self.DirtyExtent

        xc,yc=dx,dy
        NpixFacet=self.PSF.shape[1]

        M_xc=xc
        M_yc=yc
        NpixMain=self.Dirty.shape[1]
        F_xc=NpixFacet/2
        F_yc=NpixFacet/2
                
        ## X
        M_x0=M_xc-NpixFacet/2
        x0main=np.max([0,M_x0])
        dx0=x0main-M_x0
        x0facet=dx0
                
        M_x1=M_xc+NpixFacet/2
        x1main=np.min([NpixMain-1,M_x1])
        dx1=M_x1-x1main
        x1facet=NpixFacet-dx1
        x1main+=1
        ## Y
        M_y0=M_yc-NpixFacet/2
        y0main=np.max([0,M_y0])
        dy0=y0main-M_y0
        y0facet=dy0
        
        M_y1=M_yc+NpixFacet/2
        y1main=np.min([NpixMain-1,M_y1])
        dy1=M_y1-y1main
        y1facet=NpixFacet-dy1
        y1main+=1

        # self.Dirty[:,x0main:x1main,y0main:y1main]-=self.PSF[:,x0facet:x1facet,y0facet:y1facet]*(Fpol*self.Gain)
        Aedge=[x0main,x1main,y0main,y1main]
        Bedge=[x0facet,x1facet,y0facet,y1facet]
        print "***************hereree**********",self.PSF.shape
        print"************************************ debut subset here**********************************************"
        #self.PSF = (np.float32(self.PSFMachine.givePSF(xc,yc).copy()))[np.newaxis,:,:]
        print"************************************ after subset here**********************************************"
        _,n,n=self.PSF.shape

        PSF=(np.float32(self.PSFMachine.givePSF(xc,yc).copy()))#[np.newaxis,:,:]

        #PSF=self.PSF.reshape((n,n))
        
        
        picmax=np.max(PSF)#[n/2,n/2]
        
        self.PSFNormFactor=1./(picmax)
        factor=-Fpol[0,0,0]*self.Gain*self.PSFNormFactor#(1./picmax)
        
    
        
        print "pic", picmax,self.PSFNormFactor
        #stop
        #print>>log, "    Removing %f Jy at (%i %i) (peak of %f Jy)"%(Fpol[0,0,0]*self.Gain,dx,dy,Fpol[0,0,0])

        NpParallel.A_add_B_prod_factor(self.Dirty,PSF,Aedge,Bedge,factor=float(factor),NCPU=self.NCPU)

    def setChannel(self,ch=0):
        self.PSF=self._PSF[ch]
        self.Dirty=self._Dirty[ch]
        self.ModelImage=self._ModelImage[ch]

    def setSideLobeLevel(self,SideLobeLevel):
        self.SideLobeLevel=SideLobeLevel

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

        print>>log, "  Running minor cycle [MaxMinorIter = %i, CycleFactor=%3.1f]"%(Nminor,self.CycleFactor)

        NPixStats=1000
        RandomInd=np.int64(np.random.rand(NPixStats)*npix**2)
        RMS=np.std(np.real(self.Dirty.ravel()[RandomInd]))
        
        Threshold_RMS=5./(1.-self.SideLobeLevel)
        MaxDirty=np.max(np.abs(self.Dirty))
        FluxLimit=Threshold_RMS*RMS
        #FluxLimit_SideLobe=MaxDirty*(1.-self.SideLobeLevel)
        Threshold_SideLobe=self.CycleFactor*MaxDirty*(self.SideLobeLevel)

        mm0,mm1=self.Dirty.min(),self.Dirty.max()
        print>>log, "    Dirty image peak flux   = %7.3f Jy [(min, max) = (%7.3f, %7.3f) Jy]"%(MaxDirty,mm0,mm1)
        print>>log, "    RMS threshold flux      = %7.3f Jy [rms      = %7.3f Jy]"%(FluxLimit, RMS)
        print>>log, "    Sidelobs threshold flux = %7.3f Jy [sidelobe = %7.3f of peak]"%(Threshold_SideLobe,self.SideLobeLevel)

        MaxModelInit=np.max(np.abs(self.ModelImage))

        
        # Fact=4
        # self.BookKeepShape=(npix/Fact,npix/Fact)
        # BookKeep=np.zeros(self.BookKeepShape,np.float32)
        # NPixBook,_=self.BookKeepShape
        # FactorBook=float(NPixBook)/npix
        
        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=1)
        if ThisFlux < FluxLimit:
            print>>log, ModColor.Str("    Initial maximum peak %f Jy lower that rms-based limit of %f Jy (%i-sigma)" % (ThisFlux,Threshold_RMS,Threshold_RMS))
            return "DoneMinFlux"


        for i in range(Nminor):
            #print>>log, "   "
            #print>>log, "   - Iteration %i"%i
            
            # ThisFlux=np.max(np.abs(self.Dirty))
            # T.timeit("max0")
            # MaxModelNow=np.max(np.abs(self.ModelImage))
            # T.timeit("max1")
            # _,x,y=np.where(np.abs(self.Dirty)==ThisFlux)
            # x=x[0]
            # y=y[0]
            # T.timeit("where")

            x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=1)
            #print>>log, "    Maximum peak %f Jy at (%i %i)" % (ThisFlux,x,y)

            #_,_,MaxModelNow=NpParallel.A_whereMax(BookKeep,NCPU=self.NCPU,DoAbs=1)
            #MaxModelNow=np.max(np.abs(MaxModelNow))
            T.timeit("max0")

            if ThisFlux < FluxLimit:
                print>>log, "    [iter=%i] Maximum peak lower that rms-based limit of %f Jy (%i-sigma)" % (i,FluxLimit,Threshold_RMS)
                return "MinFlux"

            #MaxCleaned=MaxModelNow#-MaxModelInit

            #print>>log, "        Iteration %i maximum cleaned flux = %f Jy"%(i,MaxCleaned)
            #if MaxCleaned > FluxLimit_SideLobe:
            if ThisFlux < Threshold_SideLobe:
                print>>log, "    [iter=%i] Peak residual flux %f Jy higher than sidelobe-based limit of %f Jy" % (i,ThisFlux, Threshold_SideLobe)

                return "MinFlux"

            Fpol=self.Dirty[:,x,y].reshape(npol,1,1)
            dx=x-xc
            dy=y-xc
            # print dx,dy

            # PSF=self.GivePSF((dx,dy))
            # self.Dirty-=PSF*(Fpol*self.Gain)

            
            # xbook=np.int64(x*FactorBook)
            # ybook=np.int64(y*FactorBook)
            # BookKeep[xbook,ybook]+=Fpol[0,0,0]*self.Gain

            T.timeit("stuff")
            self.SubStep((x,y),Fpol)
            T.timeit("add0")


            for pol in range(npol):
                self.ModelImage[pol,x,y]+=Fpol[pol,0,0]*self.Gain*self.PSFNormFactor
            
            # pylab.clf()
            # pylab.subplot(1,2,1)
            # pylab.imshow(self.Dirty[0],interpolation="nearest")#,vmin=m0,vmax=m1)
            # pylab.colorbar()
            # pylab.subplot(1,2,2)
            # pylab.imshow(self.ModelImage[0],interpolation="nearest")#,vmin=0,vmax=1)
            # pylab.colorbar()
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)

            T.timeit("add1")



        print>>log, ModColor.Str("    [iter=%i] Reached maximum number of iterations" % (Nminor))
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

