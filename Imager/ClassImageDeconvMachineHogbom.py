"""
This minimal implementation of standard Hogbom CLEAN algorithm should serve
as a minimal reference interface of how to incorporate new deconvolution
algorithms into DDFacet.
"""

import numpy as np
import pylab
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.Other import ClassTimeIt
from pyrap.images import image
from ClassPSFServer import ClassPSFServer
import ClassModelMachine
from DDFacet.Other.progressbar import ProgressBar
import ClassGainMachine # Currently required by model machine but fixed to static mode


class ClassImageDeconvMachine():
    """
    Currently constructor inputs match those in MSMF, should figure out which are truly generic and put the rest in
    parset's MinorCycleConfig option.
    These methods may be called from ClassDeconvMachine
        Init(**kwargs) - contains minor cycle specific initialisations which are only used once
            Input: currently kwargs are minor cycle specific and should be set from ClassDeconvMachine but a
                     ideally a generic interface has these set in the parset somehow.
        Clean() - does joint deconvolution over all the channels/bands.
            Output: return_code - "MaxIter"????
                    continue - whether to continue the deconvolution
                    updated - whether the model has been updated
        GiveModelImage(freq) - returns current model at freq
            Input: freq - tuple of frequencies at which to return the model
            Output: Mod - the current model at freq
        Update(DicoDirty,**kwargs) - updates to minor cycle at the end of each major cycle
            Input:  DicoDirty - updated image dict at start of each major cycle
                    Use kwargs to pass any other minor cycle specific options
        ToFile(fname) - saves dico model to file
            Input: fname - the name of the file to write the dico image to
        FromFile(fname) - reads model dict from file
            Input: fname - the name of the file to write the dico image to
    """
    def __init__(self,Gain=0.3,
                 MaxMinorIter=100,NCPU=6,
                 CycleFactor=2.5,FluxThreshold=None,RMSFactor=3,PeakFactor=0,
                 GD=None,SearchMaxAbs=1,CleanMaskImage=None,
                 **kw    # absorb any unknown keywords arguments into this
                 ):
        self.ModelImage=None
        self.MaxMinorIter=MaxMinorIter
        self.NCPU=NCPU
        self.GD=GD
        self.MultiFreqMode = (self.GD["MultiFreqs"]["NFreqBands"] > 1)
        self.FluxThreshold = FluxThreshold 
        self.CycleFactor = CycleFactor
        self.RMSFactor = RMSFactor
        self.PeakFactor = PeakFactor
        self.GainMachine = ClassGainMachine.ClassGainMachine(GainMin=Gain)
        self.ModelMachine = ClassModelMachine.ClassModelMachine(self.GD,GainMachine=self.GainMachine)
        # reset overall iteration counter
        self._niter = 0
        if CleanMaskImage!=None:
            print>>log, "Reading mask image: %s"%CleanMaskImage
            MaskArray=image(CleanMaskImage).getdata()
            nch,npol,_,_=MaskArray.shape
            self._MaskArray=np.zeros(MaskArray.shape,np.bool8)
            for ch in range(nch):
                for pol in range(npol):
                    self._MaskArray[ch,pol,:,:]=np.bool8(1-MaskArray[ch,pol].T[::-1].copy())[:,:]
            self.MaskArray=self._MaskArray[0]


    def Init(self, **kwargs):
        self.SetPSF(kwargs["PSFVar"])
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])

    def GiveModelImage(self,*args): return self.ModelMachine.GiveModelImage(*args)

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe
        

    def SetPSF(self,DicoVariablePSF):
        self.PSFServer=ClassPSFServer(self.GD)
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF)
        self.DicoVariablePSF=DicoVariablePSF

    def SetDirty(self,DicoDirty):
        self.DicoDirty=DicoDirty
        self._Dirty = self.DicoDirty["ImagData"]
        self._MeanDirty = self.DicoDirty["MeanImage"]

        NPSF=self.PSFServer.NPSF
        _,_,NDirty,_=self._Dirty.shape

        off=(NPSF-NDirty)/2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)

        if self.ModelImage==None:
            self._ModelImage=np.zeros_like(self._Dirty)
        if self.MaskArray==None:
            self._MaskArray=np.zeros(self._Dirty.shape,dtype=np.bool8)

    def GiveEdges(self,(xc0,yc0),N0,(xc1,yc1),N1):
        """
        Each pixel in the image is associated with a different facet each with
        a different PSF.
        This finds the indices corresponding to the edges of a local psf centered
        on a specific pixel, here xc0,yc0.
        """
        M_xc=xc0
        M_yc=yc0
        NpixMain=N0
        F_xc=xc1
        F_yc=yc1
        NpixFacet=N1
                
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

        Aedge=[x0main,x1main,y0main,y1main]
        Bedge=[x0facet,x1facet,y0facet,y1facet]
        return Aedge,Bedge

    def SubStep(self,(dx,dy),LocalSM):
        """
        This is where subtraction in the image domain happens
        """
        npol,_,_=self.Dirty.shape
        x0,x1,y0,y1=self.DirtyExtent

        xc,yc=dx,dy
        N0=self.Dirty.shape[-1]
        N1=LocalSM.shape[-1]

        #Get overlap indices where psf should be subtracted
        Aedge,Bedge=self.GiveEdges((xc,yc),N0,(N1/2,N1/2),N1)

        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge

        #Subtract from each channel/band
        self._Dirty[:,:,x0d:x1d,y0d:y1d]-=LocalSM[:,:,x0p:x1p,y0p:y1p]
        #Subtract from the average
        if self.MultiFreqMode:  #If multiple frequencies are present construct the weighted mean
            W=np.float32(self.DicoDirty["WeightChansImages"])  #Get the weights
            self._MeanDirty[0,:,x0d:x1d,y0d:y1d]-=np.sum(LocalSM[:,:,x0p:x1p,y0p:y1p]*W.reshape((W.size,1,1,1)),axis=0) #Sum over frequency

    def setChannel(self,ch=0):
        """
        In case we ever want to deconvolve per channel.
        Currently just sets self.Dirty to average over freq bands.
        """
        #self.PSF=self._MeanPSF[ch]
        self.Dirty=self._MeanDirty[ch]
        self.ModelImage=self._ModelImage[ch]
        self.MaskArray=self._MaskArray[ch]


    def Clean(self,ch=0):
        """
        Runs minor cycle over image channel 'ch'.
        initMinor is number of minor iteration (keeps continuous count through major iterations)
        Nminor is max number of minor iterations

        Returns tuple of: return_code,continue,updated
        where return_code is a status string;
        continue is True if another cycle should be executed;
        update is True if model has been updated (note that update=False implies continue=False)
        """
        if self._niter >= self.MaxMinorIter:
            return "MaxIter", False, False

        #No need to set the channel when doing joint deconvolution
        self.setChannel(ch)

        _,npix,_=self.Dirty.shape
        xc=(npix)/2

        npol,_,_=self.Dirty.shape

        m0,m1=self.Dirty[0].min(),self.Dirty[0].max()

        #These options should probably be moved into MinorCycleConfig in parset
        DoAbs=int(self.GD["ImagerDeconv"]["SearchMaxAbs"])
        print>>log, "  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter,self.MaxMinorIter,DoAbs)

        ## Determine which stopping criterion to use for flux limit
        #Get RMS stopping criterion
        NPixStats = self.GD["ImagerDeconv"]["NumRMSSamples"]
        if NPixStats:
            RandomInd=np.int64(np.random.rand(NPixStats)*npix**2)
            RMS=np.std(np.real(self.Dirty.ravel()[RandomInd]))
        else:
            RMS=np.std(self.Dirty)
        self.RMS=RMS

        self.GainMachine.SetRMS(RMS)
        
        Fluxlimit_RMS = self.RMSFactor*RMS

        #Find position and intensity of first peak
        x,y,MaxDirty=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)
        #Get peak factor stopping criterion
        Fluxlimit_Peak = MaxDirty*self.PeakFactor

        #Get side lobe stopping criterion
        Fluxlimit_Sidelobe = ((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*MaxDirty if self.CycleFactor else 0

        mm0,mm1=self.Dirty.min(),self.Dirty.max()

        # Choose whichever threshold is highest
        StopFlux = max(Fluxlimit_Peak, Fluxlimit_RMS, Fluxlimit_Sidelobe, self.FluxThreshold)

        print>>log, "    Dirty image peak flux      = %10.6f Jy [(min, max) = (%.3g, %.3g) Jy]"%(MaxDirty,mm0,mm1)
        print>>log, "      RMS-based threshold      = %10.6f Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor)
        print>>log, "      Sidelobe-based threshold = %10.6f Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor)
        print>>log, "      Peak-based threshold     = %10.6f Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor)
        print>>log, "      Absolute threshold       = %10.6f Jy"%(self.FluxThreshold)
        print>>log, "    Stopping flux              = %10.6f Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty)
        
        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        ThisFlux=MaxDirty
        #print x,y

        if ThisFlux < StopFlux:
            print>>log, ModColor.Str("    Initial maximum peak %g Jy below threshold, we're done here" % (ThisFlux),col="green" )
            return "FluxThreshold", False, False

        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="Cleaning   ", HeaderSize=20,TitleSize=30)
        # pBAR.disable()

        self.GainMachine.SetFluxMax(ThisFlux)
        pBAR.render(0,"g=%3.3f"%self.GainMachine.GiveGain())

        def GivePercentDone(ThisMaxFlux):
            fracDone=1.-(ThisMaxFlux-StopFlux)/(MaxDirty-StopFlux)
            return max(int(round(100*fracDone)),100)

        #Do minor cycle deconvolution loop
        try:
            for i in range(self._niter+1,self.MaxMinorIter+1):
                self._niter = i

                x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)

                self.GainMachine.SetFluxMax(ThisFlux)

                T.timeit("max0")

                if ThisFlux <= StopFlux:
                    pBAR.render(100,"peak %.3g"%(ThisFlux,))
                    print>>log, ModColor.Str("    [iter=%i] peak of %.3g Jy lower than stopping flux" % (i,ThisFlux),col="green")
                    cont = ThisFlux > self.FluxThreshold
                    if not cont:
                          print>>log, ModColor.Str("    [iter=%i] absolute flux threshold of %.3g Jy has been reached" % (i,self.FluxThreshold),col="green",Bold=True)

                    return "MinFluxRms", cont, True    # stop deconvolution if hit absolute treshold; update model

                if (i>0)&((i%100)==0):
                    PercentDone=GivePercentDone(ThisFlux)                
                    pBAR.render(PercentDone,"peak %.3g i%d"%(ThisFlux,self._niter))

                nch,npol,_,_=self._Dirty.shape
                #I think Fpol contains the intensities at (x,y) per freq and polarisation
                Fpol=np.float32((self._Dirty[:,:,x,y].reshape((nch,npol,1,1))).copy())

                # dx=x-xc
                # dy=y-xc

                T.timeit("stuff")

                self.PSFServer.setLocation(x,y) #Selects the facet closest to (x,y)
                PSF = self.PSFServer.GivePSF()  #Get corresonding PSF


                T.timeit("FindScale")

                CurrentGain = self.GainMachine.GiveGain()
                #Subtract LocalSM*CurrentGain from dirty image
                self.SubStep((x,y),PSF*Fpol*CurrentGain)
                T.timeit("SubStep")

                T.timeit("End")

        except KeyboardInterrupt:
            print>>log, ModColor.Str("    [iter=%i] minor cycle interrupted with Ctrl+C, peak flux %.3g" % (self._niter, ThisFlux))
            return "MaxIter", False, True   # stop deconvolution but do update model

        print>>log, ModColor.Str("    [iter=%i] Reached maximum number of iterations, peak flux %.3g" % (self._niter, ThisFlux))

        return "MaxIter", False, True   # stop deconvolution but do update model

    def Update(self,DicoDirty,**kwargs):
        """
        Method to update attributes from ClassDeconvMachine
        """
        #Update image dict
        self.SetDirty(DicoDirty)

    def ToFile(self,fname):
        """
        Method to write model image to file
        """
        self.ModelMachine.ToFile(fname)


    def FromFile(self, fname):
        """
        Read model dict from file SubtractModel
        """
        self.ModelMachine.FromFile(fname)