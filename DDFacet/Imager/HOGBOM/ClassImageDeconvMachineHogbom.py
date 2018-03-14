'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

"""
This minimal implementation of standard Hogbom CLEAN algorithm should serve
as a minimal reference interface of how to incorporate new deconvolution
algorithms into DDFacet.
"""

import numpy as np
import numexpr
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.Other import ClassTimeIt
from pyrap.images import image
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Imager import ClassGainMachine # Currently required by model machine but fixed to static mode
from DDFacet.ToolsDir import GiveEdges


class ClassImageDeconvMachine():
    """
    Currently constructor inputs match those in MSMF, should figure out which are truly generic and put the rest in
    parset's MinorCycleConfig option.
    These methods may be called from ClassDeconvMachine
        Init(**kwargs) - contains minor cycle specific initialisations which are only used once
            Input: currently kwargs are minor cycle specific and should be set from ClassDeconvMachine but a
                     ideally a generic interface has these set in the parset somehow.
        Deconvolve() - does joint deconvolution over all the channels/bands.
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
                 GD=None,SearchMaxAbs=1,CleanMaskImage=None,ImagePolDescriptor=["I"],ModelMachine=None,
                 **kw    # absorb any unknown keywords arguments into this
                 ):
        self.SearchMaxAbs = SearchMaxAbs
        self.ModelImage=None
        self.MaxMinorIter=MaxMinorIter
        self.NCPU=NCPU
        self.MaskArray = None
        self.GD=GD
        self.MultiFreqMode = (self.GD["Freq"]["NBand"] > 1)
        self.NFreqBand = self.GD["Freq"]["NBand"]
        self.FluxThreshold = FluxThreshold 
        self.CycleFactor = CycleFactor
        self.RMSFactor = RMSFactor
        self.PeakFactor = PeakFactor
        self.GainMachine = ClassGainMachine.ClassGainMachine(GainMin=Gain)
        if ModelMachine is None:
            import ClassModelMachineHogbom as ClassModelMachine
            self.ModelMachine = ClassModelMachine.ClassModelMachine(self.GD, GainMachine=self.GainMachine)
        else:
            self.ModelMachine = ModelMachine
        self.GainMachine = self.ModelMachine.GainMachine
        self.GiveEdges = GiveEdges.GiveEdges
        self._niter = 0
        if CleanMaskImage is not None:
            print>>log, "Reading mask image: %s"%CleanMaskImage
            MaskArray=image(CleanMaskImage).getdata()
            nch,npol,_,_=MaskArray.shape
            self._MaskArray=np.zeros(MaskArray.shape,np.bool8)
            for ch in range(nch):
                for pol in range(npol):
                    self._MaskArray[ch,pol,:,:]=np.bool8(1-MaskArray[ch,pol].T[::-1].copy())[:,:]
            self.MaskArray=self._MaskArray[0]
        self._peakMode = "normal"

        self.CurrentNegMask = None
        self._NoiseMap = None
        self._PNRStop = None  # in _peakMode "sigma", provides addiitonal stopping criterion


    def Init(self, **kwargs):
        self.SetPSF(kwargs["PSFVar"])
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
        self.SetModelRefFreq(kwargs["RefFreq"])
        self.ModelMachine.setFreqMachine(kwargs["GridFreqs"], kwargs["DegridFreqs"])
        self.Freqs = kwargs["GridFreqs"]


    def Reset(self):
        pass

    def setMaskMachine(self,MaskMachine):
        self.MaskMachine=MaskMachine


    def SetModelRefFreq(self, RefFreq):
        """
        Sets ref freq in ModelMachine.
        """
        AllFreqs = []
        AllFreqsMean = np.zeros((self.NFreqBand,), np.float32)
        for iChannel in range(self.NFreqBand):
            AllFreqs += self.DicoVariablePSF["freqs"][iChannel]
            AllFreqsMean[iChannel] = np.mean(self.DicoVariablePSF["freqs"][iChannel])
        #assume that the frequency variance is somewhat the same in all the stokes images:
        #RefFreq = np.sum(AllFreqsMean.ravel() * np.mean(self.DicoVariablePSF["WeightChansImages"],axis=1).ravel())
        self.ModelMachine.setRefFreq(RefFreq)


    def SetModelShape(self):
        """
        Sets the shape params of model, call in every update step
        """
        self.ModelMachine.setModelShape(self._Dirty.shape)

    def GiveModelImage(self, *args): return self.ModelMachine.GiveModelImage(*args)

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe
        

    def SetPSF(self,DicoVariablePSF):
        self.PSFServer=ClassPSFServer(self.GD)
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF)
        self.DicoVariablePSF=DicoVariablePSF

    def setNoiseMap(self, NoiseMap, PNRStop=10):
        """Sets the noise map. The mean dirty will be divided by the noise map before peak finding.
        If PNRStop is set, an additional stopping criterion (peak-to-noisemap) will be applied.
            Peaks are reported in units of sigmas.
        If PNRStop is not set, NoiseMap is treated as simply an (inverse) weighting that will bias
            peak selection in the minor cycle. In this mode, peaks are reported in units of flux.
        """
        self._NoiseMap = NoiseMap
        self._PNRStop = PNRStop
        self._peakMode = "sigma"

    def SetDirty(self,DicoDirty):
        self.DicoDirty=DicoDirty
        self._Dirty = self.DicoDirty["ImageCube"]
        self._MeanDirty = self.DicoDirty["MeanImage"]

        NPSF=self.PSFServer.NPSF
        _,_,NDirty,_=self._Dirty.shape

        off=(NPSF-NDirty)/2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)

        if self._peakMode is "sigma":
            print>> log, "Will search for the peak in the SNR-weighted dirty map"
            a, b = self._MeanDirty, self._NoiseMap.reshape(self._MeanDirty.shape)
            self._PeakSearchImage = numexpr.evaluate("a/b")
        # elif self._peakMode is "weighted":   ######## will need to get a PeakWeightImage from somewhere for this option
        #     print>> log, "Will search for the peak in the weighted dirty map"
        #     a, b = self._MeanDirty, self._peakWeightImage
        #     self._PeakSearchImage = numexpr.evaluate("a*b")
        else:
            print>> log, "Will search for the peak in the unweighted dirty map"
            self._PeakSearchImage = self._MeanDirty

        if self.ModelImage is None:
            self._ModelImage=np.zeros_like(self._Dirty)
        if self.MaskArray is None:
            self._MaskArray=np.zeros(self._Dirty.shape,dtype=np.bool8)


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
            W=np.mean(np.float32(self.DicoDirty["WeightChansImages"]),axis=1)  #Get the weights (assuming they stay relatively the same over stokes terms)
            self._MeanDirty[0,:,x0d:x1d,y0d:y1d]-=np.sum(LocalSM[:,:,x0p:x1p,y0p:y1p]*W.reshape((W.size,1,1,1)),axis=0) #Sum over frequency

    def setChannel(self,ch=0):
        """
        In case we ever want to deconvolve per channel.
        Currently just sets self.Dirty to average over freq bands.
        """
        #self.PSF=self._MeanPSF[ch]
        if self.MultiFreqMode:
            self.Dirty = self._MeanDirty.view()[ch]
        else:
            self.Dirty = self._Dirty.view()[ch]
        self.ModelImage = self._ModelImage.view()[ch]
        self.MaskArray = self._MaskArray.view()[ch]


    def Deconvolve(self, ch=0, **kwargs):
        """
        Runs minor cycle over image channel 'ch'.
        initMinor is number of minor iteration (keeps continuous count through major iterations)
        Nminor is max number of minor iterations

        Returns tuple of: return_code,continue,updated
        where return_code is a status string;
        continue is True if another cycle should be executed (one or more polarizations still need cleaning);
        update is True if one or more polarization models have been updated
        """
        #No need to set the channel when doing joint deconvolution
        self.setChannel(ch)

        exit_msg = ""
        continue_deconvolution = False
        update_model = False

        _,npix,_=self.Dirty.shape
        xc=(npix)/2

        npol,_,_=self.Dirty.shape

        # Get the PeakMap (first index will always be 0 because we only support I cleaning)
        PeakMap = self.Dirty[0,:,:]

        m0,m1=PeakMap.min(),PeakMap.max()

        #These options should probably be moved into MinorCycleConfig in parset
        DoAbs=int(self.GD["Deconv"]["AllowNegative"])
        print>>log, "  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter, self.MaxMinorIter, DoAbs)

        ## Determine which stopping criterion to use for flux limit
        #Get RMS stopping criterion
        NPixStats = self.GD["Deconv"]["NumRMSSamples"]
        if NPixStats:
            RandomInd=np.int64(np.random.rand(NPixStats)*npix**2)
            RMS=np.std(np.real(PeakMap.ravel()[RandomInd]))
        else:
            RMS = np.std(PeakMap)

        self.RMS=RMS

        self.GainMachine.SetRMS(RMS)

        Fluxlimit_RMS = self.RMSFactor*RMS

        #Find position and intensity of first peak
        x,y,MaxDirty=NpParallel.A_whereMax(PeakMap,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)

        #Get peak factor stopping criterion
        Fluxlimit_Peak = MaxDirty*self.PeakFactor

        #Get side lobe stopping criterion
        Fluxlimit_Sidelobe = ((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*MaxDirty if self.CycleFactor else 0

        mm0,mm1=PeakMap.min(),PeakMap.max()

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
            print>>log, ModColor.Str("    Initial maximum peak %g Jy below threshold, we're done CLEANing" % (ThisFlux),col="green" )
            exit_msg = exit_msg + " " + "FluxThreshold"
            continue_deconvolution = False or continue_deconvolution
            update_model = False or update_model
            # No need to do anything further if we are already at the stopping flux
            return exit_msg, continue_deconvolution, update_model

        # set peak in GainMachine (deprecated?)
        self.GainMachine.SetFluxMax(ThisFlux)

        # def GivePercentDone(ThisMaxFlux):
        #     fracDone=1.-(ThisMaxFlux-StopFlux)/(MaxDirty-StopFlux)
        #     return max(int(round(100*fracDone)),100)

        #Do minor cycle deconvolution loop
        try:
            for i in range(self._niter+1,self.MaxMinorIter+1):
                self._niter = i
                #grab a new peakmap
                PeakMap = self.Dirty[0, :, :]

                x,y,ThisFlux=NpParallel.A_whereMax(PeakMap,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)

                # deprecated?
                self.GainMachine.SetFluxMax(ThisFlux)

                T.timeit("max0")

                if ThisFlux <= StopFlux:
                    print>>log, ModColor.Str("    CLEANing [iter=%i] peak of %.3g Jy lower than stopping flux" % (i,ThisFlux),col="green")
                    cont = ThisFlux > self.FluxThreshold
                    if not cont:
                          print>>log, ModColor.Str("    CLEANing [iter=%i] absolute flux threshold of %.3g Jy has been reached" % (i,self.FluxThreshold),col="green",Bold=True)
                    exit_msg = exit_msg + " " + "MinFluxRms"
                    continue_deconvolution = cont or continue_deconvolution
                    update_model = True or update_model

                    break # stop cleaning if threshold reached

                # This is used to track Cleaning progress
                rounded_iter_step = 1 if i < 10 else (
                    10 if i < 200 else (
                        100 if i < 2000
                        else 1000))
                # min(int(10**math.floor(math.log10(i))), 10000)
                if i >= 10 and i % rounded_iter_step == 0:
                    # if self.GD["Debug"]["PrintMinorCycleRMS"]:
                    #rms = np.std(np.real(self._CubeDirty.ravel()[self.IndStats]))
                    print>> log, "    [iter=%i] peak residual %.3g" % (i, ThisFlux)

                nch,npol,_,_=self._Dirty.shape
                #Fpol contains the intensities at (x,y) per freq and polarisation
                Fpol = np.zeros([nch, npol, 1, 1], dtype=np.float32)
                if self.MultiFreqMode:
                    if self.GD["Hogbom"]["FreqMode"] == "Poly":
                        Ncoeffs = self.GD["Hogbom"]["PolyFitOrder"]
                    elif self.GD["Hogbom"]["FreqMode"] == "GPR":
                        Ncoeffs = self.GD["Hogbom"]["NumBasisFuncs"]
                    else:
                        raise NotImplementedError("FreqMode %s not supported" % self.GD["Hogbom"]["FreqMode"])
                    Coeffs = np.zeros([npol, Ncoeffs])
                else:
                    Coeffs = np.zeros([npol, nch])  # to support per channel cleaning

                # Get the JonesNorm
                JonesNorm = (self.DicoDirty["JonesNorm"][:, :, x, y]).reshape((nch, npol, 1, 1))

                # Get the solution
                Fpol[:, 0, 0, 0] = self._Dirty[:, 0, x, y]/np.sqrt(JonesNorm[:, 0, 0, 0])
                # Fit a polynomial to get coeffs
                # tmp = self.ModelMachine.FreqMachine.Fit(Fpol[:, 0, 0, 0])
                # print tmp.shape
                Coeffs[0, :] = self.ModelMachine.FreqMachine.Fit(Fpol[:, 0, 0, 0])
                # Overwrite with polynoimial fit
                Fpol[:, 0, 0, 0] = self.ModelMachine.FreqMachine.Eval(Coeffs[0, :])

                T.timeit("stuff")

                #Find PSF corresponding to location (x,y)
                self.PSFServer.setLocation(x,y) #Selects the facet closest to (x,y)
                PSF, meanPSF = self.PSFServer.GivePSF()  #Gives associated PSF
                _, _, PSFnx, PSFny = PSF.shape
                # Normalise PSF in each channel
                PSF /= np.amax(PSF.reshape(nch, npol, PSFnx * PSFny), axis=2, keepdims=True).reshape(nch, npol, 1, 1)

                T.timeit("FindScale")

                CurrentGain = self.GainMachine.GiveGain()

                #Update model
                self.ModelMachine.AppendComponentToDictStacked((x, y), 1.0, Coeffs[0, :], 0)

                # Subtract LocalSM*CurrentGain from dirty image
                self.SubStep((x,y),PSF*Fpol*CurrentGain*np.sqrt(JonesNorm))
                T.timeit("SubStep")

                T.timeit("End")

        except KeyboardInterrupt:
            print>>log, ModColor.Str("    CLEANing [iter=%i] minor cycle interrupted with Ctrl+C, peak flux %.3g" % (self._niter, ThisFlux))
            exit_msg = exit_msg + " " + "MaxIter"
            continue_deconvolution = False or continue_deconvolution
            update_model = True or update_model
            return exit_msg, continue_deconvolution, update_model

        if self._niter >= self.MaxMinorIter: #Reached maximum number of iterations:
            print>> log, ModColor.Str("    CLEANing [iter=%i] Reached maximum number of iterations, peak flux %.3g" % (self._niter, ThisFlux))
            exit_msg = exit_msg + " " + "MaxIter"
            continue_deconvolution = False or continue_deconvolution
            update_model = True or update_model

        return exit_msg, continue_deconvolution, update_model

    def Update(self,DicoDirty,**kwargs):
        """
        Method to update attributes from ClassDeconvMachine
        """
        #Update image dict
        self.SetDirty(DicoDirty)
        #self.SetModelRefFreq()
        self.SetModelShape()

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

    def updateRMS(self):
        _,npol,npix,_ = self._MeanDirty.shape
        NPixStats = self.GD["Deconv"]["NumRMSSamples"]
        if NPixStats:
            #self.IndStats=np.int64(np.random.rand(NPixStats)*npix**2)
            self.IndStats=np.int64(np.linspace(0,self._PeakSearchImage.size-1,NPixStats))
        else:
            self.IndStats = slice(None)
        self.RMS=np.std(np.real(self._PeakSearchImage.ravel()[self.IndStats]))

    def resetCounter(self):
        self._niter = 0