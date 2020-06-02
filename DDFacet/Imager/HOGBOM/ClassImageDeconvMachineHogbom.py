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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

"""
This minimal implementation of standard Hogbom CLEAN algorithm should serve
as a minimal reference interface of how to incorporate new deconvolution
algorithms into DDFacet.
"""

import numpy as np
import numexpr
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("ClassImageDeconvMachine")
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
        self.GainMachine = ClassGainMachine.get_instance()
        if ModelMachine is None:
            from DDFacet.Imager.HOGBOM import ClassModelMachineHogbom as ClassModelMachine
            self.ModelMachine = ClassModelMachine.ClassModelMachine(self.GD, GainMachine=self.GainMachine)
        else:
            self.ModelMachine = ModelMachine
        self.GiveEdges = GiveEdges.GiveEdges
        self._niter = 0
        self._peakMode = "normal"

        self.CurrentNegMask = None
        self._NoiseMap = None
        self._PNRStop = None  # in _peakMode "sigma", provides addiitonal stopping criterion

        numexpr.set_num_threads(self.NCPU)


    def Init(self, **kwargs):
        self.SetPSF(kwargs["PSFVar"])
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
        self.Freqs = kwargs["GridFreqs"]
        AllDegridFreqs = []
        for i in kwargs["DegridFreqs"].keys():
            AllDegridFreqs.append(kwargs["DegridFreqs"][i])
        self.Freqs_degrid = np.asarray(AllDegridFreqs).flatten()
        self.SetPSF(kwargs["PSFVar"])
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
        self.ModelMachine.setPSFServer(self.PSFServer)
        self.ModelMachine.setFreqMachine(self.Freqs, self.Freqs_degrid,
                                         weights=kwargs["PSFVar"]["WeightChansImages"], PSFServer=self.PSFServer)


    def Reset(self):
        pass

    def setMaskMachine(self,MaskMachine):
        self.MaskMachine=MaskMachine
        if self.MaskMachine.ExternalMask is not None:
            print("Applying external mask", file=log)
            MaskArray=self.MaskMachine.ExternalMask
            nch,npol,_,_=MaskArray.shape
            self._MaskArray=np.zeros(MaskArray.shape,np.bool8)
            for ch in range(nch):
                for pol in range(npol):
                    self._MaskArray[ch,pol,:,:]=np.bool8(1-MaskArray[ch,pol].copy())[:,:]
            self._MaskArray = np.ascontiguousarray(self._MaskArray)
            self.MaskArray = np.ascontiguousarray(self._MaskArray[0])

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
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF, NormalisePSF=True)
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
        self.WeightsChansImages = DicoDirty["WeightChansImages"].squeeze()
        self._Dirty = self.DicoDirty["ImageCube"]
        self._MeanDirty = self.DicoDirty["MeanImage"]

        self.NpixPSF = self.PSFServer.NPSF
        self.Nchan, self.Npol, self.Npix, _ = self._Dirty.shape

        # if self._peakMode is "sigma":
        #     print("Will search for the peak in the SNR-weighted dirty map", file=log)
        #     a, b = self._MeanDirty, self._NoiseMap.reshape(self._MeanDirty.shape)
        #     self._PeakSearchImage = numexpr.evaluate("a/b")
        # # elif self._peakMode is "weighted":   ######## will need to get a PeakWeightImage from somewhere for this option
        # #     print("Will search for the peak in the weighted dirty map", file=log)
        # #     a, b = self._MeanDirty, self._peakWeightImage
        # #     self._PeakSearchImage = numexpr.evaluate("a*b")
        # else:
        #     print("Will search for the peak in the unweighted dirty map", file=log)
        #     self._PeakSearchImage = self._MeanDirty

        if self.ModelImage is None:
            self._ModelImage=np.zeros_like(self._Dirty)
        if self.MaskArray is None:
            self._MaskArray=np.zeros(self._Dirty.shape,dtype=np.bool8)


    def SubStep(self,xc,yc,LocalSM):
        """
        This is where subtraction in the image domain happens
        
        Parameters
        ----------
        (xc, yc) - The location of the component
        LocalSM - array of shape (nchan, npol, nx, ny)
                  Local Sky Model = comp * PSF * gain where the PSF should be
                  normalised to unity at the center.
        """
        #Get overlap indices where psf should be subtracted
        Aedge,Bedge=self.GiveEdges(xc,yc, self.Npix, self.NpixPSF//2,self.NpixPSF//2,self.NpixPSF)

        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge

        cube, sm = self._Dirty[:,:,x0d:x1d,y0d:y1d], LocalSM[:,:,x0p:x1p,y0p:y1p]
        numexpr.evaluate('cube-sm',out=cube,casting="unsafe")

        #Subtract from each channel/band
        # self._Dirty[:,:,x0d:x1d,y0d:y1d]-=LocalSM[:,:,x0p:x1p,y0p:y1p]
        # If multiple frequencies are present construct the weighted mean
        meanimage = self._MeanDirty[:, :, x0d:x1d, y0d:y1d]
        if self.MultiFreqMode:
            W = self.WeightsChansImages.reshape((self.Nchan,1,1,1))
            meanimage[...] = (cube*W).sum(axis=0) #Sum over frequency
        else:
            meanimage[0,...] = cube[0,...]

    def Deconvolve(self, **kwargs):
        """
        Runs minor cycle over image channel 'ch'.
        initMinor is number of minor iteration (keeps continuous count through major iterations)
        Nminor is max number of minor iterations

        Returns tuple of: return_code,continue,updated
        where return_code is a status string;
        continue is True if another cycle should be executed (one or more polarizations still need cleaning);
        update is True if one or more polarization models have been updated
        """
        exit_msg = ""
        continue_deconvolution = False
        update_model = False

        # # Get the PeakMap (first index will always be 0 because we only support I cleaning)
        PeakMap = self._MeanDirty[0, 0, :, :]

        #These options should probably be moved into MinorCycleConfig in parset
        DoAbs=int(self.GD["Deconv"]["AllowNegative"])
        print("  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter, self.MaxMinorIter, DoAbs), file=log)

        ## Determine which stopping criterion to use for flux limit
        #Get RMS stopping criterion
        NPixStats = self.GD["Deconv"]["NumRMSSamples"]
        if NPixStats:
            RandomInd=np.int64(np.random.rand(NPixStats)*self.Npix**2)
            RMS=np.std(np.real(PeakMap.ravel()[RandomInd]))
        else:
            RMS = np.std(PeakMap)

        self.RMS=RMS

        Fluxlimit_RMS = self.RMSFactor*RMS

        # Find position and intensity of first peak
        x,y,MaxDirty=NpParallel.A_whereMax(PeakMap,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)

        # Get peak factor stopping criterion
        Fluxlimit_Peak = MaxDirty*self.PeakFactor

        # Get side lobe stopping criterion
        Fluxlimit_Sidelobe = ((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*MaxDirty if self.CycleFactor else 0

        mm0, mm1 = PeakMap.min(), PeakMap.max()

        # Choose whichever threshold is highest
        StopFlux = max(Fluxlimit_Peak, Fluxlimit_RMS, Fluxlimit_Sidelobe, self.FluxThreshold)

        print("    Dirty image peak flux      = %10.6f Jy [(min, max) = (%.3g, %.3g) Jy]"%(MaxDirty,mm0,mm1), file=log)
        print("      RMS-based threshold      = %10.6f Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor), file=log)
        print("      Sidelobe-based threshold = %10.6f Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor), file=log)
        print("      Peak-based threshold     = %10.6f Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor), file=log)
        print("      Absolute threshold       = %10.6f Jy"%(self.FluxThreshold), file=log)
        print("    Stopping flux              = %10.6f Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty), file=log)

        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        ThisFlux=MaxDirty

        if ThisFlux < StopFlux:
            print(ModColor.Str("    Initial maximum peak %g Jy below threshold, we're done CLEANing" % (ThisFlux),col="green" ), file=log)
            exit_msg = exit_msg + " " + "FluxThreshold"
            continue_deconvolution = False or continue_deconvolution
            update_model = False or update_model
            # No need to do anything further if we are already at the stopping flux
            return exit_msg, continue_deconvolution, update_model

        #Do minor cycle deconvolution loop
        try:
            for i in range(self._niter+1,self.MaxMinorIter+1):
                self._niter = i
                #grab a new peakmap
                PeakMap = self._MeanDirty[0, 0, :, :]

                x,y,ThisFlux=NpParallel.A_whereMax(PeakMap,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)

                T.timeit("max0")

                if ThisFlux <= StopFlux:
                    print(ModColor.Str("    CLEANing [iter=%i] peak of %.3g Jy lower than stopping flux" % (i,ThisFlux),col="green"), file=log)
                    cont = ThisFlux > self.FluxThreshold
                    if not cont:
                          print(ModColor.Str("    CLEANing [iter=%i] absolute flux threshold of %.3g Jy has been reached" % (i,self.FluxThreshold),col="green",Bold=True), file=log)
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
                    print("    [iter=%i] peak residual %.3g" % (i, ThisFlux), file=log)

                # Find PSF corresponding to location (x,y)
                self.PSFServer.setLocation(x, y)  # Selects the facet closest to (x,y)

                # Get the JonesNorm
                JonesNorm = self.DicoDirty["JonesNorm"][:, 0, x, y]

                # Get the solution (division by JonesNorm handled in fit)
                Iapp = self._Dirty[:, 0, x, y]

                # Fit a polynomial to get coeffs
                Coeffs = self.ModelMachine.FreqMachine.Fit(Iapp, JonesNorm, self.WeightsChansImages)

                # Overwrite with polynoimial fit
                Iapp = self.ModelMachine.FreqMachine.Eval(Coeffs)
                T.timeit("stuff")

                PSF, meanPSF = self.PSFServer.GivePSF()  #Gives associated PSF

                T.timeit("FindScale")

                #Update model
                self.ModelMachine.AppendComponentToDictStacked((x, y), Coeffs)

                # Subtract LocalSM*CurrentGain from dirty image
                self.SubStep(x, y, PSF * Iapp[:, None, None, None] * self.GD["Deconv"]["Gain"])

                T.timeit("SubStep")

                T.timeit("End")

        except KeyboardInterrupt:
            print(ModColor.Str("    CLEANing [iter=%i] minor cycle interrupted with Ctrl+C, peak flux %.3g" % (self._niter, ThisFlux)), file=log)
            exit_msg = exit_msg + " " + "MaxIter"
            continue_deconvolution = False or continue_deconvolution
            update_model = True or update_model
            return exit_msg, continue_deconvolution, update_model

        if self._niter >= self.MaxMinorIter: #Reached maximum number of iterations:
            print(ModColor.Str("    CLEANing [iter=%i] Reached maximum number of iterations, peak flux %.3g" % (self._niter, ThisFlux)), file=log)
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
