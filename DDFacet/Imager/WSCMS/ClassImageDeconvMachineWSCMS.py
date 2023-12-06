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

"""
This is an implementation of the multi-scale algorithm implemented in wsclean
"""

from DDFacet.compatibility import range

import numpy as np
from scipy.integrate import cumtrapz
import numexpr
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.Other import ClassTimeIt
from pyrap.images import image
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Imager import ClassGainMachine  # Currently required by model machine but fixed to static mode
from DDFacet.Imager.ClassMaskMachine import ClassMaskMachine
from DDFacet.ToolsDir.GiveEdges import GiveEdges
# from DDFacet.Other.AsyncProcessPool import APP
# from DDFacet.ToolsDir.ModFFTW import FFTW_Scale_Manager  # usage just to register job handlers but has no effect atm

class ClassImageDeconvMachine():
    """
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
    def __init__(self,Gain=0.1,
                 MaxMinorIter=50000,
                 NCPU=0,
                 CycleFactor=2.5,
                 FluxThreshold=None,
                 RMSFactor=3,
                 PeakFactor=0,
                 GD=None,
                 SearchMaxAbs=1,
                 CleanMaskImage=None,
                 ImagePolDescriptor=["I"],
                 ModelMachine=None,
                 MainCache=None,
                 CacheFileName='WSCMS',
                 **kw    # absorb any unknown keywords arguments here
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
        if ModelMachine is None:
            # raise RuntimeError("You need to supply ImageDeconvMachine with a instantiated ModelMachine")
            from DDFacet.Imager.WSCMS import ClassModelMachineWSCMS as ClassModelMachine
            self.ModelMachine = ClassModelMachine.ClassModelMachine(self.GD, GainMachine=ClassGainMachine.get_instance())
        else:
            self.ModelMachine = ModelMachine
        self.GainMachine = self.ModelMachine.GainMachine
        self._niter = 0

        # cache options
        self.maincache = MainCache
        self.CacheFileName = CacheFileName
        self.PSFHasChanged = False
        self.LastScale = 99999

        self._peakMode = "normal"

        self.CacheFileName = CacheFileName
        self.CurrentNegMask = None
        self._NoiseMap = None
        self._PNRStop = None  # in _peakMode "sigma", provides addiitonal stopping criterion

        # # this is so that the relevant functions are registered as job handlers with APP
        # # pass to ModelMachine.setScaleMachine to set workers
        # self.FTMachine = FFTW_Scale_Manager(wisdom_file=self.GD["Cache"]["DirWisdomFFTW"])
        #
        # APP.registerJobHandlers(self)


    def Init(self, cache=None, facetcache=None, FacetMachine=None, BaseName=None, **kwargs):
        # check for valid cache
        cachehash = dict(
            [(section, self.GD[section]) for section in (
                "Data", "Beam", "Selection", "Freq",
                "Image", "Facets", "Weight", "RIME",
                "Comp", "CF", "WSCMS")])

        cachepath, valid = self.maincache.checkCache(self.CacheFileName, cachehash, directory=True,
                                                     reset=cache or self.PSFHasChanged)
        # export the hash
        self.maincache.saveCache(name='WSCMS')

        # required to save intermediate images
        self.FacetMachine = FacetMachine
        self.BaseName = BaseName
        self.ModelMachine.setFacetMachine(FacetMachine=self.FacetMachine, BaseName=self.BaseName)

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

        from africanus.constants import c as lightspeed
        minlambda = lightspeed/self.Freqs.min()
        # LB - note MaskArray might be modified by ScaleMachine if GD{"WSCMS"]["AutoMask"] is True
        # so we should avoid keeping it as None
        self.Nchan, self.Npol, self.Npix, _ = self.DicoVariablePSF["OutImShape"]
        if self.MaskArray is None:
            self.MaskArray = np.zeros([1, 1, self.Npix, self.Npix], dtype=np.bool8)
        else:  # Make sure mask is correct shape
            if self.MaskArray.shape != (1, 1, self.Npix, self.Npix):
                raise ValueError("Mask is incorrect shape. Expected %s but got %s" % ((1,1,self.Npix, self.Npix), self.MaskArray.shape))

        self.ModelMachine.setScaleMachine(self.PSFServer, NCPU=self.NCPU, MaskArray=self.MaskArray,
                                          cachepath=cachepath, MaxBaseline=kwargs["MaxBaseline"] / minlambda)


    def Reset(self):
        pass

    def setMaskMachine(self,MaskMachine):
        self.MaskMachine=MaskMachine
        if self.MaskMachine.ExternalMask is not None:
            print("Applying external mask", file=log)
            MaskArray=self.MaskMachine.ExternalMask
            nch,npol,_,_=MaskArray.shape
            self.MaskArray=np.zeros(MaskArray.shape,np.bool8)
            for ch in range(nch):
                for pol in range(npol):
                    self.MaskArray[ch,pol,:,:]=np.bool8(1-MaskArray[ch,pol].copy())[:,:]
            self.MaskArray = np.ascontiguousarray(self.MaskArray)


    def SetModelRefFreq(self, RefFreq):
        """
        Sets ref freq in ModelMachine.
        """
        self.ModelMachine.setRefFreq(RefFreq)


    def SetModelShape(self):
        """
        Sets the shape params of model, call in every update step
        """
        assert self._Dirty.shape == (self.Nchan, self.Npol, self.Npix, self.Npix)
        self.ModelMachine.setModelShape(self._Dirty.shape)
        self.NpixFacet = self.Npix//self.GD["Facets"]["NFacets"]

    def GiveModelImage(self, *args): return self.ModelMachine.GiveModelImage(*args)

    def setSideLobeLevel(self, SideLobeLevel, OffsetSideLobe):
        self.SideLobeLevel = SideLobeLevel
        self.OffsetSideLobe = OffsetSideLobe
        

    def SetPSF(self,DicoVariablePSF):
        """
        The keys in DicoVariablePSF and what they mean:
         'MeanFacetPSF' -    
         'MeanImage' -
         'ImageCube' -
         'CellSizeRad' -
         'ChanMappingGrid' -
         'ChanMappingGridChan' -
         'CubeMeanVariablePSF' -
         'CubeVariablePSF' -
         'SumWeights'           -
         'MeanJonesBand'        -
         'PeakNormed_CubeMeanVariablePSF'
         'PeakNormed_CubeVariablePSF'
         'OutImShape'
         'JonesNorm'
         'Facets'
         'PSFSidelobes'
         'ImageInfo'
         'CentralFacet'
         'freqs'
         'SumJonesChan'
         'SumJonesChanWeightSq'
         'EstimatesAvgPSF'
         'WeightChansImages'
         'FacetNorm'
         'PSFGaussPars'
         'FWHMBeam'
        
        """
        self.PSFServer = ClassPSFServer(self.GD)
        # NormalisePSF must be true here for the beam to be applied correctly
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF, NormalisePSF=True)
        self.DicoVariablePSF = DicoVariablePSF

    def setNoiseMap(self, NoiseMap, PNRStop=10):
        """
        Sets the noise map. The mean dirty will be divided by the noise map before peak finding.
        If PNRStop is set, an additional stopping criterion (peak-to-noisemap) will be applied.
            Peaks are reported in units of sigmas.
        If PNRStop is not set, NoiseMap is treated as simply an (inverse) weighting that will bias
            peak selection in the minor cycle. In this mode, peaks are reported in units of flux.
        """
        self._NoiseMap = NoiseMap
        self._PNRStop = PNRStop
        self._peakMode = "sigma"

    def SetDirty(self,DicoDirty):
        """
        The keys in DicoDirty and what they mean (see also FacetMachine.FacetsToIm docs)
         'JonesNorm' - array containing norm of Jones terms as an image 
         'ImageInfo' - dictionary containing 'CellSizeRad' and 'OutImShape'
         'ImageCube' - array containing residual
         'MeanImage' - array containing mean of the residual
         'freqs' - dictionary keyed by band number containing the actual frequencies that got binned into that band 
         'SumWeights' - sum of visibility weights used in normalizing the gridded correlations
         'FacetMeanResidual' - ???
         'WeightChansImages' - Weights corresponding to imaging bands (how is this computed?)
         'FacetNorm' - self.FacetImage (grid-correcting map) see FacetMachine
        """
        self.DicoDirty = DicoDirty
        self._Dirty = self.DicoDirty["ImageCube"]
        self._MeanDirty = self.DicoDirty["MeanImage"]
        self._JonesNorm = self.DicoDirty["JonesNorm"]
        self.WeightsChansImages = np.mean(np.float32(self.DicoDirty["WeightChansImages"]), axis=1)[:, None, None, None]


    def SubStep(self,dx,dy,LocalSM):
        """
        This is where subtraction in the image domain happens
        """
        xc, yc = dx, dy
        N1 = LocalSM.shape[-1]

        # Get overlap indices where psf should be subtracted
        Aedge, Bedge = GiveEdges(xc, yc, self.Npix, N1//2, N1//2, N1)

        x0d, x1d, y0d, y1d = Aedge
        x0p, x1p, y0p, y1p = Bedge

        cube, sm = self._Dirty[:,:,x0d:x1d,y0d:y1d], LocalSM[:,:,x0p:x1p,y0p:y1p]
        numexpr.evaluate('cube-sm',out=cube,casting="unsafe")

        # Subtract from the average
        meanimage = self._MeanDirty[:, :, x0d:x1d, y0d:y1d]
        if self.MultiFreqMode:
            W = self.WeightsChansImages.reshape((self.Nchan,1,1,1))
            meanimage[...] = (cube*W).sum(axis=0) #Sum over frequency
        else:
            meanimage[0,...] = cube[0,...]

    def track_progress(self, i, ThisFlux):
        # This is used to track Cleaning progress
        rounded_iter_step = 1 if i < 10 else (
            10 if i < 200 else (
                100 if i < 2000
                else 1000))
        # min(int(10**math.floor(math.log10(i))), 10000)
        if i >= 10 and i % rounded_iter_step == 0:
            # if self.GD["Debug"]["PrintMinorCycleRMS"]:
            # rms = np.std(np.real(self._CubeDirty.ravel()[self.IndStats]))
            print("    [iter=%i] peak residual %.3g" % (i, ThisFlux), file=log)

    def check_stopping_criteria(self):
        # Get RMS stopping criterion
        RMS = np.std(self._MeanDirty)
        Fluxlimit_RMS = self.RMSFactor*RMS

        # Find position and intensity of first peak
        x, y, MaxDirty = NpParallel.A_whereMax(self._MeanDirty, NCPU=self.NCPU,
                                               DoAbs=self.GD["Deconv"]["AllowNegative"], Mask=self.MaskArray)


        # Get peak factor stopping criterion
        Fluxlimit_Peak = MaxDirty*self.PeakFactor

        # Get side lobe stopping criterion
        Fluxlimit_Sidelobe = ((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*MaxDirty if self.CycleFactor else 0

        mm0, mm1 = self._MeanDirty.min(), self._MeanDirty.max()

        # Choose whichever threshold is highest
        StopFlux = max(Fluxlimit_Peak, Fluxlimit_RMS, Fluxlimit_Sidelobe, self.FluxThreshold)

        print("    Dirty image peak flux      = %10.6f Jy [(min, max) = (%.3g, %.3g) Jy]"%(MaxDirty,mm0,mm1), file=log)
        print("      RMS-based threshold      = %10.6f Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor), file=log)
        print("      Sidelobe-based threshold = %10.6f Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor), file=log)
        print("      Peak-based threshold     = %10.6f Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor), file=log)
        print("      Absolute threshold       = %10.6f Jy"%(self.FluxThreshold), file=log)
        print("    Stopping flux              = %10.6f Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty), file=log)

        return StopFlux, MaxDirty, RMS

    def Deconvolve(self):
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

        # These options should probably be moved into MinorCycleConfig in parset
        print("  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter, self.MaxMinorIter,
                                                                              int(self.GD["Deconv"]["AllowNegative"])),
                                                                              file=log)

        # Determine which stopping criterion to use for flux limit
        StopFlux, MaxDirty, RMS = self.check_stopping_criteria()

        TrackRMS = RMS.copy()

        ThisFlux = MaxDirty.copy()

        if ThisFlux < self.FluxThreshold:
            print(ModColor.Str("    Initial maximum peak %g Jy below threshold, we're done CLEANing" % (ThisFlux),col="green" ), file=log)
            exit_msg = exit_msg + " " + "FluxThreshold"
            continue_deconvolution = False or continue_deconvolution
            update_model = False or update_model
            # No need to do anything further if we are already at the stopping flux
            return exit_msg, continue_deconvolution, update_model

        # Do minor cycle deconvolution loop
        TrackFlux = MaxDirty.copy()
        diverged = False
        diverged_count = 0
        stalled = False
        scale_stall_count = {}
        scales_stalled = np.zeros(self.ModelMachine.ScaleMachine.Nscales, dtype=bool)
        # reset retired scales at the start of each major cycle
        self.ModelMachine.ScaleMachine.retired_scales = []
        for scale in self.ModelMachine.ScaleMachine.forbidden_scales:
            self.ModelMachine.ScaleMachine.retired_scales.append(scale)
            scales_stalled[scale] = 1
        try:
            while self._niter <= self.MaxMinorIter:
                # Check if diverging
                if np.abs(ThisFlux) > self.GD["WSCMS"]["MinorDivergenceFactor"] * np.abs(TrackFlux):
                    diverged_count += 1
                    if diverged_count > 5:
                        diverged = True

                TrackFlux = ThisFlux.copy()

                if ThisFlux <= StopFlux or diverged or stalled:
                    if diverged:
                        print(ModColor.Str("    At [iter=%i] minor cycle is diverging so it has been force stopped at a flux of %.3g Jy" % (self._niter, ThisFlux),col="green"), file=log)
                    elif stalled:
                        print(ModColor.Str("    At [iter=%i] minor cycle has stalled so it has been force stopped at a flux of %.3g Jy" % (self._niter, ThisFlux), col="green"), file=log)
                    else:
                        print(ModColor.Str("    CLEANing [iter=%i] peak of %.3g Jy lower than stopping flux" % (self._niter, ThisFlux),col="green"), file=log)
                    cont = ThisFlux > self.FluxThreshold
                    if not cont:
                          print(ModColor.Str("    CLEANing [iter=%i] absolute flux threshold of %.3g Jy has been reached" % (self._niter, StopFlux),col="green",Bold=True), file=log)
                    exit_msg = exit_msg + " " + "MinFluxRms"
                    continue_deconvolution = cont or continue_deconvolution
                    update_model = True or update_model

                    break  # stop cleaning if threshold reached

                # Find the relevant scale and do sub-minor loop. Note that the dirty cube is updated during the
                # sub-minor loop by subtracting the once convolved PSF's as components are added to the model.
                # The model is updated by adding components to the ModelMachine dictionary.
                niter, iScale = self.ModelMachine.do_minor_loop(self._Dirty, self._MeanDirty, self._JonesNorm,
                                                                self.WeightsChansImages, ThisFlux, StopFlux, RMS)

                # compute the new mean image from the weighted sum of over frequency
                self._MeanDirty = np.sum(self._Dirty * self.WeightsChansImages, axis=0, keepdims=True)

                ThisRMS = np.std(self._MeanDirty * ~self.MaskArray)

                # check for and retire scales that cause stalls
                if np.abs(TrackRMS - ThisRMS) < self.GD['WSCMS']['MinorStallThreshold']:
                    scale_stall_count.setdefault(iScale, 0)
                    scale_stall_count[iScale] += 1
                    # retire scale if it causes a stall more than x number of times
                    if scale_stall_count[iScale] > 5:
                        self.ModelMachine.ScaleMachine.retired_scales.append(iScale)
                        scales_stalled[iScale] = 1
                        print("Retired scale %i because it was stalling." % iScale, file=log)
                    # if all scales have stalled then we trigger a new major cycle
                    if np.all(scales_stalled):
                        stalled = True
                TrackRMS = ThisRMS.copy()

                # find peak
                x, y, ThisFlux = NpParallel.A_whereMax(self._MeanDirty, NCPU=self.NCPU,
                                                       DoAbs=self.GD["Deconv"]["AllowNegative"],
                                                       Mask=self.MaskArray)

                # update counter
                self._niter += niter

                if iScale != self.LastScale:
                    print("    [iter=%i] peak residual %.8g, rms = %.8g, scale = %i" % (self._niter, ThisFlux, TrackRMS, iScale), file=log)
                    self.LastScale = iScale

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