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
This is an implementation of the multi-scale algorithm implemented in wsclean
"""

import numpy as np
from scipy.integrate import cumtrapz
import numexpr
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.Other import ClassTimeIt
from pyrap.images import image
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Imager import ClassGainMachine  # Currently required by model machine but fixed to static mode
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.Other.AsyncProcessPool import APP
from DDFacet.ToolsDir.ModFFTW import FFTW_Scale_Manager  # usage just to register job handlers but has no effect atm

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
            import ClassModelMachineWSCMS as ClassModelMachine
            self.ModelMachine = ClassModelMachine.ClassModelMachine(self.GD, GainMachine=ClassGainMachine.get_instance())
        else:
            self.ModelMachine = ModelMachine
        self.GainMachine = self.ModelMachine.GainMachine
        self._niter = 0

        # cache options
        self.maincache = MainCache
        self.CacheFileName = CacheFileName
        self.PSFHasChanged = False

        #  TODO - use MaskMachine for this
        CleanMaskImage = self.GD["Mask"]["External"]
        if CleanMaskImage is not None:
            print>>log, "Reading mask image: %s"%CleanMaskImage
            MaskArray=image(CleanMaskImage).getdata()
            nch,npol,nxmask,nymask=MaskArray.shape
            # if (nch > 1) or (npol > 1):
            #     print>>log, "Warning - only single channel and pol mask supported. Will use mask for ch 0 pol 0"
            # MaskArray = MaskArray[0,0]
            # _, _, nxmod, nymod = self.ModelMachine.ModelShape
            # if (nxmod != nxmask) or (nymod !=nymask):
            #     print>>log, "Warning - shape of mask != shape of your model. Will pad/trncate to match model shape"
            #     nxdiff = nxmod - nxmask
            #     nydiff = nymod - nymask
            #     if nxdiff < 0:
            #         MaskArray = MaskArray
            self._MaskArray=np.zeros(MaskArray.shape,np.bool8)
            for ch in range(nch):
                for pol in range(npol):
                    self._MaskArray[ch,pol,:,:]=np.bool8(1-MaskArray[ch,pol].T[::-1].copy())[:,:]
            self.MaskArray=np.ascontiguousarray(self._MaskArray)

        # import matplotlib.pyplot as plt
        # plt.imshow(self.MaskArray[0,0])
        # plt.colorbar()
        # plt.show()
        #
        # import sys
        # sys.exit(0)

        self._peakMode = "normal"

        self.CacheFileName = CacheFileName
        self.CurrentNegMask = None
        self._NoiseMap = None
        self._PNRStop = None  # in _peakMode "sigma", provides addiitonal stopping criterion

        # this is so that the relevant functions are registered as job handlers with APP
        self.FTMachine = FFTW_Scale_Manager(wisdom_file=self.GD["Cache"]["DirWisdomFFTW"])

        APP.registerJobHandlers(self)


    def Init(self, cache=None, facetcache=None, **kwargs):
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

        self.Freqs = kwargs["GridFreqs"]
        AllDegridFreqs = []
        for i in kwargs["DegridFreqs"].keys():
            AllDegridFreqs.append(kwargs["DegridFreqs"][i])
        self.Freqs_degrid = np.asarray(AllDegridFreqs).flatten()
        self.SetPSF(kwargs["PSFVar"])
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])

        self.ModelMachine.setPSFServer(self.PSFServer)
        #self.SetModelRefFreq(kwargs["RefFreq"])
        self.ModelMachine.setFreqMachine(self.Freqs, self.Freqs_degrid,
                                         weights=kwargs["PSFVar"]["WeightChansImages"], PSFServer=self.PSFServer)

        self.ModelMachine.setScaleMachine(self.PSFServer, NCPU=self.NCPU, MaskArray=self.MaskArray,
                                          FTMachine=self.FTMachine, cachepath=cachepath)


    def Reset(self):
        pass

    def setMaskMachine(self,MaskMachine):
        self.MaskMachine = MaskMachine


    def SetModelRefFreq(self, RefFreq):
        """
        Sets ref freq in ModelMachine.
        """
        AllFreqs = []
        AllFreqsMean = np.zeros((self.NFreqBand,), np.float32)
        for iChannel in range(self.NFreqBand):
            AllFreqs += self.DicoVariablePSF["freqs"][iChannel]
            AllFreqsMean[iChannel] = np.mean(self.DicoVariablePSF["freqs"][iChannel])
        # assume that the frequency variance is somewhat the same in all the stokes images:
        # RefFreq = np.sum(AllFreqsMean.ravel() * np.mean(self.DicoVariablePSF["WeightChansImages"],axis=1).ravel())
        self.ModelMachine.setRefFreq(RefFreq)


    def SetModelShape(self):
        """
        Sets the shape params of model, call in every update step
        """
        self.ModelMachine.setModelShape(self._Dirty.shape)
        self.Nchan, self.Npol, self.Npix, _ = self._Dirty.shape
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

        if self._peakMode is "sigma":
            print>> log, "Will search for the peak in the SNR-weighted dirty map"
            a, b = self._MeanDirty, self._NoiseMap.reshape(self._MeanDirty.shape)
            self._PeakSearchImage = numexpr.evaluate("a/b")
        else:
            print>> log, "Will search for the peak in the unweighted dirty map"
            self._PeakSearchImage = self._MeanDirty

        if self.ModelImage is None:
            self._ModelImage = np.zeros_like(self._Dirty)
        if self.MaskArray is None:
            self._MaskArray = np.zeros(self._Dirty.shape, dtype=np.bool8)


    def SubStep(self,(dx,dy),LocalSM):
        """
        This is where subtraction in the image domain happens
        """
        xc, yc = dx, dy
        N1 = LocalSM.shape[-1]

        # Get overlap indices where psf should be subtracted
        Aedge, Bedge = GiveEdges((xc, yc), self.Npix, (N1//2, N1//2), N1)

        x0d, x1d, y0d, y1d = Aedge
        x0p, x1p, y0p, y1p = Bedge

        self._Dirty[:, :, x0d:x1d, y0d:y1d] -= LocalSM[:, :, x0p:x1p, y0p:y1p]

        # Subtract from the average
        if self.MultiFreqMode:  # If multiple frequencies are present construct the weighted mean
            self._MeanDirty[:, 0, x0d:x1d, y0d:y1d] -= np.sum(LocalSM[:, :, x0p:x1p, y0p:y1p] * self.WeightsChansImages,
                                                              axis=0)  # Sum over freq
        else:
            self._MeanDirty = self._Dirty

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
            print>> log, "    [iter=%i] peak residual %.3g" % (i, ThisFlux)

    def check_stopping_criteria(self, PeakMap, npix, DoAbs):
        # Get RMS stopping criterion
        NPixStats = self.GD["Deconv"]["NumRMSSamples"]
        if NPixStats:
            RandomInd = np.int64(np.random.rand(NPixStats)*npix**2)
            RMS = np.std(np.real(PeakMap.ravel()[RandomInd]))
        else:
            RMS = np.std(PeakMap)

        self.RMS = RMS

        self.GainMachine.SetRMS(RMS)

        Fluxlimit_RMS = self.RMSFactor*RMS

        # Find position and intensity of first peak
        x, y, MaxDirty = NpParallel.A_whereMax(PeakMap, NCPU=self.NCPU, DoAbs=DoAbs, Mask=self.MaskArray)

        # Get peak factor stopping criterion
        Fluxlimit_Peak = MaxDirty*self.PeakFactor

        # Get side lobe stopping criterion
        Fluxlimit_Sidelobe = ((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*MaxDirty if self.CycleFactor else 0

        mm0, mm1 = PeakMap.min(), PeakMap.max()

        # Choose whichever threshold is highest
        StopFlux = max(Fluxlimit_Peak, Fluxlimit_RMS, Fluxlimit_Sidelobe, self.FluxThreshold)

        print>>log, "    Dirty image peak flux      = %10.6f Jy [(min, max) = (%.3g, %.3g) Jy]"%(MaxDirty,mm0,mm1)
        print>>log, "      RMS-based threshold      = %10.6f Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor)
        print>>log, "      Sidelobe-based threshold = %10.6f Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor)
        print>>log, "      Peak-based threshold     = %10.6f Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor)
        print>>log, "      Absolute threshold       = %10.6f Jy"%(self.FluxThreshold)
        print>>log, "    Stopping flux              = %10.6f Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty)

        return StopFlux, MaxDirty

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

        # Get the PeakMap (first index will always be 0 because we only support I cleaning)
        PeakMap = self._MeanDirty[0, 0, :, :]

        # These options should probably be moved into MinorCycleConfig in parset
        DoAbs = int(self.GD["Deconv"]["AllowNegative"])
        print>>log, "  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter, self.MaxMinorIter, DoAbs)

        # Determine which stopping criterion to use for flux limit
        StopFlux, MaxDirty = self.check_stopping_criteria(PeakMap, self.Npix, DoAbs)


        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        ThisFlux=MaxDirty

        if ThisFlux < StopFlux:
            print>>log, ModColor.Str("    Initial maximum peak %g Jy below threshold, we're done CLEANing" % (ThisFlux),col="green" )
            exit_msg = exit_msg + " " + "FluxThreshold"
            continue_deconvolution = False or continue_deconvolution
            update_model = False or update_model
            # No need to do anything further if we are already at the stopping flux
            return exit_msg, continue_deconvolution, update_model

        # set peak in GainMachine (LB - deprecated?)
        # self.GainMachine.SetFluxMax(ThisFlux)

        # alphamap, alphastdmap, alphacomps = self.ModelMachine.GiveNewSpectralIndexMap(GaussPars=self.PSFServer.DicoVariablePSF["FWHMBeam"][0],
        #                                                                   ResidCube=self.DicoDirty["ImageCube"], GiveComponents=True)
        #
        # alphamap = alphamap[0,0]
        # alphamap = np.where(alphamap==0.0, -0.7, alphamap)
        #
        # import matplotlib.pyplot as plt
        # plt.imshow(alphamap)
        # plt.show()
        #
        # minalpha = alphacomps.min()
        # maxalpha = alphacomps.max()
        # delalpha = maxalpha - minalpha
        # Nbins = 7
        # alphatmp = alphacomps*Nbins/delalpha
        # alphaprobs, _ = np.histogram(alphatmp, Nbins, density=True)
        # alphaprobsinv = 1.0/alphaprobs
        # alphaprobsinv /= np.sum(alphaprobsinv)
        # print alphaprobsinv
        # alphabins = minalpha + maxalpha * np.cumsum(alphaprobsinv)
        # print alphabins
        # alphamask = np.digitize(alphamap, alphabins, right=True)
        #
        #
        # plt.imshow(alphamask)
        # plt.colorbar()
        # plt.show()
        #
        # import sys
        # sys.exit()

        # Do minor cycle deconvolution loop
        TrackFlux = MaxDirty.copy()
        diverged = False
        stalled = False
        stall_count = 0
        diverged_count = 0
        try:
            for i in range(self._niter+1,self.MaxMinorIter+1):
                self._niter = i

                # find peak
                x, y, ThisFlux = NpParallel.A_whereMax(self._MeanDirty, NCPU=self.NCPU, DoAbs=DoAbs, Mask=self.MaskArray)

                # Crude hack to prevent divergences
                if np.abs(ThisFlux) > self.GD["WSCMS"]["MinorDivergenceFactor"] * np.abs(TrackFlux):
                    diverged_count += 1
                    if diverged_count > 5:
                        diverged = True
                elif np.abs((ThisFlux - TrackFlux)/TrackFlux) < self.GD['WSCMS']['MinorStallThreshold']:
                    stall_count += 1
                    if stall_count > 5:
                        stalled = True
                else:
                    TrackFlux = ThisFlux

                # LB - deprecated?
                # self.GainMachine.SetFluxMax(ThisFlux)

                T.timeit("max0")

                if ThisFlux <= StopFlux or diverged or stalled:
                    if diverged:
                        print>>log, ModColor.Str("    At [iter=%i] minor cycle is diverging so it has been force stopped at a flux of %.3g Jy" % (i,ThisFlux),col="green")
                    elif stalled:
                        print>> log, ModColor.Str("    At [iter=%i] minor cycle has stalled so it has been force stopped at a flux of %.3g Jy" % (i, ThisFlux), col="green")
                    else:
                        print>>log, ModColor.Str("    CLEANing [iter=%i] peak of %.3g Jy lower than stopping flux" % (i,ThisFlux),col="green")
                    cont = ThisFlux > self.FluxThreshold
                    if not cont:
                          print>>log, ModColor.Str("    CLEANing [iter=%i] absolute flux threshold of %.3g Jy has been reached" % (i,self.FluxThreshold),col="green",Bold=True)
                    exit_msg = exit_msg + " " + "MinFluxRms"
                    continue_deconvolution = cont or continue_deconvolution
                    update_model = True or update_model

                    break # stop cleaning if threshold reached

                self.track_progress(i, ThisFlux)

                # run minor loop
                if self.GD["WSCMS"]["MultiScale"]:
                    # Find the relevant scale and do sub-minor loop. Returns the model constructed during the
                    # sub-minor loop.
                    # If the delta scale is found then self._Dirty and
                    # self._MeanDirty have already had the components subtracted from them so we don't need to do
                    # anything further.
                    ScaleModel, iScale, x, y = self.ModelMachine.do_minor_loop(x, y, self._Dirty,
                                                         self._MeanDirty, self._JonesNorm,
                                                         self.WeightsChansImages, ThisFlux, StopFlux)

                    # convolve scale model with PSF and subtract from residual (if not delta scale)
                    if iScale:
                        for iFacet in xrange(self.GD["Facets"]["NFacets"]**2):
                            # set PSF in this facet
                            self.PSFServer.setFacet(iFacet)

                            # Note this logic assumes square facets!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
                            # get location of facet center
                            xc, yc = self.DicoVariablePSF["Facets"][iFacet]["pixCentral"]
                            # xl, xu, yl, yu = self.DicoVariablePSF["FacetInfo"][iFacet]["pixExtent"]
                            # LB - why is the -1 necessary here when Nfacets = 1
                            LocalSM = ScaleModel[:, :, xc-self.NpixFacet//2:xc+self.NpixFacet//2 + 1,
                                                 yc-self.NpixFacet//2:yc+self.NpixFacet//2 + 1]

                            # print "iFacet = ", iFacet
                            # print "(xc, yc) = ", (xc, yc)
                            # print "x indices = ", xl, xu
                            # print "y indices = ", yl, yu
                            # print "NpixFacet = ", self.DicoVariablePSF["FacetInfo"][iFacet]["NpixFacet"], xu - xl, yu-yl

                            # convolve local sky model with PSF
                            if (LocalSM > 1e-8).any():
                                SM = self.ModelMachine.ScaleMachine.SMConvolvePSF(iFacet, LocalSM)

                                # print "Local Sm - ", SM.max(), SM.min()

                                _, _, ntmp, _ = SM.shape
                                if ntmp < 2*self.NpixFacet:
                                    print "Warning - your LocalSM is too small"

                                # subtract facet model
                                self.SubStep((xc, yc), SM)

                        # import sys
                        # sys.exit(0)

                else:  # TODO - remove this and advise users to use Hogbom for SS clean
                    # Get the JonesNorm
                    JonesNorm = (self._JonesNorm[:, :, x, y]).reshape((self.Nchan, self.Npol, 1, 1))

                    # Get the solution
                    Fpol = np.zeros([self.Nchan, self.Npol, 1, 1], dtype=np.float32)

                    # the dirty image has been stitched so we need to divide by the
                    # stitched sqrt(JonesNorm) to get the intrinsic (done in freqmachine)
                    Fpol[:, 0, 0, 0] = self._Dirty[:, 0, x, y].copy()  # /np.sqrt(JonesNorm[:, 0, 0, 0])

                    # Find PSF corresponding to location (x,y) (need to set PSF before doing WPoly)
                    self.PSFServer.setLocation(x, y)  # Selects the facet closest to (x,y)

                    # Fit a polynomial to get coeffs (coeffs are for intrinsic flux)
                    self.ModelMachine.Coeffs = self.ModelMachine.FreqMachine.Fit(Fpol[:, 0, 0, 0],
                                                                                 JonesNorm[:, 0, 0, 0],
                                                                                 self._MeanDirty[0, 0, x, y])

                    # Overwrite with polynoimial fit (this returns the apparent flux)
                    Fpol[:, 0, 0, 0] = self.ModelMachine.FreqMachine.Eval(self.ModelMachine.Coeffs)

                    T.timeit("stuff")

                    # get the PSF in this facet
                    PSF, _ = self.PSFServer.GivePSF()

                    T.timeit("FindScale")

                    CurrentGain = self.GD["Deconv"]["Gain"]  # GainMachine.GiveGain()

                    #Update model
                    self.ModelMachine.AppendComponentToDictStacked((x, y), self.ModelMachine.Coeffs[:], 0, CurrentGain)

                    # Subtract LocalSM*CurrentGain from dirty image (not JonesNorm since Fpol is already for apparent)
                    self.SubStep((x, y), PSF*Fpol*CurrentGain)
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