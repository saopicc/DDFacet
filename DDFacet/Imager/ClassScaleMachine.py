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

import numpy as np
from DDFacet.Other import MyLogger
log = MyLogger.getLogger("ClassScaleMachine")
from DDFacet.Array import NpParallel
from DDFacet.ToolsDir.ModFFTW import LB_FFT_and_Gauss_Tools

class ClassScaleMachine(object):
    def __init__(self, GD=None, NCPU=6, MaskArray=None):
        self.GD = GD
        self.DoAbs = int(self.GD["Deconv"]["AllowNegative"])
        self.MaskArray = MaskArray
        self.PeakFactor = self.GD["WSCMS"]["SubMinorPeakFact"]
        self.NSubMinorIter = self.GD["WSCMS"]["NSubMinorIter"]
        self.NCPU = NCPU
        self.DoAbs = self.GD["Deconv"]["AllowNegative"]

    def Init(self, PSFserver=None, FreqMachine=None):
        """
        Sets everything required to perform multi-scale CLEAN. 
        :param PSFserver: Mandatory if NScales > 1
        :param FWHMBeamAvg: Mandatory if NScales > 1 and setting scales automatically
        :param NFacetpix: The number of pixels in a facet. Facets are assumed to be identical for the moment.
                          Note it is not possible to deconvolve scales larger than a facet. If larger scales are present 
                          we need to set scales per facet. 
        :param FreqMachine: FreqMachine is required if doing simultaneous multi-frequency and multi-scale deconv 
        :return: None
        """
        # set PSF server
        if PSFserver is None:
            raise Exception('You have to pass PSFserver to ScaleMachine for multi-scale CLEAN')
        else:
            self.PSFServer = PSFserver
            (self.FWHMBeamAvg, _, _)=self.PSFServer.DicoVariablePSF["EstimatesAvgPSF"]
            self.FWHMBeam = self.PSFServer.DicoVariablePSF["FWHMBeam"]

        # TODO: need to catch cases when FreqMachine is None and mf-deconv is True
        self.FreqMachine = FreqMachine
        self.Nchan = self.FreqMachine.nchan
        self.NpixPSF = self.PSFServer.NPSF  # need this to initialise the FTMachine
        self.NpixPaddedPSF = self.PSFServer.DicoVariablePSF["PaddedPSFInfo"][0]  # hack for now

        self.Npad = (self.NpixPaddedPSF - self.NpixPSF)//2

        # create dicts to store FT of PSF (TODO - keep these in a fixed length ordered dict to save memory)
        self.FT_PSF = {}
        self.FT_meanPSF = {}
        self.ConvPSFFreqPeaks = {}

        # get scales (in pixel units)
        self.set_scales()

        # get the FFT and convolution utility
        self.FTMachine = LB_FFT_and_Gauss_Tools(self.NpixPaddedPSF//2, self.Nchan, 1, self.Nscales)

        # set bias factors
        self.set_bias()

        # get the Gaussian pars, volume factors and FWHMs corresponding to dictionary functions
        self.set_kernels()

        # for scale dependent masking
        self.ScaleMaskArray = {}

        # set the scale and facet dependent gains
        self.set_gains()

    # TODO: Set max scale with minimum baseline
    def set_scales(self):
        if self.GD["WSCMS"]["Scales"] is None:
            print>>log, "Setting scales automatically from FWHM of average beam"
            # FWHM is in degrees so first convert to radians
            # The sqrt(2) factor corrects for the fact that we fit to first null instead of the FWHM
            FWHM0 = 1.0/np.sqrt(2)*((self.FWHMBeamAvg[0] + self.FWHMBeamAvg[1])*np.pi / 180) / \
                    (2.0 * self.GD['Image']['Cell'] * np.pi / 648000)
            FWHMs = [FWHM0, 2.0*FWHM0]  # impirically determined 2.25 to work pretty well
            i = 1
            while FWHMs[i] < 100:  # hardcoded for now
                FWHMs.append(1.5*FWHMs[i])
                i += 1
            self.FWHMs = np.asarray(FWHMs)
        else:
            print>>log, "Using user defined scales"
            self.FWHMs = np.asarray(self.GD["WSCMS"]["Scales"])

        self.Nscales = self.FWHMs.size

        print>>log, "Using %i scales with FWHMs of %s pixels" % (self.Nscales, self.FWHMs)

    def set_bias(self):
        # get scale bias factor
        self.beta = self.GD["WSCMS"]["MultiScaleBias"]

        # set scale bias according to Offringa definition implemented i.t.o. inverse bias
        self.bias = np.ones(self.Nscales, dtype=np.float64)
        self.bias[1::] = self.beta**(-1.0 - np.log2(self.FWHMs[1::]/self.FWHMs[1]))

    def set_bias_new(self):
        """
        This is a new way of setting the bias based on the following idea. Suppose we have a number of 
        Gaussian components with scales sigma0, sigma1, ... and they all have an integrated flux of 1Jy.
        Now we want our algorithm to clean the large scales first. The components in the dirty image
        have all been convolved by the PSF i.e they are K_alpha * PSF where K_alpha is the normalised 
        scale kernel. Thus convolving by a scale again we end up with K_alpha * K_alpha * PSF and we usually
        apply the bias function to the peaks of these images. We can get all the images to the same peak of one
        by dividing by the peaks of K_alpha * K_alpha * PSF. However, if we want the largest scale to be
        cleaned first, we should artificially shift the peak of the largest scale by just more than the 
        peak of the next largest scale and so on. We can, for example, make this a function of the FWHM of
        the Gaussian.
        the 
        :return: 
        """

    def set_kernels(self):
        """
        Computes Gauss pars corresponding to scales given in number of pixels pixels
        :return: 
        """
        self.sigmas = np.zeros(self.Nscales, dtype=np.float64)
        self.volumes = np.zeros(self.Nscales, dtype=np.float64)
        for i in xrange(self.Nscales):
            self.sigmas[i] = self.FWHMs[i]/(2*np.sqrt(2*np.log(2.0)))
            self.volumes[i] = 2*np.pi*self.sigmas[i]**2  # volume = normalisation constant of 2D Gaussian

            print>>log, "scale %i kernel peak = %f" % (i, 1.0/self.volumes[i])
            print>> log, "scale %i sigma = %f" % (i, self.sigmas[i])
        self.VolumeNorms = self.volumes/self.volumes[0]
        return

    def set_gains(self):
        self.PSFpeaksmean = {}
        self.PSFpeaks = {}
        self.Nfacets = self.GD['Facets']['NFacets']**2
        self.gains = np.zeros([self.Nfacets, self.Nscales], dtype=np.float64)

        # get PSF at central facet
        CentralFacetID = self.Nfacets//2
        self.PSFServer.setFacet(CentralFacetID)
        PSF, PSFmean = self.PSFServer.GivePSF()
        _, _, NPixFacet, _ = PSFmean.shape
        PSFmean /= PSFmean.max()

        # keep track of FT_meanPSF
        self.FT_meanPSF[CentralFacetID] = self.FTMachine.SimpleFT(PSFmean, unpad=False)
        #self.FT_PSF[CentralFacetID] = self.FTMachine.SimpleFT(PSF, cube=True, unpad=False)

        # convolve with scales (duplicate of FT but this only happens once so should be ok)
        ConvPSFs = self.FTMachine.ConvolveGaussianScales(PSFmean, self.sigmas)

        Scale0PSFmax = ConvPSFs[0, NPixFacet // 2, NPixFacet // 2]
        self.PSFpeaksmean[CentralFacetID] = ConvPSFs[:, NPixFacet // 2, NPixFacet // 2]

        # get the normalisation factor for the ConvPSF's (must set ConvPSF for scale zero to unity)
        Conv2PSF0 = self.FTMachine.TwiceConvolvePSF(self.FT_meanPSF[CentralFacetID], self.sigmas[0], self.Npad)
        self.ConvPSFNormFactor = Conv2PSF0[0, 0, NPixFacet // 2, NPixFacet // 2]

        # set gains
        gamma = self.GD['Deconv']['Gain']
        self.gains[CentralFacetID, :] = gamma * Scale0PSFmax / ConvPSFs[:, NPixFacet // 2, NPixFacet // 2]

        for iFacet in np.delete(xrange(self.Nfacets), CentralFacetID):
            self.PSFServer.setFacet(iFacet)
            PSF, PSFmean = self.PSFServer.GivePSF()
            _, _, NPixFacet, _ = PSFmean.shape

            # if we want to clean slower towards the edges of the beam we should not do this step
            PSFmean /= PSFmean.max()
            PSF /= np.amax(PSF.reshape(self.Nchan, 1, NPixFacet * NPixFacet), axis=2,
                           keepdims=True).reshape(self.Nchan, 1, 1, 1)

            # keep track of FT_meanPSF. TODO - put these in an ordered dict of fixed length
            self.FT_meanPSF[iFacet] = self.FTMachine.SimpleFT(PSFmean, unpad=False)
            #self.FT_PSF[iFacet] = self.FTMachine.SimpleFT(PSF, cube=True, unpad=False)

            ConvPSFs = self.FTMachine.ConvolveGaussianScales(PSFmean, self.sigmas)
            self.PSFpeaksmean[iFacet] = ConvPSFs[:, NPixFacet // 2, NPixFacet // 2]
            self.gains[iFacet, :] = gamma * Scale0PSFmax / ConvPSFs[:, NPixFacet // 2, NPixFacet // 2]

        print>> log, "gains = ", self.gains

    def do_scale_convolve_new(self, Dirty, MeanDirty):
        """
        Finds current most relevant scale based on total power instead of highest peak. To make this work we need to 
        do the peak finding differently
        :param Dirty: 
        :param MeanDirty: 
        :return: 
        """
        # get the power factors
        scale_power = self.FTMachine.GiveTotalScalePower(MeanDirty, self.sigmas)

        # apply bias factors
        biased_scale_power = self.bias*scale_power

        # set most relevant scale
        CurrentScale = np.argwhere(biased_scale_power == biased_scale_power.max()).squeeze()
        # print>> log, "Found most relevant scale at %i" % CurrentScale

        # convolve Dirty and MeanDirty by this scale unless it's the delta scale
        if CurrentScale == 0:
            x, y, MaxDirty = NpParallel.A_whereMax(MeanDirty, NCPU=self.NCPU, DoAbs=self.DoAbs,
                                                            Mask=self.MaskArray)
        else:
            MeanDirty = self.FTMachine.ConvolveGaussian(MeanDirty, self.sigmas[CurrentScale])
            Dirty = self.FTMachine.ConvolveGaussian(Dirty, self.sigmas[CurrentScale], cube=True)
            x, y, MaxDirty = NpParallel.A_whereMax(MeanDirty, NCPU=self.NCPU, DoAbs=self.DoAbs, Mask=self.MaskArray)
        return x, y, MaxDirty, MeanDirty, Dirty, CurrentScale

    def do_scale_convolve(self, Dirty, MeanDirty):
        # get scale convolved dirty
        ConvMeanDirtys = self.FTMachine.ConvolveGaussianScales(MeanDirty, self.sigmas)

        # find the one with the highest peak
        maxvals = np.zeros(self.Nscales)
        for iScale in xrange(self.Nscales):
            xtmp, ytmp, ConvMaxDirty = NpParallel.A_whereMax(ConvMeanDirtys[iScale], NCPU=self.NCPU, DoAbs=self.DoAbs,
                                                   Mask=self.MaskArray)
            maxvals[iScale] = ConvMaxDirty * self.bias[iScale]
            if iScale == 0:
                x = xtmp
                y = ytmp
                BiasedMaxVal = ConvMaxDirty * self.bias[iScale]
                MaxDirty = ConvMaxDirty
                CurrentDirty = ConvMeanDirtys[iScale][None, None, :, :]
                CurrentScale = iScale
            else:
                # only update if new scale is more significant
                if ConvMaxDirty * self.bias[iScale] >= BiasedMaxVal:
                    x = xtmp
                    y = ytmp
                    BiasedMaxVal = ConvMaxDirty * self.bias[iScale]
                    MaxDirty = ConvMaxDirty
                    CurrentDirty = ConvMeanDirtys[iScale][None, None, :, :]
                    CurrentScale = iScale

        # convolve Dirty and MeanDirty by this scale unless it's the delta scale
        if CurrentScale != 0:
            MeanDirty = CurrentDirty
            Dirty = self.FTMachine.ConvolveGaussian(Dirty, self.sigmas[CurrentScale], cube=True)
            x, y, MaxDirty = NpParallel.A_whereMax(MeanDirty, NCPU=self.NCPU, DoAbs=self.DoAbs, Mask=self.MaskArray)

        return x, y, MaxDirty, MeanDirty, Dirty, CurrentScale

    def GivePSFFreqPeaks(self, iFacet, iScale):
        if iFacet not in self.FT_PSF:
            PSF, PSFmean = self.PSFServer.GivePSF()
            self.FT_PSF[iFacet] = self.FTMachine.SimpleFT(PSF, cube=True, unpad=False)
        if iFacet not in self.ConvPSFFreqPeaks:
            self.ConvPSFFreqPeaks[iFacet] = {}
        if not iScale in self.ConvPSFFreqPeaks[iFacet]:
            self.ConvPSFFreqPeaks[iFacet] = {}
            ConvolvedPSFCube = self.FTMachine.ConvolvePSF(self.FT_PSF[iFacet], cube=True, sig=self.sigmas[iScale])
            self.ConvPSFFreqPeaks[iFacet][iScale] = np.amax(ConvolvedPSFCube.reshape(self.Nchan, 1, self.NpixPaddedPSF**2), axis=2, keepdims=True).reshape(self.Nchan, 1, 1, 1)
        return self.ConvPSFFreqPeaks[iFacet][iScale]

    def SMConvolvePSF(self, iFacet, FT_SM):
        """
        Here we pass in the FT of a sky model and convolve it with the psf 
        :param iFacet: 
        :param FT_SM: 
        :return: 
        """
        if iFacet not in self.FT_PSF:
            PSF, PSFmean = self.PSFServer.GivePSF()
            self.FT_PSF[iFacet] = self.FTMachine.SimpleFT(PSF, cube=True, unpad=False)
        return self.FTMachine.ConvolvePSF(self.FT_PSF[iFacet], FT_SM, npad=self.Npad, cube=FT_SM.shape[0]>1)

    def GiveTwiceConvolvedPSF(self, iFacet, iScale):
        # compute and store the FT of the PSF cube if it doesn't exist
        if iFacet not in self.FT_PSF:
            PSF, PSFmean = self.PSFServer.GivePSF()
            self.FT_PSF[iFacet] = self.FTMachine.SimpleFT(PSF, cube=True, unpad=False)
        self.ConvPSF = self.FTMachine.TwiceConvolvePSF(self.FT_PSF[iFacet], self.sigmas[iScale], self.Npad, cube=True)
        self.ConvPSFmean = self.FTMachine.TwiceConvolvePSF(self.FT_meanPSF[iFacet], self.sigmas[iScale], self.Npad, cube=False)
        return self.ConvPSF, self.ConvPSFmean

