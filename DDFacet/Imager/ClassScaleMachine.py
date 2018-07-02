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
import numexpr
from DDFacet.Other import MyLogger
log = MyLogger.getLogger("ClassScaleMachine")
from DDFacet.Array import NpParallel
from DDFacet.ToolsDir.ModFFTW import LB_FFT_and_Gauss_Tools
from DDFacet.ToolsDir.ModFFTW import FFTW_Scale_Manager

Fs = np.fft.fftshift
iFs = np.fft.ifftshift

class ClassScaleMachine(object):
    def __init__(self, GD=None, NCPU=6, MaskArray=None):
        self.GD = GD
        self.DoAbs = int(self.GD["Deconv"]["AllowNegative"])
        self.MaskArray = MaskArray
        self.PeakFactor = self.GD["WSCMS"]["SubMinorPeakFact"]
        self.NSubMinorIter = self.GD["WSCMS"]["NSubMinorIter"]
        self.NCPU = NCPU
        self.DoAbs = self.GD["Deconv"]["AllowNegative"]

    def Init(self, PSFserver=None, FreqMachine=None, FTMachine2=None):
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
        _, _, self.Npix, _ = self.PSFServer.ImageShape
        self.NpixPadded = int(np.ceil(self.GD["Facets"]["Padding"]*self.Npix))
        # make sure it is odd numbered
        if self.NpixPadded % 2 == 0:
            self.NpixPadded += 1
        self.Npad = (self.NpixPadded - self.Npix)//2
        self.NpixPSF = self.PSFServer.NPSF  # need this to initialise the FTMachine
        self.NpixPaddedPSF = self.PSFServer.DicoVariablePSF["PaddedPSFInfo"][0]  # hack for now
        self.NpadPSF = (self.NpixPaddedPSF - self.NpixPSF) // 2

        self.set_coordinates(self.NpixPadded, self.NpixPaddedPSF)

        # create dicts to store FT of PSF (TODO - keep these in a fixed length ordered dict to save memory)
        self.FT_PSF = {}
        self.FT_meanPSF = {}
        self.ConvPSFFreqPeaks = {}

        # get scales (in pixel units)
        self.set_scales()

        self.PSFServer.setFacet(0)
        PSF, meanPSF = self.PSFServer.GivePSF()

        # get the FFT and convolution utility
        self.FTMachine = LB_FFT_and_Gauss_Tools(self.NpixPaddedPSF//2, self.Nchan, 1, self.Nscales)
        self.FTMachine2 = FTMachine2
        self.FTMachine2.Init(self.Npix, self.NpixPadded, self.NpixPSF, self.NpixPaddedPSF, self.Nchan, 1, self.Nscales)

        import time
        for i in xrange(5):
            ti = time.time()
            result1 = self.FTMachine.ConvolveGaussian(PSF.copy(), 5.0, cube=True)
            tf = time.time()
            print "Old version took ", tf - ti
            ti = time.time()
            result2 = self.ConvolveGaussian(PSF.copy(), 5.0, mode='Facet')
            tf = time.time()
            print "New version took ", tf - ti
            ti = time.time()
            result3 = self.ConvolveGaussian_new(PSF.copy(), 5.0, mode='Facet')
            tf = time.time()
            print "Newest version took ", tf - ti
            print np.amax(np.abs(result3.reshape(self.Nchan, 1, self.NpixPSF**2) - result2.reshape(self.Nchan, 1, self.NpixPSF**2)),
                      axis=2)

        # SM = self.GaussianSymmetricFT(5.0) #, amp=np.ones(self.Nchan))
        # result = self.ConvolvePSF(0, SM, use_mean=True)
        #
        # print result.shape
        # import matplotlib.pyplot as plt
        # for i in xrange(self.Nchan):
        #     plt.figure('1')
        #     plt.imshow(np.abs(result[i, 0]))
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()
        #
        # plt.figure('2')
        # plt.imshow(np.abs(result2[0,0]))
        # plt.colorbar()

        #plt.show()

        # print np.amax(np.abs(result1.reshape(self.Nchan, 1, 1755 * 1755) - result3.reshape(self.Nchan, 1, 1755 * 1755)), axis=2)
        #
        import sys
        sys.exit(0)

        # set bias factors
        self.set_bias()

        # get the Gaussian pars, volume factors and FWHMs corresponding to dictionary functions
        self.set_kernels()

        # for scale dependent masking
        self.ScaleMaskArray = {}

        # set the scale and facet dependent gains
        self.set_gains()

    def set_coordinates(self, NpixPaddedImage, NpixPaddedFacet):
        # TODO - cache and evaluate with numexpr
        # get pixel coordinates for image
        n = NpixPaddedImage//2
        self.x_image, self.y_image = np.mgrid[-n:n:1.0j * self.NpixPadded, -n:n:1.0j * self.NpixPadded]
        # set corresponding frequencies (note they are fourier shifted so no need to iFs them later)
        freqs = np.fft.fftfreq(self.NpixPadded)
        self.u_image, self.v_image = np.meshgrid(freqs, freqs)
        self.rhosq_image = self.u_image ** 2 + self.v_image ** 2

        # get pixel coordinates for facet
        n = NpixPaddedFacet//2
        self.x_facet, self.y_facet = np.mgrid[-n:n:1.0j * self.NpixPaddedPSF, -n:n:1.0j * self.NpixPaddedPSF]
        # set corresponding frequencies (note they are fourier shifted so no need to iFs them later)
        freqs = np.fft.fftfreq(self.NpixPaddedPSF)
        self.u_facet, self.v_facet = np.meshgrid(freqs, freqs)
        self.rhosq_facet = self.u_facet ** 2 + self.v_facet ** 2

    def GaussianSymmetric(self, sig, x0=0, y0=0, amp=np.array([1.0])):
        """
        Evaluates symmetric normalised 2D Gaussian centered at (x0, y0). Note this is only used in 
        GiveModelImage so always adds onto a grid the size of the unpadded image.
        The image space coordiantes are must be computed in advance and tored in self.x and self.y
        :param sig: std deviation of Gaussian in signal space
        :param x0: x coordinate relative to centre
        :param y0: y coordinate relative to centre
        :param amp: amplitude (at delta scale) of Gaussian component (if amp.size > 1 cube must be True)
        :return: 
        """
        # broadcast amplitude array to cube
        amp = amp[:, None, None, None]
        # for slicing array
        I  = slice(0, self.Npix)
        # evaluate slice with numexpr and broadcast to cube
        loc_dict = {'x': self.x, 'x0': x0, 'y': self.y, 'y0': y0, 'sig': sig, 'pi': np.pi}
        out = numexpr.evaluate('exp(-((x-x0)**2 + (y-y0)**2)/ (2 * sig ** 2))/(2 * pi * sig ** 2)',
                               local_dict=loc_dict)[None, None, I, I]
        # multiply by amplitude and return
        return numexpr.evaluate('amp * out')

    def GaussianSymmetricFT(self, sig, x0=0, y0=0, amp=np.array([1.0]), mode='Facet'):
        """
        Gives the FT of a symmetric Gaussian analytically. Note since we will always be using this to 
        convolve with real valued inputs we only need to compute the result at the positive frequencies. 
        Note only the mean image ever gets convolved with multiple scales and we are usually convolving a 
        cube with a single scale.
        :param sig: std deviation of Gaussian in signal space
        :param x0: center x coordinate relative to center
        :param y0: center y coordinate relative to center
        :param amp: amplitude (at delta scale) of Gaussian component (if amp.size > 1 cube must be True)
        :return: 
        """
        if not x0 and not y0:
            if mode=='Facet':
                loc_dict = {'sig': sig, 'pi': np.pi, 'rhosq': self.rhosq_facet}
            elif mode=='Image':
                loc_dict = {'sig': sig, 'pi': np.pi, 'rhosq': self.rhosq_image}
            result = numexpr.evaluate('exp(-2 * pi ** 2 * rhosq * sig ** 2)', local_dict=loc_dict)
        else:
            if mode=='Facet':
                loc_dict = {'sig': sig, 'pi': np.pi, 'rhosq': self.rhosq_facet, 'v': self.v_facet, 'x0': x0,
                            'u': self.u_facet, 'y0': y0}
            elif mode=='Image':
                loc_dict = {'sig': sig, 'pi': np.pi, 'rhosq': self.rhosq_image, 'v': self.v_image, 'x0': x0,
                            'u': self.u_image, 'y0': y0}
            result = numexpr.evaluate('exp(-2.0j * pi * v * x0 - 2.0j * pi * u * y0 - 2 * pi ** 2 * rhosq * sig ** 2)',
                                   local_dict=loc_dict)
        loc_dict = {'amp': amp[:, None, None, None], 'result': result[None, None, :, :]}
        return numexpr.evaluate('amp*result', local_dict=loc_dict)


    def ConvolveGaussian(self, A, sig, mode='Facet'):
        """
        Colnvolves A with a symmetric Guassian kernel.   
        :param A: [nch, npol, NpixFacet, NpixFacet] array to be convolved
        :param sig: std deviation of Gaussian kernel in signal space
        :param mode: whether we are convoling something the size of a facet or the size of the image
        :return: 
        """
        nslices, _, _, _ = A.shape
        # get FT of data
        self.FTMachine2.FFT(A, unpad=False, mode=mode)
        # multiply by FT of Gaussian
        if mode=='Facet':
            loc_dict = {'sig': sig, 'pi': np.pi, 'rhosq': self.rhosq_facet[None, None],
                        'Ahat': self.FTMachine2._PaddedFacetArray[0:nslices]}
            numexpr.evaluate('Ahat * exp(-2 * pi ** 2 * rhosq * sig ** 2)', local_dict=loc_dict,
                             out=self.FTMachine2._PaddedFacetArray[0:nslices])
        elif mode=='Image':
            loc_dict = {'sig': sig, 'pi': np.pi, 'rhosq': self.rhosq_image[None, None],
                        'Ahat': self.FTMachine2._PaddedImageArray[0:nslices]}
            numexpr.evaluate('Ahat * exp(-2 * pi ** 2 * rhosq * sig ** 2)', local_dict=loc_dict,
                             out=self.FTMachine2._PaddedImageArray[0:nslices])
        else:
            raise Exception('This is a bug. We should never get here')
        # take inverse FT
        A = self.FTMachine2.iFFT(nslices, unpad=True, mode=mode)
        if nslices==1:
            return A[None]
        else:
            return A

    def ConvolveGaussian_new(self, A, sig, mode='Facet'):
        """
        Colnvolves A with a symmetric Guassian kernel.   
        :param A: [nch, npol, NpixFacet, NpixFacet] array to be convolved
        :param sig: std deviation of Gaussian kernel in signal space
        :param mode: whether we are convoling something the size of a facet or the size of the image
        :return: 
        """
        nslices, _, _, _ = A.shape
        # get FT of data
        self.FTMachine2.FFT_new(A, mode=mode)
        # multiply by FT of Gaussian
        if mode=='Facet':
            loc_dict = {'sig': sig, 'pi': np.pi, 'rhosq': self.rhosq_facet[None, None],
                        'Ahat': self.FTMachine2._PaddedFacetArray[0:nslices]}
            numexpr.evaluate('Ahat * exp(-2 * pi ** 2 * rhosq * sig ** 2)', local_dict=loc_dict,
                             out=self.FTMachine2._PaddedFacetArray[0:nslices])
        elif mode=='Image':
            loc_dict = {'sig': sig, 'pi': np.pi, 'rhosq': self.rhosq_image[None, None],
                        'Ahat': self.FTMachine2._PaddedImageArray[0:nslices]}
            numexpr.evaluate('Ahat * exp(-2 * pi ** 2 * rhosq * sig ** 2)', local_dict=loc_dict,
                             out=self.FTMachine2._PaddedImageArray[0:nslices])
        else:
            raise Exception('This is a bug. We should never get here')
        # take inverse FT
        A = self.FTMachine2.iFFT_new(nslices, unpad=True, mode=mode)
        if nslices==1:
            return A[None]
        else:
            return A

    def ConvolvePSF(self, iFacet, SM, FT=True, mode='Facet', use_mean=False):
        """
        This convolves an image with the PSF. We need to cater for the following cases:
        1) PSF * K_alpha
        2) meanPSF * K_alpha
        3) PSF * Image
        4) meanPSF * meanImage
        
        :param iFacet: facet label 
        :param SM: sky model or FT of sky model
        :param FT: specifies if sky model is in Fourier space or not 
        :param sig: 
        :return: 
        """
        # set array to perform compute on
        nslices, _, npixin, _ = SM.shape
        if mode=='Facet':
            compute_array = self.FTMachine2._PaddedFacetArray.view()
        elif mode=='Image':
            compute_array = self.FTMachine2._PaddedImageArray.view()
        else:
            raise Exception('Bug!!! Never supposed to get here')

        # get the FT of the PSF (note we always do PSF convolves in mode='Facet')
        if use_mean:
            if iFacet not in self.FT_meanPSF:
                _, PSFmean = self.PSFServer.GivePSF()
                self.FTMachine2.FFT(PSFmean, unpad=False, mode='Facet')
                self.FT_meanPSF[iFacet] = self.FTMachine2._PaddedFacetArray[0][None].copy()
            FT_PSF = self.FT_meanPSF[iFacet]
        else:
            if iFacet not in self.FT_PSF:
                PSF, _ = self.PSFServer.GivePSF()
                self.FTMachine2.FFT(PSF, unpad=False, mode='Facet')
                self.FT_PSF[iFacet] = self.FTMachine2._PaddedFacetArray.copy()
            FT_PSF = self.FT_PSF[iFacet][0:nslices]

        # get FT of sky model
        if not FT:
            self.FTMachine2.FFT(SM, mode=mode)
            SM = compute_array[0:nslices].copy()

        # write product to compute array
        numexpr.evaluate('FT_PSF * SM', out=compute_array[0:nslices])

        # do inverse FT
        A = self.FTMachine2.iFFT(nslices, unpad=True, mode=mode)
        return A

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

