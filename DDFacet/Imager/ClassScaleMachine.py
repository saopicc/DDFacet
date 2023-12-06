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

import numpy as np
import numexpr
from DDFacet.Other import logger
log = logger.getLogger("ClassScaleMachine")
from DDFacet.Array import NpParallel
from DDFacet.ToolsDir.ModFFTW import FFTW_Manager

Fs = np.fft.fftshift
iFs = np.fft.ifftshift

from glob import glob
import pylru

# we should probably move this to CacheManager or something
class Store(object):
    """
    This is a dictionary like interface to access .npy files on disk
    in the folder specified by cache_dir. These files should be keyed
    on the string pattern in between cache_dir and .npy
    Useful when creating an lru cache with disk spillover
    when the size exceeds some maximum. 
    """

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        # check if there any cached items in cache_dir
        self.valid_keys = []
        for fle in glob(cache_dir + '*.npy'):
            self.valid_keys.append(fle.replace(cache_dir, '').replace('.npy', ''))

    def __contains__(self, key):
        return key in self.valid_keys

    def __getitem__(self, key):
        value = np.load(self.cache_dir + key + '.npy')
        return value

    def __setitem__(self, key, value):
        self.valid_keys.append(key)
        np.save(self.cache_dir + str(key) + '.npy', value)

class ClassScaleMachine(object):
    def __init__(self, GD=None, NCPU=0, MaskArray=None):
        self.GD = GD
        #self.GD["Facets"]["Padding"] = 1.2  # shouldn't need anything bigger than this for the minor cycle
        self.DoAbs = int(self.GD["Deconv"]["AllowNegative"])
        self.MaskArray = MaskArray
        self.PeakFactor = self.GD["WSCMS"]["SubMinorPeakFact"]
        self.NSubMinorIter = self.GD["WSCMS"]["NSubMinorIter"]
        if NCPU == 0:
            import multiprocessing
            NCPU = multiprocessing.cpu_count()
        self.NCPU = NCPU
        self.DoAbs = self.GD["Deconv"]["AllowNegative"]

    def Init(self, PSFServer, FreqMachine, cachepath=None, MaxBaseline=None):
        """
        Sets everything required to perform multi-scale CLEAN. 
        :param PSFserver: Mandatory if NScales > 1
        :param FreqMachine: FreqMachine is required if doing simultaneous multi-frequency and multi-scale deconv 
        :param FTMachine: Initialsed FFTW_Scale_Manager object. Needs to be initialised for the handlers to be
                          registered with APP
        :param cachepath: path to cache. The following items are cached as .npy arrays:
                            - per facet and scale gains
                            - per facet and scale twice convolved PSF keyed as 'S%iF%i' % (iScale, iFacet) 
                              (scale 0's ConvPSF's are not cached since we get them from PSFServer where they already are)
                            - Fourier transform of per facet PSF's and meanPSF's
                            
        :return: None
        """
        # WriteThroughCacheManager have an ordinary dict like interface and will automatically keep the number of items
        # kept in memory limited to the number specified in self.GD['WSCMS']['CacheSize']. All new entries are also
        # automatically stored in the folder specified by cachepath as name<key>.npy files.
        conv_psf_store = Store(cachepath+'/convpsf')
        self.ConvPSFs = pylru.WriteThroughCacheManager(conv_psf_store, self.GD['WSCMS']['CacheSize'])
        conv_psf_mean_store = Store(cachepath+'/convpsfmean')
        self.Conv2PSFmean = pylru.WriteThroughCacheManager(conv_psf_mean_store, self.GD['WSCMS']['CacheSize'])
        ft_psf_store = Store(cachepath+'/ft_psf')
        self.FT_PSF = pylru.WriteThroughCacheManager(ft_psf_store, self.GD['WSCMS']['CacheSize'])
        ft_meanpsf_store = Store(cachepath+'/ft_meanpsf')
        self.FT_meanPSF = pylru.WriteThroughCacheManager(ft_meanpsf_store, self.GD['WSCMS']['CacheSize'])

        # dictionary to store per facet per scale gains
        self.gains = {}

        # set PSF server
        self.PSFServer = PSFServer
        (self.FWHMBeamAvg, _, _) = self.PSFServer.DicoVariablePSF["EstimatesAvgPSF"]
        self.FWHMBeam = self.PSFServer.DicoVariablePSF["FWHMBeam"]

        # set reference key
        _, _, self.Npix, _ = self.PSFServer.ImageShape
        self.CentralFacetID = self.PSFServer.giveFacetID2(self.Npix//2, self.Npix//2)

        # Set freqmachine
        self.FreqMachine = FreqMachine
        self.Nchan = self.FreqMachine.nchan

        # set scale convolve and FFTW related params
        self.NpixPadded = int(np.ceil(self.GD["WSCMS"]["Padding"]*self.Npix))
        # make sure it is odd numbered
        if not self.NpixPadded % 2:
            self.NpixPadded += 1
        self.Npad = (self.NpixPadded - self.Npix)//2
        self.NpixPSF = self.PSFServer.NPSF  # need this to initialise the FTMachine
        self.NpixPaddedPSF = int(np.ceil(self.GD["WSCMS"]["Padding"]*self.NpixPSF))
        if not self.NpixPaddedPSF % 2:
            self.NpixPaddedPSF += 1
        self.NpadPSF = (self.NpixPaddedPSF - self.NpixPSF) // 2

        self.set_coordinates()

        # get scales (in pixel units)
        self.MaxBaseline=MaxBaseline
        self.set_scales()

        # get the FFT and convolution utility
        self.FTMachine = FFTW_Manager(self.GD, self.Nchan, 1, self.Nscales,
                                      self.Npix, self.NpixPadded, self.NpixPSF,
                                      self.NpixPaddedPSF, nthreads=self.NCPU)

        # set bias factors
        self.set_bias()

        # get the Gaussian pars, volume factors and FWHMs corresponding to dictionary functions
        self.set_kernels()

        # initialise Scale dependent masks (everything masked initially)
        self.ScaleMaskArray = {}
        if self.GD["WSCMS"]["AutoMask"]:
            self.AppendMaskComponents = True  # set to false once masking kicks in
            for iScale in range(self.Nscales):
                self.ScaleMaskArray[str(iScale)] = np.ones((1, 1, self.Npix, self.Npix), dtype=bool)
        else:
            self.AppendMaskComponents = False

        # Initialise gains for central facet (logs scale info)
        for i in range(self.Nscales):
            self.set_gains(self.CentralFacetID, i)
            key = 'S' + str(i) + 'F' + str(self.CentralFacetID)
            print(" - Scale %i, bias factor=%f, psfpeak=%f, gain=%f, kernel peak=%f" % \
                               (self.alphas[i], self.bias[i], self.ConvPSFmeanMax,
                                self.gains[key], self.kernels[i].max()), file=log)
        
        # these are permanent
        self.forbidden_scales = []
        # these get reset at the start of every major cycle
        self.retired_scales = []

    def set_coordinates(self):
        # get pixel coordinates for unpadded image
        n = self.Npix//2
        x_unpadded, y_undpadded = np.mgrid[-n:n:1.0j * self.Npix, -n:n:1.0j * self.Npix]
        self.rsq_unpadded = x_unpadded**2 + y_undpadded**2

        # get pixel coordinates for padded image
        n = self.NpixPadded//2
        self.x_image, self.y_image = np.mgrid[-n:n:1.0j * self.NpixPadded, -n:n:1.0j * self.NpixPadded]
        # set corresponding frequencies (note they are fourier shifted so no need to iFs them later)
        freqs = Fs(np.fft.fftfreq(self.NpixPadded))
        self.u_image, self.v_image = np.meshgrid(freqs, freqs)
        self.rhosq_image = self.u_image ** 2 + self.v_image ** 2

        # get pixel coordinates for facet
        n = self.NpixPaddedPSF//2
        self.x_facet, self.y_facet = np.mgrid[-n:n:1.0j * self.NpixPaddedPSF, -n:n:1.0j * self.NpixPaddedPSF]
        # set corresponding frequencies (note they are fourier shifted so no need to iFs them later)
        freqs = Fs(np.fft.fftfreq(self.NpixPaddedPSF))
        self.u_facet, self.v_facet = np.meshgrid(freqs, freqs)
        self.rhosq_facet = self.u_facet ** 2 + self.v_facet ** 2

    def GaussianSymmetric(self, sig, amp=np.array([1.0]), support=None):
        """
        Evaluates symmetric normalised 2D Gaussian centered at (x0, y0). Note this is only used in 
        GiveModelImage so always adds onto a grid the size of the unpadded image.
        The image space coordiantes must be computed in advance and stored in self.x and self.y
        :param sig: std deviation of Gaussian in signal space
        :param x0: x coordinate relative to centre
        :param y0: y coordinate relative to centre
        :param amp: amplitude (at delta scale) of Gaussian component
        :return: 
        """
        I = np.argwhere(sig == self.sigmas).squeeze()
        out = self.kernels[I]
        # broadcast amplitude array to cube
        amp = amp[:, None, None, None]
        return numexpr.evaluate('amp * out')

    def GaussianSymmetricFT(self, sig, x0=0, y0=0, mode='Facet'):
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
                #rhosq = self.rhosq_facet
            elif mode=='Image':
                loc_dict = {'sig': sig, 'pi': np.pi, 'rhosq': self.rhosq_image, 'v': self.v_image, 'x0': x0,
                            'u': self.u_image, 'y0': y0}
                #rhosq = self.rhosq_image
            result = numexpr.evaluate('exp(-2.0j * pi * v * x0 - 2.0j * pi * u * y0 - 2 * pi ** 2 * rhosq * sig ** 2)',
                                      local_dict=loc_dict)
        return result

    # TODO - min scale needs to be set from theoretical min beam size. MaxBaseline taken from VS._uvmax but
    # doesn't seem correct hence the sqrt(2) factor. Is this baseline length or baseline?
    # TODO - Set max scale with minimum baseline or facet size?
    def set_scales(self):
        if self.GD['WSCMS']["MaxScale"] is None:
            MaxScale = self.Npix//4
        else:
            MaxScale = self.GD['WSCMS']["MaxScale"]
        if self.GD["WSCMS"]["Scales"] is None:
            print("Setting scales automatically from theoretical minimum beam size", file=log)
            min_beam = 1.0/self.MaxBaseline  # computed at max frequency
            cell_size_rad = self.GD["Image"]["Cell"] * np.pi / (180 * 3600)
            FWHM0_pix = np.sqrt(2) * min_beam/cell_size_rad  # sqrt(2) is fiddle factor which gives approx same scales as wsclean
            alpha0 = np.ceil(FWHM0_pix / 0.45)
            if alpha0 % 2:
                alpha0 += 1
            alphas = [alpha0, 4*alpha0]
            i = 1
            while alphas[i] < MaxScale:  # hardcoded for now
                alphas.append(1.5*alphas[i])
                i += 1
            self.alphas = np.asarray(alphas[0:-1])
            self.Nscales = self.alphas.size
        else:
            print("Using user defined scales", file=log)
            self.alphas = np.asarray(self.GD["WSCMS"]["Scales"], dtype=float)
            self.Nscales = self.alphas.size

        for i in range(self.Nscales):
            if self.alphas[i] % 2 == 0:
                self.alphas[i] += 1

        self.FWHMs = self.alphas/0.45

    def dilate_scale_masks(self):
        """
        For each scale we compute minor and major axes of Gaussian fit to PSF convolved 
        with the scale kernel and then dilate the scale masks by the average of this value
        :return: 
        """
        from DDFacet.ToolsDir import ModFitPSF
        from scipy.ndimage import binary_dilation
        import matplotlib.pyplot as plt
        for iScale in range(self.Nscales):
            tmpMask = ~self.ScaleMaskArray[str(iScale)].squeeze()
            if tmpMask.any():
                # get PSF and make FWHM mask
                key = 'S' + str(iScale) + 'F' + str(self.CentralFacetID)
                ScalePSF = self.Conv2PSFmean[key].squeeze()
                PSFmax = ScalePSF.max()
                FWHMmask = np.where(ScalePSF.squeeze() > PSFmax/2.0, True, False)
                # get bounding box
                rows = np.any(FWHMmask, axis=1).squeeze()
                cols = np.any(FWHMmask, axis=0).squeeze()
                tmpr = np.where(rows)[0]
                tmpc = np.where(cols)[0]
                rmin, rmax = tmpr[0],  tmpr[-1]
                cmin, cmax = tmpc[0],  tmpc[-1]
                structure = np.asarray(FWHMmask[rmin:rmax+1, cmin:cmax+1])
                self.ScaleMaskArray[str(iScale)] = ~binary_dilation(tmpMask, structure=structure)[None, None]
                # make sure they conform to the external mask
                if self.GD["Mask"]["External"] is not None:
                    self.ScaleMaskArray[str(iScale)] = np.where(self.MaskArray, self.MaskArray,
                                                                self.ScaleMaskArray[str(iScale)])

    def CheckScaleMasks(self, DicoSMStacked):
        DicoComp = DicoSMStacked.setdefault("Comp", {})
        for iScale in DicoComp.keys():
            for key in DicoComp[iScale].keys():
                if key != "NumComps":  # LB - dirty dirty hack needs to die!!!
                    x, y = key
                    self.ScaleMaskArray[str(iScale)][0, 0, x, y] = 0

    def set_bias(self):
        # get scale bias factor
        self.beta = self.GD["WSCMS"]["MultiScaleBias"]

        # set scale bias according to Offringa definition implemented i.t.o. inverse bias
        self.bias = np.ones(self.Nscales, dtype=np.float64)
        for scale in range(1, self.Nscales):
            self.bias[scale] = self.beta**(-1.0 - np.log2(self.alphas[scale]/self.alphas[1]))


    def set_kernels(self):
        """
        Computes Gauss pars corresponding to scales given in number of pixels pixels
        :return: 
        """
        self.sigmas = np.zeros(self.Nscales, dtype=np.float64)
        self.extents = np.zeros(self.Nscales, dtype=np.int32)
        self.volumes = np.zeros(self.Nscales, dtype=np.float64)
        self.kernels = np.empty(self.Nscales, dtype=object)
        for i in range(self.Nscales):
            self.sigmas[i] = 3.0 * self.alphas[i] / 16.0

            # support of Gaussian components in pixels
            half_alpha = self.alphas[i] / 2.0

            # set grid of x, y coordinates
            x, y = np.mgrid[-half_alpha:half_alpha:self.alphas[i] * 1j, -half_alpha:half_alpha:self.alphas[i] * 1j]

            # evaluate scale kernel
            self.kernels[i] = np.exp(-(x ** 2 + y ** 2) / (2 * self.sigmas[i] ** 2))

            self.extents[i] = self.alphas[i]

            self.volumes[i] = np.sum(self.kernels[i])

            self.kernels[i] /= self.volumes[i]

    def give_gain(self, iFacet, iScale):
        """
        Returns gain for facet and scale and add it to the gains dict if it doesn't exist yet. 
        Also initialises the relevant PSF's for facet and scale
        :param iFacet:
        :param iScale:
        :return:
        """
        # get key
        key = 'S'+str(iScale)+'F'+str(iFacet)
        if key not in self.gains:
            self.set_gains(iFacet, iScale)
        return self.gains[key]

    def set_gains(self, iFacet, iScale):
        """
        Initially we need to at least initialise gain for scale 0 at central facet
        :return: 
        """
        key = 'S' + str(iScale) + 'F' + str(iFacet)
        # get PSF for facet
        self.PSFServer.setFacet(iFacet)
        PSF, PSFmean = self.PSFServer.GivePSF()
        npad = (self.NpixPaddedPSF - self.NpixPSF) // 2
        I = slice(npad, self.NpixPaddedPSF - npad)

        # keep track of FT_meanPSF (we don't need to do this for every scale)
        if str(iFacet) not in self.FT_meanPSF:
            self.FTMachine.xhat[...] = iFs(np.pad(PSFmean, ((0, 0), (0, 0), (npad, npad), (npad, npad)), mode='constant'), axes=(2, 3))
            self.FTMachine.FFT()
            self.FT_meanPSF[str(iFacet)] = Fs(self.FTMachine.xhat.copy(), axes=(2, 3))

        # keep track of FT_PSF
        if str(iFacet) not in self.FT_PSF:
            self.FTMachine.Chat[...] = iFs(np.pad(PSF, ((0, 0), (0, 0), (npad, npad), (npad, npad)), mode='constant'), axes=(2, 3))
            self.FTMachine.CFFT()
            self.FT_PSF[str(iFacet)] = Fs(self.FTMachine.Chat.copy(), axes=(2, 3))

        # Get max of mean PSF convolved with scale kernel
        scale_kernel = iFs(self.GaussianSymmetricFT(self.sigmas[iScale])[None, None], axes=(2,3))
        self.FTMachine.xhat[...] = iFs(self.FT_meanPSF[str(iFacet)], axes=(2,3)) * scale_kernel
        # TODO - this FT should not be necessary since we only need the value at the central pixel
        self.FTMachine.iFFT()
        ConvPSFmean = Fs(self.FTMachine.xhat.real.copy(), axes=(2,3))[:, :, I, I]
        self.ConvPSFmeanMax = ConvPSFmean.max()  # just for logging scale info in Init()

        # a few things we need to keep track of if we are at the central facet and scale 0
        if iFacet == self.CentralFacetID and iScale == 0:
            # get the normalisation factor for the ConvPSF's (must set ConvPSF for scale zero to unity)
            self.FTMachine.xhat[...] = iFs(self.FT_meanPSF[str(iFacet)], axes=(2, 3)) * scale_kernel ** 2
            self.FTMachine.iFFT()
            Conv2PSF0 = Fs(self.FTMachine.xhat.real.copy(), axes=(2,3))[:, :, I, I]
            # LB - this normalisation factor is used to ensure that the twice convolved PSF for scale 0
            # (if we were using it) would be normalised to have a maximum of 1 and is applied to normalise
            # the PSF's
            self.ConvPSFNormFactor = ConvPSFmean.max()
            self.Conv2PSFNormFactor = Conv2PSF0.max()

        # get PSF convolved with scale kernel for subtracting components from dirty cube
        self.FTMachine.Chat[...] = iFs(self.FT_PSF[str(iFacet)], axes=(2,3)) * scale_kernel
        self.FTMachine.iCFFT()
        self.ConvPSFs[key] = Fs(self.FTMachine.Chat.real.copy(), axes=(2,3))[:, :, I, I]

        # get twice convolved mean PSF for running sub-minor loop
        self.FTMachine.xhat[...] = iFs(self.FT_meanPSF[str(iFacet)], axes=(2,3)) * scale_kernel**2
        self.FTMachine.iFFT()
        self.Conv2PSFmean[key] = Fs(self.FTMachine.xhat.real.copy(), axes=(2,3))[:, :, I, I]/self.Conv2PSFNormFactor

        # set gains
        gamma = self.GD['Deconv']['Gain']
        if iScale:
            self.gains[key] = gamma / ConvPSFmean.max()
        else:
            self.gains[key] = gamma

    def do_scale_convolve(self, MeanDirty):
        # convolve mean dirty with each scale in parallel
        I = slice(self.Npad, self.NpixPadded - self.Npad)
        self.FTMachine.xhatim[...] = iFs(np.pad(MeanDirty[0:1], ((0, 0), (0,0), (self.Npad, self.Npad),
                                                          (self.Npad, self.Npad)), mode='constant'), axes=(2, 3))
        self.FTMachine.FFTim()
        self.FTMachine.Shat[...] = self.FTMachine.xhatim
        kernels = self.GaussianSymmetricFT(self.sigmas[:, None, None, None], mode='Image')
        self.FTMachine.Shat *= iFs(kernels, axes=(2, 3))
        self.FTMachine.iSFFT()
        ConvMeanDirtys = np.ascontiguousarray(Fs(self.FTMachine.Shat.real, axes=(2, 3))[:, :, I, I])

        # reset the zero scale
        # LB - for scale 0 we might want to do scale selection based
        # on the convolved image instead of MeanDirty
        ConvMeanDirtys[0:1] = MeanDirty.copy()

        # initialise to zero so we always trigger the
        # if statement below at least once
        BiasedMaxVal = 0.0
        # find most relevant scale
        for iScale in range(self.Nscales):
            if iScale not in self.retired_scales:
                # get mask for scale (once auto-masking kicks in we use that instead of external mask)
                if self.AppendMaskComponents or not self.GD["WSCMS"]["AutoMask"]:
                    ScaleMask = self.MaskArray
                else:
                    ScaleMask = self.ScaleMaskArray[str(iScale)]

                xtmp, ytmp, ConvMaxDirty = NpParallel.A_whereMax(ConvMeanDirtys[iScale:iScale+1],
                                                                 NCPU=self.NCPU, DoAbs=self.DoAbs,
                                                                 Mask=ScaleMask)

                if ConvMaxDirty * self.bias[iScale] >= BiasedMaxVal:
                    x = xtmp
                    y = ytmp
                    BiasedMaxVal = ConvMaxDirty * self.bias[iScale]
                    MaxDirty = ConvMaxDirty
                    CurrentDirty = ConvMeanDirtys[iScale:iScale+1]
                    CurrentScale = iScale
                    CurrentMask = ScaleMask
        if BiasedMaxVal == 0:
            print("No scale has been selected. This should never happen. Bug!")
            print("Forbidden scales = ", self.forbidden_scales)

        return x, y, MaxDirty, CurrentDirty, CurrentScale, CurrentMask
