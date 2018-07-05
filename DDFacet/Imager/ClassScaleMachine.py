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
        #print "Found valid keys ", self.valid_keys

    def __contains__(self, key):
        return key in self.valid_keys

    def __getitem__(self, key):
        value = np.load(self.cache_dir + key + '.npy')
        return value

    def __setitem__(self, key, value):
        np.save(self.cache_dir + str(key) + '.npy', value)

class ClassScaleMachine(object):
    def __init__(self, GD=None, NCPU=6, MaskArray=None):
        self.GD = GD
        self.DoAbs = int(self.GD["Deconv"]["AllowNegative"])
        self.MaskArray = MaskArray
        self.PeakFactor = self.GD["WSCMS"]["SubMinorPeakFact"]
        self.NSubMinorIter = self.GD["WSCMS"]["NSubMinorIter"]
        self.NCPU = NCPU
        self.DoAbs = self.GD["Deconv"]["AllowNegative"]

    def Init(self, PSFServer, FreqMachine, FTMachine2=None, cachepath=None):
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
        # If we have a cachepath then load all elements already precomputed in the cache into WriteThroughCacheManager
        # objects. These objects have an ordinary dict like interface and will automatically keep the number of items
        # kept in memory limited to the number specified in self.GD['WSCMS']['CacheSize']. All new entries are also
        # automatically stored in the folder specified by cachepath as name<key>.npy files.
        if cachepath is not None:
            conv_psf_store = Store(cachepath+'/convpsf')
            self.Conv2PSFs = pylru.WriteThroughCacheManager(conv_psf_store, self.GD['WSCMS']['CacheSize'])
            gains_store = Store(cachepath+'/gains')
            self.gains = pylru.WriteThroughCacheManager(gains_store, self.GD['WSCMS']['CacheSize'])
            ft_psf_store = Store(cachepath+'/ft_psf')
            self.FT_PSF = pylru.WriteThroughCacheManager(ft_psf_store, self.GD['WSCMS']['CacheSize'])
            ft_meanpsf_store = Store(cachepath+'/ft_meanpsf')
            self.FT_meanPSF = pylru.WriteThroughCacheManager(ft_meanpsf_store, self.GD['WSCMS']['CacheSize'])
            conv_psf_freq_peak_store = Store(cachepath+'/convpsfpeaks')
            self.ConvPSFFreqPeaks = pylru.WriteThroughCacheManager(conv_psf_freq_peak_store, self.GD['WSCMS']['CacheSize'])
        else:
            self.Conv2PSFs = {}
            self.gains = {}
            self.FT_PSF = {}
            self.FT_meanPSF = {}
            self.ConvPSFFreqPeaks = {}

        # set PSF server
        self.PSFServer = PSFServer
        (self.FWHMBeamAvg, _, _)=self.PSFServer.DicoVariablePSF["EstimatesAvgPSF"]
        self.FWHMBeam = self.PSFServer.DicoVariablePSF["FWHMBeam"]

        # set reference key
        self.CentralFacetID = self.GD['Facets']['NFacets']**2//2  # assumes odd number of facets

        # Set freqmachine
        self.FreqMachine = FreqMachine
        self.Nchan = self.FreqMachine.nchan

        # set scale convolve and FFTW related params
        _, _, self.Npix, _ = self.PSFServer.ImageShape
        self.NpixPadded = int(np.ceil(self.GD["Facets"]["Padding"]*self.Npix))
        # make sure it is odd numbered
        if self.NpixPadded % 2 == 0:
            self.NpixPadded += 1
        self.Npad = (self.NpixPadded - self.Npix)//2
        self.NpixPSF = self.PSFServer.NPSF  # need this to initialise the FTMachine
        self.NpixPaddedPSF = self.PSFServer.DicoVariablePSF["PaddedPSFInfo"][0]  # hack for now
        self.NpadPSF = (self.NpixPaddedPSF - self.NpixPSF) // 2

        #print self.Npix, self.NpixPadded, self.Npad
        #print self.NpixPSF, self.NpixPaddedPSF, self.NpadPSF

        self.set_coordinates(self.NpixPadded, self.NpixPaddedPSF)

        # get scales (in pixel units)
        self.set_scales()

        # Testing!!!!
        #self.PSFServer.setFacet(0)
        #PSF, meanPSF = self.PSFServer.GivePSF()

        # get the FFT and convolution utility
        self.FTMachine = LB_FFT_and_Gauss_Tools(self.NpixPaddedPSF//2, self.Nchan, 1, self.Nscales,
                                                wisdom_file=cachepath+'/wisdom.npy', npix=self.Npix,
                                                npixpadded=self.NpixPadded, npixpsf=self.NpixPSF,
                                                npixpaddedpsf=self.NpixPaddedPSF)
        #self.FTMachine2 = FTMachine2
        #self.FTMachine2.Init(self.Npix, self.NpixPadded, self.NpixPSF, self.NpixPaddedPSF, self.Nchan, 1, self.Nscales)

        # import time
        # for i in xrange(5):
        #     ti = time.time()
        #     result1 = self.FTMachine.ConvolveGaussian(PSF.copy(), 5.0, cube=True)
        #     tf = time.time()
        #     print "Old version took ", tf - ti
        #     ti = time.time()
        #     result2 = self.ConvolveGaussian(PSF.copy(), 5.0, mode='Facet')
        #     tf = time.time()
        #     print "New version took ", tf - ti
        #     ti = time.time()
        #     result3 = self.ConvolveGaussian_new(PSF.copy(), 5.0, mode='Facet')
        #     tf = time.time()
        #     print "Newest version took ", tf - ti
        #     print np.amax(np.abs(result3.reshape(self.Nchan, 1, self.NpixPSF**2) - result2.reshape(self.Nchan, 1, self.NpixPSF**2)),
        #               axis=2)
        #
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

        # set bias factors
        self.set_bias()

        # get the Gaussian pars, volume factors and FWHMs corresponding to dictionary functions
        self.set_kernels()

        # result1 = self.FTMachine.GaussianSymmetricFT(self.sigmas[0], x0=12, y0=25)
        # result2 = self.GaussianSymmetricFT(self.sigmas[0], x0=12, y0=25)
        #
        # print np.abs(result1 - result2).max()
        #
        # result1 = self.FTMachine.GaussianSymmetricFT(self.sigmas[0])
        # result2 = self.GaussianSymmetricFT(self.sigmas[0])
        #
        # print np.abs(result1 - result2).max()
        #
        # import sys
        # sys.exit(0)

        # for scale dependent masking (do we want to cache this?)
        self.ScaleMaskArray = {}

        # we always need to set the gain for the central facet and scale 0 to initialise
        self.set_gains(self.CentralFacetID, 0)

    def set_coordinates(self, NpixPaddedImage, NpixPaddedFacet):
        # TODO - cache and evaluate with numexpr
        # get pixel coordinates for image
        n = NpixPaddedImage//2
        self.x_image, self.y_image = np.mgrid[-n:n:1.0j * self.NpixPadded, -n:n:1.0j * self.NpixPadded]
        # set corresponding frequencies (note they are fourier shifted so no need to iFs them later)
        freqs = Fs(np.fft.fftfreq(self.NpixPadded))
        self.u_image, self.v_image = np.meshgrid(freqs, freqs)
        self.rhosq_image = self.u_image ** 2 + self.v_image ** 2

        # get pixel coordinates for facet
        n = NpixPaddedFacet//2
        self.x_facet, self.y_facet = np.mgrid[-n:n:1.0j * self.NpixPaddedPSF, -n:n:1.0j * self.NpixPaddedPSF]
        # set corresponding frequencies (note they are fourier shifted so no need to iFs them later)
        freqs = Fs(np.fft.fftfreq(self.NpixPaddedPSF))
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
            elif mode=='Image':
                loc_dict = {'sig': sig, 'pi': np.pi, 'rhosq': self.rhosq_image, 'v': self.v_image, 'x0': x0,
                            'u': self.u_image, 'y0': y0}
            result = numexpr.evaluate('exp(-2.0j * pi * v * x0 - 2.0j * pi * u * y0 - 2 * pi ** 2 * rhosq * sig ** 2)',
                                   local_dict=loc_dict)
        return result


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
        try:
            self.bias[1::] = self.beta**(-1.0 - np.log2(self.FWHMs[1::]/self.FWHMs[1]))
        except:  # in case there is only a single scale it will be the delta scale
            self.bias = np.array([1.0])

    def set_bias_new(self):  # work in progress
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

            #print>>log, "scale %i kernel peak = %f" % (i, 1.0/self.volumes[i])
            #print>> log, "scale %i sigma = %f" % (i, self.sigmas[i])
        #self.VolumeNorms = self.volumes/self.volumes[0]
        return

    def give_gain(self, iFacet, iScale):
        """
        Returns gain for facet and scale and add it to the gains dict if it doesn't exist yet. The gain is
        automatically cached if a cachepath is provided in Init()
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

        # a few things we need to keep track of if we are at the central facet and scale 0
        if iFacet == self.CentralFacetID and iScale == 0:
            self.Scale0PSFmax = ConvPSFmean[0, 0, self.NpixPSF // 2, self.NpixPSF // 2]
            # get the normalisation factor for the ConvPSF's (must set ConvPSF for scale zero to unity)
            self.FTMachine.xhat[...] = iFs(self.FT_meanPSF[str(iFacet)], axes=(2, 3)) * scale_kernel ** 2
            self.FTMachine.iFFT()
            Conv2PSF0 = Fs(self.FTMachine.xhat.real.copy(), axes=(2,3))[:, :, I, I]
            self.ConvPSFNormFactor = Conv2PSF0[0, 0, self.NpixPSF // 2, self.NpixPSF // 2]

            #print "Scale0PSFMax = ", self.Scale0PSFmax
            #print "ConvPSF norm factor is %f"%self.ConvPSFNormFactor

        # get max of PSF convolved with scale kernel
        self.FTMachine.Chat[...] = iFs(self.FT_PSF[str(iFacet)], axes=(2,3)) * scale_kernel
        self.FTMachine.iCFFT()
        ConvPSF = Fs(self.FTMachine.Chat.real.copy(), axes=(2,3))[:, :, I, I]
        # print "mean Conv PSF peak = ", ConvPSFmean[0, 0, self.NpixPSF // 2, self.NpixPSF // 2]
        # print "Conv PSF peaks = ", ConvPSF[:, 0, self.NpixPSF // 2, self.NpixPSF // 2]
        self.ConvPSFFreqPeaks[key] = (ConvPSFmean[0, 0, self.NpixPSF // 2, self.NpixPSF // 2] /
                                      ConvPSF[:, 0, self.NpixPSF // 2, self.NpixPSF // 2]).reshape(self.Nchan, 1, 1, 1)

        # get twice convolved PSF
        self.FTMachine.Chat[...] = iFs(self.FT_PSF[str(iFacet)], axes=(2,3)) * scale_kernel**2
        self.FTMachine.iCFFT()
        self.Conv2PSFs[key] = Fs(self.FTMachine.Chat.real.copy(), axes=(2,3))[:, :, I, I]/self.ConvPSFNormFactor

        # set gains
        gamma = self.GD['Deconv']['Gain']
        self.gains[key] = gamma * self.Scale0PSFmax / ConvPSFmean[0, 0, self.NpixPSF // 2, self.NpixPSF // 2]
        #print "gain for scale %i = " % iScale, self.gains[key]

    def do_scale_convolve(self, Dirty, MeanDirty):
        I = slice(self.Npad, self.NpixPadded - self.Npad)
        self.FTMachine.xhatim[...] = iFs(np.pad(MeanDirty[0:1], ((0, 0), (0,0), (self.Npad, self.Npad),
                                                          (self.Npad, self.Npad)), mode='constant'), axes=(2, 3))
        self.FTMachine.FFTim()
        self.FTMachine.Shat[...] = self.FTMachine.xhatim
        kernels = self.GaussianSymmetricFT(self.sigmas[:, None, None, None], mode='Image')
        self.FTMachine.Shat *= iFs(kernels, axes=(2, 3))
        self.FTMachine.iSFFT()
        ConvMeanDirtys = np.ascontiguousarray(Fs(self.FTMachine.Shat.real, axes=(2, 3))[:, :, I, I])

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
            self.FTMachine.Chatim[...] = iFs(np.pad(Dirty, ((0, 0), (0, 0), (self.Npad, self.Npad),
                                                          (self.Npad, self.Npad)), mode='constant'), axes=(2,3))
            self.FTMachine.CFFTim()
            self.FTMachine.Chatim *= iFs(self.GaussianSymmetricFT(self.sigmas[CurrentScale], mode='Image')[None, None],
                                       axes=(2, 3))
            self.FTMachine.iCFFTim()
            Dirty = Fs(self.FTMachine.Chatim.real.copy(), axes=(2,3))[:, :, I, I]
            x, y, MaxDirty = NpParallel.A_whereMax(MeanDirty, NCPU=self.NCPU, DoAbs=self.DoAbs, Mask=self.MaskArray)

        return x, y, MaxDirty, MeanDirty, Dirty, CurrentScale

    def SMConvolvePSF(self, iFacet, FT_SM):
        """
        Here we pass in the FT of a sky model and convolve it with the psf 
        :param iFacet: 
        :param FT_SM: 
        :return: 
        """
        if str(iFacet) not in self.FT_PSF:
            raise Exception("This should never happen. Bug!!!!")

        I = slice(self.NpadPSF, self.NpixPaddedPSF - self.NpadPSF)
        self.FTMachine.Chat[...] = iFs(self.FT_PSF[str(iFacet)], axes=(2, 3)) * iFs(FT_SM, axes=(2, 3))
        self.FTMachine.iCFFT()
        return Fs(self.FTMachine.Chat.real.copy(), axes=(2, 3))[:, :, I, I]

