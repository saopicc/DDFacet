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

import pdb
import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)
from scipy.optimize import curve_fit, fmin_l_bfgs_b
from DDFacet.Other import logger
from DDFacet.Imager.ClassScaleMachine import Store
log = logger.getLogger("ClassFreqMachine")

class ClassFrequencyMachine(object):
    """
    Interface to fit frequency axis in model image. All fitting is currently based on some polynomial model in normalised frequencies (v/v_0).
    For the alpha map the fit happens in log space.
        Initialisation:
                ModelCube   = A cube containing the model image with shape [NChannel,NPol,Npix,Npix]
                Freqs       = The gridding frequencies
                Freqsp      = The degridding frequencies
                ref_freq    = The reference frequency
        Methods:
                getFitMask  : Creates a mask to fit models to
                setDesMat   : Creates a design matrix in either log or normal space
                FitAlphaMap : Fits an alpha map to pixels in the model image above a user specified threshold
                EvalAlphaMap: Evaluates the model image from the alpha map
                FitPolyCube : Fits a polynomial to pixels in the model image above a user specified threshhold
                EvalPoly    : Evaluates the polynomial from the computed coefficients
                FitGP       : Fits a Gaussian process to the spectral axis of the model image to pixels above a user specified threshold

    """
    def __init__(self, Freqs, Freqsp, ref_freq, GD=None, weights=None, PSFServer=None):
        self.Freqs = np.asarray(Freqs,dtype=np.float32)
        # Use the longer of the two frequency arrays
        self.Freqsp = np.asarray(Freqsp,dtype=np.float32)
        self.nchan = self.Freqs.size
        self.nchan_degrid = self.Freqsp.size
        self.ref_freq = ref_freq
        freq_diffs = np.abs(self.Freqsp - self.ref_freq)
        self.ref_freq_index = np.argwhere(freq_diffs == freq_diffs.min()).squeeze()
        if self.ref_freq_index.size > 1:
            self.ref_freq_index = self.ref_freq_index[0]
        self.GD = GD
        self.CurrentFacet = 999999
        self.DicoBeamFactors = {}
        self.BeamEnable = self.GD["Beam"]["Model"] is not None
        self.weight = weights if weights is not None else np.ones(self.nchan, dtype=np.float32)

        self.DeconvMode = self.GD["Deconv"]["Mode"]
        # in case we want to use the full channel resolution we need to get the full freqs from somewhere
        if PSFServer is not None:
            self.PSFServer = PSFServer
        else:
            mode = self.GD['Deconv']['Mode']
            if mode == 'Hogbom' or mode == 'WSCMS':
                print("No PSFServer provided, unable to use new freq fit mode", file=log)

    def set_Method(self, mode="Poly"):
        """
        Here we set the method used to fit the frequency axis
        :param mode: the mode to use. options are Poly for normal polynomial, GPR for reduced rank GPR (deprecated)
                     or iPoly for integrated polynomial (similar to what wsclean does)
        :return: 
        """
        if self.nchan==1: #hack to deal with a single channel
            self.Fit = lambda vals, a, b: vals
            self.Eval = lambda vals: vals # this will just be the value in that channel
            self.Eval_Degrid = lambda vals, Freqs: np.repeat(vals[0], Freqs.size)
            # Freqs unused here - nothing to be done but use the same model through
            # the entire passband. LB - Give SPI of -0.7 maybe?
        else:
            if mode == "Poly":
                self.Eval_Degrid = lambda coeffs, Freqs: self.EvalPoly(coeffs, Freqsp=Freqs)
                if self.GD['Output']["Mode"] != 'Predict':  # None of this is needed in Predict mode
                    # set order
                    if self.GD["Deconv"]["Mode"] == 'Hogbom':
                        self.order = self.GD["Hogbom"]["PolyFitOrder"]
                    elif self.GD["Deconv"]["Mode"] == 'WSCMS':
                        self.order = self.GD["WSCMS"]["NumFreqBasisFuncs"]
                    print("Using %i order polynomial for frequency fit" % self.order, file=log)
                    # construct design matrix at gridding channel resolution
                    self.Xdes = self.setDesMat(self.Freqs, order=self.order, mode="Mono")
                    self.Xdesp = self.setDesMat(self.Freqsp, order=self.order, mode="Mono")
                    self.Xdes_ref = self.setDesMat(self.ref_freq, order=self.order, mode="Mono")


                    # get frequencies at full channel resolution
                    self.freqs_full = []
                    for iCh in range(self.nchan):
                        self.freqs_full.append(self.PSFServer.DicoVariablePSF["freqs"][iCh]
                        if hasattr(self, "PSFServer") and self.PSFServer is not None else [self.Freqs[iCh]])
                    self.freqs_full = np.concatenate(self.freqs_full)
                    self.nchan_full = np.size(self.freqs_full)

                    # set design matrix at full channel resolution
                    self.Xdes_full = self.setDesMat(self.freqs_full, order=self.order,
                                                    mode="Mono")

                    # # build the S matrix
                    # ChanMappingGrid = self.PSFServer.DicoMappingDesc["ChanMappingGrid"] 
                    # self.S = np.zeros([self.nchan, self.nchan_full], dtype=np.float32)
                    # for iChannel in range(self.nchan):
                    #     active_chans = [ChanMappingGrid[iMS]==iChannel for iMS in ChanMappingGrid.keys()]
                    #     nchunk = sum([a.sum() for a in active_chans])   # count all active channels
                    #     w = 1.0/nchunk if nchunk else 0
                    #     for a in active_chans:
                    #         self.S[iChannel, a] = 1.0/nchunk

                    # dictionaries to hold pseudo inverses and design matrices
                    self.pinv_dict = {}
                    self.sax_dict = {}

                    self.Fit = self.FitPoly
                    self.Eval = self.EvalPolyApparent

            else:
                raise NotImplementedError("Frequency fit mode %s not supported" % mode)

    def FitSPIComponents(self, FitCube, nu, nu0):
        """
        Slow version using serial scipy.optimise.curve_fit to fit the spectral indices.
        Used as a fallback if africanus version not found
        :param FitCube: (ncomps, nfreqs) data array  
        :param nu: freqs
        :param nu0: ref freq
        :return: 
        """
        def spi_func(nu, I0, alpha):
            return I0 * nu ** alpha
        nchan, ncomps = FitCube.shape
        Iref = np.zeros([ncomps])
        varIref = np.zeros([ncomps])
        alpha = np.zeros([ncomps])
        varalpha = np.zeros([ncomps])
        I0 = 1.0
        alpha0 = -0.7
        for i in range(ncomps):
            popt, pcov = curve_fit(spi_func, nu/nu0, FitCube[:, i], p0=np.array([I0, alpha0]))
            Iref[i] = popt[0]
            varIref[i] = pcov[0,0]
            alpha[i] = popt[1]
            varalpha[i] = pcov[1,1]
        return alpha, varalpha, Iref, varIref

    def setDesMat(self, Freqs, order=None, mode="Normal"):
        """
        This function creates the design matrix. Use any linear model your heart desires
        Args:
            order   = The number of basis functions to use
            mode    = sets the kind of basis functions to use. Options are
                    - "Normal" for monomial model
                    - "log" for log monomial model
                    - "Cyril" for Cyril's sum of power law model
                    - "Andre" for Andre's integrated polynomial model
        Returns:
            Xdesign    = The design matrix [1, (v/v_0), (v/v_0)**2, ...]

        """
        if order is None:
            order = self.order
        if mode == "Mono" or mode == "Poly":
            # Construct vector of frequencies
            w = (Freqs / self.ref_freq).reshape(Freqs.size, 1)
            # create tiled array and raise each column to the correct power
            Xdesign = np.tile(w, order) ** np.arange(0, order)
        elif mode == "log":
            # Construct vector of frequencies
            w = np.log(Freqs / self.ref_freq).reshape(Freqs.size, 1)
            # create tiled array and raise each column to the correct power
            Xdesign = np.tile(w, order) ** np.arange(0, order)
        else:
            raise NotImplementedError("Frequency basis %s not supported" % mode)
        return Xdesign

    # this is taken directly from ClassSpectralFunctions (its all the functionality we need in here)
    def GiveBeamFactorsFacet(self, iFacet):
        if iFacet in self.DicoBeamFactors:
            return self.DicoBeamFactors[iFacet]

        SumJonesChan = self.PSFServer.DicoMappingDesc["SumJonesChan"]
        ChanMappingGrid = self.PSFServer.DicoMappingDesc["ChanMappingGrid"]

        ChanMappingGridChan = self.PSFServer.DicoMappingDesc["ChanMappingGridChan"]
        ListBeamFactor = []
        ListBeamFactorWeightSq = []
        for iChannel in range(self.nchan):
            nfreq = len(self.PSFServer.DicoMappingDesc["freqs"][iChannel])
            ThisSumJonesChan = np.zeros(nfreq, np.float64)
            ThisSumJonesChanWeightSq = np.zeros(nfreq, np.float64)
            for iMS in SumJonesChan.keys():
                ind = np.where(ChanMappingGrid[iMS] == iChannel)[0]
                channels = ChanMappingGridChan[iMS][ind]
                ThisSumJonesChan[channels] += SumJonesChan[iMS][iFacet, 0, ind]
                ThisSumJonesChanWeightSq[channels] += SumJonesChan[iMS][iFacet, 1, ind]
            ListBeamFactor.append(ThisSumJonesChan)
            ListBeamFactorWeightSq.append(ThisSumJonesChanWeightSq)

        self.DicoBeamFactors[iFacet] = ListBeamFactor, ListBeamFactorWeightSq, self.PSFServer.DicoMappingDesc['MeanJonesBand'][iFacet]

        return ListBeamFactor, ListBeamFactorWeightSq, self.PSFServer.DicoMappingDesc['MeanJonesBand'][iFacet]

    def compute_pseudo_inverse(self, A, W):
        """
        Computes the pseudo-inverse of design matrix A for the facet 
        weighted by the WeightsChansImages
        """
        sqrtW = np.sqrt(W)
        WX = sqrtW[:, None] * A
        if self.nchan >= self.order:
            # get left pinv
            XTX = WX.T.dot(WX)
            XTXinv = np.linalg.inv(XTX)
            pinv = XTXinv.dot(WX.T)
        else:
            # get right pinv
            XXT = WX.dot(WX.T)
            XXTinv = np.linalg.inv(XXT)
            pinv = WX.T.dot(XXTinv)
        return pinv

    def give_pseudo_inverse(self, JonesNorm, WeightsChansImages):
        """
        Checks if we need to recompute the pseudo-inverse and recomputes it if necessary
        :return: 
        """
        if self.BeamEnable:
            key = self.PSFServer.iFacet
            if key not in self.sax_dict:
                # SumJonesChanList, SumJonesChanWeightSqList, MeanJonesBand = self.GiveBeamFactorsFacet(key)
                # SumJonesChan = np.concatenate(SumJonesChanList)
                # SumJonesChanWeightSq = np.concatenate(SumJonesChanWeightSqList)
                # BeamFactor = np.sqrt(SumJonesChan / SumJonesChanWeightSq)  # unstitched sqrt(JonesNorm) at full resolution
                # # incorporate stitched JonesNorm
                # JonesFactor = np.sqrt(MeanJonesBand / JonesNorm)
                # # The division by JonesFactor corrects for the fact that the PSF is normalised
                # SAmat = self.S * BeamFactor[None, :] / JonesFactor[:, None]
                # SAX = SAmat.dot(self.Xdes_full)
                SAX = np.sqrt(JonesNorm)[:, None] * self.Xdes
                self.sax_dict[key] = SAX
            if key not in self.pinv_dict:
                SAX = self.sax_dict[key]
                pinv = self.compute_pseudo_inverse(SAX, WeightsChansImages)
                self.pinv_dict[key] = pinv
        else:
            key = 0
            if key not in self.sax_dict:
                # SAX = self.S.dot(self.Xdes_full)
                SAX = self.Xdes
                self.sax_dict[key] = SAX
            if key not in self.pinv_dict:
                SAX = self.sax_dict[key]
                pinv = self.compute_pseudo_inverse(SAX, WeightsChansImages)
                self.pinv_dict[key] = pinv
        return self.sax_dict[key], self.pinv_dict[key]

    def FitPoly(self, Iapp, JonesNorm, WeightsChansImages):
        """
        This is for the new frequency fit mode incorporating FreqBandsFluxRation
        :param Iapp: Apparent values in the imaging bands
        :param JonesNorm: the value of the JonesNorm at the location of vals
        :param JNWeight: the distance of the location of vals from facet center used to weight the fit between the 
                        unstitched JonesNorm (BeamFactor) at full resolution and the stitched one (at gridding res) 
        :return: coeffs
        Here is what I have figured out regarding the dictionary keywords so far:
        MeanJonesBand - holds the unstitched JonesNorm in each Facet for each imaging band accessed as [iFacet][iChannel]
        SumJonesChan - This is the sum of the unstitched Jones terms at full channel resolution for each facet
        SumJonesChanWeightSq - this is a normalisation factor of sorts for SumJonesChan. I think is there are no 
                               flagged data these terms are all the same. Bottom line is that the unstitched Jonesnorm 
                               in a specific frequency chunk can be computed as 
                               np.sum(SumJonesChan[0][chunk])/np.sum(SumJonesChanWeightSq[0][chunk])
        ChanMappingGrid - specifies the band that each channel falls into for each MS [iMS][iChannel]
        ChanMappingGridChan - 
        """
        # get design matrix and pseudo-inverse
        SAX, pinv = self.give_pseudo_inverse(JonesNorm, WeightsChansImages)

        # fit to whitened data
        sqrtW = np.sqrt(WeightsChansImages)
        Wy = sqrtW * Iapp
        theta = pinv.dot(Wy)
        return theta

    def FitLin(self, Iapp, JonesNorm, WeightsChansImages):
        """
        simple linear fitting between each adjacent channel pairs (for the gridding frequency, self.Freqs)
        The output is a numpy array of linear fitting coeffciencies with a shape of (# of frequency pairs,2)
        Hightest order first, following numpy polyfit
        """
        if self.Freqs.size == 1:
            n_pairs = 1
            coeffs = np.zeros((n_pairs, 2))
            coeffs[0, 1] = Iapp[0]
            coeffs[0, 0] = 0
        else:
            n_pairs = len(self.Freqs) - 1
            coeffs = np.zeros((n_pairs, 2))
            for i in range(n_pairs):
                coeffs[i,:] = np.polyfit(self.Freqs[i:i+2], Iapp[i:i+2], deg=1, w = WeightsChansImages[i:i+2])
        return coeffs

    def EvalLin(self, coeffs, Freqsp):
        """
        calculate the flux at the given  frequencies Freqsp using the linear fitting coefficiencies from FitLin
        """
        if self.Freqs.size == 1:
            if coeffs.size != 2:
                raise RuntimeError("Fit coefficients must constant pair for single band data")
            return coeffs[0, 1].reshape(1)
        else:
            if coeffs.size != 2 * (self.Freqs.size - 1):
                raise RuntimeError("Fit coefficients must be pairs of the size or number of gridding bands - 1")
            out_freq_array = np.array([Freqsp, np.ones(len(Freqsp))])
            out_freq_array = out_freq_array.T

            # first check if the given frequencies is the same as the gridding frequency self.Freqs
            if np.array_equal(Freqsp, self.Freqs):
                out_coeffs = np.concatenate((coeffs,np.array([coeffs[-1,:]])))
                out_coeffs = out_coeffs.T
            else:
                #construct the frequency array of middle freqency between adjacent gridding frequency pairs 
                if len(self.Freqs) >= 2:
                    mid_grid_freq_array = np.array([np.mean(self.Freqs[i:i+2]) for i in range(len(self.Freqs)-1)])
                else:
                    mid_grid_freq_array = self.Freqs
                
                sel_coeffs = list(map(lambda gridfreq: np.argmin(np.abs(mid_grid_freq_array - gridfreq)), 
                                    Freqsp))
                out_coeffs = coeffs[sel_coeffs].reshape(coeffs[sel_coeffs].size//2, 2).T
            
            # freq array dotted with closest linear fit coefficients (Nfreq x 2 dot 2 x Nfreq)
            out = np.dot(out_freq_array, out_coeffs)
            
            return np.diagonal(out)

    def EvalPoly(self, coeffs, Freqsp=None):
        """
        Evaluates a polynomial at Freqs with coefficients coeffs
        Args:
            coeffs: the coefficients of the polynomial in order corresponding to (1,v,v**2,...)
            Freqs: the frequencies at which to evaluate the polynomial
        Returns:
            The polynomial evaluated at Freqs
        """
        if np.array_equal(Freqsp, self.Freqs):
            # Here we don't need to reset the design matrix
            return np.dot(self.Xdes, coeffs)
        elif np.array_equal(Freqsp, self.Freqsp):
            return np.dot(self.Xdesp, coeffs)
        elif Freqsp.size == 1 and Freqsp == self.ref_freq:
            return np.dot(self.Xdes_ref, coeffs)
        else:
            # Here we do
            Xdes = self.setDesMat(Freqsp, order=self.order, mode="Mono")
            # evaluate poly and return result
            return np.dot(Xdes, coeffs)

    def EvalPolyApparent(self, coeffs):
        """
        Gives the apparent flux for coeffs given beam in this facet
        Args:
            coeffs: the coefficients of the polynomial in order corresponding to (1,v,v**2,...)
            Freqs: the frequencies at which to evaluate the polynomial
        Returns:
            The polynomial evaluated at Freqs
        """
        if self.BeamEnable:
            key = self.PSFServer.iFacet
        else:
            key = 0
        SAX = self.sax_dict[key]
        return SAX.dot(coeffs)
