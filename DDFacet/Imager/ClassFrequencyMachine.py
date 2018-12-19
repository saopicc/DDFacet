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
from scipy.optimize import curve_fit, fmin_l_bfgs_b
from DDFacet.Other import MyLogger
log = MyLogger.getLogger("ClassScaleMachine")

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
        self.Freqs = np.asarray(Freqs)
        # Use the longer of the two frequency arrays
        self.Freqsp = np.asarray(Freqsp)
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
            print>>log, "No PSFServer provided, unable to use new freq fit mode"

    def set_Method(self, mode="Poly"):
        """
        Here we set the method used to fit the frequency axis
        :param mode: the mode to use. options are Poly for normal polynomial, GPR for reduced rank GPR (deprecated)
                     or iPoly for integrated polynomial (similar to what wsclean does)
        :return: 
        """
        if self.nchan==1: #hack to deal with a single channel
            self.Fit = lambda vals: vals
            self.Eval = lambda vals: vals # this will just be the value in that channel
            self.Eval_Degrid = lambda vals, Freqs: np.tile(vals, Freqs.size) # Freqs unused - nothing to be done but use the same model through the entire passband
        else:
            if mode == "WSCMS":
                self.Eval_Degrid = lambda coeffs, Freqs: self.EvalPoly(coeffs, Freqsp=Freqs)
                if self.GD['Output']["Mode"] != 'Predict':  # None of this is needed in Predict mode
                    # set order
                    self.order = self.GD["WSCMS"]["NumFreqBasisFuncs"]
                    # get polynomial coeffs for prior when I0 is positive
                    self.alpha_prior = self.GD["WSCMS"]["AlphaPrior"]
                    if self.alpha_prior is not None:
                        I = (self.Freqsp/self.ref_freq)**(self.alpha_prior)
                        coeffs, Kfull = np.polyfit(self.Freqsp/self.ref_freq, I, deg=self.order-1, cov=True)
                        self.prior_theta = coeffs[::-1]
                        self.prior_invcov = 1.0/np.diag(Kfull)[::-1]
                    else:
                        self.prior_theta = np.zeros(self.order, dtype=np.float64)
                        self.prior_invcov = np.zeros(self.order, dtype=np.float64)
                    # get polynomial coeffs for prior when I0 is negative
                    self.alpha_prior_neg = self.GD["WSCMS"]["AlphaPriorNeg"]
                    if self.alpha_prior_neg is not None:
                        I = (self.Freqsp/self.ref_freq)**(self.alpha_prior_neg)
                        coeffs, Kfull = np.polyfit(self.Freqsp/self.ref_freq, I, deg=self.order-1, cov=True)
                        self.prior_theta_neg = coeffs[::-1]
                        self.prior_invcov_neg = 1.0/np.diag(Kfull)[::-1]
                    else:
                        self.prior_theta_neg = np.zeros(self.order, dtype=np.float64)
                        self.prior_invcov_neg = np.zeros(self.order, dtype=np.float64)

                    if (self.alpha_prior is not None) or (self.alpha_prior_neg is not None):
                        self.bnds = ((None, None),)
                        for param in xrange(self.order-1):
                            self.bnds += ((None, None),)

                    # construct design matrix at gridding channel resolution
                    self.Xdes = self.setDesMat(self.Freqs, order=self.order, mode=self.GD['WSCMS']['FreqMode'])

                    # there is no need to recompute this every time if the beam is not enabled because same everywhere
                    if not self.BeamEnable:
                        self.SAX = self.setDesMat(self.Freqs, order=self.order, mode='Andre')  # this fits the integrated polynomial
                    else:
                        self.freqs_full = []
                        for iCh in xrange(self.nchan):
                            self.freqs_full.append(self.PSFServer.DicoVariablePSF["freqs"][iCh])
                        self.freqs_full = np.concatenate(self.freqs_full)
                        self.nchan_full = np.size(self.freqs_full)

                        self.Xdes_full = self.setDesMat(self.freqs_full, order=self.order,
                                                        mode=self.GD['WSCMS']['FreqMode'])
                        # build the S matrix
                        ChanMappingGrid = self.PSFServer.DicoMappingDesc["ChanMappingGrid"]
                        ChanMappingFull = []
                        for iMS in ChanMappingGrid.keys():
                            ChanMappingFull.append(ChanMappingGrid[iMS])
                        ChanMappingFull = np.concatenate(ChanMappingFull)
                        self.S = np.zeros([self.nchan, self.nchan_full], dtype=np.float32)
                        for iChannel in range(self.nchan):
                            ind = np.argwhere(ChanMappingFull == iChannel).squeeze()
                            nchunk = np.size(ind)
                            if nchunk:
                                self.S[iChannel, ind] = 1.0/nchunk
                            else:
                                self.S[iChannel, ind] = 0.0

                    self.Fit = self.FitPolyNew
                    self.Eval = self.EvalPolyApparent
            elif mode == "Poly":
                # set order
                self.order = self.GD["Hogbom"]["PolyFitOrder"]
                # construct design matrix at full channel resolution
                self.Xdes = self.setDesMat(self.Freqs, order=self.order, mode="Mono")
                if self.nchan >= self.order: # use left pseudo inverse
                    self.AATinvAT = np.linalg.inv(self.Xdes.T.dot(self.Xdes)).dot(self.Xdes.T)
                else: # use right pseudo inverse
                    self.AATinvAT = self.Xdes.T.dot(np.linalg.inv(self.Xdes.dot(self.Xdes.T)))
                self.Fit = lambda vals: self.FitPoly(vals)
                self.Eval = lambda coeffs: self.EvalPoly(coeffs, Freqsp=self.Freqs)
                self.Eval_Degrid = lambda coeffs, Freqs: self.EvalPoly(coeffs, Freqsp=Freqs)
            elif mode is None:  # TODO - test that this does the expected thing when Nchan != Nchan_degrid
                self.Fit = lambda vals: vals
                self.Eval = lambda vals: vals
                self.Eval_Degrid = lambda vals, Freqs: self.DistributeFreqs(vals, Freqs)
            else:
                raise NotImplementedError("Frequency fit mode %s not supported" % mode)

    def FitSPIComponents(self, FitCube, nu, nu0):
        def spi_func(nu, I0, alpha):
            return I0 * nu ** alpha
        nchan, ncomps = FitCube.shape
        Iref = np.zeros([ncomps])
        varIref = np.zeros([ncomps])
        alpha = np.zeros([ncomps])
        varalpha = np.zeros([ncomps])
        I0 = 1.0
        alpha0 = -0.7
        for i in xrange(ncomps):
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
        elif mode == "Laplace":
            def Laplace_Eigenfunc(self, i, nu, L=1.5):
                return np.sin(np.pi * i * (nu + L) / (2.0 * L)) / np.sqrt(L)
            w = (Freqs / self.ref_freq)
            Nch = np.size(Freqs)
            Xdesign = np.zeros([Nch, self.order])
            for i in xrange(order):
                Xdesign[:, i] = Laplace_Eigenfunc(i + 1, w)
        elif mode == "PowLaws":
            w = (Freqs / self.ref_freq).reshape(Freqs.size, 1)
            alphas = np.linspace(-1, 1, self.order)
            return np.tile(w, order) ** alphas
        elif mode == "log":
            # Construct vector of frequencies
            w = np.log(Freqs / self.ref_freq).reshape(Freqs.size, 1)
            # create tiled array and raise each column to the correct power
            Xdesign = np.tile(w, order) ** np.arange(0, order)
        elif mode == "Andre":
            # we are given frequencies at bin centers convert to bin edges
            delta_freq = Freqs[1] - Freqs[0]
            wlow = (Freqs - delta_freq/2.0)/self.ref_freq
            whigh = (Freqs + delta_freq/2.0)/self.ref_freq
            wdiff = whigh - wlow
            Xdesign = np.zeros([Freqs.size, self.order])
            for i in xrange(1, self.order+1):
                Xdesign[:, i-1] = (whigh**i - wlow**i)/(i*wdiff)
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

    def logp_and_dlogp(self, x, Iapp, SAX, W, K, theta0):
        theta = x[0:-1]
        I0 = x[-1]
        thetap = I0 * theta0
        Kp = I0 ** 2 * K
        resI = Iapp - SAX.dot(theta)
        logp = 0.5 * resI.dot(W * resI)
        restheta = theta - thetap
        logp += 0.5 * restheta.dot(restheta / Kp)
        dlogp = np.zeros(x.size)
        dlogp[0:-1] = - SAX.T.dot(W * (resI)) + restheta / Kp
        dlogp[-1] = -2 * theta.T.dot(theta / K) / I0 ** 3 + 2 * theta0.T.dot(theta / K) / I0 ** 2
        return logp, dlogp

    def solve_MAP(self, Iapp, A, W, Kinv, theta):
        Dinv = A.T.dot(W[:, None] * A) + np.diag(Kinv)
        res = np.linalg.solve(Dinv, A.T.dot(W * Iapp) + theta * Kinv)
        return res

    def FitPolyNew(self, Vals, JonesNorm, MaxDirty):
        """
        This is for the new frequency fit mode incorporating FreqBandsFluxRation
        :param Vals: Values in the imaging bands
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
        self.CurrentFacet = self.PSFServer.iFacet
        if self.BeamEnable:
            # get BeamFactor
            SumJonesChanList, SumJonesChanWeightSqList, MeanJonesBand = self.GiveBeamFactorsFacet(self.CurrentFacet)
            SumJonesChan = np.concatenate(SumJonesChanList)
            SumJonesChanWeightSq = np.concatenate(SumJonesChanWeightSqList)
            BeamFactor = np.sqrt(SumJonesChan/SumJonesChanWeightSq)  # unstitched sqrt(JonesNorm) at full resolution
            # incorporate stitched JonesNorm
            JonesFactor = np.sqrt(MeanJonesBand/JonesNorm)
            # scale I0 by beam at ref_freq
            I0 = MaxDirty / BeamFactor[self.ref_freq_index]  # TODO - learn optimal I0 for prior from evidence
            # next compute the product of the averaging matrix and beam matrix
            # ChanMappingGrid = self.PSFServer.DicoMappingDesc["ChanMappingGrid"]
            # ChanMappingGridChan = self.PSFServer.DicoMappingDesc["ChanMappingGridChan"]
            SAmat = self.S * BeamFactor[None, :] / JonesFactor[:, None]  # The division by JonesFactor corrects for the fact that the PSF is normalised
            self.SAX = SAmat.dot(self.Xdes_full)
        else:
            I0 = MaxDirty
        # get MFS weights
        W = self.PSFServer.DicoVariablePSF['SumWeights'].squeeze().astype(np.float64)
        Ig = Vals.astype(np.float64)
        if I0 > 0.0:
            # get initial MAP estimate as initial guess
            theta = self.solve_MAP(Ig, self.SAX, W, self.prior_invcov/I0**2, I0*self.prior_theta)
            if self.GD["WSCMS"]["AlphaPrior"] is not None:
                x0 = np.concatenate((theta, np.array([I0])))
                params = fmin_l_bfgs_b(self.logp_and_dlogp, x0,
                                       args=(Ig, self.SAX, W, 1.0/self.prior_invcov, self.prior_theta),
                                       approx_grad=False, bounds=self.bnds + ((1e-6, None),))
                theta = params[0][0:-1]
        else:
            theta = self.solve_MAP(Vals.astype(np.float64), self.SAX, W, self.prior_invcov_neg/I0**2, I0*self.prior_theta_neg)
            if self.GD["WSCMS"]["AlphaPriorNeg"] is not None:
                x0 = np.concatenate((theta, np.array([I0])))
                params = fmin_l_bfgs_b(self.logp_and_dlogp, x0,
                                       args=(Ig, self.SAX, W, 1.0/self.prior_invcov_neg, self.prior_theta_neg),
                                       approx_grad=False, bounds=self.bnds + ((None, -1e-6),))
                theta = params[0][0:-1]
        return theta

    def FitPoly(self, Vals):
        """
        Fits a polynomial to Vals. The order of the polynomial is set when the class is instantiated and defaults to 5.
        The input frequencies are also set in the constructor.
        Args:
            Vals: Function values at input frequencies
        Returns:
            Coefficients of polynomial in order (1,v,v**2,...)
        """
        return np.dot(self.AATinvAT, Vals)

    def EvalPoly(self, coeffs, Freqsp=None):
        """
        Evaluates a polynomial at Freqs with coefficients coeffs
        Args:
            coeffs: the coefficients of the polynomial in order corresponding to (1,v,v**2,...)
            Freqs: the frequencies at which to evaluate the polynomial
        Returns:
            The polynomial evaluated at Freqs
        """
        if np.all(Freqsp == self.Freqs):
            # Here we don't need to reset the design matrix
            return np.dot(self.Xdes, coeffs)
        else:
            # Here we do
            Xdes = self.setDesMat(Freqsp, order=self.order, mode=self.GD[self.DeconvMode]['FreqMode'])
            # evaluate poly and return result
            return np.dot(Xdes, coeffs)

    # IMPORTANT!!!! If beam is enabled assumes self.SAX is set in previous call to FitPolyNew
    # TODO - more reliable way to do this, maybe store in dict keyed on components
    def EvalPolyApparent(self, coeffs, Freqsp=None):
        """
        Gives the apparent flux for coeffs given beam in this facet
        Args:
            coeffs: the coefficients of the polynomial in order corresponding to (1,v,v**2,...)
            Freqs: the frequencies at which to evaluate the polynomial
        Returns:
            The polynomial evaluated at Freqs
        """
        return self.SAX.dot(coeffs)

    def DistributeFreqs(self, vals, freqs):
        """
        Distribute (equally for now but should be done according to channel mapping) Nchan vals into Nchan_degrid channels
        :param vals: 
        :param freqs: 
        :return: 
        """
        Nchan = vals.size
        Nchan_degrid = freqs.size
        Fpol = np.zeros(Nchan_degrid, dtype=np.float32)
        bin_width = Nchan_degrid // Nchan_degrid
        for ch in xrange(Nchan):
            if ch == Nchan-1:
                Fpol[ch * bin_width::] = vals[ch]
            else:
                Fpol[ch*bin_width:(ch+1)*bin_width] = vals[ch]
        return Fpol