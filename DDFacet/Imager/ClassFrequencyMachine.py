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
from DDFacet.ToolsDir import ClassRRGP
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
                # get polynomial coeffs for prior when I0 is positive
                self.alpha_prior_neg = self.GD["WSCMS"]["AlphaPriorNeg"]
                if self.alpha_prior_neg is not None:
                    I = (self.Freqsp/self.ref_freq)**(self.alpha_prior_neg)
                    coeffs, Kfull = np.polyfit(self.Freqsp/self.ref_freq, I, deg=self.order-1, cov=True)
                    self.prior_theta_neg = coeffs[::-1]
                    self.prior_invcov_neg = 1.0/np.diag(Kfull)[::-1]
                else:
                    self.prior_theta_neg = np.zeros(self.order, dtype=np.float64)
                    self.prior_invcov_neg = np.zeros(self.order, dtype=np.float64)
                # construct design matrix at full channel resolution
                self.Xdes = self.setDesMat(self.Freqsp, order=self.order, mode=self.GD['WSCMS']['FreqBasis'])
                ChanMappingGrid = self.PSFServer.DicoMappingDesc["ChanMappingGrid"]
                self.nchan_full = np.size(ChanMappingGrid[0])
                self.freqs_full = []
                for iCh in xrange(self.nchan):
                    self.freqs_full.append(self.PSFServer.DicoVariablePSF["freqs"][iCh])
                self.freqs_full = np.concatenate(self.freqs_full)

                self.Xdes_full = self.setDesMat(self.freqs_full, order=self.order, mode=self.GD['WSCMS']['FreqBasis'])

                print "                      1 = ", np.shape(self.Xdes_full), self.nchan_full, self.freqs_full

                # there is no need to recompute this every time if the beam is not enabled because same everywhere
                if not self.BeamEnable:
                    self.SAX = self.setDesMat(self.Freqs, order=self.order, mode='Andre')  # this fits the integrated polynomial
                self.Fit = self.FitPolyNew
                self.Eval = self.EvalPolyApparent
                self.Eval_Degrid = lambda coeffs, Freqs: self.EvalPoly(coeffs, Freqsp=Freqs)
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
            elif mode == "GPR":
                # Instantiate the GP
                self.GP = ClassRRGP.RR_GP(self.Freqs/self.ref_freq,self.Freqsp/self.ref_freq,
                                          self.GD["Hogbom"]["MaxLengthScale"],self.GD["Hogbom"]["NumBasisFuncs"])
                # Set default initial length scale
                self.l0 = (self.GP.x.max() - self.GP.x.min()) / 2
                # Set the fit and eval methods
                self.Fit = lambda vals: self.FitGP(vals)
                self.Eval = lambda coeffs : self.EvalGP(coeffs, Freqsp=self.Freqs)
                self.Eval_Degrid = lambda coeffs, Freqs : self.EvalGP(coeffs, Freqsp=Freqs)
            elif mode is None:  # TODO - test that this does the expected thing when Nchan != Nchan_degrid
                self.Fit = lambda vals: vals
                self.Eval = lambda vals: vals
                self.Eval_Degrid = lambda vals, Freqs: self.DistributeFreqs(vals, Freqs)
            else:
                raise NotImplementedError("Frequency fit mode %s not supported" % mode)

    def getFitMask(self, FitCube, Threshold=0.0, SetNegZero=False, ResidCube=None):
        """
        Args:
            FitCube     = The cube to fit an alpha map to
            Threshold   = The threshold above which to fit. Defaults to zero.
            SetNegZero  = Whether to set negative pixels to zero. This is required if we want to fit the alhpa map for example. Defaults to False. Only use with PolMode = 'I'
            ResidCube   = The spectral cube of residuals
        Returns:
            FitMask     = A 0/1 mask image
            MaskIndices = The indices at which the mask is non-zero (i.e. the mask is extracted at the indices MaskIndices[:,0],MaskIndices[:,1])
        """
        if ResidCube is not None:
            if SetNegZero:
                    # Find negative indices (these are just set to zero for now)
                    ineg = np.argwhere(FitCube + ResidCube < 0.0)
                    FitCube[ineg[:, 0], ineg[:, 1], ineg[:, 2]] = 0.0

            # Find where I is above threshold (in any frequency band)
            # FitMax = np.amax(FitCube, axis=0)
            # Ip = FitMax > Threshold
            FitMin = np.amin(FitCube, axis=0)
            In = FitMin > Threshold
            #mind = Ip & In
            MaskIndices = np.argwhere(In)
            print>>log, "Adding in residuals"
            FitMask = FitCube[:, MaskIndices[:, 0], MaskIndices[:, 1]] + ResidCube[:, MaskIndices[:, 0], MaskIndices[:, 1]]
        else:
            if SetNegZero:
                # Find negative indices (these are just set to zero for now)
                ineg = np.argwhere(FitCube < 0.0)
                FitCube[ineg[:, 0], ineg[:, 1], ineg[:, 2]] = 0.0

                # Find where I is above threshold (in any frequency band)
                # FitMax = np.amax(FitCube, axis=0)
                # Ip = FitMax > Threshold
            FitMin = np.amin(FitCube, axis=0)
            In = FitMin > Threshold
            # mind = Ip & In
            MaskIndices = np.argwhere(In)
            FitMask = FitCube[:, MaskIndices[:, 0], MaskIndices[:, 1]]
        return FitMask, MaskIndices

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
        if mode == "Mono":
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
            print "Design matrix set Andre style"
        else:
            raise NotImplementedError("Frequency basis %s not supported" % mode)
        return Xdesign

    def FitAlphaMap(self, FitCube, threshold=0.1, ResidCube=None):
        """
        Here we fit a spectral index model to each pixel in the model image above threshold. Note only positive pixels can be used.
        Args:
            FitCube     = The cube to fit the alpha map to ; shape = [Nch, Nx,Ny]
            threshold   = The threshold above which to fit the model
        """
        if self.GD["AlphaMap"]['Mode'] == 0:
            # Get Stokes I components
            FitCube = FitCube[:, 0, :, :]
            if ResidCube is not None:
                ResidCube = ResidCube[:, 0, :, :]
            # Get size of image
            nchan = FitCube.shape[0]
            Nx = FitCube.shape[1]
            Ny = FitCube.shape[2]

            # Get the > 0 components
            IMask,MaskInd = self.getFitMask(FitCube, Threshold=threshold, SetNegZero=True, ResidCube=ResidCube)
            ix = MaskInd[:, 0]
            iy = MaskInd[:, 1]

            # Get array to fit model to
            nsource = ix.size
            IFlat = IMask.reshape([nchan, nsource])

            # Get the model Image as a function of frequency at all these locations
            logI = np.log(IFlat)

            # Create the design matrix (at full freq resolution)
            p = 2 #polynomial order
            XDes = self.setDesMat(self.full_freqs, order=p, mode="log")
            # get the channel mapping
            ChanMappingGrid = self.PSFServer.DicoMappingDesc["ChanMappingGrid"]
            # set the averaging matrix
            Smat = np.zeros([self.nchan, self.nchan_full])
            for iCh in xrange(self.nchan):  # TODO - modify for multiple MS
                I = np.argwhere(ChanMappingGrid[0] == iCh).squeeze()
                nchunk = np.size(I)
                Smat[iCh, I] = 1.0 / nchunk
            # get SX
            SX = Smat.dot(XDes)
            # Solve the system
            Sol = np.dot(np.linalg.inv(SX.T.dot(self.weights[:, None]*SX)), np.dot(SX.T, self.weights[:, None]*logI))
            logIref = Sol[0, :]
            #self.logIref = logIref
            alpha = Sol[1::,:].reshape(logIref.size)
            #self.alpha = alpha
            # Create the alpha map
            self.alpha_map = np.zeros([Nx, Ny])
            if int(np.version.version.split('.')[1]) > 9: #check numpy version > 9 (broadcasting fails for older versions)
                self.alpha_map[ix, iy] = alpha
            else:
                for j in xrange(ix.size):
                    self.alpha_map[ix[j],iy[j]] = alpha[j]

            # Get I0 map
            self.Iref = np.zeros([Nx, Ny])
            if int(np.version.version.split('.')[1]) > 9: # check numpy version > 9 (broadcasting fails for older versions)
                self.Iref[ix, iy] = np.exp(logIref)
            else:
                for j in xrange(ix.size):
                    self.Iref[ix[j], iy[j]] = np.exp(logIref[j])

            # Re-weight the alphas according to flux of model component
            self.weighted_alpha_map = self.alpha_map*self.Iref
            #print self.weighted_alpha_map.min(), self.weighted_alpha_map.max()

            # Create a dict to store model components spi's
            self.alpha_dict = {}
            for j, key in enumerate(zip(ix,iy)):
                self.alpha_dict[key] = {}
                self.alpha_dict[key]['alpha'] = alpha[j]
                self.alpha_dict[key]['Iref'] = np.exp(logIref[j])

            # # Get the variance estimate of residuals
            # epshat = logI - np.dot(XDes, Sol)
            # epsvar = np.diag(np.dot(epshat.T, self.weights[:, None]*epshat))/(nchan - p)
            # #print epsvar.min(), epsvar.max()
            # self.var_map = np.zeros([Nx, Ny])
            # if int(np.version.version.split('.')[1]) > 9: #check numpy version > 9 (broadcasting fails for older versions)
            #     self.var_map[ix, iy] = epsvar
            # else:
            #     for j in xrange(ix.size):
            #         self.var_map[ix[j],iy[j]] = epsvar[j]
            #
            # # Get the variance estimate of the alphas (assuming normally distributed errors, might want to compute confidence intervals)
            # w = self.Freqs/self.ref_freq
            # wbar = np.mean(w)
            # alphavar = np.sqrt(epsvar/np.sum((w-wbar)**2))
            # self.alpha_var_map = np.zeros([Nx, Ny])
            # if int(np.version.version.split('.')[1]) > 9:  # check numpy version > 9 (broadcasting fails for older versions)
            #     self.alpha_var_map[ix, iy] = alphavar
            # else:
            #     for j in xrange(ix.size):
            #         self.alpha_var_map[ix[j], iy[j]] = alphavar[j]
            # self.weighted_alpha_var_map = self.alpha_var_map*self.Iref

        elif self.GD["AlphaMap"]['Mode'] == 1:
            return 1

    def EvalAlphamap(self, Freqs):
        """

        Args:
            Freqs   = The frequencies at which to evulaute the model image from the alpha map

        Returns:
            IM      = The model evaluated at Freqs
        """
        # Compute basis functions
        w = Freqs/self.ref_freq

        nfreqs = Freqs.size
        # Reconstruct the model
        Nx, Ny = self.alpha_map.shape
        IM = np.zeros([nfreqs, Nx, Ny])
        for i in xrange(nfreqs):
            IM[i, :, :] = self.Iref*w[i]**self.alpha_map
        return IM

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
            ChanMappingGrid = self.PSFServer.DicoMappingDesc["ChanMappingGrid"]
            SAmat = np.zeros([self.nchan, self.nchan_full])
            for iCh in xrange(self.nchan):
                I = np.argwhere(ChanMappingGrid[0] == iCh).squeeze()  # TODO - test on multiple MSs
                nchunk = np.size(I)
                if nchunk:
                    SAmat[iCh, I] = BeamFactor[I]/(nchunk*JonesFactor[iCh])  # The division by JonesFactor corrects for the fact that the PSF is normalised
                else:
                    SAmat[iCh, I] = 0.0  # if the chunk is empty this avoids division by zero but weights should also be zero here
                    Wtmp = self.PSFServer.DicoVariablePSF['SumWeights'].squeeze().astype(np.float64)[iCh]
                    if Wtmp != 0:
                        print "Your weights for chunk %i should be zero but its %f" % (iCh, Wtmp)
            print "                                   2 = ", np.shape(SAmat), np.shape(self.Xdes_full)
            self.SAX = SAmat.dot(self.Xdes_full)
        else:
            I0 = MaxDirty
        # get MFS weights
        W = self.PSFServer.DicoVariablePSF['SumWeights'].squeeze().astype(np.float64)
        if I0 > 0.0:
            theta = self.solve_MAP(Vals.astype(np.float64), self.SAX, W, self.prior_invcov/I0**2, I0*self.prior_theta)
        else:
            theta = self.solve_MAP(Vals.astype(np.float64), self.SAX, W, self.prior_invcov_neg/I0**2, I0*self.prior_theta_neg)
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
        if np.all(Freqsp == self.Freqsp):
            # Here we don't need to reset the design matrix
            return np.dot(self.Xdes, coeffs)
        else:
            # Here we do
            Xdes = self.setDesMat(Freqsp, order=self.order, mode=self.GD[self.DeconvMode]['FreqBasis'])
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

    def FitGP(self, Vals):
        """
        Here we fit a reduced rank GP to the frequency axis
        Args:
            Freqs       = The frequencies at which to evulaute the GP

        Returns:
            IM          = The model image at Freqs

        """
        # Set initial guess for hypers
        sigmaf0 = np.maximum(Vals.max() - Vals.min(), 1.1e-5)
        sigman0 = np.maximum(np.var(Vals), 1.1e-4)
        theta = np.array([sigmaf0, self.l0, sigman0])

        # Fit and evaluate GP
        coeffs, thetaf = self.GP.RR_EvalGP(theta, Vals)

        # if (coeffs <= 1.0e-8).all():
        #     print "Something went wrong with GPR"
        #     print self.GP.SolverFlag
        #     print thetaf

        return coeffs

    def EvalGP(self, coeffs, Freqsp=None):
        if np.all(Freqsp == self.Freqs):
            return self.GP.RR_From_Coeffs(coeffs)
        elif np.all(Freqsp == self.Freqsp):
            return self.GP.RR_From_Coeffs_Degrid(coeffs)
        elif np.all(Freqsp == self.ref_freq):
            return self.GP.RR_From_Coeffs_Degrid_ref(coeffs)
        else:
            raise NotImplementedError('GPR mode only predicts to GridFreqs, DegridFreqs and ref_freq')
