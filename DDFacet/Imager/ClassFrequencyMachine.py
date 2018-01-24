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
# import matplotlib.pyplot as plt
from DDFacet.Other import MyPickle

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
    def __init__(self, Freqs, Freqsp, ref_freq, GD=None):
        self.Freqs = np.asarray(Freqs)
        # Use the longer of the two frequency arrays
        if len(Freqs)>len(Freqsp):
            self.Freqsp = np.asarray(Freqs)
        else:
            self.Freqsp = np.asarray(Freqsp)
        self.nchan = self.Freqs.size
        self.nchan_degrid = self.Freqsp.size
        #print "Nchan =", self.nchan
        # # Get Stokes parameters
        # self.IStokes = ModelCube[:, 0, :, :]
        # if self.npol > 1:
        #     self.QStokes = ModelCube[:, 1, :, :]
        # if self.npol > 2:
        #     self.UStokes = ModelCube[:, 2, :, :]
        # if self.npol > 3:
        #     self.VStokes = ModelCube[:, 3, :, :]
        # self.ModelCube = ModelCube
        self.ref_freq = ref_freq
        self.GD = GD

    def set_Method(self, mode="Poly"):
        if self.nchan==1: #hack to deal with a single channel
            self.Fit = lambda vals: vals
            self.Eval = lambda vals: vals # this will just be the value in that channel
            self.Eval_Degrid = lambda vals: np.tile(vals, self.nchan_degrid)
        else:
            if mode == "Poly":
                self.order = self.GD["Hogbom"]["PolyFitOrder"]
                self.Xdes = self.setDesMat(self.Freqs, order=self.order)
                if self.nchan >= self.order: # use left pseudo inverse
                    self.AATinvAT = np.linalg.inv(self.Xdes.T.dot(self.Xdes)).dot(self.Xdes.T)
                else: # use right pseudo inverse
                    self.AATinvAT = self.Xdes.T.dot(np.linalg.inv(self.Xdes.dot(self.Xdes.T)))
                #print "PI shape = ", self.AATinvAT.shape
                # Set the fit and eval methods
                self.Fit = lambda vals: self.FitPoly(vals)
                self.Eval = lambda coeffs : self.EvalPoly(coeffs, Freqsp=self.Freqs)
                self.Eval_Degrid = lambda coeffs, Freqs : self.EvalPoly(coeffs, Freqsp=Freqs)
            elif mode == "GPR":
                # Instantiate the GP
                self.GP = ClassRRGP.RR_GP(self.Freqs/self.ref_freq,self.Freqsp/self.ref_freq,self.GD["Hogbom"]["MaxLengthScale"],self.GD["Hogbom"]["NumBasisFuncs"])
                # Set default initial length scale
                self.l0 = (self.GP.x.max() - self.GP.x.min()) / 2
                # Set the fit and eval methods
                self.Fit = lambda vals: self.FitGP(vals)
                self.Eval = lambda coeffs : self.EvalGP(coeffs, Freqsp=self.Freqs)
                self.Eval_Degrid = lambda coeffs, Freqs : self.EvalGP(coeffs, Freqsp=Freqs)

    def getFitMask(self, FitCube, Threshold=0.0, SetNegZero=False):
        """
        Args:
            FitCube     = The cube to fit an alpha map to
            Threshold   = The threshold above which to fit. Defaults to zero.
            SetNegZero  = Whether to set negative pixels to zero. This is required if we want to fit the alhpa map for example. Defaults to False. Only use with PolMode = 'I'
        Returns:
            FitMask     = A 0/1 mask image
            MaskIndices = The indices at which the mask is non-zero (i.e. the mask is extracted at the indices MaskIndices[:,0],MaskIndices[:,1])
        """
        # Remove redundant axis
        if SetNegZero:
                # Find negative indices (these are just set to zero for now)
                ineg = np.argwhere(FitCube < 0.0)
                FitCube[ineg[:, 0], ineg[:, 1], ineg[:, 2]] = 0.0

        # Find where I is above threshold (in any frequency band)
        # FitMax = np.amax(FitCube, axis=0)
        # Ip = FitMax > Threshold
        FitMin = np.amin(FitCube, axis=0)
        In = FitMin > Threshold
        #mind = Ip & In
        MaskIndices = np.argwhere(In)
        FitMask = FitCube[:, MaskIndices[:, 0], MaskIndices[:, 1]]
        return FitMask, MaskIndices

    def setDesMat(self, Freqs, order=None, mode="Normal"):
        """
        This function creates the design matrix
        Args:
            order   = The order of the polynomial fit
            mode    = "Normal" or "log" determines if the design matrix should be built in log space or not
        Returns:
            Xdesign    = The design matrix [1, (v/v_0), (v/v_0)**2, ...]

        """
        if order is None:
            order = self.order
        if mode=="Normal":
            # Construct vector of frequencies
            w = (Freqs / self.ref_freq).reshape(Freqs.size, 1)
            # create tiled array and raise each column to the correct power
            Xdesign = np.tile(w, order) ** np.arange(0, order)
        elif mode=="log":
            # Construct vector of frequencies
            w = np.log(Freqs / self.ref_freq).reshape(Freqs.size, 1)
            # create tiled array and raise each column to the correct power
            Xdesign = np.tile(w, order) ** np.arange(0, order)
        else:
            raise NotImplementedError("mode %s not supported" % mode)
        return Xdesign

    def FitAlphaMap(self, FitCube, threshold=0.1):
        """
        Here we fit a spectral index model to each pixel in the model image above threshold. Note only positive pixels can be used.
        Args:
            FitCube     = The cube to fit the alpha map to ; shape = [Nch, Nx,Ny]
            threshold   = The threshold above which to fit the model
        """
        # Get size of image
        nchan = FitCube.shape[0]
        Nx = FitCube.shape[1]
        Ny = FitCube.shape[2]

        # Get the > 0 components
        IMask,MaskInd = self.getFitMask(FitCube, Threshold=threshold, SetNegZero=True)
        ix = MaskInd[:, 0]
        iy = MaskInd[:, 1]

        # Get array to fit model to
        nsource = ix.size
        IFlat = IMask.reshape([nchan, nsource])

        # Get the model Image as a function of frequency at all these locations
        logI = np.log(IFlat)

        # Create the design matrix
        p = 2 #polynomial order
        XDes = self.setDesMat(self.Freqsp, order=p, mode="log")
        # Solve the system
        Sol = np.dot(np.linalg.inv(XDes.T.dot(XDes)), np.dot(XDes.T, logI))
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
        # epsvar = np.diag(np.dot(epshat.T, epshat))/(nchan - p)
        # #print epsvar.min(), epsvar.max()
        # self.var_map = np.zeros([Nx, Ny])
        # if int(np.version.version.split('.')[1]) > 9: #check numpy version > 9 (broadcasting fails for older versions)
        # 	self.var_map[ix, iy] = epsvar
        # else:
        # 	for j in xrange(ix.size):
        # 		self.var_map[ix[j],iy[j]] = epsvar[j]
        #
        # # Get the variance estimate of the alphas (assuming normally distributed errors, might want to compute confidence intervals)
        # w = self.Freqs/self.ref_freq
        # wbar = np.mean(w)
        # alphavar = np.sqrt(epsvar/np.sum((w-wbar)**2))
        # self.alpha_var_map = np.zeros([Nx, Ny])
        # if int(np.version.version.split('.')[1]) > 9:  # check numpy version > 9 (broadcasting fails for older versions)
        # 	self.alpha_var_map[ix, iy] = alphavar
        # else:
        # 	for j in xrange(ix.size):
        # 		self.alpha_var_map[ix[j], iy[j]] = alphavar[j]
        # self.weighted_alpha_var_map = self.alpha_var_map*self.Iref
        return

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
        elif np.all(Freqsp == self.Freqsp):
            # Here we do
            order = coeffs.size
            Xdes = self.setDesMat(Freqsp, order=order)
            # evaluate poly and return result
            return np.dot(Xdes, coeffs)
        elif np.all(Freqsp == self.ref_freq):
            order = coeffs.size
            Xdes = self.setDesMat(Freqsp, order=order)
            return np.dot(Xdes, coeffs)
        else:
            #frequencies changed so we need a new design matrix
            order = coeffs.size
            Xdes = self.setDesMat(Freqsp, order=order)
            # evaluate poly and return result
            return np.dot(Xdes, coeffs)


    def FitGP(self,Vals):
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
