import numpy as np
from DDFacet.ToolsDir import ClassGP

class ClassFrequencyMachine(object):
    """
    Interface to fit frequency axis in model image. All fitting is currently based on some polynomial model in normalised frequencies (v/v_0).
    For the alpha map the fit happens in log space.
        Initialisation:
                ModelCube   = A cube containing the model image with shape [NChannel,NPol,Npix,Npix]
                Freqs       = The Frequencies corresponding to the model image
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
    def __init__(self, Freqs, ref_freq, order=5):
        self.nchan = Freqs.size
        # # Get Stokes parameters
        # self.IStokes = ModelCube[:, 0, :, :]
        # if self.npol > 1:
        #     self.QStokes = ModelCube[:, 1, :, :]
        # if self.npol > 2:
        #     self.UStokes = ModelCube[:, 2, :, :]
        # if self.npol > 3:
        #     self.VStokes = ModelCube[:, 3, :, :]
        # self.ModelCube = ModelCube
        self.Freqs = Freqs
        self.ref_freq = ref_freq
        self.order = order
        self.Xdes = self.setDesMat(Freqs, order=self.order)
        self.AATinvAT = np.dot(np.linalg.inv(self.XDes.T.dot(self.XDes)), self.XDes.T)

    def getFitMask(self, Threshold=0.0, SetNegZero=False, PolMode='I'):
        """
        Args:
            Threshold   = The threshold above which to fit. Defaults to zero.
            SetNegZero  = Whether to set negative pixels to zero. This is required if we want to fit the alhpa map for example. Defaults to False. Only use with PolMode = 'I'
            PolMode     = Which Stokes parameter to mask. Defaults to I.
        Returns:
            FitMask     = A 0/1 mask image
            MaskIndices = The indices at which the mask is non-zero (i.e. the mask is extracted at the indices MaskIndices[:,0],MaskIndices[:,1])
        """
        if PolMode == "I":
            if SetNegZero:
                # Find negative indices (these are just set to zero for now)
                ineg = np.argwhere(self.IStokes < 0.0)
                FitCube = self.IStokes
                FitCube[ineg[:, 0], ineg[:, 1], ineg[:, 2]] = 0.0
            else:
                FitCube = self.IStokes
        elif PolMode == "Q":  # For anything but I we will have to figure out how to use the threshold. Maybe using abs value?
            FitCube = self.QStokes
        elif PolMode == "U":
            FitCube = self.UStokes
        elif PolMode == "V":
            FitCube = self.VStokes

        # Find where I is above threshold (in any frequency band)
        FitMax = np.amax(FitCube,axis=0)
        MaskIndices = np.argwhere(FitMax > Threshold)
        FitMask = FitCube[:, MaskIndices[:, 0], MaskIndices[:, 1]]
        return FitMask, MaskIndices

    def setDesMat(self, Freqs, order=5, mode="Normal"):
        """
        This function creates the design matrix
        Args:
            order   = The order of the polynomial fit
            mode    = "Normal" or "log" determines if the design matrix should be built in log space or not
        Returns:
            Xdesign    = The design matrix [1, (v/v_0), (v/v_0)**2, ...]

        """
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

    def FitAlphaMap(self,threshold=0.1,order=2):
        """
        Here we fit a spectral index model to each pixel in the model image above threshold. Note only positive pixels can be used.
        Args:
            threshold   = The threshold above which to fit the model
            order       = The order of the spectral model fit. The default is to fit I(v) = I(v_0) (v/v_0)**alpha but we can also
                        allow for spectral curvature etc. by doing a higher order fit in log space (i.e. order = 3 gives spectral
                        curvature).
        """
        # Get the mask and mask indices
        IMask,MaskInd = self.getFitMask(Threshold=threshold, SetNegZero=True, PolMode="I")
        ix = MaskInd[:, 0]
        iy = MaskInd[:, 1]

        # Get array to fit model to
        nsource = ix.size
        IFlat = IMask.reshape([self.nchan, nsource])

        # Get the model Image as a function of frequency at all these locations
        logI = np.log(IFlat)

        # Create the design matrix
        XDes = self.setDesMat(self.Freqs, order=order, mode="log")

        # Solve the system
        Sol = np.dot(np.linalg.inv(XDes.T.dot(XDes)), np.dot(XDes.T, logI))
        logIref = Sol[0, :]
        #self.logIref = logIref
        alpha = Sol[1::,:]
        #self.alpha = alpha
        # Create the alpha map
        self.alpha_map = np.zeros([self.Nx, self.Ny])
        self.alpha_map[ix, iy] = alpha[:, 0]

        # Get I0 map
        self.Iref = np.zeros([self.Nx, self.Ny])
        self.Iref[ix, iy] = np.exp(logIref)
        return

    def EvalAlphamap(self,Freqs):
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
        IM = np.zeros([nfreqs, self.Nx, self.Ny])
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

    def EvalPoly(self,coeffs,Freqs):
        """
        Evaluates a polynomial at Freqs with coefficients coeffs
        Args:
            coeffs: the coefficients of the polynomial in order corresponding to (1,v,v**2,...)
            Freqs: the frequencies at which to evaluate the polynomial
        Returns:
            The polynomial evaluated at Freqs
        """
        order = coeffs.size
        Xdes = self.setDesMat(Freqs,order=order)
        # evaluate poly and return result
        return np.dot(Xdes,coeffs.reshape(order,1))

    def FitPolyCube(self, deg=4, threshold = 0.0, PolMode = "I", weights="Default"):
        """
        Fits polynomial of degree=deg with weights=weights to each pixel in the model image above threshold
        """
        # Set weights to identity if default
        if weights == "Default":
            weights = np.ones(deg)

        if deg > self.nchan:
            print "Warning: The degree of the polynomial should not be greater than the number of bands/channels. The system is ill conditioned"
            deg = self.nchan

        # Initialise array to store coefficients
        self.coeffs = np.zeros([deg, self.Nx, self.Ny])

        #Get the fit mask
        IMask, MaskInd = self.getFitMask(Threshold=0.0, SetNegZero=False, PolMode="I")
        ix = MaskInd[:, 0]
        iy = MaskInd[:, 1]

        # Get array to fit model to
        nsource = ix.size
        IFlat = IMask.reshape([self.nchan,nsource])

        # Create the design matrix
        XDes = self.setDesMat(order=deg, mode="Normal")

        # Solve the system
        Sol = np.dot(np.linalg.inv(XDes.T.dot(XDes)), np.dot(XDes.T, IFlat))

        print Sol

        self.coeffs[:, ix, iy] = Sol
        return


    def EvalPolyCube(self, Freqs):  # ,Ix,Iy
        """
        Evaluates the polynomial at locations (Ix,Iy) and frequencies Freqs
        """
        # Get the degree of the polynomial
        deg, _, _, _ = self.coeffs.shape
        w = Freqs / self.ref_freq
        tmp = self.coeffs[0, :, :]
        for i in xrange(1, deg):
            tmp += self.coeffs[i, :, :] * w[:, np.newaxis, np.newaxis] ** deg
        return tmp

    def FitGP(self,Freqs):
        """
        Here we fit a GP to the frequency axis of the model cube to pixels above a certain threshold
        Args:
            Freqs       = The frequencies at which to evulaute the GP

        Returns:
            IM          = The model image at Freqs

        """
        # Initialise GP
        GP = ClassGP.ClassGP(self.Freqs,Freqs)

        # Get the mask
        IMask,MaskInd = self.getFitMask(Threshold=0.0, SetNegZero=False, PolMode="I")
        ix = MaskInd[:, 0]
        iy = MaskInd[:, 1]

        # Get array to fit model to
        nsource = ix.size
        Iflat = IMask.reshape([self.nchan,nsource])

        # Set initial guess for theta
        theta = np.ones(3)

        # Create storage arrays
        IMFlat = np.zeros([Freqs.size, nsource])
        IM = np.zeros([Freqs.size, self.Nx, self.Ny])
        for i in xrange(nsource):
            IMFlat[:, i] = GP.EvalGP(Iflat[:, i], theta)
        # Get model in 2D shape
        IM[:, ix, iy] = IMFlat
        return IM


def testFM():
    #Create array to hold model image
    N = 100
    Nch = 10
    IM = np.zeros([Nch,1,N,N])

    # Choose some random indices to populate
    nsource = 25
    ix = np.random.randint(0, N, nsource)
    iy = np.random.randint(0, N, nsource)

    #Populate model
    IM[:,:,ix,iy] = 1.0

    #Set frequencies
    Freqs = np.linspace(1.0,3.0,Nch)
    ref_freq = 2.0

    #Create frequency machine
    fmachine = ClassFrequencyMachine(IM,Freqs,ref_freq)

    #Fit an alpha map (should be getting all zeros)
    #fmachine.FitAlphaMap(threshold=0.1,order=2)

    #print np.exp(fmachine.logIref), fmachine.alpha

    # Fit the polynomial
    fmachine.FitPolyCube(4,0.1)



    return