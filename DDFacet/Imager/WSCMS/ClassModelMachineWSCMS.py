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
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassModelMachine")
from DDFacet.Array import NpParallel
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Other import MyPickle
from DDFacet.Other import reformat
from DDFacet.ToolsDir.Gaussian import GaussianSymmetric
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.Imager import ClassModelMachine as ClassModelMachinebase
from DDFacet.Imager import ClassFrequencyMachine, ClassScaleMachine
import os

class ClassModelMachine(ClassModelMachinebase.ClassModelMachine):
    def __init__(self,*args,**kwargs):
        ClassModelMachinebase.ClassModelMachine.__init__(self, *args, **kwargs)
        self.DicoSMStacked={}
        self.DicoSMStacked["Type"]="WSCMS"
        self.n_sub_minor_iter = 0

    def setRefFreq(self, RefFreq, Force=False):
        if self.RefFreq is not None and not Force:
            print>>log, ModColor.Str("Reference frequency already set to %f MHz" % (self.RefFreq/1e6))
            return

        self.RefFreq = RefFreq
        self.DicoSMStacked["RefFreq"] = RefFreq

    def setPSFServer(self, PSFServer):
        self.PSFServer = PSFServer

    def setFreqMachine(self, GridFreqs, DegridFreqs, weights=None, PSFServer=None):
        self.PSFServer = PSFServer
        # Initiaise the Frequency Machine
        self.DegridFreqs = DegridFreqs
        self.GridFreqs = GridFreqs
        self.FreqMachine = ClassFrequencyMachine.ClassFrequencyMachine(GridFreqs, DegridFreqs,
                                                                       self.DicoSMStacked["RefFreq"], self.GD,
                                                                       weights=weights, PSFServer=self.PSFServer)
        self.FreqMachine.set_Method(mode="WSCMS")

        if (self.GD["Freq"]["NBand"] > 1):
            self.Coeffs = np.zeros(self.GD["WSCMS"]["NumFreqBasisFuncs"])
        else:
            self.Coeffs = np.zeros([1])

        self.Nchan = self.FreqMachine.nchan
        self.Npol = 1

        # self.DicoSMStacked["Eval_Degrid"] = self.FreqMachine.Eval_Degrid


    def setScaleMachine(self, PSFServer, NCPU=None, MaskArray=None, FTMachine=None, cachepath=None):
        if self.GD["WSCMS"]["MultiScale"]:
            if NCPU is None:
                self.NCPU = self.GD['Parallel'][NCPU]
                if self.NCPU == 0:
                    import multiprocessing

                    self.NCPU = multiprocessing.cpu_count()
            else:
                 self.NCPU = NCPU
            self.DoAbs = self.GD["Deconv"]["AllowNegative"]
            self.ScaleMachine = ClassScaleMachine.ClassScaleMachine(GD=self.GD, NCPU=NCPU, MaskArray=MaskArray)
            self.FTMachine = FTMachine
            self.ScaleMachine.Init(PSFServer, self.FreqMachine, FTMachine2=self.FTMachine,
                                   cachepath=cachepath)
            self.Nscales = self.ScaleMachine.Nscales
            # Initialise CurrentScale variable
            self.CurrentScale = 999999
            # Initialise current facet variable
            self.CurrentFacet = 999999

            self.DicoSMStacked["Scale_Info"] = {}
            for i, sigma in enumerate(self.ScaleMachine.sigmas):
                if sigma not in self.DicoSMStacked["Scale_Info"].keys():
                    self.DicoSMStacked["Scale_Info"][sigma] = {}
                self.DicoSMStacked["Scale_Info"][sigma]["kernel"] = self.ScaleMachine.kernels[i]
                self.DicoSMStacked["Scale_Info"][sigma]["extent"] = self.ScaleMachine.extents[i]

        else:
            # we need to keep track of what the sigma value of the delta scale corresponds to
            # even if we don't do multiscale (because we need it in GiveModelImage)
            (self.FWHMBeamAvg, _, _) = PSFServer.DicoVariablePSF["EstimatesAvgPSF"]
            self.ListScales = [1.0/np.sqrt(2)*((self.FWHMBeamAvg[0] + self.FWHMBeamAvg[1])*np.pi / 180) / \
                                (2.0 * self.GD['Image']['Cell'] * np.pi / 648000)]

    def ToFile(self, FileName, DicoIn=None):
        print>> log, "Saving dico model to %s" % FileName
        if DicoIn is None:
            D = self.DicoSMStacked
        else:
            D = DicoIn

        D["GD"] = self.GD
        D["Type"] = "WSCMS"
        try:
            D["ListScales"] = list(self.ScaleMachine.sigmas)  # list containing std of Gaussian components
        except:
            D["ListScales"] = self.ListScales
        D["ModelShape"] = self.ModelShape
        MyPickle.Save(D, FileName)

    def FromFile(self, FileName):
        print>> log, "Reading dico model from %s" % FileName
        self.DicoSMStacked = MyPickle.Load(FileName)
        self.FromDico(self.DicoSMStacked)

    def FromDico(self, DicoSMStacked):
        self.DicoSMStacked = DicoSMStacked
        self.RefFreq = self.DicoSMStacked["RefFreq"]
        self.ListScales = self.DicoSMStacked["ListScales"]
        self.ModelShape = self.DicoSMStacked["ModelShape"]

    def setModelShape(self, ModelShape):
        self.ModelShape = ModelShape
        self.Npix = self.ModelShape[-1]

    def AppendComponentToDictStacked(self, key, Sols, Scale, Gain):
        """
        Adds component to model dictionary at a scale specified by Scale. 
        The dictionary corresponding to each scale is keyed on pixel values (l,m location tupple). 
        Each model component is therefore represented parametrically by a pixel value a scale and a set of coefficients
        associated with the spectral function that is fit to the frequency axis.
        Currently only Stokes I is supported.
        Args:
            key: the (l,m) centre of the component
            Fpol: Weight of the solution
            Sols: Nd array of solutions with length equal to the number of basis functions representing the component.
            Scale: the sigma value (Gaussian standard deviation) of the scale at which to append the component    
        Post conditions:
        Added component list to dictionary for particular scale. This dictionary is stored in
        self.DicoSMStacked["Comp"][Scale] and has keys:
            "SolsArray": solutions ndArray with shape [#basis_functions,#stokes_terms]
            "SumWeights": weights ndArray with shape [#stokes_terms]
            
            
        LB - Note I have added and extra scale layer to the dictionary structure
        """
        DicoComp = self.DicoSMStacked.setdefault("Comp", {})

        if Scale not in DicoComp.keys():
            DicoComp[Scale] = {}

        if key not in DicoComp[Scale].keys():
            DicoComp[Scale][key] = {}
            DicoComp[Scale][key]["SolsArray"] = np.zeros(Sols.size, np.float32)
            DicoComp[Scale][key]["NumComps"] = np.zeros(1, np.float32)

        DicoComp[Scale][key]["NumComps"] += 1
        DicoComp[Scale][key]["SolsArray"] += Sols.ravel() * Gain

    def GiveModelImage(self, FreqIn=None, out=None):
        RefFreq=self.DicoSMStacked["RefFreq"]
        # Default to reference frequency if no input given
        if FreqIn is None:
            FreqIn=np.array([RefFreq], dtype=np.float32)

        FreqIn = np.array([FreqIn.ravel()], dtype=np.float32).flatten()

        DicoComp = self.DicoSMStacked.setdefault("Comp", {})
        _, npol, nx, ny = self.ModelShape

        # The model shape has nchan = len(GridFreqs)
        nchan = FreqIn.size
        if out is not None:  # LB - is this for appending components to an existing model?
            if out.shape != (nchan,npol,nx,ny) or out.dtype != np.float32:
                raise RuntimeError("supplied image has incorrect type (%s) or shape (%s)" % (out.dtype, out.shape))
            ModelImage = out
        else:
            ModelImage = np.zeros((nchan,npol,nx,ny),dtype=np.float32)

        # get the zero scale
        try:
            zero_scale = self.ScaleMachine.sigmas[0]
        except:
            zero_scale = self.ListScales[0]

        for scale in DicoComp.keys():
            # Note here we are building a spectral cube delta function representation first and then convolving by
            # the scale at the end
            ScaleModel = np.zeros((nchan, npol, nx, ny), dtype=np.float32)
            # get the extent of the Gaussian
            # I = np.argwhere(scale == self.ScaleMachine.sigmas).squeeze()
            # extent = self.ScaleMachine.extents[I]
            kernel = self.DicoSMStacked["Scale_Info"][scale]["kernel"]
            for key in DicoComp[scale].keys():
                Sol = DicoComp[scale][key]["SolsArray"]
                # TODO - try soft thresholding components
                x, y = key
                extent = self.DicoSMStacked["Scale_Info"][scale]["extent"]
                Aedge, Bedge = GiveEdges((x, y), nx, (extent // 2, extent // 2), extent)

                x0d, x1d, y0d, y1d = Aedge
                x0p, x1p, y0p, y1p = Bedge

                try:  # LB - Should we drop support for anything other than polynomials maybe?
                    interp = self.FreqMachine.Eval_Degrid(Sol, FreqIn)
                except:
                    interp = np.polyval(Sol[::-1], FreqIn/RefFreq)

                if interp is None:
                    raise RuntimeError("Could not interpolate model onto degridding bands. Inspect your data, check "
                                       "'WSCMS-NumFreqBasisFuncs' or if you think this is a bug report it.")

                if self.GD["WSCMS"]["MultiScale"] and scale != zero_scale:
                    try:
                        out = np.atleast_1d(interp)[:, None, None, None] * kernel
                        ScaleModel[:, :, x0d:x1d, y0d:y1d] += out[:, :, x0p:x1p, y0p:y1p]
                        # ScaleModel[:, :, x0d:x1d, y0d:y1d] += self.ScaleMachine.GaussianSymmetric(scale, amp=np.atleast_1d(interp),
                        #                                                                           support=extent)[:, :, x0p:x1p, y0p:y1p]
                    except:
                        x -= nx // 2
                        y -= ny // 2
                        ScaleModel += GaussianSymmetric(scale, nx, x0=x or None,
                                                        y0=y or None, amp=interp, cube=True)
                else:
                    ScaleModel[:, 0, x, y] += interp

            ModelImage += ScaleModel
        return ModelImage

    def GiveSpectralIndexMap(self,CellSizeRad=1.,GaussPars=[(1,1,0)],DoConv=True,MaxSpi=100,MaxDR=1e+6):
        """
        Cyril's old method for computing alpha maps. 
        :param CellSizeRad: 
        :param GaussPars: 
        :param DoConv: 
        :param MaxSpi: 
        :param MaxDR: 
        :return: 
        """
        # TODO - Modify this to convolve over each frequency band instead of only two.
        RefFreq=self.DicoSMStacked["RefFreq"]
        f0=RefFreq/1.5
        f1=RefFreq*1.5

        M0=self.GiveModelImage(f0)
        M1=self.GiveModelImage(f1)
        if DoConv:
            M0=ModFFTW.ConvolveGaussianSimpleWrapper(M0, CellSizeRad=CellSizeRad, GaussPars=GaussPars)
            M1=ModFFTW.ConvolveGaussianSimpleWrapper(M1, CellSizeRad=CellSizeRad, GaussPars=GaussPars)

        # compute threshold for alpha computation by rounding DR threshold to .1 digits (i.e. 1.65e-6 rounds to 1.7e-6)
        if not np.all(M0==0):
            minmod = float("%.1e"%(np.max(np.abs(M0))/MaxDR))
        else:
            minmod=1e-6

        # mask out pixels above threshold
        mask=(M1<minmod)|(M0<minmod)
        print>>log,"computing alpha map for model pixels above %.1e Jy (based on max DR setting of %g)"%(minmod,MaxDR)
        M0[mask]=minmod
        M1[mask]=minmod
        alpha = (np.log(M0)-np.log(M1))/(np.log(f0/f1))
        alpha[mask] = 0

        # mask out |alpha|>MaxSpi. These are not physically meaningful anyway
        mask = alpha>MaxSpi
        alpha[mask]  = MaxSpi
        masked = mask.any()
        mask = alpha<-MaxSpi
        alpha[mask] = -MaxSpi
        if masked or mask.any():
            print>>log,ModColor.Str("WARNING: some alpha pixels outside +/-%g. Masking them."%MaxSpi,col="red")
        return alpha

    def SubStep(self, (dx, dy), LocalSM, Residual):
        """
        Sub-minor loop subtraction
        """
        xc, yc = dx, dy
        N0 = Residual.shape[-1]
        N1 = LocalSM.shape[-1]

        # Get overlap indices where psf should be subtracted
        Aedge, Bedge = GiveEdges((xc, yc), N0, (N1 // 2, N1 // 2), N1)

        x0d, x1d, y0d, y1d = Aedge
        x0p, x1p, y0p, y1p = Bedge

        # Subtract from each channel/band
        Residual[:, :, x0d:x1d, y0d:y1d] -= LocalSM[:, :, x0p:x1p, y0p:y1p]

        return Residual

    def set_ConvPSF(self, iFacet, iScale):
        # we only need to compute the PSF if Facet or Scale has changed
        # note will always be set initially since comparison to 999999 will fail
        if iFacet != self.CurrentFacet or self.CurrentScale != iScale:
            key = 'S' + str(iScale) + 'F' + str(iFacet)
            # update facet (need to make sure PSFserver has been updated when we get here)
            self.CurrentFacet = iFacet

            # update scale
            self.CurrentScale = iScale

            # get the gain in this Facet for this scale. This function actually does most of the work.
            # If the PSF's for this facet and scale have not yet been computed it will compute them and store
            # all the relevant information in the LRU cache which spills to disk automatically if the number
            # of elements exceeds the pre-set maximum in --WSCMS-CacheSize
            self.CurrentGain = self.ScaleMachine.give_gain(iFacet, iScale)

            # twice convolve PSF with scale if not delta scale
            if not iScale:
                PSF, PSFmean = self.PSFServer.GivePSF()
                self.ConvPSF = PSF
                #self.ConvPSFmean = PSFmean
                # delta scale is not cleaned with the ConvPSF so these should all be unity
                self.PSFFreqNormFactors = np.ones([self.Nchan, 1, 1, 1], dtype=np.float32)
                self.FpolNormFactor = 1.0

            else:
                self.ConvPSF = self.ScaleMachine.Conv2PSFs[key]

                # To normalise by the frequency response of ConvPSF implicitly contained in residual
                # we need to keep track of the ConvPSF peaks. Note this is not done in wsclean but should give more
                # even per band residuals
                self.PSFFreqNormFactors = self.ScaleMachine.ConvPSFFreqPeaks[key]

                # Finally normalise PSF by peak of ConvPSF0
                # This would set the peak of ConvPSF0 to unity if we were cleaning
                # the delta scale with the convolved PSF
                #self.ConvPSF /= self.ScaleMachine.ConvPSFNormFactor  # done in set_gains

                # This normalisation for Fpol is required so that we don't see jumps between minor cycles.
                # Basically since the PSF is normalised by this factor the components also need to be normalised
                # by the same factor for the subtraction in the sub-minor cycle to be the same as the subtraction
                # in the minor and major cycles.
                self.FpolNormFactor = self.ScaleMachine.ConvPSFNormFactor

            # from astropy.io import fits
            # key = 'S' + str(iScale) + 'F' + str(iFacet)
            # hdu = fits.PrimaryHDU(self.ConvPSF[:, 0, :, :] * self.FpolNormFactor)
            # hdul = fits.HDUList([hdu])
            # hdul.writeto(
            #     '/home/landman/Projects/Data/MS_dir/ddfacet_test_data/WSCMS_MSMF_TestSuite/DDF_out/compare/Conv2PSF' + key + '.fits',
            #     overwrite=True)
            # hdul.close()

        #print self.CurrentScale, self.CurrentGain

        # import matplotlib.pyplot as plt
        # for iCh in xrange(self.Nchan):
        #     plt.figure('PSF')
        #     plt.imshow(self.ConvPSF[iCh, 0, :, :])
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()


    # not working yet!!!
    def give_scale_mask(self, meanDirty, meanPSF, gain, JonesNorm=None):
        """
        Automatically creates a mask for this scale by doing a shallow clean meanDirty 
        :param meanDirty: The mean dirty image convolved with scale function for the current scale 
        :param meanPSF: The mean PSF twice convolved with the scale function for the current scale
        :param gain: scale dependent gain
        :param JonesNorm: Not sure if I should even include this here
        :return: 
        """
        x, y, MaxDirty = NpParallel.A_whereMax(meanDirty, NCPU=self.NCPU, DoAbs=self.DoAbs,
                                                            Mask=self.ScaleMachine.MaskArray)

        mask = np.zeros_like(meanDirty, dtype=np.bool)

        threshold = 0.9*np.abs(MaxDirty)

        i = 0
        maxit = 500
        while i < maxit and np.abs(MaxDirty) > threshold:
            mask[:, :, x, y] = 1
            meanDirty = self.SubStep((x, y), gain*MaxDirty*meanPSF, meanDirty)
            x, y, MaxDirty = NpParallel.A_whereMax(meanDirty, NCPU=self.NCPU, DoAbs=self.DoAbs,
                                                   Mask=self.ScaleMachine.MaskArray)
            i += 1

        return mask


    def do_minor_loop(self, x, y, Dirty, meanDirty, JonesNorm, W, MaxDirty, Stopping_flux=None):
        """
        Runs the sub-minor loop at a specific scale 
        :param Dirty: 
        :param meanDirty: 
        :param JonesNorm: 
        :return: The scale convolved model and a list of components 
        """
        #print "Entering subminor cycle. MaxDirty = ", MaxDirty
        # determine most relevant scale. This is the most time consuming step hence the sub-minor cycle
        xscale, yscale, ConvMaxDirty, CurrentDirty, ConvDirtyCube, iScale,  = \
            self.ScaleMachine.do_scale_convolve(Dirty.copy(), meanDirty.copy())
        if iScale == 0:
            xscale = x
            yscale = y
            ConvMaxDirty = MaxDirty
            CurrentDirty = meanDirty.view()
            ConvDirtyCube = Dirty.view()

        # print "Max at start = ", ConvMaxDirty, iScale

        self.PSFServer.setLocation(xscale, yscale)

        # get GaussPars for scale
        sigma = self.ScaleMachine.sigmas[iScale]

        # For automasking
        MaskArray = self.ScaleMachine.MaskArray
        ExternalMask = MaskArray if MaskArray is not None else np.ones([1, 1, self.Npix, self.Npix], dtype=np.bool)
        self.ScaleMachine.ScaleMaskArray.setdefault(sigma, ExternalMask.copy())

        # set twice convolve PSF for scale and facet if either has changed
        self.set_ConvPSF(self.PSFServer.iFacet, iScale)

        # update scale dependent mask
        if self.GD["WSCMS"]["AutoMask"]:
            mask = self.give_scale_mask(CurrentDirty.copy(), self.ConvPSFmean, self.CurrentGain)
            self.ScaleMachine.ScaleMaskArray[sigma] &= mask
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(self.ScaleMachine.ScaleMaskArray[sigma][0, 0].astype(np.int8))
            # plt.colorbar()
            # plt.show()

        #print "Mask indices = ", np.argwhere(self.ScaleMachine.ScaleMaskArray[sigma].squeeze()).squeeze()

        # create a model for this scale
        ScaleModel = np.zeros_like(ConvDirtyCube, dtype=np.float32)

        # Create new component list
        Component_list = {}

        # Set stopping threshold.
        # We cannot use StoppingFlux from minor cycle directly since the maxima of the scale convolved residual
        # is different from the maximum of the residual
        Threshold = self.ScaleMachine.PeakFactor * ConvMaxDirty
        if Stopping_flux is not None:
            # # If the location of the peak has changed we need to set the threshold relative to the new peak
            # # Nope this is wrong, we don't care about the location of the peak only what the maximum was
            # if x != xscale or y != yscale:
            #     MaxDirty = meanDirty[0, 0, xscale, yscale]
            DirtyRatio = ConvMaxDirty / MaxDirty
            Threshold = np.maximum(Threshold, Stopping_flux * DirtyRatio)

        # run subminor loop
        k = 0
        while np.abs(ConvMaxDirty) > Threshold and k < self.ScaleMachine.NSubMinorIter:
            # get JonesNorm
            JN = JonesNorm[:, 0, xscale, yscale]

            # set facet location
            self.PSFServer.setLocation(xscale, yscale)

            # set PSF and gain
            self.set_ConvPSF(self.PSFServer.iFacet, iScale)

            # keep track of components in the facet
            Component_list.setdefault(self.CurrentFacet, [])  # in case the facet has changed and uninitialised
            if [xscale, yscale] not in Component_list[self.CurrentFacet]:
                Component_list[self.CurrentFacet].append([xscale, yscale])

            # JonesNorm is corrected for in FreqMachine so we just need to pass in the apparent
            Fpol = np.zeros([self.Nchan, 1, 1, 1], dtype=np.float32)
            Fpol[:, 0, 0, 0] = ConvDirtyCube[:, 0, xscale, yscale].copy()

            # correct for frequency response of PSF when convolving with scale function
            Fpol *= self.PSFFreqNormFactors

            # Fit frequency axis to get coeffs (coeffs correspond to intrinsic flux)
            self.Coeffs = self.FreqMachine.Fit(Fpol[:, 0, 0, 0], JN, ConvMaxDirty)

            # Overwrite with polynoimial fit (Fpol is apparent flux)
            Fpol[:, 0, 0, 0] = self.FreqMachine.Eval(self.Coeffs)

            self.AppendComponentToDictStacked((xscale, yscale), self.Coeffs / self.FpolNormFactor, sigma, self.CurrentGain)

            # keep track of apparent model array (this is for subtraction in the upper minor loop)
            # Note apparent since we convolve with a normalised PSF
            ScaleModel[:, 0, xscale, yscale] += self.CurrentGain * Fpol[:, 0, 0, 0] / self.FpolNormFactor

            # Restore ConvPSF frequency response and subtract component from residual
            Fpol /= self.PSFFreqNormFactors
            ConvDirtyCube = self.SubStep((xscale, yscale), self.ConvPSF * Fpol * self.CurrentGain,
                                         ConvDirtyCube)

            # get the weighted mean over freq axis
            CurrentDirty[...] = np.sum(ConvDirtyCube * W.reshape((W.size, 1, 1, 1)), axis=0)[None, :, :, :]

            # find the peak
            #PeakMap = np.ascontiguousarray(CurrentDirty*self.ScaleMachine.ScaleMaskArray[sigma])
            xscale, yscale, ConvMaxDirty = NpParallel.A_whereMax(CurrentDirty, NCPU=self.NCPU, DoAbs=self.DoAbs,
                                                                 Mask=None)

            # Update counters TODO - should add subminor cycle count to minor cycle count
            k += 1
            self.n_sub_minor_iter += 1

        # print "Max at end = ", ConvMaxDirty, iScale
        # report if max sub-iterations exceeded
        if k >= self.ScaleMachine.NSubMinorIter:
            print>>log, "Maximum subiterations reached. "
            # print "Max delta component = ", np.abs(ScaleModel).max()
            # print "Current Scale, facet and gain = ", self.CurrentScale, self.CurrentFacet, self.CurrentGain

        return ScaleModel, Component_list, sigma



###################### Dark magic below this line ###################################
    def PutBackSubsComps(self):
        # if self.GD["Data"]["RestoreDico"] is None: return

        SolsFile = self.GD["DDESolutions"]["DDSols"]
        if not (".npz" in SolsFile):
            Method = SolsFile
            ThisMSName = reformat.reformat(os.path.abspath(self.GD["Data"]["MS"]), LastSlash=False)
            SolsFile = "%s/killMS.%s.sols.npz" % (ThisMSName, Method)
        DicoSolsFile = np.load(SolsFile)
        SourceCat = DicoSolsFile["SourceCatSub"]
        SourceCat = SourceCat.view(np.recarray)
        # RestoreDico=self.GD["Data"]["RestoreDico"]
        RestoreDico = DicoSolsFile["ModelName"][()][0:-4] + ".DicoModel"

        print>> log, "Adding previously subtracted components"
        ModelMachine0 = ClassModelMachine(self.GD)

        ModelMachine0.FromFile(RestoreDico)

        _, _, nx0, ny0 = ModelMachine0.DicoSMStacked["ModelShape"]

        _, _, nx1, ny1 = self.ModelShape
        dx = nx1 - nx0

        for iSource in range(SourceCat.shape[0]):
            x0 = SourceCat.X[iSource]
            y0 = SourceCat.Y[iSource]

            x1 = x0 + dx
            y1 = y0 + dx

            if not ((x1, y1) in self.DicoSMStacked["Comp"].keys()):
                self.DicoSMStacked["Comp"][(x1, y1)] = ModelMachine0.DicoSMStacked["Comp"][(x0, y0)]
            else:
                self.DicoSMStacked["Comp"][(x1, y1)] += ModelMachine0.DicoSMStacked["Comp"][(x0, y0)]
