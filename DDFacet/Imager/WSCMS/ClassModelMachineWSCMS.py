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
import numba
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("ClassModelMachine")
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Other import MyPickle
from DDFacet.Other import reformat
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.Imager import ClassModelMachine as ClassModelMachinebase
from DDFacet.Imager import ClassFrequencyMachine, ClassScaleMachine
import os

@numba.jit(nopython=True, nogil=True, cache=True)
def substep(A, psf, sol, Ip, Iq, pq, npixpsf):
    """
    Subtract psf from the set A where they overlap
    :param A: the active set
    :param psf: the psf (not necessarily the size of the image)
    :param Ip, Iq: indices of active set relative to image
    :param pq: the location of a component relative to the active set
    :return:
    """
    halfnpixpsf = npixpsf//2
    p = Ip[pq]
    q = Iq[pq]
    # loop over active indices
    for i, ipq in enumerate(zip(Ip, Iq)):
        ip, iq = ipq
        # check that there is an overlap with psf
        delp = p - ip
        delq = q - iq
        if abs(delp) <= halfnpixpsf and abs(delq) <= halfnpixpsf:
            pp = halfnpixpsf - delp
            qq = halfnpixpsf - delq
            A[i] -= sol * psf[pp, qq]
    return A

class ClassModelMachine(ClassModelMachinebase.ClassModelMachine):
    def __init__(self,*args,**kwargs):
        ClassModelMachinebase.ClassModelMachine.__init__(self, *args, **kwargs)
        self.DicoSMStacked={}
        self.DicoSMStacked["Type"]="WSCMS"

    def setRefFreq(self, RefFreq, Force=False):
        if self.RefFreq is not None and not Force:
            print(ModColor.Str("Reference frequency already set to %f MHz" % (self.RefFreq/1e6)), file=log)
            return

        self.RefFreq = RefFreq
        self.DicoSMStacked["RefFreq"] = RefFreq

    def setPSFServer(self, PSFServer):
        self.PSFServer = PSFServer

        _, _, self.Npix, _ = self.PSFServer.ImageShape
        self.NpixPadded = int(np.ceil(self.GD["Facets"]["Padding"] * self.Npix))
        # make sure it is odd numbered
        if self.NpixPadded % 2 == 0:
            self.NpixPadded += 1
        self.Npad = (self.NpixPadded - self.Npix) // 2

    def setFreqMachine(self, GridFreqs, DegridFreqs, weights=None, PSFServer=None):
        self.PSFServer = PSFServer
        # Initiaise the Frequency Machine
        self.DegridFreqs = DegridFreqs
        self.GridFreqs = GridFreqs
        self.FreqMachine = ClassFrequencyMachine.ClassFrequencyMachine(GridFreqs, DegridFreqs,
                                                                       self.DicoSMStacked["RefFreq"], self.GD,
                                                                       weights=weights, PSFServer=self.PSFServer)
        self.FreqMachine.set_Method()

        if (self.GD["Freq"]["NBand"] > 1):
            self.Coeffs = np.zeros(self.GD["WSCMS"]["NumFreqBasisFuncs"])
        else:
            self.Coeffs = np.zeros([1])

        self.Nchan = self.FreqMachine.nchan
        self.Npol = 1

        # self.DicoSMStacked["Eval_Degrid"] = self.FreqMachine.Eval_Degrid

    def setFacetMachine(self, FacetMachine, BaseName):
        self.FacetMachine = FacetMachine
        self.BaseName = BaseName

    def setScaleMachine(self, PSFServer, NCPU=None, MaskArray=None, cachepath=None, MaxBaseline=None):
        # if self.GD["WSCMS"]["MultiScale"]:
        if NCPU is None:
            self.NCPU = self.GD['Parallel'][NCPU]
            if self.NCPU == 0:
                import multiprocessing

                self.NCPU = multiprocessing.cpu_count()
        else:
             self.NCPU = NCPU
        self.DoAbs = self.GD["Deconv"]["AllowNegative"]
        self.ScaleMachine = ClassScaleMachine.ClassScaleMachine(GD=self.GD, NCPU=self.NCPU, MaskArray=MaskArray)
        self.ScaleMachine.Init(PSFServer, self.FreqMachine, cachepath=cachepath, MaxBaseline=MaxBaseline)
        self.NpixPSF = self.ScaleMachine.NpixPSF
        self.halfNpixPSF = self.NpixPSF//2
        self.Nscales = self.ScaleMachine.Nscales
        # Initialise CurrentScale variable
        self.CurrentScale = 999999
        # Initialise current facet variable
        self.CurrentFacet = 999999


        self.DicoSMStacked["Scale_Info"] = {}
        for iScale, sigma in enumerate(self.ScaleMachine.sigmas):
            if iScale not in self.DicoSMStacked["Scale_Info"].keys():
                self.DicoSMStacked["Scale_Info"][iScale] = {}
            self.DicoSMStacked["Scale_Info"][iScale]["sigma"] = self.ScaleMachine.sigmas[iScale]
            self.DicoSMStacked["Scale_Info"][iScale]["kernel"] = self.ScaleMachine.kernels[iScale]
            self.DicoSMStacked["Scale_Info"][iScale]["extent"] = self.ScaleMachine.extents[iScale]

    def ToFile(self, FileName, DicoIn=None):
        print("Saving dico model to %s" % FileName, file=log)
        if DicoIn is None:
            D = self.DicoSMStacked
        else:
            D = DicoIn

        if self.GD is None:
            print("Warning - you are haven't initialised GD before writing to the DicoModel")
        D["GD"] = self.GD
        D["Type"] = "WSCMS"
        D["ModelShape"] = self.ModelShape
        MyPickle.Save(D, FileName)

    def FromFile(self, FileName):
        print("Reading dico model from %s" % FileName, file=log)
        self.DicoSMStacked = MyPickle.Load(FileName)
        self.FromDico(self.DicoSMStacked)

    def FromDico(self, DicoSMStacked):
        self.DicoSMStacked = DicoSMStacked
        self.RefFreq = self.DicoSMStacked["RefFreq"]
        # self.ListScales = self.DicoSMStacked["ListScales"]
        self.ModelShape = self.DicoSMStacked["ModelShape"]

    def setModelShape(self, ModelShape):
        self.ModelShape = ModelShape
        self.Npix = self.ModelShape[-1]

    def AppendComponentToDictStacked(self, key, Sols, iScale, Gain):
        """
        Adds component to model dictionary at a scale specified by Scale.
        The dictionary corresponding to each scale is keyed on pixel values (l,m location tupple).
        Each model component is therefore represented parametrically by a pixel value a scale and a set of coefficients
        describing the spectral axis.
        Currently only Stokes I is supported.
        Args:
            key: the (l,m) centre of the component in pixels
            Sols: Nd array of coeffs with length equal to the number of basis functions representing the component.
            iScale: the scale index
            Gain: clean loop gain

        Added component list to dictionary for particular scale. This dictionary is stored in
        self.DicoSMStacked["Comp"][iScale] and has keys:
            "SolsArray": solutions ndArray with shape [#basis_functions,#stokes_terms]
            "NumComps": scalar keeps tracl of the number of components found at a particular scale
        """
        DicoComp = self.DicoSMStacked.setdefault("Comp", {})

        if iScale not in DicoComp.keys():
            DicoComp[iScale] = {}
            DicoComp[iScale]["NumComps"] = np.zeros(1, np.int16)  # keeps track of number of components at this scale

        if key not in DicoComp[iScale].keys():
            DicoComp[iScale][key] = {}
            DicoComp[iScale][key]["SolsArray"] = np.zeros(Sols.size, np.float32)

        DicoComp[iScale]["NumComps"] += 1
        DicoComp[iScale][key]["SolsArray"] += Sols.ravel() * Gain

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

        for iScale in DicoComp.keys():
            ScaleModel = np.zeros((nchan, npol, nx, ny), dtype=np.float32)
            # get scale kernel
            if self.GD["WSCMS"]["MultiScale"]:
                sigma = self.DicoSMStacked["Scale_Info"][iScale]["sigma"]
                kernel = self.DicoSMStacked["Scale_Info"][iScale]["kernel"]
                extent = self.DicoSMStacked["Scale_Info"][iScale]["extent"]

            for key in DicoComp[iScale].keys():
                if key != "NumComps":  # LB - dirty dirty hack needs to die!!!
                    Sol = DicoComp[iScale][key]["SolsArray"]
                    # TODO - try soft thresholding components
                    x, y = key
                    try:  # LB - Should we drop support for anything other than polynomials maybe?
                        interp = self.FreqMachine.Eval_Degrid(Sol, FreqIn)
                    except:
                        interp = np.polyval(Sol[::-1], FreqIn/RefFreq)

                    if interp is None:
                        raise RuntimeError("Could not interpolate model onto degridding bands. Inspect your data, check "
                                           "'WSCMS-NumFreqBasisFuncs' or if you think this is a bug report it.")

                    if self.GD["WSCMS"]["MultiScale"] and iScale != 0:
                        Aedge, Bedge = GiveEdges(x, y, nx, extent // 2, extent // 2, extent)

                        x0d, x1d, y0d, y1d = Aedge
                        x0p, x1p, y0p, y1p = Bedge

                        out = np.atleast_1d(interp)[:, None, None, None] * kernel
                        ScaleModel[:, :, x0d:x1d, y0d:y1d] += out[:, :, x0p:x1p, y0p:y1p]
                    else:
                        ScaleModel[:, 0, x, y] += interp

            ModelImage += ScaleModel
        return ModelImage

    def GiveSpectralIndexMap(self, GaussPars=[(1, 1, 0)], ResidCube=None,
                             GiveComponents=False, ChannelWeights=None):

        # convert to radians
        ex, ey, pa = GaussPars
        ex *= np.pi/180/np.sqrt(2)/2
        ey *= np.pi/180/np.sqrt(2)/2
        epar = (ex + ey)/2.0
        pa = 0.0

        # get in terms of number of cells
        CellSizeRad = self.GD['Image']['Cell'] * np.pi / 648000

        # get Gaussian kernel
        GaussKern = ModFFTW.GiveGauss(self.Npix, CellSizeRad=CellSizeRad, GaussPars=(epar, epar, pa), parallel=False)

        # take FT
        Fs = np.fft.fftshift
        iFs = np.fft.ifftshift

        import pyfftw
        nthreads = int(self.GD['Parallel']['NCPU'])
        if not nthreads:
            import multiprocessing
            nthreads = multiprocessing.cpu_count()
        else:
            from multiprocessing.pool import ThreadPool
            import dask

            dask.config.set(pool=ThreadPool(nthreads))

        FFT = lambda x: pyfftw.interfaces.numpy_fft.fft2(x, axes=(-2, -1), planner_effort='FFTW_ESTIMATE', threads=nthreads) #, norm='ortho')
        iFFT = lambda x: pyfftw.interfaces.numpy_fft.ifft2(x, axes=(-2, -1), planner_effort='FFTW_ESTIMATE', threads=nthreads) #, norm='ortho')

        # evaluate model
        ModelImage = self.GiveModelImage(self.GridFreqs)

        # pad GausKern and take FT
        GaussKern = np.pad(GaussKern, self.Npad, mode='constant')
        FTshape, _ = GaussKern.shape
        from scipy import fftpack as FT
        #GaussKernhat = FT.fft2(iFs(GaussKern))
        GaussKernhat = FFT(iFs(GaussKern))

        # pad and FT of ModelImage
        ModelImagehat = np.zeros((self.Nchan, FTshape, FTshape), dtype=np.complex128)
        ConvModelImage = np.zeros((self.Nchan, self.Npix, self.Npix), dtype=np.float64)
        I = slice(self.Npad, -self.Npad)
        for i in range(self.Nchan):
            tmp_array = np.pad(ModelImage[i, 0], self.Npad, mode='constant')
            # ModelImagehat[i] = FT.fft2(iFs(tmp_array)) * GaussKernhat
            ModelImagehat[i] = FFT(iFs(tmp_array)) * GaussKernhat
            # ConvModelImage[i] = Fs(FT.ifft2(ModelImagehat[i]))[I, I].real
            ConvModelImage[i] = Fs(iFFT(ModelImagehat[i]))[I, I].real

        if ResidCube is not None:
            #ConvModelImage += ResidCube.squeeze()
            RMS = np.std(ResidCube.flatten())
            Threshold = self.GD["SPIMaps"]["AlphaThreshold"] * RMS
        else:
            AbsModel = np.abs(ModelImage).squeeze()
            MinAbsImage = np.amin(AbsModel, axis=0)
            RMS = np.min(np.abs(MinAbsImage.flatten())) # base cutoff on smallest value in model
            Threshold = self.GD["SPIMaps"]["AlphaThreshold"] * RMS

        # get minimum along any freq axis
        MinImage = np.amin(ConvModelImage, axis=0)
        MaskIndices = np.argwhere(MinImage > Threshold)
        FitCube = ConvModelImage[:, MaskIndices[:, 0], MaskIndices[:, 1]]


        if ChannelWeights is None:
            weights = np.ones(self.Nchan, dtype=np.float32)
        else:
            weights = ChannelWeights.astype(np.float32)
            if ChannelWeights.size != self.Nchan:
                import warnings
                warnings.warn("The provided channel weights are of incorrect length. Ignoring weights.", RuntimeWarning)
                weights = np.ones(self.Nchan, dtype=np.float32)

        try:
            import traceback
            from africanus.model.spi.dask import fit_spi_components
            import dask.array as da
            _, ncomps = FitCube.shape
            FitCubeDask = da.from_array(FitCube.T.astype(np.float64),
                                        chunks=(np.maximum(100, ncomps//nthreads), self.Nchan))
            weightsDask = da.from_array(weights.astype(np.float64), chunks=(self.Nchan))
            freqsDask = da.from_array(np.array(self.GridFreqs).astype(np.float64), chunks=(self.Nchan))

            alpha, varalpha, Iref, varIref = fit_spi_components(FitCubeDask, weightsDask,
                                                                freqsDask, self.RefFreq).compute()
        except Exception as e:
            raise(e)

        _, _, nx, ny = ModelImage.shape
        alphamap = np.zeros([nx, ny])
        Irefmap = np.zeros([nx, ny])
        alphastdmap = np.zeros([nx, ny])
        Irefstdmap = np.zeros([nx, ny])

        alphamap[MaskIndices[:, 0], MaskIndices[:, 1]] = alpha
        Irefmap[MaskIndices[:, 0], MaskIndices[:, 1]] = Iref
        alphastdmap[MaskIndices[:, 0], MaskIndices[:, 1]] = np.sqrt(varalpha)
        Irefstdmap[MaskIndices[:, 0], MaskIndices[:, 1]] = np.sqrt(varIref)

        if GiveComponents:
            return alphamap[None, None], alphastdmap[None, None], alpha
        else:
            return alphamap[None, None], alphastdmap[None, None]

    def SubStep(self, xc, yc, LocalSM, Residual):
        """
        Sub-minor loop subtraction
        """
        N0 = Residual.shape[-1]
        N1 = LocalSM.shape[-1]

        # Get overlap indices where psf should be subtracted
        Aedge, Bedge = GiveEdges(xc, yc, N0, N1 // 2, N1 // 2, N1)

        x0d, x1d, y0d, y1d = Aedge
        x0p, x1p, y0p, y1p = Bedge

        # Subtract from each channel/band
        Residual[:, :, x0d:x1d, y0d:y1d] -= LocalSM[:, :, x0p:x1p, y0p:y1p]

    def set_ConvPSF(self, iFacet, iScale):
        # we only need to compute the PSF if Facet or Scale has changed
        # note will always be set initially since comparison to 999999 will fail
        if iFacet != self.CurrentFacet or self.CurrentScale != iScale:
            key = 'S' + str(iScale) + 'F' + str(iFacet)
            # update facet (NB - ensure PSFServer has been updated before we get here)
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
                self.Conv2PSFmean = PSFmean

                # delta scale is not cleaned with the ConvPSF so these should all be unity
                self.FpolNormFactor = 1.0
            else:
                self.ConvPSF = self.ScaleMachine.ConvPSFs[key]
                self.Conv2PSFmean = self.ScaleMachine.Conv2PSFmean[key]

                # This normalisation for Fpol is required so that we don't see jumps between minor cycles.
                # Basically, since the PSF is normalised by this factor the components also need to be normalised
                # by the same factor for the subtraction in the sub-minor cycle to be the same as the subtraction
                # in the minor and major cycles.
                self.FpolNormFactor = self.ScaleMachine.Conv2PSFNormFactor


    def do_minor_loop(self, Dirty, meanDirty, JonesNorm, WeightsChansImages, MaxDirty, Stopping_flux=None, RMS=None):
        """
        Runs the sub-minor loop at a specific scale
        :param Dirty: dirty cube
        :param meanDirty: mean dirty
        :param JonesNorm: "average" DDE map
        :param WeightsChansImages: The sum of the weights in each channel normalised to sum to one
        :param MaxDirty: maximum of mean dirty computed in last minor loop
        :param Stopping_flux: stopping flux for unconvolved mean image
        :param RMS: RMS of mean dirty computed in last minor loop (for auto-masking)
        :return: number of iterations k, the dominant scale

        Components are only searched for in the active set A defined as all pixels in s scale convolved dirty
        above GD[WSCMS][SubMinorPeakFactor] * AbsConvMaxDirty. The PSF for A is the PSF twice convolved with the
        dominant scale kernel. At the same time, once a component has been found in A, it is subtracted from the
        dirty cube using the PSF once convolved with the scale kernel. The actual MeanDirty image is only updated
        once we drop back into the minor loop by computing the weighted sum over channels.
        """
        # Select scale mask and check if auto-masking has kicked in
        if self.GD["WSCMS"]["AutoMask"]:
            if self.GD["WSCMS"]["AutoMaskThreshold"] is not None:
                MaskThreshold = self.GD["WSCMS"]["AutoMaskThreshold"]
            else:
                MaskThreshold = self.GD["WSCMS"]["AutoMaskRMSFactor"] * RMS
            if MaxDirty <= MaskThreshold:
                # This should only happen once
                if self.ScaleMachine.AppendMaskComponents:
                    print("Starting auto-masking at a threshold of %f" % MaskThreshold, file=log)
                    # we shan't be updating the mask any longer
                    self.ScaleMachine.AppendMaskComponents = False
                    # check that all components in the dictionary are in trhe scale masks
                    self.ScaleMachine.CheckScaleMasks(self.DicoSMStacked)
                    # bit flip first if no external mask (since initialised to all zeros)
                    if not self.ScaleMachine.MaskArray.any():
                        self.ScaleMachine.MaskArray |= True

                        # get global mask in which we check convergence criteria
                        for i in range(self.ScaleMachine.Nscales):
                            ScaleMask = self.ScaleMachine.ScaleMaskArray[str(i)]
                            self.ScaleMachine.MaskArray &= ScaleMask
                            # retire scale if there are no components in the mask
                            if ScaleMask.all():
                                self.ScaleMachine.forbidden_scales.append(i)
                                self.ScaleMachine.retired_scales.append(i)
                                print("Retired scale %i permanently because auto-masking "
                                      "kicked in and mask is empty thus far"%i, file=log)
                    # dilate all masks
                    self.ScaleMachine.dilate_scale_masks()

                    # save all masks
                    savestr = self.GD["Output"]["Images"]
                    if savestr.lower() == 'all' or 'k' in list(savestr):
                        self.FacetMachine.ToCasaImage(np.float32(self.ScaleMachine.MaskArray),
                                                      ImageName="%s.GlobalMask" % (self.BaseName),
                                                      Fits=True)
                        for i in range(self.ScaleMachine.Nscales):
                            ScaleMask = self.ScaleMachine.ScaleMaskArray[str(i)]
                            self.FacetMachine.ToCasaImage(np.float32(ScaleMask),
                                                          ImageName="%s.ScaleMask%i" % (self.BaseName, i),
                                                          Fits=True)

        # determine most relevant scale (note AbsConvMaxDirty given as absolute value)
        xscale, yscale, AbsConvMaxDirty, CurrentDirty, iScale, CurrentMask = self.ScaleMachine.do_scale_convolve(meanDirty)

        # set PSF at current location
        self.PSFServer.setLocation(xscale, yscale)

        # set convolved PSFs for scale and facet if either has changed
        # the once convolved PSF cubes used to subtract from the dirty cube are stored in self.ConvPSF
        # the twice convolved PSF used to subtract from the mean convolved dirty is held in self.Conv2PSFmean
        self.set_ConvPSF(self.PSFServer.iFacet, iScale)

        # set stopping threshold.
        Threshold = self.ScaleMachine.PeakFactor * AbsConvMaxDirty
        # DirtyRatio = AbsConvMaxDirty / MaxDirty  # should be 1 for zero scale
        # Threshold = np.maximum(Threshold, Stopping_flux * DirtyRatio)

        # get the set A (in which we do peak finding)
        # assumes mask is 0 where we should do peak finding (same as Cyril's convention)
        absdirty = np.where(CurrentMask.squeeze(), 0.0, np.abs(CurrentDirty.squeeze()))
        I = np.argwhere(absdirty > Threshold)
        Ip = I[:, 0]
        Iq = I[:, 1]
        A = CurrentDirty[0, 0, Ip, Iq]
        absA = np.abs(A)
        try:
            pq = int(np.argwhere(absA == AbsConvMaxDirty))
        except:
            raise RuntimeError("Somehow Threshold > CurrentDirty.max()? This is a bug!")
        ConvMaxDirty = A[pq]


        # run subminor loop
        k = 0
        while AbsConvMaxDirty > Threshold and k < self.ScaleMachine.NSubMinorIter:
            # get JonesNorm
            JN = JonesNorm[:, 0, xscale, yscale]

            # set facet location
            self.PSFServer.setLocation(xscale, yscale)

            # set PSF and gain
            self.set_ConvPSF(self.PSFServer.iFacet, iScale)

            # JonesNorm is corrected for in FreqMachine so we just need to pass in the apparent
            Fpol = Dirty[:, 0, xscale, yscale].copy()

            # Fit frequency axis to get coeffs (coeffs correspond to intrinsic flux)
            self.Coeffs = self.FreqMachine.Fit(Fpol, JN, WeightsChansImages.squeeze())

            # Overwrite with polynoimial fit (Fpol is apparent flux)
            Fpol = self.FreqMachine.Eval(self.Coeffs)

            # append component to dico
            self.AppendComponentToDictStacked((xscale, yscale), self.Coeffs, self.CurrentScale, self.CurrentGain)
            # Subtract fitted component from residual cube
            self.SubStep(xscale, yscale, self.ConvPSF * Fpol[:, None, None, None] * self.CurrentGain, Dirty.view())
            # subtract component from convolved dirty image
            A = substep(A, self.Conv2PSFmean[0, 0], float(ConvMaxDirty * self.CurrentGain), Ip, Iq, pq, self.NpixPSF)

            # update scale dependent mask
            if self.ScaleMachine.AppendMaskComponents:
                ScaleMask = self.ScaleMachine.ScaleMaskArray[str(iScale)].view()
                ScaleMask[0, 0, xscale, yscale] = 0
            else:
                # If auto-masking has kicked in we keep track of where new components are being added
                # so we can check convergence in the minor cycle
                self.ScaleMachine.MaskArray[0, 0, xscale, yscale] = 0

            # find new peak
            absA = np.abs(A)
            AbsConvMaxDirty = absA.max()
            # TODO - How does this happen? It seems sometimes we have two components with the same max flux
            try:
                pq = int(np.argwhere(absA == AbsConvMaxDirty))
            except:
                pq = int(np.argwhere(absA == AbsConvMaxDirty)[0])
            ConvMaxDirty = A[pq]

            # get location of component in residual frame
            xscale = Ip[pq]
            yscale = Iq[pq]

            # Update counters
            k += 1

        return k, iScale

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

        print("Adding previously subtracted components", file=log)
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
