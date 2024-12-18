from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import numpy as np
from scipy.integrate import cumtrapz
import numexpr
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.Other import ClassTimeIt
from pyrap.images import image
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Imager import ClassGainMachine  # Currently required by model machine but fixed to static mode
from DDFacet.Imager.ClassMaskMachine import ClassMaskMachine
from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric
from DDFacet.Imager import ClassFrequencyMachine
from . import ClassScaleMachine
import numba

# from DDFacet.Other.AsyncProcessPool import APP
# from DDFacet.ToolsDir.ModFFTW import FFTW_Scale_Manager  # usage just to register job handlers but has no effect atm

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
    npixpsf_x,npixpsf_y=npixpsf
    
    halfnpixpsf_x = npixpsf_x//2
    halfnpixpsf_y = npixpsf_y//2
    p = Ip[pq]
    q = Iq[pq]
    # loop over active indices
    for i, ipq in enumerate(zip(Ip, Iq)):
        ip, iq = ipq
        # check that there is an overlap with psf
        delp = p - ip
        delq = q - iq
        if abs(delp) <= halfnpixpsf_x and abs(delq) <= halfnpixpsf_y:
            pp = halfnpixpsf_x - delp
            qq = halfnpixpsf_y - delq
            A[i] -= sol * psf[pp, qq]
    return A

class ClassWSCMS_MinorLoop():

    def __init__(self,GD):
        self.GD=GD
        self.NCall_to_do_scale_convolve=0
        
    def setPSFServer(self, PSFServer):
        self.PSFServer = PSFServer

        _, _, self.Npix_x, self.Npix_y = self.PSFServer.ImageShape
        self.NpixPadded_x = int(np.ceil(self.GD["Facets"]["Padding"] * self.Npix_x))
        self.NpixPadded_y = int(np.ceil(self.GD["Facets"]["Padding"] * self.Npix_y))
        
        # make sure it is odd numbered
        if self.NpixPadded_x % 2 == 0:
            self.NpixPadded_x += 1
        self.Npad_x = (self.NpixPadded_x - self.Npix_x) // 2
        
        if self.NpixPadded_y % 2 == 0:
            self.NpixPadded_y += 1
        self.Npad_y = (self.NpixPadded_y - self.Npix_y) // 2

    def setModelMachine(self,ModelMachine):
        self.ModelMachine=ModelMachine
        self.FreqMachine=self.ModelMachine.FreqMachine
        
    def setScaleMachine(self, PSFServer, NCPU=None, MaskArray=None, cachepath=None, MaxBaseline=None):
        # if self.GD["WSCMS"]["MultiScale"]:
        if NCPU is None:
            self.NCPU = self.GD['Parallel'][NCPU]
            if self.NCPU == 0:
                import multiprocessing

                self.NCPU = multiprocessing.cpu_count()
        else:
             self.NCPU = NCPU

        self.MaskArray=MaskArray.copy()
        self.DoAbs = self.GD["Deconv"]["AllowNegative"]
        self.ScaleMachine = ClassScaleMachine.ClassScaleMachine(GD=self.GD, NCPU=self.NCPU,
                                                                MaskArray=self.MaskArray)

        self.ScaleMachine.Init(PSFServer, self.FreqMachine, cachepath=cachepath, MaxBaseline=MaxBaseline)
        self.NpixPSF_x,self.NpixPSF_y = self.ScaleMachine.NpixPSF
        self.halfNpixPSF_x = self.NpixPSF_x//2
        self.halfNpixPSF_y = self.NpixPSF_y//2
        self.Nscales = self.ScaleMachine.Nscales
        # Initialise CurrentScale variable
        self.CurrentScale = 999999
        # Initialise current facet variable
        self.CurrentFacet = 999999


        self.ModelMachine.DicoSMStacked["Scale_Info"] = {}
        for iScale, sigma in enumerate(self.ScaleMachine.sigmas):
            if iScale not in self.ModelMachine.DicoSMStacked["Scale_Info"].keys():
                self.ModelMachine.DicoSMStacked["Scale_Info"][iScale] = {}
            self.ModelMachine.DicoSMStacked["Scale_Info"][iScale]["sigma"] = self.ScaleMachine.sigmas[iScale]
            self.ModelMachine.DicoSMStacked["Scale_Info"][iScale]["kernel"] = self.ScaleMachine.kernels[iScale]
            self.ModelMachine.DicoSMStacked["Scale_Info"][iScale]["extent"] = self.ScaleMachine.extents[iScale]
            
    def SubStep(self, xc, yc, LocalSM, Residual):
        """
        Sub-minor loop subtraction
        """
        N0x,N0y = Residual.shape[-2:]
        N1x,N1y = LocalSM.shape[-2:]

        # Get overlap indices where psf should be subtracted
        Aedge, Bedge = GiveEdgesDissymetric(xc, yc, N0x, N0y, N1x // 2, N1y // 2, N1x,N1y)

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
                    self.ScaleMachine.CheckScaleMasks(self.ModelMachine.DicoSMStacked)
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
                    savestr = self.GD["Output"]["Images"]+self.GD["Output"]["Also"]
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
        
        print(xscale, yscale, AbsConvMaxDirty, iScale)
        self.NCall_to_do_scale_convolve+=1
        if self.NCall_to_do_scale_convolve==2: stop

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
        print("GFLDFLJ",AbsConvMaxDirty, Threshold, k, self.ScaleMachine.NSubMinorIter)
        while AbsConvMaxDirty > Threshold and k < self.ScaleMachine.NSubMinorIter:
            print("   ************")
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
            print("Fpol,self.Coeffs",Fpol,self.Coeffs)
            
            # Overwrite with polynoimial fit (Fpol is apparent flux)
            Fpol = self.FreqMachine.Eval(self.Coeffs)

            # append component to dico
            self.ModelMachine.AppendComponentToDictStacked((xscale, yscale), self.Coeffs, self.CurrentScale, self.CurrentGain)
            # Subtract fitted component from residual cube
            print("self.ConvPSF",self.ConvPSF.shape,self.ConvPSF.reshape((3,-1)).max(axis=1))
            self.SubStep(xscale, yscale, self.ConvPSF * Fpol[:, None, None, None] * self.CurrentGain, Dirty.view())
            # subtract component from convolved dirty image
            A = substep(A, self.Conv2PSFmean[0, 0], float(ConvMaxDirty * self.CurrentGain), Ip, Iq, pq, (self.NpixPSF_x,self.NpixPSF_y))
            
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
            print("GFLDFLJ AbsConvMaxDirty,ConvMaxDirty, Threshold, k, self.ScaleMachine.NSubMinorIter,Fpol,pq",AbsConvMaxDirty,ConvMaxDirty, Threshold, k, self.ScaleMachine.NSubMinorIter,Fpol,pq)
            stop

        return k, iScale
