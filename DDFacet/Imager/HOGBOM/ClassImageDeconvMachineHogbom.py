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
from DDFacet.Imager.HOGBOM import ClassModelMachineHogbom

from DDFacet.compatibility import range

"""
This minimal implementation of standard Hogbom CLEAN algorithm should serve
as a minimal reference interface of how to incorporate new deconvolution
algorithms into DDFacet.
"""

import numpy as np
import numexpr
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.Other import ClassTimeIt
from pyrap.images import image
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Imager import ClassGainMachine # Currently required by model machine but fixed to static mode
from DDFacet.ToolsDir import GiveEdges


class ClassImageDeconvMachine():
    """
    Currently constructor inputs match those in MSMF, should figure out which are truly generic and put the rest in
    parset's MinorCycleConfig option.
    These methods may be called from ClassDeconvMachine
        Init(**kwargs) - contains minor cycle specific initialisations which are only used once
            Input: currently kwargs are minor cycle specific and should be set from ClassDeconvMachine but a
                     ideally a generic interface has these set in the parset somehow.
        Deconvolve() - does joint deconvolution over all the channels/bands.
            Output: return_code - "MaxIter"????
                    continue - whether to continue the deconvolution
                    updated - whether the model has been updated
        GiveModelImage(freq) - returns current model at freq
            Input: freq - tuple of frequencies at which to return the model
            Output: Mod - the current model at freq
        Update(DicoDirty,**kwargs) - updates to minor cycle at the end of each major cycle
            Input:  DicoDirty - updated image dict at start of each major cycle
                    Use kwargs to pass any other minor cycle specific options
        ToFile(fname) - saves dico model to file
            Input: fname - the name of the file to write the dico image to
        FromFile(fname) - reads model dict from file
            Input: fname - the name of the file to write the dico image to
    """
    def __init__(self,Gain=0.3,
                 MaxMinorIter=100,NCPU=6,
                 CycleFactor=2.5,FluxThreshold=None,RMSFactor=3,PeakFactor=0,
                 GD=None,SearchMaxAbs=1,CleanMaskImage=None,ImagePolDescriptor=["I"],ModelMachine=None,
                 **kw    # absorb any unknown keywords arguments into this
                 ):
        self.SearchMaxAbs = SearchMaxAbs
        self.ModelImage=None
        self.MaxMinorIter=MaxMinorIter
        self.NCPU=NCPU
        self.MaskArray = None
        self.GD=GD
        self.MultiFreqMode = (self.GD["Freq"]["NBand"] > 1)
        self.NFreqBand = self.GD["Freq"]["NBand"]
        if self.GD["Hogbom"]["LinearPeakfinding"].lower() == "joint":
            self.ComplexClean = True
        elif self.GD["Hogbom"]["LinearPeakfinding"].lower() == "separate":
            self.ComplexClean = False
        else:
            raise ValueError("Hogbom-LinearPeakfinding must be Joint or Separate. You specified {}".format(
                self.GD["Hogbom"]["LinearPeakfinding"]))
        self.FluxThreshold = FluxThreshold
        self.CycleFactor = CycleFactor
        self.RMSFactor = RMSFactor
        self.PeakFactor = PeakFactor
        self.GainMachine = ClassGainMachine.get_instance()
        
        # FGAO: added back the polarization related part:
        self.PolarizationDescriptor = ImagePolDescriptor
        self.PolarizationCleanTasks = []
        if "I" in self.PolarizationDescriptor:
            self.PolarizationCleanTasks.append("I")
            print("Found Stokes I. I will be CLEANed independently", file=log)
        else:
            print("Stokes I not available. Not performing intensity CLEANing.",file=log)
        if set(["Q","U"]) < set(self.PolarizationDescriptor):
            complex_clean = self.ComplexClean
            if complex_clean:
                self.PolarizationCleanTasks.append("Q+iU") #Luke Pratley's complex polarization CLEAN
                print("Will perform joint linear (Pratley-Johnston-Hollitt) CLEAN.",file=log)
            else:
                self.PolarizationCleanTasks.append("Q") #back to the normal way
                self.PolarizationCleanTasks.append("U")
                print("Will CLEAN Q and U separately.",file=log)
        elif set(["Q"]) < set(self.PolarizationDescriptor):
            self.PolarizationCleanTasks.append("Q")
            print("Will CLEAN Q without cleaning U as per user request.",file=log)
        elif set(["U"]) < set(self.PolarizationDescriptor):
            self.PolarizationCleanTasks.append("U")
            print("Will CLEAN U without cleaning Q as per user request.",file=log)
        else:
            print("Neither Stokes Q nor U synthesized. Will not CLEAN these.",file=log)

        if "V" in self.PolarizationDescriptor:
            self.PolarizationCleanTasks.append("V")
            print("Found Stokes V. V will be CLEANed independently",file=log)
        else:
            print("Did not find stokes V image. Will not clean Circular Polarization.",file=log)
        # FGAO: end of the new section on polarization
        
        if ModelMachine is None:
            from DDFacet.Imager.HOGBOM import ClassModelMachineHogbom as ClassModelMachine
            self.ModelMachine = ClassModelMachine.ClassModelMachine(
                                self.GD, 
                                GainMachine=self.GainMachine)
        else:
            self.ModelMachine = ModelMachine
        self.GiveEdges = GiveEdges.GiveEdges
        
        # FGAO: put back the self._niter which includes polarizations
        #self._niter = 0
        self._niter = np.zeros([len(self.PolarizationCleanTasks)],dtype=np.int64)
        
        self._peakMode = "normal"

        self.CurrentNegMask = None
        self._NoiseMap = None
        self._PNRStop = None  # in _peakMode "sigma", provides addiitonal stopping criterion

        numexpr.set_num_threads(self.NCPU)


    def Init(self, **kwargs):
        self.SetPSF(kwargs["PSFVar"])
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
        self.Freqs = kwargs["GridFreqs"]
        AllDegridFreqs = []
        for i in kwargs["DegridFreqs"].keys():
            AllDegridFreqs.append(kwargs["DegridFreqs"][i])
        self.Freqs_degrid = np.unique(np.concatenate(AllDegridFreqs).flatten())
        self.SetPSF(kwargs["PSFVar"])
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
        self.ModelMachine.setPSFServer(self.PSFServer)
        self.ModelMachine.setFreqMachine(self.Freqs, self.Freqs_degrid,
                                         weights=kwargs["PSFVar"]["WeightChansImages"], PSFServer=self.PSFServer)


    def Reset(self):
        pass

    def setMaskMachine(self,MaskMachine):
        self.MaskMachine=MaskMachine
        if self.MaskMachine.ExternalMask is not None:
            print("Applying external mask", file=log)
            MaskArray=self.MaskMachine.ExternalMask
            nch,npol,_,_=MaskArray.shape
            self._MaskArray=np.zeros(MaskArray.shape,np.bool8)
            for ch in range(nch):
                for pol in range(npol):
                    self._MaskArray[ch,pol,:,:]=np.bool8(1-MaskArray[ch,pol].copy())[:,:]
            self._MaskArray = np.ascontiguousarray(self._MaskArray)
            self.MaskArray = np.ascontiguousarray(self._MaskArray[0])

    def SetModelRefFreq(self, RefFreq):
        """
        Sets ref freq in ModelMachine.
        """
        AllFreqs = []
        AllFreqsMean = np.zeros((self.NFreqBand,), np.float32)
        for iChannel in range(self.NFreqBand):
            AllFreqs += self.DicoVariablePSF["freqs"][iChannel]
            AllFreqsMean[iChannel] = np.mean(self.DicoVariablePSF["freqs"][iChannel])
        #assume that the frequency variance is somewhat the same in all the stokes images:
        #RefFreq = np.sum(AllFreqsMean.ravel() * np.mean(self.DicoVariablePSF["WeightChansImages"],axis=1).ravel())
        self.ModelMachine.setRefFreq(RefFreq)


    def SetModelShape(self):
        """
        Sets the shape params of model, call in every update step
        """
        self.ModelMachine.setModelShape(self._Dirty.shape)

    def GiveModelImage(self, *args): return self.ModelMachine.GiveModelImage(*args)

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe
        

    def SetPSF(self,DicoVariablePSF):
        self.PSFServer=ClassPSFServer(self.GD)
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF, NormalisePSF=True)
        self.DicoVariablePSF=DicoVariablePSF

    def setNoiseMap(self, NoiseMap, PNRStop=10):
        """Sets the noise map. The mean dirty will be divided by the noise map before peak finding.
        If PNRStop is set, an additional stopping criterion (peak-to-noisemap) will be applied.
            Peaks are reported in units of sigmas.
        If PNRStop is not set, NoiseMap is treated as simply an (inverse) weighting that will bias
            peak selection in the minor cycle. In this mode, peaks are reported in units of flux.
        """
        self._NoiseMap = NoiseMap
        self._PNRStop = PNRStop
        self._peakMode = "sigma"

    def SetDirty(self,DicoDirty):
        self.DicoDirty=DicoDirty
        self.WeightsChansImages = DicoDirty["WeightChansImages"].squeeze()
        self._Dirty = self.DicoDirty["ImageCube"]
        self._MeanDirty = self.DicoDirty["MeanImage"]

        self.NpixPSF = self.PSFServer.NPSF
        self.Nchan, self.Npol, self.Npix, _ = self._Dirty.shape


        # if self._peakMode is "sigma":
        #     print("Will search for the peak in the SNR-weighted dirty map", file=log)
        #     a, b = self._MeanDirty, self._NoiseMap.reshape(self._MeanDirty.shape)
        #     self._PeakSearchImage = numexpr.evaluate("a/b")
        # # elif self._peakMode is "weighted":   ######## will need to get a PeakWeightImage from somewhere for this option
        # #     print("Will search for the peak in the weighted dirty map", file=log)
        # #     a, b = self._MeanDirty, self._peakWeightImage
        # #     self._PeakSearchImage = numexpr.evaluate("a*b")
        # else:
        #     print("Will search for the peak in the unweighted dirty map", file=log)
        #     self._PeakSearchImage = self._MeanDirty

        if self.ModelImage is None:
            self._ModelImage=np.zeros_like(self._Dirty)
        if self.MaskArray is None:
            self._MaskArray=np.zeros(self._Dirty.shape,dtype=np.bool8)


    def SubStep(self,xc,yc,indexPol,LocalSM):
        """
        This is where subtraction in the image domain happens
        
        Parameters
        ----------
        (xc, yc) - The location of the component
        LocalSM - array of shape (nchan, npol, nx, ny)
                  Local Sky Model = comp * PSF * gain where the PSF should be
                  normalised to unity at the center.
        """
        #Get overlap indices where psf should be subtracted
        Aedge,Bedge=self.GiveEdges(xc,yc, self.Npix, self.NpixPSF//2,self.NpixPSF//2,self.NpixPSF)

        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge
        
        cube, sm = self._Dirty[:,indexPol,x0d:x1d,y0d:y1d], \
            LocalSM[:,:,x0p:x1p,y0p:y1p]
        cube = cube[:, None, :, :]
        numexpr.evaluate('cube-sm',out=cube,casting="unsafe")
        
        # If multiple frequencies are present construct the weighted mean
        meanimage = self._MeanDirty[:, indexPol, x0d:x1d, y0d:y1d]
        #print("meanimage.shape= (at definition)",meanimage.shape)
        if self.MultiFreqMode:
            selWCI = self.WeightsChansImages[:, indexPol] if self.WeightsChansImages.ndim == 2 else self.WeightsChansImages
            W = selWCI.reshape((self.Nchan,1,1,1))
            meanimage[...] = (cube*W).sum(axis=0) #Sum over frequency
        else:
            # ori
            #meanimage[0,indexPol,...] = cube[0,indexPol,...]
            # try
            meanimage[...] = cube[0,...]

    def Deconvolve(self, **kwargs):
        """
        Runs minor cycle over image channel 'ch'.
        initMinor is number of minor iteration (keeps continuous count through major iterations)
        Nminor is max number of minor iterations

        Returns tuple of: return_code,continue,updated
        where return_code is a status string;
        continue is True if another cycle should be executed (one or more polarizations still need cleaning);
        update is True if one or more polarization models have been updated
        """
        exit_msg = ""
        continue_deconvolution = np.zeros((len(self.PolarizationCleanTasks), ), 
                                          dtype=np.bool8)
        update_model = False
        complex_clean = self.ComplexClean

        # starts the outside loop for polarizations:
        for pol_task_id, pol_task in enumerate(self.PolarizationCleanTasks):
            print(ModColor.Str("Now deconvolving %s for this major iteration" % (pol_task), col="red"), file=log)
            #first check the iteration counts
            if self._niter[pol_task_id] >= self.MaxMinorIter:
                print(ModColor.Str("Minor cycle CLEANing of %s has already reached the maximum number of minor " % (pol_task), col="red"), file=log)
                
                print(ModColor.Str("cycles... won't CLEAN this polarization further." ,col="red"),file=log)
                exit_msg = exit_msg +" " + "MaxIter"
                continue_deconvolution[pol_task_id] = False
                continue #Previous minor clean on this polarization has reached the maximum number of minor cycles.... onwards to the next polarization
            else:
                continue_deconvolution[pol_task_id] = True # continue if stop flag has already been raised

            PeakMap = None
            # FG 20220620:


            _,_,npix,_=self._Dirty.shape
            xc=(npix)/2

            nchan,npol,_,_=self._Dirty.shape
            

            # FG20210604: the dirty image axis are n_chan, n_pol, n_pix, n_pix,
            # FG20211217: here I just use the MeanDirty instead of the Dirty, to follow what has been done before
            if pol_task == "I":
                indexI = self.PolarizationDescriptor.index("I")
                PeakMap = self._MeanDirty[0, indexI, :, :]
            elif pol_task == "Q":
                indexQ = self.PolarizationDescriptor.index("Q")
                PeakMap = self._MeanDirty[0, indexQ, :, :]
            elif pol_task == "U":
                indexU = self.PolarizationDescriptor.index("U")
                PeakMap = self._MeanDirty[0, indexU, :, :]
            elif pol_task == "Q+iU":
                # FG here the _MeanDirty already suffered from smearing due to unknown RM, so cannot use it.
                indexQ = self.PolarizationDescriptor.index("Q")
                indexU = self.PolarizationDescriptor.index("U")
                Q2U2_sum_map  = np.zeros((npix*npix),dtype=np.float32).reshape(npix,npix)
                for j in range(nchan):
                    Q2U2_sum_map += np.abs(self._Dirty[j, indexQ,:,:] + 1.0j * self._Dirty[j, indexU, :, :]) **2
                PeakMap = Q2U2_sum_map/nchan
                #PeakMap = np.abs(self._MeanDirty[0, indexQ, :, :] + 1.0j * self._MeanDirty[0, indexU, :, :]) ** 2
            elif pol_task == "V":
                indexV = self.PolarizationDescriptor.index("V")
                PeakMap = self._MeanDirty[0, indexV, :, :]
            else:
                raise ValueError("Invalid polarization cleaning task: %s. This is a bug" % pol_task)

            if len(self.PolarizationDescriptor) > 1:
                PeakMap = PeakMap.copy() #Argmax must take non-strided array

            m0,m1=PeakMap.min(),PeakMap.max()

            #These options should probably be moved into MinorCycleConfig in parset
            # Since Q, U and V can be negative we should always clean the negatives
            # Q+iU complex clean is in quadrature space so will be positive
            DoAbs=int(self.GD["Deconv"]["AllowNegative"] or pol_task in ["Q", "U", "V"])
            print("  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter[pol_task_id], self.MaxMinorIter, DoAbs), file=log)

            ## Determine which stopping criterion to use for flux limit
            #Get RMS stopping criterion
            NPixStats = self.GD["Deconv"]["NumRMSSamples"]
            ## FG: what is this NPixStats doing here?
            if NPixStats:
                # here self.Npix is new compared to before
                RandomInd=np.int64(np.random.rand(NPixStats)*self.Npix**2)
                if pol_task == "Q+iU":
                    RMS = np.std(np.sqrt(np.real(PeakMap).ravel()[RandomInd]))
                else:
                    RMS=np.std(np.real(PeakMap.ravel()[RandomInd]))
            else:
                if pol_task == "Q+iU":
                    RMS=np.std(np.sqrt(PeakMap))
                else:
                    RMS = np.std(PeakMap)

            self.RMS = RMS

            Fluxlimit_RMS = self.RMSFactor*RMS

            # Find position and intensity of first peak
            x,y,MaxDirty=NpParallel.A_whereMax(PeakMap,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)

            if pol_task == "I":
                pass
            elif pol_task == "Q+iU":
                MaxDirty = np.sqrt(MaxDirty)
            elif pol_task == "Q":
                pass
            elif pol_task == "U":
                pass
            elif pol_task == "V":
                pass
            else:
                raise ValueError("Invalid polarization cleaning task: %s. This is a bug" % pol_task)
                
            # Get peak factor stopping criterion
            Fluxlimit_Peak = MaxDirty*self.PeakFactor

            # Get side lobe stopping criterion
            Fluxlimit_Sidelobe = ((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*MaxDirty if self.CycleFactor else 0

            mm0, mm1 = PeakMap.min(), PeakMap.max()

            # Choose whichever threshold is highest
            StopFlux = max(Fluxlimit_Peak, Fluxlimit_RMS, Fluxlimit_Sidelobe, self.FluxThreshold)

            print("    Dirty image peak flux      = %10.6f Jy [(min, max) = (%.3g, %.3g) Jy]"%(MaxDirty,mm0,mm1), file=log)
            print("      RMS-based threshold      = %10.6f Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor), file=log)
            print("      Sidelobe-based threshold = %10.6f Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor), file=log)
            print("      Peak-based threshold     = %10.6f Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor), file=log)
            print("      Absolute threshold       = %10.6f Jy"%(self.FluxThreshold), file=log)
            print("    Stopping flux              = %10.6f Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty), file=log)

            T=ClassTimeIt.ClassTimeIt()
            T.disable()
            
            # FG: here ThisFlux is just the peak before the minor loop
            ThisFlux=MaxDirty

            if ThisFlux < StopFlux:
                print(ModColor.Str("    Initial maximum peak %g Jy below threshold, we're done CLEANing" % (ThisFlux),col="green" ), file=log)
                exit_msg = exit_msg + " " + "FluxThreshold"
                continue_deconvolution[pol_task_id] = False
                # FG: since we are looping through different polarization, now cannot just return
                continue

            #Do minor cycle deconvolution loop
            try:
                istart = self._niter[pol_task_id]
                for i in range(self._niter[pol_task_id]+1,self.MaxMinorIter+1):
                    self._niter[pol_task_id] = i
                    PeakMap = None
                    
                    if pol_task == "I":
                        indexI = self.PolarizationDescriptor.index("I")
                        PeakMap = self._MeanDirty[0, indexI, :, :]
                        
                    elif pol_task == "Q":
                        indexQ = self.PolarizationDescriptor.index("Q")
                        PeakMap = self._MeanDirty[0, indexQ, :, :]
                        
                    elif pol_task == "U":
                        indexU = self.PolarizationDescriptor.index("U")
                        PeakMap = self._MeanDirty[0, indexU, :, :]
                        
                    elif pol_task == "Q+iU":
                        indexQ = self.PolarizationDescriptor.index("Q")
                        indexU = self.PolarizationDescriptor.index("U")
                        #PeakMap = np.abs(self._MeanDirty[0, indexQ, :, :] + 1.0j * self._MeanDirty[0, indexU, :, :]) ** 2        
                        Q2U2_sum_map  = np.zeros((npix*npix),dtype=np.float32).reshape(npix,npix)
                        for j in range(nchan):
                            Q2U2_sum_map += np.abs(self._Dirty[j, indexQ,:,:] + 1.0j * self._Dirty[j, indexU, :, :]) **2
                        PeakMap = Q2U2_sum_map/nchan

                    elif pol_task == "V":
                        indexV = self.PolarizationDescriptor.index("V")
                        PeakMap = self._MeanDirty[0, indexV, :, :]
                    else:
                        raise ValueError("Invalid polarization cleaning task: %s. This is a bug" % pol_task)
                    
                    if len(self.PolarizationDescriptor) > 1:
                        PeakMap = PeakMap.copy() #Argmax must take non-strided array

                    # FG: now x,y is the location of the peak in the current minor loop
                    x,y,ThisFlux=NpParallel.A_whereMax(PeakMap,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)
                    
                    # FG: just added back
                    if pol_task == "I":
                        pass
                    elif pol_task == "Q+iU":
                        ThisFlux = np.sqrt(ThisFlux)
                    elif pol_task == "Q":
                        pass
                    elif pol_task == "U":
                        pass
                    elif pol_task == "V":
                        pass
                    else:
                        raise ValueError("Invalid polarization cleaning task: %s. This is a bug" % pol_task)
                    
                    # FG: not sure if the following line is useful, needs to check
                    #self.GainMachine.SetFluxMax(ThisFlux)

                    T.timeit("max0")
                    
                    # FG: following line 423-428 is new, needs to check
                    from DDFacet.Imager import ClassDeconvMachine
                    if ClassDeconvMachine.user_stopped:
                        print(ModColor.Str("    user stop signal received"))
                        exit_msg = exit_msg + " " + "UserStop"
                        continue_deconvolution[...] = False
                        break # stop cleaning if threshold reached

                    if ThisFlux <= StopFlux:
                        print(ModColor.Str("    CLEANing [iter=%i] peak of %.3g Jy lower than stopping flux" % (i,ThisFlux),col="green"), file=log)
                        cont = ThisFlux > self.FluxThreshold
                        if not cont:
                              print(ModColor.Str("    CLEANing [iter=%i] absolute flux threshold of %.3g Jy has been reached" % (i,self.FluxThreshold),col="green",Bold=True), file=log)
                        exit_msg = exit_msg + " " + "MinFluxRms"
                        
                        # Unless stopped because of other criteria we continue in next cycle if needed
                        continue_deconvolution[pol_task_id] = cont and continue_deconvolution[pol_task_id]
                        # Clean threshold reached Check next polarization
                        break

                    # This is used to track Cleaning progress
                    rounded_iter_step = 1 if i < 10 else (
                        10 if i < 200 else (
                            100 if i < 2000
                            else 1000))
                    
                    # FG: here just comment out this if statement
                    #if (i >= 1 and i % rounded_iter_step == 0) or (istart > 0 and self._niter[pol_task_id] - 1 == istart):
                    #print("    [iter=%i] peak residual %.3g" % (i, ThisFlux), file=log)
                    #print("silly test,i=",i)

                    
                    
                    # Find PSF corresponding to location (x,y)
                    self.PSFServer.setLocation(x, y)  # Selects the facet closest to (x,y)

                    # Get the JonesNorm
                    # BH: This is a single weight per PSF, so since
                    # the PSF is polarization invarient we use the
                    # same as Stokes I
                    JonesNorm = self.DicoDirty["JonesNorm"][:, 0, x, y]

                    # Get the solution (division by JonesNorm handled in fit)
                    # next step the polynomial fitting will get the coeffs
                    if pol_task == "I":
                        Iapp = self._Dirty[:, indexI, x, y]
                    elif pol_task == "Q":
                        Iapp = self._Dirty[:, indexQ, x, y]
                    elif pol_task == "U":
                        Iapp = self._Dirty[:, indexU, x, y]
                    elif pol_task == "V":
                        Iapp = self._Dirty[:, indexV, x, y]
                        
                    elif pol_task == "Q+iU":
                        Iapp_Q = self._Dirty[:, indexQ, x, y]
                        Iapp_U = self._Dirty[:, indexU, x, y]
                    
                    # Fit a polynomial to get coeffs
                    # We fit Q and U separately
                    # Another option is to fit angle, but just as with fitting phase
                    # this can be troublesome due to the discontinuities with angles
                    if pol_task == "Q+iU":
                        indx = indexQ
                        W = self.WeightsChansImages[:, indx] if self.WeightsChansImages.ndim == 2 else self.WeightsChansImages
                        # FG:the Coeffs_Q will have the length of the exact polynomial order used in the fitting
                        Coeffs_Q = self.ModelMachine.FreqMachine.Fit(Iapp_Q, JonesNorm, W)
                        Coeffs_Q_dev = self.ModelMachine.FreqMachine.FitLin(Iapp_Q, JonesNorm, W) 
                        indx = indexU
                        W = self.WeightsChansImages[:, indx] if self.WeightsChansImages.ndim == 2 else self.WeightsChansImages
                        Coeffs_U = self.ModelMachine.FreqMachine.Fit(Iapp_U, JonesNorm, W)
                        Coeffs_U_dev = self.ModelMachine.FreqMachine.FitLin(Iapp_U, JonesNorm, W)
                    elif pol_task == "Q" :
                        indx = indexQ
                        W = self.WeightsChansImages[:, indx] if self.WeightsChansImages.ndim == 2 else self.WeightsChansImages
                        # FG:the Coeffs_Q will have the length of the exact polynomial order used in the fitting
                        Coeffs_Q = self.ModelMachine.FreqMachine.Fit(Iapp, JonesNorm, W)
                        Coeffs_Q_dev = self.ModelMachine.FreqMachine.FitLin(Iapp, JonesNorm, W) 
                    elif pol_task == "U":
                        indx = indexU
                        W = self.WeightsChansImages[:, indx] if self.WeightsChansImages.ndim == 2 else self.WeightsChansImages
                        Coeffs_U = self.ModelMachine.FreqMachine.Fit(Iapp, JonesNorm, W)
                        Coeffs_U_dev = self.ModelMachine.FreqMachine.FitLin(Iapp, JonesNorm, W)
                    else:
                        indx = indexI if pol_task == "I" else indexV
                        W = self.WeightsChansImages[:, indx] if self.WeightsChansImages.ndim == 2 else self.WeightsChansImages
                        Coeffs = self.ModelMachine.FreqMachine.Fit(Iapp, JonesNorm, W)
                    
                    # Overwrite with polynoimial fit
                    if pol_task == "Q+iU":
                        Iapp_Q = self.ModelMachine.FreqMachine.Eval(Coeffs_Q)
                        Iapp_U = self.ModelMachine.FreqMachine.Eval(Coeffs_U)
                        Iapp_Q_dev = self.ModelMachine.FreqMachine.EvalLin(Coeffs_Q_dev, self.ModelMachine.FreqMachine.Freqs)
                        Iapp_U_dev = self.ModelMachine.FreqMachine.EvalLin(Coeffs_U_dev, self.ModelMachine.FreqMachine.Freqs)
                    elif pol_task == "Q" :
                        Iapp_Q = self.ModelMachine.FreqMachine.Eval(Coeffs_Q)
                        Iapp_Q_dev = self.ModelMachine.FreqMachine.EvalLin(Coeffs_Q_dev, self.ModelMachine.FreqMachine.Freqs)
                    elif pol_task == "U":
                        Iapp_U = self.ModelMachine.FreqMachine.Eval(Coeffs_U)
                        Iapp_U_dev = self.ModelMachine.FreqMachine.EvalLin(Coeffs_U_dev, self.ModelMachine.FreqMachine.Freqs)
                    else:
                        Iapp = self.ModelMachine.FreqMachine.Eval(Coeffs)
                    
                    T.timeit("stuff")


                    PSF, meanPSF = self.PSFServer.GivePSF()  #Gives associated PSF
                    PSF = PSF[:, 0, :, :] # PSF same for all stokes
                    meanPSF = meanPSF[:, 0, :, :] # PSF same for all stokes
                    PSF = PSF[:, None, :, :]
                    meanPSF = meanPSF[:, None, :, :]

                    T.timeit("FindScale")

                    if pol_task == "I":
                        self.ModelMachine.AppendComponentToDictStacked((x, y), Coeffs, indexI)
                    elif pol_task == "Q":
                        self.ModelMachine.AppendComponentToDictStacked((x, y), Coeffs_Q_dev, indexQ)
                    elif pol_task == "U":
                        self.ModelMachine.AppendComponentToDictStacked((x, y), Coeffs_U_dev, indexU)
                    elif pol_task == "V":
                        self.ModelMachine.AppendComponentToDictStacked((x, y), Coeffs, indexV)
                    elif pol_task == "Q+iU": 
                        self.ModelMachine.AppendComponentToDictStacked((x, y), Coeffs_Q_dev, indexQ)
                        self.ModelMachine.AppendComponentToDictStacked((x, y), Coeffs_U_dev, indexU)
                        
                    update_model = True

                    # Subtract LocalSM*CurrentGain from dirty image
                    if pol_task == "I":
                        self.SubStep(x, y, indexI, PSF * Iapp[:, None, None, None] * self.GD["Deconv"]["Gain"])
                    elif pol_task == "Q":
                        self.SubStep(x, y, indexQ, PSF * Iapp[:, None, None, None] * self.GD["Deconv"]["Gain"])
                    elif pol_task == "U":
                        self.SubStep(x, y, indexU, PSF * Iapp[:, None, None, None] * self.GD["Deconv"]["Gain"])
                    elif pol_task == "V":
                        self.SubStep(x, y, indexV, PSF * Iapp[:, None, None, None] * self.GD["Deconv"]["Gain"])
                    elif pol_task == "Q+iU":
                        self.SubStep(x, y, indexQ, PSF * Iapp_Q_dev[:, None, None, None] * self.GD["Deconv"]["Gain"])
                        self.SubStep(x, y, indexU, PSF * Iapp_U_dev[:, None, None, None] * self.GD["Deconv"]["Gain"])
                        
                    T.timeit("SubStep")

                    T.timeit("End")

            except KeyboardInterrupt:
                print(ModColor.Str("    CLEANing [iter=%i] minor cycle interrupted with Ctrl+C, peak flux %.3g" % (self._niter[pol_task_id], ThisFlux)), file=log)
                exit_msg = exit_msg + " " + "MaxIter"
                continue_deconvolution[...] = False
                return exit_msg, np.any(continue_deconvolution), update_model

            if self._niter[pol_task_id] >= self.MaxMinorIter: #Reached maximum number of iterations:
                print(ModColor.Str("    CLEANing [iter=%i] Reached maximum number of iterations, peak flux %.3g" % (self._niter[pol_task_id], ThisFlux)), file=log)
                exit_msg = exit_msg + " " + "MaxIter"
                continue_deconvolution[pol_task_id] = False

        # if any polarizations still need cleaning then we will continue in the next major
        return exit_msg, np.any(continue_deconvolution), update_model

    def Update(self,DicoDirty,**kwargs):
        """
        Method to update attributes from ClassDeconvMachine
        """
        #Update image dict
        self.SetDirty(DicoDirty)
        #self.SetModelRefFreq()
        self.SetModelShape()

    def ToFile(self,fname):
        """
        Method to write model image to file
        """
        self.ModelMachine.ToFile(fname)


    def FromFile(self, fname):
        """
        Read model dict from file SubtractModel
        """
        self.ModelMachine.FromFile(fname)

    def updateRMS(self):
        _,npol,npix,_ = self._MeanDirty.shape
        NPixStats = self.GD["Deconv"]["NumRMSSamples"]
        if NPixStats:
            #self.IndStats=np.int64(np.random.rand(NPixStats)*npix**2)
            self.IndStats=np.int64(np.linspace(0,self._PeakSearchImage.size-1,NPixStats))
        else:
            self.IndStats = slice(None)
        self.RMS=np.std(np.real(self._PeakSearchImage.ravel()[self.IndStats]))

    def resetCounter(self):
        self._niter = 0

