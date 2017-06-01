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

"""
This minimal implementation of standard Hogbom CLEAN algorithm should serve
as a minimal reference interface of how to incorporate new deconvolution
algorithms into DDFacet.
"""

import numpy as np
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.Other import ClassTimeIt
from pyrap.images import image
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Imager import ClassGainMachine # Currently required by model machine but fixed to static mode


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
        self.FluxThreshold = FluxThreshold 
        self.CycleFactor = CycleFactor
        self.RMSFactor = RMSFactor
        self.PeakFactor = PeakFactor
        self.GainMachine = ClassGainMachine.ClassGainMachine(GainMin=Gain)
        if ModelMachine is None:
            from DDFacet.Imager import ClassModelMachineHogbom as ClassModelMachine
            self.ModelMachine=ClassModelMachine.ClassModelMachine(self.GD,GainMachine=self.GainMachine)
        else:
            self.ModelMachine = ModelMachine
        self.GainMachine = self.ModelMachine.GainMachine
        self.PolarizationDescriptor = ImagePolDescriptor
        self.PolarizationCleanTasks = []
        if "I" in self.PolarizationDescriptor:
            self.PolarizationCleanTasks.append("I")
            print>>log,"Found Stokes I. I will be CLEANed independently"
        else:
            print>>log, "Stokes I not available. Not performing intensity CLEANing."
        if "V" in self.PolarizationDescriptor:
            self.PolarizationCleanTasks.append("V")
            print>> log, "Found Stokes V. V will be CLEANed independently"
        else:
            print>>log, "Did not find stokes V image. Will not clean Circular Polarization."
        if set(["Q","U"]) < set(self.PolarizationDescriptor):
            self.PolarizationCleanTasks.append("Q+iU") #Luke Pratley's complex polarization CLEAN
            print>> log, "Found Stokes Q and U. Will perform joint linear (Pratley-Johnston-Hollitt) CLEAN."
        else:
            print>>log, "Stokes Images Q and U must both be synthesized to CLEAN linear polarization. " \
                        "Will not CLEAN these."
        # reset overall iteration counter
        self._niter = np.zeros([len(self.PolarizationCleanTasks)],dtype=np.int64)
        if CleanMaskImage is not None:
            print>>log, "Reading mask image: %s"%CleanMaskImage
            MaskArray=image(CleanMaskImage).getdata()
            nch,npol,_,_=MaskArray.shape
            self._MaskArray=np.zeros(MaskArray.shape,np.bool8)
            for ch in range(nch):
                for pol in range(npol):
                    self._MaskArray[ch,pol,:,:]=np.bool8(1-MaskArray[ch,pol].T[::-1].copy())[:,:]
            self.MaskArray=self._MaskArray[0]


    def Init(self, **kwargs):
        self.SetPSF(kwargs["PSFVar"])
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
        self.SetModelRefFreq(kwargs["RefFreq"])
        self.ModelMachine.setFreqMachine(kwargs["GridFreqs"], kwargs["DegridFreqs"])
        self.Freqs = kwargs["GridFreqs"]
        # tmp = [{'Alpha': 0.0, 'Scale': 0, 'ModelType': 'Delta'}]
        # AlphaMin, AlphaMax, NAlpha = self.GD["HMP"]["Alpha"]
        # if NAlpha > 1:
        #     print>>log, "Multi-frequency synthesis not supported in Hogbom CLEAN. Setting alpha to zero"
        #
        # self.ModelMachine.setListComponants(tmp)

        # Set gridding Freqs

    def Reset(self):
        pass

    def setMaskMachine(self,MaskMachine):
        self.MaskMachine=MaskMachine


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

    def GiveModelImage(self,*args): return self.ModelMachine.GiveModelImage(*args)

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe
        

    def SetPSF(self,DicoVariablePSF):
        self.PSFServer=ClassPSFServer(self.GD)
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF)
        self.DicoVariablePSF=DicoVariablePSF

    def SetDirty(self,DicoDirty):
        self.DicoDirty=DicoDirty
        self._Dirty = self.DicoDirty["ImageCube"]
        self._MeanDirty = self.DicoDirty["MeanImage"]

        NPSF=self.PSFServer.NPSF
        _,_,NDirty,_=self._Dirty.shape

        off=(NPSF-NDirty)/2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)

        if self.ModelImage is None:
            self._ModelImage=np.zeros_like(self._Dirty)
        if self.MaskArray is None:
            self._MaskArray=np.zeros(self._Dirty.shape,dtype=np.bool8)

    def GiveEdges(self,(xc0,yc0),N0,(xc1,yc1),N1):
        """
        Each pixel in the image is associated with a different facet each with
        a different PSF.
        This finds the indices corresponding to the edges of a local psf centered
        on a specific pixel, here xc0,yc0.
        """
        M_xc=xc0
        M_yc=yc0
        NpixMain=N0
        F_xc=xc1
        F_yc=yc1
        NpixFacet=N1
                
        ## X
        M_x0=M_xc-NpixFacet/2
        x0main=np.max([0,M_x0])
        dx0=x0main-M_x0
        x0facet=dx0
                
        M_x1=M_xc+NpixFacet/2
        x1main=np.min([NpixMain-1,M_x1])
        dx1=M_x1-x1main
        x1facet=NpixFacet-dx1
        x1main+=1

        ## Y
        M_y0=M_yc-NpixFacet/2
        y0main=np.max([0,M_y0])
        dy0=y0main-M_y0
        y0facet=dy0
        
        M_y1=M_yc+NpixFacet/2
        y1main=np.min([NpixMain-1,M_y1])
        dy1=M_y1-y1main
        y1facet=NpixFacet-dy1
        y1main+=1

        Aedge=[x0main,x1main,y0main,y1main]
        Bedge=[x0facet,x1facet,y0facet,y1facet]
        return Aedge,Bedge

    def SubStep(self,(dx,dy),LocalSM):
        """
        This is where subtraction in the image domain happens
        """
        npol,_,_=self.Dirty.shape
        x0,x1,y0,y1=self.DirtyExtent

        xc,yc=dx,dy
        N0=self.Dirty.shape[-1]
        N1=LocalSM.shape[-1]

        #Get overlap indices where psf should be subtracted
        Aedge,Bedge=self.GiveEdges((xc,yc),N0,(N1/2,N1/2),N1)

        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge

        #Subtract from each channel/band
        self._Dirty[:,:,x0d:x1d,y0d:y1d]-=LocalSM[:,:,x0p:x1p,y0p:y1p]
        #Subtract from the average
        if self.MultiFreqMode:  #If multiple frequencies are present construct the weighted mean
            W=np.mean(np.float32(self.DicoDirty["WeightChansImages"]),axis=1)  #Get the weights (assuming they stay relatively the same over stokes terms)
            self._MeanDirty[0,:,x0d:x1d,y0d:y1d]-=np.sum(LocalSM[:,:,x0p:x1p,y0p:y1p]*W.reshape((W.size,1,1,1)),axis=0) #Sum over frequency

    def setChannel(self,ch=0):
        """
        In case we ever want to deconvolve per channel.
        Currently just sets self.Dirty to average over freq bands.
        """
        #self.PSF=self._MeanPSF[ch]
        if self.MultiFreqMode:
            self.Dirty = self._MeanDirty.view()[ch]
        else:
            self.Dirty = self._Dirty.view()[ch]
        self.ModelImage = self._ModelImage.view()[ch]
        self.MaskArray = self._MaskArray.view()[ch]


    def Deconvolve(self, ch=0):
        """
        Runs minor cycle over image channel 'ch'.
        initMinor is number of minor iteration (keeps continuous count through major iterations)
        Nminor is max number of minor iterations

        Returns tuple of: return_code,continue,updated
        where return_code is a status string;
        continue is True if another cycle should be executed (one or more polarizations still need cleaning);
        update is True if one or more polarization models have been updated
        """
        #No need to set the channel when doing joint deconvolution
        self.setChannel(ch)

        exit_msg = ""
        continue_deconvolution = False
        update_model = False
        for pol_task_id, pol_task in enumerate(self.PolarizationCleanTasks):
            if self._niter[pol_task_id] >= self.MaxMinorIter:
                print>>log, ModColor.Str("Minor cycle CLEANing of %s has already reached the maximum number of minor "
                                         "cycles... won't CLEAN this polarization further." % pol_task)
                exit_msg = exit_msg +" " + "MaxIter"
                continue_deconvolution = continue_deconvolution or False
                update_model = update_model or False
                continue #Previous minor clean on this polarization has reached the maximum number of minor cycles.... onwards to the next polarization
            PeakMap = None
            if pol_task == "I":
                indexI = self.PolarizationDescriptor.index("I")
                PeakMap = self.Dirty[indexI, :, :]
            elif pol_task == "Q+iU":
                indexQ = self.PolarizationDescriptor.index("Q")
                indexU = self.PolarizationDescriptor.index("U")
                PeakMap = np.abs(self.Dirty[indexQ, :, :] + 1.0j * self.Dirty[indexU, :, :]) ** 2
            elif pol_task == "V":
                indexV = self.PolarizationDescriptor.index("V")
                PeakMap = self.Dirty[indexV, :, :]
            else:
                raise ValueError("Invalid polarization cleaning task: %s. This is a bug" % pol_task)

            _,npix,_=self.Dirty.shape
            xc=(npix)/2

            npol,_,_=self.Dirty.shape

            m0,m1=PeakMap.min(),PeakMap.max()

            #These options should probably be moved into MinorCycleConfig in parset
            DoAbs=int(self.GD["Deconv"]["AllowNegative"])
            print>>log, "  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter[pol_task_id],self.MaxMinorIter,DoAbs)

            ## Determine which stopping criterion to use for flux limit
            #Get RMS stopping criterion
            NPixStats = self.GD["Deconv"]["NumRMSSamples"]
            if NPixStats:
                RandomInd=np.int64(np.random.rand(NPixStats)*npix**2)
                if pol_task == "Q+iU":
                    RMS = np.std(np.sqrt(np.real(PeakMap).ravel()[RandomInd]))
                else:
                    RMS=np.std(np.real(PeakMap.ravel()[RandomInd]))
            else:
                if pol_task == "Q+iU":
                    RMS=np.std(np.sqrt(PeakMap))
                else:
                    RMS = np.std(PeakMap)
            self.RMS=RMS

            self.GainMachine.SetRMS(RMS)

            Fluxlimit_RMS = self.RMSFactor*RMS

            #Find position and intensity of first peak
            x,y,MaxDirty=NpParallel.A_whereMax(PeakMap,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)
            if pol_task == "I":
                pass
            elif pol_task == "Q+iU":
                MaxDirty = np.sqrt(MaxDirty)
            elif pol_task == "V":
                pass
            else:
                raise ValueError("Invalid polarization cleaning task: %s. This is a bug" % pol_task)
            #Itmp = np.argwhere(self.Dirty[0] == np.max(self.Dirty[0]))
            #x, y = Itmp[0]
            #MaxDirty = self.Dirty[0][x,y]

            #Get peak factor stopping criterion
            Fluxlimit_Peak = MaxDirty*self.PeakFactor

            #Get side lobe stopping criterion
            Fluxlimit_Sidelobe = ((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*MaxDirty if self.CycleFactor else 0

            mm0,mm1=PeakMap.min(),PeakMap.max()

            # Choose whichever threshold is highest
            StopFlux = max(Fluxlimit_Peak, Fluxlimit_RMS, Fluxlimit_Sidelobe, self.FluxThreshold)

            print>>log, "    Dirty %s image peak flux      = %10.6f Jy [(min, max) = (%.3g, %.3g) Jy]"%(pol_task,MaxDirty,mm0,mm1)
            print>>log, "      RMS-based threshold      = %10.6f Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor)
            print>>log, "      Sidelobe-based threshold = %10.6f Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor)
            print>>log, "      Peak-based threshold     = %10.6f Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor)
            print>>log, "      Absolute threshold       = %10.6f Jy"%(self.FluxThreshold)
            print>>log, "    Stopping flux              = %10.6f Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty)

            T=ClassTimeIt.ClassTimeIt()
            T.disable()

            ThisFlux=MaxDirty
            #print x,y

            if ThisFlux < StopFlux:
                print>>log, ModColor.Str("    Initial maximum peak %g Jy below threshold, we're done CLEANing %s" % (ThisFlux, pol_task),col="green" )
                exit_msg = exit_msg + " " + "FluxThreshold"
                continue_deconvolution = False or continue_deconvolution
                update_model = False or update_model
                continue #onto the next polarization

            #pBAR= ProgressBar(Title="Cleaning  %s " %  pol_task)
            # pBAR.disable()

            self.GainMachine.SetFluxMax(ThisFlux)
            #pBAR.render(0,100) # "g=%3.3f"%self.GainMachine.GiveGain())

            def GivePercentDone(ThisMaxFlux):
                fracDone=1.-(ThisMaxFlux-StopFlux)/(MaxDirty-StopFlux)
                return max(int(round(100*fracDone)),100)

            #Do minor cycle deconvolution loop
            try:
                for i in range(self._niter[pol_task_id]+1,self.MaxMinorIter+1):
                    self._niter[pol_task_id] = i
                    #grab a new peakmap
                    PeakMap = None
                    if pol_task == "I":
                        indexI = self.PolarizationDescriptor.index("I")
                        PeakMap = self.Dirty[indexI, :, :]
                    elif pol_task == "Q+iU":
                        indexQ = self.PolarizationDescriptor.index("Q")
                        indexU = self.PolarizationDescriptor.index("U")
                        PeakMap = np.abs(self.Dirty[indexQ, :, :] + 1.0j * self.Dirty[indexU, :, :]) ** 2
                    elif pol_task == "V":
                        indexV = self.PolarizationDescriptor.index("V")
                        PeakMap = self.Dirty[indexV, :, :]
                    else:
                        raise ValueError("Invalid polarization cleaning task: %s. This is a bug" % pol_task)

                    x,y,ThisFlux=NpParallel.A_whereMax(PeakMap,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)
                    if pol_task == "I":
                        pass
                    elif pol_task == "Q+iU":
                        ThisFlux = np.sqrt(ThisFlux)
                    elif pol_task == "V":
                        pass
                    else:
                        raise ValueError("Invalid polarization cleaning task: %s. This is a bug" % pol_task)

                    self.GainMachine.SetFluxMax(ThisFlux)

                    T.timeit("max0")

                    if ThisFlux <= StopFlux:
                        #pBAR.render(100,100) #"peak %.3g"%(ThisFlux,))
                        print>>log, ModColor.Str("    CLEANing %s [iter=%i] peak of %.3g Jy lower than stopping flux" % (pol_task,i,ThisFlux),col="green")
                        cont = ThisFlux > self.FluxThreshold
                        if not cont:
                              print>>log, ModColor.Str("    CLEANing %s [iter=%i] absolute flux threshold of %.3g Jy has been reached" % (pol_task,i,self.FluxThreshold),col="green",Bold=True)
                        exit_msg = exit_msg + " " + "MinFluxRms"
                        continue_deconvolution = cont or continue_deconvolution
                        update_model = True or update_model

                        break # stop cleaning this polariztion and move on to the next polarization job

                    rounded_iter_step = 1 if i < 10 else (
                        10 if i < 200 else (
                            100 if i < 2000
                            else 1000))
                    # min(int(10**math.floor(math.log10(i))), 10000)
                    if i >= 10 and i % rounded_iter_step == 0:
                        # if self.GD["Debug"]["PrintMinorCycleRMS"]:
                        #rms = np.std(np.real(self._CubeDirty.ravel()[self.IndStats]))
                        print>> log, "    [iter=%i] peak residual %.3g" % (i, ThisFlux)

                    nch,npol,_,_=self._Dirty.shape
                    #Fpol contains the intensities at (x,y) per freq and polarisation
                    Fpol = np.zeros([nch, npol, 1, 1], dtype=np.float32)
                    if self.GD["Hogbom"]["FreqMode"] == "Poly":
                        Ncoeffs = self.GD["Hogbom"]["PolyFitOrder"]
                    elif self.GD["Hogbom"]["FreqMode"] == "GPR":
                        Ncoeffs = self.GD["Hogbom"]["NumBasisFuncs"]
                    else:
                        raise NotImplementedError("FreqMode %s not supported" % self.GD["Hogbom"]["FreqMode"])
                    Coeffs = np.zeros([npol, Ncoeffs])

                    # Get the JonesNorm
                    JonesNorm = (self.DicoDirty["JonesNorm"][:, :, x, y]).reshape((nch, npol, 1, 1))
                    if pol_task == "I":
                        indexI = self.PolarizationDescriptor.index("I")
                        # Get the solution
                        Fpol[:, indexI, 0, 0] = self._Dirty[:, indexI, x, y]/np.sqrt(JonesNorm[:, indexI, 0, 0])
                        # Fit a polynomial to get coeffs
                        Coeffs[indexI, :] = self.ModelMachine.FreqMachine.Fit(Fpol[:, indexI, 0, 0])
                        # Overwrite with polynoimial fit
                        Fpol[:, indexI, 0, 0] = self.ModelMachine.FreqMachine.Eval(Coeffs[indexI, :])
                    elif pol_task == "Q+iU":
                        indexQ = self.PolarizationDescriptor.index("Q")
                        indexU = self.PolarizationDescriptor.index("U")
                        Fpol[:, indexQ, 0, 0] = self._Dirty[:, indexQ, x, y]
                        Coeffs[indexQ, :] = self.ModelMachine.FreqMachine.Fit(Fpol[:, indexQ, 0, 0])
                        Fpol[:, indexQ, 0, 0] = self.ModelMachine.FreqMachine.Eval(Coeffs[indexQ, :])
                        Fpol[:, indexU, 0, 0] = self._Dirty[:, indexU, x, y]
                        Coeffs[indexU, :] = self.ModelMachine.FreqMachine.Fit(Fpol[:, indexU, 0, 0])
                        Fpol[:, indexU, 0, 0] = self.ModelMachine.FreqMachine.Eval(Coeffs[indexU, :])
                    elif pol_task == "V":
                        indexV = self.PolarizationDescriptor.index("V")
                        Fpol[:, indexV, 0, 0] = self._Dirty[:, indexV, x, y]
                        Coeffs[indexV, :] = self.ModelMachine.FreqMachine.Fit(Fpol[:, indexV, 0, 0])
                        Fpol[:, indexV, 0, 0] = self.ModelMachine.FreqMachine.Eval(Coeffs[indexV, :])
                    else:
                        raise ValueError("Invalid polarization cleaning task: %s. This is a bug" % pol_task)

                    # if self.ModelMachine.FreqMachine.GP.SolverFlag:
                    #     print " Flag set at location ", x, y
                    #nchan, npol, _, _ = Fpol.shape
                    #JonesNorm = (self.DicoDirty["JonesNorm"][:, :, x, y]).reshape((nchan, npol, 1, 1))
                    # dx=x-xc
                    # dy=y-xc
                    T.timeit("stuff")

                    #Find PSF corresponding to location (x,y)
                    self.PSFServer.setLocation(x,y) #Selects the facet closest to (x,y)
                    PSF, meanPSF = self.PSFServer.GivePSF()  #Gives associated PSF

                    T.timeit("FindScale")

                    CurrentGain = self.GainMachine.GiveGain()
                    #Update model
                    if pol_task == "I":
                        indexI = self.PolarizationDescriptor.index("I")
                        self.ModelMachine.AppendComponentToDictStacked((x, y), 1.0, Coeffs[indexI, :], indexI)
                    elif pol_task == "Q+iU":
                        indexQ = self.PolarizationDescriptor.index("Q")
                        indexU = self.PolarizationDescriptor.index("U")
                        self.ModelMachine.AppendComponentToDictStacked((x, y), 1.0, Coeffs[indexQ, :], indexQ)
                        self.ModelMachine.AppendComponentToDictStacked((x, y), 1.0, Coeffs[indexU, :], indexU)
                    elif pol_task == "V":
                        indexV = self.PolarizationDescriptor.index("V")
                        self.ModelMachine.AppendComponentToDictStacked((x, y), 1.0, Coeffs[indexV, :], indexV)
                    else:
                        raise ValueError("Invalid polarization cleaning task: %s. This is a bug" % pol_task)

                    # Subtract LocalSM*CurrentGain from dirty image
                    _,_,PSFnx,PSFny = PSF.shape
                    PSF /= np.amax(PSF.reshape(nch,npol,PSFnx*PSFny), axis=2, keepdims=True).reshape(nch,npol,1,1) #Normalise PSF in each channel
                    #tmp = PSF*Fpol*np.sqrt(JonesNorm)
                    self.SubStep((x,y),PSF*Fpol*CurrentGain*np.sqrt(JonesNorm))
                    T.timeit("SubStep")

                    T.timeit("End")

            except KeyboardInterrupt:
                print>>log, ModColor.Str("    CLEANing %s [iter=%i] minor cycle interrupted with Ctrl+C, peak flux %.3g" % (pol_task,self._niter[pol_task_id], ThisFlux))
                exit_msg = exit_msg + " " + "MaxIter"
                continue_deconvolution = False or continue_deconvolution
                update_model = True or update_model
                break #stop cleaning this polarization and move on to the next one

            if self._niter[pol_task_id] >= self.MaxMinorIter: #Reached maximum iterations for this polarization:
                print>> log, ModColor.Str("    CLEANing %s [iter=%i] Reached maximum number of iterations, peak flux %.3g" % (pol_task, self._niter[pol_task_id], ThisFlux))
                exit_msg = exit_msg + " " + "MaxIter"
                continue_deconvolution = False or continue_deconvolution
                update_model = True or update_model
            #onwards to the next polarization <--

        return exit_msg, continue_deconvolution, update_model

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
