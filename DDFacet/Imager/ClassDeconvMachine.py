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

from ClassFacetMachineTessel import ClassFacetMachineTessel as ClassFacetMachine
import numpy as np
from pyrap.images import image
from DDFacet.Other import MyPickle
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Array import NpShared
import os
from DDFacet.ToolsDir import ModFitPSF
from DDFacet.Data import ClassVisServer
from DDFacet.Data import ClassMS
import ClassCasaImage
from ModModelMachine import ClassModModelMachine
import time
import glob
from DDFacet.Other import ModColor
from DDFacet.Other import MyLogger
import traceback
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
from DDFacet.ToolsDir.ClassAdaptShape import ClassAdaptShape
import copy
from DDFacet.Other import AsyncProcessPool
from DDFacet.Other.AsyncProcessPool import APP
import cPickle
log=MyLogger.getLogger("ClassImagerDeconv")
import DDFacet.Data.ClassBeamMean as ClassBeamMean
from DDFacet.Imager import ClassMaskMachine
from DDFacet.Array import shared_dict
from DDFacet.Other import ClassTimeIt
import numexpr
from DDFacet.Imager import ClassImageNoiseMachine
from DDFacet.Data import ClassStokes


# from astropy import wcs
# from astropy.io import fits
#
# def load_wcs_from_file(filename):
#     # Load the FITS hdulist using astropy.io.fits
#     hdulist = fits.open(filename)
#
#     # Parse the WCS keywords in the primary HDU
#     w = wcs.WCS(hdulist[0].header)
#
#     # Print out the "name" of the WCS, as defined in the FITS header
#     print w.wcs.name
#
#     # Print out all of the settings that were parsed from the header
#     w.wcs.print_contents()
#
#     # Some pixel coordinates of interest.
#     pixcrd = np.array([[0, 0], [24, 38], [45, 98]], numpy.float_)
#
#     # Convert pixel coordinates to world coordinates
#     # The second argument is "origin" -- in this case we're declaring we
#     # have 1-based (Fortran-like) coordinates.
#     world = w.wcs_pix2world(pixcrd, 1)
#     print world
#
#     # Convert the same coordinates back to pixel coordinates.
#     pixcrd2 = w.wcs_world2pix(world, 1)
#     print pixcrd2
#
#     # These should be the same as the original pixel coordinates, modulo
#     # some floating-point error.
#     assert np.max(np.abs(pixcrd - pixcrd2)) < 1e-6


class ClassImagerDeconv():
    def __init__(self, GD=None,
                 PointingID=0,BaseName="ImageTest2",ReplaceDico=None,
                 predict_only=False, data=True, psf=True, readcol=True, deconvolve=True):
        # if ParsetFile is not None:
        #     GD=ClassGlobalData(ParsetFile)
        #     self.GD=GD

        if GD is not None:
            self.GD=GD

        self.BaseName=BaseName
        self.DicoModelName="%s.DicoModel"%self.BaseName
        self.DicoMetroModelName="%s.Metro.DicoModel"%self.BaseName
        self.PointingID=PointingID
        self.do_predict_only = predict_only
        self.do_data, self.do_psf, self.do_readcol, self.do_deconvolve = data, psf, readcol, deconvolve
 
        self.FacetMachine=None
        self.FWHMBeam = None
        self.PSFGaussPars = None
        self.PSFSidelobes = None
        self.FWHMBeamAvg = None
        self.PSFGaussParsAvg = None
        self.PSFSidelobesAvg = None
        self.HasFittedPSFBeam=False
        self.fit_stat = None # PSF fit status

        self.DicoDirty = None         # shared dict with current dirty/residual image
        self.DicoImagesPSF = None     # shared dict with current PSF images
        self.DATA = None              # shared dict with current data chunk

        self.Precision=self.GD["RIME"]["Precision"]#"S"
        self.PolMode=self.GD["RIME"]["PolMode"]
        self.PSFFacets = self.GD["Facets"]["PSFFacets"]
        self.HasDeconvolved=False
        self.Parallel = self.GD["Parallel"]["NCPU"] != 1
        self.ModConstructor = ClassModModelMachine(self.GD)

        self.PredictMode = self.GD["RIME"]["ForwardMode"]

        #self.PNGDir="%s.png"%self.BaseName
        #os.system("mkdir -p %s"%self.PNGDir)
        #os.system("rm %s/*.png 2> /dev/null"%self.PNGDir)

        # Oleg's "new" interface: set up which output images will be generated
        # --SaveImages abc means save defaults plus abc
        # --SaveOnly abc means only save abc
        # --SaveImages all means save all
        saveimages = self.GD["Output"]["Also"]
        saveonly = self.GD["Output"]["Images"]
        savecubes = self.GD["Output"]["Cubes"]
        allchars = set([chr(x) for x in range(128)])
        if saveimages.lower() == "all" or saveonly.lower() == "all":
            self._saveims = allchars
        else:
            self._saveims = set(saveimages) | set(saveonly)
        self._savecubes = allchars if savecubes.lower() == "all" else set(savecubes)

        self.do_stokes_residue = (self.GD["Output"]["StokesResidues"] != self.GD["RIME"]["PolMode"] and
                                  ("r" in self._saveims or "r" in self._savecubes or
                                   "R" in self._saveims or "R" in self._savecubes))
        ## disabling this, as it doesn't play nice with in-place FFTs
        # self._save_intermediate_grids = self.GD["Debug"]["SaveIntermediateDirtyImages"]

        APP.registerJobHandlers(self)

    def Init(self):
        DC = self.GD
        mslist = ClassMS.expandMSList(DC["Data"]["MS"],
                                      defaultDDID=DC["Selection"]["DDID"],
                                      defaultField=DC["Selection"]["Field"])
        AsyncProcessPool.init(ncpu=self.GD["Parallel"]["NCPU"],
                              affinity=self.GD["Parallel"]["Affinity"],
                              parent_affinity=self.GD["Parallel"]["MainProcessAffinity"],
                              verbose=self.GD["Debug"]["APPVerbose"],
                              pause_on_start=self.GD["Debug"]["PauseWorkers"])

        self.VS = ClassVisServer.ClassVisServer(mslist,ColName=self.do_readcol and DC["Data"]["ColName"],
                                                TChunkSize=DC["Data"]["ChunkHours"],
                                                GD=self.GD)

        self.NMajor=self.GD["Deconv"]["MaxMajorIter"]
        del(self.GD["Deconv"]["MaxMajorIter"])
        # If we do the deconvolution construct a model according to what is in MinorCycleConfig
        MinorCycleConfig=dict(self.GD["Deconv"])
        MinorCycleConfig["NCPU"] = self.GD["Parallel"]["NCPU"]
        MinorCycleConfig["NBand"]=MinorCycleConfig["NFreqBands"]=self.VS.NFreqBands
        MinorCycleConfig["GD"] = self.GD
        MinorCycleConfig["ImagePolDescriptor"] = self.VS.StokesConverter.RequiredStokesProducts()

        SubstractModel=self.GD["Predict"]["InitDicoModel"]
        DoSub=(SubstractModel!="")&(SubstractModel is not None)
        if DoSub:
            print>>log, ModColor.Str("Initialise sky model using %s"%SubstractModel,col="blue")
            ModelMachine = self.ModConstructor.GiveInitialisedMMFromFile(SubstractModel)
            modeltype = ModelMachine.DicoSMStacked["Type"]
            if modeltype == "GA":
                modeltype = "SSD"
            elif modeltype == "MSMF":
                modeltype = "HMP"
            if self.GD["Deconv"]["Mode"] != modeltype:
                raise NotImplementedError("You want to use different minor cycle and IniDicoModel types [%s vs %s]"\
                                          %(self.GD["Deconv"]["Mode"], modeltype))

            if ModelMachine.RefFreq!=self.VS.RefFreq:
                print>>log, ModColor.Str("Taking reference frequency from the model machine %f MHz (instead of %f MHz from the data)"%(ModelMachine.RefFreq/1e6,self.VS.RefFreq/1e6))
            self.RefFreq=self.VS.RefFreq=ModelMachine.RefFreq

            if self.BaseName==self.GD["Predict"]["InitDicoModel"][0:-10]:
                self.BaseName+=".continue"
                self.DicoModelName="%s.DicoModel"%self.BaseName
                self.DicoMetroModelName="%s.Metro.DicoModel"%self.BaseName
            self.DoDirtySub=1
            # enable that to be able to restore even if we don't deconvolve
            self.HasDeconvolved=True
        else:
            ModelMachine = self.ModConstructor.GiveMM(Mode=self.GD["Deconv"]["Mode"])
            ModelMachine.setRefFreq(self.VS.RefFreq)
            self.RefFreq=self.VS.RefFreq
            self.DoDirtySub=0

        self.ModelMachine=ModelMachine



        self.ImageNoiseMachine=ClassImageNoiseMachine.ClassImageNoiseMachine(self.GD,self.ModelMachine,
                                                                        DegridFreqs=self.VS.FreqBandChannelsDegrid[0],
                                                                        GridFreqs=self.VS.FreqBandCenters,
                                                                        MainCache=self.VS.maincache)
        self.MaskMachine=ClassMaskMachine.ClassMaskMachine(self.GD)
        self.MaskMachine.setImageNoiseMachine(self.ImageNoiseMachine)

        MinorCycleConfig["RefFreq"] = self.RefFreq
        MinorCycleConfig["ModelMachine"] = ModelMachine


        if self.do_deconvolve:
            # Specify which deconvolution algorithm to use
            if self.GD["Deconv"]["Mode"] == "HMP":
                if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                    raise NotImplementedError("Multi-polarization CLEAN is not supported in MSMF")
                from DDFacet.Imager.MSMF import ClassImageDeconvMachineMSMF
                self.DeconvMachine=ClassImageDeconvMachineMSMF.ClassImageDeconvMachine(MainCache=self.VS.maincache, **MinorCycleConfig)
                print>>log,"Using MSMF algorithm"
            elif self.GD["Deconv"]["Mode"]=="SSD":
                if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                    raise NotImplementedError("Multi-polarization is not supported in SSD")
                from DDFacet.Imager.SSD import ClassImageDeconvMachineSSD
                self.DeconvMachine=ClassImageDeconvMachineSSD.ClassImageDeconvMachine(MainCache=self.VS.maincache, **MinorCycleConfig)
                print>>log,"Using SSD with %s Minor Cycle algorithm"%self.GD["SSDClean"]["IslandDeconvMode"]
            elif self.GD["Deconv"]["Mode"] == "Hogbom":
                if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                    raise NotImplementedError("Multi-polarization CLEAN is not supported in Hogbom")
                from DDFacet.Imager.HOGBOM import ClassImageDeconvMachineHogbom
                self.DeconvMachine=ClassImageDeconvMachineHogbom.ClassImageDeconvMachine(**MinorCycleConfig)
                print>>log,"Using Hogbom algorithm"
            elif self.GD["Deconv"]["Mode"]=="MORESANE":
                if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                    raise NotImplementedError("Multi-polarization is not supported in MORESANE")
                from DDFacet.Imager.MORESANE import ClassImageDeconvMachineMoresane
                self.DeconvMachine=ClassImageDeconvMachineMoresane.ClassImageDeconvMachine(MainCache=self.VS.maincache, **MinorCycleConfig)
                print>>log,"Using MORESANE algorithm"
            elif self.GD["Deconv"]["Mode"]=="MUFFIN":
                if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                    raise NotImplementedError("Multi-polarization is not supported in MORESANE")
                from DDFacet.Imager.MUFFIN import ClassImageDeconvMachineMUFFIN
                self.DeconvMachine=ClassImageDeconvMachineMUFFIN.ClassImageDeconvMachine(MainCache=self.VS.maincache, **MinorCycleConfig)
                print>>log,"Using MUFFIN algorithm"
            else:
                raise NotImplementedError("Unknown --Deconvolution-Mode setting '%s'" % self.GD["Deconv"]["Mode"])
            self.DeconvMachine.setMaskMachine(self.MaskMachine)
        self.CreateFacetMachines()
        self.VS.setFacetMachine(self.FacetMachine or self.FacetMachinePSF)

        self.DoSmoothBeam=(self.GD["Beam"]["Smooth"] and self.GD["Beam"]["Model"]) and self.FacetMachine is not None
        if self.DoSmoothBeam:
            AverageBeamMachine=ClassBeamMean.ClassBeamMean(self.VS)
            self.FacetMachine.setAverageBeamMachine(AverageBeamMachine)
            self.StokesFacetMachine and self.StokesFacetMachine.setAverageBeamMachine(AverageBeamMachine)
        # tell VisServer to not load weights
        if self.do_predict_only:
            self.VS.IgnoreWeights()

        # all internal state initialized -- start the worker threads
        APP.startWorkers()
        # and proceed with background tasks
        self.VS.CalcWeightsBackground()
        self.FacetMachine and self.FacetMachine.initCFInBackground()
        # FacetMachinePSF will skip CF init if they match those of FacetMachine
        if self.FacetMachinePSF is not None:
            self.FacetMachinePSF.initCFInBackground(other_fm=self.FacetMachine)

    def CreateFacetMachines (self):
        """Creates FacetMachines for data and/or PSF"""
        self.StokesFacetMachine = self.FacetMachine = self.FacetMachinePSF = None
        MainFacetOptions = self.GiveMainFacetOptions()
        if self.do_stokes_residue:
            self.StokesFacetMachine = ClassFacetMachine(self.VS,
                                                        self.GD,
                                                        Precision=self.Precision,
                                                        PolMode=self.GD["Output"]["StokesResidues"],
                                                        custom_id="STOKESFM")
            self.StokesFacetMachine.appendMainField(ImageName="%s.image"%self.BaseName,**MainFacetOptions)
            self.StokesFacetMachine.Init()

        if self.do_data:
            self.FacetMachine = ClassFacetMachine(self.VS, self.GD,
                                                Precision=self.Precision, PolMode=self.PolMode)
            self.FacetMachine.appendMainField(ImageName="%s.image"%self.BaseName,**MainFacetOptions)
            self.FacetMachine.Init()
        if self.do_psf:
            if self.PSFFacets:
                print>> log, "the PSFFacets version is currently not supported, using 0 (i.e. same facets as image)"
                self.PSFFacets = 0
            oversize = self.GD["Facets"]["PSFOversize"] or 1
            if self.PSFFacets:
                MainFacetOptions["NFacets"] = self.PSFFacets
                print>> log, "using %d facets to compute the PSF" % self.PSFFacets
                if self.PSFFacets == 1:
                    oversize = 1
                    print>> log, "PSFFacets=1 implies PSFOversize=1"
            print>> log, "using PSFOversize=%.2f" % oversize
            self.FacetMachinePSF = ClassFacetMachine(self.VS, self.GD,
                                                Precision=self.Precision, PolMode=self.PolMode,
                                                DoPSF=True, Oversize=oversize)
            self.FacetMachinePSF.appendMainField(ImageName="%s.psf" % self.BaseName, **MainFacetOptions)
            self.FacetMachinePSF.Init()

        self.CellArcSec = (self.FacetMachine or self.FacetMachinePSF).Cell
        self.CellSizeRad = (self.CellArcSec/3600.)*np.pi/180

    def GiveMainFacetOptions(self):
        MainFacetOptions=self.GD["Image"].copy()
        MainFacetOptions.update(self.GD["CF"].copy())
        MainFacetOptions.update(self.GD["Image"].copy())
        MainFacetOptions.update(self.GD["Facets"].copy())
        MainFacetOptions.update(self.GD["RIME"].copy())
        MainFacetOptions.update(self.GD["Weight"].copy())
        del(MainFacetOptions['Precision'],
            MainFacetOptions['PolMode'],MainFacetOptions['Mode'],MainFacetOptions['Robust'])
        return MainFacetOptions

    def _createDirtyPSFCacheKey(self, sparsify=0):
        """Creates cache key used for Dirty and PSF caches"""
        key = dict([("MSNames", [ms.MSName for ms in self.VS.ListMS])] +
                    [(section, self.GD[section]) for section in
                     "Data", "Beam", "Selection",
                     "Freq", "Image", "Comp",
                     "CF", "RIME","Facets","Weight","DDESolutions"]+
                   [("InitDicoModel",self.GD["Predict"]["InitDicoModel"])]
               )

        key["Comp"]["Sparsification"] = sparsify
        return key

    def _checkForCachedPSF (self, sparsify, key=None):
        mode = self.GD["Cache"]["PSF"]
        if mode in (0, False, None, 'off'):
            return None, False, False
        elif mode == 'reset':
            print>> log, ModColor.Str("Forcing to reset the cached PSF image", col="red")
            cachepath, valid = self.VS.maincache.checkCache("PSF", key or self._createDirtyPSFCacheKey(sparsify),
                                                                      reset=True)
            writecache = True
        elif mode in (1, True, 'auto'):
            cachepath, valid = self.VS.maincache.checkCache("PSF", key or self._createDirtyPSFCacheKey(sparsify))
            writecache = not valid
        elif mode == 'force':
            cachepath = self.VS.maincache.getElementPath("PSF")
            valid = os.path.exists(cachepath)
            print>> log, ModColor.Str("Forcing to read the cached PSF", col="red")
            writecache = False
        else:
            raise ValueError("unknown --Cache-PSF setting %s"%self.GD["Cache"]["PSF"])
        return cachepath, valid, writecache

    def _checkForCachedDirty (self, sparsify, key=None):
        mode = self.GD["Cache"]["Dirty"]
        if mode in (0, False, None, 'off'):
            cachepath, valid = None, False
            writecache = False
        elif mode == 'reset':
            print>> log, ModColor.Str("Forcing to reset the cached dirty image", col="red")
            cachepath, valid = self.VS.maincache.checkCache("Dirty", key or self._createDirtyDirtyCacheKey(sparsify),
                                                                      reset=True)
            writecache = True
        elif mode in (1, True, 'auto'):
            cachepath, valid = self.VS.maincache.checkCache("Dirty", key or self._createDirtyDirtyCacheKey(sparsify))
            writecache = not valid
        elif mode == 'forcedirty':
            cachepath = self.VS.maincache.getElementPath("Dirty")
            valid = os.path.exists(cachepath)
            if not valid:
                print>> log, ModColor.Str("Can't force-read cached dirty %s: does not exist" % cachepath, col="red")
                raise RuntimeError("--Cache-Dirty forcedirty in effect, but no cached dirty image found")
            print>> log, ModColor.Str("Forcing reading the cached dirty image", col="red")
            writecache = False
        elif mode == 'forceresidual':
            cachepath = self.VS.maincache.getElementPath("LastResidual")
            valid = os.path.exists(cachepath)

            if not valid:
                print>> log, ModColor.Str("Can't force-read cached last residual %s: does not exist" % cachepath, col="red")
                raise RuntimeError("--Cache-Dirty forceresidual in effect, but no cached residual image found")
            print>> log, ModColor.Str("Forcing reading the cached last residual image", col="red")

            writecache = False
        else:
            raise ValueError("unknown --Cache-Dirty setting %s"%mode)
        return cachepath, valid, writecache

    def _loadCachedPSF (self, cachepath):
        self.DicoImagesPSF = shared_dict.create("FMPSF_AllImages")
        self.DicoImagesPSF.restore(cachepath)

        #DicoImagesPSF = MyPickle.FileToDicoNP(cachepath)
        #self.DicoImagesPSF = SharedDict.dict_to_shm("FMPSF_AllImages",DicoImagesPSF)
        #del(DicoImagesPSF)

        # if we load a cached PSF, mark these as None so that we don't re-save a PSF image in _fitAndSavePSF()
        self._psfmean = self._psfcube = None
        self.FWHMBeam=self.DicoImagesPSF["FWHMBeam"]
        self.PSFGaussPars=self.DicoImagesPSF["PSFGaussPars"]
        self.PSFSidelobes=self.DicoImagesPSF["PSFSidelobes"]
        (self.FWHMBeamAvg, self.PSFGaussParsAvg, self.PSFSidelobesAvg)=self.DicoImagesPSF["EstimatesAvgPSF"]

        # #########################"
        # Needed if cached PSF is there but --Output-RestoringBeam set differently
        forced_beam=self.GD["Output"]["RestoringBeam"]
        if forced_beam is not None:
            if isinstance(forced_beam,float) or isinstance(forced_beam,int):
                forced_beam=[float(forced_beam),float(forced_beam),0]
            elif len(forced_beam)==1:
                forced_beam=[forced_beam[0],forced_beam[0],0]
            f_beam=(forced_beam[0]/3600.0,forced_beam[1]/3600.0,forced_beam[2])
            FWHMFact = 2. * np.sqrt(2. * np.log(2.))
            f_gau=(np.deg2rad(f_beam[0])/FWHMFact,np.deg2rad(f_beam[1])/FWHMFact,np.deg2rad(f_beam[2]))
            print>>log, 'Will use user-specified beam: bmaj=%f, bmin=%f, bpa=%f degrees' % f_beam
            beam, gausspars = f_beam, f_gau
            self.FWHMBeamAvg, self.PSFGaussParsAvg = beam, gausspars
        # #########################"

        self.HasFittedPSFBeam=True



    def _finalizeComputedPSF (self, FacetMachinePSF, cachepath=None):
        self.DicoImagesPSF = FacetMachinePSF.FacetsToIm(NormJones=True)
        FacetMachinePSF.releaseGrids()
        self._psfmean, self._psfcube = self.DicoImagesPSF["MeanImage"], self.DicoImagesPSF["ImageCube"]  # this is only for the casa image saving
        self.HasFittedPSFBeam = False
        self.fit_stat = self.FitPSF()
        if cachepath:
            try:
                self.DicoImagesPSF["FWHMBeam"]=self.FWHMBeam
                self.DicoImagesPSF["PSFGaussPars"]=self.PSFGaussPars
                self.DicoImagesPSF["PSFSidelobes"]=self.PSFSidelobes
                self.DicoImagesPSF["EstimatesAvgPSF"]=(self.FWHMBeamAvg, self.PSFGaussParsAvg, self.PSFSidelobesAvg)
                #cPickle.dump(self.DicoImagesPSF, file(self._psf_cachepath, 'w'), 2)
                self.DicoImagesPSF.save(cachepath)
                MyPickle.DicoNPToFile(self.DicoImagesPSF,"%s.DicoPickle"%cachepath)
                self.VS.maincache.saveCache("PSF")
                if self.fit_stat is not None:
                    raise self.fit_stat #delay fitting errors
            except:
                print>> log, traceback.format_exc()
                print>> log, ModColor.Str(
                    "WARNING: PSF cache could not be written, see error report above. Proceeding anyway.")

    def _fitAndSavePSF (self, FacetMachinePSF, save=True, cycle=None):
        if not self.HasFittedPSFBeam:
            self.fit_stat = self.FitPSF()
        if save and self._psfmean is not None:
            cycle_label = ".%02d"%cycle if cycle else ""
            if "P" in self._saveims or "p" in self._saveims:
                FacetMachinePSF.ToCasaImage(self._psfmean, ImageName="%s%s.psf" % (self.BaseName, cycle_label),
                                            Fits=True, beam=self.FWHMBeamAvg,
                                            Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            if "P" in self._savecubes or "p" in self._savecubes:
                FacetMachinePSF.ToCasaImage(self._psfcube,
                                              ImageName="%s%s.cube.psf" % (self.BaseName, cycle_label),
                                              Fits=True, beam=self.FWHMBeamAvg, Freqs=self.VS.FreqBandCenters,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        try:
            if self.fit_stat is not None:
                raise self.fit_stat
        except:
            print>> log, traceback.format_exc()
            raise RuntimeError("there was an error fitting the PSF. Something is really wrong with the data?")

    def MakePSF(self, sparsify=0):
        """
        Generates PSF.

        Args:
            sparsify: sparsification factor applied to data. 0 means calculate the most precise dirty possible.
        """
        if self.DicoImagesPSF is not None:
            return
 
        cachepath, valid, writecache = self._checkForCachedPSF(sparsify)


        if valid:
            print>>log, ModColor.Str("============================ Loading cached PSF ==========================")
            print>> log, "found valid cached PSF in %s" % cachepath
            if not self.GD["Cache"]["PSF"].startswith("force"):
                print>>log, ModColor.Str("As near as we can tell, we can reuse this cache because it was produced")
                print>>log, ModColor.Str("with the same set of relevant DDFacet settings. If you think this is in error,")
                print>>log, ModColor.Str("or if your MS has changed, please remove the cache, or run with --Cache-PSF reset.")
            self._loadCachedPSF(cachepath)
        else:
            print>>log, ModColor.Str("=============================== Making PSF ===============================")

            # tell the I/O thread to go load the first chunk
            self.VS.ReInitChunkCount()
            self.VS.startChunkLoadInBackground()

            while True:
                # note that collectLoadedChunk() will destroy the current DATA dict, so we must make sure
                # the gridding jobs of the previous chunk are finished
                self.FacetMachinePSF.collectGriddingResults()
                # Polarization psfs is not going to be supported. We can only make dirty maps
                if self.VS.StokesConverter.RequiredStokesProducts() != ['I']:
                    raise RuntimeError("Unsupported: Polarization PSF creation is not defined")
                # get loaded chunk from I/O thread, schedule next chunk
                # self.VS.startChunkLoadInBackground()
                DATA = self.VS.collectLoadedChunk(start_next=True)
                if type(DATA) is str:
                    print>> log, ModColor.Str("no more data: %s" % DATA, col="red")
                    break
                # None weights indicates an all-flagged chunk: go on to the next chunk
                if DATA["Weights"] is None:
                    continue

                self.FacetMachinePSF.putChunkInBackground(DATA)

            self._finalizeComputedPSF(self.FacetMachinePSF,cachepath=writecache and cachepath)

        self._fitAndSavePSF(self.FacetMachinePSF)


    def GiveDirty(self, psf=False, sparsify=0, last_cycle=False):
        """
        Generates dirty image (& PSF)

        Args:
            psf: if True, PSF is also generated
            sparsify: sparsification factor applied to data. 0 means calculate the most precise dirty possible.
        """
        if sparsify <= 1:
            sparsify = 0
        cache_key = self._createDirtyPSFCacheKey()
        dirty_cachepath, dirty_valid, dirty_writecache = self._checkForCachedDirty(sparsify, key=cache_key)
        if psf:
            psf_cachepath, psf_valid, psf_writecache = self._checkForCachedPSF(sparsify, key=cache_key)
        else:
            psf_valid = psf_writecache = False

        current_model_freqs = np.array([])
        ModelImage = None
        # load from cache
        if dirty_valid:
            if self.GD["Cache"]["Dirty"] == "forceresidual":
                print>>log, ModColor.Str("============================ Loading last residual image ========================")
                print>>log, "found valid cached residual image in %s"%dirty_cachepath
            else:
                print>>log, ModColor.Str("============================ Loading cached dirty image =========================")
                print>> log, "found valid cached residual image in %s" % dirty_cachepath
            if type(self.GD["Cache"]["Dirty"]) is not str or not self.GD["Cache"]["Dirty"].startswith("force"):
                print>>log, ModColor.Str("As near as we can tell, we can reuse this cache because it was produced")
                print>>log, ModColor.Str("with the same set of relevant DDFacet settings. If you think this is in error,")
                print>>log, ModColor.Str("or if your MS has changed, please remove the cache, or run with --Cache-Dirty reset.")

            self.DicoDirty = shared_dict.create("FM_AllImages")
            self.DicoDirty.restore(dirty_cachepath)


            if self.DicoDirty["JonesNorm"] is not None:
                self.FacetMachine.setNormImages(self.DicoDirty)
                self.MeanJonesNorm = self.FacetMachine.MeanJonesNorm
                self.JonesNorm = self.FacetMachine.JonesNorm
                if self.FacetMachinePSF is not None:
                    self.FacetMachinePSF.setNormImages(self.DicoDirty)
            elif self.DicoImagesPSF["JonesNorm"] is not None:
                self.FacetMachine.setNormImages(self.DicoImagesPSF)
                self.FacetMachinePSF.setNormImages(self.DicoImagesPSF)
                self.MeanJonesNorm = self.FacetMachinePSF.MeanJonesNorm
                self.JonesNorm = self.FacetMachinePSF.JonesNorm
            else:
                self.MeanJonesNorm = None
                self.JonesNorm = None

            if self.DicoDirty.get("LastMask") is not None and self.GD["Mask"]["Auto"]:
                self.MaskMachine.joinExternalMask(self.DicoDirty["LastMask"])

        if psf_valid:
            print>>log, ModColor.Str("============================ Loading cached PSF =================================")
            print>> log, "found valid cached PSF in %s" % psf_cachepath
            if type(self.GD["Cache"]["PSF"]) is not str or not self.GD["Cache"]["PSF"].startswith("force"):
                print>>log, ModColor.Str("As near as we can tell, we can reuse this cache because it was produced")
                print>>log, ModColor.Str("with the same set of relevant DDFacet settings. If you think this is in error,")
                print>>log, ModColor.Str("or if your MS has changed, please remove the cache, or run with --Cache-PSF reset.")
            self._loadCachedPSF(psf_cachepath)

        # run FM loop if need to generate either
        if not (dirty_valid and psf_valid):
            print>>log, ModColor.Str("============================== Making Dirty Image and/or PSF ====================")
            # tell the I/O thread to go load the first chunk
            self.VS.ReInitChunkCount()
            self.VS.startChunkLoadInBackground()
            if not dirty_valid:
                self.FacetMachine.ReinitDirty()
            if psf and not psf_valid and self.FacetMachinePSF is not None:
                self.FacetMachinePSF.ReinitDirty()


            iloop = 0
            while True:
                # note that collectLoadedChunk() will destroy the current DATA dict, so we must make sure
                # the gridding jobs of the previous chunk are finished
                if not dirty_valid:
                    self.FacetMachine.collectGriddingResults()
                if psf and not psf_valid and self.FacetMachinePSF is not None:
                    self.FacetMachinePSF.collectGriddingResults()

                # get loaded chunk from I/O thread, schedule next chunk
                # self.VS.startChunkLoadInBackground()
                DATA = self.VS.collectLoadedChunk(start_next=True)

                if type(DATA) is str:
                    print>>log,ModColor.Str("no more data: %s"%DATA, col="red")
                    break

                # Allow for predict mode when a residual only is computed
                predict_colname = None
                if self.GD["Output"]["Mode"]=="Dirty":
                    predict_colname = self.GD["Predict"]["ColName"]
                if self.DoDirtySub and predict_colname:
                    predict = DATA.addSharedArray("predict", DATA["datashape"], DATA["datatype"])
                    visdata = DATA["data"]
                    np.copyto(predict, visdata)


                # None weights indicates an all-flagged chunk: go on to the next chunk
                if DATA["Weights"] is None:
                    continue
                print>>log,"sparsify %f"%sparsify
                self.FacetMachine.applySparsification(DATA, sparsify)

                if self.DoDirtySub:
                    ## redo model image if needed
                    model_freqs = DATA["FreqMappingDegrid"]
                    if not np.array_equal(model_freqs, current_model_freqs):
                        ModelImage = self.FacetMachine.setModelImage(self.ModelMachine.GiveModelImage(model_freqs))
                        self.FacetMachine.ToCasaImage(ModelImage,ImageName="%s.model"%(self.BaseName),
                                                      Fits=True,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                        current_model_freqs = model_freqs
                        print>> log, "model image @%s MHz (min,max) = (%f, %f)" % (
                        str(model_freqs / 1e6), ModelImage.min(), ModelImage.max())
                    else:
                        print>> log, "reusing model image from previous chunk"
                    if not dirty_valid:
                        self.FacetMachine.getChunkInBackground(DATA)
                        self.FacetMachine.collectDegriddingResults()

                    if predict_colname:
                        predict -= visdata
                        # schedule jobs for saving visibilities, then start reading next chunk (both are on io queue)
                        self.VS.startVisPutColumnInBackground(DATA, "predict", predict_colname, likecol=self.GD["Data"]["ColName"])
                        

                        
                # crude but we need it here, since FacetMachine computes/loads CFs, which FacetMachinePSF uses.
                # so even if we're not using FM to make a dirty, we still need this call to make sure the CFs come in.
                self.FacetMachine.awaitInitCompletion()

                # Stacks average beam if not computed
                self.FacetMachine.StackAverageBeam(DATA)

                if not dirty_valid:
                    # commented out. @cyriltasse to uncomment when fixed
                    self.FacetMachine.putChunkInBackground(DATA)


                if psf and not psf_valid and self.FacetMachinePSF is not None:
                    self.FacetMachinePSF.putChunkInBackground(DATA)
                ## disabled this, doesn't like in-place FFTs
                # # collect intermediate grids, if asked to
                # if self._save_intermediate_grids:
                #     self.DicoDirty=self.FacetMachine.FacetsToIm(NormJones=True)
                #     self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.dirty.%d."%(self.BaseName,iloop),
                #                                   Fits=True,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                #     if 'g' in self._savecubes:
                #         self.FacetMachine.ToCasaImage(self.DicoDirty["ImageCube"],ImageName="%s.cube.dirty.%d"%(self.BaseName,iloop),
                #             Fits=True,Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                #     self.FacetMachine.JonesNorm = None
                #     self.FacetMachine.NormImage = None

                iloop += 1


            if not dirty_valid:
                # if Smooth beam enabled, either compute it from the stack, or get it from cache
                # else do nothing
                self.FacetMachine.finaliseSmoothBeam()

                # stitch facets and release grids
                self.DicoDirty = self.FacetMachine.FacetsToIm(NormJones=True)
                self.FacetMachine.releaseGrids()

                self.SaveDirtyProducts()

                # dump dirty to cache
                if dirty_writecache:
                    try:
                        #cPickle.dump(self.DicoDirty, file(cachepath, 'w'), 2)
                        self.DicoDirty.save(dirty_cachepath)
                        self.VS.maincache.saveCache("Dirty")
                    except:
                        print>> log, traceback.format_exc()
                        print>> log, ModColor.Str("WARNING: Dirty image cache could not be written, see error report above. Proceeding anyway.")

            if psf and not psf_valid:
                self._finalizeComputedPSF(self.FacetMachinePSF, psf_writecache and psf_cachepath)

        # self.SaveDirtyProducts()

        # This call needs to be here to attach the cached smooth beam to FacetMachine if it exists
        # and if dirty has been initialised from cache
        self.FacetMachine.finaliseSmoothBeam()

        # If we have used InitDicoModel to substracted to the original dirty,
        # no need to anymore in the subsequent call to GiveDirty
        self.DoDirtySub=0

        ## we get here whether we recomputed dirty/psf or not
        # finalize other PSF initialization
        if psf:
            self._fitAndSavePSF(self.FacetMachinePSF)

        return self.DicoDirty["MeanImage"]

    def SaveDirtyProducts(self):

        if "d" in self._saveims:
            self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.dirty"%self.BaseName,Fits=True,
                                          Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        if "d" in self._savecubes:
            self.FacetMachine.ToCasaImage(self.DicoDirty["ImageCube"],ImageName="%s.cube.dirty"%self.BaseName,
                                          Fits=True,Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())

        if "n" in self._saveims:
            FacetNormReShape = self.FacetMachine.getNormDict()["FacetNormReShape"]
            self.FacetMachine.ToCasaImage(FacetNormReShape,
                                          ImageName="%s.NormFacets"%self.BaseName,
                                          Fits=True)

        if self.DicoDirty["JonesNorm"] is not None:
            DirtyCorr = self.DicoDirty["ImageCube"]/np.sqrt(self.DicoDirty["JonesNorm"])
            nch,npol,nx,ny = DirtyCorr.shape
            if "D" in self._saveims:
                MeanCorr = np.mean(DirtyCorr, axis=0).reshape((1, npol, nx, ny))
                self.FacetMachine.ToCasaImage(MeanCorr,ImageName="%s.dirty.corr"%self.BaseName,Fits=True,
                                                  Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            if "D" in self._savecubes:
                self.FacetMachine.ToCasaImage(DirtyCorr,ImageName="%s.cube.dirty.corr"%self.BaseName,
                                              Fits=True,Freqs=self.VS.FreqBandCenters,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())

            self.JonesNorm = self.DicoDirty["JonesNorm"]
            self.MeanJonesNorm = np.mean(self.JonesNorm,axis=0).reshape((1,npol,nx,ny))

            if self.DoSmoothBeam and self.FacetMachine.SmoothJonesNorm is not None:
                self.FacetMachine.ToCasaImage(self.FacetMachine.MeanSmoothJonesNorm,ImageName="%s.MeanSmoothNorm"%self.BaseName,Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                self.FacetMachine.ToCasaImage(self.FacetMachine.SmoothJonesNorm,ImageName="%s.SmoothNorm"%self.BaseName,Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts(),
                                              Freqs=self.VS.FreqBandCenters)


            if "N" in self._saveims:
                self.FacetMachine.ToCasaImage(self.MeanJonesNorm,ImageName="%s.Norm"%self.BaseName,Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            if "N" in self._savecubes:
                self.FacetMachine.ToCasaImage(self.JonesNorm, ImageName="%s.cube.Norm" % self.BaseName,
                                              Fits=True, Freqs=self.VS.FreqBandCenters,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        else:
            self.MeanJonesNorm = None


    def GivePredict(self, subtract=False, from_fits=True):
        if subtract:
            print>>log, ModColor.Str("============================== Making Predict/Subtract =====================")
        else:
            print>>log, ModColor.Str("============================== Making Predict ==============================")

        if not self.GD["Predict"]["ColName"]:
            raise ValueError("--Predict-ColName must be set")
        if not self.GD["Predict"]["FromImage"] and not self.GD["Predict"]["InitDicoModel"]:
            raise ValueError("--Predict-FromImage or --Predict-InitDicoModel must be set")
	
        # tell the I/O thread to go load the first chunk
        self.VS.ReInitChunkCount()
        self.VS.startChunkLoadInBackground()

        self.FacetMachine.ReinitDirty()

        # BaseName=self.GD["Output"]["Name"]
        # #ModelMachine=ClassModelMachine(self.GD)
        # try:
        #     NormImageName="%s.NormFacets.fits"%BaseName
        #     CasaNormImage = image(NormImageName)
        #     NormImage = CasaNormImage.getdata()
        #     print NormImage.shape
        # except:
        #     NormImage = self.FacetMachine.BuildFacetNormImage()
        #     NormImage = NormImage.reshape([1,1,NormImage.shape[0],NormImage.shape[1]])
        # nch,npol,nx,_=NormImage.shape
        # for ch in range(nch):
        #     for pol in range(npol):
        #         NormImage[ch,pol]=NormImage[ch,pol].T[::-1]
        # self.FacetMachine.NormImage=NormImage.reshape((nx,nx))

        CleanMaskImage=None
        CleanMaskImageName=self.GD["Mask"]["External"]
        # if CleanMaskImageName is not None and CleanMaskImageName is not "":
        #     print>>log,ModColor.Str("Will use mask image %s for the predict"%CleanMaskImageName)
        #     CleanMaskImage = np.bool8(ClassCasaImage.FileToArray(CleanMaskImageName,True))


        modelfile = self.GD["Predict"]["FromImage"]
        # if model image is specified, we'll use that, rather than the ModelMachine
        if modelfile is not None and modelfile is not "":
            print>>log,ModColor.Str("Reading image file for the predict: %s" % modelfile)
            FixedModelImage = ClassCasaImage.FileToArray(modelfile,True)
            nch,npol,_,NPix=self.FacetMachine.OutImShape
            nchModel,npolModel,_,NPixModel=FixedModelImage.shape
            if NPixModel!=NPix:
                print>>log,ModColor.Str("Model image spatial shape does not match DDFacet settings [%i vs %i]"%(FixedModelImage.shape[-1],NPix))
                CA=ClassAdaptShape(FixedModelImage)
                FixedModelImage=CA.giveOutIm(NPix)

            if len(FixedModelImage.shape) != 4:
                raise RuntimeError("Expect FITS file with 4 axis: NX, NY, NPOL, NCH. Cannot continue.")
            nch, npol, ny, nx = FixedModelImage.shape
            if ny != nx:
                raise RuntimeError("Currently non-square images are not supported")
            npixest, _ = EstimateNpix(float(self.GD["Image"]["NPix"]), Padding=1)
            if nx != npixest:
                raise RuntimeError("Number of pixels in FITS file (%d) does not match "
                                   "image size (%d). Cannot continue." % (nx, npixest))
            if npol != 1:
                raise RuntimeError("Unsupported: Polarization prediction is not defined")
            for msi in self.VS.FreqBandChannelsDegrid:
                nband = self.GD["Freq"]["NDegridBand"] if self.GD["Freq"]["NDegridBand"] != 0 \
                                                       else len(self.VS.FreqBandChannelsDegrid[msi])
                if nch != nband:
                    raise RuntimeError("Number of predict frequency bands (%d) do not correspond to number of "
                                       "frequency bands (%d) in input FITS file. Cannot continue." % (nband, nch))
        else:
            FixedModelImage = None

        current_model_freqs = np.array([])
        ModelImage = None

        self.FacetMachine.awaitInitCompletion()
        self.FacetMachine.BuildFacetNormImage()
        while True:
            # get loaded chunk from I/O thread, schedule next chunk
            # self.VS.startChunkLoadInBackground()
            DATA = self.VS.collectLoadedChunk(start_next=True)
            if self.VS.StokesConverter.RequiredStokesProducts() != ['I']:
                raise RuntimeError("Unsupported: Polarization prediction is not defined")
            if type(DATA) is str:
                print>> log, ModColor.Str("no more data: %s" % DATA, col="red")
                break
            # None weights indicates an all-flagged chunk: go on to the next chunk
            if DATA["Weights"] is None:
                continue
            # insert null array for predict
            predict = DATA.addSharedArray("data", DATA["datashape"], DATA["datatype"])


            model_freqs = DATA["FreqMappingDegrid"]
            if FixedModelImage is None:
                ## redo model image if needed
                if not np.array_equal(model_freqs, current_model_freqs):
                    ModelImage = self.FacetMachine.setModelImage(self.ModelMachine.GiveModelImage(model_freqs))
                    current_model_freqs = model_freqs
                    print>> log, "model image @%s MHz (min,max) = (%f, %f)" % (
                    str(model_freqs / 1e6), ModelImage.min(), ModelImage.max())
                else:
                    print>> log, "reusing model image from previous chunk"
            else:
                if ModelImage is None:
                    nch=model_freqs.size
                    nchModel=FixedModelImage.shape[0]
                    ThisChFixedModelImage=FixedModelImage # so it's initialized
                    if nch!=nchModel:
                        print>>log,ModColor.Str("Model image spectral shape does not match DDFacet settings [%i vs %i]"%(nchModel,nch))
                        if nchModel>nch:
                            print>>log,ModColor.Str("  taking the model's %i first channels only"%(nch))
                            ThisChFixedModelImage=FixedModelImage[0:nch].copy()
                        else:
                            print>>log,ModColor.Str("  Replicating %i-times the 1st channel"%(nch))
                            ThisChFixedModelImage=FixedModelImage[0].reshape((1,npol,NPix,NPix))*np.ones((DATA["ChanMappingDegrid"].size,1,1,1))
                        self.FacetMachine.ToCasaImage(ThisChFixedModelImage,
                                                      ImageName="%s.cube.model"%(self.BaseName),
                                                      Fits=True,
                                                      Freqs=model_freqs,
                                                      Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                    ModelImage = self.FacetMachine.setModelImage(ThisChFixedModelImage)


            if self.GD["Predict"]["MaskSquare"]:
                # MaskInside: choose mask inside (0) or outside (1)
                # NpixInside: Size of the masking region
                MaskOutSide,NpixInside = self.GD["Predict"]["MaskSquare"]
                if MaskOutSide==0:
                    SquareMaskMode="Inside"
                elif MaskOutSide==1:
                    SquareMaskMode="Outside"
                NpixInside, _ = EstimateNpix(float(NpixInside), Padding=1)
                print>>log,"  Zeroing model %s square [%i pixels]"%(SquareMaskMode,NpixInside)
                dn=NpixInside/2
                n=self.FacetMachine.Npix
                InSquare=np.zeros(ModelImage.shape,bool)
                InSquare[:,:,n/2-dn:n/2+dn+1,n/2-dn:n/2+dn+1]=1
                if SquareMaskMode=="Inside":
                    ModelImage[InSquare]=0
                elif SquareMaskMode=="Outside":
                    ModelImage[np.logical_not(InSquare)]=0
                #ModelImage = self.FacetMachine.setModelImage(ModelImage)

            ## OMS 16/04/17: @cyriltasse this code looks all wrong and was giving me errors. ChanMappingDegrid has size equal to the
            ## number of channels. I guess this is meant for the case where we predict from a FixedModelImage
            ## rather than a DicoModel, but in this case we probably need to recalculate ChanMappingDegrid specifically
            ## based on the FixedModelImage freq axis? Disabling for now.
            # if ModelImage.shape[0]!=DATA["ChanMappingDegrid"].size:
            #     print>>log, "The image model channels and targetted degridded visibilities channels have different sizes (%i vs %i respectively)"%(
            #         ModelImage.shape[0], DATA["ChanMappingDegrid"].size)
            #     if ModelImage.shape[0]==1:
            #         print>>log, " Matching freq size of model image to visibilities"
            #         ModelImage=ModelImage*np.ones((DATA["ChanMappingDegrid"].size,1,1,1))

            # if CleanMaskImage is not None:
            #     nch,npol,_,_=ModelImage.shape
            #     indZero=(CleanMaskImage[0,0]!=0)
            #     for ich in range(nch):
            #         for ipol in range(npol):
            #             ModelImage[ich,ipol][indZero]=0

            # self.FacetMachine.ToCasaImage(ModelImage,ImageName="%s.modelPredict"%self.BaseName,Fits=True,
            #                               Stokes=self.VS.StokesConverter.RequiredStokesProducts())


            if self.PredictMode == "BDA-degrid" or self.PredictMode == "Classic":  # latter for backwards compatibility
                self.FacetMachine.getChunkInBackground(DATA)
            elif self.PredictMode == "Montblanc":
                from ClassMontblancMachine import ClassMontblancMachine
                model = self.ModelMachine.GiveModelList()
                mb_machine = ClassMontblancMachine(self.GD, self.FacetMachine.Npix, self.FacetMachine.CellSizeRad)
                mb_machine.getChunk(DATA, predict, model, self.VS.ListMS[DATA["iMS"]])
                mb_machine.close()
            else:
                raise ValueError("Invalid PredictMode '%s'" % self.PredictMode)
            self.FacetMachine.collectDegriddingResults()
            if not subtract:
                predict *= -1   # model was subtracted from (zero) data, so need to invert sign
            # run job in I/O thread


            self.VS.startVisPutColumnInBackground(DATA, "data", self.GD["Predict"]["ColName"], likecol=self.GD["Data"]["ColName"])
            # and wait for it to finish (we don't want DATA destroyed, which collectLoadedChunk() above will)
            self.VS.collectPutColumnResults()



            # #Convert to radians
            # ra = ra_d*np.pi/180.0
            # dec = dec_d*np.pi/180
            # #Convert to lm coords
            # l,m = self.VS.CurrentMS.radec2lm_scalar(ra,dec)
            # #Find pixels corresponding to these coords
            # #get pointing center
            # ra0 = self.VS.CurrentMS.rarad
            # dec0 = self.VS.CurrentMS.decrad
            # #Now get deltapix in radians
            #
            # print self.VS.CurrentMS.Field


    def main(self,NMajor=None):
        if NMajor is None:
            NMajor=self.NMajor

        restart_time = time.time()

        # see if we're working in "sparsification" mode
        # 'sparsify' is a list of factors. E.g. [100,10] means data in first major cycle is sparsified
        # by a factor of 100, second cycle by 10, from third cyrcle onwards is precise
        sparsify_list = self.GD["Comp"]["Sparsification"]
        if sparsify_list:
            if not isinstance(sparsify_list, list):
                sparsify_list = [ sparsify_list ]
        else:
            sparsify_list = []

        # use an approximate PSF when sparsification is above this value. 0 means never
        approximate_psf_above = self.GD["Deconv"]["ApproximatePSF"] or 999999
        print>>log,"allowing PSF approximations for sparsification > %.1f" % approximate_psf_above

        # previous_sparsify keeps track of the sparsity factor at which the dirty/PSF was computed
        # in the previous cycle. We use this to decide when to recompute the PSF.
        sparsify = previous_sparsify = sparsify_list[0] if sparsify_list else 0
        if sparsify <= 1:
            sparsify = previous_sparsify = 0
        if sparsify:
            print>> log, "applying a sparsification factor of %f to data for dirty image" % sparsify
        # if running in NMajor=0 mode, then we simply want to subtract/predict the model probably
        self.GiveDirty(psf=True, sparsify=sparsify, last_cycle=(NMajor==0))

        # Polarization clean is not going to be supported. We can only make dirty maps
        if self.VS.StokesConverter.RequiredStokesProducts() != ['I']:
            raise RuntimeError("Unsupported: Polarization cleaning is not"\
                               " supported. Maybe you meant Output-StokesResidues"\
                               " instead?")

        # if we reached a sparsification of 1, we shan't be re-making the PSF
        if not sparsify:
            self.FacetMachinePSF.releaseGrids()
            self.FacetMachinePSF = None

        # if asked to conserve memory, DeconvMachine and ImageNoiseMachine
        # will be reset and reinitialized each major cycle (since they may have memory-hungry HDM dicts
        # inside them)
        conserve_memory = self.GD["Misc"]["ConserveMemory"]

        # flags keep track of whether the machines need to be (re)initialized
        deconvmachine_init = imagenoisemachine_init = False

        continue_deconv = True

        for iMajor in range(1, NMajor+1):
            # previous minor loop indicated it has reached bottom? Break out
            if not continue_deconv:
                break

            print>>log, ModColor.Str("========================== Running major cycle %i ========================="%(iMajor-1))

            # noise mask first (this may be RAM-hungry due to HMP inside, but ClassImageNoiseMachine.giveBrutalRestored()
            # eventually Reset()s its HMP machine, releasing memory)

            # we have to give the PSF to the image-noise machine since it may have to run an HMP deconvolution
            self.ImageNoiseMachine.setPSF(self.DicoImagesPSF)
            # now update the mask - it will eventually call for ImageNoiseMachine to compute a noise image
            self.MaskMachine.updateMask(self.DicoDirty)
            if self.MaskMachine.CurrentMask is not None:
                if "k" in self._saveims:
                    self.FacetMachine.ToCasaImage(np.float32(self.MaskMachine.CurrentMask),
                                                  ImageName="%s.mask%2.2i"%(self.BaseName,iMajor),Fits=True,
                                                  Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                if "z" in self._saveims and self.ImageNoiseMachine.NoiseMap is not None:
                    self.FacetMachine.ToCasaImage(np.float32(self.ImageNoiseMachine.NoiseMapReShape),
                                                  ImageName="%s.noise%2.2i"%(self.BaseName,iMajor),Fits=True,
                                                  Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                    self.FacetMachine.ToCasaImage(np.float32(self.ImageNoiseMachine.Restored),
                                                  ImageName="%s.brutalRestored%2.2i"%(self.BaseName,iMajor),Fits=True,
                                                  Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                    self.FacetMachine.ToCasaImage(np.float32(self.ImageNoiseMachine.ModelConv),
                                                  ImageName="%s.brutalModelConv%2.2i"%(self.BaseName,iMajor),Fits=True,
                                                  Stokes=self.VS.StokesConverter.RequiredStokesProducts())

            # now, finally, initialize the deconv machine (this may also be RAM-heavy thanks to HMP,
            # so we only do this after the mask machine has done its business)

            if not deconvmachine_init:
                # Pass minor cycle specific options into Init as kwargs
                self.DeconvMachine.Init(PSFVar=self.DicoImagesPSF, PSFAve=self.PSFSidelobesAvg,
                                        approx=(sparsify > approximate_psf_above), cache=not sparsify,
                                        GridFreqs=self.VS.FreqBandCenters, DegridFreqs=self.VS.FreqBandChannelsDegrid[0],
                                        RefFreq=self.VS.RefFreq)
                deconvmachine_init = True

            # To make the package more robust against memory leaks, we restart the worker processes every now and then.
            # As a rule of thumb, we do this every major cycle, but no more often than every N minutes.
            if time.time() > restart_time + 600:
                APP.restartWorkers()
                restart_time = time.time()

            self.DeconvMachine.Update(self.DicoDirty)

            repMinor, continue_deconv, update_model = self.DeconvMachine.Deconvolve()
            try:
                self.FacetMachine.ToCasaImage(self.DeconvMachine.LabelIslandsImage,
                                              ImageName="%s.labelIslands%2.2i"%(self.BaseName,iMajor),Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                # ModelImage = self.DeconvMachine.GiveModelImage(np.array([100e6]))
                # self.FacetMachine.ToCasaImage(ModelImage,ImageName="%s.model%2.2i"%(self.BaseName,iMajor),
                #                               Fits=True)#,Freqs=current_model_freqs,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            except:
                pass


            ###
            self.ModelMachine.ToFile(self.DicoModelName)
            # ###
            model_freqs=np.array([self.RefFreq],np.float64)
            ModelImage = self.FacetMachine.setModelImage(self.DeconvMachine.GiveModelImage(model_freqs))
            # write out model image, if asked to
            current_model_freqs = model_freqs
            print>>log,"model image @%s MHz (min,max) = (%f, %f)"%(str(model_freqs/1e6),ModelImage.min(),ModelImage.max())
            if "o" in self._saveims:
                self.FacetMachine.ToCasaImage(ModelImage, ImageName="%s.model%2.2i" % (self.BaseName, iMajor),
                                              Fits=True, Freqs=current_model_freqs,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            # stop
            # ###

            ## returned with nothing done in minor cycle? Break out
            if not update_model or iMajor == NMajor:
                continue_deconv = False
                print>> log, "This is the last major cycle"
            else:
                print>> log, "Finished Deconvolving for this major cycle... Going back to visibility space."
            predict_colname = not continue_deconv and self.GD["Predict"]["ColName"]

            # in the meantime, tell the I/O thread to go reload the first data chunk
            self.VS.ReInitChunkCount()
            self.VS.startChunkLoadInBackground()

            # determine whether data still needs to be sparsified
            # last major cycle is always done at full precision, but also if the sparsification_list ends we go to full precision
            if not continue_deconv or iMajor >= len(sparsify_list):
                sparsify = 0
            else:
                sparsify = sparsify_list[iMajor]
                if sparsify <= 1:
                    sparsify = 0

            # recompute PSF in sparsification mode, or the first time we go from sparsified to full precision,
            # unless this is the last major cycle, in which case we never recompute the PSF
            do_psf = (sparsify or previous_sparsify) and continue_deconv

            if self.DicoDirty is not None:
                self.DicoDirty.delete()
                self.DicoDirty = None
            self.FacetMachine.ReinitDirty()
            if sparsify or previous_sparsify:
                print>>log, "applying a sparsification factor of %f (was %f in previous cycle)" % (sparsify, previous_sparsify)
            if do_psf:
                print>>log, "the PSF will be recomputed"
                self.FacetMachinePSF.ReinitDirty()
            if do_psf or not continue_deconv:
                if self.DicoImagesPSF is not None:
                    self.DicoImagesPSF.delete()
                    self.DicoImagesPSF = None


            # release PSFs and memory in DeconvMachine, if it's going to be reinitialized with a new PSF anyway, or if
            # we're not going to use it again, or if we're asked to conserve memory
#            import pdb;
#            pdb.set_trace()
            if do_psf or not continue_deconv or conserve_memory:
                # if DeconvMachine has a reset method, use it
                if hasattr(self.DeconvMachine, 'Reset'):
                    self.DeconvMachine.Reset()
                    deconvmachine_init = False

            previous_sparsify = sparsify

            current_model_freqs = np.array([])
            ModelImage = None
            HasWrittenModel=False
            while True:
                # note that collectLoadedChunk() will destroy the current DATA dict, so we must make sure
                # the gridding jobs of the previous chunk are finished
                self.FacetMachine.collectGriddingResults()
                if self.FacetMachinePSF is not None:
                    self.FacetMachinePSF.collectGriddingResults()
                self.VS.collectPutColumnResults()  # if these were going on
                # get loaded chunk from I/O thread, schedule next chunk
                # note that if we're writing predict data out, DON'T schedule until we're done writing this one
                DATA = self.VS.collectLoadedChunk(start_next=not predict_colname)
                if type(DATA) is str:
                    print>>log,ModColor.Str("no more data: %s"%DATA, col="red")
                    break
                # None weights indicates an all-flagged chunk: go on to the next chunk
                if DATA["Weights"] is None:
                    continue
                visdata = DATA["data"]
                if predict_colname:
                    predict = DATA.addSharedArray("predict", DATA["datashape"], DATA["datatype"])
                    np.copyto(predict, visdata)
                # sparsify the data according to current levels
                self.FacetMachine.applySparsification(DATA, sparsify)
                ## redo model image if needed
                model_freqs = DATA["FreqMappingDegrid"]
                if not np.array_equal(model_freqs, current_model_freqs):
                    ModelImage = self.FacetMachine.setModelImage(self.DeconvMachine.GiveModelImage(model_freqs))
                    # write out model image, if asked to
                    current_model_freqs = model_freqs
                    print>>log,"model image @%s MHz (min,max) = (%f, %f)"%(str(model_freqs/1e6),ModelImage.min(),ModelImage.max())
                    if "o" in self._saveims and not HasWrittenModel:
                        self.FacetMachine.ToCasaImage(ModelImage, ImageName="%s.model%2.2i" % (self.BaseName, iMajor),
                                                      Fits=True, Freqs=current_model_freqs,
                                                      Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                        HasWrittenModel=True
                else:
                    print>>log,"reusing model image from previous chunk"

                ## @cyriltasse added this but it seems unnecessary
                # if "o" in self._saveims:
                #     # self.FacetMachine.ToCasaImage(ModelImage,ImageName="%s.model%2.2i"%(self.BaseName,iMajor),
                #     #     Fits=True,Freqs=current_model_freqs,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                #     nf,npol,nx,nx=ModelImage.shape
                #     ModelImageAvg=np.mean(ModelImage,axis=0).reshape((1,npol,nx,nx))
                #
                #     self.FacetMachine.ToCasaImage(ModelImageAvg,ImageName="%s.model%2.2i"%(self.BaseName,iMajor),
                #                                   Fits=True)#,Freqs=current_model_freqs,Stokes=self.VS.StokesConverter.RequiredStokesProducts())


                if predict_colname:
                    print>>log,"last major cycle: model visibilities will be stored to %s"%predict_colname

                if self.PredictMode == "BDA-degrid" or self.PredictMode == "DeGridder":
                    self.FacetMachine.getChunkInBackground(DATA)
                elif self.PredictMode == "Montblanc":
                    from ClassMontblancMachine import ClassMontblancMachine
                    model = self.ModelMachine.GiveModelList()
                    mb_machine = ClassMontblancMachine(self.GD, self.FacetMachine.Npix, self.FacetMachine.CellSizeRad)
                    mb_machine.getChunk(DATA, DATA["data"], model, self.VS.ListMS[DATA["iMS"]])
                    mb_machine.close()
                else:
                    raise ValueError("Invalid PredictMode '%s'" % self.PredictMode)

                if predict_colname:
                    self.FacetMachine.collectDegriddingResults()
                    # predict had original data -- subtract residuals to arrive at model
                    predict -= visdata
                    # schedule jobs for saving visibilities, then start reading next chunk (both are on io queue)
                    self.VS.startVisPutColumnInBackground(DATA, "predict", predict_colname, likecol=self.GD["Data"]["ColName"])
                    self.VS.startChunkLoadInBackground()

                # Stacks average beam if not computed
                self.FacetMachine.StackAverageBeam(DATA)

                self.FacetMachine.putChunkInBackground(DATA)
                if do_psf:
                    self.FacetMachinePSF.putChunkInBackground(DATA)

            # wait for gridding to finish
            self.FacetMachine.collectGriddingResults()
            self.VS.collectPutColumnResults()  # if these were going on
            # release model image from memory
            ModelImage = None
            self.FacetMachine.releaseModelImage()
            # create new residual image
            self.DicoDirty = self.FacetMachine.FacetsToIm(NormJones=True)

            self.DicoDirty["LastMask"] = self.MaskMachine.CurrentMask

            # was PSF re-generated?
            if do_psf:
                self._finalizeComputedPSF(self.FacetMachinePSF, cachepath=None)
                self._fitAndSavePSF(self.FacetMachinePSF, cycle=iMajor)
                deconvmachine_init = False  # force re-init above

            # if we reached a sparsification of 1, we shan't be re-making the PSF
            if sparsify <= 1:
                self.FacetMachinePSF = None

            # if "SmoothMeanNormImage" in self.DicoDirty.keys():
            #     self.SmoothMeanNormImage=self.DicoDirty["SmoothMeanNormImage"]

            if "e" in self._saveims:
                self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.residual%2.2i"%(self.BaseName,iMajor),
                                              Fits=True,Stokes=self.VS.StokesConverter.RequiredStokesProducts())

            # write out current model, using final or intermediate name
            if continue_deconv:
                self.DeconvMachine.ToFile("%s.%2.2i.DicoModel" % (self.BaseName, iMajor) )
            else:
                self.DeconvMachine.ToFile(self.DicoModelName)


            self.HasDeconvolved=True
            # dump dirty to cache
            if self.GD["Cache"]["LastResidual"] and self.DicoDirty is not None:
                cachepath, valid = self.VS.maincache.checkCache("LastResidual", 
                                                                dict(
                                                                    [("MSNames", [ms.MSName for ms in self.VS.ListMS])] +
                                                                    [(section, self.GD[section]) for section in "Data", "Beam", "Selection",
                                                                     "Freq", "Image", "Comp",
                                                                     "RIME","Weight","Facets",
                                                                     "DDESolutions"]
                                                                ), 
                                                                reset=False)
                try:
                    print>>log,"Saving last residual image to %s"%cachepath
                    self.DicoDirty.save(cachepath)
                    MyPickle.DicoNPToFile(self.DicoDirty,"%s.DicoPickle"%cachepath)
                    self.VS.maincache.saveCache("LastResidual")
                except:
                    print>> log, traceback.format_exc()
                    print>> log, ModColor.Str("WARNING: Residual image cache could not be written, see error report above. Proceeding anyway.")

        self.FacetMachine.finaliseSmoothBeam()

        # dump dirty to cache
        if self.GD["Cache"]["LastResidual"] and self.DicoDirty is not None:
            cachepath, valid = self.VS.maincache.checkCache("LastResidual",
                                                            dict(
                                                                [("MSNames", [ms.MSName for ms in self.VS.ListMS])] +
                                                                [(section, self.GD[section]) for section in "Data", "Beam", "Selection",
                                                                 "Freq", "Image", "Comp",
                                                                 "RIME","Weight","Facets",
                                                                 "DDESolutions"]
                                                            ),
                                                            reset=False)
            try:
                print>>log,"Saving last residual image to %s"%cachepath
                self.DicoDirty.save(cachepath)
                self.VS.maincache.saveCache("LastResidual")
            except:
                print>> log, traceback.format_exc()
                print>> log, ModColor.Str("WARNING: Residual image cache could not be written, see error report above. Proceeding anyway.")

        # delete shared dicts that are no longer needed, since Restore() may need a lot of memory
        self.VS.releaseLoadedChunk()
        self.FacetMachine.releaseGrids()
        self.FacetMachine.releaseCFs()
        if self.FacetMachinePSF is not None:
            self.FacetMachinePSF.releaseGrids()
            self.FacetMachinePSF.releaseCFs()
        if self.DicoImagesPSF is not None:
            self.DicoImagesPSF.delete()
            self.DicoImagesPSF = None

        # we still need the normdict, and DicoDirty (for the residuals), so keep those around
        # self.Restore()

        if self.HasDeconvolved:
            self.Restore()

            # Last major cycle may output residues other than Stokes I
            # Since the current residue images are for Stokes I only
            # we need to redo them in all required stokes
            self.do_stokes_residue and self._dump_stokes_residues()

    def _dump_stokes_residues(self):
         """
            Precondition: Must have already initialized a Facet Machine in self.FacetMachine
            Post-conditions: Dump out stokes residues to disk as requested in
            Output-StokesResidues, Stokes residues stored in self.DicoDirty
         """
         print>>log, ModColor.Str("============================== Making Stokes residue maps ====================")
         print>>log, ModColor.Str ("W.A.R.N.I.N.G: Stokes parameters other than I have not been deconvolved. Use these maps"
                                  " only as a debugging tool.", col="yellow")

         # tell the I/O thread to go load the first chunk
         self.VS.ReInitChunkCount()
         self.VS.startChunkLoadInBackground()

         # init new grids for Stokes residues
         self.StokesFacetMachine.Init()
         self.FacetMachine.Init()
         self.StokesFacetMachine.ReinitDirty()
         self.FacetMachine.initCFInBackground()
         self.StokesFacetMachine.initCFInBackground(other_fm=self.FacetMachine) #same weighting map and CF support
         current_model_freqs = np.array([]) #invalidate
         while True:
            # note that collectLoadedChunk() will destroy the current DATA dict, so we must make sure
            # the gridding jobs of the previous chunk are finished
            self.StokesFacetMachine.collectGriddingResults()

            # get loaded chunk from I/O thread, schedule next chunk
            # self.VS.startChunkLoadInBackground()
            DATA = self.VS.collectLoadedChunk(start_next=True)
            if type(DATA) is str:
                print>>log,ModColor.Str("no more data: %s"%DATA, col="red")
                break
            # None weights indicates an all-flagged chunk: go on to the next chunk
            if DATA["Weights"] is None:
                continue

            # Stacks average beam if not computed
            self.StokesFacetMachine.StackAverageBeam(DATA)
            self.FacetMachine.StackAverageBeam(DATA)

            # Predict and subtract from current MS vis data
            model_freqs = DATA["FreqMappingDegrid"]

            # switch subband if necessary
            self.FacetMachine.awaitInitCompletion()
            if not np.array_equal(model_freqs, current_model_freqs):
                ModelImage = self.FacetMachine.setModelImage(self.DeconvMachine.GiveModelImage(model_freqs))
                # write out model image, if asked to
                current_model_freqs = model_freqs
                print>>log,"model image @%s MHz (min,max) = (%f, %f)"%(str(model_freqs/1e6),ModelImage.min(),ModelImage.max())
            else:
                print>>log,"reusing model image from previous chunk"
            if self.PredictMode == "BDA-degrid" or self.PredictMode == "DeGridder":
                self.FacetMachine.getChunkInBackground(DATA)
            elif self.PredictMode == "Montblanc":
                from ClassMontblancMachine import ClassMontblancMachine
                model = self.ModelMachine.GiveModelList()
                mb_machine = ClassMontblancMachine(self.GD, fm_predict.Npix, self.FacetMachine.CellSizeRad)
                mb_machine.getChunk(DATA, DATA["data"], model, self.VS.ListMS[DATA["iMS"]])
                mb_machine.close()
            else:
                raise ValueError("Invalid PredictMode '%s'" % self.PredictMode)

            # Ensure degridding and subtraction has finished before firing up
            # gridding
            self.FacetMachine.collectDegriddingResults()

            # Grid residue vis
            self.StokesFacetMachine.awaitInitCompletion()
            self.StokesFacetMachine.putChunkInBackground(DATA)

         # if Smooth beam enabled, either compute it from the stack, or get it from cache
         # else do nothing
         self.StokesFacetMachine.finaliseSmoothBeam()

         # fourier transform, stitch facets and release grids
         self.DicoDirty = self.StokesFacetMachine.FacetsToIm(NormJones=True)
         self.StokesFacetMachine.releaseGrids()

         # All done dump the stokes residues
         if "r" in self._saveims:
            self.StokesFacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],
                                                ImageName="%s.stokes.app.residual"%self.BaseName,
                                                Fits=True,
                                                Stokes=self.StokesFacetMachine.StokesConverter.RequiredStokesProducts())
         if "r" in self._savecubes:
            self.FacetMachine.ToCasaImage(self.DicoDirty["ImageCube"],
                                          ImageName="%s.cube.stokes.app.residual"%self.BaseName,
                                          Fits=True,
                                          Freqs=self.VS.FreqBandCenters,
                                          Stokes=self.StokesFacetMachine.StokesConverter.RequiredStokesProducts())
         if self.DicoDirty["JonesNorm"] is not None:
            # this assumes the matrix squareroot is the same for all
            # stokes parameters. This may or may not be true so take this
            # with a bag of salt
            nch,npol,nx,ny = self.DicoDirty["ImageCube"].shape
            DirtyCorr = self.DicoDirty["ImageCube"]/np.sqrt(self.DicoDirty["JonesNorm"])
            if "R" in self._saveims:
                MeanCorr = np.mean(DirtyCorr, axis=0).reshape((1, npol, nx, ny))
                self.FacetMachine.ToCasaImage(MeanCorr,
                                              ImageName="%s.stokes.int.residual"%self.BaseName,
                                              Fits=True,
                                              Stokes=self.StokesFacetMachine.StokesConverter.RequiredStokesProducts())
            if "R" in self._savecubes:
                self.FacetMachine.ToCasaImage(DirtyCorr,
                                              ImageName="%s.cube.stokes.int.residual"%self.BaseName,
                                              Fits=True,
                                              Freqs=self.VS.FreqBandCenters,
                                              Stokes=self.StokesFacetMachine.StokesConverter.RequiredStokesProducts())

    def fitSinglePSF(self, PSF, off, label="mean"):
        """
            Fits a PSF given by argument
        Args:
            PSF: PSF array
            label: string label used in output to describe this PSF
        Returns:
            tuple of ((fwhm_xdeg,fwhm_deg,pa_deg),(gx,gy,theta),sidelobes)
        """
        x, y = np.where(PSF == np.max(PSF))[-2:]
        nx, ny = PSF.shape[-2:]
        off = self.GD["Image"]["SidelobeSearchWindow"] // 2
        if x.shape[0] == 0:
            raise RuntimeError("Empty PSF slice detected while fitting. Check your data.")
        off = min(off, x[0], nx-x[0], y[0], ny-y[0])
        print>> log, "Fitting %s PSF in a [%i,%i] box ..." % (label, off * 2, off * 2)
        P = PSF[0, x[0] - off:x[0] + off, y[0] - off:y[0] + off].copy()
        bmaj, bmin, theta = ModFitPSF.FitCleanBeam(P)
        sidelobes = ModFitPSF.FindSidelobe(P)
        print>>log, "PSF max is %f"%P.max()

        FWHMFact = 2. * np.sqrt(2. * np.log(2.))

        fwhm = (bmaj * self.CellArcSec * FWHMFact / 3600.,
                bmin * self.CellArcSec * FWHMFact / 3600.,
                np.rad2deg(theta))
        gausspars = (bmaj * self.CellSizeRad, bmin * self.CellSizeRad, theta)
        print>> log, "\tsigma is %f, %f (FWHM is %f, %f), PA is %f deg" % (bmaj * self.CellArcSec,
                                                                           bmin * self.CellArcSec,
                                                                           bmaj * self.CellArcSec * FWHMFact,
                                                                           bmin * self.CellArcSec * FWHMFact,
                                                                           90-np.rad2deg(theta))
        print>> log, "\tSecondary sidelobe at the level of %5.1f at a position of %i from the center" % sidelobes
        return fwhm, gausspars, sidelobes

    def FitPSF(self):
        """
            Fits the PSF to get the parameters for the clean beam used in restoring
            Post conditions:
                self.FWHMBeam: The maj (deg), min (deg), theta (deg) gaussian parameters for the full width half power
                               fits. This should be passed to the FITS file outputs
                self.PSFGaussPars: The maj (rad), min (rad), theta (rad) parameters for the fit of the gaussian
                self.PSFSidelobes: Position of the highest sidelobes (px)
            Returns:
                None if there was no error during fitting, the original exception otherwise
        """
        if self.HasFittedPSFBeam:
            return

        self.HasFittedPSFBeam=True

        # If set, use the parameter RestoringBeam to fix the clean beam parameters
        forced_beam=self.GD["Output"]["RestoringBeam"]
        if forced_beam is not None:
            FWHMFact = 2. * np.sqrt(2. * np.log(2.))

            if isinstance(forced_beam,float) or isinstance(forced_beam,int):
                forced_beam=[float(forced_beam),float(forced_beam),0]
            elif len(forced_beam)==1:
                forced_beam=[forced_beam[0],forced_beam[0],0]
            f_beam=(forced_beam[0]/3600.0,forced_beam[1]/3600.0,forced_beam[2])
            f_gau=(np.deg2rad(f_beam[0])/FWHMFact,np.deg2rad(f_beam[1])/FWHMFact,np.deg2rad(f_beam[2]))
        PSF = self.DicoImagesPSF["CubeVariablePSF"][self.FacetMachinePSF.iCentralFacet]
        meanPSF = self.DicoImagesPSF["CubeMeanVariablePSF"][self.FacetMachinePSF.iCentralFacet]

        off=self.GD["Image"]["SidelobeSearchWindow"] // 2
        try:
            fit_err = None
            beam, gausspars, sidelobes = self.fitSinglePSF(meanPSF[0,...], "mean")
        except Exception as e:
            beam = (0,0,0)
            gausspars = (0,0,0)
            sidelobes=(0,0)
            fit_err = e

        if forced_beam is not None:
            print>>log, 'Will use user-specified beam: bmaj=%f, bmin=%f, bpa=%f degrees' % f_beam
            beam, gausspars = f_beam, f_gau
            fit_err = None # don't care if user provided parameters

        self.FWHMBeamAvg, self.PSFGaussParsAvg, self.PSFSidelobesAvg = beam, gausspars, sidelobes
        # MeanFacetPSF has a shape of 1,1,nx,ny, so need to cut that extra one off
        if self.VS.MultiFreqMode:
            self.FWHMBeam = []
            self.PSFGaussPars = []
            self.PSFSidelobes = []
            for band in range(self.VS.NFreqBands):
                try:
                    beam, gausspars, sidelobes = self.fitSinglePSF(PSF[band,...],off,"band %d"%band)
                except Exception as e:
                    beam = (0,0,0)
                    gausspars = (0,0,0)
                    sidelobes=(0,0)
                    fit_err = e # last error stored

                if forced_beam is not None:
                    beam = f_beam
                    gausspars = f_gau

                self.FWHMBeam.append(beam)
                self.PSFGaussPars.append(gausspars)
                self.PSFSidelobes.append(sidelobes)
        else:
            self.FWHMBeam = [self.FWHMBeamAvg]
            self.PSFGaussPars = [self.PSFGaussParsAvg]
            self.PSFSidelobes = [self.PSFSidelobesAvg]

        self.DicoImagesPSF["PSFGaussPars"]=self.PSFGaussPars
        self.DicoImagesPSF["PSFSidelobes"]=self.PSFSidelobes
        self.DicoImagesPSF["EstimatesAvgPSF"]=(self.FWHMBeamAvg, self.PSFGaussParsAvg, self.PSFSidelobesAvg)
        return fit_err

    def GiveMetroModel(self):
        model_freqs=self.VS.CurrentChanMappingDegrid
        ModelImage = self.DeconvMachine.GiveModelImage(model_freqs)
        nf,npol,nx,nx=ModelImage.shape
        ModelImageAvg=np.mean(ModelImage,axis=0).reshape((1,npol,nx,nx))

        self.FacetMachine.ToCasaImage(ModelImageAvg,
                                      ImageName="%s.model.pre_metro"%(self.BaseName),
                                      Fits=True)

        GD=copy.deepcopy(self.GD)
        # # ####################
        # # Initialise Model machine for MetroClean
        # ThisMode=["S","Alpha","GSig"]
        # GD["SSDClean"]["SSDSolvePars"]=ThisMode
        # DicoAugmentedModel=self.DeconvMachine.ModelMachine.GiveConvertedSolveParamDico(ThisMode)
        # MinorCycleConfig=dict(GD["ImagerDeconv"])
        # MinorCycleConfig["NCPU"]=GD["Parallel"]["NCPU"]
        # MinorCycleConfig["NFreqBands"]=self.VS.NFreqBands
        # MinorCycleConfig["GD"] = GD
        # MinorCycleConfig["ImagePolDescriptor"] = self.VS.StokesConverter.RequiredStokesProducts()
        # MinorCycleConfig["IdSharedMem"] = self.IdSharedMem
        # ModelMachine = self.ModConstructor.GiveMM(Mode=GD["ImagerDeconv"]["MinorCycleMode"])
        # ModelMachine.FromDico(DicoAugmentedModel)
        # MinorCycleConfig["ModelMachine"] = ModelMachine

        # # ####################
        # # Initialise Image deconv machine for MetroClean
        # from DDFacet.Imager.SSD import ClassImageDeconvMachineSSD
        # DeconvMachine=ClassImageDeconvMachineSSD.ClassImageDeconvMachine(**MinorCycleConfig)
        # DeconvMachine.Init(PSFVar=self.DicoImagesPSF,PSFAve=self.PSFSidelobesAvg)

        DeconvMachine=self.DeconvMachine
        ModelMachine=self.ModelMachine

        # Create a ModelMachine to sture the sigma values
        DicoErrModel=copy.deepcopy(ModelMachine.giveDico())
        del(DicoErrModel["Comp"])
        DicoErrModel["Comp"]={}
        ModConstructor = ClassModModelMachine(self.GD)
        ErrorModelMachine = ModConstructor.GiveMM(Mode=GD["Deconv"]["Mode"])
        ErrorModelMachine.FromDico(DicoErrModel)

        DeconvMachine.ErrorModelMachine=ErrorModelMachine

        # ####################
        # Run MetroClean
        print>>log,"Running a Metropolis-Hastings MCMC on islands larger than %i pixels"%self.GD["SSDClean"]["RestoreMetroSwitch"]
        DeconvMachine.setDeconvMode(Mode="MetroClean")
        DeconvMachine.Update(self.DicoDirty)
        repMinor, continue_deconv, update_model = DeconvMachine.Deconvolve()
        DeconvMachine.ToFile(self.DicoMetroModelName)

        ErrModelImage = DeconvMachine.ErrorModelMachine.GiveModelImage(model_freqs)
        nf,npol,nx,nx=ErrModelImage.shape
        ErrModelImageAvg=np.mean(ErrModelImage,axis=0).reshape((1,npol,nx,nx))
        self.FacetMachine.ToCasaImage(ErrModelImageAvg,
                                      ImageName="%s.metro.model.sigma"%(self.BaseName),
                                      Fits=True)

        ModelImage = DeconvMachine.GiveModelImage(model_freqs)
        nf,npol,nx,nx=ModelImage.shape
        ModelImageAvg=np.mean(ModelImage,axis=0).reshape((1,npol,nx,nx))
        self.FacetMachine.ToCasaImage(ModelImageAvg,
                                      ImageName="%s.metro.model"%(self.BaseName),
                                      Fits=True)
        self.FacetMachine.ToCasaImage(DeconvMachine.SelectedIslandsMask,
                                      ImageName="%s.metro.mask"%(self.BaseName),
                                      Fits=True)
        return ModelMachine

    def _saveImage_worker (self, sd, field, ImageName, delete=False,
                           Fits=True, beam=None, beamcube=None, Freqs=None,
                           Stokes=None):
        """Worker function to save an image to disk, and optionally to delete it"""
        image = sd[field]
        self.FacetMachine.ToCasaImage(image, ImageName=ImageName, Fits=Fits, beam=beam, beamcube=beamcube,
                                      Freqs=Freqs, Stokes=Stokes)
        if delete:
#            print>> log, "releasing %s image" % field
            sd.delete_item(field)

    def _delSharedImage_worker(self, sd, *args):
        """Worker function to delete an imaged from the shared dict. Enqueued after that image is no longer useful."""
        for field in args:
            if field in sd:
 #               print>>log, "releasing %s image" % field
                sd.delete_item(field)



    
    def RestoreAndShift(self):
        dirty_cachepath = self.VS.maincache.getElementPath("LastResidual")
        #dirty_cachepath = self.VS.maincache.getElementPath("Dirty")
        valid = os.path.exists(dirty_cachepath)
        
        if not valid:
            print>> log, ModColor.Str("Can't force-read cached last residual %s: does not exist" % dirty_cachepath, col="red")
            raise RuntimeError("--Cache-Dirty forceresidual in effect, but no cached residual image found")
        print>> log, ModColor.Str("Forcing reading the cached last residual image", col="red")
        
        self.DicoDirty = shared_dict.create("FM_AllImages")
        self.DicoDirty.restore(dirty_cachepath)
        
        
        cachepath = self.VS.maincache.getElementPath("PSF")
        valid = os.path.exists(cachepath)
        if not valid:
            print>> log, ModColor.Str("Can't force-read cached PSF %s: does not exist" % cachepath, col="red")
            raise RuntimeError("--Cache-PSF force in effect, but no cached PSF image found")
        print>> log, ModColor.Str("Forcing to read the cached PSF", col="red")
        self.DicoImagesPSF = shared_dict.create("FMPSF_AllImages")
        self.DicoImagesPSF.restore(cachepath)
        self.FWHMBeam=self.DicoImagesPSF["FWHMBeam"]
        self.PSFGaussPars=self.DicoImagesPSF["PSFGaussPars"]
        self.PSFSidelobes=self.DicoImagesPSF["PSFSidelobes"]
        (self.FWHMBeamAvg, self.PSFGaussParsAvg, self.PSFSidelobesAvg)=self.DicoImagesPSF["EstimatesAvgPSF"]
        
        if self.DicoDirty["JonesNorm"] is not None:
            self.FacetMachine.setNormImages(self.DicoDirty)
            self.FacetMachinePSF.setNormImages(self.DicoDirty)
            self.MeanJonesNorm = self.FacetMachinePSF.MeanJonesNorm
            self.JonesNorm = self.FacetMachinePSF.JonesNorm
        elif self.DicoImagesPSF["JonesNorm"] is not None:
            self.FacetMachine.setNormImages(self.DicoImagesPSF)
            self.FacetMachinePSF.setNormImages(self.DicoImagesPSF)
            self.MeanJonesNorm = self.FacetMachinePSF.MeanJonesNorm
            self.JonesNorm = self.FacetMachinePSF.JonesNorm
        else:
            self.MeanJonesNorm = None
            self.JonesNorm = None

        Norm=None
        havenorm = self.MeanJonesNorm is not None and (self.MeanJonesNorm != 1).any()
        ModelImage=self.ModelMachine.GiveModelImage()
        if havenorm:
            Norm = self.MeanJonesNorm 
            sqrtNorm=np.sqrt(Norm)
            if self.FacetMachine.MeanSmoothJonesNorm is None:
                SmoothNorm=Norm
                sqrtSmoothNorm=sqrtNorm
            else:
                print>>log,ModColor.Str("Using the freq-averaged smooth beam to normalise the apparent images",col="blue")
                SmoothNorm=self.FacetMachine.MeanSmoothJonesNorm
                sqrtSmoothNorm=np.sqrt(SmoothNorm)

            ModelImage=ModelImage*sqrtNorm


        ModelImage = self.FacetMachine.setModelImage(ModelImage)
        
        Restored=self.FacetMachine.giveRestoredFacets(self.DicoDirty,
                                                      self.PSFGaussParsAvg,
                                                      ShiftFile=self.GD["Output"]["ShiftFacetsFile"])
        self.FacetMachine.ToCasaImage(Restored, ImageName="%s.app.facetRestored" % self.BaseName, 
                                      Fits=True,
                                      beam=self.FWHMBeamAvg, Stokes=self.VS.StokesConverter.RequiredStokesProducts())

        if havenorm:
            IntRestored=Restored/sqrtSmoothNorm
            self.FacetMachine.ToCasaImage(IntRestored, ImageName="%s.int.facetRestored" % self.BaseName, 
                                          Fits=True,
                                          beam=self.FWHMBeamAvg, Stokes=self.VS.StokesConverter.RequiredStokesProducts())





    def Restore(self):


        print>>log, "Create restored image"

        if self.PSFGaussPars is None:
            self.fit_stat = self.FitPSF()
            if self.fit_stat is not None:
                raise self.fit_stat #postponed fitting error (in absence of user-supplied parameters)
        #self.DeconvMachine.ToFile(self.DicoModelName)

        RefFreq = self.VS.RefFreq

        if self.GD["Deconv"]["Mode"]=="SSD" and self.GD["SSDClean"]["RestoreMetroSwitch"]>0:
            ModelMachine=self.GiveMetroModel()
        else:
            ModelMachine = self.DeconvMachine.ModelMachine

        # Putting back substracted componants
        if self.GD["DDESolutions"]["RestoreSub"]:
            try:
                ModelMachine.PutBackSubsComps()
            except:
                print>>log, ModColor.Str("Failed putting back subtracted components")

        # do we have a non-trivial norm (i.e. DDE solutions or beam)?
        # @cyriltasse: maybe there's a quicker way to check?
        havenorm = self.MeanJonesNorm is not None and (self.MeanJonesNorm != 1).any()

        T = ClassTimeIt.ClassTimeIt()
        T.disable()

        # Make a SharedDict of images to save the intermediate images for when we need them.
        # SharedDict because we'll have a job on the I/O queue doing the actual saving.
        # Since these images and cubes can take a lot of SHM, we will also schedule I/O jobs to delete them
        # after we don't need them anymore.

        _images = shared_dict.create("OutputImages")
        _final_RMS = {}
        def sqrtnorm():
            label = 'sqrtnorm'
            if label not in _images:
                if havenorm:
                    a = self.MeanJonesNorm 
                else:
                    a=np.array([1])
                out = _images.addSharedArray(label, a.shape, a.dtype)
                numexpr.evaluate('sqrt(a)', out=out)
            return _images[label]
        def sqrtnormcube():
            label = 'sqrtnormcube'
            if label not in _images:
                if havenorm:
                    a = self.JonesNorm
                else:
                    a=np.array([1])
#                a = self.JonesNorm if havenorm else np.array([1])
                out = _images.addSharedArray(label, a.shape, a.dtype)
                numexpr.evaluate('sqrt(a)', out=out)
            return _images[label]
        def smooth_sqrtnorm():
            label = 'smooth_sqrtnorm'
            if label not in _images:
                if havenorm:
                    if self.FacetMachine.MeanSmoothJonesNorm is None:
                        a = self.MeanJonesNorm 
                    else:
                        print>>log,ModColor.Str("Using the freq-averaged smooth beam to normalise the apparent images",col="blue")
                        a=self.FacetMachine.MeanSmoothJonesNorm
                else:
                    a=np.array([1])
                out = _images.addSharedArray(label, a.shape, a.dtype)
                numexpr.evaluate('sqrt(a)', out=out)
            return _images[label]
        def smooth_sqrtnormcube():
            label = 'smooth_sqrtnormcube'
            if label not in _images:
                if havenorm:
                    if self.FacetMachine.MeanSmoothJonesNorm is None:
                        a = self.JonesNorm 
                    else:
                        print>>log,ModColor.Str("Using the smooth beam to normalise the apparent images",col="blue")
                        a=self.FacetMachine.SmoothJonesNorm
                else:
                    a=np.array([1])
#                a = self.JonesNorm if havenorm else np.array([1])
                out = _images.addSharedArray(label, a.shape, a.dtype)
                numexpr.evaluate('sqrt(a)', out=out)
            return _images[label]
        def appres():
            return self.DicoDirty["MeanImage"]
        def intres():
            label = 'intres'
            if label not in _images:
                if havenorm:
                    a, b = appres(), smooth_sqrtnorm()
                    out = _images.addSharedArray(label, a.shape, a.dtype)
                    numexpr.evaluate('a/b', out=out)
                    out[~np.isfinite(out)] = 0
                else:
                    _images[label] = appres()
            return _images[label]
        def apprescube():
            return self.DicoDirty["ImageCube"]
        def intrescube():
            label = 'intrescube'
            if label not in _images:
                if havenorm:
                    a, b = apprescube(), smooth_sqrtnormcube()
                    out = _images.addSharedArray(label, a.shape, a.dtype)
                    numexpr.evaluate('a/b', out=out)
                    out[~np.isfinite(out)] = 0
                else:
                    _images[label] = apprescube()
            return _images[label]
        def appmodel():
            label = 'appmodel'
            if label not in _images:
                if havenorm:
                    a, b = intmodel(), smooth_sqrtnorm()
                    out = _images.addSharedArray(label, a.shape, a.dtype)
                    numexpr.evaluate('a*b', out=out)
                else:
                    _images[label] = intmodel()
            return _images[label]
        def intmodel():
            label = 'intmodel'
            if label not in _images:
                out=ModelMachine.GiveModelImage(RefFreq)
                if havenorm:
                    a, b, c = out, sqrtnorm(), smooth_sqrtnorm()
                    numexpr.evaluate('a*b/c', out=out)
                _images[label] = out
            return _images[label]
        def appmodelcube():
            label = 'appmodelcube'
            if label not in _images:
                if havenorm:
                    a, b = intmodelcube(), smooth_sqrtnormcube()
                    out = _images.addSharedArray(label, a.shape, a.dtype)
                    numexpr.evaluate('a*b', out=out)
                else:
                    _images[label] = intmodelcube()
            return _images[label]
        def intmodelcube():
            label = 'intmodelcube'
            if label not in _images:
                shape = list(ModelMachine.ModelShape)
                shape[0] = len(self.VS.FreqBandCenters)
                out = _images.addSharedArray(label, shape, np.float32)
                ModelMachine.GiveModelImage(self.VS.FreqBandCenters, out=out)
                if havenorm:
                    a, b, c = out, sqrtnormcube(), smooth_sqrtnormcube()
                    numexpr.evaluate('a*b/c', out=out)
            return _images[label]
        def appconvmodel():
            label = 'appconvmodel'
            if label not in _images:
                if havenorm:
                    out = _images.addSharedArray(label, appmodel().shape, np.float32)
                    ModFFTW.ConvolveGaussian(shareddict={"in": appmodel(),
                                                         "out": out},
                                             field_in = "in",
                                             field_out = "out",
                                             ch = 0,
                                             CellSizeRad=self.CellSizeRad,
                                             GaussPars_ch=self.PSFGaussParsAvg)
                    T.timeit(label)
                else:
                    _images[label] = intconvmodel()
            return _images[label]
        def intconvmodel():
            label = 'intconvmodel'
            if label not in _images:
                out = _images.addSharedArray(label, intmodel().shape,
                                             np.float32)
                ModFFTW.ConvolveGaussian(shareddict={"in": intmodel(),
                                                     "out": out},
                                         field_in = "in",
                                         field_out = "out",
                                         ch = 0,
                                         CellSizeRad=self.CellSizeRad,
                                         GaussPars_ch=self.PSFGaussParsAvg)
                T.timeit(label)
            return _images[label]
        def appconvmodelcube():
            label = 'appconvmodelcube'
            if label not in _images:
                if havenorm:
                    out = _images.addSharedArray(label, appmodelcube().shape, np.float32)
                    ModFFTW.ConvolveGaussianParallel(shareddict=_images,
                                                     field_in = "appmodelcube",
                                                     field_out = label,
                                                     CellSizeRad=self.CellSizeRad,
                                                     GaussPars=self.PSFGaussPars)
                    T.timeit(label)
                else:
                    _images[label] = intconvmodelcube()
            return _images[label]
        def intconvmodelcube():
            label = 'intconvmodelcube'
            if label not in _images:
                out = _images.addSharedArray(label, intmodelcube().shape, np.float32)
                ModFFTW.ConvolveGaussianParallel(shareddict=_images,
                                                 field_in = "intmodelcube",
                                                 field_out =label,
                                                 CellSizeRad=self.CellSizeRad,
                                                 GaussPars=self.PSFGaussPars)

                ModFFTW.ConvolveGaussianParallel(_images, 'intmodelcube', label,
                                                 CellSizeRad=self.CellSizeRad, GaussPars=self.PSFGaussPars)
                T.timeit(label)
            return _images[label]
        def posintmod():
            label = 'posintmod'
            if label not in _images:
                _images.addSharedArray(label, intmodel().shape, np.float32)
                _images[label] = ModelMachine.FreqMachine.Iref.reshape(intmodel().shape)
            return _images[label]
        def give_final_RMS():
            try:
                return _final_RMS["RMS"]
            except:
                _final_RMS["RMS"] = np.std(intres().ravel())
                return _final_RMS["RMS"]
        def weighted_alphamap():
            label = 'weighted_alphamap'
            if label not in _images:
                _images.addSharedArray(label, intmodel().shape, np.float32)
                # compute the RMS of the final residual
                RMS = give_final_RMS()
                # get the RMS threshold
                RMSthreshold = self.GD["Output"]["alphathreshold"]
                _images[label] = ModelMachine.GiveSpectralIndexMap(threshold=RMS*RMSthreshold)
                _images['posintmod'] = ModelMachine.FreqMachine.Iref.reshape(intmodel().shape)
            return _images[label]
        def alphamap():
            label = 'alphamap'
            if label not in _images:
                _images.addSharedArray(label, intmodel().shape, np.float32)
                
                # ##############################
                # # Reverting for issue458
                #_images[label] = ModelMachine.FreqMachine.alpha_map.reshape(intmodel().shape)
                _images[label] = ModelMachine.GiveSpectralIndexMap()
                # ##############################

            return _images[label]
        def alphaconvmap():
            label = 'alphaconvmap'
            if label not in _images:
                # Get weighted alpha map
                a = _images.addSharedArray('alphaconvmap', weighted_alphamap().shape, np.float32)
                # Convolve with Gaussian
                ModFFTW.ConvolveGaussian(shareddict={"in": weighted_alphamap(),
                                                     "out": a},
                                         field_in = "in",
                                         field_out = "out",
                                         ch = 0,
                                         CellSizeRad=self.CellSizeRad,
                                         GaussPars_ch=self.PSFGaussParsAvg)

                # Get positive part of restored image
                b = _images.addSharedArray('posconvmod', alphamap().shape, np.float32)
                ModFFTW.ConvolveGaussian(shareddict={"in": alphamap(),
                                                     "out": b},
                                         field_in = "in",
                                         field_out = "out",
                                         ch = 0,
                                         CellSizeRad=self.CellSizeRad,
                                         GaussPars_ch=self.PSFGaussParsAvg)

                c = intconvmodel()
                # Get mask based on restored image and positive restored image
                RMS = give_final_RMS()
                RMSmaskfact = self.GD["Output"]["alphamaskthreshold"]
                I1 = c[0, 0, :, :] > RMSmaskfact*RMS
                I2 = b[0, 0, :, :] > RMSmaskfact*RMS
                IC = I1 & I2
                I = np.argwhere(IC)
                #print I.size
                ix = I[:,0]
                iy = I[:,1]
                d = np.zeros_like(a)
                d[0, 0, ix, iy] = a[0, 0, ix, iy]/b[0, 0, ix, iy]
                #print a.min(), a.max()
                _images.addSharedArray(label, alphamap().shape, np.float32)
                _images[label] = d
                T.timeit(label)
            return _images[label]

        # norm
        if havenorm and ("S" in self._saveims or "s" in self._saveims):
            sqrtnorm()
            APP.runJob("save:sqrtnorm", self._saveImage_worker, io=0, args=( _images.readonly(), "sqrtnorm",),
                            kwargs=dict( ImageName="%s.fluxscale"%(self.BaseName),
                                          Fits=True,Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        # apparent-flux residuals
        if "r" in self._saveims:
            appres()
            APP.runJob("save:appres", self._saveImage_worker, io=0, args=( self.DicoDirty.readonly(), "MeanImage",),
                            kwargs=dict( ImageName="%s.app.residual"%(self.BaseName),
                                          Fits=True,Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        # intrinsic-flux residuals
        if havenorm and "R" in self._saveims:
            intres()
            APP.runJob("save:intres", self._saveImage_worker, io=0, args=( _images.readonly(), "intres",),
                            kwargs=dict(ImageName="%s.int.residual"%(self.BaseName),Fits=True,
                                          Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        # apparent-flux model
        if "m" in self._saveims:
            appmodel()
            APP.runJob("save:appmodel", self._saveImage_worker, io=0, args=(_images.readonly(), "appmodel",),
                       kwargs=dict(ImageName="%s.app.model" % self.BaseName, Fits=True,
                                   Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        # intrinsic-flux model
        if havenorm and "M" in self._saveims:
            intmodel()
            APP.runJob("save:intmodel", self._saveImage_worker, io=0, args=(_images.readonly(), "intmodel",),
                       kwargs=dict(ImageName="%s.int.model" % self.BaseName, Fits=True,
                                   Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        # convolved-model image in apparent flux
        if "c" in self._saveims:
            appconvmodel()
            APP.runJob("save:appconvmodel", self._saveImage_worker, io=0, args=(_images.readonly(), "appconvmodel",),
                       kwargs=dict(ImageName="%s.app.convmodel" % self.BaseName, Fits=True,
                                   beam=self.FWHMBeamAvg, Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        # convolved-model image in intrinsic flux
        if havenorm and "C" in self._saveims:
            intconvmodel()
            APP.runJob("save:intconvmodel", self._saveImage_worker, io=0, args=(_images.readonly(), "intconvmodel",),
                       kwargs=dict(ImageName="%s.int.convmodel" % self.BaseName, Fits=True,
                                   beam=self.FWHMBeamAvg, Stokes=self.VS.StokesConverter.RequiredStokesProducts()))

        # norm cube
        if havenorm and ("S" in self._savecubes or "s" in self._savecubes):
            sqrtnormcube()
            APP.runJob("save:sqrtnormcube", self._saveImage_worker, io=0, args=(_images.readonly(), "sqrtnormcube",),
                       kwargs=dict(ImageName="%s.cube.fluxscale" % (self.BaseName), Fits=True,
                                   Freqs=self.VS.FreqBandCenters,
                                   Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        # apparent-flux restored image
        if "i" in self._saveims:
            _images["apprestored"] = appres()
            _images["apprestored"] += appconvmodel()
            APP.runJob("save:apprestored", self._saveImage_worker, io=0, args=(_images.readonly(), "apprestored",),
                       kwargs=dict(ImageName="%s.app.restored" % self.BaseName, Fits=True,
                                   beam=self.FWHMBeamAvg, Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        # intrinsic-flux restored image
        if havenorm and "I" in self._saveims:
            _images["intrestored"] = intres()
            _images["intrestored"] += intconvmodel()
            APP.runJob("save:intrestored", self._saveImage_worker, io=0, args=(_images.readonly(), "intrestored",),
                       kwargs=dict(ImageName="%s.int.restored" % self.BaseName, Fits=True,
                                   beam=self.FWHMBeamAvg, Stokes=self.VS.StokesConverter.RequiredStokesProducts()))

        # # intrinsic-flux restored image
        # if havenorm and self.DoSmoothBeam:
        #     if self.FacetMachine.SmoothJonesNorm is None:
        #         print>> log, ModColor.Str("You requested a restored imaged but the smooth beam is not in there")
        #         print>> log, ModColor.Str("  so just not doing it")
        #     else:
        #         a, b, c = appres(), appconvmodel(), self.FacetMachine.MeanSmoothJonesNorm
        #         out = _images.addSharedArray('smoothrestored', a.shape, a.dtype)
        #         numexpr.evaluate('(a+b)/sqrt(c)', out=out)
        #         APP.runJob("save:smoothrestored", self._saveImage_worker, io=0,
        #                    args=(_images.readwrite(), "smoothrestored",), kwargs=dict(
        #                 ImageName="%s.smooth.int.restored" % self.BaseName, Fits=True, delete=True,
        #                 beam=self.FWHMBeamAvg, Stokes=self.VS.StokesConverter.RequiredStokesProducts()))

        # mixed-flux restored image
        # (apparent noise + intrinsic model) if intrinsic model is available
        # (apparent noise + apparent model) otherwise, (JonesNorm ~= 1)
        if "x" in self._saveims:
            a, b = (appres(), intconvmodel()) if havenorm else \
                   (appres(), appconvmodel())
            out = _images.addSharedArray('mixrestored', a.shape, a.dtype)
            numexpr.evaluate('a+b', out=out)
            APP.runJob("save:mixrestored", self._saveImage_worker, io=0, args=(_images.readwrite(), "mixrestored",),
                       kwargs=dict(
                           ImageName="%s.restored" % self.BaseName, Fits=True, delete=True,
                           beam=self.FWHMBeamAvg, Stokes=self.VS.StokesConverter.RequiredStokesProducts()))

        # Alpha image
        if "A" in self._saveims and self.VS.MultiFreqMode:
            # ##############################
            # # Reverting for issue458
            # _images['alphaconvmap'] = alphaconvmap()
            # APP.runJob("save:alphaconv", self._saveImage_worker, io=0, args=(_images.readwrite(), 'alphaconvmap',), kwargs=dict(
            #     ImageName="%s.alphaconv" % self.BaseName, Fits=True, delete=True, beam=self.FWHMBeamAvg,
            #     Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
            # ##############################
            _images['alphamap'] = alphamap()
            APP.runJob("save:alpha", self._saveImage_worker, io=0, args=(_images.readwrite(), 'alphamap',), kwargs=dict(
                ImageName="%s.alpha" % self.BaseName, Fits=True, delete=True, beam=self.FWHMBeamAvg,
                Stokes=self.VS.StokesConverter.RequiredStokesProducts()))

        #  done saving images -- schedule a job to delete them all from the dict to save RAM
        APP.runJob("del:images", self._delSharedImage_worker, io=0, args=[_images.readwrite()] + list(_images.keys()))

        # now form up cubes
        # apparent-flux model cube
        if "m" in self._savecubes:
            appmodelcube()
            APP.runJob("save:appmodelcube", self._saveImage_worker, io=0, args=(_images.readonly(), "appmodelcube", ),
                       kwargs=dict(ImageName="%s.cube.app.model" % self.BaseName, Fits=True,
                                   Freqs=self.VS.FreqBandCenters,
                                   Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        # intrinsic-flux model cube
        if havenorm and "M" in self._savecubes:
            intmodelcube()
            APP.runJob("save:intmodelcube", self._saveImage_worker, io=0, args=(_images.readonly(), "intmodelcube",),
                       kwargs=dict(ImageName="%s.cube.int.model" % self.BaseName, Fits=True,
                                   Freqs=self.VS.FreqBandCenters,
                                   Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        # convolved-model cube in apparent flux
        if "c" in self._savecubes:
            appconvmodelcube()
            APP.runJob("save:appconvmodelcube", self._saveImage_worker, io=0,
                       args=(_images.readonly(), "appconvmodelcube",),
                       kwargs=dict(ImageName="%s.cube.app.convmodel" % self.BaseName, Fits=True,
                                   beam=self.FWHMBeamAvg, beamcube=self.FWHMBeam, Freqs=self.VS.FreqBandCenters,
                                   Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        #  can delete this one now
        APP.runJob("del:appmodelcube", self._delSharedImage_worker, io=0, args=[_images.readwrite(), "appmodelcube"])
        # convolved-model cube in intrinsic flux
        if havenorm and "C" in self._savecubes:
            intconvmodelcube()
            APP.runJob("save:intconvmodelcube", self._saveImage_worker, io=0, args=( _images.readwrite(), "intconvmodelcube",), kwargs=dict(ImageName="%s.cube.int.convmodel"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,beamcube=self.FWHMBeam,Freqs=self.VS.FreqBandCenters,
                Stokes=self.VS.StokesConverter.RequiredStokesProducts()))


        # intrinsic-flux restored image cube
        if havenorm and "I" in self._savecubes:
            a, b = intrescube(), intconvmodelcube()
            out = _images.addSharedArray('intrestoredcube', a.shape, a.dtype)
            numexpr.evaluate('a+b', out=out)
            APP.runJob("save:intrestoredcube", self._saveImage_worker, io=0,
                       args=(_images.readwrite(), "intrestoredcube",), kwargs=dict(
                    ImageName="%s.cube.int.restored" % self.BaseName, Fits=True, delete=True,
                    beam=self.FWHMBeamAvg, beamcube=self.FWHMBeam, Freqs=self.VS.FreqBandCenters,
                    Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        APP.runJob("del:intcubes", self._delSharedImage_worker, io=0, args=[_images.readwrite(), "intconvmodelcube", "intrestoredcube"])

        #  can delete this one now
        APP.runJob("del:intmodelcube", self._delSharedImage_worker, io=0, args=[_images.readwrite(), "intmodelcube"])
        # apparent-flux residual cube
        if "r" in self._savecubes:
            APP.runJob("save:apprescube", self._saveImage_worker, io=0, args=( self.DicoDirty.readonly(), "ImageCube",),
                       kwargs=dict(ImageName="%s.cube.app.residual"%(self.BaseName),Fits=True,
                                Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        # apparent-flux restored image cube
        if "i" in self._savecubes:
            a, b = apprescube(), appconvmodelcube()
            out = _images.addSharedArray('apprestoredcube', a.shape, a.dtype)
            numexpr.evaluate('a+b', out=out)
            APP.runJob("save:apprestoredcube", self._saveImage_worker, io=0,
                       args=(_images.readwrite(), "apprestoredcube",), kwargs=dict(
                    ImageName="%s.cube.app.restored" % self.BaseName, Fits=True, delete=True,
                    beam=self.FWHMBeamAvg, beamcube=self.FWHMBeam, Freqs=self.VS.FreqBandCenters,
                    Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        #  can delete this one now
        APP.runJob("del:appcubes", self._delSharedImage_worker, io=0, args=[_images.readwrite(), "appconvmodelcube", "apprescube"])
        # intrinsic-flux residual cube
        if havenorm and "R" in self._savecubes:
            intrescube()
            APP.runJob("save:intrescube", self._saveImage_worker, io=0, args=( _images.readonly(), "intrescube",),
                       kwargs=dict(ImageName="%s.cube.int.residual"%(self.BaseName),Fits=True,
                                   Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts()))
        #  can delete this one now
        APP.runJob("del:sqrtnormcube", self._delSharedImage_worker, io=0, args=[_images.readwrite(), "sqrtnormcube"])

        APP.awaitJobResults(["save:*", "del:*"])

    def testDegrid(self):
        import pylab
        self.InitFacetMachine()

        self.FacetMachine.ReinitDirty()
        Res=self.setNextData()
        #if Res=="EndChunk": break

        DATA=self.DATA


        # ###########################################
        # self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),DATA["Weights"],doStack=True)
        # testImage=self.FacetMachine.FacetsToIm()
        # testImage.fill(0)
        # _,_,nx,_=testImage.shape
        # print "shape image:",testImage.shape
        # xc=nx/2
        # n=2
        # dn=200
        # #for i in range(-n,n+1):
        # #   for j in range(-n,n+1):
        # #       testImage[0,0,int(xc+i*dn),int(xc+j*dn)]=100.
        # # for i in range(n+1):
        # #     testImage[0,0,int(xc+i*dn),int(xc+i*dn)]=100.
        # testImage[0,0,200,400]=100.
        # #testImage[0,0,xc+200,xc+300]=100.
        # self.FacetMachine.ToCasaImage(ImageIn=testImage, ImageName="testImage",Fits=True)
        # stop
        # ###########################################

        #testImage=np.zeros((1, 1, 1008, 1008),np.complex64)

        im=image("lala2.nocompDeg3.model.fits")
        testImageIn=im.getdata()
        nchan,npol,_,_=testImageIn.shape
        print testImageIn.shape
        testImage=np.zeros_like(testImageIn)
        for ch in range(nchan):
            for pol in range(npol):
                testImage[ch,pol,:,:]=testImageIn[ch,pol,:,:].T[::-1,:]#*1.0003900000000001

        visData=DATA["data"].copy()
        DATA["data"].fill(0)
        PredictedDataName="%s%s"%(self.IdSharedMem,"predicted_data")
        visPredict=NpShared.zeros(PredictedDataName,visData.shape,visData.dtype)

        _=self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],visPredict,DATA["flags"],(DATA["A0"],DATA["A1"]),testImage)


        DATA["data"]*=-1

        A0,A1=DATA["A0"],DATA["A1"]
        fig=pylab.figure(1)
        os.system("rm -rf png/*.png")
        op0=np.real
        op1=np.angle
        for iAnt in [0]:#range(36)[::-1]:
            for jAnt in [26]:#range(36)[::-1]:

                ind=np.where((A0==iAnt)&(A1==jAnt))[0]
                if ind.size==0: continue
                d0=visData[ind,0,0]
                u,v,w=DATA["uvw"][ind].T
                if np.max(d0)<1e-6: continue

                d1=DATA["data"][ind,0,0]
                pylab.clf()
                pylab.subplot(3,1,1)
                pylab.plot(op0(d0))
                pylab.plot(op0(d1))
                pylab.plot(op0(d0)-op0(d1))
                pylab.plot(np.zeros(d0.size),ls=":",color="black")
                pylab.subplot(3,1,2)
                #pylab.plot(op1(d0))
                #pylab.plot(op1(d1))
                pylab.plot(op1(d0/d1))
                pylab.plot(np.zeros(d0.size),ls=":",color="black")
                pylab.title("%s"%iAnt)
                pylab.subplot(3,1,3)
                pylab.plot(w)
                pylab.draw()
                #fig.savefig("png/resid_%2.2i_%2.2i.png"%(iAnt,jAnt))
                pylab.show(False)


        DATA["data"][:,:,:]=visData[:,:,:]-DATA["data"][:,:,:]

        self.FacetMachine.putChunk()
        Image=self.FacetMachine.FacetsToIm()
        self.ResidImage=Image
        #self.FacetMachine.ToCasaImage(ImageName="test.residual",Fits=True)
        self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="test.residual",Fits=True)


        m0=-0.02
        m1=0.02
        pylab.figure(2)
        pylab.clf()
        pylab.imshow(Image[0,0],interpolation="nearest",vmin=m0,vmax=m1)
        pylab.colorbar()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)

        time.sleep(2)

