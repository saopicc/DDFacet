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
import copy
from DDFacet.Other import Multiprocessing
from DDFacet.Other import AsyncProcessPool
from DDFacet.Other.AsyncProcessPool import APP
import cPickle
log=MyLogger.getLogger("ClassImagerDeconv")
import pyfits
from DDFacet.Array import SharedDict

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
                 PointingID=0,BaseName="ImageTest2",ReplaceDico=None,IdSharedMem="CACA.",
                 data=True, psf=True, readcol=True, deconvolve=True):
        # if ParsetFile is not None:
        #     GD=ClassGlobalData(ParsetFile)
        #     self.GD=GD

        if GD is not None:
            self.GD=GD

        self.BaseName=BaseName
        self.DicoModelName="%s.DicoModel"%self.BaseName
        self.DicoMetroModelName="%s.Metro.DicoModel"%self.BaseName
        self.PointingID=PointingID
        self.do_data, self.do_psf, self.do_readcol, self.do_deconvolve = data, psf, readcol, deconvolve
        self.CurrentDicoResidImage=None
        self.FacetMachine=None
        self.PSF=None
        self.FWHMBeam = None
        self.PSFGaussPars = None
        self.PSFSidelobes = None
        self.FWHMBeamAvg = None
        self.PSFGaussParsAvg = None
        self.PSFSidelobesAvg = None
        self.HasFittedPSFBeam=False


        self.VisWeights=None
        self.DATA=None
        self.Precision=self.GD["RIME"]["Precision"]#"S"
        self.PolMode=self.GD["RIME"]["PolMode"]
        self.PSFFacets = self.GD["Facets"]["PSFFacets"]
        self.HasDeconvolved=False
        self.Parallel = self.GD["Parallel"]["NCPU"] != 1
        self.IdSharedMem=IdSharedMem
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

        ## disabling this, as it doesn't play nice with in-place FFTs
        # self._save_intermediate_grids = self.GD["Debug"]["SaveIntermediateDirtyImages"]

        # init process pool for parallelization
        Multiprocessing.initDefaultPool(GD=self.GD)

    def Init(self):
        DC = self.GD
        mslist = ClassMS.expandMSList(DC["Data"]["MS"],
                                      defaultDDID=DC["Selection"]["DDID"],
                                      defaultField=DC["Selection"]["Field"])
        AsyncProcessPool.init(ncpu=self.GD["Parallel"]["NCPU"], affinity=self.GD["Parallel"]["Affinity"],
                              verbose=self.GD["Debug"]["APPVerbose"])

        self.VS = ClassVisServer.ClassVisServer(mslist,ColName=self.do_readcol and DC["Data"]["ColName"],
                                                TChunkSize=DC["Data"]["ChunkHours"],
                                                GD=self.GD)

        if self.do_deconvolve:
            self.NMajor=self.GD["Deconv"]["MaxMajorIter"]
            del(self.GD["Deconv"]["MaxMajorIter"])
            # If we do the deconvolution construct a model according to what is in MinorCycleConfig
            MinorCycleConfig=dict(self.GD["Deconv"])
            MinorCycleConfig["NCPU"]=self.GD["Parallel"]["NCPU"]
            MinorCycleConfig["NBand"]=MinorCycleConfig["NFreqBands"]=self.VS.NFreqBands
            MinorCycleConfig["GD"] = self.GD
            MinorCycleConfig["ImagePolDescriptor"] = self.VS.StokesConverter.RequiredStokesProducts()


            SubstractModel=self.GD["Data"]["InitDicoModel"]
            DoSub=(SubstractModel!="")&(SubstractModel is not None)
            if DoSub:
                print>>log, ModColor.Str("Initialise sky model using %s"%SubstractModel,col="blue")
                ModelMachine = self.ModConstructor.GiveInitialisedMMFromFile(SubstractModel)
                if self.GD["Deconv"]["Mode"] != ModelMachine.DicoSMStacked["Type"]:
                    raise NotImplementedError("You want to use different minor cycle and IniDicoModel types [%s vs %s]"\
                                              %(self.GD["Deconv"]["Mode"],ModelMachine.DicoSMStacked["Type"]))
            
                print>>log, ModColor.Str("Taking reference frequency from the model machine %f MHz (instead of %f MHz from the data)"%(ModelMachine.RefFreq/1e6,self.VS.RefFreq/1e6))
                self.RefFreq=self.VS.RefFreq=ModelMachine.RefFreq
                
                if self.BaseName==self.GD["Data"]["InitDicoModel"][0:-10]:
                    self.BaseName+=".continue"
                    self.DicoModelName="%s.DicoModel"%self.BaseName
                    self.DicoMetroModelName="%s.Metro.DicoModel"%self.BaseName
            else:
                ModelMachine = self.ModConstructor.GiveMM(Mode=self.GD["Deconv"]["Mode"])
                ModelMachine.setRefFreq(self.VS.RefFreq)
                self.RefFreq=self.VS.RefFreq
            
            self.ModelMachine=ModelMachine

            MinorCycleConfig["ModelMachine"] = ModelMachine

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
                from DDFacet.Imager.HOGBOM import ClassImageDeconvMachineHogbom
                self.DeconvMachine=ClassImageDeconvMachineHogbom.ClassImageDeconvMachine(**MinorCycleConfig)
                print>>log,"Using Hogbom algorithm"
            else:
                raise NotImplementedError("Unknown --Deconvolution-Mode setting '%s'" % self.GD["Deconv"]["Mode"])

        self.CreateFacetMachines()
        self.VS.setFacetMachine(self.FacetMachine or self.FacetMachinePSF)


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
        self.FacetMachine = self.FacetMachinePSF = None
        MainFacetOptions = self.GiveMainFacetOptions()
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


    def _createDirtyPSFCacheKey(self):
        """Creates cache key used for Dirty and PSF caches"""
        return dict([("MSNames", [ms.MSName for ms in self.VS.ListMS])] +
                    [(section, self.GD[section]) for section in 
                     "Data", "Beam", "Selection",
                     "Freq", "Image", "Comp",
                     "CF", "RIME","Facets","Weight","DDESolutions"]
                )


    def _checkForCachedPSF (self, sparsify, key=None):
        self._psf_cachepath, valid = self.VS.maincache.checkCache("PSF", key or self._createDirtyPSFCacheKey(),
                                            reset=self.GD["Cache"]["ResetPSF"] or sparsify)
        return self._psf_cachepath, valid

    def _loadCachedPSF (self, cachepath):
        import cPickle
        #self.DicoVariablePSF = cPickle.load(file(cachepath))
        self.DicoVariablePSF = MyPickle.FileToDicoNP(cachepath)
        self.DicoVariablePSF = SharedDict.dict_to_shm("dictPSF",self.DicoVariablePSF)

        # if we load a cached PSF, mark these as None so that we don't re-save a PSF image in _fitAndSavePSF()
        self._psfmean = self._psfcube = None
        self.PSF = self.MeanFacetPSF = self.DicoVariablePSF["MeanFacetPSF"]
        self.FWHMBeam=self.DicoVariablePSF["FWHMBeam"]
        self.PSFGaussPars=self.DicoVariablePSF["PSFGaussPars"]
        self.PSFSidelobes=self.DicoVariablePSF["PSFSidelobes"]
        (self.FWHMBeamAvg, self.PSFGaussParsAvg, self.PSFSidelobesAvg)=self.DicoVariablePSF["EstimatesAvgPSF"]
        self.HasFittedPSFBeam=True



    def _finalizeComputedPSF (self, FacetMachinePSF, sparsify):
        psfdict = FacetMachinePSF.FacetsToIm(NormJones=False)
        self._psfmean, self._psfcube = psfdict["MeanImage"], psfdict["ImagData"]  # this is only for the casa image saving
        self.DicoVariablePSF = FacetMachinePSF.DicoPSF
        self.PSF = self.MeanFacetPSF = self.DicoVariablePSF["MeanFacetPSF"]
        if self.GD["Cache"]["PSF"] and not sparsify:
            try:
                self.FitPSF()
                self.DicoVariablePSF["FWHMBeam"]=self.FWHMBeam
                self.DicoVariablePSF["PSFGaussPars"]=self.PSFGaussPars
                self.DicoVariablePSF["PSFSidelobes"]=self.PSFSidelobes
                self.DicoVariablePSF["EstimatesAvgPSF"]=(self.FWHMBeamAvg, self.PSFGaussParsAvg, self.PSFSidelobesAvg)
                #cPickle.dump(self.DicoVariablePSF, file(self._psf_cachepath, 'w'), 2)
                MyPickle.DicoNPToFile(self.DicoVariablePSF, self._psf_cachepath)
                self.VS.maincache.saveCache("PSF")
            except:
                print>> log, traceback.format_exc()
                print>> log, ModColor.Str(
                    "WARNING: PSF cache could not be written, see error report above. Proceeding anyway.")

    def _fitAndSavePSF (self, FacetMachinePSF, save=True, cycle=None):
        if not self.HasFittedPSFBeam:
            self.FitPSF()
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

    def MakePSF(self, sparsify=0):
        """
        Generates PSF.

        Args:
            sparsify: sparsification factor applied to data. 0 means calculate the most precise dirty possible.
        """
        if self.PSF is not None:
            return

        cachepath, valid = self._checkForCachedPSF(sparsify)


        if valid or self.GD["Cache"]["ResetPSF"]==-1:
            print>>log, ModColor.Str("============================ Loading cached PSF ==========================")
            print>>log, "found valid cached PSF in %s"%cachepath
            print>>log, ModColor.Str("As near as we can tell, we can reuse this cached PSF because it was produced")
            print>>log, ModColor.Str("with the same set of relevant DDFacet settings. If you think this is in error,")
            print>>log, ModColor.Str("or if your MS has been substantially flagged or otherwise had its uv-coverage")
            print>>log, ModColor.Str("affected, please remove the cache, or else run with --ResetPSF 1.")
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

            self._finalizeComputedPSF(self.FacetMachinePSF,1)

        self._fitAndSavePSF(self.FacetMachinePSF)


    def GiveDirty(self, psf=False, sparsify=0):
        """
        Generates dirty image (& PSF)

        Args:
            psf: if True, PSF is also generated
            sparsify: sparsification factor applied to data. 0 means calculate the most precise dirty possible.
        """
        # cache only used in "precise" mode. In approximative mode, things are super-quick anyway
        if not sparsify:
            cache_key = self._createDirtyPSFCacheKey()
            dirty_cachepath, dirty_valid = self.VS.maincache.checkCache("Dirty", cache_key,
                                                                        reset=self.GD["Cache"]["ResetDirty"])
            if psf:
                psf_cachepath, psf_valid = self._checkForCachedPSF(sparsify, key=cache_key)
        else:
            dirty_valid = psf_valid = False


        if dirty_valid and (not psf or psf_valid):
            if psf:
                print>>log, ModColor.Str("============================ Loading cached dirty image & PSF ===================")
                print>>log, "found valid cached dirty image in %s"%dirty_cachepath
                print>>log, "found valid cached PSF in %s"%psf_cachepath
            else:
                print>>log, ModColor.Str("============================ Loading cached dirty image =========================")
                print>>log, "found valid cached dirty image in %s"%dirty_cachepath
            print>>log, ModColor.Str("As near as we can tell, we can reuse this cache because it was produced")
            print>>log, ModColor.Str("with the same set of relevant DDFacet settings. If you think this is in error,")
            print>>log, ModColor.Str("or if your MS has changed, please remove the cache, or run with --ResetDirty 1.")

            #self.DicoDirty = cPickle.load(file(dirty_cachepath))
            self.DicoDirty = MyPickle.FileToDicoNP(dirty_cachepath)

            if self.DicoDirty["JonesNorm"] is not None:
                self.FacetMachine.setNormImages(self.DicoDirty)
                self.FacetMachinePSF.setNormImages(self.DicoDirty)
            else:
                self.MeanNormImage = None
            if psf:
                self._loadCachedPSF(psf_cachepath)
        else:
            if psf:
                print>>log, ModColor.Str("============================== Making Dirty Image & PSF ========================")
            else:
                print>>log, ModColor.Str("============================== Making Dirty Image ==============================")
            # tell the I/O thread to go load the first chunk
            self.VS.ReInitChunkCount()
            self.VS.startChunkLoadInBackground()


            self.FacetMachine.ReinitDirty()
            if self.FacetMachinePSF is not None:
                self.FacetMachinePSF.ReinitDirty()


            SubtractModel = self.GD["Data"]["InitDicoModel"]
            if SubtractModel:
                print>>log, ModColor.Str("Initialise sky model using %s"%SubtractModel,col="blue")
                # Load model dict
            	#DicoSMStacked = MyPickle.Load(SubtractModel)
                # Get the correct model machine from SubtractModel file
            	ModelMachine = self.ModelMachine#self.ModConstructor.GiveMMFromDico(DicoSMStacked)
                #ModelMachine.FromDico(DicoSMStacked)
                #self.FacetMachine.BuildFacetNormImage()
                # NormFacetsFile="%s.NormFacets.fits"%InitBaseName
                # if InitBaseName!=BaseName:
                #     print>>log, ModColor.Str("You are substracting a model build from a different facetting mode")
                #     print>>log, ModColor.Str("  This is rather dodgy because of the ")
                # self.FacetMachine.NormImage=ClassCasaImage.FileToArray(NormFacetsFile,True)
                # _,_,nx,nx=self.FacetMachine.NormImage.shape
                # self.FacetMachine.NormImage=self.FacetMachine.NormImage.reshape((nx,nx))
                if self.BaseName==self.GD["Data"]["InitDicoModel"][0:-10]:
                    self.BaseName+=".continue"
                current_model_freqs = np.array([])
                ModelImage = None

            iloop = 0
            while True:
                # note that collectLoadedChunk() will destroy the current DATA dict, so we must make sure
                # the gridding jobs of the previous chunk are finished
                self.FacetMachine.collectGriddingResults()
                if self.FacetMachinePSF is not None:
                    self.FacetMachinePSF.collectGriddingResults()


                # get loaded chunk from I/O thread, schedule next chunk
                # self.VS.startChunkLoadInBackground()
                DATA = self.VS.collectLoadedChunk(start_next=True)
                if type(DATA) is str:
                    print>>log,ModColor.Str("no more data: %s"%DATA, col="red")
                    break
                # None weights indicates an all-flagged chunk: go on to the next chunk
                if DATA["Weights"] is None:
                    continue
                print>>log,"sparsify %f"%sparsify
                self.FacetMachine.applySparsification(DATA, sparsify)

                if SubtractModel:
                    ## redo model image if needed
                    model_freqs = DATA["FreqMappingDegrid"]
                    if not np.array_equal(model_freqs, current_model_freqs):
                        ModelImage = self.FacetMachine.setModelImage(self.DeconvMachine.GiveModelImage(model_freqs))
                        current_model_freqs = model_freqs
                        print>> log, "model image @%s MHz (min,max) = (%f, %f)" % (
                        str(model_freqs / 1e6), ModelImage.min(), ModelImage.max())
                    else:
                        print>> log, "reusing model image from previous chunk"
                    self.FacetMachine.getChunkInBackground(DATA)
                    self.FacetMachine.collectDegriddingResults()

                self.FacetMachine.putChunkInBackground(DATA)
                if self.FacetMachinePSF is not None:
                    self.FacetMachinePSF.putChunkInBackground(DATA)
                ## disabled this, doesn't like in-place FFTs
                # # collect intermediate grids, if asked to
                # if self._save_intermediate_grids:
                #     self.DicoDirty=self.FacetMachine.FacetsToIm(NormJones=True)
                #     self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.dirty.%d."%(self.BaseName,iloop),
                #                                   Fits=True,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                #     if 'g' in self._savecubes:
                #         self.FacetMachine.ToCasaImage(self.DicoDirty["ImagData"],ImageName="%s.cube.dirty.%d"%(self.BaseName,iloop),
                #             Fits=True,Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                #     self.FacetMachine.JonesNorm = None
                #     self.FacetMachine.NormImage = None

                iloop += 1

            self.DicoDirty=self.FacetMachine.FacetsToIm(NormJones=True)
            
            self.FacetMachine.ComputeSmoothBeam()
            self.SaveDirtyProducts()


            if "d" in self._saveims:
                self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.dirty"%self.BaseName,Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            if "d" in self._savecubes:
                self.FacetMachine.ToCasaImage(self.DicoDirty["ImagData"],ImageName="%s.cube.dirty"%self.BaseName,
                        Fits=True,Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())

            if "n" in self._saveims:
                self.FacetMachine.ToCasaImage(self.FacetMachine.FacetNormReShape,ImageName="%s.NormFacets"%self.BaseName,
                                              Fits=True)

            if self.DicoDirty["JonesNorm"] is not None:
                DirtyCorr = self.DicoDirty["ImagData"]/np.sqrt(self.DicoDirty["JonesNorm"])
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
                if "N" in self._saveims:
                    self.FacetMachine.ToCasaImage(self.MeanJonesNorm,ImageName="%s.Norm"%self.BaseName,Fits=True,
                                                  Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                if "N" in self._savecubes:
                    self.FacetMachine.ToCasaImage(self.JonesNorm, ImageName="%s.cube.Norm" % self.BaseName,
                                                  Fits=True, Freqs=self.VS.FreqBandCenters,
                                                  Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            else:
                self.MeanJonesNorm = None

            # dump dirty to cache
            if self.GD["Cache"]["Dirty"] and not sparsify:
                try:
                    #cPickle.dump(self.DicoDirty, file(cachepath, 'w'), 2)
                    MyPickle.DicoNPToFile(self.DicoDirty, dirty_cachepath)
                    self.VS.maincache.saveCache("Dirty")
                except:
                    print>> log, traceback.format_exc()
                    print>> log, ModColor.Str("WARNING: Dirty image cache could not be written, see error report above. Proceeding anyway.")
            if psf:
                self._finalizeComputedPSF(self.FacetMachinePSF, sparsify)

        # finalize other PSF initialization
        if psf:
            self._fitAndSavePSF(self.FacetMachinePSF)

        self.CurrentDicoResidImage = self.DicoDirty
        self.ResidCube  = self.CurrentDicoResidImage["ImagData"] #get residuals cube
        self.ResidImage = self.CurrentDicoResidImage["MeanImage"]
        return self.DicoDirty["MeanImage"]

    def SaveDirtyProducts(self):
        if "d" in self._saveims:
            self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.dirty"%self.BaseName,Fits=True,
                                          Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        if "d" in self._savecubes:
            self.FacetMachine.ToCasaImage(self.DicoDirty["ImagData"],ImageName="%s.cube.dirty"%self.BaseName,
                                          Fits=True,Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())


        if "n" in self._saveims:
            self.FacetMachine.ToCasaImage(self.FacetMachine.FacetNormReShape,ImageName="%s.NormFacets"%self.BaseName,
                                          Fits=True)
                
        if self.DicoDirty["JonesNorm"] is not None:
            DirtyCorr = self.DicoDirty["ImagData"]/np.sqrt(self.DicoDirty["JonesNorm"])
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

            if "H" in self._saveims and self.FacetMachine.SmoothMeanJonesNorm is not None:
                self.FacetMachine.ToCasaImage(self.FacetMachine.SmoothMeanJonesNorm,ImageName="%s.SmoothNorm"%self.BaseName,Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())


            if "N" in self._saveims:
                self.FacetMachine.ToCasaImage(self.MeanJonesNorm,ImageName="%s.Norm"%self.BaseName,Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            if "N" in self._savecubes:
                self.FacetMachine.ToCasaImage(self.JonesNorm, ImageName="%s.cube.Norm" % self.BaseName,
                                              Fits=True, Freqs=self.VS.FreqBandCenters,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        else:
            self.MeanJonesNorm = None


    def GivePredict(self,from_fits=True):
        print>>log, ModColor.Str("============================== Making Predict ==============================")

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
        CleanMaskImageName=self.GD["Deconv"]["CleanMaskImage"]
        if CleanMaskImageName is not None and CleanMaskImageName is not "":
            print>>log,ModColor.Str("Will use mask image %s for the predict"%CleanMaskImageName)
            CleanMaskImage = np.bool8(ClassCasaImage.FileToArray(CleanMaskImageName,True))


        modelfile = self.GD["Data"]["PredictFrom"]


        # if model is a dict, init model machine with that
        # else we use a model image and hope for the best (need to fix frequency axis...)
        # 
        if modelfile is not None and modelfile is not "":
            print>>log,ModColor.Str("Reading image file for the predict: %s"%modelfile)
            FixedModelImage = ClassCasaImage.FileToArray(modelfile,True)

        current_model_freqs = np.array([])
        ModelImage = None

        while True:
            # get loaded chunk from I/O thread, schedule next chunk
            # self.VS.startChunkLoadInBackground()
            DATA = self.VS.collectLoadedChunk(start_next=True)
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
                    ModelImage = self.FacetMachine.setModelImage(self.DeconvMachine.GiveModelImage(model_freqs))
                    current_model_freqs = model_freqs
                    print>> log, "model image @%s MHz (min,max) = (%f, %f)" % (
                    str(model_freqs / 1e6), ModelImage.min(), ModelImage.max())
                else:
                    print>> log, "reusing model image from previous chunk"
            else:
                if ModelImage is None:
                    ModelImage = self.FacetMachine.setModelImage(FixedModelImage)

            if self.GD["Images"]["MaskSquare"] is not None:
                # MaskInside: choose mask inside (0) or outside (1) 
                # NpixInside: Size of the masking region
                MaskOutSide,NpixInside = self.GD["Images"]["MaskSquare"]
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

            if ModelImage.shape[0]!=self.VS.CurrentChanMappingDegrid.size:
                print>>log, "The image model channels and targetted degridded visibilities channels have different sizes (%i vs %i respectively)"%(ModelImage.shape[0],self.VS.CurrentChanMappingDegrid.size)
                if ModelImage.shape[0]==1:
                    print>>log, " Matching freq size of model image to visibilities"
                    ModelImage=ModelImage*np.ones((self.VS.CurrentChanMappingDegrid.size,1,1,1))


            if CleanMaskImage is not None:
                nch,npol,_,_=ModelImage.shape
                indZero=(CleanMaskImage[0,0]!=0)
                for ich in range(nch):
                    for ipol in range(npol):
                        ModelImage[ich,ipol][indZero]=0

            self.FacetMachine.ToCasaImage(ModelImage,ImageName="%s.modelPredict"%self.BaseName,Fits=True,
                                          Stokes=self.VS.StokesConverter.RequiredStokesProducts())


            if self.PredictMode == "BDA-degrid" or self.PredictMode == "DeGridder":  # latter for backwards compatibility
                self.FacetMachine.getChunkInBackground(DATA)
            elif self.PredictMode == "Montblanc":
                from ClassMontblancMachine import ClassMontblancMachine
                model = self.DeconvMachine.ModelMachine.GiveModelList()
                mb_machine = ClassMontblancMachine(self.GD, self.FacetMachine.Npix, self.FacetMachine.CellSizeRad)
                mb_machine.getChunk(DATA, predict, model, self.VS.CurrentMS)
                mb_machine.close()
            else:
                raise ValueError("Invalid PredictMode '%s'" % self.PredictMode)
            self.FacetMachine.collectDegriddingResults()
            predict *= -1   # model was subtracted from (zero) predict, so need to invert sign
            # run job in I/O thread
            self.VS.startVisPutColumnInBackground(DATA, "data", self.GD["Data"]["PredictColName"], likecol=self.GD["Data"]["ColName"])
            # and wait for it to finish (we don't want DATA destroyed, which collectLoadedChunk() above will)
            self.VS.collectPutColumnResults()
        # if from_fits:
        #     print "Predicting from fits and saving in %s",(PredictColName)
        #     #Read in the LSM
        #     Imtodegrid = pyfits.open("/home/landman/Projects/Processed_Images/ddfacet_out/Test-D147-HI-NOIFS-NOPOL-4M5Sa/model-true.fits")[0].data
        #     print ModelImage.shape, Imtodegrid.shape
        #     self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),Imtodegrid)
        #     vis = - DATA["data"]
        #     PredictColName = self.GD["Data"]["PredictColName"]
        #
        #     MSName = self.VS.CurrentMS.MSName
        #     print>>log, "Writing data predicted from Tigger LSM to column %s of %s"%(PredictColName,MSName)
        #     self.VS.CurrentMS.PutVisColumn(PredictColName, vis)



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
            sparsify = 0
        if sparsify:
            print>> log, "applying a sparsification factor of %f to data for dirty image" % sparsify
        self.GiveDirty(psf=True, sparsify=sparsify)

        # if we reached a sparsification of 1, we shan't be re-making the PSF
        if not sparsify:
            self.FacetMachinePSF = None

        #Pass minor cycle specific options into Init as kwargs
        self.DeconvMachine.Init(PSFVar=self.DicoVariablePSF, PSFAve=self.PSFSidelobesAvg,
                                approx=(sparsify > approximate_psf_above), cache=not sparsify,
                                GridFreqs=self.VS.FreqBandCenters)

        DicoImage=self.DicoDirty
        continue_deconv = True

        for iMajor in range(1, NMajor+1):
            # previous minor loop indicated it has reached bottom? Break out
            if not continue_deconv:
                break

            print>>log, ModColor.Str("========================== Running major cycle %i ========================="%(iMajor-1))

            self.DeconvMachine.Update(DicoImage)

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

            self.DeconvMachine.ModelMachine.ToFile(self.DicoModelName) 

            ## returned with nothing done in minor cycle? Break out
            if not update_model or iMajor == NMajor:
                continue_deconv = False
                print>> log, "This is the last major cycle"
            else:
                print>> log, "Finished Deconvolving for this major cycle... Going back to visibility space."
            predict_colname = not continue_deconv and self.GD["Data"]["PredictColName"]

            # in the meantime, tell the I/O thread to go reload the first data chunk
            self.VS.ReInitChunkCount()
            self.VS.startChunkLoadInBackground()


            #self.ResidImage=DicoImage["MeanImage"]
            #self.FacetMachine.ToCasaImage(DicoImage["MeanImage"],ImageName="%s.residual_sub%i"%(self.BaseName,iMajor),Fits=True)

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
            self.FacetMachine.ReinitDirty()
            if sparsify or previous_sparsify:
                print>>log, "applying a sparsification factor of %f (was %f in previous cycle)" % (sparsify, previous_sparsify)
            if do_psf:
                print>>log, "the PSF will be recomputed"
                # check PSF cache to make sure paths are set up
                if not sparsify:
                    self._checkForCachedPSF(self, sparsify)
                self.FacetMachinePSF.ReinitDirty()
            previous_sparsify = sparsify

            current_model_freqs = np.array([])
            ModelImage = None

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
                    current_model_freqs = model_freqs
                    print>>log,"model image @%s MHz (min,max) = (%f, %f)"%(str(model_freqs/1e6),ModelImage.min(),ModelImage.max())
                else:
                    print>>log,"reusing model image from previous chunk"

                if "o" in self._saveims:
                    # self.FacetMachine.ToCasaImage(ModelImage,ImageName="%s.model%2.2i"%(self.BaseName,iMajor),
                    #     Fits=True,Freqs=current_model_freqs,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                    nf,npol,nx,nx=ModelImage.shape
                    ModelImageAvg=np.mean(ModelImage,axis=0).reshape((1,npol,nx,nx))
                    
                    self.FacetMachine.ToCasaImage(ModelImageAvg,ImageName="%s.model%2.2i"%(self.BaseName,iMajor),
                                                  Fits=True)#,Freqs=current_model_freqs,Stokes=self.VS.StokesConverter.RequiredStokesProducts())


                if predict_colname:
                    print>>log,"last major cycle: model visibilities will be stored to %s"%predict_colname

                if self.PredictMode == "BDA-degrid" or self.PredictMode == "DeGridder":
                    self.FacetMachine.getChunkInBackground(DATA)
                elif self.PredictMode == "Montblanc":
                    from ClassMontblancMachine import ClassMontblancMachine
                    model = self.DeconvMachine.ModelMachine.GiveModelList()
                    mb_machine = ClassMontblancMachine(self.GD, self.FacetMachine.Npix, self.FacetMachine.CellSizeRad)
                    mb_machine.getChunk(DATA, DATA["data"], model, self.VS.CurrentMS)
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

                self.FacetMachine.putChunkInBackground(DATA)
                if do_psf:
                    self.FacetMachinePSF.putChunkInBackground(DATA)

            # write out model image, if asked to
            if "o" in self._saveims:
                self.FacetMachine.ToCasaImage(ModelImage, ImageName="%s.model%2.2i" % (self.BaseName, iMajor),
                                              Fits=True, Freqs=current_model_freqs,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            # wait for gridding to finish
            self.FacetMachine.collectGriddingResults()
            self.VS.collectPutColumnResults()  # if these were going on
            # release model image from memory
            ModelImage = None
            self.FacetMachine.releaseModelImage()
            # create new residual image
            DicoImage = self.FacetMachine.FacetsToIm(NormJones=True)

            self.ResidCube  = DicoImage["ImagData"] #get residuals cube
            self.ResidImage = DicoImage["MeanImage"]
            # was PSF re-generated?
            if do_psf:
                self._finalizeComputedPSF(self.FacetMachinePSF, sparsify)
                self._fitAndSavePSF(self.FacetMachinePSF, cycle=iMajor)
                self.DeconvMachine.Init(PSFVar=self.DicoVariablePSF, PSFAve=self.PSFSidelobesAvg,
                                        approx=(sparsify > approximate_psf_above),
                                        cache=not sparsify)

            # if we reached a sparsification of 1, we shan't be re-making the PSF
            if sparsify <= 1:
                self.FacetMachinePSF = None

            # if "SmoothMeanNormImage" in DicoImage.keys():
            #     self.SmoothMeanNormImage=DicoImage["SmoothMeanNormImage"]

            if "e" in self._saveims:
                self.FacetMachine.ToCasaImage(self.ResidImage,ImageName="%s.residual%2.2i"%(self.BaseName,iMajor),
                                              Fits=True,Stokes=self.VS.StokesConverter.RequiredStokesProducts())

            # write out current model, using final or intermediate name
            if continue_deconv:
                self.DeconvMachine.ToFile("%s.%2.2i.DicoModel" % (self.BaseName, iMajor) )
            else:
                self.DeconvMachine.ToFile(self.DicoModelName)


            self.HasDeconvolved=True

        # dump dirty to cache
        if self.GD["Cache"]["LastResidual"] and self.CurrentDicoResidImage is not None:
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
                MyPickle.DicoNPToFile(self.CurrentDicoResidImage, cachepath)
                self.VS.maincache.saveCache("LastResidual")
            except:
                print>> log, traceback.format_exc()
                print>> log, ModColor.Str("WARNING: Dirty image cache could not be written, see error report above. Proceeding anyway.")


        self.Restore()

        # if self.HasDeconvolved:
        #     self.Restore()

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

        off = min(off, x[0], nx-x[0], y[0], ny-y[0])
        print>> log, "Fitting %s PSF in a [%i,%i] box ..." % (label, off * 2, off * 2)
        P = PSF[0, x[0] - off:x[0] + off, y[0] - off:y[0] + off]
        

        sidelobes = ModFitPSF.FindSidelobe(P)
        bmaj, bmin, theta = ModFitPSF.FitCleanBeam(P)

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
        """
        # If set, use the parameter RestoringBeam to fix the clean beam parameters
        forced_beam=self.GD["Output"]["RestoringBeam"]
        if forced_beam is not None:
            FWHMFact = 2. * np.sqrt(2. * np.log(2.))

            if isinstance(forced_beam,float):
                forced_beam=[forced_beam,forced_beam,0]
            elif len(forced_beam)==1:
                forced_beam=[forced_beam[0],forced_beam[0],0]
            f_beam=(forced_beam[0]/3600.0,forced_beam[1]/3600.0,forced_beam[2])
            f_gau=(np.deg2rad(f_beam[0])/FWHMFact,np.deg2rad(f_beam[1])/FWHMFact,np.deg2rad(f_beam[2]))
        PSF = self.DicoVariablePSF["CubeVariablePSF"][self.FacetMachine.iCentralFacet]

        off=self.GD["Image"]["SidelobeSearchWindow"] // 2
        beam, gausspars, sidelobes = self.fitSinglePSF(self.MeanFacetPSF[0,...], "mean")
        if forced_beam is not None:
            print>>log, 'Will use user-specified beam: bmaj=%f, bmin=%f, bpa=%f degrees' % f_beam
            beam, gausspars = f_beam, f_gau
        self.FWHMBeamAvg, self.PSFGaussParsAvg, self.PSFSidelobesAvg = beam, gausspars, sidelobes

        # MeanFacetPSF has a shape of 1,1,nx,ny, so need to cut that extra one off
        if self.VS.MultiFreqMode:
            self.FWHMBeam = []
            self.PSFGaussPars = []
            self.PSFSidelobes = []
            for band in range(self.VS.NFreqBands):
                beam, gausspars, sidelobes = self.fitSinglePSF(PSF[band,...],off,"band %d"%band)
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
        self.HasFittedPSFBeam=True
        
        
        ## LB - Remove his chunk ?
        #theta=np.pi/2-theta
        #
        #FWHMFact=2.*np.sqrt(2.*np.log(2.))
        #bmaj=np.max([sigma_x, sigma_y])*self.CellArcSec*FWHMFact
        #bmin=np.min([sigma_x, sigma_y])*self.CellArcSec*FWHMFact
        #self.FWHMBeam=(bmaj/3600.,bmin/3600.,theta)
        #self.PSFGaussPars = (sigma_x*self.CellSizeRad, sigma_y*self.CellSizeRad, theta)
        #print>>log, "Fitted PSF (sigma): (Sx, Sy, Th)=(%f, %f, %f)"%(sigma_x*self.CellArcSec, sigma_y*self.CellArcSec, theta)
        #print>>log, "Fitted PSF (FWHM):  (Sx, Sy, Th)=(%f, %f, %f)"%(sigma_x*self.CellArcSec*FWHMFact, sigma_y*self.CellArcSec*FWHMFact, theta)
        #print>>log, "Secondary sidelobe at the level of %5.1f at a position of %i from the center"%(self.SideLobeLevel,self.OffsetSideLobe)

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
        # DeconvMachine.Init(PSFVar=self.DicoVariablePSF,PSFAve=self.PSFSidelobesAvg)
        
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
        print>>log,"Runing a Metropolis-Hastings MCMC on islands larger than %i pixels"%self.GD["SSDClean"]["RestoreMetroSwitch"]
        DeconvMachine.setDeconvMode(Mode="MetroClean")
        DeconvMachine.Update(self.CurrentDicoResidImage)
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


    def Restore(self):
        print>>log, "Create restored image"
        if self.PSFGaussPars is None:
            self.FitPSF()
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
                print>>log, ModColor.Str("Failed Putting back substracted components")

        # do we have a non-trivial norm (i.e. DDE solutions or beam)?
        # @cyriltasse: maybe there's a quicker way to check?
        havenorm = self.MeanJonesNorm is not None and (self.MeanJonesNorm != 1).any()

        # make a dict of _images to save the intermediate images for when we need them
        _images = {}
        def sqrtnorm():
            label = 'sqrtnorm'
            if label not in _images:
                _images[label] = np.sqrt(self.MeanJonesNorm) if havenorm else 1
            return _images[label]
        def sqrtnormcube():
            label = 'sqrtnormcube'
            if label not in _images:
                _images[label] = np.sqrt(self.JonesNorm) if havenorm else 1
            return _images[label]
        def appres():
            return self.ResidImage
        def intres():
            label = 'intres'
            if label not in _images:
                _images[label] = x = appres()/sqrtnorm() if havenorm else appres()
                x[~np.isfinite(x)] = 0
            return _images[label]
        def apprescube():
            return self.ResidCube
        def intrescube():
            label = 'intrescube'
            if label not in _images:
                _images[label] = x = apprescube()/sqrtnormcube() if havenorm else apprescube()
                x[~np.isfinite(x)] = 0
            return _images[label]
        def appmodel():
            label = 'appmodel'
            if label not in _images:
                _images[label] = intmodel()*sqrtnorm() if havenorm else intmodel()
            return _images[label]
        def intmodel():
            label = 'intmodel'
            if label not in _images:
                _images[label] = ModelMachine.GiveModelImage(RefFreq)
            return _images[label]
        def appmodelcube():
            label = 'appmodelcube'
            if label not in _images:
                _images[label] = intmodelcube()*sqrtnormcube() if havenorm else intmodel()
            return _images[label]
        def intmodelcube():
            label = 'intmodelcube'
            if label not in _images:
                _images[label] = ModelMachine.GiveModelImage(self.VS.FreqBandCenters)
            return _images[label]
        def appconvmodel():
            label = 'appconvmodel'
            if label not in _images:
                _images[label] = ModFFTW.ConvolveGaussian(appmodel(),CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussParsAvg]) \
                                    if havenorm else intconvmodel()
            return _images[label]
        def intconvmodel():
            label = 'intconvmodel'
            if label not in _images:
                _images[label] = ModFFTW.ConvolveGaussian(intmodel(),CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussParsAvg])
            return _images[label]
        def appconvmodelcube():
            label = 'appconvmodelcube'
            if label not in _images:
                _images[label] = ModFFTW.ConvolveGaussian(appmodelcube(),CellSizeRad=self.CellSizeRad,GaussPars=self.PSFGaussPars) \
                                    if havenorm else intconvmodelcube()
            return _images[label]
        def intconvmodelcube():
            label = 'intconvmodelcube'
            if label not in _images:
                _images[label] = ModFFTW.ConvolveGaussian(intmodelcube(),CellSizeRad=self.CellSizeRad,GaussPars=self.PSFGaussPars)
            return _images[label]

        # norm
        if havenorm and ("S" in self._saveims or "s" in self._saveims):
            self.FacetMachine.ToCasaImage(sqrtnorm(),ImageName="%s.fluxscale"%(self.BaseName),
                                          Fits=True,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        if havenorm and ("S" in self._savecubes or "s" in self._savecubes):
            self.FacetMachine.ToCasaImage(sqrtnormcube(), ImageName="%s.cube.fluxscale" % (self.BaseName), Fits=True,
                Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())

            # apparent-flux residuals
        if "r" in self._saveims:
            self.FacetMachine.ToCasaImage(appres(),ImageName="%s.app.residual"%(self.BaseName),
                                          Fits=True,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        # intrinsic-flux residuals
        if havenorm and "R" in self._saveims:
            self.FacetMachine.ToCasaImage(intres(),ImageName="%s.int.residual"%(self.BaseName),Fits=True,
                                          Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        
        # apparent-flux residual cube
        if "r" in self._savecubes:
            self.FacetMachine.ToCasaImage(apprescube(),ImageName="%s.cube.app.residual"%(self.BaseName),Fits=True,
                Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        # intrinsic-flux residual cube
        if havenorm and "R" in self._savecubes:
            self.FacetMachine.ToCasaImage(intrescube(),ImageName="%s.cube.int.residual"%(self.BaseName),Fits=True,
                Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())

        # apparent-flux model
        if "m" in self._saveims:
            self.FacetMachine.ToCasaImage(appmodel(),ImageName="%s.app.model"%self.BaseName,Fits=True,
                                          Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        # intrinsic-flux model
        if havenorm and "M" in self._saveims:
            self.FacetMachine.ToCasaImage(intmodel(),ImageName="%s.int.model"%self.BaseName,Fits=True,
                                          Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        # apparent-flux model cube
        if "m" in self._savecubes:
            self.FacetMachine.ToCasaImage(appmodelcube(),ImageName="%s.cube.app.model"%self.BaseName,Fits=True,
                Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        # intrinsic-flux model cube
        if havenorm and "M" in self._savecubes:
            self.FacetMachine.ToCasaImage(intmodelcube(),ImageName="%s.cube.int.model"%self.BaseName,Fits=True,
                Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())

        # convolved-model image in apparent flux
        if "c" in self._saveims:
            self.FacetMachine.ToCasaImage(appconvmodel(),ImageName="%s.app.convmodel"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        # convolved-model image in intrinsic flux
        if havenorm and "C" in self._saveims:
            self.FacetMachine.ToCasaImage(intconvmodel(),ImageName="%s.int.convmodel"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        # convolved-model cube in apparent flux
        if "c" in self._savecubes:
            self.FacetMachine.ToCasaImage(appconvmodelcube(),ImageName="%s.cube.app.convmodel"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,beamcube=self.FWHMBeam,Freqs=self.VS.FreqBandCenters,
                Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        # convolved-model cube in intrinsic flux
        if havenorm and "C" in self._savecubes:
            self.FacetMachine.ToCasaImage(intconvmodelcube(),ImageName="%s.cube.int.convmodel"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,beamcube=self.FWHMBeam,Freqs=self.VS.FreqBandCenters,
                Stokes=self.VS.StokesConverter.RequiredStokesProducts())

        # apparent-flux restored image
        if "i" in self._saveims:
            print np.max(appres()),np.max(appconvmodel())
            self.FacetMachine.ToCasaImage(appres()+appconvmodel(),ImageName="%s.app.restored"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        # intrinsic-flux restored image
        if havenorm and "I" in self._saveims:
            self.FacetMachine.ToCasaImage(intres()+intconvmodel(),ImageName="%s.int.restored"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,Stokes=self.VS.StokesConverter.RequiredStokesProducts())

        # intrinsic-flux restored image
        if havenorm and ("H" in self._saveims):
            if self.FacetMachine.SmoothMeanJonesNorm is None:
                print>>log, ModColor.Str("You requested a restored imaged but the smooth beam is not in there")
                print>>log, ModColor.Str("  so just not doing it")

            else:
                SmoothRestored=(appres()+appconvmodel())/np.sqrt(self.FacetMachine.SmoothMeanJonesNorm)
                self.FacetMachine.ToCasaImage(SmoothRestored,ImageName="%s.smooth.int.restored"%self.BaseName,Fits=True,
                                              beam=self.FWHMBeamAvg,Stokes=self.VS.StokesConverter.RequiredStokesProducts())

        # apparent-flux restored image cube
        if "i" in self._savecubes:
            self.FacetMachine.ToCasaImage(apprescube()+appconvmodelcube(),ImageName="%s.cube.app.restored"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,beamcube=self.FWHMBeam,Freqs=self.VS.FreqBandCenters,
                Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        # intrinsic-flux restored image cube
        if havenorm and "I" in self._savecubes:
            self.FacetMachine.ToCasaImage(intrescube()+intconvmodelcube(),ImageName="%s.cube.int.restored"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,beamcube=self.FWHMBeam,Freqs=self.VS.FreqBandCenters,
                Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        # mixed-flux restored image
        if havenorm and "x" in self._saveims:
            self.FacetMachine.ToCasaImage(appres()+intconvmodel(),ImageName="%s.restored"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,Stokes=self.VS.StokesConverter.RequiredStokesProducts())

        # Alpha image
        if "A" in self._saveims and self.VS.MultiFreqMode:
            IndexMap=ModelMachine.GiveSpectralIndexMap(CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussParsAvg])
            # IndexMap=ModFFTW.ConvolveGaussian(IndexMap,CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars],Normalise=True)
            self.FacetMachine.ToCasaImage(IndexMap,ImageName="%s.alpha"%self.BaseName,Fits=True,beam=self.FWHMBeamAvg,
                                          Stokes=self.VS.StokesConverter.RequiredStokesProducts())

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
        self.FacetMachine.ToCasaImage(self.ResidImage,ImageName="test.residual",Fits=True)


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

