from ClassFacetMachineTessel import ClassFacetMachineTessel as ClassFacetMachine
import numpy as np
from pyrap.images import image
from DDFacet.Other import MyPickle
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Array import NpShared
import os
from DDFacet.ToolsDir import ModFitPSF
from DDFacet.Data import ClassVisServer
import ClassCasaImage
from ModModelMachine import ClassModModelMachine
import time
import glob
from DDFacet.Other import ModColor
from DDFacet.Other import MyLogger
import traceback
from DDFacet.ToolsDir.ModToolBox import EstimateNpix

log=MyLogger.getLogger("ClassImagerDeconv")
import pyfits

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
    def __init__(self,ParsetFile=None,GD=None,
                 PointingID=0,BaseName="ImageTest2",ReplaceDico=None,IdSharedMem="CACA."):
        # if ParsetFile is not None:
        #     GD=ClassGlobalData(ParsetFile)
        #     self.GD=GD

        if GD is not None:
            self.GD=GD

        self.BaseName=BaseName
        self.DicoModelName="%s.DicoModel"%self.BaseName
        self.DicoMetroModelName="%s.Metro.DicoModel"%self.BaseName
        self.PointingID=PointingID

        self.FacetMachine=None
        self.PSF=None
        self.FWHMBeam = None
        self.PSFGaussPars = None
        self.PSFSidelobes = None
        self.FWHMBeamAvg = None
        self.PSFGaussParsAvg = None
        self.PSFSidelobesAvg = None


        self.VisWeights=None
        self.DATA=None
        self.Precision=self.GD["ImagerGlobal"]["Precision"]#"S"
        self.PolMode=self.GD["ImagerGlobal"]["PolMode"]
        self.PSFFacets = self.GD["ImagerGlobal"]["PSFFacets"]
        self.HasDeconvolved=False
        self.Parallel=self.GD["Parallel"]["Enable"]
        self.IdSharedMem=IdSharedMem
        self.ModConstructor = ClassModModelMachine(self.GD)

        self.PredictMode = self.GD["ImagerGlobal"]["PredictMode"]

        #self.PNGDir="%s.png"%self.BaseName
        #os.system("mkdir -p %s"%self.PNGDir)
        #os.system("rm %s/*.png 2> /dev/null"%self.PNGDir)

        # Oleg's "new" interface: set up which output images will be generated
        # --SaveImages abc means save defaults plus abc
        # --SaveOnly abc means only save abc
        # --SaveImages all means save all
        saveimages = self.GD["Images"]["SaveImages"]
        saveonly = self.GD["Images"]["SaveOnly"]
        savecubes = self.GD["Images"]["SaveCubes"]
        allchars = set([chr(x) for x in range(128)])
        if saveimages.lower() == "all" or saveonly.lower() == "all":
            self._saveims = allchars
        else:
            self._saveims = set(saveimages) | set(saveonly)
        self._savecubes = allchars if savecubes.lower() == "all" else set(savecubes)

        old_interface_saveims = self.GD["Images"]["SaveIms"]
        if "Model" in old_interface_saveims:
            self._saveims.update("M")
        if "Alpha" in old_interface_saveims:
            self._saveims.update("A")
        if "Model_i" in old_interface_saveims:
            self._saveims.update("o")
        if "Residual_i" in old_interface_saveims:
            self._saveims.update("e")
        self._save_intermediate_grids = self.GD["Debugging"]["SaveIntermediateDirtyImages"]


    def Init(self):
        DC=self.GD


        MSName0 = MSName = DC["VisData"]["MSName"]

        if type(MSName) is list:
            print>>log,"multi-MS mode"
        elif MSName.endswith(".txt"):
            f=open(MSName)#DC["VisData"]["MSListFile"])
            Ls=f.readlines()
            f.close()
            MSName=[]
            for l in Ls:
                ll=l.replace("\n","")
                MSName.append(ll)
            print>>log,"list file %s contains %d MSs"%(MSName0, len(MSName))
        elif ("*" in MSName)|("?" in MSName):
            MSName=sorted(glob.glob(MSName))
            print>>log,"found %d MSs matching %s"%(len(MSName), MSName0)
        else:
            MSName = sorted(glob.glob(MSName))
            print>>log,"found %d MSs matching %s"%(len(MSName), MSName0)
            if len(MSName) == 1:
                MSName = MSName[0]
                submss = os.path.join(MSName,"SUBMSS")
                if os.path.exists(submss) and os.path.isdir(submss):
                    MSName = [ ms for ms in sorted(glob.glob(os.path.join(submss,"*.[mM][sS]"))) if os.path.isdir(ms) ]
                    print>>log,"multi-MS mode for %s, found %d sub-MSs"%(MSName0, len(MSName))
                else:
                    print>>log,"single-MS mode for %s"%MSName
            else:
                print>>log,"multi-MS mode"


        self.VS=ClassVisServer.ClassVisServer(MSName,
                                              ColName=DC["VisData"]["ColName"],
                                              TVisSizeMin=DC["VisData"]["ChunkHours"]*60,
                                              #DicoSelectOptions=DicoSelectOptions,
                                              TChunkSize=DC["VisData"]["ChunkHours"],
                                              IdSharedMem=self.IdSharedMem,
                                              Robust=DC["ImagerGlobal"]["Robust"],
                                              Weighting=DC["ImagerGlobal"]["Weighting"],
                                              MFSWeighting=DC["ImagerGlobal"]["MFSWeighting"],
                                              Super=DC["ImagerGlobal"]["Super"],
                                              DicoSelectOptions=dict(DC["DataSelection"]),
                                              NCPU=self.GD["Parallel"]["NCPU"],
                                              GD=self.GD)
        


        self.NMajor=self.GD["ImagerDeconv"]["MaxMajorIter"]
        del(self.GD["ImagerDeconv"]["MaxMajorIter"])

        # Construct a model according to what is in MinorCycleConfig
        MinorCycleConfig=dict(self.GD["ImagerDeconv"])
        MinorCycleConfig["NCPU"]=self.GD["Parallel"]["NCPU"]
        MinorCycleConfig["NFreqBands"]=self.VS.NFreqBands
        MinorCycleConfig["GD"] = self.GD
        MinorCycleConfig["ImagePolDescriptor"] = self.VS.StokesConverter.RequiredStokesProducts()
        MinorCycleConfig["IdSharedMem"] = self.IdSharedMem

        SubstractModel=self.GD["VisData"]["InitDicoModel"]
        DoSub=(SubstractModel!="")&(SubstractModel is not None)
        if DoSub:
            print>>log, ModColor.Str("Initialise sky model using %s"%SubstractModel,col="blue")
            ModelMachine = self.ModConstructor.GiveInitialisedMMFromFile(SubstractModel)
            if self.GD["ImagerDeconv"]["MinorCycleMode"] != ModelMachine.DicoSMStacked["Type"]:
                raise NotImplementedError("You want to use different minor cycle and IniDicoModel types [%s vs %s]"\
                                          %(self.GD["ImagerDeconv"]["MinorCycleMode"],ModelMachine.DicoSMStacked["Type"]))
            if self.BaseName==self.GD["VisData"]["InitDicoModel"][0:-10]:
                self.BaseName+=".continue"
        else:
            ModelMachine = self.ModConstructor.GiveMM(Mode=self.GD["ImagerDeconv"]["MinorCycleMode"])
        self.ModelMachine=ModelMachine
        MinorCycleConfig["ModelMachine"] = ModelMachine

        # Specify which deconvolution algorithm to use
        if self.GD["ImagerDeconv"]["MinorCycleMode"] == "MSMF":
            if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                raise NotImplementedError("Multi-polarization CLEAN is not supported in MSMF")
            from DDFacet.Imager.MSMF import ClassImageDeconvMachineMSMF
            self.DeconvMachine=ClassImageDeconvMachineMSMF.ClassImageDeconvMachine(MainCache=self.VS.maincache, **MinorCycleConfig)
            print>>log,"Using MSMF algorithm"
        elif self.GD["ImagerDeconv"]["MinorCycleMode"]=="SSD":
            if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                raise NotImplementedError("Multi-polarization CLEAN is not supported in SSD")
            from DDFacet.Imager.SSD import ClassImageDeconvMachineSSD
            self.DeconvMachine=ClassImageDeconvMachineSSD.ClassImageDeconvMachine(**MinorCycleConfig)
            print>>log,"Using SSD algorithm"
        elif self.GD["ImagerDeconv"]["MinorCycleMode"] == "Hogbom":
            from DDFacet.Imager.HOGBOM import ClassImageDeconvMachineHogbom
            self.DeconvMachine=ClassImageDeconvMachineHogbom.ClassImageDeconvMachine(**MinorCycleConfig)
            print>>log,"Using Hogbom algorithm"
        else:
            raise NotImplementedError("Mode %s is not valid"%self.GD["ImagerDeconv"]["MinorCycleMode"])


        self.InitFacetMachine()
        
        self.FacetMachine.DoComputeSmoothBeam=("H" in self._saveims)

        self.VS.setFacetMachine(self.FacetMachine)
        self.VS.CalcWeights()


    def InitFacetMachine(self):
        if self.FacetMachine is not None:
            return

        ApplyCal=False
        SolsFile=self.GD["DDESolutions"]["DDSols"]
        if (SolsFile!="")|(self.GD["Beam"]["BeamModel"] is not None): ApplyCal=True

        self.FacetMachine=ClassFacetMachine(self.VS,
                                            self.GD,
                                            Precision=self.Precision,
                                            PolMode=self.PolMode,
                                            Parallel=self.Parallel,
                                            IdSharedMem=self.IdSharedMem,
                                            ApplyCal=ApplyCal)

        MainFacetOptions=self.GiveMainFacetOptions()

        self.FacetMachine.appendMainField(ImageName="%s.image"%self.BaseName,**MainFacetOptions)
        self.FacetMachine.Init()

        self.CellSizeRad=(self.FacetMachine.Cell/3600.)*np.pi/180
        self.CellArcSec=self.FacetMachine.Cell

    def setNextData (self, keep_data=False, null_data=False):
        try:
            del(self.DATA)
        except:
            pass

        try:
            NpShared.DelAll("%s%s"%(self.IdSharedMem,"DicoData"))
        except:
            pass

        Load = self.VS.LoadNextVisChunk(keep_data=keep_data, null_data=null_data)
        if Load == "EndOfObservation":
            return "EndOfObservation"

        if Load == "EndChunk":
            print>>log, ModColor.Str("Reached end of data chunk")
            return "EndChunk"

        self.DATA = self.VS.VisChunkToShared()
        self.WEIGHTS = self.VS.CurrentVisWeights

        return True

    def GiveMainFacetOptions(self):
        MainFacetOptions=self.GD["ImagerMainFacet"].copy()
        MainFacetOptions.update(self.GD["ImagerCF"].copy())
        MainFacetOptions.update(self.GD["ImagerGlobal"].copy())
        del(MainFacetOptions['ConstructMode'],MainFacetOptions['Precision'],
            MainFacetOptions['PolMode'],MainFacetOptions['Mode'],MainFacetOptions['Robust'],
            MainFacetOptions['Weighting'])
        return MainFacetOptions

    def MakePSF(self):
        if self.PSF is not None: return

        import cPickle

        cachepath, valid = self.VS.maincache.checkCache("PSF",dict(
                [ ("MSNames", [ms.MSName for ms in self.VS.ListMS]) ] +
                [ (section, self.GD[section]) for section in "VisData", "Beam", "DataSelection",
                                                             "MultiFreqs", "ImagerGlobal", "Compression",
                                                             "ImagerCF", "ImagerMainFacet","DDESolutions" ]
            ), reset=self.GD["Caching"]["ResetPSF"])

        if valid or self.GD["Caching"]["ResetPSF"]==-1:
            print>>log, ModColor.Str("============================ Loading cached PSF ==========================")
            print>>log, "found valid cached PSF in %s"%cachepath
            print>>log, ModColor.Str("As near as we can tell, we can reuse this cached PSF because it was produced")
            print>>log, ModColor.Str("with the same set of relevant DDFacet settings. If you think this is in error,")
            print>>log, ModColor.Str("or if your MS has been substantially flagged or otherwise had its uv-coverage")
            print>>log, ModColor.Str("affected, please remove the cache, or else run with --ResetPSF 1.")
            psfmean, psfcube = None, None
            #self.DicoVariablePSF = cPickle.load(file(cachepath))
            self.DicoVariablePSF = MyPickle.FileToDicoNP(cachepath)
            self.FWHMBeamAvg, self.PSFGaussParsAvg, self.PSFSidelobesAvg=self.DicoVariablePSF["EstimatesAvgPSF"]


        else:
            print>>log, ModColor.Str("=============================== Making PSF ===============================")
            if self.PSFFacets:
                print>>log,"the PSFFacets version is currently not supported, using 0 (i.e. same facets as image)"
                self.PSFFacets = 0
            oversize = self.GD["ImagerGlobal"]["PSFOversize"] or 1
            if oversize == 1 and not self.PSFFacets:
                print>>log,"PSFOversize=1 and PSFFacets=0, same facet machine will be reused for PSF"
                FacetMachinePSF=self.FacetMachine
            else:
                MainFacetOptions=self.GiveMainFacetOptions()
                if self.PSFFacets:
                    MainFacetOptions["NFacets"] = self.PSFFacets
                    print>>log,"using %d facets to compute the PSF"%self.PSFFacets
                    if self.PSFFacets == 1:
                        oversize = 1
                        print>>log,"PSFFacets=1 implies PSFOversize=1"
                print>>log,"PSFOversize=%.2f, making a separate facet machine for the PSFs"%oversize
                FacetMachinePSF=ClassFacetMachine(self.VS,self.GD,
                    Precision=self.Precision,PolMode=self.PolMode,Parallel=self.Parallel,
                    IdSharedMem=self.IdSharedMem+"psf.",
                    IdSharedMemData=self.IdSharedMem,
                    DoPSF=True,
                    Oversize=oversize)
                FacetMachinePSF.appendMainField(ImageName="%s.psf"%self.BaseName,**MainFacetOptions)
                FacetMachinePSF.Init()

            FacetMachinePSF.ReinitDirty()
            FacetMachinePSF.DoPSF=True


            while True:
                Res=self.setNextData()
                if Res=="EndOfObservation": break
                FacetMachinePSF.putChunk(Weights=self.WEIGHTS)

            psfdict = FacetMachinePSF.FacetsToIm(NormJones=True)
            psfmean, psfcube = psfdict["MeanImage"], psfdict["ImagData"]   # this is only for the casa image saving
            self.DicoVariablePSF = FacetMachinePSF.DicoPSF
            #FacetMachinePSF.ToCasaImage(self.DicoImagePSF["ImagData"],ImageName="%s.psf"%self.BaseName,Fits=True)
            self.PSF=self.MeanFacetPSF=self.DicoVariablePSF["MeanFacetPSF"]
            self.FitPSF()

            self.DicoVariablePSF["FWHMBeam"]=self.FWHMBeam
            self.DicoVariablePSF["PSFGaussPars"]=self.PSFGaussPars
            self.DicoVariablePSF["PSFSidelobes"]=self.PSFSidelobes
            self.DicoVariablePSF["EstimatesAvgPSF"]=(self.FWHMBeamAvg, self.PSFGaussParsAvg, self.PSFSidelobesAvg)

            if self.GD["Caching"]["CachePSF"]:
                try:
                    #cPickle.dump(self.DicoVariablePSF, file(cachepath,'w'), 2)
                    print>>log,"Put PSF dico into %s"%cachepath
                    MyPickle.DicoNPToFile(self.DicoVariablePSF,cachepath)
                    self.VS.maincache.saveCache("PSF")
                except:
                    print>>log,traceback.format_exc()
                    print>>log,ModColor.Str("WARNING: PSF cache could not be written, see error report above. Proceeding anyway.")
            FacetMachinePSF.DoPSF = False

        # self.PSF = self.DicoImagePSF["MeanImage"]#/np.sqrt(self.DicoImagePSF["NormData"])
        self.PSF = self.MeanFacetPSF=self.DicoVariablePSF["MeanFacetPSF"]

        self.FWHMBeam=self.DicoVariablePSF["FWHMBeam"]
        self.PSFGaussPars=self.DicoVariablePSF["PSFGaussPars"]
        self.PSFSidelobes=self.DicoVariablePSF["PSFSidelobes"]

#        MyPickle.Save(self.DicoImagePSF,"DicoPSF")


        # # Image=FacetMachinePSF.FacetsToIm()
        # pylab.clf()
        # pylab.imshow(self.PSF[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop



        # so strange... had to put pylab statement after ToCasaimage, otherwise screw fits header
        # and even sending a copy of PSF to imshow doesn't help...
        # Error validating header for HDU 0 (note: PyFITS uses zero-based indexing).
        # Unparsable card (BZERO), fix it first with .verify('fix').
        # There may be extra bytes after the last HDU or the file is corrupted.
        # Edit: Only with lastest matplotlib!!!!!!!!!!!!!
        # WHOOOOOWWWW... AMAZING!

        # m0=-1;m1=1
        # pylab.clf()
        # FF=self.PSF[0,0].copy()
        # pylab.imshow(FF,interpolation="nearest")#,vmin=m0,vmax=m1)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # time.sleep(1)



        # self.FWHMBeamAvg=(10.,10.,10.)
        # FacetMachinePSF.ToCasaImage(self.PSF)



        #FacetMachinePSF.ToCasaImage(self.PSF,ImageName="%s.psf"%self.BaseName,Fits=True)

        if psfmean is not None:
            if "P" in self._saveims or "p" in self._saveims:
                FacetMachinePSF.ToCasaImage(psfmean,ImageName="%s.psf"%self.BaseName,Fits=True,beam=self.FWHMBeamAvg,
                                            Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            if "P" in self._savecubes or "p" in self._savecubes:
                self.FacetMachine.ToCasaImage(psfcube,
                                              ImageName="%s.cube.psf"%self.BaseName,
                                              Fits=True,beam=self.FWHMBeamAvg,Freqs=self.VS.FreqBandCenters,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())

        FacetMachinePSF = None

        # if self.VS.MultiFreqMode:
        #     for Channel in range(self.VS.NFreqBands):
        #         Im=self.DicoImagePSF["ImagData"][Channel]
        #         npol,n,n=Im.shape
        #         Im=Im.reshape((1,npol,n,n))
        #         FacetMachinePSF.ToCasaImage(Im,ImageName="%s.psf.ch%i"%(self.BaseName,Channel),Fits=True,beam=self.FWHMBeamAvg)

        #self.FitPSF()
        #FacetMachinePSF.ToCasaImage(self.PSF,Fits=True)



        #del(FacetMachinePSF)


    ### NB: this is not really used anywhere anymore. OMS 6 Sep 2016
    # def LoadPSF(self,CasaFilePSF):
    #     self.CasaPSF=image(CasaFilePSF)
    #     self.PSF=self.CasaPSF.getdata()
    #     self.CellArcSec=np.abs(self.CasaPSF.coordinates().dict()["direction0"]["cdelt"][0]*60)
    #     self.CellSizeRad=(self.CellArcSec/3600.)*np.pi/180
    #
    #


    def GiveDirty(self):

        self.InitFacetMachine()

        import cPickle




        SubstractModel=self.GD["VisData"]["InitDicoModel"]
        DoSub=(SubstractModel!="")&(SubstractModel is not None)

        CacheName="Dirty"
        if self.GD["Caching"]["DirtyFromLastResid"]:
            print>>log,"Setting dirty image cache name to last residual image cache, and not doing any substraction"
            CacheName="LastResidual"
            DoSub=False

        cachepath, valid = self.VS.maincache.checkCache(CacheName, dict(
            [("MSNames", [ms.MSName for ms in self.VS.ListMS])] +
            [(section, self.GD[section]) for section in "VisData", "Beam", "DataSelection",
                                                        "MultiFreqs", "ImagerGlobal", "Compression",
                                                        "ImagerCF", "ImagerMainFacet","DDESolutions"]
        ), reset=self.GD["Caching"]["ResetDirty"])

        if valid or self.GD["Caching"]["ResetDirty"]==-1:
            print>>log, ModColor.Str("============================ Loading cached dirty image =======================")
            print>>log, "found valid cached dirty image in %s"%cachepath
            print>>log, ModColor.Str("As near as we can tell, we can reuse this cached dirty because it was produced")
            print>>log, ModColor.Str("with the same set of relevant DDFacet settings. If you think this is in error,")
            print>>log, ModColor.Str("or if your MS has changed, please remove the cache, or run with --ResetDirty 1.")

            #self.DicoDirty = cPickle.load(file(cachepath))
            self.DicoDirty = MyPickle.FileToDicoNP(cachepath)
            if self.DicoDirty["NormData"] is not None:
                nch, npol, nx, ny = self.DicoDirty["ImagData"].shape
                self.NormImage = self.DicoDirty["NormData"]
                self.MeanNormImage = np.mean(self.NormImage, axis=0).reshape((1, npol, nx, ny))

                self.FacetMachine.NormImage = self.DicoDirty["NormImage"]
                NpShared.ToShared("%sNormImage"%self.IdSharedMem,self.DicoDirty["NormImage"])
                self.FacetMachine.NormImageReShape = self.DicoDirty["NormImage"].reshape([1,1,
                                                                                          self.FacetMachine.NormImage.shape[0],
                                                                                          self.FacetMachine.NormImage.shape[1]])
                self.FacetMachine.NormData = self.DicoDirty["NormData"] 
                self.FacetMachine.MeanResidual = self.DicoDirty["MeanImage"]
                self.FacetMachine.DoCalcNormData = False

                #if "SmoothMeanNormImage" in self.DicoDirty.keys():
                #    #print>>log,"A smooth beam is used for the averaged Muller "
                #    self.SmoothMeanNormImage=self.DicoDirty["SmoothMeanNormImage"]
                #    self.FacetMachine.SmoothMeanNormImage = self.DicoDirty["SmoothMeanNormImage"]
                #    self.FacetMachine.DoComputeSmoothBeam = False

                self.FacetMachine.ComputeSmoothBeam()
                # self.SaveDirtyProducts()
                DirtyCorr = self.DicoDirty["ImagData"]/np.sqrt(self.DicoDirty["NormData"])
                nch,npol,nx,ny = DirtyCorr.shape
            else:
                self.MeanNormImage = None
        else:
            print>>log, ModColor.Str("============================== Making Residual Image ==============================")

            self.FacetMachine.ReinitDirty()
            isPlotted=False

            # if self.GD["Stores"]["Dirty"] is not None:
            #     print>>log, "Reading Dirty image from %s"%self.GD["Stores"]["Dirty"]
            #     CasaDirty=image(self.GD["Stores"]["Dirty"])
            #     Dirty=CasaDirty.getdata()
            #     nch,npol,_,_=Dirty.shape
            #     for ch in range(nch):
            #         for pol in range(npol):
            #             Dirty[ch,pol]=Dirty[ch,pol].T[::-1]
            #     return Dirty
            #


            iloop = 0
            while True:
                Res=self.setNextData()
                # if not(isPlotted):
                #     isPlotted=True
                #     self.FacetMachine.PlotFacetSols()
                #     stop
                #if Res=="EndChunk": break
                if Res=="EndOfObservation": break

                if DoSub:
                    ThisMeanFreq=self.VS.CurrentChanMappingDegrid#np.mean(DATA["freqs"])
                    ModelImage=self.ModelMachine.GiveModelImage(ThisMeanFreq)
                    print>>log, "Model image @%s MHz (min,max) = (%f, %f)"%(str(ThisMeanFreq/1e6),ModelImage.min(),ModelImage.max())

                    # self.FacetMachine.ToCasaImage(ModelImage,ImageName="%s.modelSub"%self.BaseName,Fits=True,
                    #                               Stokes=self.VS.StokesConverter.RequiredStokesProducts())

                    _=self.FacetMachine.getChunk(ModelImage)


                self.FacetMachine.putChunk(Weights=self.WEIGHTS)

                if self._save_intermediate_grids:
                    self.DicoDirty=self.FacetMachine.FacetsToIm(NormJones=True)
                    self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.dirty.%d."%(self.BaseName,iloop),
                                                  Fits=True,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                    if 'g' in self._savecubes:
                        self.FacetMachine.ToCasaImage(self.DicoDirty["ImagData"],ImageName="%s.cube.dirty.%d"%(self.BaseName,iloop),
                            Fits=True,Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                    self.FacetMachine.NormData = None
                    self.FacetMachine.NormImage = None

                iloop += 1

            self.DicoDirty=self.FacetMachine.FacetsToIm(NormJones=True)

            self.FacetMachine.ComputeSmoothBeam()
            self.SaveDirtyProducts()

            # dump dirty to cache
            if self.GD["Caching"]["CacheDirty"]:
                try:
                    #cPickle.dump(self.DicoDirty, file(cachepath, 'w'), 2)
                    MyPickle.DicoNPToFile(self.DicoDirty, cachepath)
                    self.VS.maincache.saveCache("Dirty")
                except:
                    print>> log, traceback.format_exc()
                    print>> log, ModColor.Str("WARNING: Dirty image cache could not be written, see error report above. Proceeding anyway.")

        return self.DicoDirty["MeanImage"]

    def SaveDirtyProducts(self):
        if "d" in self._saveims:
            self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.dirty"%self.BaseName,Fits=True,
                                          Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        if "d" in self._savecubes:
            self.FacetMachine.ToCasaImage(self.DicoDirty["ImagData"],ImageName="%s.cube.dirty"%self.BaseName,
                                          Fits=True,Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())


        if "n" in self._saveims:
            self.FacetMachine.ToCasaImage(self.FacetMachine.NormImageReShape,ImageName="%s.NormFacets"%self.BaseName,
                                          Fits=True)
                
        if self.DicoDirty["NormData"] is not None:
            DirtyCorr = self.DicoDirty["ImagData"]/np.sqrt(self.DicoDirty["NormData"])
            nch,npol,nx,ny = DirtyCorr.shape
            if "D" in self._saveims:
                MeanCorr = np.mean(DirtyCorr, axis=0).reshape((1, npol, nx, ny))
                self.FacetMachine.ToCasaImage(MeanCorr,ImageName="%s.dirty.corr"%self.BaseName,Fits=True,
                                                  Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            if "D" in self._savecubes:
                self.FacetMachine.ToCasaImage(DirtyCorr,ImageName="%s.cube.dirty.corr"%self.BaseName,
                                              Fits=True,Freqs=self.VS.FreqBandCenters,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())

            self.NormImage = self.DicoDirty["NormData"]
            self.MeanNormImage = np.mean(self.NormImage,axis=0).reshape((1,npol,nx,ny))

            if "H" in self._saveims and self.FacetMachine.SmoothMeanNormImage is not None:
                self.FacetMachine.ToCasaImage(self.FacetMachine.SmoothMeanNormImage,ImageName="%s.SmoothNorm"%self.BaseName,Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())


            if "N" in self._saveims:
                self.FacetMachine.ToCasaImage(self.MeanNormImage,ImageName="%s.Norm"%self.BaseName,Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            if "N" in self._savecubes:
                self.FacetMachine.ToCasaImage(self.NormImage, ImageName="%s.cube.Norm" % self.BaseName,
                                              Fits=True, Freqs=self.VS.FreqBandCenters,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        else:
            self.MeanNormImage = None


    def GivePredict(self,from_fits=True):
        print>>log, ModColor.Str("============================== Making Predict ==============================")
        self.InitFacetMachine()

        self.FacetMachine.ReinitDirty()
        BaseName=self.GD["Images"]["ImageName"]

        #ModelMachine=ClassModelMachine(self.GD)
        # try:
        #     NormImageName="%s.NormFacets.fits"%BaseName
        #     CasaNormImage = image(NormImageName)
        #     NormImage = CasaNormImage.getdata()
        #     print NormImage.shape
        # except:
        #     NormImage = self.FacetMachine.BuildFacetNormImage()
        #     NormImage = NormImage.reshape([1,1,NormImage.shape[0],NormImage.shape[1]])

        NormImage = self.FacetMachine.BuildFacetNormImage()
        NormImage = NormImage.reshape([1,1,NormImage.shape[0],NormImage.shape[1]])

        # nch,npol,nx,_=NormImage.shape
        # for ch in range(nch):
        #     for pol in range(npol):
        #         NormImage[ch,pol]=NormImage[ch,pol].T[::-1]
        # self.FacetMachine.NormImage=NormImage.reshape((nx,nx))

        modelfile = self.GD["Images"]["PredictModelName"]
        FixedModelImage = None

        # if model is a dict, init model machine with that
        # else we use a model image and hope for the best (need to fix frequency axis...)
        # 
        if modelfile is not None and modelfile is not "":
            print>>log,ModColor.Str("Reading image file for the predict: %s"%modelfile)
            FixedModelImage = ClassCasaImage.FileToArray(modelfile,True)

        current_model_freqs = np.array([])

        while True:
            null_data=(self.GD["ImagerGlobal"]["Mode"] != "Substract")
            Res=self.setNextData(null_data=null_data)
            if Res=="EndOfObservation": break

            model_freqs = self.VS.CurrentChanMappingDegrid
            ## redo model image if needed
            if FixedModelImage is None:
                if (np.array(model_freqs != current_model_freqs).any()) or (model_freqs.size != current_model_freqs.size):
                    ModelImage = self.DeconvMachine.GiveModelImage(model_freqs)
                    current_model_freqs = model_freqs
                    print>>log, "Model image @%s MHz (min,max) = (%f, %f)"%(str(model_freqs/1e6),ModelImage.min(),ModelImage.max())
                else:
                    print>>log,"reusing model image from previous chunk"
            else:
                ModelImage = FixedModelImage

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




            if self.PredictMode == "DeGridder":
                self.FacetMachine.getChunk(ModelImage)
            elif self.PredictMode == "Montblanc":
                from ClassMontblancMachine import ClassMontblancMachine
                model = self.DeconvMachine.ModelMachine.GiveModelList()
                mb_machine = ClassMontblancMachine(self.GD, self.FacetMachine.Npix, self.FacetMachine.CellSizeRad)
                mb_machine.getChunk(self.DATA, self.VS.getVisibilityResiduals(), model, self.VS.CurrentMS)
                mb_machine.close()
            else:
                raise ValueError("Invalid PredictMode '%s'" % PredictMode)

            # #######################
            # vis = self.VS.getVisibilityResiduals() # that's crap, gives only zeros, I could not reverse engineer why
            vis = self.DATA["data"]
            # #######################
            if self.GD["ImagerGlobal"]["Mode"]!="Substract":
                vis *= -1 # model was subtracted from null data, so need to invert

            PredictColName=self.GD["VisData"]["PredictColName"]

            self.VS.CurrentMS.PutVisColumn(PredictColName, vis)

        # if from_fits:
        #     print "Predicting from fits and saving in %s",(PredictColName)
        #     #Read in the LSM
        #     Imtodegrid = pyfits.open("/home/landman/Projects/Processed_Images/ddfacet_out/Test-D147-HI-NOIFS-NOPOL-4M5Sa/model-true.fits")[0].data
        #     print ModelImage.shape, Imtodegrid.shape
        #     self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),Imtodegrid)
        #     vis = - DATA["data"]
        #     PredictColName = self.GD["VisData"]["PredictColName"]
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

        self.GiveDirty()
        self.MakePSF()

        #Pass minor cycle specific options into Init as kwargs
        self.DeconvMachine.Init(PSFVar=self.DicoVariablePSF,PSFAve=self.PSFSidelobesAvg)

        DicoImage=self.DicoDirty
        continue_deconv = True


        for iMajor in range(NMajor):
            # previous minor loop indicated it has reached bottom? Break out
            if not continue_deconv:
                break

            print>>log, ModColor.Str("========================== Running major Cycle %i ========================="%iMajor)

            self.DeconvMachine.Update(DicoImage)

            repMinor, continue_deconv, update_model = self.DeconvMachine.Deconvolve()
            self.DeconvMachine.ModelMachine.ToFile(self.DicoModelName) 

            ## returned with nothing done in minor cycle? Break out
            if not update_model or iMajor == NMajor-1:
                continue_deconv = False
                print>> log, "This is the last major cycle"
            else:
                print>> log, "Finished Deconvolving for this major cycle... Going back to visibility space."
            predict_colname = not continue_deconv and self.GD["VisData"]["PredictColName"]

            # for some reason predict gives zeros 
            if predict_colname is "" or predict_colname is None:
                predict_colname=False



            #self.ResidImage=DicoImage["MeanImage"]
            #self.FacetMachine.ToCasaImage(DicoImage["MeanImage"],ImageName="%s.residual_sub%i"%(self.BaseName,iMajor),Fits=True)

            self.FacetMachine.ReinitDirty()
            
            current_model_freqs = np.array([])
            
            while True:
                # if writing predicted visibilities, tell VisServer to keep the original data
                Res = self.setNextData(keep_data=predict_colname)

                #if Res=="EndChunk": break
                if Res=="EndOfObservation":
                    break

                model_freqs = self.VS.CurrentChanMappingDegrid
                ## redo model image if needed
                if not np.array_equal(model_freqs, current_model_freqs):
                    ModelImage = self.DeconvMachine.GiveModelImage(model_freqs)
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

                if self.PredictMode == "DeGridder":
                    self.FacetMachine.getChunk(ModelImage)
                elif self.PredictMode == "Montblanc":
                    from ClassMontblancMachine import ClassMontblancMachine
                    model = self.DeconvMachine.ModelMachine.GiveModelList()
                    mb_machine = ClassMontblancMachine(self.GD, self.FacetMachine.Npix, self.FacetMachine.CellSizeRad)
                    mb_machine.getChunk(self.DATA, self.VS.getVisibilityResiduals(), model, self.VS.CurrentMS)
                    mb_machine.close()
                else:
                    raise ValueError("Invalid PredictMode '%s'" % PredictMode)

                if predict_colname:
                    data = self.VS.getVisibilityData()
                    resid = self.VS.getVisibilityResiduals()
                    # model is data minus residuals
                    model = data-resid
                    self.VS.CurrentMS.PutVisColumn(predict_colname, model)
                    data = resid = None

                self.FacetMachine.putChunk(Weights=self.WEIGHTS)


            self.CurrentDicoResidImage=DicoImage=self.FacetMachine.FacetsToIm(NormJones=True)
            self.ResidCube  = DicoImage["ImagData"] #get residuals cube
            self.ResidImage = DicoImage["MeanImage"]

            # if "SmoothMeanNormImage" in DicoImage.keys():
            #     self.SmoothMeanNormImage=DicoImage["SmoothMeanNormImage"]

            if "e" in self._saveims:
                self.FacetMachine.ToCasaImage(self.ResidImage,ImageName="%s.residual%2.2i"%(self.BaseName,iMajor),
                                              Fits=True,Stokes=self.VS.StokesConverter.RequiredStokesProducts())


            #self.DeconvMachine.ToFile(self.DicoModelName)


            self.HasDeconvolved=True

        # dump dirty to cache
        if self.GD["Caching"]["CacheLastResid"]:
            cachepath, valid = self.VS.maincache.checkCache("LastResidual", 
                                                            dict(
                                                                [("MSNames", [ms.MSName for ms in self.VS.ListMS])] +
                                                                [(section, self.GD[section]) for section in "VisData", "Beam", "DataSelection",
                                                                 "MultiFreqs", "ImagerGlobal", "Compression",
                                                                 "ImagerCF", "ImagerMainFacet","DDESolutions"]
                                                            ), 
                                                            reset=False)
            try:
                print>>log,"Saving last residual image to %s"%cachepath
                MyPickle.DicoNPToFile(self.CurrentDicoResidImage, cachepath)
                self.VS.maincache.saveCache("LastResidual")
            except:
                print>> log, traceback.format_exc()
                print>> log, ModColor.Str("WARNING: Dirty image cache could not be written, see error report above. Proceeding anyway.")

        if self.HasDeconvolved:
            self.Restore()

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
        #off = offStart

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
        forced_beam=self.GD["ImagerDeconv"]["RestoringBeam"]
        if forced_beam is not None:
            FWHMFact = 2. * np.sqrt(2. * np.log(2.))
            
            if isinstance(forced_beam,float):
                forced_beam=[forced_beam,forced_beam,0]
            elif len(forced_beam)==1:
                forced_beam=[forced_beam[0],forced_beam[0],0]
            f_beam=(forced_beam[0]/3600.0,forced_beam[1]/3600.0,forced_beam[2])
            f_gau=(np.deg2rad(f_beam[0])/FWHMFact,np.deg2rad(f_beam[1])/FWHMFact,np.deg2rad(f_beam[2]))
        PSF = self.DicoVariablePSF["CubeVariablePSF"][self.FacetMachine.iCentralFacet]

        off=self.GD["ImagerDeconv"]["SidelobeSearchWindow"] // 2

        beam, gausspars, sidelobes = self.fitSinglePSF(self.MeanFacetPSF[0,...], off, "mean")
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

    def Restore(self):
        print>>log, "Create restored image"
        if self.PSFGaussPars is None:
            self.FitPSF()
        #self.DeconvMachine.ToFile(self.DicoModelName)

        RefFreq = self.VS.RefFreq

        if self.GD["ImagerDeconv"]["MinorCycleMode"]=="SSD" and self.GD["SSDClean"]["RestoreMetroSwitch"]>0:
            print>>log,"Runing and Metropolis-Hastings MCMC on islands larger than %i pixels"%self.GD["SSDClean"]["RestoreMetroSwitch"]
            self.DeconvMachine.setDeconvMode(Mode="MetroClean")
            self.DeconvMachine.Update(self.CurrentDicoResidImage)
            repMinor, continue_deconv, update_model = self.DeconvMachine.Deconvolve()
            self.DeconvMachine.ToFile(self.DicoMetroModelName)


        ModelMachine = self.DeconvMachine.ModelMachine
        # Putting back substracted componants
        if self.GD["DDESolutions"]["RestoreSub"]:
            try:
                ModelMachine.PutBackSubsComps()
            except:
                print>>log, ModColor.Str("Failed Putting back substracted components")


        # do we have a non-trivial norm (i.e. DDE solutions or beam)?
        # @cyriltasse: maybe there's a quicker way to check?
        havenorm = self.MeanNormImage is not None and (self.MeanNormImage != 1).any()

        # make a dict of _images to save the intermediate images for when we need them
        _images = {}
        def sqrtnorm():
            label = 'sqrtnorm'
            if label not in _images:
                _images[label] = np.sqrt(self.MeanNormImage) if havenorm else 1
            return _images[label]
        def sqrtnormcube():
            label = 'sqrtnormcube'
            if label not in _images:
                _images[label] = np.sqrt(self.NormImage) if havenorm else 1
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
            self.FacetMachine.ToCasaImage(appres()+appconvmodel(),ImageName="%s.app.restored"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        # intrinsic-flux restored image
        if havenorm and "I" in self._saveims:
            self.FacetMachine.ToCasaImage(intres()+intconvmodel(),ImageName="%s.int.restored"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,Stokes=self.VS.StokesConverter.RequiredStokesProducts())

        # intrinsic-flux restored image
        if havenorm and ("H" in self._saveims):
            if self.FacetMachine.SmoothMeanNormImage is None:
                print>>log, ModColor.Str("You requested a restored imaged but the smooth beam is not in there")
                print>>log, ModColor.Str("  so just not doing it")

            else:
                SmoothRestored=(appres()+appconvmodel())/np.sqrt(self.FacetMachine.SmoothMeanNormImage)
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

        self.FacetMachine.putChunk(Weights=self.WEIGHTS)
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

