from ClassFacetMachineTessel import ClassFacetMachineTessel as ClassFacetMachine
import numpy as np
import pylab
from pyrap.images import image
import ClassImageDeconvMachineHogbom
import ClassImageDeconvMachineMSMF
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Array import NpShared
import os
from DDFacet.ToolsDir import ModFitPSF
from DDFacet.Data import ClassVisServer
import ClassCasaImage
from ClassModelMachine import ClassModelMachine
import time
import glob
from DDFacet.Other import ModColor
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassImagerDeconv")


class ClassImagerDeconv():
    def __init__(self,ParsetFile=None,GD=None,
                 PointingID=0,BaseName="ImageTest2",ReplaceDico=None,IdSharedMem="CACA.",DoDeconvolve=True):
        if ParsetFile!=None:
            GD=ClassGlobalData(ParsetFile)
            self.GD=GD

        if GD!=None:
            self.GD=GD

        self.BaseName=BaseName
        self.DicoModelName="%s.DicoModel"%self.BaseName
        self.PointingID=PointingID
        self.DoDeconvolve=DoDeconvolve
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
        self.HasCleaned=False
        self.Parallel=self.GD["Parallel"]["Enable"]
        self.IdSharedMem=IdSharedMem
        #self.PNGDir="%s.png"%self.BaseName
        #os.system("mkdir -p %s"%self.PNGDir)
        #os.system("rm %s/*.png 2> /dev/null"%self.PNGDir)
        self._save_intermediate_grids = self.GD["Debugging"]["SaveIntermediateDirtyImages"]
        
        # Oleg's "new" interface: set up which output images will be generated
        # --SaveImages abc means save defaults plus abc
        # --SaveOnly abc means only save abc
        # --SaveImages all means save all
        saveimages = self.GD["Images"]["SaveImages"]
        saveonly = self.GD["Images"]["SaveOnly"]
        if saveimages.lower() == "all" or saveonly.lower() == "all":
            self._saveims = set([chr(x) for x in range(128)])  # all characters
        else:
            self._saveims = set(saveimages) | set(saveonly)
        old_interface_saveims = self.GD["Images"]["SaveIms"] 

        if "Model" in old_interface_saveims:
            self._saveims.update("M")
        if "Alpha" in old_interface_saveims:
            self._saveims.update("A")
        if "Model_i" in old_interface_saveims:
            self._saveims.update("o")
        if "Residual_i" in old_interface_saveims:
            self._saveims.update("e")


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
                                              TVisSizeMin=DC["VisData"]["TChunkSize"]*60,
                                              #DicoSelectOptions=DicoSelectOptions,
                                              TChunkSize=DC["VisData"]["TChunkSize"],
                                              IdSharedMem=self.IdSharedMem,
                                              Robust=DC["ImagerGlobal"]["Robust"],
                                              Weighting=DC["ImagerGlobal"]["Weighting"],
                                              Super=DC["ImagerGlobal"]["Super"],
                                              DicoSelectOptions=dict(DC["DataSelection"]),
                                              NCPU=self.GD["Parallel"]["NCPU"],
                                              GD=self.GD)

        if self.DoDeconvolve:
            self.NMajor=self.GD["ImagerDeconv"]["MaxMajorIter"]
            del(self.GD["ImagerDeconv"]["MaxMajorIter"])
            MinorCycleConfig=dict(self.GD["ImagerDeconv"])
            MinorCycleConfig["NCPU"]=self.GD["Parallel"]["NCPU"]
            MinorCycleConfig["GD"] = self.GD

            if self.GD["MultiScale"]["MSEnable"]:
                print>>log, "Minor cycle deconvolution in Multi Scale Mode"
                self.MinorCycleMode="MS"
                # Specify which deconvolution algorithm to use
                if self.GD["ImagerDeconv"]["MinorCycleMode"] == "MSMF":
                    self.DeconvMachine=ClassImageDeconvMachineMSMF.ClassImageDeconvMachine(**MinorCycleConfig)
                else:
                    print>>log, "Currently MSMF is the only multi-scale algorithm"
            else:
                print>>log, "Minor cycle deconvolution in Single Scale Mode"
                self.MinorCycleMode="SS"
                if self.GD["ImagerDeconv"]["MinorCycleMode"] == "Hogbom":
                    self.DeconvMachine=ClassImageDeconvMachineHogbom.ClassImageDeconvMachine(**MinorCycleConfig)

            self.InitFacetMachine()
            self.VS.setFacetMachine(self.FacetMachine)
            self.VS.CalcWeights()


    def InitFacetMachine(self):
        if self.FacetMachine!=None:
            return

        ApplyCal=False
        SolsFile=self.GD["DDESolutions"]["DDSols"]
        if (SolsFile!="")|(self.GD["Beam"]["BeamModel"]!=None): ApplyCal=True

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

    def setNextData(self):
        try:
            del(self.DATA)
        except:
            pass

        try:
            NpShared.DelAll("%s%s"%(self.IdSharedMem,"DicoData"))
        except:
            pass

        Load=self.VS.LoadNextVisChunk()
        if Load=="EndOfObservation":
            return "EndOfObservation"

        DATA=self.VS.VisChunkToShared()
        if DATA=="EndOfObservation":
            print>>log, ModColor.Str("Reached end of Observation")
            return "EndOfObservation"
        if DATA=="EndChunk":
            print>>log, ModColor.Str("Reached end of data chunk")
            return "EndChunk"
        self.DATA=DATA
        
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
        if self.PSF!=None: return

        if self.GD["Stores"]["PSF"]!=None:
            print>>log, "Reading PSF image from %s"%self.GD["Stores"]["PSF"]
            CasaPSF=image(self.GD["Stores"]["PSF"])
            PSF=CasaPSF.getdata()
            nch,npol,_,_=PSF.shape
            for ch in range(nch):
                for pol in range(npol):
                    PSF[ch,pol]=PSF[ch,pol].T[::-1]
                    
            self.PSF=PSF
            self.FitPSF()
            return PSF



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
            #if Res=="EndChunk": break
            if Res=="EndOfObservation": break
            DATA=self.DATA

            FacetMachinePSF.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],
                                     (DATA["A0"],DATA["A1"]),
                                     DATA["Weights"],
                                     doStack=True)

        self.DicoImagePSF=FacetMachinePSF.FacetsToIm(NormJones=True)
        self.DicoVariablePSF=FacetMachinePSF.DicoPSF
        #FacetMachinePSF.ToCasaImage(self.DicoImagePSF["ImagData"],ImageName="%s.psf"%self.BaseName,Fits=True)

        self.PSF=self.DicoImagePSF["MeanImage"]#/np.sqrt(self.DicoImagePSF["NormData"])
        
        self.MeanFacetPSF=self.DicoVariablePSF["MeanFacetPSF"]

        FacetMachinePSF.DoPSF=False

        self.FitPSF()
        if "P" in self._saveims or "p" in self._saveims:
            FacetMachinePSF.ToCasaImage(self.PSF,ImageName="%s.psf"%self.BaseName,Fits=True,beam=self.FWHMBeamAvg)
        
        FacetMachinePSF = None

    def LoadPSF(self,CasaFilePSF):
        self.CasaPSF=image(CasaFilePSF)
        self.PSF=self.CasaPSF.getdata()
        self.CellArcSec=np.abs(self.CasaPSF.coordinates().dict()["direction0"]["cdelt"][0]*60)
        self.CellSizeRad=(self.CellArcSec/3600.)*np.pi/180




    def GiveDirty(self):

        print>>log, ModColor.Str("============================== Making Residual Image ==============================")

        #self.InitFacetMachine()
        
        self.FacetMachine.ReinitDirty()
        isPlotted=False
        
        if self.GD["Stores"]["Dirty"]!=None:
            print>>log, "Reading Dirty image from %s"%self.GD["Stores"]["Dirty"]
            CasaDirty=image(self.GD["Stores"]["Dirty"])
            Dirty=CasaDirty.getdata()
            nch,npol,_,_=Dirty.shape
            for ch in range(nch):
                for pol in range(npol):
                    Dirty[ch,pol]=Dirty[ch,pol].T[::-1]
            return Dirty

        SubstractModel=self.GD["VisData"]["InitDicoModel"]
        DoSub=(SubstractModel!="")&(SubstractModel!=None)
        if DoSub:
            print>>log, ModColor.Str("Initialise sky model using %s"%SubstractModel,col="blue")
            self.DeconvMachine.FromFile(SubstractModel)
            InitBaseName=".".join(SubstractModel.split(".")[0:-1])
            self.FacetMachine.BuildFacetNormImage()

            if self.BaseName==self.GD["VisData"]["InitDicoModel"][0:-10]:
                self.BaseName+=".continue"

        iloop = 0
        while True:
            Res=self.setNextData()
            # if not(isPlotted):
            #     isPlotted=True
            #     self.FacetMachine.PlotFacetSols()
            #     stop
            #if Res=="EndChunk": break
            if Res=="EndOfObservation": break
            DATA=self.DATA

            if DoSub:
                ThisMeanFreq=self.VS.CurrentChanMappingDegrid#np.mean(DATA["freqs"])
                ModelImage=self.DeconvMachine.GiveModelImage(ThisMeanFreq)
                print>>log, "Model image @%s MHz (min,max) = (%f, %f)"%(str(ThisMeanFreq/1e6),ModelImage.min(),ModelImage.max())
                _=self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),ModelImage)

            self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],
                                       (DATA["A0"],DATA["A1"]),
                                       DATA["Weights"],
                                       doStack=True)
            
            if self._save_intermediate_grids:
                self.DicoDirty=self.FacetMachine.FacetsToIm(NormJones=True)
                self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.dirty.%d."%(self.BaseName,iloop),Fits=True)
                self.FacetMachine.NormData = None
                self.FacetMachine.NormImage = None

            iloop += 1

        self.DicoDirty=self.FacetMachine.FacetsToIm(NormJones=True)

        if "d" in self._saveims:
            self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.dirty"%self.BaseName,Fits=True)


        if "n" in self._saveims:
            self.FacetMachine.ToCasaImage(self.FacetMachine.NormImageReShape,ImageName="%s.NormFacets"%self.BaseName,Fits=True)

        if self.DicoDirty["NormData"]!=None:
            #MeanCorr=self.DicoDirty["ImagData"]*self.DicoDirty["NormData"]
            MeanCorr=self.DicoDirty["ImagData"]/np.sqrt(self.DicoDirty["NormData"])
            #MeanCorr=self.DicoDirty["ImagData"]*(self.DicoDirty["NormData"])
            nch,npol,nx,ny=MeanCorr.shape
            MeanCorr=np.mean(MeanCorr,axis=0).reshape((1,npol,nx,ny))
            if "D" in self._saveims:
                self.FacetMachine.ToCasaImage(MeanCorr,ImageName="%s.dirty.corr"%self.BaseName,Fits=True)

            self.MeanNormImage = np.mean(self.DicoDirty["NormData"],axis=0).reshape((1,npol,nx,ny))
            if "N" in self._saveims:
                self.FacetMachine.ToCasaImage(self.MeanNormImage,ImageName="%s.Norm"%self.BaseName,Fits=True)
        else:
            self.MeanNormImage = None

        return self.DicoDirty["MeanImage"]
        

    def GivePredict(self):

        print>>log, ModColor.Str("============================== Making Predict ==============================")
        #self.InitFacetMachine()
        
        self.FacetMachine.ReinitDirty()
        BaseName=self.GD["Images"]["ImageName"]

        ModelMachine=ClassModelMachine(self.GD)
        NormImageName="%s.NormFacets.fits"%BaseName
        CasaNormImage=image(NormImageName)
        NormImage=CasaNormImage.getdata()
        nch,npol,nx,_=NormImage.shape
        for ch in range(nch):
            for pol in range(npol):
                NormImage[ch,pol]=NormImage[ch,pol].T[::-1]

        
        self.FacetMachine.NormImage=NormImage.reshape((nx,nx))

        ModelImage=ClassCasaImage.FileToArray(self.GD["Images"]["PredictModelName"],True)

        while True:
            Res=self.setNextData()
            if Res=="EndOfObservation": break
            DATA=self.DATA
            ThisMeanFreq=np.mean(DATA["freqs"])

            DATA["data"].fill(0)
            self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),ModelImage)
            vis=-DATA["data"]
            PredictColName=self.GD["VisData"]["PredictColName"]

            MSName=self.VS.CurrentMS.MSName
            print>>log, "Writing predicted data in column %s of %s"%(PredictColName,MSName)
            self.VS.CurrentMS.PutVisColumn(PredictColName, vis)


    def main(self,NMajor=None):
        if NMajor==None:
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

            print>>log, ModColor.Str("========================== Runing major Cycle %i ========================="%iMajor)
            
            self.DeconvMachine.Update(DicoImage)

            repMinor, continue_deconv, update_model = self.DeconvMachine.Clean()
            if not update_model:
                break

            self.FacetMachine.ReinitDirty()

            while True:
                #print>>log, "Max model image: %f"%(np.max(self.DeconvMachine._ModelImage))
                #DATA=self.VS.GiveNextVisChunk()            
                #if (DATA==None): break
                Res=self.setNextData()
                #if Res=="EndChunk": break
                if Res=="EndOfObservation": break
                DATA=self.DATA
                
                #visData=DATA["data"]

                ThisMeanFreq=self.VS.CurrentChanMappingDegrid#np.mean(DATA["freqs"])

                ModelImage=self.DeconvMachine.GiveModelImage(ThisMeanFreq)

                print>>log, "Model image @%s MHz (min,max) = (%f, %f)"%(str(ThisMeanFreq/1e6),ModelImage.min(),ModelImage.max())

                self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),ModelImage)


                self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),DATA["Weights"],doStack=True)#,Channel=self.VS.CurrentFreqBand)
                
                # NpShared.DelArray(PredictedDataName)
                del(DATA)


            DicoImage=self.FacetMachine.FacetsToIm(NormJones=True)
            if self.GD["Images"]["MultiFreqMap"]:
                raise NotImplementedError("TODO issue #51: Not yet implemented")
                self.ResidImage=DicoImage["ImagData"] #get residuals cube
            else:
                self.ResidImage=DicoImage["MeanImage"]

            if "e" in self._saveims:
                self.FacetMachine.ToCasaImage(DicoImage["MeanImage"],ImageName="%s.residual%2.2i"%(self.BaseName,iMajor),Fits=True)

            if "o" in self._saveims:
                ModelImage=self.DeconvMachine.GiveModelImage(ThisMeanFreq)
                self.FacetMachine.ToCasaImage(ModelImage,ImageName="%s.model%2.2i"%(self.BaseName,iMajor),Fits=True)

            self.DeconvMachine.ToFile(self.DicoModelName)

            self.HasCleaned=True

        if self.HasCleaned:
            self.Restore()

    def FitPSF(self):
        """
            Fits the PSF to get the parameters for the clean beam used in restoring
            Post conditions:
                self.FWHMBeam: The maj (deg), min (deg), theta (deg) gaussian parameters for the full width half power
                               fits. This should be passed to the FITS file outputs
                self.PSFGaussPars: The maj (rad), min (rad), theta (rad) parameters for the fit of the gaussian
                self.PSFSidelobes: Position of the highest sidelobes (px)
        """
        if self.GD["Images"]["MultiFreqMap"]:
            raise NotImplementedError("TODO issue #51: Not yet implemented")
        else: #fit to a mean PSF image
            PSF = self.MeanFacetPSF
            _, _, x, y = np.where(PSF == np.max(PSF))
            self.FWHMBeam = []
            self.PSFGaussPars = []
            self.PSFSidelobes = []
            off=self.GD["ImagerDeconv"]["SidelobeSearchWindow"]//2
            print>>log, "Try fitting PSF in a [%i,%i] box ..."%(off*2,off*2)
            P=PSF[0, 0, x[0]-off:x[0]+off,y[0]-off:y[0]+off]
            self.PSFSidelobes.append(ModFitPSF.FindSidelobe(P))
            bmaj, bmin, theta = ModFitPSF.FitCleanBeam(P)
            print>>log, "   ... done"

            FWHMFact=2.*np.sqrt(2.*np.log(2.))

            self.FWHMBeam.append((bmaj*self.CellArcSec*FWHMFact/3600.,
                                  bmin*self.CellArcSec*FWHMFact/3600.,
                                  np.rad2deg(theta)))
            self.PSFGaussPars.append((bmaj*self.CellSizeRad, bmin*self.CellSizeRad, theta))
            PA = 90-np.rad2deg(theta) #image is only transposed when written to FITS, so print the correct angle
            print>>log, "\tFitted PSF (sigma): (Sx, Sy, Th)=(%f, %f, %f deg)"%(bmaj*self.CellArcSec,
                                                                           bmin*self.CellArcSec,
                                                                           PA)
            print>>log, "\tFitted PSF (FWHM):  (Sx, Sy, Th)=(%f, %f, %f deg)"%(bmaj*self.CellArcSec*FWHMFact,
                                                                           bmin*self.CellArcSec*FWHMFact,
                                                                           PA)
            print>>log, "\tSecondary sidelobe at the level of %5.1f at a position of %i from the center" % self.PSFSidelobes[0]
        self.FWHMBeamAvg = (np.average(np.array([tup[0] for tup in self.FWHMBeam])),
                            np.average(np.array([tup[1] for tup in self.FWHMBeam])),
                            np.average(np.array([tup[2] for tup in self.FWHMBeam])))
        self.PSFGaussParsAvg = (np.average(np.array([tup[0] for tup in self.PSFGaussPars])),
                                np.average(np.array([tup[1] for tup in self.PSFGaussPars])),
                                np.average(np.array([tup[2] for tup in self.PSFGaussPars])))
        self.PSFSidelobesAvg = (np.average(np.array([tup[0] for tup in self.PSFSidelobes])),
                                int(np.round(np.average(np.array([tup[1] for tup in self.PSFSidelobes])))))

    def Restore(self):
        print>>log, "Create restored image"
        if self.PSFGaussPars==None:
            self.FitPSF()
        self.DeconvMachine.ToFile(self.DicoModelName)

        RefFreq=self.VS.RefFreq
        ModelMachine=self.DeconvMachine.ModelMachine

        # Putting back substracted componants
        if self.GD["DDESolutions"]["RestoreSub"]:
            try:
                ModelMachine.PutBackSubsComps()
            except:
                print>>log, ModColor.Str("Failed Putting back substracted components")


        # model image
        if self.GD["Images"]["MultiFreqMap"]:
            NotImplementedError("TODO issue #51: Not yet implemented")
            modelImage = ModelMachine.GiveModelImage(np.array(
                [np.average(np.array(self.VS.FreqBandsInfos[band])) for band in self.VS.FreqBandsInfos]))
        else:
            modelImage = ModelMachine.GiveModelImage(RefFreq) #Save only reference frequency model

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
        def appres():
            return self.ResidImage
        def intres():
            label = 'intres'
            if label not in _images:
                _images[label] = intres = self.ResidImage/sqrtnorm() if havenorm else self.ResidImage 
                intres[~np.isfinite(intres)] = 0
            return _images[label]
        def appmodel():
            label = 'appmodel'
            if label not in _images:
                _images[label] = intmodel()*sqrtnorm() if havenorm else intmodel()
            return _images[label]
        def intmodel():
            return modelImage
        def appconvmodel():
            label = 'appconvmodel'
            if label not in _images:
                _images[label] = ModFFTW.ConvolveGaussian(appmodel(),CellSizeRad=self.CellSizeRad,GaussPars=self.PSFGaussPars) \
                                    if havenorm else intconvmodel()
            return _images[label]
        def intconvmodel():
            label = 'intconvmodel'
            if label not in _images:
                _images[label] = ModFFTW.ConvolveGaussian(intmodel(),CellSizeRad=self.CellSizeRad,GaussPars=self.PSFGaussPars)
            return _images[label]

        # norm
        if havenorm and ("S" in self._saveims or "s" in self._saveims):
            self.FacetMachine.ToCasaImage(sqrtnorm(),ImageName="%s.fluxscale"%(self.BaseName),Fits=True)

        # apparent-flux residuals
        if "r" in self._saveims:
            self.FacetMachine.ToCasaImage(appres(),ImageName="%s.app.residual"%(self.BaseName),Fits=True)
        # intrinsic-flux residuals
        if havenorm and "R" in self._saveims:
            self.FacetMachine.ToCasaImage(intres(),ImageName="%s.int.residual"%(self.BaseName),Fits=True)

        # apparent-flux model
        if "m" in self._saveims:
            self.FacetMachine.ToCasaImage(appmodel(),ImageName="%s.app.model"%self.BaseName,Fits=True)
        # intrinsic-flux model
        if havenorm and "M" in self._saveims:
            self.FacetMachine.ToCasaImage(intmodel(),ImageName="%s.int.model"%self.BaseName,Fits=True)

        # convolved-model image in apparent flux
        if "c" in self._saveims:
            self.FacetMachine.ToCasaImage(appconvmodel(),ImageName="%s.app.convmodel"%self.BaseName,Fits=True,beam=self.FWHMBeamAvg)
        # convolved-model image in intrinsic flux
        if havenorm and "C" in self._saveims: 
            self.FacetMachine.ToCasaImage(intconvmodel(),ImageName="%s.int.convmodel"%self.BaseName,Fits=True,beam=self.FWHMBeamAvg)

        # apparent-flux restored image
        if "i" in self._saveims:
            self.FacetMachine.ToCasaImage(appres()+appconvmodel(),ImageName="%s.app.restored"%self.BaseName,Fits=True,beam=self.FWHMBeamAvg)
        # intrinsic-flux restored image
        if havenorm and "I" in self._saveims:
            self.FacetMachine.ToCasaImage(intres()+intconvmodel(),ImageName="%s.int.restored"%self.BaseName,Fits=True,beam=self.FWHMBeamAvg)
        # mixed-flux restored image
        if havenorm and "x" in self._saveims:
            self.FacetMachine.ToCasaImage(appres()+intconvmodel(),ImageName="%s.restored"%self.BaseName,Fits=True,beam=self.FWHMBeamAvg)
        
        # Alpha image
        if "A" in self._saveims and self.VS.MultiFreqMode:
            IndexMap=ModelMachine.GiveSpectralIndexMap(CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussParsAvg])
            # IndexMap=ModFFTW.ConvolveGaussian(IndexMap,CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars],Normalise=True)
            self.FacetMachine.ToCasaImage(IndexMap,ImageName="%s.alpha"%self.BaseName,Fits=True,beam=self.FWHMBeamAvg)

    def testDegrid(self):
        self.InitFacetMachine()
        
        self.FacetMachine.ReinitDirty()
        Res=self.setNextData()
        #if Res=="EndChunk": break

        DATA=self.DATA

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
        
        self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],visData,DATA["flags"],(DATA["A0"],DATA["A1"]),DATA["Weights"])
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
