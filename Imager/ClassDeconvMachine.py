

from ClassFacetMachine import ClassFacetMachine
from ClassFacetMachineTessel2 import ClassFacetMachineTessel as ClassFacetMachine
import numpy as np
import pylab
#import ToolsDir
from DDFacet.Other import MyPickle
from pyrap.images import image
import ClassImageDeconvMachineMultiScale
import ClassImageDeconvMachineSingleScale
import ClassImageDeconvMachineMSMF
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Other import ModColor
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassImagerDeconv")
from DDFacet.Array import NpShared
import os
from DDFacet.ToolsDir import ModFitPSF
#from ClassData import ClassMultiPointingData,ClassSinglePointingData,ClassGlobalData
from DDFacet.Data import ClassVisServer
from DDFacet.Other import MyPickle
import ClassCasaImage
from ClassModelMachine import ClassModelMachine
from pyrap.tables import table

import time
import glob

def test():
    Imager=ClassImagerDeconv(ParsetFile="ParsetDDFacet.txt")
    #Imager.MakePSF()
    #Imager.LoadPSF("PSF.image")
    # Imager.FitPSF()
    # Imager.main(NMajor=5)
    # Imager.Restore()


    Imager.Init()
    #Model=np.zeros(Imager.FacetMachine.OutImShape,np.complex64)
    #Model[0,0,100,100]=1
    #Imager.GivePredict(Model)
    #Imager.MakePSF()
    #Imager.GiveDirty()
    #Imager.main()
    Imager.testDegrid()
    return Imager

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
        
        # self.VS.setFOV([1,1,1000,1000],[1,1,1000,1000],[1,1,1000,1000],2./3600.*np.pi/180)
        # self.VS.CalcWeights()
        # for i in range(10):
        #     print>>log, self.setNextData()
        # stop
        

        if self.DoDeconvolve:
            self.NMajor=self.GD["ImagerDeconv"]["MaxMajorIter"]
            del(self.GD["ImagerDeconv"]["MaxMajorIter"])
            MinorCycleConfig=dict(self.GD["ImagerDeconv"])
            MinorCycleConfig["NCPU"]=self.GD["Parallel"]["NCPU"]
            MinorCycleConfig["NFreqBands"]=self.VS.NFreqBands
            
            if self.GD["MultiScale"]["MSEnable"]:
                print>>log, "Minor cycle deconvolution in Multi Scale Mode" 
                self.MinorCycleMode="MS"
                MinorCycleConfig["GD"]=self.GD
                #self.DeconvMachine=ClassImageDeconvMachineMultiScale.ClassImageDeconvMachine(**MinorCycleConfig)
                self.DeconvMachine=ClassImageDeconvMachineMSMF.ClassImageDeconvMachine(
                    **MinorCycleConfig)
            else:
                print>>log, "Minor cycle deconvolution in Single Scale Mode" 
                self.MinorCycleMode="SS"
                self.DeconvMachine=ClassImageDeconvMachineSingleScale.ClassImageDeconvMachine(
                    **MinorCycleConfig)

        self.InitFacetMachine()
        #self.VS.SetImagingPars(self.FacetMachine.OutImShape,self.FacetMachine.CellSizeRad)
        #self.VS.CalcWeights(self.FacetMachine.OutImShape,self.FacetMachine.CellSizeRad)
        self.VS.setFacetMachine(self.FacetMachine)
        self.VS.CalcWeights()




    def InitFacetMachine(self):
        if self.FacetMachine!=None:
            return

        
        #print "initFacetMachine deconv0"; self.IM.CI.E.clear()
        ApplyCal=False
        SolsFile=self.GD["DDESolutions"]["DDSols"]
        if (SolsFile!="")|(self.GD["Beam"]["BeamModel"]!=None): ApplyCal=True

        self.FacetMachine=ClassFacetMachine(self.VS,self.GD,Precision=self.Precision,PolMode=self.PolMode,Parallel=self.Parallel,
                                                              IdSharedMem=self.IdSharedMem,ApplyCal=ApplyCal)#,Sols=SimulSols)

        
        #print "initFacetMachine deconv1"; self.IM.CI.E.clear()
        MainFacetOptions=self.GiveMainFacetOptions()
        self.FacetMachine.appendMainField(ImageName="%s.image"%self.BaseName,**MainFacetOptions)
        self.FacetMachine.Init()
        #print "initFacetMachine deconv2"; self.IM.CI.E.clear()

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
        # self.CellSizeRad=(FacetMachinePSF.Cell/3600.)*np.pi/180
        # self.CellArcSec=FacetMachinePSF.Cell

        # #FacetMachinePSF.ToCasaImage(None)

        FacetMachinePSF.ReinitDirty()
        FacetMachinePSF.DoPSF=True

        while True:
            Res=self.setNextData()
            #if Res=="EndChunk": break
            if Res=="EndOfObservation": break
            DATA=self.DATA

            FacetMachinePSF.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),DATA["Weights"],doStack=True)#,Channel=self.VS.CurrentFreqBand)


            # Image=FacetMachinePSF.FacetsToIm()
            # pylab.clf()
            # pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            # break

        self.DicoImagePSF=FacetMachinePSF.FacetsToIm(NormJones=True)
        self.DicoVariablePSF=FacetMachinePSF.DicoPSF
        #FacetMachinePSF.ToCasaImage(self.DicoImagePSF["ImagData"],ImageName="%s.psf"%self.BaseName,Fits=True)

        #np.savez("PSF.npz",ImagData=self.DicoImagePSF["ImagData"],MeanImage=self.DicoImagePSF["MeanImage"])

        self.PSF=self.DicoImagePSF["MeanImage"]#/np.sqrt(self.DicoImagePSF["NormData"])
        
        self.MeanFacetPSF=self.DicoVariablePSF["MeanFacetPSF"]

        # ImageName="%s.psf.corr.MF"%self.BaseName
        # ImagData=self.DicoImagePSF["ImagData"]#/np.sqrt(self.DicoImagePSF["NormData"])
        # im=ClassCasaImage.ClassCasaimage(ImageName,ImagData.shape,self.FacetMachine.Cell,self.FacetMachine.MainRaDec)
        # im.setdata(ImagData,CorrT=True)
        # im.ToFits()
        # im.close()

        FacetMachinePSF.DoPSF=False

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

        self.FitPSF()
        if "P" in self._saveims or "p" in self._saveims:
            FacetMachinePSF.ToCasaImage(self.PSF,ImageName="%s.psf"%self.BaseName,Fits=True,beam=self.FWHMBeamAvg)
        
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


    def LoadPSF(self,CasaFilePSF):
        self.CasaPSF=image(CasaFilePSF)
        self.PSF=self.CasaPSF.getdata()
        self.CellArcSec=np.abs(self.CasaPSF.coordinates().dict()["direction0"]["cdelt"][0]*60)
        self.CellSizeRad=(self.CellArcSec/3600.)*np.pi/180




    def GiveDirty(self):

        print>>log, ModColor.Str("============================== Making Residual Image ==============================")

        self.InitFacetMachine()
        
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
            self.DeconvMachine.ModelMachine.FromFile(SubstractModel)        
            InitBaseName=".".join(SubstractModel.split(".")[0:-1])
            self.FacetMachine.BuildFacetNormImage()
            # NormFacetsFile="%s.NormFacets.fits"%InitBaseName
            # if InitBaseName!=BaseName:
            #     print>>log, ModColor.Str("You are substracting a model build from a different facetting mode")
            #     print>>log, ModColor.Str("  This is rather dodgy because of the ")
            # self.FacetMachine.NormImage=ClassCasaImage.FileToArray(NormFacetsFile,True)
            # _,_,nx,nx=self.FacetMachine.NormImage.shape
            # self.FacetMachine.NormImage=self.FacetMachine.NormImage.reshape((nx,nx))

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

            self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),DATA["Weights"],doStack=True)#,Channel=self.VS.CurrentFreqBand)
            
            if self._save_intermediate_grids:
                self.DicoDirty=self.FacetMachine.FacetsToIm(NormJones=True)
                self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.dirty.%d."%(self.BaseName,iloop),Fits=True)
                if 'g' in self._savecubes:
                    self.FacetMachine.ToCasaImage(self.DicoDirty["ImagData"],ImageName="%s.cube.dirty.%d"%(self.BaseName,iloop),
                        Fits=True,Freqs=self.VS.FreqBandCenters) 
                self.FacetMachine.NormData = None
                self.FacetMachine.NormImage = None

            iloop += 1
            
            
            # Image=self.FacetMachine.FacetsToIm()
            # pylab.clf()
            # pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
            # pylab.colorbar()
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)

        self.DicoDirty=self.FacetMachine.FacetsToIm(NormJones=True)
        
        # self.DicoDirty=self.FacetMachine.FacetsToIm()

        if "d" in self._saveims:
            self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.dirty"%self.BaseName,Fits=True)
        if "d" in self._savecubes:
            self.FacetMachine.ToCasaImage(self.DicoDirty["ImagData"],ImageName="%s.cube.dirty"%self.BaseName,
                    Fits=True,Freqs=self.VS.FreqBandCenters) 

        # ImageName="%s.dirty.MF"%self.BaseName
        # ImagData=self.DicoDirty["ImagData"]
        # im=ClassCasaImage.ClassCasaimage(ImageName,ImagData.shape,self.FacetMachine.Cell,self.FacetMachine.MainRaDec)
        # im.setdata(ImagData,CorrT=True)
        # im.ToFits()
        # im.close()


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

        #if self.VS.MultiFreqMode:
        #    for Channel in range(

        #np.savez("Dirty.npz",ImagData=self.DicoDirty["ImagData"],MeanImage=self.DicoDirty["MeanImage"],NormData=self.DicoDirty["NormData"])
        #print self.DicoDirty["freqs"]

        #MyPickle.Save(DicoImage,"DicoDirty")

        return self.DicoDirty["MeanImage"]


        #m0,m1=Image.min(),Image.max()
        
        # pylab.clf()
        # pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        

    def GivePredict(self):

        print>>log, ModColor.Str("============================== Making Predict ==============================")
        self.InitFacetMachine()
        
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

        modelfile = self.GD["Images"]["PredictModelName"]

        # if model is a dict, init model machine with that
        # else we use a model image and hope for the best (need to fix frequency axis...)
        if modelfile.endswith(".DicoModel"):
            ModelMachine.FromFile(modelfile)
            FixedModelImage = None
        else:
            FixedModelImage = ClassCasaImage.FileToArray(modelfile,True)

        current_model_freqs = np.array([])

        while True:
            Res=self.setNextData()
            if Res=="EndOfObservation": break
            DATA=self.DATA

            model_freqs = self.VS.CurrentChanMappingDegrid
            ## redo model image if needed
            if FixedModelImage is None:
                if np.array(model_freqs != current_model_freqs).any():
                    ModelImage = ModelMachine.GiveModelImage(model_freqs)
                    current_model_freqs = model_freqs
                    print>>log, "Model image @%s MHz (min,max) = (%f, %f)"%(str(model_freqs/1e6),ModelImage.min(),ModelImage.max())
                else:
                    print>>log,"reusing model image from previous chunk"
            else:
                ModelImage = FixedModelImage
            

            DATA["data"].fill(0)
            self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),ModelImage)
            vis=-DATA["data"]
            PredictColName=self.GD["VisData"]["PredictColName"]

            MSName=self.VS.CurrentMS.MSName
            print>>log, "Writing predicted data to column %s of %s"%(PredictColName,MSName)
            self.VS.CurrentMS.PutVisColumn(PredictColName, vis)
            

    def setPSF(self):

        self.MakePSF()
        self.DeconvMachine.SetPSF(self.DicoVariablePSF)
        #initialize the deconvolve machine with the first side lobe level and offset:
        self.DeconvMachine.setSideLobeLevel(self.PSFSidelobesAvg[0], self.PSFSidelobesAvg[1])
        self.DeconvMachine.InitMSMF()
        


    def main(self,NMajor=None):
        if NMajor==None:
            NMajor=self.NMajor

        self.GiveDirty()
        self.setPSF()
        
        DicoImage=self.DicoDirty
        continue_deconv = True

        
        for iMajor in range(NMajor):
            # previous minor loop indicated it has reached bottom? Break out
            if not continue_deconv:
                break

            print>>log, ModColor.Str("========================== Running major Cycle %i ========================="%iMajor)
            
            self.DeconvMachine.SetDirty(DicoImage)
            #self.DeconvMachine.setSideLobeLevel(0.2,10)

            repMinor, continue_deconv, update_model = self.DeconvMachine.Clean()
            ## returned with nothing done in minor cycle? Break out
            if not update_model or iMajor == NMajor-1:
                continue_deconv = False

            predict_colname = not continue_deconv and self.GD["VisData"]["PredictColName"]

            #self.ResidImage=DicoImage["MeanImage"]
            #self.FacetMachine.ToCasaImage(DicoImage["MeanImage"],ImageName="%s.residual_sub%i"%(self.BaseName,iMajor),Fits=True)

            self.FacetMachine.ReinitDirty()

            current_model_freqs = np.array([])

            while True:
                #print>>log, "Max model image: %f"%(np.max(self.DeconvMachine._ModelImage))
                #DATA=self.VS.GiveNextVisChunk()            
                #if (DATA==None): break
                Res=self.setNextData()
                #if Res=="EndChunk": break
                if Res=="EndOfObservation": break
                DATA=self.DATA
                

                model_freqs = self.VS.CurrentChanMappingDegrid
                ## redo model image if needed
                if np.array(model_freqs != current_model_freqs).any():
                    ModelImage = self.DeconvMachine.GiveModelImage(model_freqs)
                    current_model_freqs = model_freqs
                    print>>log,"model image @%s MHz (min,max) = (%f, %f)"%(str(model_freqs/1e6),ModelImage.min(),ModelImage.max())
                else:
                    print>>log,"reusing model image from previous chunk"

                if predict_colname:
                    print>>log,"last major cycle: model visibilities will be stored to %s"%predict_colname
                    modelvis = DATA["data"].copy()

                self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),ModelImage)

                if predict_colname:
                    modelvis -= DATA["data"]
                    print>>log, "writing model visibilities to column %s" % predict_colname
                    self.VS.CurrentMS.PutVisColumn(predict_colname, modelvis)
                    del modelvis

                self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),DATA["Weights"],doStack=True)

                # NpShared.DelArray(PredictedDataName)
                del DATA


            DicoImage=self.FacetMachine.FacetsToIm(NormJones=True)
            self.ResidCube  = DicoImage["ImagData"] #get residuals cube
            self.ResidImage = DicoImage["MeanImage"]

            if "e" in self._saveims:
                self.FacetMachine.ToCasaImage(self.ResidImage,ImageName="%s.residual%2.2i"%(self.BaseName,iMajor),Fits=True)

            if "o" in self._saveims:
                self.FacetMachine.ToCasaImage(ModelImage,ImageName="%s.model%2.2i"%(self.BaseName,iMajor),
                    Fits=True,Freqs=current_model_freqs)

            self.DeconvMachine.ModelMachine.ToFile(self.DicoModelName)

            # fig=pylab.figure(1)
            # pylab.clf()
            # ax=pylab.subplot(1,2,1)
            # pylab.imshow(self.ResidImage[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
            # pylab.colorbar()
            # pylab.subplot(1,2,2,sharex=ax,sharey=ax)
            # pylab.imshow(ModelImage[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
            # pylab.colorbar()
            # pylab.draw()
            # #PNGName="%s/Residual%3.3i.png"%(self.PNGDir,iMajor)
            # #fig.savefig(PNGName)
            # pylab.show(False)
            # pylab.pause(0.1)
            # #stop

            self.HasCleaned=True

        #self.FacetMachine.ToCasaImage(Image,ImageName="%s.residual"%self.BaseName,Fits=True)
        if self.HasCleaned:
            self.Restore()

    def fitSinglePSF(self, PSF, label="mean"):
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
        off = self.GD["ImagerDeconv"]["SidelobeSearchWindow"] // 2
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
                                                                           np.rad2deg(theta))
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
        PSF = self.DicoVariablePSF["CubeVariablePSF"][self.FacetMachine.iCentralFacet]

        self.FWHMBeamAvg, self.PSFGaussParsAvg, self.PSFSidelobesAvg = \
            self.fitSinglePSF(self.MeanFacetPSF[0,...], "mean")
        # MeanFacetPSF has a shape of 1,1,nx,ny, so need to cut that extra one off

        if self.VS.MultiFreqMode:
            self.FWHMBeam = []
            self.PSFGaussPars = []
            self.PSFSidelobes = []
            for band in range(self.VS.NFreqBands):
                beam, gausspars, sidelobes = self.fitSinglePSF(PSF[band,...],"band %d"%band)
                self.FWHMBeam.append(beam)
                self.PSFGaussPars.append(gausspars)
                self.PSFSidelobes.append(sidelobes)
        else:
            self.FWHMBeam = [self.FWHMBeamAvg]
            self.PSFGaussPars = [self.PSFGaussParsAvg]
            self.PSFSidelobes = [self.PSFSidelobesAvg]

    def Restore(self):
        print>>log, "Create restored image"
        if self.PSFGaussPars==None:
            self.FitPSF()
        self.DeconvMachine.ModelMachine.ToFile(self.DicoModelName)

        RefFreq = self.VS.RefFreq
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
                _images[label] = x = apprescube()/sqrtnorm() if havenorm else apprescube()
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
                _images[label] = intmodelcube()*sqrtnorm() if havenorm else intmodel()
            return _images[label]
        def intmodelcube():
            label = 'intmodelcube'
            if label not in _images:
                _images[label] = ModelMachine.GiveModelImage(self.VS.FreqBandCenters)
            return _images[label]
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
            self.FacetMachine.ToCasaImage(sqrtnorm(),ImageName="%s.fluxscale"%(self.BaseName),Fits=True)

        # apparent-flux residuals
        if "r" in self._saveims:
            self.FacetMachine.ToCasaImage(appres(),ImageName="%s.app.residual"%(self.BaseName),Fits=True)
        # intrinsic-flux residuals
        if havenorm and "R" in self._saveims:
            self.FacetMachine.ToCasaImage(intres(),ImageName="%s.int.residual"%(self.BaseName),Fits=True)
        # apparent-flux residual cube
        if "r" in self._savecubes:
            self.FacetMachine.ToCasaImage(apprescube(),ImageName="%s.cube.app.residual"%(self.BaseName),Fits=True,
                Freqs=self.VS.FreqBandCenters)
        # intrinsic-flux residual cube
        if havenorm and "R" in self._savecubes:
            self.FacetMachine.ToCasaImage(intrescube(),ImageName="%s.cube.int.residual"%(self.BaseName),Fits=True,
                Freqs=self.VS.FreqBandCenters)

        # apparent-flux model
        if "m" in self._saveims:
            self.FacetMachine.ToCasaImage(appmodel(),ImageName="%s.app.model"%self.BaseName,Fits=True)
        # intrinsic-flux model
        if havenorm and "M" in self._saveims:
            self.FacetMachine.ToCasaImage(intmodel(),ImageName="%s.int.model"%self.BaseName,Fits=True)
        # apparent-flux model cube
        if "m" in self._savecubes:
            self.FacetMachine.ToCasaImage(appmodelcube(),ImageName="%s.cube.app.model"%self.BaseName,Fits=True,
                Freqs=self.VS.FreqBandCenters)
        # intrinsic-flux model cube
        if havenorm and "M" in self._savecubes:
            self.FacetMachine.ToCasaImage(intmodelcube(),ImageName="%s.cube.int.model"%self.BaseName,Fits=True,
                Freqs=self.VS.FreqBandCenters)

        # convolved-model image in apparent flux
        if "c" in self._saveims:
            self.FacetMachine.ToCasaImage(appconvmodel(),ImageName="%s.app.convmodel"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg)
        # convolved-model image in intrinsic flux
        if havenorm and "C" in self._saveims: 
            self.FacetMachine.ToCasaImage(intconvmodel(),ImageName="%s.int.convmodel"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg)
        # convolved-model cube in apparent flux 
        if "c" in self._savecubes:
            self.FacetMachine.ToCasaImage(appconvmodelcube(),ImageName="%s.cube.app.convmodel"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,beamcube=self.FWHMBeam,Freqs=self.VS.FreqBandCenters)
        # convolved-model cube in intrinsic flux
        if havenorm and "C" in self._savecubes: 
            self.FacetMachine.ToCasaImage(intconvmodelcube(),ImageName="%s.cube.int.convmodel"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,beamcube=self.FWHMBeam,Freqs=self.VS.FreqBandCenters)

        # apparent-flux restored image
        if "i" in self._saveims:
            self.FacetMachine.ToCasaImage(appres()+appconvmodel(),ImageName="%s.app.restored"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg)
        # intrinsic-flux restored image
        if havenorm and "I" in self._saveims:
            self.FacetMachine.ToCasaImage(intres()+intconvmodel(),ImageName="%s.int.restored"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg)
        # apparent-flux restored image cube
        if "i" in self._savecubes:
            self.FacetMachine.ToCasaImage(apprescube()+appconvmodelcube(),ImageName="%s.cube.app.restored"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,beamcube=self.FWHMBeam,Freqs=self.VS.FreqBandCenters)
        # intrinsic-flux restored image cube
        if havenorm and "I" in self._savecubes:
            self.FacetMachine.ToCasaImage(intrescube()+intconvmodelcube(),ImageName="%s.cube.int.restored"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg,beamcube=self.FWHMBeam,Freqs=self.VS.FreqBandCenters)
        # mixed-flux restored image
        if havenorm and "x" in self._saveims:
            self.FacetMachine.ToCasaImage(appres()+intconvmodel(),ImageName="%s.restored"%self.BaseName,Fits=True,
                beam=self.FWHMBeamAvg)
        
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
        

