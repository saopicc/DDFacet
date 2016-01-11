

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
        self.PSFGaussPars = None
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
        

    def Init(self):
        DC=self.GD

        
        MSName0 = MSName = DC["VisData"]["MSName"]

        if ".txt" in MSName:#DC["VisData"]["MSListFile"]!="":
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
            submss = os.path.join(MSName,"SUBMSS") 
            if os.path.exists(submss) and os.path.isdir(submss):
                MSName = sorted(glob.glob(os.path.join(submss,"*")))
                print>>log,"multi-MS mode for %s, found %d sub-MSs"%(MSName0, len(MSName))
            else:
                print>>log,"single-MS mode for %s"%MSName


        self.VS=ClassVisServer.ClassVisServer(MSName,
                                              ColName=DC["VisData"]["ColName"],
                                              Field=DC["VisData"]["Field"],
                                              TVisSizeMin=DC["VisData"]["TChunkSize"]*60,
                                              #DicoSelectOptions=DicoSelectOptions,
                                              TChunkSize=DC["VisData"]["TChunkSize"],
                                              IdSharedMem=self.IdSharedMem,
                                              Robust=DC["ImagerGlobal"]["Robust"],
                                              Weighting=DC["ImagerGlobal"]["Weighting"],
                                              DicoSelectOptions=dict(DC["DataSelection"]),
                                              NCPU=self.GD["Parallel"]["NCPU"],
                                              GD=self.GD)
        
        # self.VS.setFOV([1,1,1000,1000],[1,1,1000,1000],[1,1,1000,1000],2./3600.*np.pi/180)
        # self.VS.CalcWeigths()
        # for i in range(10):
        #     print>>log, self.setNextData()
        # stop
        

        if self.DoDeconvolve:
            self.NMajor=self.GD["ImagerDeconv"]["MaxMajorIter"]
            del(self.GD["ImagerDeconv"]["MaxMajorIter"])
            MinorCycleConfig=dict(self.GD["ImagerDeconv"])
            MinorCycleConfig["NCPU"]=self.GD["Parallel"]["NCPU"]
            
            if self.GD["MultiScale"]["MSEnable"]:
                print>>log, "Minor cycle deconvolution in Multi Scale Mode" 
                self.MinorCycleMode="MS"
                MinorCycleConfig["GD"]=self.GD
                #self.DeconvMachine=ClassImageDeconvMachineMultiScale.ClassImageDeconvMachine(**MinorCycleConfig)
                self.DeconvMachine=ClassImageDeconvMachineMSMF.ClassImageDeconvMachine(**MinorCycleConfig)
            else:
                print>>log, "Minor cycle deconvolution in Single Scale Mode" 
                self.MinorCycleMode="SS"
                self.DeconvMachine=ClassImageDeconvMachineSingleScale.ClassImageDeconvMachine(**MinorCycleConfig)

        self.InitFacetMachine()
        #self.VS.SetImagingPars(self.FacetMachine.OutImShape,self.FacetMachine.CellSizeRad)
        #self.VS.CalcWeigths(self.FacetMachine.OutImShape,self.FacetMachine.CellSizeRad)
        self.VS.setFacetMachine(self.FacetMachine)
        self.VS.CalcWeigths()




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
        oversize = self.GD["ImagerGlobal"]["PSFOversize"]
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

        self.PSF=self.DicoImagePSF["MeanImage"]/np.sqrt(self.DicoImagePSF["NormData"])

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



        # self.FWHMBeam=(10.,10.,10.)
        # FacetMachinePSF.ToCasaImage(self.PSF)



        # FacetMachinePSF.ToCasaImage(self.PSF,ImageName="%s.psf"%self.BaseName,Fits=True)

        self.FitPSF()
        FacetMachinePSF.ToCasaImage(self.PSF,ImageName="%s.psf"%self.BaseName,Fits=True,beam=self.FWHMBeam)

        FacetMachinePSF = None

        # if self.VS.MultiFreqMode:
        #     for Channel in range(self.VS.NFreqBands):
        #         Im=self.DicoImagePSF["ImagData"][Channel]
        #         npol,n,n=Im.shape
        #         Im=Im.reshape((1,npol,n,n))
        #         FacetMachinePSF.ToCasaImage(Im,ImageName="%s.psf.ch%i"%(self.BaseName,Channel),Fits=True,beam=self.FWHMBeam)

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
                ThisMeanFreq=np.mean(DATA["freqs"])
                ModelImage=self.DeconvMachine.GiveModelImage(ThisMeanFreq)
                print>>log, "Model image @%f MHz (min,max) = (%f, %f)"%(ThisMeanFreq/1e6,ModelImage.min(),ModelImage.max())
                _=self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),ModelImage)

            self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),DATA["Weights"],doStack=True)#,Channel=self.VS.CurrentFreqBand)
            
            
            
            # Image=self.FacetMachine.FacetsToIm()
            # pylab.clf()
            # pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
            # pylab.colorbar()
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)

        self.DicoDirty=self.FacetMachine.FacetsToIm(NormJones=True)
        
        # self.DicoDirty=self.FacetMachine.FacetsToIm()

        self.FacetMachine.ToCasaImage(self.DicoDirty["MeanImage"],ImageName="%s.dirty"%self.BaseName,Fits=True)

        # ImageName="%s.dirty.MF"%self.BaseName
        # ImagData=self.DicoDirty["ImagData"]
        # im=ClassCasaImage.ClassCasaimage(ImageName,ImagData.shape,self.FacetMachine.Cell,self.FacetMachine.MainRaDec)
        # im.setdata(ImagData,CorrT=True)
        # im.ToFits()
        # im.close()


        self.FacetMachine.ToCasaImage(self.FacetMachine.NormImageReShape,ImageName="%s.NormFacets"%self.BaseName,Fits=True)

        if self.DicoDirty["NormData"]!=None:
            #MeanCorr=self.DicoDirty["ImagData"]*self.DicoDirty["NormData"]
            MeanCorr=self.DicoDirty["ImagData"]/np.sqrt(self.DicoDirty["NormData"])
            #MeanCorr=self.DicoDirty["ImagData"]*(self.DicoDirty["NormData"])
            nch,npol,nx,ny=MeanCorr.shape
            MeanCorr=np.mean(MeanCorr,axis=0).reshape((1,npol,nx,ny))
            self.FacetMachine.ToCasaImage(MeanCorr,ImageName="%s.dirty.corr"%self.BaseName,Fits=True)

            MeanNorm=np.mean(self.DicoDirty["NormData"],axis=0).reshape((1,npol,nx,ny))
            self.FacetMachine.ToCasaImage(MeanNorm,ImageName="%s.Norm"%self.BaseName,Fits=True)
        
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

        ModelImage=ClassCasaImage.FileToArray(self.GD["Images"]["PredictModelName"],True)

        while True:
            Res=self.setNextData()
            if Res=="EndOfObservation": break
            DATA=self.DATA
            ThisMeanFreq=np.mean(DATA["freqs"])

            

            # ######################################
            # ModelMachine.DicoSMStacked["Comp"][(153, 570)]['SolsArray']=np.array([ 10], dtype=np.float32)
            # #ModelMachine.DicoSMStacked["Comp"][(11275, 9821)]['SolsArray']=np.array([ 10], dtype=np.float32)
            # ######################################
                
            # ModelImage=ModelMachine.GiveModelImage(ThisMeanFreq)

            DATA["data"].fill(0)
            self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),ModelImage)
            vis=-DATA["data"]
            PredictColName=self.GD["VisData"]["PredictColName"]

            MSName=self.VS.CurrentMS.MSName
            print>>log, "Writing predicted data in column %s of %s"%(PredictColName,MSName)
            self.VS.CurrentMS.AddCol(PredictColName)
            t=table(MSName,readonly=False,ack=False)
            t.putcol(PredictColName,vis)
            t.close()
            

    def setPSF(self):

        self.MakePSF()
        self.DeconvMachine.SetPSF(self.DicoVariablePSF)
        self.DeconvMachine.setSideLobeLevel(self.SideLobeLevel,self.OffsetSideLobe)
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

            print>>log, ModColor.Str("========================== Runing major Cycle %i ========================="%iMajor)
            
            self.DeconvMachine.SetDirty(DicoImage)
            #self.DeconvMachine.setSideLobeLevel(0.2,10)

            repMinor, continue_deconv, update_model = self.DeconvMachine.Clean()
            ## returned with nothing done in minor cycle? Break out
            if not update_model:
                break


            #self.ResidImage=DicoImage["MeanImage"]
            #self.FacetMachine.ToCasaImage(DicoImage["MeanImage"],ImageName="%s.residual_sub%i"%(self.BaseName,iMajor),Fits=True)

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

                # # stop
                # # # ModelImage.fill(0)
                # # # ModelImage[:,:,487, 487]=0.88
                # # # ####################
                # # # testImage=np.zeros((1, 1, 1008, 1008),np.complex64)
                # # # testImage[0,0,200,650]=100.
                # # # self.DeconvMachine._ModelImage=testImage
                # # # ####################
                
                # # # PredictedDataName="%s%s"%(self.IdSharedMem,"predicted_data")
                # # # visPredict=NpShared.zeros(PredictedDataName,visData.shape,visData.dtype)
                # # # _=self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],visPredict,DATA["flags"],(DATA["A0"],DATA["A1"]),self.DeconvMachine._ModelImage)
                # # # visData[:,:,:]=visData[:,:,:]-visPredict[:,:,:]
            

                # ModelImage=np.zeros(self.FacetMachine.OutImShape,np.float32)
                # # _,_,nx,_=ModelImage.shape
                # # #ModelImage[0,0].T[::-1,:][1519,508]=-10.
                # # #ModelImage[0,0].T[::-1,:][1519,508]=-10.
                # # ModelImage[0,0].T[::-1,:][1625,557]=-10.
                #ind=np.where(Image==np.max(Image))
                #print ind

                # ind=np.where(ModelImage==np.max(ModelImage))
                # #print ind
                # Max=np.max(ModelImage)
                # ModelImage.fill(0)
                # ModelImage[ind]=Max

                # # #ModelImage[0,0,:,:]=ModelImage[0,0]#[::-1].T
                #d0=DATA["data"].copy()
                #DATA["data"].fill(0)

                # #ind=np.where(ModelImage==np.max(ModelImage))
                # ModelImage.fill(0)
                # ModelImage[0,0,2381,6610]=10.
                # #ModelImage[ind]=-10
                # #ModelImage=-ModelImage

                #DATA["data"].fill(0)

                self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),ModelImage)



                # d1=DATA["data"]
                # A0,A1=DATA["A0"],DATA["A1"]
                # for iFreq in [0]:#range(20):
                #     #ind=np.where((A0==49)&(A1==55))[0]
                #     ind=range(d0.shape[0])
                #     op0=np.abs
                #     op1=np.real
                #     pylab.clf()
                #     pylab.subplot(2,1,1)
                #     pylab.plot(op0(d0[ind,iFreq,0]))
                #     pylab.plot(op0(d1[ind,iFreq,0]))
                #     pylab.plot(op0(d1[ind,iFreq,0])-op0(d0[ind,iFreq,0]))
                #     # pylab.title(iAnt)
                #     pylab.subplot(2,1,2)
                #     pylab.plot(op1(d0[ind,iFreq,0]))
                #     pylab.plot(op1(d1[ind,iFreq,0]))
                #     pylab.plot(op1(d1[ind,iFreq,0])-op1(d0[ind,iFreq,0]))
                #     pylab.draw()
                #     pylab.show(False)

                # stop


                self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),DATA["Weights"],doStack=True)#,Channel=self.VS.CurrentFreqBand)
                
                # NpShared.DelArray(PredictedDataName)
                del(DATA)


            DicoImage=self.FacetMachine.FacetsToIm(NormJones=True)
            #DicoImage["NormData"]=self.NormImage
            # self.DeconvMachine.MSMachine.ModelMachine.ToFile("%s.DicoModel"%self.BaseName)

            self.ResidImage=DicoImage["MeanImage"]
            if "Residual_i" in self.GD["Images"]["SaveIms"]:
                self.FacetMachine.ToCasaImage(DicoImage["MeanImage"],ImageName="%s.residual%2.2i"%(self.BaseName,iMajor),Fits=True)


            if "Model_i" in self.GD["Images"]["SaveIms"]:
                ModelImage=self.DeconvMachine.GiveModelImage(ThisMeanFreq)
                self.FacetMachine.ToCasaImage(ModelImage,ImageName="%s.model%2.2i"%(self.BaseName,iMajor),Fits=True)

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

    def FitPSF(self):
        #PSF=self.PSF
        PSF=self.MeanFacetPSF
        PSF=self.DicoVariablePSF["CubeMeanVariablePSF"][self.FacetMachine.iCentralFacet]

        _,_,x,y=np.where(PSF==np.max(PSF))
        FitOK=False
        off=100
        while FitOK==False:

            # print>>log, "Try fitting PSF in a [%i,%i] box ..."%(off*2,off*2)
            # P=PSF[0,0,x[0]-off:x[0]+off,y[0]-off:y[0]+off]
            # self.SideLobeLevel,self.OffsetSideLobe=ModFitPSF.FindSidelobe(P)
            # sigma_x, sigma_y, theta = ModFitPSF.DoFit(P)
            # FitOK=True
            # print>>log, "   ... done"

            # P=PSF[0,0,:,:]
            # self.SideLobeLevel,self.OffsetSideLobe=ModFitPSF.FindSidelobe(P)
            # sigma_x, sigma_y, theta = ModFitPSF.DoFit(P)
            # FitOK=True
            # print>>log, "   ... done"

            print>>log, "Try fitting PSF in a [%i,%i] box ..."%(off*2,off*2)
            P=PSF[0,0,x[0]-off:x[0]+off,y[0]-off:y[0]+off]
            self.SideLobeLevel,self.OffsetSideLobe=ModFitPSF.FindSidelobe(P)
            sigma_x, sigma_y, theta = ModFitPSF.DoFit(P)
            FitOK=True
            print>>log, "   ... done"

            # try:
            #     print>>log, "Try fitting PSF in a [%i,%i] box ..."%(off*2,off*2)
            #     P=PSF[0,0,x[0]-off:x[0]+off,y[0]-off:y[0]+off]
            #     self.SideLobeLevel,self.OffsetSideLobe=ModFitPSF.FindSidelobe(P)
            #     sigma_x, sigma_y, theta = ModFitPSF.DoFit(P)
            #     FitOK=True
            #     print>>log, "   ... done"
            # except:
            #     print>>log, "   ... failed"
            #     off+=100
                

        theta=np.pi/2-theta
        
        FWHMFact=2.*np.sqrt(2.*np.log(2.))
        bmaj=np.max([sigma_x, sigma_y])*self.CellArcSec*FWHMFact
        bmin=np.min([sigma_x, sigma_y])*self.CellArcSec*FWHMFact
        self.FWHMBeam=(bmaj/3600.,bmin/3600.,theta)
        self.PSFGaussPars = (sigma_x*self.CellSizeRad, sigma_y*self.CellSizeRad, theta)
        print>>log, "Fitted PSF (sigma): (Sx, Sy, Th)=(%f, %f, %f)"%(sigma_x*self.CellArcSec, sigma_y*self.CellArcSec, theta)
        print>>log, "Fitted PSF (FWHM):  (Sx, Sy, Th)=(%f, %f, %f)"%(sigma_x*self.CellArcSec*FWHMFact, sigma_y*self.CellArcSec*FWHMFact, theta)
        print>>log, "Secondary sidelobe at the level of %5.1f at a position of %i from the center"%(self.SideLobeLevel,self.OffsetSideLobe)
            
            
    def Restore(self):
        print>>log, "Create restored image"
        if self.PSFGaussPars==None:
            self.FitPSF()
        self.DeconvMachine.ModelMachine.ToFile(self.DicoModelName)

        RefFreq=self.VS.RefFreq
        ModelMachine=self.DeconvMachine.ModelMachine



        self.FacetMachine.ToCasaImage(self.ResidImage,ImageName="%s.residual"%(self.BaseName),Fits=True)

        # Putting back substracted componants
        if self.GD["DDESolutions"]["RestoreSub"]:
            try:
                ModelMachine.PutBackSubsComps()
            except:
                print>>log, ModColor.Str("Failed Putting back substracted componants")


        # model image
        ModelImage=ModelMachine.GiveModelImage(RefFreq)
        if "Model" in self.GD["Images"]["SaveIms"]:
            self.FacetMachine.ToCasaImage(ModelImage,ImageName="%s.model"%self.BaseName,Fits=True)

        # restored image
        self.RestoredImage=ModFFTW.ConvolveGaussian(ModelImage,CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars])
        self.RestoredImageRes=self.RestoredImage+self.ResidImage
        self.FacetMachine.ToCasaImage(self.RestoredImageRes,ImageName="%s.restored"%self.BaseName,Fits=True,beam=self.FWHMBeam)

        # Alpha image
        if ("Alpha" in self.GD["Images"]["SaveIms"])&(self.VS.MultiFreqMode):
            IndexMap=ModelMachine.GiveSpectralIndexMap(CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars])
            # IndexMap=ModFFTW.ConvolveGaussian(IndexMap,CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars],Normalise=True)
            self.FacetMachine.ToCasaImage(IndexMap,ImageName="%s.alpha"%self.BaseName,Fits=True,beam=self.FWHMBeam)


        # self.RestoredImageRes=self.RestoredImage+self.ResidImage/np.sqrt(self.NormImage)
        # self.FacetMachine.ToCasaImage(self.RestoredImageRes,ImageName="%s.restored.corr"%self.BaseName,Fits=True,beam=self.FWHMBeam)

        # self.FacetMachine.ToCasaImage(self.RestoredImage,ImageName="%s.modelConv"%self.BaseName,Fits=True,beam=self.FWHMBeam)




        
        # pylab.clf()
        # pylab.imshow(self.RestoredImage[0,0],interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

################################################

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
        

