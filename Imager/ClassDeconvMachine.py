

import ClassFacetMachine
import numpy as np
import pylab
import ToolsDir
import MyPickle
from pyrap.images import image
import ClassImageDeconvMachine
import ModFFTW
import MyLogger
import ModColor
log=MyLogger.getLogger("ClassImagerDeconv")
import NpShared
import os
import ModFitPSF
from ClassData import ClassMultiPointingData,ClassSinglePointingData,ClassGlobalData
import ClassVisServer
import time

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
                 PointingID=0,BaseName="ImageTest2",ReplaceDico=None,IdSharedMem="CACA."):
        if ParsetFile!=None:
            GD=ClassGlobalData(ParsetFile)
            self.GD=GD
            
        if GD!=None:
            self.GD=GD

        self.BaseName=BaseName
        self.PointingID=PointingID
        self.NMajor=self.GD["ImagerDeconv"]["MaxMajorIter"]
        del(self.GD["ImagerDeconv"]["MaxMajorIter"])
        MinorCycleConfig=dict(self.GD["ImagerDeconv"])
        MinorCycleConfig["NCPU"]=self.GD["Parallel"]["NCPU"]
        self.DeconvMachine=ClassImageDeconvMachine.ClassImageDeconvMachine(**MinorCycleConfig)
        self.FacetMachine=None
        self.PSF=None
        self.PSFGaussPars = None
        self.VisWeights=None
        self.DATA=None
        self.Precision=self.GD["ImagerGlobal"]["Precision"]#"S"
        self.PolMode=self.GD["ImagerGlobal"]["PolMode"]
        self.HasCleaned=False
        self.Parallel=self.GD["Parallel"]["Enable"]
        self.IdSharedMem=IdSharedMem
        self.PNGDir="%s.png"%self.BaseName
        os.system("mkdir -p %s"%self.PNGDir)
        os.system("rm %s/*.png 2> /dev/null"%self.PNGDir)
        

    def Init(self):
        DC=self.GD

        
        MSName=DC["VisData"]["MSName"]
        if DC["VisData"]["MSListFile"]!="":
            f=open(DC["VisData"]["MSListFile"])
            Ls=f.readlines()
            f.close()
            MSName=[]
            for l in Ls:
                ll=l.replace("\n","")
                MSName.append(ll)
            

        self.VS=ClassVisServer.ClassVisServer(MSName,
                                              ColName=DC["VisData"]["ColName"],
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



        self.InitFacetMachine()
        #self.VS.SetImagingPars(self.FacetMachine.OutImShape,self.FacetMachine.CellSizeRad)
        #self.VS.CalcWeigths(self.FacetMachine.OutImShape,self.FacetMachine.CellSizeRad)
        self.VS.setFOV(self.FacetMachine.OutImShape,self.FacetMachine.PaddedGridShape,self.FacetMachine.FacetShape,self.FacetMachine.CellSizeRad)
        self.VS.CalcWeigths()




    def InitFacetMachine(self):
        if self.FacetMachine!=None:
            return

        
        #print "initFacetMachine deconv0"; self.IM.CI.E.clear()
        ApplyCal=False
        SolsFile=self.GD["DDESolutions"]["DDSols"]
        if SolsFile!="": ApplyCal=True

        self.FacetMachine=ClassFacetMachine.ClassFacetMachine(self.VS,self.GD,Precision=self.Precision,PolMode=self.PolMode,Parallel=self.Parallel,
                                                              IdSharedMem=self.IdSharedMem,ApplyCal=ApplyCal)#,Sols=SimulSols)

        
        #print "initFacetMachine deconv1"; self.IM.CI.E.clear()
        MainFacetOptions=self.GiveMainFacetOptions()
        self.FacetMachine.appendMainField(ImageName="%s.image"%self.BaseName,**MainFacetOptions)
        self.FacetMachine.Init()
        #print "initFacetMachine deconv2"; self.IM.CI.E.clear()

        self.CellSizeRad=(self.FacetMachine.Cell/3600.)*np.pi/180
        self.CellArcSec=self.FacetMachine.Cell

    def setNextData(self):
        #del(self.DATA)
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
        print>>log, ModColor.Str("=============================== Making PSF ===============================")
        # FacetMachinePSF=ClassFacetMachine.ClassFacetMachine(self.VS,self.GD,Precision=self.Precision,PolMode=self.PolMode,Parallel=self.Parallel,
        #                                                     IdSharedMem=self.IdSharedMem,DoPSF=True)#,Sols=SimulSols)

        FacetMachinePSF=self.FacetMachine

        # MainFacetOptions=self.GiveMainFacetOptions()
        # FacetMachinePSF.appendMainField(ImageName="%s.psf"%self.BaseName,**MainFacetOptions)
        # FacetMachinePSF.Init()
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

            FacetMachinePSF.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),DATA["Weights"],doStack=True)


            # Image=FacetMachinePSF.FacetsToIm()
            # pylab.clf()
            # pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            # break

        self.PSF=FacetMachinePSF.FacetsToIm().copy()

        FacetMachinePSF.DoPSF=False
        
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

        self.FitPSF()
        #self.FWHMBeam=(10.,10.,10.)
        #FacetMachinePSF.ToCasaImage(self.PSF)
        FacetMachinePSF.ToCasaImage(self.PSF,ImageName="%s.psf"%self.BaseName,Fits=True,beam=self.FWHMBeam)

        #self.FitPSF()
        #FacetMachinePSF.ToCasaImage(self.PSF,Fits=True)


        
        #del(FacetMachinePSF)


    def LoadPSF(self,CasaFilePSF):
        self.CasaPSF=image(CasaFilePSF)
        self.PSF=self.CasaPSF.getdata()
        self.CellArcSec=np.abs(self.CasaPSF.coordinates().dict()["direction0"]["cdelt"][0]*60)
        self.CellSizeRad=(self.CellArcSec/3600.)*np.pi/180




    def GiveDirty(self):

        print>>log, ModColor.Str("============================== Making Dirty ==============================")
        self.InitFacetMachine()
        
        self.FacetMachine.ReinitDirty()

        while True:
            Res=self.setNextData()
            #if Res=="EndChunk": break
            if Res=="EndOfObservation": break
            DATA=self.DATA
            
            self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),DATA["Weights"],doStack=True)
            
            # Image=self.FacetMachine.FacetsToIm()
            # pylab.clf()
            # pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
            # pylab.colorbar()
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)

        Image=self.FacetMachine.FacetsToIm()
        self.FacetMachine.ToCasaImage(Image,ImageName="%s.dirty"%self.BaseName,Fits=True)
        #m0,m1=Image.min(),Image.max()
        
        # pylab.clf()
        # pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        return Image

    def GivePredict(self,ModelImage):

        print>>log, ModColor.Str("============================== Making Dirty ==============================")
        self.InitFacetMachine()
        
        self.FacetMachine.ReinitDirty()

        while True:
            Res=self.setNextData()
            if Res=="EndOfObservation": break
            DATA=self.DATA
            
            vis=self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),ModelImage)


        return Image


    def main(self,NMajor=None):
        if NMajor==None:
            NMajor=self.NMajor

        self.MakePSF()
        
        Image=self.GiveDirty()

        for iMajor in range(NMajor):

            print>>log, ModColor.Str("========================== Runing major Cycle %i ========================="%iMajor)
            self.DeconvMachine.SetDirtyPSF(Image,self.PSF)
            self.DeconvMachine.setSideLobeLevel(self.SideLobeLevel)
            repMinor=self.DeconvMachine.Clean()
            if repMinor=="DoneMinFlux":
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
                
                visData=DATA["data"]


                # ####################
                # testImage=np.zeros((1, 1, 1008, 1008),np.complex64)
                # testImage[0,0,200,650]=100.
                # self.DeconvMachine._ModelImage=testImage
                # ####################

                # PredictedDataName="%s%s"%(self.IdSharedMem,"predicted_data")
                # visPredict=NpShared.zeros(PredictedDataName,visData.shape,visData.dtype)
                # _=self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],visPredict,DATA["flags"],(DATA["A0"],DATA["A1"]),self.DeconvMachine._ModelImage)
                # visData[:,:,:]=visData[:,:,:]-visPredict[:,:,:]
            
                _=self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],(DATA["A0"],DATA["A1"]),self.DeconvMachine._ModelImage)
            
                self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],visData,DATA["flags"],(DATA["A0"],DATA["A1"]),DATA["Weights"],doStack=True)
                
                # NpShared.DelArray(PredictedDataName)

            Image=self.FacetMachine.FacetsToIm()
            self.ResidImage=Image
            self.FacetMachine.ToCasaImage(Image,ImageName="%s.residual"%self.BaseName,Fits=True)
            # fig=pylab.figure(1)
            # pylab.clf()
            # pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
            # pylab.colorbar()
            # pylab.draw()
            # PNGName="%s/Residual%3.3i.png"%(self.PNGDir,iMajor)
            # fig.savefig(PNGName)
            # pylab.show(False)
            # pylab.pause(0.1)

            self.HasCleaned=True
            if repMinor=="MaxIter": break

        #self.FacetMachine.ToCasaImage(Image,ImageName="%s.residual"%self.BaseName,Fits=True)
        if self.HasCleaned:
            self.Restore()

    def FitPSF(self):
        _,_,x,y=np.where(self.PSF==np.max(self.PSF))
        FitOK=False
        off=100
        while FitOK==False:
            try:
                print>>log, "Try fitting PSF in a [%i,%i] box ..."%(off*2,off*2)
                PSF=self.PSF[0,0,x[0]-off:x[0]+off,y[0]-off:y[0]+off]
                self.SideLobeLevel=ModFitPSF.FindSidelobe(PSF)
                sigma_x, sigma_y, theta = ModFitPSF.DoFit(PSF)
                FitOK=True
                print>>log, "   ... done"
            except:
                print>>log, "   ... failed"
                off+=100
                

        theta=np.pi/2-theta
        
        FWHMFact=2.*np.sqrt(2.*np.log(2.))
        bmaj=np.max([sigma_x, sigma_y])*self.CellArcSec*FWHMFact
        bmin=np.min([sigma_x, sigma_y])*self.CellArcSec*FWHMFact
        self.FWHMBeam=(bmaj/3600.,bmin/3600.,theta)
        self.PSFGaussPars = (sigma_x*self.CellSizeRad, sigma_y*self.CellSizeRad, theta)
        print>>log, "Fitted PSF (sigma): (Sx, Sy, Th)=(%f, %f, %f)"%(sigma_x*self.CellArcSec, sigma_y*self.CellArcSec, theta)
        print>>log, "Fitted PSF (FWHM):  (Sx, Sy, Th)=(%f, %f, %f)"%(sigma_x*self.CellArcSec*FWHMFact, sigma_y*self.CellArcSec*FWHMFact, theta)
        print>>log, "Secondary sidelobe at the level of %5.1f"%(self.SideLobeLevel)
            
            
    def Restore(self):
        print>>log, "Create restored image"
        if self.PSFGaussPars==None:
            self.FitPSF()
        
        self.RestoredImage=ModFFTW.ConvolveGaussian(self.DeconvMachine._ModelImage,CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars])
        self.RestoredImageRes=self.RestoredImage+self.ResidImage
        self.FacetMachine.ToCasaImage(self.RestoredImageRes,ImageName="%s.restored"%self.BaseName,Fits=True,beam=self.FWHMBeam)

        self.FacetMachine.ToCasaImage(self.DeconvMachine._ModelImage,ImageName="%s.model"%self.BaseName,Fits=True)
        self.FacetMachine.ToCasaImage(self.RestoredImage,ImageName="%s.modelConv"%self.BaseName,Fits=True,beam=self.FWHMBeam)




        
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

        im=image("testImage")
        testImageIn=im.getdata()
        nchan,npol,_,_=testImageIn.shape
        testImage=np.zeros_like(testImageIn)
        for ch in range(nchan):
            for pol in range(npol):
                testImage[ch,pol,:,:]=testImageIn[ch,pol,:,:].T[::-1,:]#*1.0003900000000001

        visData=DATA["data"]
        PredictedDataName="%s%s"%(self.IdSharedMem,"predicted_data")
        visPredict=NpShared.zeros(PredictedDataName,visData.shape,visData.dtype)
        
        _=self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],visPredict,DATA["flags"],(DATA["A0"],DATA["A1"]),testImage)


        A0,A1=DATA["A0"],DATA["A1"]
        fig=pylab.figure(1)
        os.system("rm -rf png/*.png")
        op0=np.real
        op1=np.angle
        for iAnt in [32]:#range(36)[::-1]:
            for jAnt in [33]:#range(36)[::-1]:
            
                ind=np.where((A0==iAnt)&(A1==jAnt))[0]
                if ind.size==0: continue
                d0=DATA["data"][ind,0,0]
                u,v,w=DATA["uvw"][ind].T
                if np.max(d0)<1e-6: continue

                d1=visPredict[ind,0,0]
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


        visData[:,:,:]=visData[:,:,:]-visPredict[:,:,:]
        
        self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],visData,DATA["flags"],(DATA["A0"],DATA["A1"]),DATA["Weights"])
        Image=self.FacetMachine.FacetsToIm()
        self.ResidImage=Image
        self.FacetMachine.ToCasaImage(ImageName="test.residual",Fits=True)

        pylab.figure(2)
        pylab.clf()
        pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
        pylab.colorbar()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)

        
        

