

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
from IPClusterDir import ClassDistributedVisServer

class ClassImagerDeconv():
    def __init__(self,ParsetFile=None,GD=None,
                 PointingID=0,BaseName="ImageTest2"):
        MDC,GD=ToolsDir.GiveMDC.GiveMDC(ParsetFile=ParsetFile,GD=GD,DoReadData=False)
        self.MDC=MDC
        self.GD=GD
        self.BaseName=BaseName
        self.PointingID=PointingID
        MinorCycleConfig=self.GD.DicoConfig["Facet"]["MinorCycleOptions"]
        self.NMajor=self.GD.DicoConfig["Facet"]["MajorCycleOptions"]["MaxMajorIter"]
        self.DeconvMachine=ClassImageDeconvMachine.ClassImageDeconvMachine(**MinorCycleConfig)
        self.FacetMachine=None
        self.PSF=None
        self.PSFGaussPars = None
        self.VisWeights=None
        self.Precision=self.GD.DicoConfig["Facet"]["Precision"]#"S"
        self.PolMode=self.GD.DicoConfig["Facet"]["PolMode"]
        self.HasCleaned=False
        self.Parallel=self.GD.DicoConfig["Cluster"]["Parallel"]
        self.IM=None

    def Init(self):
        self.InitFacetMachine(self.IM)
        self.VS=ClassDistributedVisServer.ClassDistributedVisServer(self.IM)
        self.VS.CalcWeigths(self.FacetMachine.OutImShape,self.FacetMachine.CellSizeRad)


    def MakePSF(self):
        if self.PSF!=None: return
        print>>log, ModColor.Str("   ====== Making PSF ======")
        FacetMachinePSF=ClassFacetMachine.ClassFacetMachine(self.MDC,self.GD,Precision=self.Precision,PolMode=self.PolMode,Parallel=self.Parallel,DoPSF=True)#,Sols=SimulSols)
        MainFacetOptions=self.GD.DicoConfig["Facet"]["MainFacetOptions"]
        FacetMachinePSF.setInitMachine(self.IM)
        FacetMachinePSF.appendMainField(ImageName="%s.psf"%self.BaseName,**MainFacetOptions)
        FacetMachinePSF.Init()
        self.CellSizeRad=(FacetMachinePSF.Cell/3600.)*np.pi/180
        self.CellArcSec=FacetMachinePSF.Cell

        FacetMachinePSF.ReinitDirty()

        while True:
            DATA=self.VS.GiveNextVisChunk()
            if (DATA==None): break

            FacetMachinePSF.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],DATA["A0A1"],DATA["Weights"],doStack=True)


            # Image=FacetMachinePSF.FacetsToIm()
            # pylab.clf()
            # pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            # break

        self.PSF=FacetMachinePSF.FacetsToIm()
        FacetMachinePSF.ToCasaImage()


        # Image=FacetMachinePSF.FacetsToIm()
        # pylab.clf()
        # pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop

        m0=-1;m1=1
        pylab.clf()
        pylab.imshow(self.PSF[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)

        del(FacetMachinePSF)


    def LoadPSF(self,CasaFilePSF):
        self.CasaPSF=image(CasaFilePSF)
        self.PSF=self.CasaPSF.getdata()
        self.CellArcSec=np.abs(self.CasaPSF.coordinates().dict()["direction0"]["cdelt"][0]*60)
        self.CellSizeRad=(self.CellArcSec/3600.)*np.pi/180

    def InitFacetMachine(self,IM=None):
        if self.FacetMachine!=None:
            return

        
        #print "initFacetMachine deconv0"; self.IM.CI.E.clear()
        self.FacetMachine=ClassFacetMachine.ClassFacetMachine(self.MDC,self.GD,Precision=self.Precision,PolMode=self.PolMode,Parallel=self.Parallel)#,Sols=SimulSols)
        
        if self.IM!=None:
            self.FacetMachine.setInitMachine(IM)
        #print "initFacetMachine deconv1"; self.IM.CI.E.clear()
        MainFacetOptions=self.GD.DicoConfig["Facet"]["MainFacetOptions"]
        self.FacetMachine.appendMainField(ImageName="%s.image"%self.BaseName,**MainFacetOptions)
        self.FacetMachine.Init()
        #print "initFacetMachine deconv2"; self.IM.CI.E.clear()

        self.CellSizeRad=(self.FacetMachine.Cell/3600.)*np.pi/180
        self.CellArcSec=self.FacetMachine.Cell

    def testDegrid(self):
    
        DATA=self.VS.GiveNextVisChunk()
        visPredict=np.zeros_like(DATA["data"])
        Im=np.zeros(self.FacetMachine.OutImShape,dtype=np.complex64)
        _,_,n,n=Im.shape
        Im[0,0,n/4,n/4]=1
        Im[0,0,n/4,n/2]=1.

            #visPredict=self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],visPredict,DATA["flags"],DATA["A0A1"],Im)

            # self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],DATA["A0A1"],DATA["Weights"],doStack=True)
        self.FacetMachine.ClearSharedMemory()
        self.FacetMachine.PutInShared(DATA)
        
        visPredict=np.zeros_like(DATA["data"])
        visPredict=NpShared.ToShared("%s.%s"%(self.FacetMachine.PrefixShared,"predict_data"),visPredict)
        
        _=self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],visPredict,DATA["flags"],DATA["A0A1"],Im)
        
        visResid=NpShared.GiveArray("%s.%s"%(self.FacetMachine.PrefixShared,"data"))
        visResid[:,:,:]=visPredict[:,:,:]#DATA["data"][:,:,:]-visPredict[:,:,:]
        
        self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],visResid,DATA["flags"],DATA["A0A1"],DATA["Weights"],doStack=True)

        Image=self.FacetMachine.FacetsToIm()


        pylab.clf()
        ax0=pylab.subplot(1,2,1)
        ax0.imshow(np.real(Im[0,0]),interpolation="nearest")#,vmin=m0,vmax=m1)
        ax1=pylab.subplot(1,2,2,sharex=ax0,sharey=ax0)
        ax1.imshow(np.real(Image[0,0]),interpolation="nearest")#,vmin=m0,vmax=m1)
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)


        #self.FacetMachine.reset()
        return


    def GiveDirty(self):

        print>>log, ModColor.Str("   ====== Making Dirty ======")
        self.InitFacetMachine(self.IM)
        
        self.FacetMachine.ReinitDirty()

        while True:
            DATA=self.VS.GiveNextVisChunk()
            


            if (DATA==None): break
            
            
            self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],DATA["A0A1"],DATA["Weights"],doStack=True)
            
            # Image=self.FacetMachine.FacetsToIm()
            # pylab.clf()
            # pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
            # pylab.colorbar()
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)

        self.FacetMachine.ToCasaImage(ImageName="%s.dirty"%self.BaseName,Fits=True)
        #m0,m1=Image.min(),Image.max()
        
        Image=self.FacetMachine.FacetsToIm()
        pylab.clf()
        pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)

        return Image

    def main(self,NMajor=None):
        if NMajor==None:
            NMajor=self.NMajor

        self.MakePSF()
        
        Image=self.GiveDirty()

        for iMajor in range(NMajor):

            print>>log, ModColor.Str("   ====== Runing major Cycle %i ======"%iMajor)
            self.DeconvMachine.SetDirtyPSF(Image,self.PSF)
            self.DeconvMachine.Clean()
            self.FacetMachine.ReinitDirty()
            while True:
                DATA=self.VS.GiveNextVisChunk()            
                if (DATA==None): break

                visPredict=np.zeros_like(DATA["data"])
                visPredict=NpShared.ToShared("%s.%s"%(self.VS.PrefixShared,"predict_data"),visPredict)

                ####################
                testImage=np.zeros((1, 1, 1008, 1008),np.complex64)
                testImage[0,0,200,650]=100.
                self.DeconvMachine._ModelImage=testImage
                ####################

                _=self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],visPredict,DATA["flags"],DATA["A0A1"],self.DeconvMachine._ModelImage)
                
                visResid=NpShared.GiveArray("%s.%s"%(self.VS.PrefixShared,"data"))
                visResid[:,:,:]=DATA["data"][:,:,:]-visPredict[:,:,:]
            
                self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],visResid,DATA["flags"],DATA["A0A1"],DATA["Weights"],doStack=True)
                
                NpShared.DelArray("%s.%s"%(self.VS.PrefixShared,"predict_data"))

            Image=self.FacetMachine.FacetsToIm()
            self.ResidImage=Image

            pylab.clf()
            pylab.imshow(Image[0,0],interpolation="nearest")#,vmin=m0,vmax=m1)
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)
            self.HasCleaned=True
        self.FacetMachine.ToCasaImage(ImageName="%s.residual"%self.BaseName,Fits=True)
        if self.HasCleaned:
            self.Restore()

    def FitPSF(self):
        _,_,x,y=np.where(self.PSF==np.max(self.PSF))
        off=100
        PSF=self.PSF[0,0,x[0]-off:x[0]+off,y[0]-off:y[0]+off]
        sigma_x, sigma_y, theta = ModFitPSF.DoFit(PSF)
        theta=np.pi/2-theta
        
        self.PSFGaussPars = (sigma_x*self.CellSizeRad, sigma_y*self.CellSizeRad, theta)
        
        print>>log, "Fitted PSF: (Sx, Sy, Th)=(%f, %f, %f)"%(sigma_x*self.CellArcSec, sigma_y*self.CellArcSec, theta)
            
            
    def Restore(self):
        print>>log, "Create restored image"
        if self.PSFGaussPars==None:
            self.FitPSF()
        
        self.RestoredImage=ModFFTW.ConvolveGaussian(self.DeconvMachine._ModelImage,CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars])
        self.RestoredImage+=self.ResidImage
        self.FacetMachine.ToCasaImage(ImageIn=self.RestoredImage,ImageName="%s.restored"%self.BaseName,Fits=True)
        # pylab.clf()
        # pylab.imshow(self.RestoredImage[0,0],interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

################################################

    def testDegrid(self):
        self.InitFacetMachine(self.IM)
        
        self.FacetMachine.ReinitDirty()
        DATA=self.VS.GiveNextVisChunk()

        # ###########################################
        # self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],DATA["data"],DATA["flags"],DATA["A0A1"],DATA["Weights"],doStack=True)
        # testImage=self.FacetMachine.FacetsToIm()
        # testImage.fill(0)
        # _,_,nx,_=testImage.shape
        # xc=nx/2
        # n=2
        # dn=100
        # #for i in range(-n,n+1):
        # #    for j in range(-n,n+1):
        # #        testImage[0,0,int(xc+i*dn),int(xc+j*dn)]=100.
        # # for i in range(n+1):
        # #     testImage[0,0,int(xc+i*dn),int(xc+i*dn)]=100.
        # #testImage[0,0,200,400]=100.
        # testImage[0,0,xc+200,xc+300]=100.
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

        visPredict=np.zeros_like(DATA["data"])
        visPredict=NpShared.ToShared("%s.%s"%(self.VS.PrefixShared,"predict_data"),visPredict)
        
        _=self.FacetMachine.getChunk(DATA["times"],DATA["uvw"],visPredict,DATA["flags"],DATA["A0A1"],testImage)


        A0,A1=DATA["A0A1"]
        fig=pylab.figure(1)
        os.system("rm -rf png/*.png")
        for iAnt in [0]:#range(36):
            for jAnt in [29]:#range(36):
            
                ind=np.where((A0==iAnt)&(A1==jAnt))[0]
                if ind.size==0: continue
                d0=DATA["data"][ind,0,0]
                if np.max(d0)<1e-6: continue

                d1=visPredict[ind,0,0]
                pylab.clf()
                pylab.plot(d0)
                pylab.plot(d1)
                pylab.plot(d0-d1)
                pylab.plot(np.zeros(d0.size),ls=":",color="black")
                pylab.title("%s"%iAnt)
                pylab.draw()
                fig.savefig("png/resid_%2.2i_%2.2i.png"%(iAnt,jAnt))
                # pylab.show(False)


        visResid=NpShared.GiveArray("%s.%s"%(self.VS.PrefixShared,"data"))
        visResid[:,:,:]=DATA["data"][:,:,:]-visPredict[:,:,:]
        
        self.FacetMachine.putChunk(DATA["times"],DATA["uvw"],visPredict,DATA["flags"],DATA["A0A1"],DATA["Weights"])
        Image=self.FacetMachine.FacetsToIm()
        self.ResidImage=Image
        self.FacetMachine.ToCasaImage(ImageName="test.residual",Fits=True)

        
        

def test():
    Imager=ClassImagerDeconv(ParsetFile="ParsetDDFacet.txt")
    #Imager.MakePSF()
    #Imager.LoadPSF("PSF.image")
    # Imager.FitPSF()
    Imager.main(NMajor=5)
    Imager.Restore()
    
    return Imager
