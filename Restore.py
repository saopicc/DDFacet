#!/usr/bin/env python
import optparse
import sys
import pickle
#from DDFacet.Imager.ClassModelMachine import ClassModelMachine
from DDFacet.Imager import ClassCasaImage
from pyrap.images import image
#from DDFacet.Imager.ModModelMachine import GiveModelMachine
from DDFacet.Imager.ModModelMachine import ClassModModelMachine
import numpy as np
from DDFacet.ToolsDir import ModFFTW

from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassRestoreMachine")

def read_options():
    desc="""DDFacet """
    
    opt = optparse.OptionParser(usage='Usage: %prog --Parset=somename.MS <options>',version='%prog version 1.0',description=desc)
    
    group = optparse.OptionGroup(opt, "* Data selection options")
    group.add_option('--BaseImageName',help='')
    group.add_option('--ResidualImage',help='',type="str",default="")
    group.add_option('--BeamPix',type='float',help='',default=5)
    group.add_option('--MaskName',type="str",help='',default=5)
    group.add_option('--NBands',type="int",help='',default=1)
    group.add_option('--CleanNegComp',type="int",help='',default=0)
    group.add_option('--DoAlpha',type="int",help='',default=0)
    group.add_option('--OutName',type="str",help='',default="")
    opt.add_option_group(group)
    
    options, arguments = opt.parse_args()
    f = open("last_param.obj","wb")
    pickle.dump(options,f)
    return options


class ClassRestoreMachine():
    def __init__(self,BaseImageName,BeamPix=5,ResidualImName="",DoAlpha=1,
                 MaskName="",CleanNegComp=False,NBands=1,OutName=""):
        self.DoAlpha=DoAlpha
        self.BaseImageName=BaseImageName
        self.BeamPix=BeamPix
        self.NBands=NBands
        self.OutName=OutName

        FileDicoModel="%s.DicoModel"%BaseImageName

        # ClassModelMachine,DicoModel=GiveModelMachine(FileDicoModel)
        # self.ModelMachine=ClassModelMachine(Gain=0.1)
        # self.ModelMachine.FromDico(DicoModel)

        ModConstructor = ClassModModelMachine()
        MM=ModConstructor.GiveInitialisedMMFromFile(FileDicoModel)


        if MaskName!="":
            self.ModelMachine.CleanMaskedComponants(MaskName)
        if CleanNegComp:
            self.ModelMachine.CleanNegComponants(box=10,sig=2)


        if ResidualImName=="":
            FitsFile="%s.residual.fits"%BaseImageName
        else:
            FitsFile=ResidualImName


        NormImageName="%s.Norm.fits"%BaseImageName

        self.FitsFile=FitsFile
        im=image(FitsFile)

        c=im.coordinates()
        self.radec=c.dict()["direction0"]["crval"]
        CellSizeRad,_=c.dict()["direction0"]["cdelt"]
        self.CellSizeRad=np.abs(CellSizeRad)
        self.Cell=(self.CellSizeRad*180/np.pi)*3600
        self.CellArcSec=self.Cell

        self.ResidualData=im.getdata()
        nchan,npol,_,_=self.ResidualData.shape
        testImage=np.zeros_like(self.ResidualData)

        if ResidualImName!="":
            for ch in range(nchan):
                for pol in range(npol):
                    testImage[ch,pol,:,:]=self.ResidualData[ch,pol,:,:].T[::-1,:]#*1.0003900000000001


        SqrtNormImage=np.zeros_like(self.ResidualData)
        imNorm=image(NormImageName).getdata()
        for ch in range(nchan):
            for pol in range(npol):
                SqrtNormImage[ch,pol,:,:]=np.sqrt(imNorm[ch,pol,:,:].T[::-1,:])

        _,_,nx,_=testImage.shape
        Nr=10000
        indx,indy=np.int64(np.random.rand(Nr)*nx),np.int64(np.random.rand(Nr)*nx)
        self.StdResidual=np.std(testImage[0,0,indx,indy])
        self.Residual=testImage
        self.SqrtNormImage=SqrtNormImage

    def Restore(self):
        print>>log, "Create restored image"


        ModelMachine=self.ModelMachine




        
        FWHMFact=2.*np.sqrt(2.*np.log(2.))
        BeamPix=self.BeamPix/FWHMFact
        sigma_x, sigma_y=BeamPix,BeamPix
        theta=0.
        bmaj=np.max([sigma_x, sigma_y])*self.CellArcSec*FWHMFact
        bmin=np.min([sigma_x, sigma_y])*self.CellArcSec*FWHMFact
        self.FWHMBeam=(bmaj/3600.,bmin/3600.,theta)
        self.PSFGaussPars = (sigma_x*self.CellSizeRad, sigma_y*self.CellSizeRad, theta)

        RefFreq=self.ModelMachine.RefFreq
        df=RefFreq*0.5

        #self.ModelMachine.ListScales[0]["Alpha"]=-0.8

        # model image
        #ModelMachine.GiveModelImage(RefFreq)

        FEdge=np.linspace(RefFreq-df,RefFreq+df,self.NBands+1)
        FCenter=(FEdge[0:-1]+FEdge[1::])/2.
        C=299792458.
        Lambda0=C/FCenter[-1]
        dLambda=1
        if self.NBands>1:
            dLambda=np.abs(C/FCenter[0]-C/FCenter[1])

        ListRestoredIm=[]
        Lambda=[Lambda0+i*dLambda for i in range(self.NBands)]
        ListRestoredImCorr=[]
        ListModelIm=[]
        #print C/np.array(Lambda)
        # restored image
        for l in Lambda:
            freq=C/l
            print>>log,"Get ModelImage... "
            ModelImage=ModelMachine.GiveModelImage(freq)
            ListModelIm.append(ModelImage)
            print>>log,"  ModelImage to apparant flux... "
            ModelImage=ModelImage*self.SqrtNormImage
            print>>log,"Convolve... "
            print>>log,"   MinMax = [%f , %f] @ freq = %f MHz"%(ModelImage.min(),ModelImage.max(),freq/1e6)
            RestoredImage=ModFFTW.ConvolveGaussian(ModelImage,CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars])
            RestoredImageRes=RestoredImage+self.Residual
            ListRestoredIm.append(RestoredImageRes)
            RestoredImageResCorr=RestoredImageRes/self.SqrtNormImage
            ListRestoredImCorr.append(RestoredImageResCorr)

        #print FEdge,FCenter

        print>>log,"Save... "
        _,_,nx,_=RestoredImageRes.shape
        RestoredImageRes=np.array(ListRestoredIm).reshape((self.NBands,1,nx,nx))
        RestoredImageResCorr=np.array(ListRestoredImCorr).reshape((self.NBands,1,nx,nx))

        ModelImage=np.array(ListModelIm).reshape((self.NBands,1,nx,nx))
        


        if self.OutName=="":
            ImageName="%s.restoredNew"%self.BaseImageName
            ImageNameCorr="%s.restoredNew.corr"%self.BaseImageName
            ImageNameModel="%s.model"%self.BaseImageName
        else:
            ImageName=self.OutName
            ImageNameCorr=self.OutName+".corr"

        CasaImage=ClassCasaImage.ClassCasaimage(ImageNameModel,RestoredImageRes.shape,self.Cell,self.radec,Lambda=(Lambda0,dLambda,self.NBands))
        CasaImage.setdata(ModelImage,CorrT=True)
        CasaImage.ToFits()
        CasaImage.setBeam(self.FWHMBeam)
        CasaImage.close()

        CasaImage=ClassCasaImage.ClassCasaimage(ImageName,RestoredImageRes.shape,self.Cell,self.radec,Lambda=(Lambda0,dLambda,self.NBands))
        CasaImage.setdata(RestoredImageRes,CorrT=True)
        CasaImage.ToFits()
        CasaImage.setBeam(self.FWHMBeam)
        CasaImage.close()

        CasaImage=ClassCasaImage.ClassCasaimage(ImageNameCorr,RestoredImageResCorr.shape,self.Cell,self.radec,Lambda=(Lambda0,dLambda,self.NBands))
        CasaImage.setdata(RestoredImageResCorr,CorrT=True)
        CasaImage.ToFits()
        CasaImage.setBeam(self.FWHMBeam)
        CasaImage.close()
        


        # ImageName="%s.modelConv"%self.BaseImageName
        # CasaImage=ClassCasaImage.ClassCasaimage(ImageName,ModelImage.shape,self.Cell,self.radec)
        # CasaImage.setdata(self.RestoredImage,CorrT=True)
        # CasaImage.ToFits()
        # CasaImage.setBeam(self.FWHMBeam)
        # CasaImage.close()


        # Alpha image
        if self.DoAlpha:
            print>>log,"Get Index Map... "
            IndexMap=ModelMachine.GiveSpectralIndexMap(CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars])
            ImageName="%s.alphaNew"%self.BaseImageName
            print>>log,"  Save... "
            CasaImage=ClassCasaImage.ClassCasaimage(ImageName,ModelImage.shape,self.Cell,self.radec)
            CasaImage.setdata(IndexMap,CorrT=True)
            CasaImage.ToFits()
            CasaImage.close()
            print>>log,"  Done. "



def test():
    CRM=ClassRestoreMachine("Resid.2")
    CRM.Restore()


def main(options=None):
    

    if options is None:
        f = open("last_param.obj",'rb')
        options = pickle.load(f)
    

    CRM=ClassRestoreMachine(options.BaseImageName,BeamPix=options.BeamPix,ResidualImName=options.ResidualImage,
                            DoAlpha=options.DoAlpha,
                            NBands=options.NBands,
                            CleanNegComp=options.CleanNegComp,
                            OutName=options.OutName)
    CRM.Restore()



if __name__=="__main__":
    OP=read_options()

    main(OP)
