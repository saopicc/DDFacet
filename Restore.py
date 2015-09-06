#!/usr/bin/env python
import optparse
import sys
import pickle
from DDFacet.Imager.ClassModelMachine import ClassModelMachine
from DDFacet.Imager import ClassCasaImage
from pyrap.images import image
from DDFacet.Imager import ClassCasaImage
import numpy as np
from DDFacet.ToolsDir import ModFFTW

from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassRestoreMachine")

def read_options():
    desc="""DDFacet """
    
    opt = optparse.OptionParser(usage='Usage: %prog --Parset=somename.MS <options>',version='%prog version 1.0',description=desc)
    
    group = optparse.OptionGroup(opt, "* Data selection options")
    group.add_option('--BaseImageName',help='',default=0)
    group.add_option('--BeamPix',help='',default=5)
    opt.add_option_group(group)
    
    options, arguments = opt.parse_args()
    f = open("last_param.obj","wb")
    pickle.dump(options,f)
    return options


class ClassRestoreMachine():
    def __init__(self,BaseImageName,BeamPix=5):
        self.BaseImageName=BaseImageName
        self.ModelMachine=ClassModelMachine(Gain=0.1)
        
        DicoModel="%s.DicoModel"%BaseImageName
        self.ModelMachine.FromFile(DicoModel)
        

        FitsFile="%s.residual.fits"%BaseImageName
        im=image(FitsFile)

        c=im.coordinates()
        self.radec=c.dict()["direction0"]["crval"]
        CellSizeRad,_=c.dict()["direction0"]["cdelt"]
        self.CellSizeRad=np.abs(CellSizeRad)
        self.Cell=(self.CellSizeRad*180/np.pi)*3600
        self.CellArcSec=self.Cell

        testImageIn=im.getdata()
        nchan,npol,_,_=testImageIn.shape
        testImage=np.zeros_like(testImageIn)
        for ch in range(nchan):
            for pol in range(npol):
                testImage[ch,pol,:,:]=testImageIn[ch,pol,:,:].T[::-1,:]#*1.0003900000000001
        self.Residual=testImage


    def Restore(self):
        print>>log, "Create restored image"


        ModelMachine=self.ModelMachine




        

        # model image
        ModelImage=ModelMachine.GiveModelImage()

        
        sigma_x, sigma_y=BeamPix,BeamPix
        theta=0.
        FWHMFact=2.*np.sqrt(2.*np.log(2.))
        bmaj=np.max([sigma_x, sigma_y])*self.CellArcSec*FWHMFact
        bmin=np.min([sigma_x, sigma_y])*self.CellArcSec*FWHMFact
        self.FWHMBeam=(bmaj/3600.,bmin/3600.,theta)
        self.PSFGaussPars = (sigma_x*self.CellSizeRad, sigma_y*self.CellSizeRad, theta)



        # restored image
        self.RestoredImage=ModFFTW.ConvolveGaussian(ModelImage,CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars])
        self.RestoredImageRes=self.RestoredImage+self.Residual

        ImageName="%s.restoredNew"%self.BaseImageName

        CasaImage=ClassCasaImage.ClassCasaimage(ImageName,ModelImage.shape,self.Cell,self.radec)
        CasaImage.setdata(self.RestoredImageRes,CorrT=True)
        CasaImage.ToFits()
        CasaImage.setBeam(self.FWHMBeam)
        CasaImage.close()


        # # Alpha image
        # if "Alpha" in self.GD["VisData"]["SaveIms"]:
        #     IndexMap=ModelMachine.GiveSpectralIndexMap(CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars])
        #     # IndexMap=ModFFTW.ConvolveGaussian(IndexMap,CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars],Normalise=True)
        #     self.FacetMachine.ToCasaImage(IndexMap,ImageName="%s.alpha"%self.BaseName,Fits=True,beam=self.FWHMBeam)


def test():
    CRM=ClassRestoreMachine("Resid.2")
    CRM.Restore()


def main(options=None):
    

    if options==None:
        f = open("last_param.obj",'rb')
        options = pickle.load(f)
    
    CRM=ClassRestoreMachine(options.BaseImageName,BeamPix=options.BeamPix)
    CRM.Restore()
