#!/usr/bin/env python
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
import optparse
import pickle
from pyrap.images import image
import numpy as np
from DDFacet.Imager import ClassCasaImage
#from DDFacet.Imager.ModModelMachine import GiveModelMachine
from DDFacet.Imager.ModModelMachine import ClassModModelMachine
from DDFacet.Other import MyLogger
from DDFacet.ToolsDir import ModFFTW

from DDFacet.Other import MyLogger
from DDFacet.Other import MyPickle
log=MyLogger.getLogger("ClassRestoreMachine")

import multiprocessing
NCPU_default=str(int(0.75*multiprocessing.cpu_count()))

def read_options():
    desc="""DDFacet """
    
    opt = optparse.OptionParser(usage='Usage: %prog <options>',version='%prog version 1.0',description=desc)
    
    group = optparse.OptionGroup(opt, "* Data selection options")
    group.add_option('--BaseImageName',help='')
    group.add_option('--ResidualImage',help='',type="str",default="")
    group.add_option('--BeamPix',type='float',help='',default=5)
    group.add_option('--SmoothMode',type='int',help='0 = use default beam, 1 = use smooth beam, default 0',default=0)
    group.add_option('--MakeCorrected',type='int',help='0 = no normalization correction, 1 = make corrected image, default 1',default=1)
    group.add_option('--MaskName',type="str",help='',default=5)
    group.add_option('--NBands',type="int",help='',default=1)
    group.add_option('--CleanNegComp',type="int",help='',default=0)
    group.add_option('--DoAlpha',type="int",help='',default=0)
    group.add_option('--OutName',type="str",help='',default="")
    group.add_option('--PSFCache',type="str",help='',default="")
    group.add_option('--NCPU',type="int",help='',default=NCPU_default)
    opt.add_option_group(group)
    
    options, arguments = opt.parse_args()
    f = open("last_param.obj","wb")
    pickle.dump(options,f)
    return options


class ClassRestoreMachine():
    def __init__(self,BaseImageName,BeamPix=5,ResidualImName="",DoAlpha=1,
                 MaskName="",CleanNegComp=False,NBands=1,OutName="",
                 SmoothMode=0,MakeCorrected=1,options=None):
        self.DoAlpha=DoAlpha
        self.BaseImageName=BaseImageName
        self.BeamPix=BeamPix
        self.NBands=NBands
        self.OutName=OutName
        self.options=options
        self.SmoothMode=SmoothMode
        self.MakeCorrected=MakeCorrected

        FileDicoModel="%s.DicoModel"%BaseImageName

        # ClassModelMachine,DicoModel=GiveModelMachine(FileDicoModel)
        # self.ModelMachine=ClassModelMachine(Gain=0.1)
        # self.ModelMachine.FromDico(DicoModel)

        ModConstructor = ClassModModelMachine()
        self.ModelMachine=ModConstructor.GiveInitialisedMMFromFile(FileDicoModel)


        if MaskName!="":
            self.ModelMachine.CleanMaskedComponants(MaskName)
        if CleanNegComp:
            self.ModelMachine.CleanNegComponants(box=10,sig=2)


        if ResidualImName=="":
            #if "App" in self.ModeNorm:
            #    FitsFile="%s.app.residual.fits"%BaseImageName
            #else:
            #    FitsFile="%s.int.residual.fits"%BaseImageName
            ResidualImName=FitsFile="%s.app.residual.fits"%BaseImageName
        else:
            ResidualImName=FitsFile=ResidualImName
        if self.MakeCorrected:
            if self.SmoothMode:
                NormImageName="%s.SmoothNorm.fits"%BaseImageName
            else:
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

            
        if self.MakeCorrected:
            SqrtNormImage=np.zeros_like(self.ResidualData)
            imNorm=image(NormImageName).getdata()
            for ch in range(nchan):
                for pol in range(npol):
                    SqrtNormImage[ch,pol,:,:]=np.sqrt(imNorm[ch,pol,:,:].T[::-1,:])
        else:
            SqrtNormImage=np.ones_like(self.ResidualData)

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

        # ################################"


        if self.options.PSFCache!="":

            import os
            IdSharedMem=str(int(os.getpid()))+"."
            MeanModelImage=ModelMachine.GiveModelImage(RefFreq)

            # #imNorm=image("6SBc.KAFCA.restoredNew.fits.6SBc.KAFCA.restoredNew.fits.MaskLarge.fits").getdata()
            # imNorm=image("6SB.KAFCA.GA.BIC_00.AP.dirty.fits.mask.fits").getdata()
            # MASK=np.zeros_like(imNorm)
            # nchan,npol,_,_=MASK.shape
            # for ch in range(nchan):
            #     for pol in range(npol):
            #         MASK[ch,pol,:,:]=imNorm[ch,pol,:,:].T[::-1,:]
            # MeanModelImage[MASK==0]=0

            # MeanModelImage.fill(0)
            # MeanModelImage[0,0,100,100]=1


            from DDFacet.Imager.GA import ClassSmearSM
            from DDFacet.Imager import ClassPSFServer
            self.DicoVariablePSF = MyPickle.FileToDicoNP(self.options.PSFCache)
            
            self.PSFServer=ClassPSFServer.ClassPSFServer()
            
            self.PSFServer.setDicoVariablePSF(self.DicoVariablePSF,NormalisePSF=True)
            #return self.Residual,MeanModelImage,self.PSFServer


            # CasaImage=ClassCasaImage.ClassCasaimage("Model.fits",MeanModelImage.shape,self.Cell,self.radec)#Lambda=(Lambda0,dLambda,self.NBands))
            # CasaImage.setdata(MeanModelImage,CorrT=True)
            # CasaImage.ToFits()
            # #CasaImage.setBeam((SmoothFWHM,SmoothFWHM,0))
            # CasaImage.close()


            SmearMachine=ClassSmearSM.ClassSmearSM(self.Residual,
                                                   MeanModelImage*self.SqrtNormImage,
                                                   self.PSFServer,
                                                   DeltaChi2=4.,
                                                   IdSharedMem=IdSharedMem,
                                                   NCPU=self.options.NCPU)
            SmearedModel=SmearMachine.Smear()
            SmoothFWHM=self.CellArcSec*SmearMachine.RestoreFWHM/3600.
            ModelSmearImage="%s.RestoredSmear"%self.BaseImageName
            CasaImage=ClassCasaImage.ClassCasaimage(ModelSmearImage,SmearedModel.shape,self.Cell,self.radec)#Lambda=(Lambda0,dLambda,self.NBands))
            CasaImage.setdata(SmearedModel+self.Residual,CorrT=True)
            #CasaImage.setdata(SmearedModel,CorrT=True)
            CasaImage.ToFits()
            CasaImage.setBeam((SmoothFWHM,SmoothFWHM,0))
            CasaImage.close()
            SmearMachine.CleanUpSHM()
            stop

        # ################################"
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
            print>>log,"  ModelImage to apparent flux... "
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

        CasaImage=ClassCasaImage.ClassCasaimage(ImageNameModel,RestoredImageRes.shape,self.Cell,self.radec)#Lambda=(Lambda0,dLambda,self.NBands))
        CasaImage.setdata(ModelImage,CorrT=True)
        CasaImage.ToFits()
        CasaImage.setBeam(self.FWHMBeam)
        CasaImage.close()

        CasaImage=ClassCasaImage.ClassCasaimage(ImageName,RestoredImageRes.shape,self.Cell,self.radec)#,Lambda=(Lambda0,dLambda,self.NBands))
        CasaImage.setdata(RestoredImageRes,CorrT=True)
        CasaImage.ToFits()
        CasaImage.setBeam(self.FWHMBeam)
        CasaImage.close()

        if self.MakeCorrected:
            CasaImage=ClassCasaImage.ClassCasaimage(ImageNameCorr,RestoredImageResCorr.shape,self.Cell,self.radec)#,Lambda=(Lambda0,dLambda,self.NBands))
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
                            OutName=options.OutName,
                            SmoothMode=options.SmoothMode,
                            MakeCorrected=options.MakeCorrected,
                            options=options)
    return CRM.Restore()



if __name__=="__main__":
    OP=read_options()

    main(OP)
