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

from DDFacet.Other import AsyncProcessPool
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
from DDFacet.Other import Multiprocessing

from DDFacet.Other import MyLogger
from DDFacet.Other import MyPickle
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms
log=MyLogger.getLogger("ClassRestoreMachine")
import scipy.signal
import scipy.stats
import multiprocessing
NCPU_default=str(int(0.75*multiprocessing.cpu_count()))

            

def read_options():
    desc="""DDFacet """
    
    opt = optparse.OptionParser(usage='Usage: %prog <options>',version='%prog version 1.0',description=desc)
    
    group = optparse.OptionGroup(opt, "* Data selection options")
    group.add_option('--BaseImageName',help='')
    group.add_option('--ResidualImage',help='',type="str",default="")
    group.add_option('--BeamPix',type='float',help='',default=5.)
    group.add_option('--SmoothMode',type='int',help='0 = use default beam, 1 = use smooth beam, default 0',default=0)
    group.add_option('--MakeCorrected',type='int',help='0 = no normalization correction, 1 = make corrected image, default 1',default=1)
    group.add_option('--MaskName',type="str",help='',default=5)
    group.add_option('--NBands',type="int",help='',default=1)
    group.add_option('--CleanNegComp',type="int",help='',default=0)
    group.add_option('--Mode',type="str",help='',default="App")
    group.add_option('--RandomCat',type="int",help='',default=0)
    group.add_option('--RandomCat_TotalToPeak',type=float,help='',default=1.)
    group.add_option('--RandomCat_CountsFile',type=str,help='',default=None)
    group.add_option('--ZeroNegComp',type="int",help='',default=0)
    group.add_option('--DoAlpha',type="int",help='',default=0)
    group.add_option('--OutName',type="str",help='',default="")
    group.add_option('--PSFCache',type="str",help='',default="")
    group.add_option('--NCPU',type="int",help='',default=NCPU_default)
    group.add_option('--AddNoise',type=float,help='',default=0)
    opt.add_option_group(group)
    
    options, arguments = opt.parse_args()
    f = open("last_param.obj","wb")
    pickle.dump(options,f)
    return options






class ClassRestoreMachine():
    def __init__(self,BaseImageName,BeamPix=5,ResidualImName="",DoAlpha=1,
                 MaskName="",CleanNegComp=False,
                 NBands=1,
                 SmoothMode=0,MakeCorrected=1,options=None):
        self.DoAlpha=DoAlpha
        self.BaseImageName=BaseImageName
        self.BeamPix=BeamPix
        self.NBands=NBands
        self.OutName=options.OutName
        self.options=options
        self.SmoothMode=SmoothMode
        self.MakeCorrected=MakeCorrected
        self.header_dict={}
        FileDicoModel="%s.DicoModel"%BaseImageName

        # ClassModelMachine,DicoModel=GiveModelMachine(FileDicoModel)
        # self.ModelMachine=ClassModelMachine(Gain=0.1)
        # self.ModelMachine.FromDico(DicoModel)

        print>>log,"Building model machine"
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
                NormImageName="%s.MeanSmoothNorm.fits"%BaseImageName
            else:
                NormImageName="%s.Norm.fits"%BaseImageName
            

        print>>log,"Reading residual image"
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

        print>>log,"Transposing residual..."
        if ResidualImName!="":
            for ch in range(nchan):
                for pol in range(npol):
                    testImage[ch,pol,:,:]=self.ResidualData[ch,pol,:,:].T[::-1,:]#*1.0003900000000001

            
        if self.MakeCorrected:
            print>>log,"Reading beam..."
            SqrtNormImage=np.zeros_like(self.ResidualData)
            imNorm=image(NormImageName).getdata()
            print>>log,"Transposing beam..."
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

    def killWorkers(self):
        print>>log, "Killing workers"
        APP.terminate()
        APP.shutdown()
        Multiprocessing.cleanupShm()

    def Restore(self):
        print>>log, "Create restored image"





        
        FWHMFact=2.*np.sqrt(2.*np.log(2.))
        FWHMFact=2.*np.sqrt(2.*np.log(2.))

        BeamPix=self.BeamPix/FWHMFact
        sigma_x, sigma_y=BeamPix,BeamPix
        theta=0.
        bmaj=np.max([sigma_x, sigma_y])*self.CellArcSec*FWHMFact
        bmin=np.min([sigma_x, sigma_y])*self.CellArcSec*FWHMFact

        #bmaj=bmin=0.001666666666666667*3600
        #sigma_x=
        self.FWHMBeam=(bmaj/3600./np.sqrt(2.),bmin/3600./np.sqrt(2.),theta)
        self.PSFGaussPars = (sigma_x*self.CellSizeRad, sigma_y*self.CellSizeRad, theta)

        #print "!!!!!!!!!!!!!!!!!!!!"
        #self.PSFGaussPars = (BeamPix,BeamPix,0)



        RefFreq=self.ModelMachine.RefFreq
        df=RefFreq*0.5

        # ################################"


        if self.options.PSFCache!="":

            import os
            IdSharedMem=str(int(os.getpid()))+"."
            MeanModelImage=self.ModelMachine.GiveModelImage(RefFreq)

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
            CasaImage.setBeam((SmoothFWHM,SmoothFWHM,0))
            CasaImage.ToFits()
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
            
            if self.options.RandomCat:
                print>>log,"Create random catalog... "
                ModelImage=self.GiveRandomModelIm()
            else:
                print>>log,"Get ModelImage... "
                ModelImage=self.ModelMachine.GiveModelImage(freq)
            
            if self.options.ZeroNegComp:
                print>>log,"Zeroing negative componants... "
                ModelImage[ModelImage<0]=0
            ListModelIm.append(ModelImage)


            if self.options.Mode=="App":
                print>>log,"  ModelImage to apparent flux... "
                ModelImage=ModelImage*self.SqrtNormImage
            print>>log,"Convolve... "
            print>>log,"   MinMax = [%f , %f] @ freq = %f MHz"%(ModelImage.min(),ModelImage.max(),freq/1e6)
            #RestoredImage=ModFFTW.ConvolveGaussianScipy(ModelImage,CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars])

            
            if self.options.AddNoise>0.:
                print>>log,"Adding Noise... "
                ModelImage+=np.random.randn(*ModelImage.shape)*self.options.AddNoise

            RestoredImage,_=ModFFTW.ConvolveGaussianWrapper(ModelImage,Sig=BeamPix)
# =======
#             #RestoredImage,_=ModFFTW.ConvolveGaussianWrapper(ModelImage,Sig=BeamPix)

#             def GiveGauss(Sig0,Sig1):
#                 npix=20*int(np.sqrt(Sig0**2+Sig1**2))
#                 if not npix%2: npix+=1
#                 dx=npix/2
#                 x,y=np.mgrid[-dx:dx:npix*1j,-dx:dx:npix*1j]
#                 dsq=x**2+y**2
#                 return Sig0**2/(Sig0**2+Sig1**2)*np.exp(-dsq/(2.*(Sig0**2+Sig1**2)))
#             R2=np.zeros_like(ModelImage)

#             Sig0=BeamPix/np.sqrt(2.)
#             if self.options.RandomCat:
#                 Sig1=(self.options.RandomCat_SigFactor-1.)*Sig0
#             else:
#                 Sig1=0.
#             nch,npol,_,_=ModelImage.shape
#             for ch in range(nch):
#                 in1=ModelImage[ch,0]
#                 R2[ch,0,:,:]=scipy.signal.fftconvolve(in1,GiveGauss(Sig0,Sig1), mode='same').real
#             RestoredImage=R2

#             self.header_dict["GSIGMA"]=Sig0


#             # print np.max(np.abs(R2-RestoredImage))
#             # import pylab
#             # ax=pylab.subplot(1,3,1)
#             # pylab.imshow(RestoredImage[0,0],interpolation="nearest")
#             # pylab.colorbar()
#             # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
#             # pylab.imshow(R2[0,0],interpolation="nearest")
#             # pylab.colorbar()
#             # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
#             # pylab.imshow((RestoredImage-R2)[0,0],interpolation="nearest")
#             # pylab.colorbar()
#             # pylab.show()
#             # stop

# >>>>>>> 0457182a873da89a2758f4be8a18f55cefd88e44

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
            ImageNameModelConv="%s.modelConv"%self.BaseImageName
        else:
            ImageName=self.OutName
            ImageNameCorr=self.OutName+".corr"
            ImageNameModel="%s.model"%self.OutName
            ImageNameModelConv="%s.modelConv"%self.OutName

        CasaImage=ClassCasaImage.ClassCasaimage(ImageNameModel,RestoredImageRes.shape,self.Cell,self.radec,header_dict=self.header_dict)#Lambda=(Lambda0,dLambda,self.NBands))
        CasaImage.setdata(ModelImage,CorrT=True)
        CasaImage.setBeam(self.FWHMBeam)
        CasaImage.ToFits()
        CasaImage.close()

        CasaImage=ClassCasaImage.ClassCasaimage(ImageName,RestoredImageRes.shape,self.Cell,self.radec,Freqs=C/np.array(Lambda).ravel(),header_dict=self.header_dict)#,Lambda=(Lambda0,dLambda,self.NBands))
        CasaImage.setdata(RestoredImageRes,CorrT=True)
        CasaImage.setBeam(self.FWHMBeam)
        CasaImage.ToFits()
        CasaImage.close()
        
        CasaImage=ClassCasaImage.ClassCasaimage(ImageNameModelConv,RestoredImage.shape,self.Cell,self.radec,Freqs=C/np.array(Lambda).ravel(),header_dict=self.header_dict)#,Lambda=(Lambda0,dLambda,self.NBands))
        CasaImage.setdata(RestoredImage,CorrT=True)
        CasaImage.setBeam(self.FWHMBeam)
        CasaImage.ToFits()
        CasaImage.close()

        if self.MakeCorrected:
            CasaImage=ClassCasaImage.ClassCasaimage(ImageNameCorr,RestoredImageResCorr.shape,self.Cell,self.radec,Freqs=C/np.array(Lambda).ravel(),header_dict=self.header_dict)#,Lambda=(Lambda0,dLambda,self.NBands))
            CasaImage.setdata(RestoredImageResCorr,CorrT=True)
            CasaImage.setBeam(self.FWHMBeam)
            CasaImage.ToFits()
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
            IndexMap=self.ModelMachine.GiveSpectralIndexMap(CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars])
            ImageName="%s.alphaNew"%self.BaseImageName
            print>>log,"  Save... "
            CasaImage=ClassCasaImage.ClassCasaimage(ImageName,ModelImage.shape,self.Cell,self.radec)
            CasaImage.setdata(IndexMap,CorrT=True)
            CasaImage.ToFits()
            CasaImage.close()
            print>>log,"  Done. "

    def GiveRandomModelIm(self):
#        np.random.seed(0)
        SModel,NModel=np.load(self.options.RandomCat_CountsFile).T
        ind=np.argsort(SModel)
        SModel=SModel[ind]
        NModel=NModel[ind]
        #SModel/=1e3
        NModel/=SModel**(5/2.)
        def GiveNPerOmega(s):
            xp=np.interp(np.log10(s), np.log10(SModel), np.log10(NModel), left=None, right=None)
            # import pylab
            # pylab.clf()
            # pylab.plot(np.log10(SModel), np.log10(NModel))
            # pylab.scatter(np.log10(s),xp)
            # pylab.draw()
            # pylab.show()
            # pylab.pause(0.1)
            return 10**xp

        std=np.std(self.Residual.flat[np.int64(np.random.rand(1000)*self.Residual.size)])
        nbin=10000
        smin=2.*std
        smax=10.
        LogS=np.linspace(np.log10(smin),np.log10(smax),nbin)

        Model=np.zeros_like(self.Residual)
        Omega=self.Residual.size*self.CellSizeRad**2
        nx=Model.shape[-1]
        im=image(self.FitsFile)
        Lra=[]
        Ldec=[]
        LS=[]
        SRA=[]
        SDEC=[]
        
        f,p,_,_=im.toworld((0,0,0,0))
        for iBin in range(nbin-1):
            ThisS=(10**LogS[iBin]+10**LogS[iBin+1])/2.
            dx=10**LogS[iBin+1]-10**LogS[iBin]
            n=int(scipy.stats.poisson.rvs(GiveNPerOmega(ThisS)*Omega*dx))#int(round(GiveNPerOmega(ThisS)*Omega*dx))
            indx=np.array([np.int64(np.random.rand(n)*nx)]).ravel()
            indy=np.array([np.int64(np.random.rand(n)*nx)]).ravel()
            s0,s1=10**LogS[iBin],10**LogS[iBin+1]
            RandS=np.random.rand(n)*(s1-s0)+s0
            Model[0,0,indy,indx]=RandS
            for iS in range(indx.size):
                _,_,dec,ra=im.toworld((0,0,indy[iS],indx[iS]))
                Lra.append(ra)
                Ldec.append(dec)
                LS.append(RandS[iS])
                #SRA.append(rad2hmsdms(ra,Type="ra").replace(" ",":"))
                #SDEC.append(rad2hmsdms(dec,Type="dec").replace(" ",":"))

        #Cat=np.zeros((len(Lra),),dtype=[("ra",np.float64),("StrRA","S200"),("dec",np.float64),("StrDEC","S200"),("S",np.float64)])
        Cat=np.zeros((len(Lra),),dtype=[("ra",np.float64),("dec",np.float64),("S",np.float64)])
        Cat=Cat.view(np.recarray)
        Cat.ra=np.array(Lra)
        Cat.dec=np.array(Ldec)
        #Cat.StrRA=np.array(SRA)
        #Cat.StrDEC=np.array(SDEC)
        Cat.S=np.array(LS)
        CatName="%s.cat.npy"%self.OutName
        print>>log,"Saving simulated catalog as %s"%CatName
        np.save(CatName,Cat)

        ModelOut=np.zeros_like(Model)
        ModelOut[0,0]=Model[0,0].T[::-1]
        
        p=self.options.RandomCat_TotalToPeak
        if p>1.:
            nx=101
            x,y=np.mgrid[-nx:nx+1,-nx:nx+1]
            r2=x**2+y**2
            def G(sig):
                C0=1./(2.*np.pi*sig**2)
                C=C0*np.exp(-r2/(2.*sig**2))
                C/=np.sum(C)
                return C
            ListSig=np.linspace(0.001,10.,100)
            TotToPeak=np.array([1./np.max(G(s)) for s in ListSig])
            sig=np.interp(self.options.RandomCat_TotalToPeak,TotToPeak,ListSig)
            print>>log,"Found a sig of %f"%sig
            Gaussian=G(sig)
            ModelOut[0,0]=scipy.signal.fftconvolve(ModelOut[0,0], G(sig), mode='same')

            FWHMFact=2.*np.sqrt(2.*np.log(2.))
            BeamPix=self.BeamPix/FWHMFact
            Model=G(sig).reshape((1,1,x.shape[0],x.shape[0]))
            ConvModel,_=ModFFTW.ConvolveGaussianWrapper(Model,Sig=BeamPix)
            self.SimulObsPeak=np.max(ConvModel)
            print>>log,"  Gaussian Peak: %f"%np.max(Gaussian)
            print>>log,"  Gaussian Int : %f"%np.sum(Gaussian)
            print>>log,"  Obs peak     : %f"%self.SimulObsPeak
            self.header_dict["OPKRATIO"]=self.SimulObsPeak
            self.header_dict["GSIGMA"]=sig
            self.header_dict["RTOTPK"]=self.options.RandomCat_TotalToPeak
        

        else:
            self.header_dict["OPKRATIO"]=1.
            self.header_dict["GSIGMA"]=0.
            self.header_dict["RTOTPK"]=1.

        
        return ModelOut


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
                            SmoothMode=options.SmoothMode,
                            MakeCorrected=options.MakeCorrected,
                            options=options)
    CRM.Restore()
    CRM.killWorkers()





if __name__=="__main__":
    OP=read_options()

    main(OP)
