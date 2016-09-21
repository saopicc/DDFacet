#!/usr/bin/env python

import optparse
import pickle
import numpy as np
from SkyModel.Other import MyLogger
log=MyLogger.getLogger("MakeModel")
from Sky import ClassSM
try:
    from DDFacet.Imager.ModModelMachine import GiveModelMachine
except:
    from DDFacet.Imager.ModModelMachine import ClassModModelMachine
from DDFacet.Imager import ClassCasaImage
from pyrap.images import image

SaveName="last_MakeModel.obj"

def read_options():
    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""
    global options
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--SkyModel',help='List of targets [no default]',default='')
    group.add_option('--OutSkyModel',help='List of targets [no default]',default='')
    group.add_option('--BaseImageName',help='List of targets [no default]',default='')
    group.add_option('--MaskName',help='List of targets [no default]',default='')
    group.add_option('--CleanNegComp',help='List of targets [no default]',type="int",default=0)
    group.add_option('--NCluster',help=' Default is %default',default="0")
    group.add_option('--DoPlot',help=' Default is %default',default="1")
    group.add_option('--DoSelect',help=' Default is %default',default="0")
    group.add_option('--DoPrint',help=' Default is %default',default="0")
    group.add_option('--CMethod',help=' Clustering method [1,2,3,4]. Default is %default',default="4")
    group.add_option('--PreClusterFile',help=' PreClusterFile. Default is %default',default="")
    group.add_option('--RemoveNegComp',help=' PreClusterFile. Default is %default',type=int,default=0)
    group.add_option('--FromClusterCat',help=' PreClusterFile. Default is %default',type=str,default="")
    group.add_option('--ApparantFlux',type="int",help=' PreClusterFile. Default is %default',default=1)
    opt.add_option_group(group)


    options, arguments = opt.parse_args()
    f = open(SaveName,"wb")
    pickle.dump(options,f)
    


def main(options=None):
    if options==None:
        f = open(SaveName,'rb')
        options = pickle.load(f)

    SkyModel=options.SkyModel

    if "," in SkyModel:
        SMList=SkyModel.split(",")
        print>>log, "Concatenating SkyModels %s"%(str(SMList))
        ThisCat=np.load(SMList[0])
        ThisCat=ThisCat.view(np.recarray)
        ThisNDir=len(list(set(ThisCat.Cluster.tolist())))
        CurrentMaxCluster=ThisNDir
        CatList=[ThisCat]
        for SM in SMList[1::]:
            ThisCat=np.load(SM)
            ThisCat=ThisCat.view(np.recarray)
            ThisNDir=len(list(set(ThisCat.Cluster.tolist())))
            ThisCat.Cluster+=CurrentMaxCluster
            CurrentMaxCluster+=ThisNDir
            CatList.append(ThisCat)
        cat=np.concatenate(CatList)
        cat=cat.view(np.recarray)
        OutSkyModel=options.OutSkyModel
        print>>log, "Saving in %s"%(OutSkyModel)
        np.save(OutSkyModel,cat)
        SM=ClassSM.ClassSM(OutSkyModel+".npy",
                           ReName=True,
                           DoREG=True,
                           SaveNp=True)
        SM.Rename()
        SM.Save()

        return

    if options.BaseImageName!="":
        #from DDFacet.Imager.ClassModelMachine import ClassModelMachine

        FileDicoModel="%s.DicoModel"%options.BaseImageName

        # ClassModelMachine,DicoModel=GiveModelMachine(FileDicoModel)
        # MM=ClassModelMachine(Gain=0.1)
        # MM.FromDico(DicoModel)

        ModConstructor = ClassModModelMachine()
        MM=ModConstructor.GiveInitialisedMMFromFile(FileDicoModel)

        SqrtNormImage=None
        if options.ApparantFlux:
            FileSqrtNormImage="%s.Norm.fits"%options.BaseImageName
            imSqrtNormImage=image(FileSqrtNormImage)
            SqrtNormImage=imSqrtNormImage.getdata()
            nchan,npol,_,_=SqrtNormImage.shape
            for ch in range(nchan):
                for pol in range(npol):
                    SqrtNormImage[ch,pol,:,:]=(SqrtNormImage[ch,pol,:,:].T[::-1,:])
            

        

        #FitsFile="%s.model.fits"%options.BaseImageName
        FitsFile="%s.dirty.fits"%options.BaseImageName
        #MM.FromFile(DicoModel)
        if options.MaskName!="":
            MM.CleanMaskedComponants(options.MaskName)
        if options.CleanNegComp:
            MM.CleanNegComponants(box=10,sig=2)

        SkyModel=options.BaseImageName+".npy"
        MM.ToNPYModel(FitsFile,SkyModel,BeamImage=SqrtNormImage)

        # SkyModel="tmpSourceCat.npy"
        # ModelImage=MM.GiveModelImage()
        # im=image(FitsFile)
        # ModelOrig=im.getdata()
        # indx,indy=np.where(ModelImage[0,0]!=ModelImage[0,0])
        # print ModelImage[0,0,indx,indy],ModelImage[0,0,indx,indy]
        # cell=abs(im.coordinates().dict()["direction0"]["cdelt"][0])*180/np.pi*3600
        # ra,dec=im.coordinates().dict()["direction0"]["crval"]
        # CasaImage=ClassCasaImage.ClassCasaimage("Model",ModelImage.shape,cell,(ra,dec))
        # CasaImage.setdata(ModelImage,CorrT=True)
        # CasaImage.ToFits()
        # CasaImage.close()

    if options.RemoveNegComp==1:
        print "Removing negative component"
        Cat=np.load(SkyModel)
        print Cat.shape
        Cat=Cat.view(np.recarray)
        Cat=Cat[Cat.I>0]
        print Cat.shape
        np.save(SkyModel,Cat)
        

        
    NCluster=int(options.NCluster)
    DoPlot=(int(options.DoPlot)==1)
    DoSelect=(int(options.DoSelect)==1)
    CMethod=int(options.CMethod)

    if DoPlot==0:
        import matplotlib
        matplotlib.use('agg')


    SM=ClassSM.ClassSM(SkyModel,ReName=True,
                       DoREG=True,SaveNp=True,
                       SelSource=DoSelect,ClusterMethod=CMethod)

    if True:
        print>>log, "Removing fake gaussians"
        Cat=SM.SourceCat
        
        indG=np.where(Cat["Gmaj"]>0)[0]
        CatSel=Cat[indG]
        Gmaj=CatSel["Gmaj"]*180/np.pi*3600
        I=CatSel["I"]
        
        ind=np.where((I/Gmaj)>3)[0]
        CatSel[ind]["Gmaj"]=0.
        CatSel[ind]["Gmin"]=0.
        CatSel[ind]["Gangle"]=0.
        CatSel[ind]["Type"]=0.
        #SM.SourceCat[indG][ind]=CatSel
        

        indN=np.arange(SM.SourceCat.shape[0])[indG][ind]
        
        SM.SourceCat["Gmaj"][indN]=0.
        SM.SourceCat["Gmin"][indN]=0.
        SM.SourceCat["Gangle"][indN]=0.
        SM.SourceCat["Type"][indN]=0.
        #print SM.SourceCat[indG][ind]
        #np.save(SkyModel,Cat)
        print>>log, "  done"

    PreCluster=options.PreClusterFile
    SM.Cluster(NCluster=NCluster,DoPlot=DoPlot,PreCluster=PreCluster,FromClusterCat=options.FromClusterCat)
    SM.MakeREG()
    SM.Save()



    if options.DoPrint=="1":
        SM.print_sm2()

    


if __name__=="__main__":
    read_options()
    f = open(SaveName,'rb')
    options = pickle.load(f)

    main(options=options)
