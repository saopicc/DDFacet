
from DDFacet.ToolsDir import ModFFTW
import numpy as np
import ipdb
from ClassMoresane import ClassMoresane
from DDFacet.Other import MyPickle
from SkyModel.PSourceExtract import ClassIncreaseIsland

def test():
    nx=17
    DataTrue=np.zeros((1,1,nx,nx),np.float32)
    DataTrue[0,0,nx/2,nx/2]=1.
    DataPSF=DataTrue.copy()
    PSF=ModFFTW.ConvolveGaussian(DataPSF,CellSizeRad=1,GaussPars=[(1.,1.,0.)])

    DataTrue.fill(0)
    DataTrue[0,0,nx/2,nx/2]=1.
    DataTrue[0,0,nx/2+5,nx/2+4]=1.
    #DataTrue[0,0,nx/2,nx/2]=1.
    DataTrue=ModFFTW.ConvolveGaussian(DataTrue,CellSizeRad=1,GaussPars=[(1,.5,0.)])
    #DataTrue=np.random.randn(*DataTrue.shape)

    DataTrue[0,0,nx/2-5,nx/2+4]+=2.
    DataTrue[0,0,nx/2-2,nx/2+2]+=2.
    # DataTrue+=np.random.randn(*DataTrue.shape)*0.05

    Dirty=ModFFTW.ConvolveGaussian(DataTrue,CellSizeRad=1,GaussPars=[(1.,1.,0.)])
    #DataConv=ModFFTW.ConvolveGaussian(DataTrue,CellSizeRad=1,GaussPars=[(1.,1.,0.)])
    #DataConv=DataTrue
    #IndDataTrue=ArrayToInd(DataConv)
    FreqsInfo=None

    _,_,indx,indy=np.where(DataTrue>1e-1)
    ListPixParms=[ij for ij in zip(indx,indy)]

    _,_,indx,indy=np.where(Dirty>1e-1)
    ListPixData=[ij for ij in zip(indx,indy)]

    CEv=ClassEvolveGA(Dirty,PSF,FreqsInfo,ListPixData=ListPixData,ListPixParms=ListPixParms)
    CEv.ArrayMethodsMachine.DataTrue=DataTrue


    return CEv

def test():
    nx=17
    DataTrue=np.zeros((1,1,nx,nx),np.float32)
    DataTrue[0,0,nx/2,nx/2]=1.
    DataPSF=DataTrue.copy()
    PSF=ModFFTW.ConvolveGaussian(DataPSF,CellSizeRad=1,GaussPars=[(1.,1.,0.)])

    # DataTrue.fill(0)
    # DataTrue[0,0,nx/2,nx/2]=1.
    # DataTrue[0,0,nx/2+5,nx/2+4]=1.
    # #DataTrue[0,0,nx/2,nx/2]=1.
    # DataTrue=ModFFTW.ConvolveGaussian(DataTrue,CellSizeRad=1,GaussPars=[(1,.5,0.)])
    # #DataTrue=np.random.randn(*DataTrue.shape)

    # DataTrue[0,0,nx/2-5,nx/2+4]+=2.
    # DataTrue[0,0,nx/2-2,nx/2+2]+=2.
    # # DataTrue+=np.random.randn(*DataTrue.shape)*0.05

    Dirty=ModFFTW.ConvolveGaussian(DataTrue,CellSizeRad=1,GaussPars=[(1.,1.,0.)])
    #DataConv=ModFFTW.ConvolveGaussian(DataTrue,CellSizeRad=1,GaussPars=[(1.,1.,0.)])
    #DataConv=DataTrue
    #IndDataTrue=ArrayToInd(DataConv)
    FreqsInfo=None

    #    _,_,indx,indy=np.where(DataTrue>1e-1)
    #    ListPixParms=[ij for ij in zip(indx,indy)]

    #    _,_,indx,indy=np.where(Dirty>1e-1)
    #    ListPixData=[ij for ij in zip(indx,indy)]

    #    CEv=ClassEvolveGA(Dirty,PSF,FreqsInfo,ListPixData=ListPixData,ListPixParms=ListPixParms)
    #    CEv.ArrayMethodsMachine.DataTrue=DataTrue
    ListPixParms=None#[(2,2)]

    ListPixData=None#[(2,2),(2,3)]


    CEv=ClassEvolveGA(Dirty,PSF,FreqsInfo,ListPixData=ListPixData,ListPixParms=ListPixParms)
    CEv.ArrayMethodsMachine.DataTrue=DataTrue


    return CEv

def testMF():
    nx=17

    nf=2
    nu=np.linspace(100,200,nf)*1e6
    FreqsInfo=None
    FreqsInfo={"freqs":[nu[0:1],nu[1:2]],"WeightChansImages":np.array([0.5, 0.5])}

    # nf=1
    # nu=np.linspace(100,200,nf)*1e6
    # FreqsInfo=None

    PSF=np.zeros((nf,1,nx,nx),np.float32)
    PSF[:,0,nx/2,nx/2]=1.
    PSF=ModFFTW.ConvolveGaussian(PSF,CellSizeRad=1,GaussPars=[(1.,1.,0.)]*nf)

    DataModel=np.zeros((1,1,nx,nx),np.float32)
    Alpha=np.zeros((1,1,nx,nx),np.float32)

    DataModel[0,0,nx/2,nx/2]=1.
    #Alpha[0,0,:,:]=-.8

    #DataModel[0,0,nx/2+5,nx/2+4]=1.
    DataModel=ModFFTW.ConvolveGaussian(DataModel,CellSizeRad=1,GaussPars=[(1,.5,0.)])
    #Alpha[0,0,nx/2,nx/2]=-.8

    DataModel[0,0,nx/2,nx/2]=+1.
    Alpha[0,0,nx/2,nx/2]=.8
    DataModel[0,0,nx/2-5,nx/2+4]+=2.
    #DataModel[0,0,nx/2-2,nx/2+2]+=2.
    Alpha[0,0,nx/2-5,nx/2+4]=-.8

    FreqRef=np.mean(nu)
    nur=nu.reshape((2,1,1,1))
    DataModelMF=DataModel*(nur/FreqRef)**Alpha

    Dirty=ModFFTW.ConvolveGaussian(DataModelMF,CellSizeRad=1,GaussPars=[(1.,1.,0.)]*nf)

    #DataConv=ModFFTW.ConvolveGaussian(DataTrue,CellSizeRad=1,GaussPars=[(1.,1.,0.)])
    #DataConv=DataTrue
    #IndDataTrue=ArrayToInd(DataConv)

    indx,indy=np.where(Dirty[0,0]>1e-1)
    ListPixParms=[ij for ij in zip(indx,indy)]
    indx,indy=np.where(Dirty[0,0]>1e-2)
    ListPixParms=[ij for ij in zip(indx,indy)]
    ListPixData=ListPixParms

    GD={"GAClean":{"GASolvePars":["S","Alpha"]}}


    CEv=ClassEvolveGA(Dirty,PSF,FreqsInfo,ListPixParms=ListPixParms,ListPixData=ListPixData,GD=GD)
    CEv.ArrayMethodsMachine.DataTrue=DataModelMF
    CEv.ArrayMethodsMachine.PM.DicoIParm["S"]["DataModel"]=DataModel


    return CEv

def CropPSF(PSF, npix):
    nx_psf, ny_psf = PSF.shape
    xc_psf = nx_psf / 2

    if npix % 2 == 0:
        print "Cropping size should be odd (npix=%d) !!! Adding 1 pixel" % npix
        npix = npix + 1

    if npix > nx_psf or npix > ny_psf:
        print "Cropping size larger than PSF size !!!"
        stop

    npixside = (npix - 1) / 2  # pixel to include from PSF center.

    PSFCrop = PSF[xc_psf - npixside:xc_psf + npixside + 1,xc_psf - npixside:xc_psf + npixside + 1]
    return PSFCrop


def testMO_DATA():
    Dico=MyPickle.Load("SaveTest")
    Dirty=Dico["Dirty"]
    PSF=Dico["PSF"]
    PSF2=np.squeeze(PSF)
    ListPixData=Dico["ListPixData"]
    FreqsInfo=Dico["FreqsInfo"]
    FreqsInfo=Dico["FreqsInfo"]
    IslandBestIndiv=Dico["IslandBestIndiv"]
    ListPixParms=ListPixData
    ListSquarePix=Dico["ListSquarePix"]
    GD=Dico["GD"]
    FacetID=Dico["FacetID"]

    nch=FreqsInfo["MeanJonesBand"][FacetID].size
    WeightMeanJonesBand=FreqsInfo["MeanJonesBand"][FacetID].reshape((nch,1,1,1))
    WeightMueller=WeightMeanJonesBand.ravel()
    WeightMuellerSignal=WeightMueller*FreqsInfo["WeightChansImages"].ravel()

    #IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
    #ListPixData=IncreaseIslandMachine.IncreaseIsland(ListPixData,dx=5)

    # Load Postage Stamp around current Island
    ListSquarePix_center=ListSquarePix['Islandcenter']
    ListSquarePix_data=ListSquarePix['Squaredata']
    xisland,yisland=ListSquarePix_data.shape # size of the square postage stamp around island

    # Crop PSF to the island postage stamp
    #PSFCrop=CropPSF(PSF,xisland)
    PSF2=CropPSF(PSF2,71)
    # MORESANE requires even sized images ==> Padding by one row and one column
    if xisland % 2 != 0:
    #    PSFCrop_even = np.zeros((xisland+1, xisland+1))
    #    PSFCrop_even[:-1, :-1] = np.squeeze(PSFCrop)
        Dirty_even=np.zeros((xisland-1,xisland-1))
        Dirty_even[:,:]=ListSquarePix_data[:-1,:-1]


    xbigdirty,ybigdirty=np.squeeze(Dirty).shape
    if xbigdirty % 2 != 0:
        Dirty_even=np.zeros((xbigdirty-1,xbigdirty-1))
        Dirty_even[:,:]=np.squeeze(Dirty)[:-1,:-1]

    xbigpsf,ybigpsf=PSF2.shape
    if xbigpsf % 2 != 0:
        PSF2_even=np.zeros((xbigpsf-1,xbigpsf-1))
        PSF2_even[:,:]=PSF2[:-1,:-1]



    DictMoresaneParms=GD['MORESANE']
    Moresane=ClassMoresane(PSF2_even,PSF2_even,DictMoresaneParms,GD=GD)
    Model=Moresane.main()
    ipdb.set_trace()

    return Model
