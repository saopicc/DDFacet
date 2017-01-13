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

import numpy as np
from DDFacet.Other import MyPickle
from DDFacet.ToolsDir import ModFFTW


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

    PSFCrop = PSF[xc_psf - npixside:xc_psf  + npixside + 1,xc_psf - npixside:xc_psf + npixside + 1]
    return PSFCrop


def SquareIslandtoIsland(Model, ThisSquarePixList, ThisPixList):
    ### Build ThisPixList from Model, in the reference frame of the Dirty

    xc, yc = ThisSquarePixList['IslandCenter']  # island center in original dirty
    ListSquarePix_Data = ThisSquarePixList['IslandSquareData']  # square image of the dirty around Island center
    ListSquarePix_Mask = ThisSquarePixList['IslandSquareMask']  # Corresponding square mask image

    NIslandPix = len(ThisPixList)

    Mod_x, Mod_y = Model.shape
    SquarePix_x, SquarePix_y = ListSquarePix_Data.shape

    if Mod_x != SquarePix_x or Mod_y != SquarePix_y:
        raise NameError('Mismatch between output Model image dims and original Square image dims. Please check if the even to uneven correction worked.')

    FluxV = []
    NewThisPixList = []
    for tmpcoor in ThisPixList:
        currentx = tmpcoor[0]
        currenty = tmpcoor[1]
        x_loc_coor = (currentx - xc) + SquarePix_x / 2  # coordinates in the small Model image
        y_loc_coor = (currenty - yc) + SquarePix_y / 2  # coordinates in the small Model image
        if ListSquarePix_Mask[x_loc_coor, y_loc_coor] == 1:  # if it is not masked (e.g. part of the island)
            FluxV.append(ListSquarePix_Data[x_loc_coor, y_loc_coor])
            NewThisPixList.append([currentx, currenty])

    return FluxV,NewThisPixList


def testMO_DATA():
    Dico= MyPickle.Load("SaveTest")
    Dirty=Dico["Dirty"]
    PSF=Dico["PSF"]
    PSF2=np.squeeze(PSF)
    ListPixData=Dico["ListPixData"]
    FreqsInfo=Dico["FreqsInfo"]
    FreqsInfo=Dico["FreqsInfo"]
    IslandBestIndiv=Dico["IslandBestIndiv"]
    ListPixParms=ListPixData
    ListSquarePix=Dico["ListSquarePix"]
    ThisPixList=ListPixData
    ThisSquarePixList=ListSquarePix
    GD=Dico["GD"]
    FacetID=Dico["FacetID"]

    nch=FreqsInfo["MeanJonesBand"][FacetID].size
    WeightMeanJonesBand=FreqsInfo["MeanJonesBand"][FacetID].reshape((nch,1,1,1))
    WeightMueller=WeightMeanJonesBand.ravel()
    WeightMuellerSignal=WeightMueller*FreqsInfo["WeightChansImages"].ravel()

    #IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
    #ListPixData=IncreaseIslandMachine.IncreaseIsland(ListPixData,dx=5)

    # 0) Load Island info (center and square data)

    ListSquarePix_Center=ListSquarePix['IslandCenter']
    ListSquarePix_Data=ListSquarePix['IslandSquareData']
    ListSquarePix_Mask=ListSquarePix['IslandSquareMask']
    orixisland,oriyisland=ListSquarePix_Data.shape # size of the square postage stamp around island

    ListSquarePix_Data=PSF2

    xisland,yisland=ListSquarePix_Data.shape # size of the square postage stamp around island
    print xisland

    # 1) Shape PSF and Dirty to have even number of pixels (required by Moresane)
    # DEAL WITH SQUARE DATA OF ISLAND IF UNEVEN

    # Crop PSF to the island postage stamp
    #PSFCrop=CropPSF(PSF,xisland)
    #PSF2=CropPSF(PSF2,71)
    # MORESANE requires even sized images ==> Padding by one row and one column
    cropped_square_data_to_even = False
    if xisland % 2 != 0:
        #    PSFCrop_even = np.zeros((xisland+1, xisland+1))
        #    PSFCrop_even[:-1, :-1] = np.squeeze(PSFCrop)
        Dirty_even = np.zeros((xisland - 1, xisland - 1))
        Dirty_even[:, :] = ListSquarePix_Data[:-1, :-1]
        cropped_square_data_to_even = True
    else:
        Dirty_even = ListSquarePix_Data
    # make it even by removing one line and one column (usually outside of the interesting island region)


    # xbigdirty,ybigdirty=np.squeeze(Dirty).shape
    # if xbigdirty % 2 != 0:
    #     Dirty_even=np.zeros((xbigdirty-1,xbigdirty-1))
    #     Dirty_even[:,:]=np.squeeze(Dirty)[:-1,:-1]

    xbigpsf,ybigpsf=PSF2.shape
    cropped_square_psf_to_even = False
    if xbigpsf % 2 != 0:
        PSF2_even=np.zeros((xbigpsf-1,xbigpsf-1))
        PSF2_even[:,:]=PSF2[:-1,:-1]
        cropped_square_data_to_even = True
    else:
        PSF2_even=PSF2

    # 2) Run the actual MinorCycle algo
    DictMoresaneParms=GD['MORESANE']
    Moresane=ClassMoresane(Dirty_even,PSF2_even,DictMoresaneParms,GD=GD)

    Model_Square=Moresane.main()

    # 3) Apply Island mask to model to get rid of regions outside the island.

    cropped_square_to_even = False
    if cropped_square_data_to_even:  # then restore the model to its original uneven dimension
        Model_Square_uneven = np.zeros((xisland, xisland))
        Model_Square_uneven[:-1, :-1] = Model_Square
        Model_Square = Model_Square_uneven

    if cropped_square_psf_to_even: # restore original PSF size
        PSF_uneven=PSF2

    Model_Square=CropPSF(Model_Square, orixisland)
    Model_Square *= ListSquarePix_Mask  # masking outside the island

    # 4) Convert back to Island format ( "S" and ThisPixList )
    NewModel, NewThisPixList = SquareIslandtoIsland(Model_Square, ThisSquarePixList, ThisPixList)

    Model = NewModel
    ThisPixList = NewThisPixList

    return Model
