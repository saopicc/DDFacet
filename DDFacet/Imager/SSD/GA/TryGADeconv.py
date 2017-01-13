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
from ClassEvolveGA import ClassEvolveGA
from DDFacet.Other import MyPickle
from DDFacet.ToolsDir import ModFFTW
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


def testMF_DATA():
    Dico= MyPickle.Load("SaveTest")
    Dirty=Dico["Dirty"]
    PSF=Dico["PSF"]

    FreqsInfo=Dico["FreqsInfo"]

    IslandBestIndiv=Dico["IslandBestIndiv"]

    ListPixData=Dico["ListPixData"]
    ListPixParms=Dico["ListPixParms"]

    GD=Dico["GD"]
    FacetID=Dico["FacetID"]
    IdSharedMem=Dico["IdSharedMem"]
    iIsland=Dico["iIsland"]

    #GD["GAClean"]["GASolvePars"]=["S","Alpha","GSig"]
    #GD["GAClean"]["GASolvePars"]=["S","Alpha"]

    nch=FreqsInfo["MeanJonesBand"][FacetID].size
    WeightMeanJonesBand=FreqsInfo["MeanJonesBand"][FacetID].reshape((nch,1,1,1))
    WeightMueller=WeightMeanJonesBand.ravel()
    WeightMuellerSignal=WeightMueller*FreqsInfo["WeightChansImages"].ravel()

    # IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
    # ListPixData=IncreaseIslandMachine.IncreaseIsland(ListPixData,dx=20)

    # IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
    # ListPixData=IncreaseIslandMachine.IncreaseIsland(ListPixData,dx=5)
    #IslandBestIndiv=np.zeros((len(GD["GAClean"]["GASolvePars"])*len(
    CEv=ClassEvolveGA(Dirty,PSF,FreqsInfo,ListPixParms=ListPixParms,ListPixData=ListPixData,
                      GD=GD,
                      IslandBestIndiv=IslandBestIndiv,
                      WeightFreqBands=WeightMuellerSignal,
                      iIsland=iIsland,
                      IdSharedMem=IdSharedMem)
    CEv.main()
