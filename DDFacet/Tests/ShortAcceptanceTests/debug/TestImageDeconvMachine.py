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

from pyrap.images import image

import DDFacet.Imager.ClassImageDeconvMachineMSMF as ClassImageDeconvMachine
import numpy as np


def test():
    impsf=image("Continuous.psf")
    psf=impsf.getdata()
    imdirty=image("Continuous.dirty")
    dirty=imdirty.getdata()
    DC=ClassImageDeconvMachine.ClassImageDeconvMachine(Gain=0.051,MaxMinorIter=200,NCPU=30)
    DC.SetDirtyPSF(dirty,psf)
    DC.setSideLobeLevel(0.1)
    DC.Deconvolve()


def test2():
    
    imdirty=np.float32(np.load("imageresid.npy"))

    _,_,N0,_=imdirty.shape
    dirty=np.float32(imdirty[:,:,0:1001,0:1001]).copy()
    _,_,N1,_=dirty.shape
    x0,x1=N0/2-N1/2,N0/2+N1/2+1
    psf=np.float32(np.load("imagepsf.npy"))[:,:,x0:x1,x0:x1]
    print dirty.shape,psf.shape


    GD={"MultiScale":{}}
    GD["MultiScale"]["Scales"]=[1,2,4,8,16]
    GD["MultiScale"]["Ratios"]=[1.33,1.66,2]
    GD["MultiScale"]["NTheta"]=6
    DC=ClassImageDeconvMachine.ClassImageDeconvMachine(Gain=1.,MaxMinorIter=200,NCPU=6,GD=GD)

    DC.SetDirtyPSF(dirty.copy(),psf.copy())
    DC.setSideLobeLevel(0.2,30)
    DC.FindPSFExtent(Method="FromSideLobe")
    DC.MakeMultiScaleCube()

    DC.Deconvolve()
    
def test3():

    psfname="lala2.nocompDeg3.psf.fits"
    dirtyname="lala2.nocompDeg3.dirty.fits"

    
    impsf=image(psfname)
    psf=np.float32(impsf.getdata())
    imdirty=image(dirtyname)#Test.KAFCA.3SB.dirty.fits")
    dirty=np.float32(imdirty.getdata())
    
    GD={"MultiScale":{}}
    GD["MultiScale"]["Scales"]=[1,2,4,8,16]
    GD["MultiScale"]["Ratios"]=[1.33,1.66,2]
    GD["MultiScale"]["NTheta"]=6
    DC=ClassImageDeconvMachine.ClassImageDeconvMachine(Gain=.1,MaxMinorIter=1000,NCPU=30,GD=GD)
    DC.SetDirtyPSF(dirty,psf)
    DC.setSideLobeLevel(0.2,10)
    DC.FindPSFExtent(Method="FromSideLobe")

    DC.MakeMultiScaleCube()
    DC.Deconvolve()
    

    c=imdirty.coordinates()
    radec=c.dict()["direction0"]["crval"]

    import ClassCasaImage
    CasaImage=ClassCasaImage.ClassCasaimage("modeltest",DC._ModelImage.shape,2.,radec)
    CasaImage.setdata(DC._ModelImage)#,CorrT=True)
    CasaImage.ToFits()
    CasaImage.close()

def test4():

    DicoPSF=dict(np.load("PSF.npz"))#MyPickle.Load("DicoPSF")
    DicoDirty=dict(np.load("Dirty.npz"))#MyPickle.Load("DicoDirty")
    DicoPSF["freqs"]={0: [[99800000.0, 100000000.0, 100200000.0]], 1: [[174800000.0, 175000000.0, 175200000.0]], 2: [[249800000.0, 250000000.0, 250200000.0]]}


    # DicoPSF=dict(np.load("Bootes/PSF.npz"))#MyPickle.Load("DicoPSF")
    # DicoDirty=dict(np.load("Bootes/Dirty.npz"))#MyPickle.Load("DicoDirty")
    # DicoPSF["freqs"]={0: [[130223083.49609375, 130320739.74609375, 130418395.99609375, 130516052.24609375, 130613708.49609375, 130711364.74609375, 130809020.99609375, 130906677.24609375, 131004333.49609375, 131101989.74609375, 131199645.99609375, 131297302.24609375, 131394958.49609375, 131492614.74609375, 131590270.99609375, 131687927.24609375, 131785583.49609375, 131883239.74609375, 131980895.99609375, 132078552.24609375], [132176208.49609375, 132273864.74609375, 132371520.99609375, 132469177.24609375, 132566833.49609375, 132664489.74609375, 132762145.99609375, 132859802.24609375, 132957458.49609375, 133055114.74609375, 133152770.99609375, 133250427.24609375, 133348083.49609375, 133445739.74609375, 133543395.99609375, 133641052.24609375, 133738708.49609375, 133836364.74609375, 133934020.99609375, 134031677.24609375]], 1: [[134129333.49609375, 134226989.74609375, 134324645.99609375, 134422302.24609375, 134519958.49609375, 134617614.74609375, 134715270.99609375, 134812927.24609375, 134910583.49609375, 135008239.74609375, 135105895.99609375, 135203552.24609375, 135301208.49609375, 135398864.74609375, 135496520.99609375, 135594177.24609375, 135691833.49609375, 135789489.74609375, 135887145.99609375, 135984802.24609375]], 2: [[136082458.49609375, 136180114.74609375, 136277770.99609375, 136375427.24609375, 136473083.49609375, 136570739.74609375, 136668395.99609375, 136766052.24609375, 136863708.49609375, 136961364.74609375, 137059020.99609375, 137156677.24609375, 137254333.49609375, 137351989.74609375, 137449645.99609375, 137547302.24609375, 137644958.49609375, 137742614.74609375, 137840270.99609375, 137937927.24609375], [138035583.49609375, 138133239.74609375, 138230895.99609375, 138328552.24609375, 138426208.49609375, 138523864.74609375, 138621520.99609375, 138719177.24609375, 138816833.49609375, 138914489.74609375, 139012145.99609375, 139109802.24609375, 139207458.49609375, 139305114.74609375, 139402770.99609375, 139500427.24609375, 139598083.49609375, 139695739.74609375, 139793395.99609375, 139891052.24609375]]}


    # x0,x1=2300-200,2700+200
    # y0,y1=2500-200,2900+200
    # DicoDirty["ImagData"]=DicoDirty["ImagData"][:,:,x0:x1,y0:y1]
    # DicoDirty["MeanImage"]=DicoDirty["MeanImage"][:,:,x0:x1,y0:y1]
    # DicoPSF["ImagData"]=DicoPSF["ImagData"]#[:,:,x0:x1,y0:y1]
    # DicoPSF["MeanImage"]=DicoPSF["MeanImage"]#[:,:,x0:x1,y0:y1]




    #psfname="lala2.nocompDeg3.psf.fits"
    #dirtyname="lala2.nocompDeg3.dirty.fits"

    
    #impsf=image(psfname)
    #psf=np.float32(impsf.getdata())
    #imdirty=image(dirtyname)#Test.KAFCA.3SB.dirty.fits")
    #dirty=np.float32(imdirty.getdata())

#    DicoDirty["ImagData"]+=1
    
    GD={"MultiScale":{},"MultiFreqs":{},"ImagerDeconv":{}}
    GD["MultiScale"]["Scales"]=[0]
    GD["MultiScale"]["Ratios"]=[]
    GD["MultiScale"]["NTheta"]=6
    GD["MultiFreqs"]["NFreqBands"]=3
    GD["MultiFreqs"]["Alpha"]=[-1.,0.,2]
    GD["MultiFreqs"]["NTerms"]=2
    GD["ImagerDeconv"]["Gain"]=0.1
    DC=ClassImageDeconvMachine.ClassImageDeconvMachine(Gain=GD["ImagerDeconv"]["Gain"],MaxMinorIter=500,NCPU=30,GD=GD)
    DC.SetDirtyPSF(DicoDirty,DicoPSF)
    DC.setSideLobeLevel(0.2,10)
    DC.InitMSMF()
    DC.Deconvolve()
    

    nu=np.linspace(100,300,10)*1e6
    Flux=np.zeros_like(nu)
    for inu in range(nu.size):
        ThisNu=nu[inu]
        Model=DC.MSMachine.ModelMachine.GiveModelImage(np.mean(DicoPSF["freqs"][1]))
        Flux[inu]=np.max(Model)

    import pylab
    pylab.clf()
    pylab.plot(np.log10(nu/1e6),np.log10(Flux))
    #pylab.imshow(Model[0,0],interpolation="nearest")
    #pylab.colorbar()
    pylab.draw()
    pylab.show(False)
    

    # c=imdirty.coordinates()
    # radec=c.dict()["direction0"]["crval"]

    # import ClassCasaImage
    # CasaImage=ClassCasaImage.ClassCasaimage("modeltest",DC._ModelImage.shape,2.,radec)
    # CasaImage.setdata(DC._ModelImage)#,CorrT=True)
    # CasaImage.ToFits()
    # CasaImage.close()

