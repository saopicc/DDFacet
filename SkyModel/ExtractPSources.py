#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

import numpy as np
from SkyModel.PSourceExtract import Gaussian
import pylab
import scipy.optimize
import time
from SkyModel.PSourceExtract import ClassIslands
from SkyModel.Other import ModColor
import pickle
import optparse
#from SkyModel.PSourceExtract.ClassPointFit2 import ClassPointFit as ClassFit
from SkyModel.PSourceExtract.ClassGaussFit import ClassGaussFit as ClassFit
#import ClassPointFit as ClassPointFit

from pyrap.images import image
from SkyModel.Other.progressbar import ProgressBar
from SkyModel.Other import reformat
from SkyModel.Sky import ClassSM
from SkyModel.Other import rad2hmsdms
from astropy.io import fits
from SkyModel import MakeMask

def read_options():
    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""
    global options
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--im',help='Image name [no default]',default='')
    group.add_option('--Osm',help='Output Sky model [no default]',default='')
    group.add_option('--PSF',help='PSF (Majax,Minax,PA) in (arcsec,arcsec,deg). Default is %default',default="")
    group.add_option('--Pfact',help='PSF size multiplying factor. Default is %default',default="1")
    group.add_option('--DoPlot',help=' Default is %default',default="1")
    group.add_option('--DoPrint',help=' Default is %default',default="0")
    group.add_option('--Boost',help=' Boost is %default',default="3")
    #group.add_option('--NCluster',help=' Boost is %default',default="0")
    group.add_option('--snr',help=' SNR above which we draw an island. Default is %default',default="7")
    #group.add_option('--CMethod',help=' Cluster algorithm Method. Default is %default',default="1")
    group.add_option('--NoiseImage',type=str,help=' Mask image. Default is %default',default=None)
    
    opt.add_option_group(group)
    options, arguments = opt.parse_args()
    f = open("last_MakePModel.obj","wb")
    pickle.dump(options,f)
    
def main(options=None):
    if options==None:
        f = open("last_MakePModel.obj",'rb')
        options = pickle.load(f)

    Boost=int(options.Boost)
    #CMethod=int(options.CMethod)
    #NCluster=int(options.NCluster)
    Osm=options.Osm
    Pfact=float(options.Pfact)
    DoPlot=(options.DoPlot=="1")
    imname=options.im
    snr=float(options.snr)
    if Osm=="":
        Osm=reformat.reformat(imname,LastSlash=False)



    f=fits.open(imname)
    # fix up comments
                
    rac,decc=f[0].header["CRVAL1"]*np.pi/180,f[0].header["CRVAL2"]*np.pi/180
    dPix=abs(f[0].header["CDELT1"])*np.pi/180
    NPix=abs(f[0].header["NAXIS1"])
    D_FITS={"rac":rac,"decc":decc,"NPix":NPix,"dPix":dPix}

    im=image(imname)
    PMaj=None
    try:
        PMaj=(im.imageinfo()["restoringbeam"]["major"]["value"])
        PMin=(im.imageinfo()["restoringbeam"]["minor"]["value"])#/2
        PPA=(im.imageinfo()["restoringbeam"]["positionangle"]["value"])
        PMaj*=Pfact
        PMin*=Pfact
    except:
        print(ModColor.Str(" No psf seen in header"))
        pass

    if options.PSF!="":
        m0,m1,pa=options.PSF.split(',')
        PMaj,PMin,PPA=float(m0),float(m1),float(pa)
        PMaj*=Pfact
        PMin*=Pfact


    if PMaj is not None:
        print(ModColor.Str(" - Using psf (maj,min,pa)=(%6.2f, %6.2f, %6.2f) (mult. fact.=%6.2f)"
                           %(PMaj,PMin,PPA,Pfact),col='green',Bold=False))
    else:
        print(ModColor.Str(" - No psf info could be gotten from anywhere"))
        print(ModColor.Str("   use PSF keyword to tell what the psf is or is not"))
        exit()


    ToSig=(1./3600.)*(np.pi/180.)/(2.*np.sqrt(2.*np.log(2)))
    PMaj*=ToSig
    PMin*=ToSig

    PPA*=np.pi/180
    
    #PPA=0
    #stop
    #PPA-=np.pi/2
    #PPA+=np.pi/4
    #PPA+=np.pi/4
    #PPA+=np.pi/2

    b=im.getdata()[0,0,:,:]
    #b=b[3000:4000,3000:4000]#[120:170,300:370]
    c=im.coordinates()
    incr=np.abs(c.dict()["direction0"]["cdelt"][0])
    print(ModColor.Str("   - Psf Size Sigma_(Maj,Min) = (%5.1f,%5.1f) pixels"%(PMaj/incr,PMin/incr),col="green",Bold=False))



    NPixPSF=int(np.pi*PMaj/incr*PMin/incr)

    ImNoiseImage=image(options.NoiseImage)
    NoiseImage=ImNoiseImage.getdata()[0,0,:,:]
    Islands=ClassIslands.ClassIslands(b,
                                      snr,
                                      MinPerIsland=NPixPSF,
                                      Boost=Boost,
                                      DoPlot=DoPlot,
                                      NoiseImage=NoiseImage)
    Islands.FindAllIslands()
    
    ImOut=np.zeros_like(b)
    pBAR = ProgressBar('white', block='=', empty=' ',Title="Fit islands")

    #print "ion"
    #import pylab
    #pylab.ion()

    sourceList=[]
    import pylab
    for i in range(len(Islands.ListX)):
        comment='Isl %i/%i' % (i+1,len(Islands.ListX))
        pBAR.render(int(100* float(i+1) / len(Islands.ListX)), comment)

        xin,yin,zin=np.array(Islands.ListX[i]),np.array(Islands.ListY[i]),np.array(Islands.ListS[i])
        xm=int(np.sum(xin*zin)/np.sum(zin))
        ym=int(np.sum(yin*zin)/np.sum(zin))
        #Fit=ClassFit(xin,yin,zin,psf=(PMaj/incr,PMin/incr,PPA),noise=Islands.Noise[xm,ym])
        # for PPA in np.linspace(-np.pi,np.pi,20):
        Fit=ClassFit(xin,yin,zin,psf=(PMaj/incr,PMin/incr,PPA),noise=Islands.Noise[xm,ym],FreePars=["l", "m","s"])
            
        sourceList.append(Fit.DoAllFit())
        x,y,z=Fit.PutFittedArray(ImOut)
            
            # pylab.clf()
            # pylab.scatter(x,y,c=z)
            # pylab.draw()
            # pylab.show(block=False)
            # pylab.pause(0.1)
            

    nx,ny=ImOut.shape
    MakeMask.PutDataInNewImage(imname,imname+".GaussFit",ImOut.reshape((1,nx,ny)))
        
    Islands.FitIm=ImOut
    xlist=[]
    ylist=[]
    slist=[]

    Cat=np.zeros((50000,),dtype=[('ra',np.float),('dec',np.float),('s',np.float),('I',np.float),('Gmaj',np.float),('Gmin',np.float),('PA',np.float),('Gangle',np.float)])
    Cat=Cat.view(np.recarray)

    isource=0

    for Dico in sourceList:
        for iCompDico in sorted(list(Dico.keys())):
            CompDico=Dico[iCompDico]
            i=CompDico["l"]
            j=CompDico["m"]
            s=CompDico["s"]
            xlist.append(i)
            ylist.append(j)
            slist.append(s)
            f,d,dec,ra=im.toworld((0,0,i,j))
            Cat.ra[isource]=ra
            Cat.dec[isource]=dec
            Cat.s[isource]=s
            Cat.I[isource]=s

            Cat.Gmin[isource]=CompDico["Sm"]*incr/ToSig
            Cat.Gmaj[isource]=CompDico["SM"]*incr/ToSig
            Cat.PA[isource]=CompDico["PA"]
            Cat.Gangle[isource]=CompDico["PA"]
            isource +=1
            

    Cat=Cat[Cat.ra!=0].copy()
    Islands.FittedComps=(xlist,ylist,slist)
    Islands.plot()

    
    SM=ClassSM.ClassSM(Osm,ReName=True,DoREG=True,SaveNp=True,FromExt=Cat)#,NCluster=NCluster,DoPlot=DoPlot,ClusterMethod=CMethod)
    #SM=ClassSM.ClassSM(Osm,ReName=True,SaveNp=True,DoPlot=DoPlot,FromExt=Cat)
    SM.MakeREG()
    SM.D_FITS=D_FITS
    SM.SavePickle()





if __name__=="__main__":
    read_options()
    main()
