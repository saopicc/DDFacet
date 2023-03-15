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
from DDFacet.Other import logger
log=logger.getLogger("PEX")
from SkyModel.PSourceExtract import Gaussian

from pyrap.images import image
from DDFacet.Other.progressbar import ProgressBar
from SkyModel.Other import reformat
from SkyModel.Sky import ClassSM
from SkyModel.Other import rad2hmsdms
from astropy.io import fits
from SkyModel import MakeMask

def PutDataInNewImage(oldfits,newfits,data):
    outim=newfits+'.fits'
    log.print("writting image %s"%outim)
    hdu=fits.open(oldfits)
    hdu[0].data=data
    hdu.writeto(outim,overwrite=True)


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
    group.add_option('--ConvolveInput',help=' Convolve, defaults is %default',default=0,type=int)
    #group.add_option('--NCluster',help=' Boost is %default',default="0")
    group.add_option('--snr',help=' SNR above which we draw an island. Default is %default',default="7")
    #group.add_option('--CMethod',help=' Cluster algorithm Method. Default is %default',default="1")
    group.add_option('--NoiseImage',type=str,help=' Mask image. Default is %default',default=None)
    group.add_option('--NoiseBox',type=int,help=' Mask image. Default is %default',default=300)
    group.add_option('--MaskImage',type=str,help=' Mask image. Default is %default',default=None)
    group.add_option('--ChFreq',type=int,help=' Channel freq in MHz. Default is %default',default=0)
    group.add_option('--NodesFile',type=str,help=' Nodes file',default="")
    
    opt.add_option_group(group)
    options, arguments = opt.parse_args()
    f = open("last_MakePModel.obj","wb")
    pickle.dump(options,f)

def mainFromExt(im=None,Osm="", PSF="",Pfact=1,DoPlot=1,DoPrint=0,Boost=3,snr=7.,NoiseImage=None,NoiseBox=None,
                MaskImage=None,
                ChSlice=0,ChFreq=0):
    class O:
        def __init__(self,**kwargs):
            for key in kwargs.keys(): setattr(self,key,kwargs[key])

    options=O(im=im,Osm=Osm, PSF=PSF,Pfact=Pfact,DoPlot=DoPlot,DoPrint=DoPrint,Boost=Boost,snr=snr,
              NoiseImage=NoiseImage,
              MaskImage=MaskImage,
              NoiseBox=NoiseBox,
              ChSlice=ChSlice,
              ChFreq=ChFreq)
    return main(options)
    
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
    NChSlices=abs(f[0].header["NAXIS4"])
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
        log.print(ModColor.Str(" No psf seen in header"))
        pass

    
    if options.PSF!="":
        m0,m1,pa=options.PSF.split(',')
        PMaj,PMin,PPA=float(m0),float(m1),float(pa)
        PMaj*=Pfact
        PMin*=Pfact


    if PMaj is not None:
        log.print("Using psf (maj,min,pa)=(%6.2f, %6.2f, %6.2f) (mult. fact.=%6.2f)"%(PMaj,PMin,PPA,Pfact))
    else:
        log.print("No psf info could be gotten from anywhere")
        log.print("   use PSF keyword to tell what the psf is or is not")
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

    ChSlice=0
    if options.ChFreq!=0:
        ChFreq=options.ChFreq*1e6
        dFreq=f[0].header["CDELT4"]
        xFreq0=f[0].header["CRPIX4"]-1
        Freq0=f[0].header["CRVAL4"]
        ChSlice=int(round(xFreq0+(ChFreq-Freq0)/dFreq))
        ChSlice=np.min([ChSlice,NChSlices-1])
        ChSlice=np.max([0,ChSlice])
        log.print("  Picking input cube slice #%i (%i slices)"%(ChSlice,NChSlices))
        
    
    b=im.getdata()[ChSlice,0,:,:]

    # if options.ConvolveInput:
    #     log.print("  Convolving input...")
    #     dN=10
    #     N=2*dN+1
    #     x,y=np.mgrid[-dN:dN:N*1j,-dN:dN:N*1j]
    #     print(ThisSm,ThisSM,ThisPA)
    #     G = Gaussian.GaussianXY(x,y,1,sig=(ThisSm,ThisSM),pa=ThisPA)
    #     pylab.clf()
    #     pylab.imshow(G)
    #     pylab.draw()
    #     pylab.show(block=False)
        
    
    #b=b[3000:4000,3000:4000]#[120:170,300:370]
    c=im.coordinates()
    incr=np.abs(c.dict()["direction0"]["cdelt"][0])
    log.print("Psf Size Sigma_(Maj,Min) = (%5.1f,%5.1f) pixels"%(PMaj/incr,PMin/incr))



    NPixPSF=int(np.pi*PMaj/incr*PMin/incr)
    NoiseImage=options.NoiseImage
    
    if NoiseImage is not None:
        ImNoiseImage=image(options.NoiseImage)
        NoiseImage=ImNoiseImage.getdata()[ChSlice,0,:,:]
    else:
        log.print("Compute noise map...")
        b0=options.NoiseBox
        box=(b0,b0)
        x=np.linspace(-10,10,1000)
        f=0.5*(1.+scipy.special.erf(x/np.sqrt(2.)))
        SBox=(box[0]//Boost,box[1]//Boost)
        n=SBox[0]*SBox[1]
        F=1.-(1.-f)**n
        ratio=np.abs(np.interp(0.5,F,x))
        Noise=-scipy.ndimage.filters.minimum_filter(b,SBox)/ratio
        Noise[Noise<0]=1e-10
        NoiseImage=np.zeros_like(b)
        for i in range(Boost):
            for j in range(Boost):
                s00,s01=Noise.shape
                s10,s11=NoiseImage[i::Boost,j::Boost].shape
                s0,s1=min(s00,s10),min(s10,s11)
                NoiseImage[i::Boost,j::Boost][0:s0,0:s1]=Noise[:,:][0:s0,0:s1]
        ind=np.where(NoiseImage==0.)
        NoiseImage[ind]=1e-10



        
    PutDataInNewImage(imname,"%s.PEXNoise"%imname,NoiseImage.reshape((1,1,b.shape[0],b.shape[1])))

    MaskImage=None
    if options.MaskImage is not None:
        MaskImage=image(options.MaskImage).getdata()[ChSlice,0,:,:]
        
    
    Islands=ClassIslands.ClassIslands(b,
                                      snr,
                                      MinPerIsland=NPixPSF,
                                      Boost=Boost,
                                      DoPlot=DoPlot,
                                      MaskImage=MaskImage,
                                      NoiseImage=NoiseImage)
    
    Islands.FindAllIslands()
    PutDataInNewImage(imname,"%s.PEXMask"%imname,np.float32(Islands.MaskImage.reshape((1,1,b.shape[0],b.shape[1]))))

    
    ImOut=np.zeros_like(b)
    pBAR = ProgressBar(Title="Fit islands")

    #log.print "ion"
    #import pylab
    #pylab.ion()

    sourceList=[]
    import pylab
    for i in range(len(Islands.ListX)):
        comment='Isl %i/%i' % (i+1,len(Islands.ListX))
        pBAR.render(float(i+1) , len(Islands.ListX))

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

    
    SM=ClassSM.ClassSM(Osm,
                       ReName=True,DoREG=True,SaveNp=True,FromExt=Cat,DoPrint=0)#,NCluster=NCluster,DoPlot=DoPlot,ClusterMethod=CMethod)
    #SM=ClassSM.ClassSM(Osm,ReName=True,SaveNp=True,DoPlot=DoPlot,FromExt=Cat)
    SM.MakeREG()
    SM.D_FITS=D_FITS
    SM.Rename()
    SM.SavePickle()

    return SM




if __name__=="__main__":
    read_options()
    main()
