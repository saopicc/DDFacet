#!/usr/bin/env python

from pyrap.tables import table
from pyrap.images import image
import pyfits
from Sky import ClassSM
import optparse
import numpy as np
import glob
import os
from Other import reformat
SaveFile="last_MyCasapy2BBS.obj"
import pickle
import scipy.ndimage
from Tools import ModFFTW
from SkyModel.PSourceExtract import ClassIslands
from SkyModel.Other.ClassCasaImage import PutDataInNewImage
import scipy.special
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("MakeMask")
from killMS2.Other.progressbar import ProgressBar
import collections
import pylab
from SkyModel.Other.MyHist import MyCumulHist

def read_options():
    desc=""" cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options")
    group.add_option('--RestoredIm',type="str",help="default is %default",default=None)
    group.add_option('--Th',type="float",default=10,help="default is %default")
    group.add_option("--Box",type="str",default="30,2",help="default is %default")
    #group.add_option("--MedFilter",type="str",default="50,10")
    opt.add_option_group(group)

    
    options, arguments = opt.parse_args()

    f = open(SaveFile,"wb")
    pickle.dump(options,f)
    

            

#####################"

def test():
    FitsFile="/media/tasse/data/DDFacet/Test/MultiFreqs3.restored.fits"
    Conv=ClassMakeMask(FitsFile=FitsFile,Th=5.,Box=(50,10))
    Conv.ComputeNoiseMap()
    Conv.FindIslands()

    nx,ny=Conv.ImIsland.shape
    ImWrite=Conv.ImIsland.reshape((1,1,nx,ny))

    PutDataInNewImage(FitsFile,FitsFile+".mask",np.float32(ImWrite))

    #Conv.plot()

    # import pylab
    # pylab.clf()
    # ax=pylab.subplot(1,2,1)
    # pylab.imshow(Conv.Restored[0,0],cmap="gray")
    # pylab.subplot(1,2,2,sharex=ax,sharey=ax)
    # pylab.imshow(Conv.IslandsMachine.ImIsland,cmap="gray")
    # pylab.draw()
    # pylab.show(False)
    # stop
    


class ClassMakeMask():
    def __init__(self,FitsFile=None,
                 Th=5.,
                 Box=(50,10)):
        self.FitsFile=FitsFile
        self.Th=Th
        self.Box,self.IncrPix=Box
        self.Boost=self.IncrPix
        self.box=self.Box,self.Box
        self.CasaIm=image(self.FitsFile)
        self.Restored=self.CasaIm.getdata()

        im=self.CasaIm
        PMaj=(im.imageinfo()["restoringbeam"]["major"]["value"])
        PMin=(im.imageinfo()["restoringbeam"]["minor"]["value"])
        PPA=(im.imageinfo()["restoringbeam"]["positionangle"]["value"])
        c=im.coordinates()
        incr=np.abs(c.dict()["direction0"]["cdelt"][0])
        
        ToSig=(1./3600.)*(np.pi/180.)/(2.*np.sqrt(2.*np.log(2)))
        SigMaj_rad=PMaj*ToSig
        SigMin_rad=PMin*ToSig
        SixMaj_pix=SigMaj_rad/incr
        SixMin_pix=SigMin_rad/incr
        PPA_rad=PPA*np.pi/180
        
        _,_,nx,ny=self.Restored.shape
        from SkyModel.PSourceExtract import Gaussian
        xc,yc=nx/2,nx/2
        sup=200
        x,y=np.mgrid[-sup:sup:1,-sup:sup:1]
        
        G=Gaussian.GaussianXY(x,y,1.,sig=(30,30),pa=0.)
        self.Restored[0,0,xc:xc+2*sup,yc:yc+2*sup]=G[:,:]
        
        x,y=np.mgrid[-10:11:1,-10:11:1]
        self.RefGauss=Gaussian.GaussianXY(x,y,1.,sig=(SixMin_pix,SixMaj_pix),pa=PPA_rad)
        self.RefGauss_xy=x,y

        BeamMin_pix=SixMin_pix*(2.*np.sqrt(2.*np.log(2)))
        BeamMaj_pix=SixMaj_pix*(2.*np.sqrt(2.*np.log(2)))
        print>>log, "Restoring Beam size of (%i, %i) pixels"%(BeamMin_pix, BeamMaj_pix)
        
        #self.Restored=np.load("testim.npy")
        self.A=self.Restored[0,0]

    def GiveVal(self,A,xin,yin):
        x,y=round(xin),round(yin)
        s=A.shape[0]-1
        cond=(x<0)|(x>s)|(y<0)|(y>s)
        if cond:
            value="out"
        else:
            value="%8.2f mJy"%(A.T[x,y]*1000.)
        return "x=%4i, y=%4i, value=%10s"%(x,y,value)

    def ComputeNoiseMap(self):
        print>>log, "Compute noise map..."
        Boost=self.Boost
        Acopy=self.Restored[0,0,0::Boost,0::Boost].copy()
        SBox=(self.box[0]/Boost,self.box[1]/Boost)


        # MeanAbs=scipy.ndimage.filters.mean_filter(np.abs(Acopy),SBox)
        # Acopy[Acopy>0]=MeanAbs[Acopy>0]
        # Noise=np.sqrt(scipy.ndimage.filters.median_filter(np.abs(Acopy)**2,SBox))

        x=np.linspace(-10,10,1000)
        f=0.5*(1.+scipy.special.erf(x/np.sqrt(2.)))
        n=SBox[0]*SBox[1]
        F=1.-(1.-f)**n
        ratio=np.abs(np.interp(0.5,F,x))

        Noise=-scipy.ndimage.filters.minimum_filter(Acopy,SBox)/ratio
        #Noise[Noise<0]=0

        # indxy=(Acopy>5.*Noise)
        # Acopy[indxy]=5*Noise[indxy]
        # Noise=np.sqrt(scipy.ndimage.filters.median_filter(np.abs(Acopy)**2,SBox))

        # indxy=(Acopy>5.*Noise)
        # Acopy[indxy]=5*Noise[indxy]
        # Noise=np.sqrt(scipy.ndimage.filters.median_filter(np.abs(Acopy)**2,SBox))

        NoiseMed=np.median(Noise)
        Noise[Noise<NoiseMed]=NoiseMed

        self.Noise=np.zeros_like(self.Restored[0,0])
        for i in range(Boost):
            for j in range(Boost):
                s00,s01=Noise.shape
                s10,s11=self.Noise[i::Boost,j::Boost].shape
                s0,s1=min(s00,s10),min(s10,s11)
                self.Noise[i::Boost,j::Boost][0:s0,0:s1]=Noise[:,:][0:s0,0:s1]
        ind=np.where(self.Noise==0.)
        self.Noise[ind]=1e-10

    # def ComputeNoiseMap(self):
    #     print "Compute noise map..."
    #     Boost=self.Boost
    #     Acopy=self.Restored[0,0,0::Boost,0::Boost].copy()
    #     SBox=(self.box[0]/Boost,self.box[1]/Boost)
    #     Noise=np.sqrt(scipy.ndimage.filters.median_filter(np.abs(Acopy)**2,SBox))
    #     self.Noise=np.zeros_like(self.Restored[0,0])
    #     for i in range(Boost):
    #         for j in range(Boost):
    #             s00,s01=Noise.shape
    #             s10,s11=self.Noise[i::Boost,j::Boost].shape
    #             s0,s1=min(s00,s10),min(s10,s11)
    #             self.Noise[i::Boost,j::Boost][0:s0,0:s1]=Noise[:,:][0:s0,0:s1]
    #     print " ... done"
    #     ind=np.where(self.Noise==0.)
    #     self.Noise[ind]=1e-10

    def MakeMask(self):
        self.ImMask=(self.Restored[0,0,:,:]>self.Th*self.Noise)
        #self.ImIsland=scipy.ndimage.filters.median_filter(self.ImIsland,size=(3,3))

    def BuildIslandList(self):
        import scipy.ndimage

        print>>log,"  Labeling islands"
        self.ImIsland,NIslands=scipy.ndimage.label(self.ImMask)
        ImIsland=self.ImIsland
        NIslands+=1
        nx,_=ImIsland.shape

        print>>log,"  Found %i islands"%NIslands
        
        NMaxPix=100000
        Island=np.zeros((NIslands,NMaxPix,2),np.int32)
        NIslandNonZero=np.zeros((NIslands,),np.int32)

        print>>log,"  Extracting pixels in islands"
        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="      Extracting ", HeaderSize=10,TitleSize=13)
        comment=''



        for ipix in range(nx):
            
            pBAR.render(int(100*ipix / (nx-1)), comment)
            for jpix in range(nx):
                iIsland=self.ImIsland[ipix,jpix]
                if iIsland:
                    NThis=NIslandNonZero[iIsland]
                    Island[iIsland,NThis,0]=ipix
                    Island[iIsland,NThis,1]=jpix
                    NIslandNonZero[iIsland]+=1

        print>>log,"  Listing pixels in islands"

        NMinPixIsland=5
        DicoIslands=collections.OrderedDict()
        for iIsland in range(1,NIslands):
            ind=np.where(Island[iIsland,:,0]!=0)[0]
            if ind.size < NMinPixIsland: continue
            Npix=ind.size
            Comps=np.zeros((Npix,3),np.float32)
            for ipix in range(Npix):
                x,y=Island[iIsland,ipix,0],Island[iIsland,ipix,1]
                s=self.Restored[0,0,x,y]
                Comps[ipix,0]=x
                Comps[ipix,1]=y
                Comps[ipix,2]=s
            DicoIslands[iIsland]=Comps

        print>>log,"  Final number of islands: %i"%len(DicoIslands)
        self.DicoIslands=DicoIslands
        

    def FilterIslands(self):
        DicoIslands=self.DicoIslands
        NIslands=len(self.DicoIslands)
        print>>log, "  Filter each individual islands"
        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="      Filter ", HeaderSize=10,TitleSize=13)
        comment=''
        for iIsland in DicoIslands.keys():
            pBAR.render(int(100*iIsland / (len(DicoIslands.keys())-1)), comment)
            x,y,s=DicoIslands[iIsland].T
            #Im=self.GiveIm(x,y,s)
            #pylab.subplot(1,2,1)
            #pylab.imshow(Im,interpolation="nearest")
            # pylab.subplot(1,2,2)

            sr=self.RefGauss.copy()*np.max(s)

            xm,ym=int(np.mean(x)),int(np.mean(y))
            Th=self.Th*self.Noise[xm,ym]

            xg,yg=self.RefGauss_xy

            MaskSel=(sr>Th)
            xg_sel=xg[MaskSel].ravel()
            yg_sel=yg[MaskSel].ravel()
            sr_sel=sr[MaskSel].ravel()


            ###############
            logs=s*s.size#np.log10(s*s.size)
            X,Y=MyCumulHist(logs)
            logsr=sr_sel*sr_sel.size#np.log10(sr_sel*sr_sel.size)
            Xr,Yr=MyCumulHist(logsr)
            Cut=0.9
            ThisTh=np.interp(Cut,Yr,Xr)
            #ThisTh=(ThisTh)/sr_sel.size
            
            #Im=self.GiveIm(xg_sel,yg_sel,sr_sel)
            #pylab.subplot(1,2,2)
            #pylab.imshow(Im,interpolation="nearest")
            


            ind=np.where(s*s.size>ThisTh)[0]
            #print ThisTh,ind.size/float(s.size )
            DicoIslands[iIsland]=DicoIslands[iIsland][ind].copy()
        #     pylab.clf()
        #     pylab.plot(X,Y)
        #     pylab.plot([ThisTh,ThisTh],[0,1],color="black")
        #     pylab.plot(Xr,Yr,color="black",lw=2,ls="--")
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        #     import time
        #     time.sleep(1)
        # stop

    def IslandsToMask(self):
        self.ImMask.fill(0)
        DicoIslands=self.DicoIslands
        NIslands=len(self.DicoIslands)
        print>>log, "  Building mask image from filtered islands"
        for iIsland in DicoIslands.keys():
            x,y,s=DicoIslands[iIsland].T
            self.ImMask[np.int32(x),np.int32(y)]=1


    def GiveIm(self,x,y,s):
        dx=np.int32(x-x.min())
        dy=np.int32(y-y.min())
        nx=dx.max()+1
        ny=dy.max()+1
        print nx,ny
        Im=np.zeros((nx,ny),np.float32)
        Im[dx,dy]=s
        return Im

    def CreateMask(self):
        self.ComputeNoiseMap()
        self.MakeMask()
        # Make island list
        self.BuildIslandList()
        self.FilterIslands()
        self.IslandsToMask()
        self.plot()
        nx,ny=self.ImMask.shape
        ImWrite=self.ImMask.reshape((1,1,nx,ny))
        
        PutDataInNewImage(self.FitsFile,self.FitsFile+".mask",np.float32(ImWrite))

    def plot(self):
        pylab.clf()
        ax1=pylab.subplot(2,3,1)
        vmin,vmax=-np.max(self.Noise),5*np.max(self.Noise)
        MaxRms=np.max(self.Noise)
        ax1.imshow(self.A,vmin=vmin,vmax=vmax,interpolation="nearest",cmap="gray",origin="lower")
        ax1.format_coord = lambda x,y : self.GiveVal(self.A,x,y)
        pylab.title("Image")

        ax2=pylab.subplot(2,3,3,sharex=ax1,sharey=ax1)
        pylab.imshow(self.Noise,vmin=0.,vmax=np.max(self.Noise),interpolation="nearest",cmap="gray",origin="lower")
        ax2.format_coord = lambda x,y : self.GiveVal(self.Noise,x,y)
        pylab.title("Noise Image")
        pylab.xlim(0,self.A.shape[0]-1)
        pylab.ylim(0,self.A.shape[0]-1)


        ax3=pylab.subplot(2,3,6,sharex=ax1,sharey=ax1)
        ax3.imshow(self.ImMask,vmin=vmin,vmax=vmax,interpolation="nearest",cmap="gray",origin="lower")
        ax3.format_coord = lambda x,y : self.GiveVal(self.ImMask,x,y)
        pylab.title("Island Image")
        pylab.xlim(0,self.A.shape[0]-1)
        pylab.ylim(0,self.A.shape[0]-1)

        pylab.draw()
        pylab.show(False)

def main(options=None):
    
    if options==None:
        f = open(SaveFile,'rb')
        options = pickle.load(f)

    s0,s1=options.Box.split(",")
    Box=(int(s0),int(s1))
        
    MaskMachine=ClassMakeMask(options.RestoredIm,
                       Th=options.Th,
                       Box=Box)
    MaskMachine.CreateMask()

if __name__=="__main__":
    read_options()
    f = open(SaveFile,'rb')
    options = pickle.load(f)
    main(options=options)

