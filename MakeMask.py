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

def read_options():
    desc=""" cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options")
    group.add_option('--RestoredIm',type="str",help="default is %default",default=None)
    group.add_option('--Th',type="float",default=10,help="default is %default")
    group.add_option("--Box",type="str",default="120,2",help="default is %default")
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
        

        NMaxPix=20000
        Island=np.zeros((NIslands,NMaxPix,2),np.int32)
        NIslandNonZero=np.zeros((NIslands,),np.int32)

        print>>log,"  Extractinng pixels in islands"
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
        LIslands=[]
        for iIsland in range(1,NIslands):
            ind=np.where(Island[iIsland,:,0]!=0)[0]
            ThisIsland=[]
            Npix=ind.size
            for ipix in range(Npix):
                ThisIsland.append([Island[iIsland,ipix,0].tolist(),Island[iIsland,ipix,1]])
            LIslands.append(ThisIsland)






    def CreateMask(self):
        self.ComputeNoiseMap()
        self.MakeMask()
        # Make island list
        self.BuildIslandList()

        nx,ny=self.ImMask.shape
        ImWrite=self.ImMask.reshape((1,1,nx,ny))
        
        PutDataInNewImage(self.FitsFile,self.FitsFile+".mask",np.float32(ImWrite))

    def plot(self):
        import pylab
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
        pylab.show()

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

