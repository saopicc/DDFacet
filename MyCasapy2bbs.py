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

def read_options():
    desc=""" cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options")
    group.add_option('--ModelIm',help='Input Model image [no default]',default='')
    group.add_option('--RestoredIm',default=None)
    group.add_option('--ResidualIm',default=None)
    group.add_option('--Th',type="float",default=5)
    opt.add_option_group(group)

    
    options, arguments = opt.parse_args()

    f = open(SaveFile,"wb")
    pickle.dump(options,f)
    

            

#####################"

def test():
    Conv=MyCasapy2BBS("/media/tasse/data/DDFacet/Test/lala2.model.fits")
    Conv.GetPixCat()
    Conv.ToSM()


class MyCasapy2BBS():
    def __init__(self,Fits,
                 ImRestoredName=None,
                 ImResidualName=None,
                 Th=None,
                 box=(100,100),Boost=2,
                 ResInPix=1):
        self.Fits=Fits
        self.Th=Th
        self.Mask=None
        self.box=box
        self.Boost=Boost
        self.ImRestoredName=ImRestoredName
        self.ImResidualName=ImResidualName
        self.DoMask=False
        self.ResInPix=ResInPix
        self.XcYcDx=10000,14000,1000
        self.Init()


    def Init(self):
        self.setModelImage()
        self.setRestored()
        self.MakeMask()

    def setModelImage(self):
        print "set model image"
        self.im=image(self.Fits)
        im=self.im
        c=im.coordinates()
        dx,_=c.__dict__["_csys"]["direction0"]["cdelt"]
        self.CellSizeRad=np.abs(dx)
        self.PSFGaussPars=self.ResInPix*self.CellSizeRad,self.ResInPix*self.CellSizeRad,0.
        Fits=self.Fits
    
        self.Model=self.im.getdata()[0,0]
        if self.XcYcDx!=None:
            xc,yc,dx=self.XcYcDx
            x0,x1=xc-dx,xc+dx
            y0,y1=yc-dx,yc+dx
            self.Model=self.Model[x0:x1,y0:y1]
        self.Plot(self.Model)
        print " done set model image"
        

    def setRestored(self):
        print "set restored image"
        if self.ImRestoredName!=None:
            print " read restored image"
            im=image(self.ImRestoredName)
            self.ImRestored=im.getdata()[0,0]
        elif self.ImResidualName!=None:
            print " convolve model image"
            nx,ny=self.Model.shape
            
            self.ImRestored=ModFFTW.ConvolveGaussian(self.Model.reshape((1,1,nx,nx)),CellSizeRad=self.CellSizeRad,GaussPars=[self.PSFGaussPars])
            self.ImRestored=self.ImRestored[0,0]
            im=image(self.ImResidualName)
            ImResidual=im.getdata()[0,0]
            xc,yc,dx=self.XcYcDx
            x0,x1=xc-dx,xc+dx
            y0,y1=yc-dx,yc+dx
            self.ImResidual=ImResidual[x0:x1,y0:y1]
            print " add residual image"
            self.ImRestored+=self.ImResidual
            del(im)
            self.Plot(self.ImRestored)

        else:
            print " done nothing"
        print " done set restored image"


    def Plot(self,data,dx=None):
        return
        import pylab
        xc=data.shape[0]/2
        pylab.clf()
        if dx!=None:
            pylab.imshow(data[xc-dx:xc+dx,xc-dx:xc+dx],interpolation="nearest",cmap="gray")
        else:
            pylab.imshow(data,interpolation="nearest",cmap="gray")
        pylab.draw()
        pylab.show()


    def ComputeNoiseMap(self):
        print "Compute noise map..."
        Boost=self.Boost
        Acopy=self.ImResidual[0::Boost,0::Boost].copy()
        SBox=(self.box[0]/Boost,self.box[1]/Boost)
        Noise=np.sqrt(scipy.ndimage.filters.median_filter(np.abs(Acopy)**2,SBox))
        #ind=(np.abs(Acopy)>3.*Noise)
        #Acopy[ind]=Noise[ind]
        #Noise=np.sqrt(scipy.ndimage.filters.median_filter(np.abs(Acopy)**2,SBox))

        self.Noise=np.zeros_like(self.ImRestored)
        for i in range(Boost):
            for j in range(Boost):
                s00,s01=Noise.shape
                s10,s11=self.Noise[i::Boost,j::Boost].shape
                s0,s1=min(s00,s10),min(s10,s11)
                self.Noise[i::Boost,j::Boost][0:s0,0:s1]=Noise[:,:][0:s0,0:s1]
        print " ... done"
        ind=np.where(self.Noise==0.)
        self.Noise[ind]=1e-10
        self.Plot(self.Noise)

    def MakeMask(self):
        if self.ImRestored==None: return
        self.ComputeNoiseMap()
        self.Mask=(self.ImRestored>(self.Th*self.Noise))
        self.DoMask=True

    def ToSM(self):
        Osm=reformat.reformat(self.Fits,LastSlash=False)
        SM=ClassSM.ClassSM(Osm,ReName=True,DoREG=True,SaveNp=True,FromExt=self.Cat)
        #SM.print_sm2()

    def GetPixCat(self):
        im=self.im
        Model=self.Model
    
        pol,freq,decc,rac=im.toworld((0,0,0,0))
        #print pol,freq,rac, decc

        indx,indy=np.where(Model!=0.)
        NPix=indx.size
        Cat=np.zeros((NPix,),dtype=[('ra',np.float),('dec',np.float),('s',np.float)])
        Cat=Cat.view(np.recarray)
        X=[]
        Y=[]

        for ipix in range(indx.size):

            x,y=indx[ipix],indy[ipix]
            if self.DoMask:
                if not(self.Mask[x,y]):
                    continue

            s=Model[x,y]
            X.append(x)
            Y.append(y)
            #a,b,dec,ra=im.toworld((0,0,0,0))
            a,b,dec,ra=im.toworld((0,0,x,y))
            #print a,b,dec,ra
            #ra*=(1./60)#*np.pi/180
            #dec*=(1./60)#*np.pi/180

            
            Cat.ra[ipix]=ra
            Cat.dec[ipix]=dec
            Cat.s[ipix]=s
            # print "======="
            # #print x,y,s
            # print ra,dec#,s

        
        # if (self.Th!=None)&(self.ModelConv==None):
        #     Cat=Cat[np.abs(Cat.s)>(self.Th*np.max(np.abs(Cat.s)))]

        Cat=Cat[Cat.s!=0]

        print "ok"
        import pylab
        pylab.clf()
        pylab.imshow(self.ImRestored.T,interpolation="nearest",cmap="gray")
        pylab.scatter(X,Y,marker=".")
        pylab.draw()
        pylab.show()

        self.Cat=Cat

    
def main(options=None):
    
    if options==None:
        f = open(SaveFile,'rb')
        options = pickle.load(f)

    Conv=MyCasapy2BBS(options.ModelIm,
                      ImRestoredName=options.RestoredIm,
                      ImResidualName=options.ResidualIm,
                      Th=options.Th)
    Conv.GetPixCat()
    Conv.ToSM()
    
if __name__=="__main__":
    read_options()
    f = open(SaveFile,'rb')
    options = pickle.load(f)
    main(options=options)

