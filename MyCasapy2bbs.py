from pyrap.tables import table
from pyrap.images import image
import pyfits
from Sky import ClassSM
import numpy as np
import glob
import os
from Other import reformat

def test():
    Conv=MyCasapy2BBS("/media/tasse/data/DDFacet/Test/lala2.model.fits")
    Conv.GetPixCat()
    Conv.ToSM()

class MyCasapy2BBS():
    def __init__(self,FitsTh=0.1):
        self.Fits=Fits
        self.Th=Th

    def ToSM(self):
        Osm=reformat.reformat(self.Fits,LastSlash=False)
        SM=ClassSM.ClassSM(Osm,ReName=True,DoREG=True,SaveNp=True,FromExt=self.Cat)
        #SM.print_sm2()

    def GetPixCat(self):
        Fits=self.Fits
    
        im=image(Fits)
        Model=im.getdata()[0,0]

    
        pol,freq,decc,rac=im.toworld((0,0,0,0))
        #print pol,freq,rac, decc

        indx,indy=np.where(Model!=0.)
        NPix=indx.size
        Cat=np.zeros((NPix,),dtype=[('ra',np.float),('dec',np.float),('s',np.float)])
        Cat=Cat.view(np.recarray)

        for ipix in range(indx.size):
            x,y=indx[ipix],indy[ipix]
            s=Model[x,y]
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

        
        Cat=Cat[np.abs(Cat.s)>(self.Th*np.max(np.abs(Cat.s)))]

        self.Cat=Cat
