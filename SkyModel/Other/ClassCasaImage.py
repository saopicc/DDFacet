
from pyrap.images import image
import os
from SkyModel.Other import MyPickle
import numpy as np
from SkyModel.Other import MyLogger
log=MyLogger.getLogger("ClassCasaImage")
from SkyModel.Other import rad2hmsdms
import astropy.io.fits as pyfits
import pyrap.images

def PutDataInNewImage(ImageNameIn,ImageNameOut,data,CorrT=False):
    im=image(ImageNameIn)

    F=pyfits.open(ImageNameIn)
    F0=F[0]
    nx=F0.header["NAXIS1"]
    ny=F0.header["NAXIS2"]
    npol=F0.header["NAXIS3"]
    nch=F0.header["NAXIS4"]
    shape=(nch,npol,ny,nx)
    
    Dico=im.coordinates().dict()
    cell=abs(Dico["direction0"]["cdelt"][0])*180/np.pi*3600
    ra,dec=Dico["direction0"]["crval"]
    CasaImage=ClassCasaimage(ImageNameOut,shape,cell,(ra,dec))
    CasaImage.setdata(data,CorrT=CorrT)
    CasaImage.ToFits()
    CasaImage.close()


class ClassCasaimage():
    def __init__(self,ImageName,ImShape,Cell,radec,Freqs=None,KeepCasa=False):
        self.Cell=Cell
        self.radec=radec
        self.KeepCasa=KeepCasa
        self.Freqs=Freqs
        self.ImShape=ImShape
        self.nch,self.npol,self.Npix,_=ImShape

        self.ImageName=ImageName
        #print "image refpix:",rad2hmsdms.rad2hmsdms(radec[0],Type="ra").replace(" ",":"),", ",rad2hmsdms.rad2hmsdms(radec[1],Type="dec").replace(" ",".")
        self.createScratch()

    def createScratch(self):
        ImageName=self.ImageName
        #print>>log, "  ----> Create casa image %s"%ImageName
        #HYPERCAL_DIR=os.environ["HYPERCAL_DIR"]
        tmpIm=image(imagename=ImageName,shape=self.ImShape)
        c=tmpIm.coordinates()
        del(tmpIm)
        os.system("rm -Rf %s"%ImageName)

        incr=c.get_increment()
        incrRad=(self.Cell/60.)#*np.pi/180
        incr[-1][0]=incrRad
        incr[-1][1]=-incrRad
        #RefPix=c.get_referencepixel()
        Npix=self.Npix
        #RefPix[0][0]=Npix/2
        #RefPix[0][1]=Npix/2

        #RefPix[0][0]=Npix/2-1
        #RefPix[0][1]=Npix/2-1

        #RefPix=c.set_referencepixel(RefPix)
        RefVal=c.get_referencevalue()
        RaDecRad=self.radec
        RefVal[-1][1]=RaDecRad[0]*180./np.pi*60
        RefVal[-1][0]=RaDecRad[1]*180./np.pi*60
        if self.Freqs!=None:
            #print RefVal[0]
            RefVal[0]=self.Freqs[0]
            ich,ipol,xy=c.get_referencepixel()
            ich=0
            c.set_referencepixel((ich,ipol,xy))
            # if self.Freqs.size>1:
            #     F=self.Freqs
            #     df=np.mean(self.Freqs[1::]-self.Freqs[0:-1])
            #     print df
            #     incr[0]=df
            #     D=c.__dict__["_csys"]
            #     fmean=np.mean(self.Freqs)
            #     D["worldreplace2"]=np.array([fmean])

            #     D["spectral2"]["restfreq"]=fmean
            #     D["spectral2"]["restfreqs"]=np.array([fmean])
            #     D["spectral2"]["tabular"]={'axes': ['Frequency'],
            #                                'pc': np.array([[ 1.]]),
            #                                'units': ['Hz']}



            #     D["spectral2"]["tabular"]["cdelt"]=np.array([df])
            #     D["spectral2"]["tabular"]["crpix"]=np.array([0])
            #     D["spectral2"]["tabular"]["crval"]=np.array([fmean])
            #     D["spectral2"]["tabular"]["pixelvalues"]=np.arange(F.size)
            #     D["spectral2"]["tabular"]["worldvalues"]=F
 
            #    # Out[16]: array([  1.42040575e+09])
                
            #     # In [17]: c.dict()["spectral2"]["tabular"]
            #     # Out[17]: 
            #     # {'axes': ['Frequency'],
            #     #  'cdelt': array([ 0.]),
            #     #  'crpix': array([ 0.]),
            #     #  'crval': array([  1.41500000e+09]),
            #     #  'pc': array([[ 1.]]),
            #     #  'pixelvalues': array([ 0.]),
            #     #  'units': ['Hz'],
            #     #  'worldvalues': array([  1.41500000e+09])}


        c.set_increment(incr)
        c.set_referencevalue(RefVal)


        #self.im=image(imagename=ImageName,shape=(1,1,Npix,Npix),coordsys=c)
        #self.im=image(imagename=ImageName,shape=(Npix,Npix),coordsys=c)
        self.im=image(imagename=ImageName,shape=self.ImShape,coordsys=c)
        #data=np.random.randn(*self.ImShape)
        #self.setdata(data)
        
    def setdata(self,dataIn,CorrT=False):
        #print>>log, "  ----> put data in casa image %s"%self.ImageName

        data=dataIn.copy()
        if CorrT:
            nch,npol,_,_=dataIn.shape
            for ch in range(nch):
                for pol in range(npol):
                    data[ch,pol]=data[ch,pol][::-1].T
        
        self.im.putdata(data)

    def ToFits(self):
        FileOut=self.ImageName+".fits"
        os.system("rm -rf %s"%FileOut)
        print>>log, "  ----> Save data in casa image as FITS file %s"%FileOut
        self.im.tofits(FileOut)

    def setBeam(self,beam):
        bmaj, bmin, PA=beam
        FileOut=self.ImageName+".fits"
        #print>>log, "  ----> Save beam info in FITS file %s"%FileOut
        
        F2=pyfits.open(FileOut)
        F2[0].header["BMAJ"]=bmaj
        F2[0].header["BMIN"]=bmin
        F2[0].header["BPA"]=PA
        os.system("rm -rf %s"%FileOut)
        F2.writeto(FileOut,clobber=True)


    def close(self):
        #print>>log, "  ----> Closing %s"%self.ImageName
        del(self.im)
        #print>>log, "  ----> Closed %s"%self.ImageName
        if self.KeepCasa==False:
            #print>>log, "  ----> Delete %s"%self.ImageName
            os.system("rm -rf %s"%self.ImageName)




# def test():

#     ra=15.*(2.+30./60)*np.pi/180
#     dec=(40.+30./60)*np.pi/180
    
#     radec=(ra,dec)
#     Cell=20.
#     imShape=(1, 1, 1029, 1029)
#     #name="lala2.psf"
#     name,imShape,Cell,radec="lala2.psf", (1, 1, 1029, 1029), 20, (3.7146787856873478, 0.91111035090915093)

#     im=ClassCasaimage(name,imShape,Cell,radec)
#     im.setdata(np.random.randn(*(imShape)),CorrT=True)
#     im.ToFits()
#     im.setBeam((0.,0.,0.))
#     im.close()

def test():
    name,imShape,Cell,radec="lala2.psf", (3, 1, 1029, 1029), 20, (3.7146787856873478, 0.91111035090915093)
    im=ClassCasaimage(name,imShape,Cell,radec,Freqs=np.array([100,200,300],dtype=np.float32)*1e6)
    im.setdata(np.random.randn(*(imShape)),CorrT=True)
    im.ToFits()
    #im.setBeam((0.,0.,0.))
    im.close()

if __name__=="__main__":
    test()
