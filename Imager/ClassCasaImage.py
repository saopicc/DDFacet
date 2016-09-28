
from pyrap.images import image
import os
import os.path
from DDFacet.Other import MyPickle
import numpy as np
from DDFacet.Other import MyLogger
import DDFacet.Data.ClassStokes as ClassStokes
log=MyLogger.getLogger("ClassCasaImage")
from DDFacet.ToolsDir import rad2hmsdms
import pyfits
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

def GiveCoord(nx,npol=1,Freqs=None,Stokes=["I"]):

    df=Freqs[1]-Freqs[0]
    D={'spectral2': {'_axes_sizes': np.array([Freqs.size], dtype=np.int32),
                   '_image_axes': np.array([1], dtype=np.int32),
                   'conversion': {'direction': {'m0': {'unit': 'rad',
                                                       'value': 0.0},
                                                'm1': {'unit': 'rad',
                                                       'value': 1.5707963267948966},
                                                'refer': 'J2000',
                                                'type': 'direction'},
                                  'epoch': {'m0': {'unit': 'd', 'value': 0.0},
                                            'refer': 'LAST',
                                            'type': 'epoch'},
                                  'position': {'m0': {'unit': 'rad',
                                                      'value': 0.0},
                                               'm1': {'unit': 'rad',
                                                      'value': 0.0},
                                               'm2': {'unit': 'm',
                                                      'value': 0.0},
                                               'refer': 'ITRF',
                                               'type': 'position'},
                                  'system': 'BARY'},
                   'formatUnit': '',
                   'name': 'Frequency',
                   'restfreq': 1420405750.0,
                   'restfreqs': np.array([  1.42040575e+09]),
                   'system': 'BARY',
                   'tabular': {'axes': ['Frequency'],
                               'cdelt': np.array([ df]),
                               'crpix': np.array([ 0.]),
                               'crval': np.array([  Freqs[0]]),
                               'pc': np.array([[ 1.]]),
                               'pixelvalues': np.arange(Freqs.size),
                               'units': ['Hz'],
                               'worldvalues': Freqs},
                   'unit': 'Hz',
                   'velType': 0,
                   'velUnit': 'km/s',
                   'version': 2,
                   'waveUnit': 'mm'},
     'stokes1': {'_axes_sizes': np.array([1], dtype=np.int32),
                 '_image_axes': np.array([0], dtype=np.int32),
                 'axes': ['Stokes'],
                 'cdelt': np.array([ 1.]),
                 'crpix': np.array([ 0.]),
                 'crval': np.array([ 1.]),
                 'pc': np.array([[ 1.]]),
                 'stokes': [ClassStokes.FitsStokesTypes[sid] for sid in Stokes]},
     'telescope': 'VLA',
     'telescopeposition': {'m0': {'unit': 'rad', 'value': -1.8782942581394362},
                           'm1': {'unit': 'rad', 'value': 0.5916750987501983},
                           'm2': {'unit': 'm', 'value': 6373576.280915651},
                           'refer': 'ITRF',
                           'type': 'position'},
     'worldmap0': np.array([0, 1], dtype=np.int32),
     'worldmap1': np.array([3], dtype=np.int32),
     'worldmap2': np.array([2], dtype=np.int32),
     'worldreplace0': np.array([ 3.47746408,  0.72981776]),
     'worldreplace1': np.array([ 1.]),
     'worldreplace2': np.array([  1.41698165e+09])}
    
    return D

def FileToArray(FileName,CorrT):
    CasaNormImage=image(FileName)
    NormImage=CasaNormImage.getdata()
    nch,npol,nx,_=NormImage.shape
    if CorrT:
        for ch in range(nch):
            for pol in range(npol):
                NormImage[ch,pol]=NormImage[ch,pol].T[::-1]
    return NormImage


class ClassCasaimage():


    def __init__(self,ImageName,ImShape,Cell,radec,Freqs=None,KeepCasa=False,Stokes=["I"]):
        self.Cell=Cell
        self.radec=radec
        self.KeepCasa=KeepCasa
        self.Freqs  = Freqs
        self.Stokes = [ClassStokes.FitsStokesTypes[sid] for sid in Stokes]
        self.sorted_stokes = [s for s in self.Stokes]
        self.sorted_stokes.sort()
        #work out the FITS spacing between stokes parameters
        if len(self.sorted_stokes) > 1:
            self.delta_stokes = self.sorted_stokes[1] - self.sorted_stokes[0]
        else:
            self.delta_stokes = 1

        for si in range(len(self.sorted_stokes)-1):
            if self.sorted_stokes[si+1] - self.sorted_stokes[si] != self.delta_stokes:
                raise RuntimeError("Your selection of Stokes parameters cannot "
                                   "be stored in a FITS file. The selection must be linearly spaced."
                                   "See FITS standard 3.0 (A&A 524 A42) Table 28 for the indicies of the stokes "
                                   "parameters you want to image.")
        self.ImShape=ImShape
        self.nch,self.npol,self.Npix,_ = ImShape

        self.ImageName=ImageName
        #print "image refpix:",rad2hmsdms.rad2hmsdms(radec[0],Type="ra").replace(" ",":"),", ",rad2hmsdms.rad2hmsdms(radec[1],Type="dec").replace(" ",".")
        self.imageFlipped = False
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
        incr[-2][0]=self.delta_stokes
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
        RefVal[-2][0] = self.sorted_stokes[0]

        D = c.__dict__["_csys"]
        stokes_ids = []
        for k in ClassStokes.FitsStokesTypes:
            stokes_ids.append((k, ClassStokes.FitsStokesTypes[k]))
        images_stokes_values = []
        for key,val in stokes_ids:
            for sval in self.sorted_stokes:
                if val == sval:
                    images_stokes_values.append(key)
        D["stokes1"]["stokes"] = images_stokes_values

        if self.Freqs is not None:
            RefVal[0] = self.Freqs[0]
            ich,ipol,xy = c.get_referencepixel()
            ich=0
            c.set_referencepixel((ich,ipol,xy))

            if self.Freqs.size>1:
                F=self.Freqs
                df=np.mean(self.Freqs[1::]-self.Freqs[0:-1])

                incr[0]=df
                D=c.__dict__["_csys"]
                fmean=np.mean(self.Freqs)

                D["worldreplace2"]=np.array([F[0]])

                D["spectral2"]["restfreq"]=0.#self.Freqs[0]
                D["spectral2"]["restfreqs"]=np.array([0.])#self.Freqs[0]])
                D["spectral2"]["tabular"]={'axes': ['Frequency'],
                                           'pc': np.array([[ 1.]]),
                                           'units': ['Hz']}



                D["spectral2"]["tabular"]["cdelt"]=np.array([df])
                D["spectral2"]["tabular"]["crpix"]=np.array([0])
                D["spectral2"]["tabular"]["crval"]=np.array([F[0]])
                D["spectral2"]["tabular"]["pixelvalues"]=np.arange(F.size)
                D["spectral2"]["tabular"]["worldvalues"]=F
 
               # Out[16]: array([  1.42040575e+09])
                
                # In [17]: c.dict()["spectral2"]["tabular"]
                # Out[17]: 
                # {'axes': ['Frequency'],
                #  'cdelt': array([ 0.]),
                #  'crpix': array([ 0.]),
                #  'crval': array([  1.41500000e+09]),
                #  'pc': array([[ 1.]]),
                #  'pixelvalues': array([ 0.]),
                #  'units': ['Hz'],
                #  'worldvalues': array([  1.41500000e+09])}


        c.set_increment(incr)
        c.set_referencevalue(RefVal)
        
        # import pprint
        # pprint.pprint(c.dict())
        
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
                    #Need to place stokes data in increasing order because of the linear spacing assumption used in FITS
                    stokes_slice_id = self.Stokes.index(self.sorted_stokes[pol])
                    data[ch,pol]=dataIn[ch][stokes_slice_id][::-1].T
        self.imageFlipped = CorrT
        self.im.putdata(data)

    def ToFits(self):
        FileOut=self.ImageName+".fits"
        if os.path.exists(FileOut):
            os.unlink(FileOut)
        print>>log, "  ----> Save data in casa image as FITS file %s"%FileOut
        self.im.tofits(FileOut,overwrite=True)

    def setBeam(self,beam,beamcube=None):
        """
        Add Fitted beam info to FITS header, expects tripple for beam:
        maj: Length of major axis in degrees
        min: Length of minor axis in degrees
        pa: Beam paralactic angle in degrees (counter clockwise, starting from declination-axis)
        """
        bmaj, bmin, PA = beam
        # if the corrT parameter was specified then the image was flipped when synthesized, so we have to
        # reverse the rotation:
        if self.imageFlipped:
            PA = 90 - PA
        FileOut=self.ImageName+".fits"
        #print>>log, "  ----> Save beam info in FITS file %s"%FileOut
        
        F2=pyfits.open(FileOut)
        F2[0].header["BMAJ"] = bmaj
        F2[0].header["BMIN"] = bmin
        F2[0].header["BPA"] = PA
        if beamcube is not None:
            for band,(bmaj, bmin, bpa) in enumerate(beamcube):
                F2[0].header["BMAJ%d"%band] = bmaj
                F2[0].header["BMIN%d"%band] = bmin
                F2[0].header["BPA%d"%band] = PA
        if os.path.exists(FileOut):
            os.unlink(FileOut)
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
    name,imShape,Cell,radec="lala2.psf", (10, 1, 1029, 1029), 20, (3.7146787856873478, 0.91111035090915093)
    im=ClassCasaimage(name,imShape,Cell,radec,Lambda=[1,0.1,10])
    im.setdata(np.random.randn(*(imShape)),CorrT=True)
    im.ToFits()
    #im.setBeam((0.,0.,0.))
    im.close()

if __name__=="__main__":
    test()
