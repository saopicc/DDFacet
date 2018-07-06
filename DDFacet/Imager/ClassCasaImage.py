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

import os
import os.path
from pyrap.images import image

import DDFacet.Data.ClassStokes as ClassStokes
import numpy as np
from DDFacet.Other import MyLogger

log= MyLogger.getLogger("ClassCasaImage")
from astropy.io import fits
from astropy.wcs import WCS
from DDFacet.report_version import report_version

def FileToArray(FileName,CorrT):
    """ Read a FITS FileName file to an array """
    hdu=fits.open(FileName)
    NormImage=hdu[0].data

    
    if len(NormImage.shape)==4:
        nch,npol,_,_=NormImage.shape
    else:
        nch,nx,ny=NormImage.shape
        npol=1
        Sh=(nch,1,nx,ny)
        NormImage=NormImage.reshape(Sh)

    if CorrT:
        for ch in range(nch):
            for pol in range(npol):
                NormImage[ch,pol]=NormImage[ch,pol].T[::-1]
    return NormImage

class ClassCasaimage():

    def __init__(self,ImageName,ImShape,Cell,radec,Freqs=None,KeepCasa=False,Stokes=["I"],header_dict=None,history=None):
        """ Create internal data structures, then call CreateScratch to
        make the image itself. 
        header_dict is a dict of FITS keywords to add.
        history is a list of strings to put in the history
        """
        if KeepCasa:
            raise RuntimeError('KeepCasa = True is not implemented!')
        self.Cell=Cell
        self.radec=radec
        self.KeepCasa=KeepCasa
        self.Freqs = Freqs
        self.Stokes = [ClassStokes.FitsStokesTypes[sid] for sid in Stokes]
        self.sorted_stokes = [s for s in self.Stokes]
        self.sorted_stokes.sort()
        self.header_dict=header_dict
        #work out the FITS spacing between stokes parameters
        if len(self.sorted_stokes) > 1:
            self.delta_stokes = self.sorted_stokes[1] - self.sorted_stokes[0]
        else:
            self.delta_stokes = 1

        for si in range(len(self.sorted_stokes)-1):
            if self.sorted_stokes[si+1] - self.sorted_stokes[si] != self.delta_stokes:
                raise RuntimeError("Your selection of Stokes parameters cannot "
                                   "be stored in a FITS file. The selection must be linearly spaced."
                                   "See FITS standard 3.0 (A&A 524 A42) Table 28 for the indices of the stokes "
                                   "parameters you want to image.")
        self.ImShape=ImShape
        self.nch,self.npol,self.Npix,_ = ImShape

        self.ImageName=ImageName
        #print "image refpix:",rad2hmsdms.rad2hmsdms(radec[0],Type="ra").replace(" ",":"),", ",rad2hmsdms.rad2hmsdms(radec[1],Type="dec").replace(" ",".")
        self.imageFlipped = False
        self.createScratch()
        # Fill in some standard keywords
        self.header['ORIGIN'] = 'DDFacet '+report_version()
        self.header['BTYPE'] = 'Intensity'
        self.header['BUNIT'] = 'Jy/beam'
        self.header['SPECSYS'] = 'TOPOCENT'
        if header_dict is not None:
            for k in header_dict:
                self.header[k]=header_dict[k]
        if history is not None:
            if isinstance(history,str):
                history=[history]
            for h in history:
                self.header['HISTORY']=h

    def createScratch(self):
        """ Create the structures necessary to save the FITS image """

        self.w = WCS(naxis=4)
        self.w.wcs.ctype = ['RA---SIN','DEC--SIN','STOKES','FREQ']
        self.w.wcs.cdelt[0] = -self.Cell/3600.0
        self.w.wcs.cdelt[1] = self.Cell/3600.0
        self.w.wcs.cdelt[2] = self.delta_stokes
        self.w.wcs.cunit = ['deg','deg','','Hz']
        self.w.wcs.crval = [self.radec[0]*180.0/np.pi,self.radec[1]*180.0/np.pi,self.sorted_stokes[0],0]
        self.w.wcs.crpix = [1+(self.Npix-1)/2.0,1+(self.Npix-1)/2.0,1,1]

        self.fmean=None
        if self.Freqs is not None:
            self.w.wcs.crval[3] = self.Freqs[0]

            if self.Freqs.size>1:
                F=self.Freqs
                df=np.mean(self.Freqs[1::]-self.Freqs[0:-1])
                self.w.wcs.cdelt[3]=df
                self.fmean=np.mean(self.Freqs)
            else:
                self.fmean=self.Freqs[0]

        self.header = self.w.to_header()
        if self.fmean is not None:
            self.header['RESTFRQ'] = self.fmean

    def setdata(self, dataIn, CorrT=False):
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
        self.data = data

    def ToFits(self):
        FileOut=self.ImageName+".fits"
        hdu = fits.PrimaryHDU(header=self.header,data=self.data)
        if os.path.exists(FileOut):
            os.unlink(FileOut)
        print>>log, "  ----> Save image data as FITS file %s"%FileOut
        hdu.writeto(FileOut)

    def setBeam(self,beam,beamcube=None):
        """
        Add Fitted beam info to FITS header, expects triple for beam:
        maj: Length of major axis in degrees
        min: Length of minor axis in degrees
        pa: Beam parallactic angle in degrees (counter clockwise, starting from declination-axis)
        """
        bmaj, bmin, PA = beam
        # if the corrT parameter was specified then the image was flipped when synthesized, so we have to
        # reverse the rotation:
        if self.imageFlipped:
            PA = 90 - PA
        FileOut=self.ImageName+".fits"
        #print>>log, "  ----> Save beam info in FITS file %s"%FileOut
        
        self.header["BMAJ"] = bmaj
        self.header["BMIN"] = bmin
        self.header["BPA"] = PA
        if beamcube is not None:
            for band,(bmaj, bmin, bpa) in enumerate(beamcube):
                self.header["BMAJ%d"%band] = bmaj
                self.header["BMIN%d"%band] = bmin
                self.header["BPA%d"%band] = PA

    def close(self):
        #print>>log, "  ----> Closing %s"%self.ImageName
        del(self.data)
        del(self.header)
        #print>>log, "  ----> Closed %s"%self.ImageName


def test():
    np.random.seed(0)
    name,imShape,Cell,radec="lala3.psf", (10, 1, 1029, 1029), 20, (3.7146787856873478, 0.91111035090915093)
    im=ClassCasaimage(name,imShape,Cell,radec, Freqs=np.linspace(1400e6,1500e6,20),header_dict={'comment':'A test'},history=['Here is a history line.','Here is another'])
    im.setdata(np.random.randn(*(imShape)).astype(np.float32),CorrT=True)
    im.setBeam((10.,10.,0.))
    im.ToFits()
    im.close()

if __name__=="__main__":
    test()
