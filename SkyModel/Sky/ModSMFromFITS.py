from __future__ import division, absolute_import, print_function

import numpy as np
from astropy.io import fits

def ReadSMFromFITS(infile):

    print("Reading-FITS %s"%infile)
    f=fits.open(infile)
    FitsImage=f[1].header["INIMAGE"]
    # fix up comments
    keywords=[('CRVAL1',float),('CRVAL2',float),('CDELT1',float),('NAXIS1',int)]
    c=f[1].header['COMMENT']
    for l in c:
        for k,ty in keywords:
            if k in l:
                bits=l.split()
                print("Warning: getting keyword %s from comments" % k)
                f[1].header['I_'+k]=ty(bits[2])
                
    decc,rac=f[1].header["I_CRVAL1"],f[1].header["I_CRVAL2"]
    rac,decc=f[1].header["I_CRVAL1"],f[1].header["I_CRVAL2"]
    dPix=abs(f[1].header["I_CDELT1"])*np.pi/180
    NPix=abs(f[1].header["I_NAXIS1"])
    
    c=fits.open(infile)[1]
    c.data.RA*=np.pi/180.
    c.data.DEC*=np.pi/180.
    
    ns=c.data.RA.size
    
    Cf=c.data


    Cat=np.zeros((ns,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('Sref',np.float),('I',np.float),('Q',np.float),\
                              ('U',np.float),('V',np.float),('RefFreq',np.float),('alpha',np.float),('ESref',np.float),\
                              ('Ealpha',np.float),('kill',np.int),('Cluster',np.int),('Type',np.int),('Gmin',np.float),\
                              ('Gmaj',np.float),('Gangle',np.float),("Select",np.int),('l',np.float),('m',np.float),("Exclude",bool)])
    Cat=Cat.view(np.recarray)
    Cat.RefFreq=1.
    Cat.ra[0:ns]=Cf.RA
    Cat.dec[0:ns]=Cf.DEC
    Cat.I[0:ns]=Cf.Total_flux
    
    Cat.Gmin[0:ns]=Cf.DC_Min*np.pi/180
    Cat.Gmaj[0:ns]=Cf.DC_Maj*np.pi/180
    Cat.Gangle[0:ns]=Cf.DC_PA*np.pi/180


    Cat=Cat[Cat.ra!=0.]
    Cat.Type[Cat.Gmaj>0.]=1
    
    Cat.Sref=Cat.I
    rac=rac*np.pi/180
    decc=decc*np.pi/180
    D_FITS={"rac":rac,"decc":decc,"NPix":NPix,"dPix":dPix}

    return Cat,D_FITS
