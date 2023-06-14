from DDFacet.Imager import ClassDeconvMachine
import copy
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u

import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from DDFacet.Other import logger
log=logger.getLogger("ClassMultiField")

from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms


def readMultiFieldFile(FName):
    # Read the CSV file
    with open(FName, 'r') as file:
        lines = file.readlines()[1:]

    # Read the CSV file
    #data = ascii.read(FName, format='fast_csv', delimiter=' ', comment='#', names=['ra', 'dec', "NPix"])
    #data = ascii.read('your_file.csv', format='fast_csv', delimiter=' ', comment='#', names=['ra', 'dec', "NPix"], header_start=-1, data_start=0)
    # Parse the data using Astropy's Table
    data = ascii.read(lines, delimiter=' ', names=['ra', 'dec', "NPix"])
    # Get the columns for RA and Dec
    ra_col = data['ra']
    dec_col = data['dec']
    NPix = data['NPix']  # Assuming 'dec' is the column name for Dec
    
    # Convert to SkyCoord
    return data

    
# def readMultiFieldFile(FName):
#     data = ascii.read(FName, format='basic', delimiter='\s+|#')
    
#     ra_col = data['ra']  # Assuming 'ra' is the column name for RA
#     dec_col = data['dec']  # Assuming 'dec' is the column name for Dec
    
#     coords = SkyCoord(ra=ra_col, dec=dec_col, unit=(u.hourangle, u.deg))
    
    
class ClassImagerDeconv():

    def __init__(self, *args, **kwargs):
        self.GD=copy.deepcopy(kwargs["GD"])
        self.kwargs=copy.deepcopy(kwargs)
        self.DicoDeconvMachine={}
        if self.GD["Image"]["MultiFieldFile"] is None:
            self.DicoDeconvMachine[0]=ClassDeconvMachine.ClassDeconvMachine.__init__(self, *args, **kwargs)
        else:
            Fields=readMultiFieldFile(FName)
            for iField,ThisField in Fields:
                coords = SkyCoord(ra=ThisField["ra"],
                                  dec=ThisField["dec"],
                                  unit=(u.hourangle, u.deg))
                ras=rad2hmsdms(coords.ra.rad,Type="ra").replace(" ",":")
                decs=rad2hmsdms(coords.dec.rad,Type="dec").replace(" ",":")
                NPix=ThisField["NPix"]
                This_kwargs=copy.deepcopy(kwargs)
                This_kwargs["Image"]["PhaseCenterRADEC"]=[ras,decs]
                This_kwargs["Image"]["NPix"]=NPix
                self.DicoDeconvMachine[iField]=ClassDeconvMachine.ClassDeconvMachine.__init__(self, *args, **This_kwargs)
            

    
