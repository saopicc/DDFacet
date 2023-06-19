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
#from DDFacet.Other.AsyncProcessPool import APP
import DDFacet.Other.AsyncProcessPool as AsyncProcessPool
from DDFacet.Data import ClassVisServer
from DDFacet.Data import ClassMS

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
            Fields=None
            self.DicoDeconvMachine[0]=ClassDeconvMachine.ClassImagerDeconv(*args, **kwargs)
            self.NFields=1
        else:
            Fields=readMultiFieldFile(self.GD["Image"]["MultiFieldFile"])
            self.GD["Image"]["PhaseCenterRADEC"]=None
            for iField,ThisField in enumerate(Fields):
                coords = SkyCoord(ra=ThisField["ra"],
                                  dec=ThisField["dec"],
                                  unit=(u.hourangle, u.deg))
                ras=rad2hmsdms(coords.ra.rad,Type="ra").replace(" ",":")
                decs=rad2hmsdms(coords.dec.rad,Type="dec").replace(" ",":")
                NPix=int(ThisField["NPix"])
                ThisGD=copy.deepcopy(self.GD)
                #ThisGD["Image"]["PhaseCenterRADEC"]=[ras,decs]
                ThisGD["Image"]["NPix"]=NPix
                This_kwargs=copy.deepcopy(kwargs)
                This_kwargs["GD"]=ThisGD
                #This_kwargs["BaseName"]="%s_Field%i"%(kwargs["BaseName"],iField)
                This_kwargs["DicoField"]={"FieldID":iField,
                                          "ra0dec0":(coords.ra.rad,coords.dec.rad)}
                self.Fields=Fields
                self.DicoDeconvMachine[iField]=ClassDeconvMachine.ClassImagerDeconv(*args, **This_kwargs)
            self.NFields=len(Fields)
            
        # all internal state initialized -- start the worker threads
        AsyncProcessPool.init(ncpu=self.GD["Parallel"]["NCPU"],
                              affinity=self.GD["Parallel"]["Affinity"],
                              parent_affinity=self.GD["Parallel"]["MainProcessAffinity"],
                              verbose=self.GD["Debug"]["APPVerbose"],
                              pause_on_start=self.GD["Debug"]["PauseWorkers"])

        self.DM0=self.DicoDeconvMachine[0]
        mslist = ClassMS.expandMSList(self.GD["Data"]["MS"],
                                      defaultDDID=self.GD["Selection"]["DDID"],
                                      defaultField=self.GD["Selection"]["Field"],
                                      defaultColumn=None)
        
        self.VS = ClassVisServer.ClassVisServer(mslist,
                                                ColName=self.GD["Data"]["ColName"] if self.DM0.do_readcol else None,
                                                TChunkSize=self.GD["Data"]["ChunkHours"],
                                                GD=self.GD)
        
        for iField in range(self.NFields):
            self.DicoDeconvMachine[iField].Init(self.VS)
        
        AsyncProcessPool.APP.startWorkers()
        self.DicoDeconvMachine[0].VS.CalcWeightsBackground()
        for iField in self.DicoDeconvMachine.keys():
            self.DicoDeconvMachine[iField].InitCF()
    
    def GiveDirty(self, *args, **kwargs):
        for iField in range(self.NFields):
            self.DicoDeconvMachine[iField].GiveDirty(*args,**kwargs)
        
