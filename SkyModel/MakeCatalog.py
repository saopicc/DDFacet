#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
import sys,os
if "PYTHONPATH_FIRST" in list(os.environ.keys()) and int(os.environ["PYTHONPATH_FIRST"]):
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path
import numpy as np
from DDFacet.Other import logger
log=logger.getLogger("MakeCatalog")
#import pyfits
import optparse
import bdsf
import psutil
import pickle
import os
SaveFile="MakeCatalog.last"
from DDFacet.Other import ModColor

def read_options():
    desc="""Make catalog"""
    
    opt = optparse.OptionParser(usage='Usage: %prog <options>',version='%prog version 1.0',description=desc)
    
    group = optparse.OptionGroup(opt, "* Data selection options")
    group.add_option('--RestoredIm',type=str,help='',default=None)
    group.add_option('--bdsm_thresh_isl',type='float',help='',default=10.)
    group.add_option('--bdsm_thresh_pix',type='float',help='',default=10.)
    group.add_option('--bdsm_rms_box',type=str,help='',default='30,10')
    group.add_option('--rmsmean_map',type=str,help='',default='')
    group.add_option('--Parallel',type=int,help='',default=1)
    group.add_option('--NCPU',type=int,help='',default=0)
    
    opt.add_option_group(group)
    options, arguments = opt.parse_args()
    f = open(SaveFile,"wb")
    pickle.dump(options,f)
    return options

        
#########################################

def main(options=None):
    if options is None:
        f = open("last_param.obj",'rb')
        options = pickle.load(f)

    MakeCatMachine=MakeCatalog(**options.__dict__)
    MakeCatMachine.MakeCatalog()


class MakeCatalog():
    def __init__(self,**kwargs):
        for key, value in list(kwargs.items()): setattr(self, key, value)
        

        if self.NCPU==0:
            self.NCPU=psutil.cpu_count()-1

        self.bdsm_rms_box=[int(i) for i in self.bdsm_rms_box.split(",")]
        Files=[self.RestoredIm]
        for f in Files:
            if not os.path.isfile(f):
                raise ValueError("File %s does not exist"%f)
            print(ModColor.Str("File %s exist"%f,col="green"), file=log)

    def Exec(self,ss):
        print(ModColor.Str("Executing:"), file=log)
        print(ModColor.Str("   %s"%ss), file=log)
        os.system(ss)

    def MakeCatalog(self):
        self.RunBDSM(self.RestoredIm)
        
    def RunBDSM(self,filename,rmsmean_map_filename=None,WriteNoise=True,Ratio=None):
        CatName=filename[:-5]+".pybdsm.gaul.fits"
        if os.path.isfile(CatName):
            print(ModColor.Str("File %s exists"%CatName,col="green"), file=log)
            print(ModColor.Str("   Skipping...",col="green"), file=log)
            return

        rmsmean_map_filename=[]
        if self.rmsmean_map:
            l=self.rmsmean_map
            l=l.replace("[","")
            l=l.replace("]","")
            l=l.split(",")
            rmsmean_map_filename=l
        img=bdsf.process_image(filename,
                               thresh_isl=self.bdsm_thresh_isl,
                               thresh_pix=self.bdsm_thresh_pix,
                               rms_box=self.bdsm_rms_box,
                               rmsmean_map_filename=rmsmean_map_filename)
        img.write_catalog(catalog_type="srl",clobber=True,format="fits")
        img.write_catalog(catalog_type="gaul",clobber=True,format="fits")
        img.write_catalog(catalog_type="gaul",clobber=True,format="bbs")
        if WriteNoise:
            img.export_image(img_type="rms",clobber=True)
            img.export_image(img_type="mean",clobber=True)


    def killWorkers(self):
        print("Killing workers", file=log)
        APP.terminate()
        APP.shutdown()
        Multiprocessing.cleanupShm()

def driver():
    OP=read_options()
    main(OP)

if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()
