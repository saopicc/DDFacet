#!/usr/bin/env python
import numpy as np
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("MakeCatalog")
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
        for key, value in kwargs.items(): setattr(self, key, value)
        

        if self.NCPU==0:
            self.NCPU=psutil.cpu_count()-1

        self.bdsm_rms_box=[int(i) for i in self.bdsm_rms_box.split(",")]
        Files=[self.RestoredIm]
        for f in Files:
            if not os.path.isfile(f):
                raise ValueError("File %s does not exist"%f)
            print>>log,ModColor.Str("File %s exist"%f,col="green")

    def Exec(self,ss):
        print>>log,ModColor.Str("Executing:")
        print>>log,ModColor.Str("   %s"%ss)
        os.system(ss)

    def MakeCatalog(self):
        self.RunBDSM(self.RestoredIm)
        
    def RunBDSM(self,filename,rmsmean_map_filename=None,WriteNoise=True,Ratio=None):
        CatName=filename[:-5]+".pybdsm.gaul.fits"
        if os.path.isfile(CatName):
            print>>log,ModColor.Str("File %s exists"%CatName,col="green")
            print>>log,ModColor.Str("   Skipping...",col="green")
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
        if WriteNoise:
            img.export_image(img_type="rms",clobber=True)
            img.export_image(img_type="mean",clobber=True)


    def killWorkers(self):
        print>>log, "Killing workers"
        APP.terminate()
        APP.shutdown()
        Multiprocessing.cleanupShm()


if __name__=="__main__":
    OP=read_options()
    main(OP)
