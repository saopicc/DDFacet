#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

import optparse
import pickle
import numpy as np
from DDFacet.Other import logger
log=logger.getLogger("MaskDicoModel")

try:
    from DDFacet.Imager.ModModelMachine import GiveModelMachine
except:
    from DDFacet.Imager.ModModelMachine import ClassModModelMachine
from DDFacet.Imager import ClassCasaImage
from pyrap.images import image

SaveName="last_MaskDicoModel.obj"

def read_options():
    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""
    global options
    opt = optparse.OptionParser(usage='Usage: %prog <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--InDicoModel',help='Input DicoModel name [no default]',default='')
    group.add_option('--OutDicoModel',help='Output DicoModel name, default is %default',default=None)
    group.add_option('--MaskName',help='Name of the fits mask, default is %default',default=None)
    group.add_option('--NPixOut',help='Name of the fits mask, default is %default',default=None)
    group.add_option('--FilterNegComp',help='Name of the fits mask, default is %default',type="int",default=0)
    group.add_option('--InvertMask',help='Invert Mask, default is %default',type="int",default=0)
    opt.add_option_group(group)

    options, arguments = opt.parse_args()
    f = open(SaveName,"wb")
    pickle.dump(options,f)
    

    
def mainFromExt(InDicoModel=None,OutDicoModel=None,MaskName=None,NPixOut=None,FilterNegComp=0,InvertMask=0):
    class O:
        def __init__(self,**kwargs):
            for key in kwargs.keys(): setattr(self,key,kwargs[key])
    options=O(InDicoModel=InDicoModel,OutDicoModel=OutDicoModel,MaskName=MaskName,NPixOut=NPixOut,FilterNegComp=FilterNegComp,InvertMask=InvertMask)
    return main(options)

def main(options=None):
    if options==None:
        f = open(SaveName,'rb')
        options = pickle.load(f)


    if options.OutDicoModel is None:
        raise ValueError("--OutDicoModel should be specified")
    ModConstructor = ClassModModelMachine()
    MM=ModConstructor.GiveInitialisedMMFromFile(options.InDicoModel)
    NComp0=len(MM.DicoSMStacked["Comp"].keys())
    
    if options.MaskName:
        MM.CleanMaskedComponants(options.MaskName,InvertMask=options.InvertMask)

    if options.FilterNegComp:
        MM.RemoveNegComponants()

    if options.NPixOut:
        MM.ChangeNPix(int(options.NPixOut))

    NComp1=len(MM.DicoSMStacked["Comp"].keys())
    log.print("Kept %i componants (out of %i)"%(NComp1,NComp0))
    MM.ToFile(options.OutDicoModel)


if __name__=="__main__":
    read_options()
    f = open(SaveName,'rb')
    options = pickle.load(f)

    main(options=options)
