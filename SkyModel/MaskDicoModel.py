#!/usr/bin/env python
from __future__ import division, absolute_import, print_function

import optparse
import pickle
import numpy as np
from SkyModel.Other import MyLogger
log=MyLogger.getLogger("MaskDicoModel")

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

def main(options=None):
    if options==None:
        f = open(SaveName,'rb')
        options = pickle.load(f)


    if options.OutDicoModel is None:
        raise ValueError("--OutDicoModel should be specified")
    ModConstructor = ClassModModelMachine()
    MM=ModConstructor.GiveInitialisedMMFromFile(options.InDicoModel)
    if options.MaskName:
        MM.CleanMaskedComponants(options.MaskName,InvertMask=options.InvertMask)
        
    if options.FilterNegComp:
        MM.RemoveNegComponants()

    if options.NPixOut:
        MM.ChangeNPix(int(options.NPixOut))

    MM.ToFile(options.OutDicoModel)

def driver():
    read_options()
    f = open(SaveName,'rb')
    options = pickle.load(f)

    main(options=options)

if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()
