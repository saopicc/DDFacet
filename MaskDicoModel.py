#!/usr/bin/env python

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
    group.add_option('--OutDicoModel',help='Output DicoModel name [no default]',default='')
    group.add_option('--MaskName',help='Name of the fits mask [no default]',default='')
    opt.add_option_group(group)

    options, arguments = opt.parse_args()
    f = open(SaveName,"wb")
    pickle.dump(options,f)

def main(options=None):
    if options==None:
        f = open(SaveName,'rb')
        options = pickle.load(f)

    ModConstructor = ClassModModelMachine()
    MM=ModConstructor.GiveInitialisedMMFromFile(options.InDicoModel)
    MM.CleanMaskedComponants(options.MaskName)
    MM.ToFile(options.OutDicoModel)

if __name__=="__main__":
    read_options()
    f = open(SaveName,'rb')
    options = pickle.load(f)

    main(options=options)
