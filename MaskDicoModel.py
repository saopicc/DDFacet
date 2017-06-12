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
    group.add_option('--OutDicoModel',help='Output DicoModel name, default is %default',default=None)
    group.add_option('--MaskName',help='Name of the fits mask, default is %default',default=None)
    group.add_option('--FilterNegComp',help='Name of the fits mask, default is %default',type="int",default=0)
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
        MM.CleanMaskedComponants(options.MaskName)

    if options.FilterNegComp:
        MM.RemoveNegComponants()


    x,y=MM.DicoSMStacked["Comp"].keys()[0]
    V=MM.DicoSMStacked["Comp"][(x,y)]
    del(MM.DicoSMStacked["Comp"])
    MM.DicoSMStacked["Comp"]={}
    x=y=19845/2
    MM.DicoSMStacked["Comp"][(x,y)]={'Vals': [np.array([ 10.,  0.])]}
    MM.ToFile(options.OutDicoModel)

    # Model=MM.GiveModelImage()
    # Cell=MM.GD["Image"]["Cell"]
    # CasaImage=ClassCasaImage.ClassCasaimage(Model,Model.shape,self.Cell,self.radec)#Lambda=(Lambda0,dLambda,self.NBands))
    # CasaImage.setdata(SmearedModel+self.Residual,CorrT=True)
    # CasaImage.ToFits()
    # CasaImage.setBeam((SmoothFWHM,SmoothFWHM,0))
    # CasaImage.close()

if __name__=="__main__":
    read_options()
    f = open(SaveName,'rb')
    options = pickle.load(f)

    main(options=options)
