#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')


import optparse
SaveFile="last_DDFacet.obj"
import pickle
import os
import logo
import ClassDeconvMachine
import ModColor
import ToolsDir.ModParset
import ClassData
#import ClassInitMachine
import NpShared
import numpy as np
import ClassDeconvMachine
import os
import PrintOptParse
import ReadCFG
import sys
import MyPickle
import MyLogger
import ModColor
log=MyLogger.getLogger("DDFacet")

import MyOptParse




global Parset
Parset=ReadCFG.Parset("%s/Parset/DefaultParset.cfg"%os.environ["DDFACET_DIR"])


def read_options():


    D=Parset.DicoPars

    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""

    OP=MyOptParse.MyOptParse(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc,
                             DefaultDict=D)


    OP.OptionGroup("* Parallel", "Parallel")
    OP.add_option('Enable')
    OP.add_option('NCPU')

    OP.OptionGroup("* Data-related options","VisData")
    OP.add_option('MSName',help='Input MS')
    OP.add_option('ColName')
    OP.add_option('TChunkSize')
    OP.add_option('ImageName',help='Image name [%default]',default='DefaultName')

    OP.OptionGroup("* Selection","DataSelection")
    OP.add_option('FlagAnts')
    OP.add_option('UVRangeKm')
    OP.add_option('DistMaxToCore')

    OP.OptionGroup("* Imager Global parameters","ImagerGlobal")
    OP.add_option('Mode',help='Default %default',default="Clean")
    OP.add_option('PolMode')
    OP.add_option('Precision')
    OP.add_option('Weighting')
    OP.add_option('Robust')

    OP.OptionGroup("* Visibility compression parameters","Compression")
    OP.add_option('CompressModeGrid')
    OP.add_option('CompressDecorr')
    OP.add_option('CompressFOV')
    OP.add_option('CompressModeDeGrid')

    OP.OptionGroup("* DDE Solutions","DDESolutions")
    OP.add_option("DDSols")
    OP.add_option("GlobalNorm")
    OP.add_option("DDModeGrid")
    OP.add_option("DDModeDeGrid")
    OP.add_option("ScaleAmp")
    OP.add_option("CalibErr")
    OP.add_option('Type')
    OP.add_option('Scale')
    OP.add_option('gamma')

    OP.OptionGroup("* Convolution functions","ImagerCF")
    OP.add_option("Support")
    OP.add_option("OverS")
    OP.add_option("wmax")
    OP.add_option("Nw")

    OP.OptionGroup("* Imager's Mainfacet","ImagerMainFacet")
    OP.add_option("NFacets",help="Number of facets, default is %default. ")
    OP.add_option("Npix")
    OP.add_option("Cell")
    OP.add_option("ConstructMode")

    OP.OptionGroup("* Clean","ImagerDeconv")
    OP.add_option("MaxMajorIter")
    OP.add_option("Gain")
    OP.add_option("MaxMinorIter")
    OP.add_option("CycleFactor")

    OP.Finalise()
    OP.ReadInput()
    OP.Print()

    
    # #optcomplete.autocomplete(opt)

    # options, arguments = opt.parse_args()
    MyPickle.Save(OP,SaveFile)
    return OP
    
    

def test():
    options=read_options()


def main(OP=None):
    


    if OP==None:
        OP = MyPickle.Load(SaveFile)

    DicoConfig=OP.DicoConfig


    ImageName=DicoConfig["VisData"]["ImageName"]



    
    global IdSharedMem
    IdSharedMem=str(int(os.getpid()))+"."


    NpShared.DelAll(IdSharedMem)
    Imager=ClassDeconvMachine.ClassImagerDeconv(GD=DicoConfig,IdSharedMem=IdSharedMem,BaseName=ImageName)

    Imager.Init()
    Mode=DicoConfig["ImagerGlobal"]["Mode"]
    if "Clean"==Mode:
        Imager.main()
    elif "Dirty"==Mode:
        Imager.GiveDirty()
    elif "PSF"==Mode:
        Imager.MakePSF()

    NpShared.DelAll(IdSharedMem)

if __name__=="__main__":
    os.system('clear')
    logo.print_logo()


    ParsetFile=sys.argv[1]

    TestParset=ReadCFG.Parset(ParsetFile)
    if TestParset.Success==True:
        #global Parset
        
        Parset=TestParset
        print >>log,ModColor.Str("Successfully read %s parset"%ParsetFile)

    OP=read_options()


    #main(OP)
    try:
        main(OP)
    except:
        NpShared.DelAll(IdSharedMem)

    # main(options)
    
    
        
    
