#!/usr/bin/env python

#import matplotlib
#matplotlib.use('agg')


import optparse
SaveFile="last_DDFacet.obj"
import pickle
import os
from DDFacet.Other import logo
from DDFacet.Imager import ClassDeconvMachine
from DDFacet.Other import ModColor
from DDFacet.ToolsDir import ModParset
#import ClassData
#import ClassInitMachine
from DDFacet.Array import NpShared
import numpy as np
from DDFacet.Imager import ClassDeconvMachine
import os
from DDFacet.Other import PrintOptParse
from DDFacet.Parset import ReadCFG
import sys
from DDFacet.Other import MyPickle
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("DDFacet")

from DDFacet.Parset import MyOptParse




global Parset
Parset=ReadCFG.Parset("%s/DDFacet/Parset/DefaultParset.cfg"%os.environ["DDFACET_DIR"])


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
    OP.add_option('MSListFile',help='Input MSs')
    OP.add_option('ColName')
    OP.add_option('TChunkSize')
    OP.add_option('InitDicoModel',help='Image name [%default]')
    OP.add_option('WeightCol')
    
    OP.OptionGroup("* Images-related options","Images")
    OP.add_option('ImageName',help='Image name [%default]',default='DefaultName')
    OP.add_option('PredictModelName',help='Predict Image name [%default]',default='')
    OP.add_option('SaveIms',help='Image name [%default]')


    OP.OptionGroup("* File storing options","Stores")
    OP.add_option('DeleteDDFProducts')
    OP.add_option('PSF')
    OP.add_option('Dirty')
   

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
    OP.add_option("PSFOversize")
    OP.add_option("PSFFacets")

    OP.OptionGroup("* Visibility compression parameters","Compression")
    OP.add_option('CompGridMode')
    OP.add_option('CompGridDecorr')
    OP.add_option('CompGridFOV')
    OP.add_option('CompDeGridMode')
    OP.add_option('CompDeGridDecorr')
    OP.add_option('CompDeGridFOV')


    #OP.add_option('CompModeDeGrid')

    OP.OptionGroup("* MultiScale Options","MultiScale")
    OP.add_option("Scales")
    OP.add_option("Ratios")
    OP.add_option("NTheta")

    OP.OptionGroup("* MultiFrequency Options","MultiFreqs")
    OP.add_option("NFreqBands")
    OP.add_option("Alpha")
    OP.add_option("NChanDegridPerMS")


    OP.OptionGroup("* Primary Beam Options","Beam")
    OP.add_option("BeamModel")
    OP.add_option("LOFARBeamMode")
    OP.add_option("DtBeamMin")
    OP.add_option("BeamNFreqPerMS")
    OP.add_option("CenterNorm")
    OP.add_option("FITSFile")
    OP.add_option("FITSFeed")  # XY or RL

    OP.OptionGroup("* DDE Solutions","DDESolutions")
    OP.add_option("DDSols")
    OP.add_option("JonesMode")
    OP.add_option("GlobalNorm")
    OP.add_option("DDModeGrid")
    OP.add_option("DDModeDeGrid")
    OP.add_option("ScaleAmpGrid")
    OP.add_option("ScaleAmpDeGrid")
    OP.add_option("CalibErr")
    OP.add_option('Type')
    OP.add_option('Scale')
    OP.add_option('gamma')
    OP.add_option("RestoreSub")
    OP.add_option("ReWeightSNR")

    OP.OptionGroup("* Convolution functions","ImagerCF")
    OP.add_option("Support")
    OP.add_option("OverS")
    OP.add_option("wmax")
    OP.add_option("Nw")

    OP.OptionGroup("* Imager's Mainfacet","ImagerMainFacet")
    OP.add_option("NFacets",help="Number of facets, default is %default. ")
    OP.add_option("Npix")
    OP.add_option("Cell")
    OP.add_option("Padding")
    OP.add_option("ConstructMode")

    OP.OptionGroup("* Clean","ImagerDeconv")
    OP.add_option("MaxMajorIter")
    OP.add_option("Gain")
    OP.add_option("SearchMaxAbs")
    OP.add_option("MaxMinorIter")
    OP.add_option("CycleFactor")
    OP.add_option("CleanMaskImage")
    OP.add_option("FluxThreshold")
 
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
    


    
    global IdSharedMem
    IdSharedMem=str(int(os.getpid()))+"."

    ImageName=DicoConfig["Images"]["ImageName"]
    OP.ToParset("%s.parset"%ImageName)

    NpShared.DelAll(IdSharedMem)
    Imager=ClassDeconvMachine.ClassImagerDeconv(GD=DicoConfig,IdSharedMem=IdSharedMem,BaseName=ImageName)

    Imager.Init()
    Mode=DicoConfig["ImagerGlobal"]["Mode"]

    # Imager.testDegrid()
    # stop
    if "Predict" in Mode:
        Imager.GivePredict()

    if "Clean" in Mode:
        Imager.main()
    if "Dirty" in Mode:
        Imager.GiveDirty()
    if "PSF" in Mode:
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
        print>>log, ModColor.Str("DDFacet ended successfully",col="green")
    except:
        print>>log, ModColor.Str("There was a problem, please help yourself",col="red")
        NpShared.DelAll(IdSharedMem)

    # main(options)
    
    
        
    
