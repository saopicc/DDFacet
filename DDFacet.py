#!/usr/bin/env python

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

def read_options():
    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--Parset',help='Input MS to draw [no default]',default=None)
    group.add_option('--Mode',help='Default %default',default="Clean")
    opt.add_option_group(group)

    group = optparse.OptionGroup(opt, "* Selection")
    group.add_option('--NCPU',default=None)
    group.add_option('--TChunkSize',default=None)
    opt.add_option_group(group)

    group = optparse.OptionGroup(opt, "* Settings")
    group.add_option('--ms',help='Input MS',default=None)
    group.add_option('--ImageName',help='Image name [%default]',default='DefaultName')
    group.add_option('--ColName',default=None)
    group.add_option('--FlagAntBL',default=None)
    group.add_option('--UVRangeKm',default=None)
    group.add_option('--PolMode',default=None)
    group.add_option("--wmax",default=None)
    group.add_option("--Nw",default=None)
    group.add_option("--DDSols",default=None)
    opt.add_option_group(group)

    group = optparse.OptionGroup(opt, "* Imaging")
    group.add_option('--Robust',default=None)
    group.add_option("--NFacets",default=None)
    group.add_option("--Npix",default=None)
    group.add_option("--Cell",default=None)
    group.add_option("--ConstructMode",default=None)
    opt.add_option_group(group)

    group = optparse.OptionGroup(opt, "* Clean")
    group.add_option("--MaxMajorIter",default=None)
    group.add_option("--Gain",default=None)
    group.add_option("--MaxMinorIter",default=None)
    opt.add_option_group(group)

    #optcomplete.autocomplete(opt)

    options, arguments = opt.parse_args()
    
    f = open(SaveFile,"wb")
    pickle.dump(options,f)
    return options

def main(options=None):
    
    logo.print_logo()

    if options==None:
        f = open(SaveFile,'rb')
        options = pickle.load(f)
    
    ParsetFile=options.Parset
    if ParsetFile==None:
        print "Give a base parset name"

    ImageName=options.ImageName

    ReplaceDico={}
    #if options.ms!=None:
    #    ReplaceDico['Files'] = {'FileMSCat': {'Name': [options.ms]}}

    ListOpts=[]
    ReplaceDico={}
    if options.NCPU!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Parallel.NCPU = %s\n"%options.NCPU)
    if options.ms!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Files.FileMSCat.Name = [%s]\n"%options.ms)
    if options.DDSols!=None:
        ToolsDir.ModParset.StrToDict(ReplaceDico,"Files.killMSSolutionFile = %s\n"%options.DDSols)
    if options.ColName!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Files.ColName = %s\n"%options.ColName)
    if options.FlagAntBL!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Select.FlagAntBL = %s\n"%options.FlagAntBL)
    if options.UVRangeKm!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Select.UVRangeKm = %s\n"%options.UVRangeKm)
    if options.Robust!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Facet.Robust = %s\n"%options.Robust)
    if options.TChunkSize!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Facet.TChunkSize = %s\n"%options.TChunkSize)
    if options.NFacets!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Facet.MainFacetOptions.NFacets = %s\n"%options.NFacets)
    if options.wmax!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Facet.MainFacetOptions.wmax = %s\n"%options.wmax)
    if options.Nw!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Facet.MainFacetOptions.Nw = %s\n"%options.Nw)
    if options.Npix!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Facet.MainFacetOptions.Npix = %s\n"%options.Npix)
    if options.Cell!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Facet.MainFacetOptions.Cell = %s\n"%options.Cell)
    if options.MaxMajorIter!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Facet.MajorCycleOptions.MaxMajorIter = %s\n"%options.MaxMajorIter)
    if options.Gain!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Facet.MinorCycleOptions.Gain = %s\n"%options.Gain)
    if options.MaxMinorIter!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Facet.MinorCycleOptions.MaxMinorIter = %s\n"%options.MaxMinorIter)
    if options.ConstructMode!=None: ToolsDir.ModParset.StrToDict(ReplaceDico,"Facet.ConstructMode = %s\n"%options.ConstructMode)


    
    GD=ClassData.ClassGlobalData(ParsetFile,ReplaceDico=ReplaceDico)
    global IdSharedMem
    IdSharedMem=str(int(os.getpid()))+"."

    SolsFile=GD.DicoConfig["Files"]["killMSSolutionFile"]
    if SolsFile!=None:
        DicoSolsFile=np.load(SolsFile)
        DicoSols={}
        DicoSols["t0"]=DicoSolsFile["Sols"]["t0"]
        DicoSols["t1"]=DicoSolsFile["Sols"]["t1"]
        nt,na,nd,_,_=DicoSolsFile["Sols"]["G"].shape
        G=np.swapaxes(DicoSolsFile["Sols"]["G"],1,2).reshape((nt,nd,na,1,2,2))
        DicoSols["Jones"]=G
        NpShared.DicoToShared("%skillMSSolutionFile"%IdSharedMem,DicoSols)
        D=NpShared.SharedToDico("%skillMSSolutionFile"%IdSharedMem)
        ClusterCat=DicoSolsFile["ClusterCat"]
        ClusterCat=ClusterCat.view(np.recarray)
        DicoClusterDirs={}
        DicoClusterDirs["l"]=ClusterCat.l
        DicoClusterDirs["m"]=ClusterCat.m
        DicoClusterDirs["I"]=ClusterCat.SumI
        DicoClusterDirs["Cluster"]=ClusterCat.Cluster
        
        _D=NpShared.DicoToShared("%sDicoClusterDirs"%IdSharedMem,DicoClusterDirs)

    NpShared.DelAll(IdSharedMem)
    Imager=ClassDeconvMachine.ClassImagerDeconv(GD=GD,IdSharedMem=IdSharedMem,BaseName=ImageName)


    Imager.Init()
    if "Clean"==options.Mode:
        Imager.main()
    elif "Dirty"==options.Mode:
        Imager.GiveDirty()
    elif "PSF"==options.Mode:
        Imager.MakePSF()

    NpShared.DelAll(IdSharedMem)

if __name__=="__main__":
    options=read_options()
    os.system('clear')

    try:
        main(options)
    except:
        NpShared.DelAll(IdSharedMem)

    # main(options)
    
    
        
    
