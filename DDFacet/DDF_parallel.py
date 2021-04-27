#!/usr/bin/env python
'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''
import os
# somewhere some library is trying to be too clever for its own good
# we will set our own settings up later on
os.environ["OMP_NUM_THREADS"] = "1"

# dirty hack to fix what seems to be an easy install problem
# see https://stackoverflow.com/questions/5984523/eggs-in-path-before-pythonpath-environment-variable

# import sys,os
# sys.path = os.environ["PYTHONPATH"].split(":") + sys.path
import sys,os
if "PYTHONPATH_FIRST" in os.environ.keys() and int(os.environ["PYTHONPATH_FIRST"]):
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


#import matplotlib
# matplotlib.use('agg')
import optparse
import traceback
import atexit
SaveFile = "last_DDFacet_parallel.obj"
import errno
import re
import sys
import time
import subprocess
import psutil
import numexpr
import numpy as np
from DDFacet.Parset import ReadCFG
from DDFacet.Other import MyPickle
from DDFacet.Parset import MyOptParse
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
from DDFacet.report_version import report_version
import DDFacet.Array.shared_dict
global log
log = None

import numpy as np

import paramiko
import getpass
import copy

# # ##############################
# # Catch numpy warning
# np.seterr(all='raise')
# import warnings
# warnings.filterwarnings('error')
# #with warnings.catch_warnings():
# #    warnings.filterwarnings('error')
# # ##############################

'''
The defaults for all the user commandline arguments are stored in a parset configuration file
called DefaultParset.cfg. When you add a new option you must specify meaningful defaults in
there.

These options can be overridden by specifying a subset of the parset options in a user parset file
passed as the first commandline argument. These options will override the corresponding defaults.
'''
import DDFacet
print("DDFacet version is",report_version())
print("Using python package located at: " + os.path.dirname(DDFacet.__file__))
print("Using driver file located at: " + __file__)
global Parset
Parset = ReadCFG.Parset("%s/DefaultParset.cfg" % os.path.dirname(DDFacet.Parset.__file__))

import logging
logging.getLogger("paramiko").setLevel(logging.WARNING)

def read_options():

    default_values = Parset.value_dict
    attrs = Parset.attr_dict

    desc = """Questions and suggestions: cyril.tasse@obspm.fr"""

    OP = MyOptParse.MyOptParse(usage='Usage: %prog [parset file] <options>', version='%prog version '+report_version(),
                               description=desc, defaults=default_values, attributes=attrs)

    # create options based on contents of parset
    for section in Parset.sections:
        values = default_values[section]
        # "_Help" value in each section is its documentation string
        OP.OptionGroup(values.get("_Help", section), section)
        for name, value in default_values[section].iteritems():
            if not attrs[section][name].get("no_cmdline"):
                OP.add_option(name, value)


    OP.Finalise()
    OP.ReadInput()

    # #optcomplete.autocomplete(opt)

    # options, arguments = opt.parse_args()
    MyPickle.Save(OP, SaveFile)
    return OP


def test():
    options = read_options()
    


def GetMSListSet(mslist,WorkDir=None):
    ListMS=[s.strip() for s in open(mslist,"r").readlines()]

    DicoNodes={}
    for MSName in ListMS:
        if ":" in MSName:
            Node,MSPath=MSName.split(":")
        else:
            Node,MSPath="localhost",MSName

        MSPath=os.path.abspath(MSPath)
        if not(Node in DicoNodes.keys()):
            DicoNodes[Node]={"ListMS":[MSPath]}
        else:
            DicoNodes[Node]["ListMS"].append(MSPath)
    
    for NodeName in DicoNodes.keys():
        ThisListName="mslist.%s.txt"%NodeName
        if WorkDir:
            ThisListName="%s/%s"%(WorkDir,ThisListName)
        f=open(ThisListName,"w")
        for MSName in DicoNodes[NodeName]["ListMS"]:
            f.write("%s\n"%MSName)
        f.close()
        DicoNodes[NodeName]["NameListMS"]=ThisListName

    return DicoNodes
            
class ParamikoPool():
    def __init__(self,WorkDir):
        self.JobPool={}
        self.WorkDir=WorkDir
        pass
    
    def AppendCommand(self,JobName,Str,NodeName=None,CheckFile=None):
    
        if NodeName is None:
            os.system(Str)
        else:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            #p = getpass.getpass()
            ssh.connect(NodeName,
                        username='tasse',
                        #password=p,
                        key_filename='/home/tasse/.ssh/id_rsa')

            S="source /media/tasse/data/Wirtinger_Pack/init.sh; cd %s; %s"%(self.WorkDir,Str)
            log.print(ModColor.Str("[%s] %s"%(NodeName,S),col="blue"))
            stdin, stdout, stderr = ssh.exec_command(S)
#            stdin, stdout, stderr = ssh.exec_command("source /media/tasse/data/Wirtinger_Pack/init.sh; %s"%(Str))
            #print stdout.readlines()
            self.JobPool[JobName]={"ssh":ssh,
                                   "stdin":stdin,
                                   "stdout":stdout,
                                   "stderr":stderr,
                                   "CheckFile":CheckFile}
        
    def WaitJob(self,JobName):
        if "*" in JobName:
            JobPrefix=JobName.replace("*","")
        else:
            JobPrefix=JobName
            
        for Job in self.JobPool.keys():
            if JobPrefix in Job:
                log.print(ModColor.Str("Waiting for %s..."%Job,col="blue"))
                #self.JobPool[Job]["stdout"].channel.recv_exit_status()
                STDOUT=self.JobPool[Job]["stdout"].read()
                STDERR=self.JobPool[Job]["stderr"].read()
                STDERROUT=STDOUT+STDERR
                Cond0=("DDFacet has encountered an unexpected error" in STDERROUT)
                Cond1=("There was a problem after" in STDERROUT)
                if Cond0 or Cond1:
                    log.print(ModColor.Str("DDFacet produced an error"))
                    log.print(STDERROUT)
                    raise RuntimeError("DDFacet crashed")
                else:
                    log.print(ModColor.Str("  Job %s finished sucessfully"%Job,col="green"))
                    
                self.JobPool[Job]["ssh"].close()
                log.print("   [done] %s"%Job)
                del(self.JobPool[Job])


def main():
    DDFp=DDFParallel()
    DDFp.main()
    
class DDFParallel():
    def __init__(self,OP=None, messages=[]):
        if OP is None:
            OP = MyPickle.Load(SaveFile)
            print("Using settings from %s, then command line."%SaveFile)
        self.OP=OP
        self.messages=messages
        self.GD=OP.DicoConfig
        ImageName = self.GD["Output"]["Name"]
        self.WorkDir=os.getcwd()
        self.WorkDirNodes=os.path.abspath("%s.logNodes"%ImageName)
        os.system("mkdir -p %s"%self.WorkDirNodes)
        self.PP=ParamikoPool(self.WorkDir)
        

    def main(self):
        OP=self.OP
        messages=self.messages
        
    
        MainOutputName = ImageName = self.GD["Output"]["Name"]
        
    
        # print current options
        OP.Print(dest=log)
    
        # write parset
        ParsetName="%s.parset"%ImageName
        OP.ToParset(ParsetName)
        ParsetName=os.path.abspath(ParsetName)
        
        Mode = self.GD["Output"]["Mode"]
        
        DicoNodes=GetMSListSet(self.GD["Data"]["MS"],
                               WorkDir=self.WorkDirNodes)
        # Compute residual image_i on all nodes_i
        DicoModelName=None
        MotherNode = self.GD["Parallel"]["MotherNode"]
        NMajorCycle= self.GD["Deconv"]["MaxMajorIter"]
        PP=self.PP

        for iMajorCycle in range(NMajorCycle):

            # #########################################################
            # Compute the residual PSF and Dirty cubes in parallel
            CacheList=[]
            for ThisNodeName in DicoNodes.keys():
                ThisNameOut="%s/%s_%s"%(self.WorkDirNodes,MainOutputName,ThisNodeName)
                ThisMSlist=DicoNodes[ThisNodeName]
                if DicoModelName is not None:
                    DicoModelName=os.path.abspath(DicoModelName)
                Str="DDF.py %s --Output-Mode=Clean --Deconv-MaxMajorIter 0 --Data-MS %s "\
                    " --Output-Name=%s --Predict-InitDicoModel %s --Debug-Pdb=never"%(ParsetName,
                                                                                      DicoNodes[ThisNodeName]["NameListMS"],
                                                                                      ThisNameOut,
                                                                                      str(DicoModelName))
                DicoNodes[ThisNodeName]["Cache"]="%s.ddfcache"%DicoNodes[ThisNodeName]["NameListMS"]
                PP.AppendCommand("ComputeResidual_%s"%ThisNodeName,Str,NodeName=ThisNodeName)
            PP.WaitJob("ComputeResidual_*")

            # #########################################################
            # # Compute the average redidual from the (redidual_i, all iNodes), and create a fake <mslist.ddfcache>
            self.StackNodesImages(DicoNodes)
                
            # #########################################################
            # Make DDF to think that the cache is valid, or force it to use  <mslist.ddfcache>, and run a minor cycle on it, and generate
            ThisCycleName="%s_MinorCycle_%i"%(MainOutputName,iMajorCycle)
            PP.AppendCommand("CleanMinor",
                             "DDF.py %s --Output-Mode=CleanMinor --Cache-Reset 0 --Cache-Dirty forceresidual"\
                             " --Cache-PSF force --Output-Name %s --Debug-Pdb=never"%(ParsetName,ThisCycleName),
                             NodeName=MotherNode)
            PP.WaitJob("CleanMinor")

            DicoModelName="%s.DicoModel"%ThisCycleName

    def StackNodesImages(self,DicoNodes):

        # ############################################
        # Reading cache of the various PSF and Dirty images from the various nodes
        for ThisNodeName in DicoNodes.keys():
            if "DicoPSF" not in DicoNodes[ThisNodeName].keys():
                DicoNodes[ThisNodeName]["DicoPSF"]={}
                DicoNodes[ThisNodeName]["DicoPSF"]=DDFacet.Array.shared_dict.create("DicoPSF_%s"%ThisNodeName)
                ThisCache="%s/PSF"%(DicoNodes[ThisNodeName]["Cache"])
                log.print("Reading PSF cache of [%s] %s"%(ThisNodeName,ThisCache))
                DicoNodes[ThisNodeName]["DicoPSF"].restore(ThisCache)
                LastCacheName_PSF=ThisCache
                
            DicoNodes[ThisNodeName]["DicoDirty"]={}
            DicoNodes[ThisNodeName]["DicoDirty"]=DDFacet.Array.shared_dict.create("DicoDirty_%s"%ThisNodeName)
            ThisCache="%s/Dirty"%(DicoNodes[ThisNodeName]["Cache"])
            log.print("Reading Dirty cache of [%s] %s"%(ThisNodeName,ThisCache))
            DicoNodes[ThisNodeName]["DicoDirty"].restore(ThisCache)
            LastCacheName_Dirty=ThisCache

        # ############################################
        # Creating cache where we will store the results
        # for the Dirty image / cube
        log.print("Computing the average of the residual images...")
        DicoStackDirty=DDFacet.Array.shared_dict.create("DicoStackDirty_%s"%ThisNodeName)
        DicoStackDirty.restore(LastCacheName_Dirty)
        Cube=DicoStackDirty["ImageCube"]
        nch,npol,nx,ny=Cube.shape
        SumWeight=DicoStackDirty["SumWeights"].reshape((nch,1,1,1))
        SumWeight.fill(0)
        Cube.fill(0)
        
        # Now doing the averaging
        for ThisNodeName in DicoNodes.keys():
            Cube_i=DicoNodes[ThisNodeName]["DicoDirty"]["ImageCube"]
            SumWeights_i=DicoNodes[ThisNodeName]["DicoDirty"]["SumWeights"].reshape((nch,1,1,1))
            Cube_i *= SumWeights_i 
            Cube += Cube_i
            SumWeight += SumWeights_i
            for ich in range(nch):
                DicoStackDirty["freqs"][ich]=DicoStackDirty["freqs"][ich]+DicoNodes[ThisNodeName]["DicoDirty"]["freqs"][ich]
                
        for ich in range(nch):
            DicoStackDirty["freqs"][ich]=sorted(list(set(DicoStackDirty["freqs"][ich])))
            
        Cube /= SumWeight.reshape((nch,1,1,1))
        WBAND=SumWeight/np.sum(SumWeight)
        MeanCube = np.sum(Cube * WBAND.reshape((nch,1,1,1)), axis=0).reshape((1, npol, nx, ny))
        DicoStackDirty["WeightChansImages"].flat[:]=WBAND.flat[:]
        MainCache="%s.ddfcache"%self.GD["Data"]["MS"]
        os.system("mkdir -p %s"%MainCache)
        DirtyCacheName="%s/Dirty"%MainCache
        log.print("Saving the cache of the average residual image in %s"%DirtyCacheName)
        DicoStackDirty.save(DirtyCacheName)

        DicoStackDirty.save("HackyParallel.Dirty.ddfcache")

        # ############################################
        # Creating cache where we will store the results
        # for the PSF image / cube
        DicoStackPSF=DDFacet.Array.shared_dict.create("DicoStackPSF_%s"%ThisNodeName)
        DicoStackPSF.restore(LastCacheName_PSF)

        Cube=DicoStackPSF["ImageCube"]
        nch,npol,nx,ny=Cube.shape
        SumWeight=DicoStackPSF["SumWeights"].reshape((nch,1,1,1))
        SumWeight.fill(0)
        Cube.fill(0)
        CubeVariablePSF=DicoStackPSF['CubeVariablePSF']
        CubeVariablePSF.fill(0)
        NFacet,nch,npol,nx_facet,ny_facet=CubeVariablePSF.shape

        # ['freqs', 'ImageInfo', 'SumWeights', 'WeightChansImages', 'Facets', 'CubeVariablePSF', 'CubeMeanVariablePSF', 'CentralFacet', 'PeakNormed_CubeVariablePSF', 'PeakNormed_CubeMeanVariablePSF', 'MeanFacetPSF', 'OutImShape', 'CellSizeRad', 'MeanJonesBand', 'SumJonesChan', 'ChanMappingGrid', 'ChanMappingGridChan', 'ImageCube', 'MeanImage', 'FacetNorm', 'JonesNorm', 'FWHMBeam', 'PSFGaussPars', 'PSFSidelobes', 'EstimatesAvgPSF']
        
        # Now doing the averaging
        for ThisNodeName in DicoNodes.keys():
            Cube_i=DicoNodes[ThisNodeName]["DicoPSF"]["ImageCube"]
            SumWeights_i=DicoNodes[ThisNodeName]["DicoPSF"]["SumWeights"].reshape((nch,1,1,1))
            Cube_i *= SumWeights_i 
            Cube += Cube_i
            SumWeight += SumWeights_i
            for iFacet in range(NFacet):
                ThisCubeFacetPSF=DicoNodes[ThisNodeName]["DicoPSF"]["CubeVariablePSF"][iFacet]
                ThisCubeFacetPSF=ThisCubeFacetPSF*SumWeights_i
                CubeVariablePSF+=ThisCubeFacetPSF
                
            for ich in range(nch):
                DicoStackPSF["freqs"][ich]=DicoStackPSF["freqs"][ich]+DicoNodes[ThisNodeName]["DicoPSF"]["freqs"][ich]
            
        for ich in range(nch):
            DicoStackPSF["freqs"][ich]=sorted(list(set(DicoStackPSF["freqs"][ich])))
            
        Cube /= SumWeight.reshape((nch,1,1,1))
        WBAND=SumWeight/np.sum(SumWeight)
        MeanCube = np.sum(Cube * WBAND.reshape((nch,1,1,1)), axis=0).reshape((1, npol, nx, ny))
        CubeVariablePSF/= SumWeight.reshape((1,nch,1,1,1))
        MeanCubeVariablePSF= np.sum(CubeVariablePSF * WBAND.reshape((1,nch,1,1,1)), axis=1).reshape((NFacet, 1, npol, nx_facet, ny_facet))
        DicoStackPSF["WeightChansImages"].flat[:]=WBAND.flat[:]
        
        MainCache="%s.ddfcache"%self.GD["Data"]["MS"]
        os.system("mkdir -p %s"%MainCache)
        PSFCacheName="%s/PSF"%MainCache
        log.print("Saving the cache of the average residual image in %s"%PSFCacheName)
        DicoStackPSF.save(PSFCacheName)

        
        
##########################################################################
##########################################################################
##########################################################################
        
if __name__ == "__main__":
    # work out DDFacet version
    version=report_version()
    # parset should have been read in by now
    OP = read_options()
    args = OP.GiveArguments()

    DicoConfig = OP.DicoConfig
    
    
    
    ImageName = DicoConfig["Output"]["Name"]
    if not ImageName:
        raise Exceptions.UserInputError("--Output-Name not specified, can't continue.")
    if not DicoConfig["Data"]["MS"]:
        raise Exceptions.UserInputError("--Data-MS not specified, can't continue.")

    # create directory if it exists
    dirname = os.path.dirname(ImageName)
    if not os.path.exists(dirname) and not dirname == "":
        os.mkdir(dirname)

    # setup logging
    MyLogger.logToFile(ImageName + ".log", append=DicoConfig["Log"]["Append"])
    global log
    log = MyLogger.getLogger("DDFacet-Parallel")

    # collect messages in a list here because I don't want to log them until the logging system
    # is set up in main()
    messages = ["starting DDFacet (%s)" % " ".join(sys.argv),
                "   version is %s"%version,
                "   working directory is %s" % os.getcwd()]

    # single argument is a parset to read
    if len(args) == 1:
        ParsetFile = args[0]
        TestParset = ReadCFG.Parset(ParsetFile)
        if TestParset.success:
            Parset.update_values(TestParset, newval=False)
            if TestParset.migrated is not None:
                messages.append(ModColor.Str("WARNING: parset %s is of a deprecated version %.1f"%(ParsetFile, TestParset.migrated)))
                messages.append(ModColor.Str("We have migrated the parset to the current version (%.1f) automatically,"%(TestParset.version)))
                messages.append(ModColor.Str("but please check the settings below to make sure they're correct."))
            else:
                messages.append("Successfully read parset %s, version %.1f"%(ParsetFile, TestParset.version))
        else:
            OP.ExitWithError(
                "Argument must be a valid parset file. Use -h for help.")
            sys.exit(1)
        # re-read options, since defaults will have been updated by the parset
        OP = read_options()
        # refuse to clobber existing parsets, unless forced from command line
        new_parset = OP.DicoConfig["Output"]["Name"] + ".parset"
        if os.path.exists(new_parset) and os.path.samefile(ParsetFile, new_parset):
            if OP.DicoConfig["Output"]["Clobber"]:
                log.print(ModColor.Str("WARNING: will overwrite existing parset, since --Output-Clobber is specified."))
            else:
                log.print(ModColor.Str("Your --Output-Name setting is the same as the base name of the parset, which would\n"
                          "mean overwriting the parset. I'm sorry, Dave, I'm afraid I can't do that.\n"
                          "Please re-run with the --Output-Clobber option if you're sure this is what\n"
                          "you want to do, or set a different --Output-Name."))
                sys.exit(1)
    elif len(args):
        print(args)
        OP.ExitWithError("Incorrect number of arguments. Use -h for help.")
        sys.exit(1)

    retcode = report_error = 0

    DDFp=DDFParallel(OP, messages)
    DDFp.main()
