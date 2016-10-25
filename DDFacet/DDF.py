#!/usr/bin/env python

#import matplotlib
#matplotlib.use('agg')
import traceback
SaveFile="last_DDFacet.obj"
from DDFacet.Other import logo
from DDFacet.Array import NpShared
from DDFacet.Imager import ClassDeconvMachine
import os, errno, re, sys
from DDFacet.Parset import ReadCFG
from DDFacet.Other import MyPickle
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
from DDFacet.Other import ClassTimeIt
import SkyModel.Other.ModColor   # because it's duplicated there
from DDFacet.Other import progressbar
log = None

from DDFacet.Parset import MyOptParse
import subprocess

# # ##############################
# # Catch numpy warning
# np.seterr(all='raise')
# import warnings
# with warnings.catch_warnings():
#     warnings.filterwarnings('error')
# # ##############################

'''
The defaults for all the user commandline arguments are stored in a parset configuration file
called DefaultParset.cfg. When you add a new option you must specify meaningful defaults in
there.

These options can be overridden by specifying a subset of the parset options in a user parset file
passed as the first commandline argument. These options will override the corresponding defaults.
'''
import DDFacet
print "Using python package located at: " + os.path.dirname(DDFacet.__file__)
print "Using driver file located at: " + __file__
global Parset
Parset= ReadCFG.Parset("%s/DefaultParset.cfg" % os.path.dirname(DDFacet.Parset.__file__))


def read_options():


    D=Parset.DicoPars

    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""

    OP= MyOptParse.MyOptParse(usage='Usage: %prog [parset file] <options>', version='%prog version 1.0', description=desc,
                              DefaultDict=D)
    '''
    These options will be read from command line arguments you can specify parset options
    in Default.parset.

    A default value here will override any user parset value, so set the defaults
    with caution, as it is counter-intuitive.

    TODO: These options should be created automatically.
    '''
    OP.OptionGroup("* Parallel", "Parallel")
    OP.add_option('Enable')
    OP.add_option('NCPU')

    OP.OptionGroup("* Data-related options","VisData")
    OP.add_option('MSName',help='Input MS')
    OP.add_option('MSListFile',help='Input MSs')
    OP.add_option('ColName')
    OP.add_option('ChunkHours')
    OP.add_option('InitDicoModel',help='Image name [%default]')
    OP.add_option('WeightCol')
    OP.add_option('PredictColName')
    OP.OptionGroup("* Images-related options","Images")
    OP.add_option('ImageName',help='Image name [%default]')

    OP.add_option('PredictModelName',help='Predict Image name [%default]')
    OP.add_option('AllowColumnOverwrite', help='Whether to overwrite existing column or not [%default]')
    OP.add_option('SaveIms',help='')
    OP.add_option('SaveImages',help='')
    OP.add_option('SaveOnly',help='')
    OP.add_option('SaveCubes',help='')
    OP.add_option('OpenImages',
                  help="Opens images after exiting successfully."
                       "List, accepts any combination of: "
                       "'Dirty','DirtyCorr','PSF','Model','Residual',"
                       "'Restored','Alpha','Norm','NormFacets'.")
    OP.add_option('DefaultImageViewer', help="Default image viewer")
#    OP.add_option('MultiFreqMap', help="Outputs multi-frequency cube (NFreqBands) instead of average map")



    OP.OptionGroup("* Caching options","Caching")
    OP.add_option('ResetCache')
    OP.add_option('ResetPSF')
    OP.add_option('ResetDirty')
    OP.add_option('ResetSmoothBeam')
    OP.add_option('CachePSF')
    OP.add_option('CacheDirty')


    OP.OptionGroup("* Selection","DataSelection")
    OP.add_option('Field')
    OP.add_option('DDID')
    OP.add_option('TaQL')
    OP.add_option('ChanStart')
    OP.add_option('ChanEnd')
    OP.add_option('ChanStep')
    OP.add_option('FlagAnts')
    OP.add_option('UVRangeKm')
    OP.add_option('DistMaxToCore')

    OP.OptionGroup("* Imager Global parameters","ImagerGlobal")
    OP.add_option('Mode',help='Default %default')
    OP.add_option('PredictMode')
    OP.add_option('PolMode')
    OP.add_option('Precision')
    OP.add_option('Weighting')
    OP.add_option('MFSWeighting')
    OP.add_option('Robust')
    OP.add_option('Super')
    OP.add_option("PSFOversize")
    OP.add_option("PSFFacets")
    OP.add_option("PhaseCenterRADEC")
    OP.add_option('FFTMachine')
    OP.add_option('GriderType')
    OP.add_option('DeGriderType')

    OP.OptionGroup("* Visibility compression parameters","Compression")
    OP.add_option('CompGridDecorr')
    OP.add_option('CompGridFOV')
    OP.add_option('CompDeGridDecorr')
    OP.add_option('CompDeGridFOV')


    #OP.add_option('CompModeDeGrid')

    OP.OptionGroup("* MultiScale Options","MultiScale")
    OP.add_option("Scales")
    OP.add_option("Ratios")
    OP.add_option("NTheta")
    OP.add_option("PSFBox")
    OP.add_option("SolverMode")

    OP.OptionGroup("* MultiFrequency Options","MultiFreqs")
    OP.add_option("NFreqBands")
    OP.add_option("Alpha")
    OP.add_option("NChanDegridPerMS")
    OP.add_option("GridBandMHz")
    OP.add_option("DegridBandMHz")


    OP.OptionGroup("* Primary Beam Options","Beam")
    OP.add_option("BeamModel")
    OP.add_option("LOFARBeamMode")
    OP.add_option("DtBeamMin")
    OP.add_option("NChanBeamPerMS")
    OP.add_option("CenterNorm")
    OP.add_option("FITSFile")
    OP.add_option("FITSFeed")
    OP.add_option("FITSLAxis")
    OP.add_option("FITSMAxis")
    OP.add_option("FITSVerbosity")

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
    OP.add_option("DecorrMode")


    OP.OptionGroup("* Convolution functions","ImagerCF")
    OP.add_option("Support")
    OP.add_option("OverS")
    OP.add_option("wmax")
    OP.add_option("Nw")

    OP.OptionGroup("* Imager's Mainfacet","ImagerMainFacet")
    OP.add_option("NFacets",help="Number of facets, default is %default. ")
    OP.add_option("DiamMaxFacet")
    OP.add_option("DiamMinFacet")
    OP.add_option("Npix")
    OP.add_option("Cell")
    OP.add_option("Padding")
    OP.add_option("ConstructMode")
    OP.add_option("Circumcision")

    OP.OptionGroup("* GAClean","GAClean")
    OP.add_option("GASolvePars")
    OP.add_option("GACostFunc")
    OP.add_option("BICFactor")
    OP.add_option("NSourceKin")
    OP.add_option("NMaxGen")
    OP.add_option("NEnlargePars")
    OP.add_option("NEnlargeData")
    OP.add_option("ArtifactRobust")
    OP.add_option("ConvFFTSwitch")


    OP.OptionGroup("* Clean","ImagerDeconv")
    OP.add_option("MaxMajorIter")
    OP.add_option("Gain")
    OP.add_option("SearchMaxAbs")
    OP.add_option("MaxMinorIter")
    OP.add_option("CleanMaskImage")
    OP.add_option("FluxThreshold")
    OP.add_option("CycleFactor")
    OP.add_option("PeakFactor")
    OP.add_option("RMSFactor")
    OP.add_option("SidelobeSearchWindow")
    OP.add_option("RestoringBeam")
    OP.add_option("MinorCycleMode")

    OP.OptionGroup("* Debugging","Debugging")
    OP.add_option("SaveIntermediateDirtyImages")
    OP.add_option("PauseGridWorkers")
    OP.add_option("FacetPhaseShift")
    OP.add_option("DumpCleanSolutions")
    OP.add_option("PrintMinorCycleRMS")
    OP.add_option("CleanStallThreshold")

    OP.OptionGroup("* Logging","Logging")
    OP.add_option("MemoryLogging")
    OP.add_option("Boring")
    OP.add_option("AppendLogFile")

    OP.Finalise()
    OP.ReadInput()


    # #optcomplete.autocomplete(opt)

    # options, arguments = opt.parse_args()
    MyPickle.Save(OP, SaveFile)
    return OP



def test():
    options=read_options()


def main(OP=None,messages=[]):
    if OP is None:
        OP = MyPickle.Load(SaveFile)

    DicoConfig=OP.DicoConfig

    # determine output image name to make a log file
    ImageName=DicoConfig["Images"]["ImageName"]
    # create directory if it exists
    dirname = os.path.dirname(ImageName)
    if not os.path.exists(dirname) and not dirname == "":
        os.mkdir(dirname)

    # setup logging
    MyLogger.logToFile(ImageName + ".log", append=DicoConfig["Logging"]["AppendLogFile"])
    global log
    log = MyLogger.getLogger("DDFacet")

    # disable colors and progressbars if requested
    ModColor.silent = SkyModel.Other.ModColor.silent = progressbar.ProgressBar.silent = DicoConfig["Logging"]["Boring"]

    if messages:
        if not DicoConfig["Logging"]["Boring"]:
            #os.system('clear') # don't trash scrollback on xterm!
            logo.print_logo()
        for msg in messages:
            print>>log,msg

    # print current options
    OP.Print(dest=log)

    # enable memory logging
    MyLogger.enableMemoryLogging(DicoConfig["Logging"]["MemoryLogging"])

    # If we're using Montblanc for the Predict, we need to use a remote
    # tensorflow server as tensorflow is not fork safe
    # http://stackoverflow.com/questions/37874838/forking-a-python-process-after-loading-tensorflow
    # If a TensorFlowServerTarget is not specified, fork a child process containing one.
    if DicoConfig["ImagerGlobal"]["PredictMode"] == "Montblanc":
        if not DicoConfig["Montblanc"]["TensorflowServerTarget"]:
            from DDFacet.TensorFlowServerFork import fork_tensorflow_server
            DicoConfig["Montblanc"]["TensorflowServerTarget"] = fork_tensorflow_server()

    global IdSharedMem
    IdSharedMem = "ddf.%d."%os.getpid()
    # check for stale shared memory
    uid = os.getuid()
    # list of all files in /dev/shm/ matching ddf.PID.* and belonging to us
    shmlist = [ (filename, re.match('ddf\.([0-9]+)\..*',filename)) for filename in os.listdir("/dev/shm/")
                if os.stat("/dev/shm/"+filename).st_uid == uid ]
    # convert to list of filename,pid tuples
    shmlist = [ (filename, int(match.group(1))) for filename, match in shmlist if match ]
    # now check all PIDs to find dead ones
    # if we get ESRC error from sending signal 0 to the process, it's not running, so we mark it as dead
    dead_pids = set()
    for pid in set([x[1] for x in shmlist]):
        try:
            os.kill(pid, 0)
        except OSError, err:
            if err.errno == errno.ESRCH:
                dead_pids.add(pid)
    # ok, make list of candidates for deletion
    victims = [ filename for filename,pid in shmlist if pid in dead_pids ]
    if victims:
        print>>log, "reaping %d shared memory objects associated with %d dead DDFacet processes"%(len(victims), len(dead_pids))
        for filename in victims:
            os.unlink("/dev/shm/"+filename)

    # write parset
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

    #open default viewer, these options should match those in ClassDeconvMachine if changed:
    viewer = DicoConfig["Images"]["DefaultImageViewer"]
    for img in DicoConfig["Images"]["OpenImages"]:
        if img == "Dirty":
            ret = subprocess.call("%s %s.dirty.fits" % (viewer,DicoConfig["Images"]["ImageName"]), shell=True)
            if ret:
                print>>log, ModColor.Str("\nCan't open dirty image\n", col="yellow")
        elif img == "DirtyCorr":
            ret = subprocess.call("%s %s.dirty.corr.fits" % (viewer,DicoConfig["Images"]["ImageName"]), shell=True)
            if ret:
                print>>log, ModColor.Str("\nCan't open dirtyCorr image\n", col="yellow")
        elif img == "PSF":
            ret = subprocess.call("%s %s.psf.fits" % (viewer,DicoConfig["Images"]["ImageName"]), shell=True)
            if ret:
                print>>log, ModColor.Str("\nCan't open PSF image\n", col="yellow")
        elif img == "Model":
            ret = subprocess.call("%s %s.model.fits" % (viewer,DicoConfig["Images"]["ImageName"]), shell=True)
            if ret:
                print>>log, ModColor.Str("\nCan't open model image\n", col="yellow")
        elif img == "Residual":
            ret = subprocess.call("%s %s.residual.fits" % (viewer,DicoConfig["Images"]["ImageName"]), shell=True)
            if ret:
                print>>log, ModColor.Str("\nCan't open residual image\n", col="yellow")
        elif img == "Restored":
            ret = subprocess.call("%s %s.restored.fits" % (viewer,DicoConfig["Images"]["ImageName"]), shell=True)
            if ret:
                print>>log, ModColor.Str("\nCan't open restored image\n", col="yellow")
        elif img == "Alpha":
            ret = subprocess.call("%s %s.alpha.fits" % (viewer,DicoConfig["Images"]["ImageName"]), shell=True)
            if ret:
                print>>log, ModColor.Str("\nCan't open alpha image\n", col="yellow")
        elif img == "Norm":
            ret = subprocess.call("%s %s.Norm.fits" % (viewer,DicoConfig["Images"]["ImageName"]), shell=True)
            if ret:
                print>>log, ModColor.Str("\nCan't open norm image\n", col="yellow")
        elif img == "NormFacets":
            ret = subprocess.call("%s %s.NormFacets.fits" % (viewer,DicoConfig["Images"]["ImageName"]), shell=True)
            if ret:
                print>>log, ModColor.Str("\nCan't open normfacets image\n", col="yellow")
        else:
            print>>log, ModColor.Str("\nDon't understand %s, not opening that image\n" % img, col="yellow")

    NpShared.DelAll(IdSharedMem)

if __name__=="__main__":
    #os.system('clear')
    #logo.print_logo()

    ParsetFile=sys.argv[1]

    TestParset= ReadCFG.Parset(ParsetFile)

    if TestParset.Success==True:
        #global Parset

        Parset.update(TestParset)
        print >>log, ModColor.Str("Successfully read %s parset" % ParsetFile)

    OP=read_options()


    #main(OP)
    T = ClassTimeIt.ClassTimeIt()

    # parset should have been read in by now
    OP = read_options()
    args = OP.GiveArguments()

    # collect messages in a list here because I don't want to log them until the logging system
    # is set up in main()
    messages = [ "starting DDFacet (%s)"%" ".join(sys.argv),
                 "working directory is %s"%os.getcwd() ]

    # single argument is a parset to read
    if len(args) == 1:
        ParsetFile = args[0]
        TestParset = ReadCFG.Parset(ParsetFile)
        if TestParset.Success==True:
            Parset.update(TestParset)
            messages.append("Successfully read %s parset"%ParsetFile)
        else:
            OP.ExitWithError("Argument must be a valid parset file. Use -h for help.")
            sys.exit(1)
        # re-read options, since defaults will have been updated by the parset
        OP = read_options()
    elif len(args):
        OP.ExitWithError("Incorrect number of arguments. Use -h for help.")
        sys.exit(1)

    retcode = 0
    try:
        main(OP,messages)
        print>>log, ModColor.Str("DDFacet ended successfully after %s"%T.timehms(),col="green")
    except KeyboardInterrupt:
        print>>log, traceback.format_exc()
        print>>log, ModColor.Str("DDFacet interrupted by Ctrl+C", col="red")
        retcode = 1 #Should at least give the command line an indication of failure
    except:
        print>>log, traceback.format_exc()
        ddfacetPath = "." if os.path.dirname(__file__) == "" else os.path.dirname(__file__)
        traceback_msg = traceback.format_exc()
        try:
            commitSha = subprocess.check_output("git -C %s rev-parse HEAD" % ddfacetPath,shell=True)
        except subprocess.CalledProcessError:
            import DDFacet.version as version
            commitSha = version.__version__

        logfileName = MyLogger.getLogFilename()
        logfileName = logfileName if logfileName is not None else "[file logging is not enabled]"
        print>>log, ModColor.Str("There was a problem after %s, if you think this is a bug please open an "
                                  "issue, quote your version of DDFacet and attach your logfile" % T.timehms(), col="red")
        print>>log, ModColor.Str("You are using DDFacet revision: %s" % commitSha, col="red")
        print>>log, ModColor.Str("Your logfile is available here: %s" % logfileName, col="red")
        print>>log, traceback_msg
        retcode = 1 #Should at least give the command line an indication of failure

    NpShared.DelAll(IdSharedMem)
    sys.exit(retcode)


