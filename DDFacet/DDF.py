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
Parset = ReadCFG.Parset("%s/DefaultParset.cfg" % os.path.dirname(DDFacet.Parset.__file__))




def read_options():
    # OP.OptionGroup("* Parallel", "Parallel")
    # OP.OptionGroup("* Data-related options","VisData")
    # OP.OptionGroup("* Images-related options","Images")
    # OP.OptionGroup("* Caching options","Caching")
    # OP.OptionGroup("* Selection","DataSelection")
    # OP.OptionGroup("* Imager Global parameters","ImagerGlobal")
    # OP.OptionGroup("* Visibility compression parameters","Compression")
    # OP.OptionGroup("* MultiScale Options","MultiScale")
    # OP.OptionGroup("* MultiFrequency Options","MultiFreqs")
    # OP.OptionGroup("* Primary Beam Options","Beam")
    # OP.OptionGroup("* DDE Solutions","DDESolutions")
    # OP.OptionGroup("* Convolution functions","ImagerCF")
    # OP.OptionGroup("* Imager's Mainfacet","ImagerMainFacet")
    # OP.OptionGroup("* GAClean","GAClean")
    # OP.OptionGroup("* Clean","ImagerDeconv")
    # OP.OptionGroup("* Debugging","Debugging")
    # OP.OptionGroup("* Logging","Logging")

    default_values = Parset.value_dict
    attrs = Parset.attr_dict

    desc = """Questions and suggestions: cyril.tasse@obspm.fr"""

    OP = MyOptParse.MyOptParse(usage='Usage: %prog [parset file] <options>', version='%prog version 1.0',
                               description=desc, defaults=default_values, attributes=attrs)

    # create options based on contents of parset
    for section in Parset.sections:
        values = default_values[section]
        # "_Help" value in each section is its documentation string
        OP.OptionGroup(values.get("_Help", section), section)
        for name, value in default_values[section].iteritems():
            OP.add_option(name, value)

    OP.Finalise()
    OP.ReadInput()


    # #optcomplete.autocomplete(opt)

    # options, arguments = opt.parse_args()
    MyPickle.Save(OP, SaveFile)
    return OP



def test():
    options=read_options()


def main(OP=None,messages=[]):
    global IdSharedMem
    IdSharedMem = "ddf.%d."%os.getpid()

    if OP is None:
        OP = MyPickle.Load(SaveFile)

    DicoConfig = OP.DicoConfig

    # determine output image name to make a log file
    ImageName=DicoConfig["Images"]["ImageName"]
    if not ImageName:
        raise ValueError("Output ImageName not specified")

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
            os.system('clear')
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

if __name__ == "__main__":
    #os.system('clear')
    logo.print_logo()

    # ParsetFile = sys.argv[1]
    #
    # TestParset = ReadCFG.Parset(ParsetFile)
    #
    # if TestParset.success==True:
    #     Parset.update(TestParset)
    #     print >>log, ModColor.Str("Successfully read %s parset" % ParsetFile)
    #
    # OP = read_options()
    #
    #
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
        if TestParset.success:
            Parset.update_values(TestParset, newval=False)
            messages.append("Successfully read %s parset"%ParsetFile)
        else:
            OP.ExitWithError("Argument must be a valid parset file. Use -h for help.")
            sys.exit(1)
        # re-read options, since defaults will have been updated by the parset
        OP = read_options()
        # refuse to clobber existing parsets, unless forced from command line
        new_parset = OP.DicoConfig["Images"]["ImageName"] + ".parset"
        if os.path.samefile(ParsetFile, new_parset):
            if OP.DicoConfig["Images"]["Clobber"]:
                print>> log, ModColor.Str("WARNING: will overwrite existing parset, since --Clobber is specified.")
            else:
                print>> log, ModColor.Str("Your ImageName setting is such that the specified parset would be "
                                          "overwritten. Please re-run with the --Clobber option if you're sure "
                                          "this is what you want to do, or set a different --ImageName.")
                sys.exit(1)
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

