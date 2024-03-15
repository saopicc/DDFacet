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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


from pyrap.tables  import table

from DDFacet.compatibility import range

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
SaveFile = "last_DDFacet.obj"
import errno
import re
import sys
import time
import subprocess
import psutil
import numexpr
import numpy as np
from DDFacet.Other import logo
from DDFacet.Array import NpParallel
from DDFacet.Imager import ClassDeconvMachine
from DDFacet.Imager import ClassFacetMachine
from DDFacet.Parset import ReadCFG
from DDFacet.Other import MyPickle
from DDFacet.Parset import MyOptParse
from DDFacet.Other import logger
from DDFacet.Other import ModColor
from DDFacet.Other import Exceptions
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import Multiprocessing
import SkyModel.Other.ModColor   # because it's duplicated there
from DDFacet.Other import progressbar
from DDFacet.Other.AsyncProcessPool import APP, WorkerProcessError
import six
if six.PY3:
    from DDFacet.cbuild.Gridder import _pyArrays3x as _pyArrays
else:
    from DDFacet.cbuild.Gridder import _pyArrays27 as _pyArrays
from DDFacet.report_version import report_version
log = None

import numpy as np
import warnings

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

# hack to avoid recursion depth issues in SSD

sys.setrecursionlimit(10000)

global Parset
Parset = ReadCFG.Parset("%s/DefaultParset.cfg" % os.path.dirname(DDFacet.Parset.__file__))


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
        for name, value in getattr(default_values[section], "iteritems", default_values[section].items)():
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


def main(OP=None, messages=[]):
    if OP is None:
        OP = MyPickle.Load(SaveFile)
        print("Using settings from %s, then command line."%SaveFile)

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
    logger.logToFile(ImageName + ".log", append=DicoConfig["Log"]["Append"])
    global log
    log = logger.getLogger("DDFacet")
    logger.setSilent("app")
    
    # disable colors and progressbars if requested
    ModColor.silent = SkyModel.Other.ModColor.silent = \
                      progressbar.ProgressBar.silent = \
                      DicoConfig["Log"]["Boring"]

    if messages:
        if not DicoConfig["Log"]["Boring"]:
            #os.system('clear')
            logo.print_logo()
        for msg in messages:
            print(msg, file=log)

    print("Checking system configuration:", file=log)
    # check for SHM size
    ram_size = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    shm_stats = os.statvfs('/dev/shm')
    shm_size = shm_stats.f_bsize * shm_stats.f_blocks
    shm_relsize = shm_size / float(ram_size)
    shm_avail = shm_stats.f_bsize * shm_stats.f_bavail / float(ram_size)

    if shm_relsize < 0.6:
        print(ModColor.Str("""WARNING: max shared memory size is only {:.0%} of total RAM size.
            This can cause problems for large imaging jobs. A setting of 90% is recommended for 
            DDFacet and killMS. If your processes keep failing with SIGBUS or "bus error" messages,
            it is most likely for this reason. You can change the memory size by running
                $ sudo mount -o remount,size=90% /dev/shm
            To make the change permanent, edit /etc/defaults/tmps, and add a line saying "SHM_SIZE=90%".
            """.format(shm_relsize)), file=log)
    else:
        print("  Max shared memory size is {:.0%} of total RAM size; {:.0%} currently available".format(shm_relsize, shm_avail), file=log)

    try:
        output = subprocess.check_output(["/sbin/sysctl", "vm.max_map_count"],universal_newlines=True)
    except Exception:
        print(ModColor.Str("""WARNING: /sbin/sysctl vm.max_map_count failed. Unable to check this setting."""), file=log)
        max_map_count = None
    else:
        max_map_count = int(output.strip().rsplit(" ", 1)[-1])

    if max_map_count is not None:
        if max_map_count < 500000:
            print(ModColor.Str("""WARNING: sysctl vm.max_map_count = {}. 
            This may be too little for large DDFacet and killMS jobs. If you get strange "file exists" 
            errors on /dev/shm, them try to bribe, beg or threaten your friendly local sysadmin into 
            setting vm.max_map_count=1000000 in /etc/sysctl.conf.
                """.format(max_map_count)), file=log)
        else:
            print("  sysctl vm.max_map_count = {}".format(max_map_count), file=log)

    # check for memory lock limits
    import resource
    msoft, mhard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if msoft >=0 or mhard >=0:
        print(ModColor.Str("""WARNING: your system has a limit on memory locks configured.
            This may possibly slow down DDFacet performance. You can try removing the limit by running
                $ ulimit -l unlimited
            If this gives an "operation not permitted" error, you can try to bribe, beg or threaten 
            your friendly local sysadmin into doing
                # echo "*        -   memlock     unlimited" >> /etc/security/limits.conf
        """), file=log)


    if DicoConfig["Debug"]["Pdb"] == "always":
        print("--Debug-Pdb=always: unexpected errors will be dropped into pdb", file=log)
        Exceptions.enable_pdb_on_error(ModColor.Str("DDFacet has encountered an unexpected error. Dropping you into pdb for a post-mortem.\n" +
                                           "(This is because you're running with --Debug-Pdb set to 'always'.)"))
    elif DicoConfig["Debug"]["Pdb"] == "auto" and not DicoConfig["Log"]["Boring"]:
        print("--Debug-Pdb=auto and not --Log-Boring: unexpected errors will be dropped into pdb", file=log)
        Exceptions.enable_pdb_on_error(ModColor.Str("DDFacet has encountered an unexpected error. Dropping you into pdb for a post-mortem.\n" +
            "(This is because you're running with --Debug-Pdb set to 'auto' and --Log-Boring is off.)"))

    # print current options
    OP.Print(dest=log)

    # enable memory logging
    logger.enableMemoryLogging(DicoConfig["Log"]["Memory"])

    # get rid of old shm arrays from previous runs
    Multiprocessing.cleanupStaleShm()

    # initialize random seed from config if set, or else from system time
    if DicoConfig["Misc"]["RandomSeed"] is not None:
        DicoConfig["Misc"]["RandomSeed"]=int(DicoConfig["Misc"]["RandomSeed"])
        print("random seed=%d (explicit)" % DicoConfig["Misc"]["RandomSeed"], file=log)
    else:
        DicoConfig["Misc"]["RandomSeed"] = int(time.time())
        print("random seed=%d (automatic)" % DicoConfig["Misc"]["RandomSeed"], file=log)
    np.random.seed(DicoConfig["Misc"]["RandomSeed"])

    # init NCPU for different bits of parallelism
    ncpu = int(DicoConfig["Parallel"]["NCPU"] or psutil.cpu_count())
    DicoConfig["Parallel"]["NCPU"]=ncpu
    _pyArrays.pySetOMPNumThreads(ncpu)
    NpParallel.NCPU_global = ModFFTW.NCPU_global = ncpu
    numexpr.set_num_threads(ncpu)
    print("using up to %d CPUs for parallelism" % ncpu, file=log)

    # write parset
    OP.ToParset("%s.parset"%ImageName)

    Mode = DicoConfig["Output"]["Mode"]

    # init semaphores, as they're needed for weight calculation too
    ClassFacetMachine.ClassFacetMachine.setup_semaphores(DicoConfig)

    # data machine initialized for all cases except PSF-only mode
    # psf machine initialized for all cases except Predict-only mode
    Imager = ClassDeconvMachine.ClassImagerDeconv(GD=DicoConfig,
                                                  BaseName=ImageName,
                                                  predict_only=(Mode == "Predict" or Mode == "Subtract"),
                                                  data=(Mode != "PSF"),
                                                  psf=(Mode != "Predict" and Mode != "Dirty" and Mode != "Subtract"),
                                                  readcol=(Mode != "Predict" and Mode != "PSF"),
                                                  deconvolve=("Clean" in Mode))

    Imager.Init()
        

    # Imager.testDegrid()
    # stop
    if "Predict" in Mode or "Subtract" in Mode:
        Imager.GivePredict()
    if "Clean" in Mode:
        Imager.main()
    elif "Dirty" in Mode:
        sparsify = DicoConfig["Comp"]["Sparsification"]
        if sparsify and isinstance(sparsify, list):
            sparsify = sparsify[0]
        Imager.GiveDirty(psf="PSF" in Mode, sparsify=sparsify)
    elif "PSF" in Mode:
        sparsify = DicoConfig["Comp"]["Sparsification"]
        if sparsify and isinstance(sparsify, list):
            sparsify = sparsify[0]
        Imager.MakePSF(sparsify=sparsify)
    elif "RestoreAndShift" == Mode:
        Imager.RestoreAndShift()

    # # open default viewer, these options should match those in
    # # ClassDeconvMachine if changed:
    # viewer = DicoConfig["Output"]["DefaultImageViewer"]
    # for img in DicoConfig["Output"]["Open"]:
    #     if img == "Dirty":
    #         ret = subprocess.call(
    #             "%s %s.dirty.fits" %
    #             (viewer, DicoConfig["Output"]["Name"]),
    #             shell=True)
    #         if ret:
    #             print>>log, ModColor.Str(
    #                 "\nCan't open dirty image\n", col="yellow")
    #     elif img == "DirtyCorr":
    #         ret = subprocess.call(
    #             "%s %s.dirty.corr.fits" %
    #             (viewer, DicoConfig["Output"]["Name"]),
    #             shell=True)
    #         if ret:
    #             print>>log, ModColor.Str(
    #                 "\nCan't open dirtyCorr image\n", col="yellow")
    #     elif img == "PSF":
    #         ret = subprocess.call(
    #             "%s %s.psf.fits" %
    #             (viewer, DicoConfig["Output"]["Name"]), shell=True)
    #         if ret:
    #             print>>log, ModColor.Str(
    #                 "\nCan't open PSF image\n", col="yellow")
    #     elif img == "Model":
    #         ret = subprocess.call(
    #             "%s %s.model.fits" %
    #             (viewer, DicoConfig["Output"]["Name"]),
    #             shell=True)
    #         if ret:
    #             print>>log, ModColor.Str(
    #                 "\nCan't open model image\n", col="yellow")
    #     elif img == "Residual":
    #         ret = subprocess.call(
    #             "%s %s.residual.fits" %
    #             (viewer, DicoConfig["Output"]["Name"]),
    #             shell=True)
    #         if ret:
    #             print>>log, ModColor.Str(
    #                 "\nCan't open residual image\n", col="yellow")
    #     elif img == "Restored":
    #         ret = subprocess.call(
    #             "%s %s.restored.fits" %
    #             (viewer, DicoConfig["Output"]["Name"]),
    #             shell=True)
    #         if ret:
    #             print>>log, ModColor.Str(
    #                 "\nCan't open restored image\n", col="yellow")
    #     elif img == "Alpha":
    #         ret = subprocess.call(
    #             "%s %s.alpha.fits" %
    #             (viewer, DicoConfig["Output"]["Name"]),
    #             shell=True)
    #         if ret:
    #             print>>log, ModColor.Str(
    #                 "\nCan't open alpha image\n", col="yellow")
    #     elif img == "Norm":
    #         ret = subprocess.call(
    #             "%s %s.Norm.fits" %
    #             (viewer, DicoConfig["Output"]["Name"]),
    #             shell=True)
    #         if ret:
    #             print>>log, ModColor.Str(
    #                 "\nCan't open norm image\n", col="yellow")
    #     elif img == "NormFacets":
    #         ret = subprocess.call(
    #             "%s %s.NormFacets.fits" %
    #             (viewer, DicoConfig["Output"]["Name"]),
    #             shell=True)
    #         if ret:
    #             print>>log, ModColor.Str(
    #                 "\nCan't open normfacets image\n", col="yellow")
    #     else:
    #         print>>log, ModColor.Str(
    #             "\nDon't understand %s, not opening that image\n" %
    #             img, col="yellow")

def driver():
    #warnings.filterwarnings("default", category=DeprecationWarning)
    #os.system('clear')
    #logo.print_logo()

    # work out DDFacet version
    version=report_version()
    traceback_msg = traceback.format_exc()
    atexit.register(Multiprocessing.cleanupShm)

    T = ClassTimeIt.ClassTimeIt()

    # parset should have been read in by now
    OP = read_options()
    args = OP.GiveArguments()
    
    DicoConfig = OP.DicoConfig
    #if DicoConfig["Misc"]["IgnoreDeprecationMarking"]:
    #    warnings.filterwarnings("always", category=DeprecationWarning, module=r"DDFacet[.\w]*|__main__|SkyModel[.\w]*")
    #    warnings.filterwarnings("always", category=DeprecationWarning, module=r"DDFacet[.\w]*|__main__|SkyModel[.\w]*")
    #else:
    #    warnings.filterwarnings("error", category=DeprecationWarning, module=r"DDFacet[.\w]*|__main__|SkyModel[.\w]*")
    #    warnings.filterwarnings("error", category=DeprecationWarning, module=r"DDFacet[.\w]*|__main__|SkyModel[.\w]*")
    if six.PY2:
        warnings.warn("Python 2 has reached end of life and is no longer supported. "
                    "You can continue running the software in Python 2 (along with other deprecated modes) by setting "
                    "--Misc-IgnoreDeprecationMarking in your parset. No further Python 2 related bug fixes or support is available.",
                    DeprecationWarning)

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
                print(ModColor.Str("WARNING: will overwrite existing parset, since --Output-Clobber is specified."), file=log)
            else:
                print(ModColor.Str("Your --Output-Name setting is the same as the base name of the parset, which would\n"
                                          "mean overwriting the parset. I'm sorry, Dave, I'm afraid I can't do that.\n"
                                          "Please re-run with the --Output-Clobber option if you're sure this is what\n"
                                          "you want to do, or set a different --Output-Name."), file=log)
                sys.exit(1)
    elif len(args):
        print(args)
        OP.ExitWithError("Incorrect number of arguments. Use -h for help.")
        sys.exit(1)

    retcode = report_error = 0

    try:
        main(OP, messages)
        print(ModColor.Str(
            "DDFacet ended successfully after %s" %
            T.timehms(), col="green"), file=log)
    except KeyboardInterrupt:
        print(traceback.format_exc(), file=log)
        print(ModColor.Str("DDFacet interrupted by Ctrl+C", col="red"), file=log)
        APP.terminate()
        retcode = 1 #Should at least give the command line an indication of failure
    except Exceptions.UserInputError:
        print(ModColor.Str(sys.exc_info()[1], col="red"), file=log)
        print(ModColor.Str("There was a problem with some user input. See messages above for an indication."), file=log)
        APP.terminate()
        retcode = 1  # Should at least give the command line an indication of failure
    except WorkerProcessError:
        print(ModColor.Str("A worker process has died on us unexpectedly. This probably indicates a bug:"), file=log)
        print(ModColor.Str("  the original underlying error may be reported in the log [possibly far] above."), file=log)
        report_error = True
    except:
        if sys.exc_info()[0]!=WorkerProcessError and Exceptions.is_pdb_enabled():
            APP.terminate()
            raise
        else:
            print(traceback.format_exc(), file=log)
        report_error = True

    if report_error:
        logfileName = logger.getLogFilename()
        logfileName = logfileName if logfileName is not None else "[file logging is not enabled]"
        print("", file=log)
        print(ModColor.Str(
            "There was a problem after %s; if you think this is a bug please open an issue, "%
            T.timehms(), col = "red"), file=log)
        print(ModColor.Str("  quote your version of DDFacet and attach your logfile.", col="red"), file=log)
        print(ModColor.Str(
            "You are using DDFacet revision: %s" %
            version, col="red"), file=log)
        print(ModColor.Str(
            "Your logfile is available here: %s" %
            logfileName, col="red"), file=log)
        # print>>log, traceback_msg
        # Should at least give the command line an indication of failure
        APP.terminate()
        retcode = 1 # Should at least give the command line an indication of failure

    APP.shutdown()
    Multiprocessing.cleanupShm()
    sys.exit(retcode)

if __name__ == "__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()
