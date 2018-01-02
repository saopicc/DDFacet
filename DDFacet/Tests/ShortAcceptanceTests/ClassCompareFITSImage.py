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

import subprocess
import unittest
import os
from os import path, getenv
import re
import numpy as np
from DDFacet.Parset.ReadCFG import Parset
from astropy.io import fits
from subprocess import Popen
import time

class ClassCompareFITSImage(unittest.TestCase):
    """ Automated assurance test: reference FITS file regression (abstract class)
        Input:
            Accepts a set of reference images (defined in define_image_list),
            input measurement set and parset file.

            The parset file can be a subset of available
            options since DDFacet reads all values from the default parset. The
            overrides should specify an input measurement set or set of measurement
            sets.

            !!!Note: each measurement set should only be the name of the measurement
            !!!set since the tests must be portable. The names are automatically updated
            !!!to be relative to the data directory specified via environment variable
            !!!below.

            The parset should contain all the options that are to be tested in t
            he test case, but should not specify the output
            image file prefix, as this is overridden according to the filename
            rules below. Use setParsetOption to set a configuration option.

            Input filename convention:
                1. Input files directory: Set environment variable "DDFACET_TEST_DATA_DIR"
                                        (defaults to current directory)
                2. Reference image name: [TestClassName].[ImageIdentifier].fits
                3. Parset file: [TestClassName].parset.cfg

        Output filename convention:
            1. Output files directory: Set environment variable "DDFACET_TEST_OUTPUT_DIR"
                                        (defaults to tmp)
            2. Run-produced image name: [TestClassName].run.[ImageIdentifier].fits
            3. DDFacet logfiles: [TestClassName].run.out.log and [TestClassName].run.err.log
            4. Parset with default overrides, including image prefixes: [TestClassName].run.parset.conf
            5. Diff map as computed with fitstool.py with filename [TestClassName].diff.[ImageIdentifier].fits

        Tests cases:
            Tests the output of DDFacet against the reference images. Currently
            we are only testing the following:
                1. max (ref-output)^2 <= tolerance
                2. Mean Squared Error <= tolerance
                3. Parse the logs of the reference images and compare runtime against current test case runtime
    """

    @classmethod
    def defineImageList(cls):
        """ Method to define set of reference images to be tested.
            Can be overridden to add additional output products to the test.
            These must correspond to whatever is used in writing out the FITS files (eg. those in ClassDeconvMachine.py)
            Returns:
                List of image identifiers to reference and output products
        """
        return ['dirty', 'dirty.corr', 'psf', 'NormFacets', 'Norm',
                'int.residual', 'app.residual', 'int.model', 'app.model',
                'int.convmodel', 'app.convmodel', 'int.restored', 'app.restored',
                'restored']

    @classmethod
    def defineMaxSquaredError(cls):
        """ Method defining maximum error tolerance between any pair of corresponding
            pixels in the output and corresponding reference FITS images.
            Should be overridden if another tolerance is desired
            Returns:
                constant for maximum tolerance used in test case setup
        """
        return [1e-6,1e-6,1e-6,1e-6,1e-6,
                1e-3,1e-4,1e-3,1e-4,
                1e-3,1e-4,1e-3,1e-4,
                1e-1] #epsilons per image pair, as listed in defineImageList

    @classmethod
    def defMeanSquaredErrorLevel(cls):
        """ Method defining maximum tolerance for the mean squared error between any
            pair of FITS images. Should be overridden if another tolerance is
            desired
            Returns:
            constant for tolerance on mean squared error
        """
        return [1e-7,1e-7,1e-7,1e-7,1e-7,
                1e-5,1e-5,1e-5,1e-5,
                1e-5,1e-5,1e-5,1e-5,
                1e-5] #epsilons per image pair, as listed in defineImageList

    @classmethod
    def defExecutionTime(cls):
        """
        Relative tolerance for total execution time in comparison with reference runs
        """
        return 0.1 # 10%

    @classmethod
    def defMinorCycleTolerance(cls):
        """
        Relative tolerance on minor cycle count
        """
        return 25  # +/- 25 minor cycles

    @classmethod
    def defMajorCycleTolerance(cls):
        """
        Relative tolerance on minor cycle count
        """
        return 1  # +/- 0 major cycles

    @classmethod
    def setParsetOption(cls, section, option, value):
        """
            Sets the default option read by the configuration parser
            args:
                section: Configuration [section] name
                option: Section option name
                value: Value for option (refer to default parset for documentation)
        """
        cls._defaultParset.set(section, option, value)

    @classmethod
    def timeoutsecs(cls):
        return 21600

    @classmethod
    def setUpClass(cls):
        unittest.TestCase.setUpClass()
        cls._inputDir = getenv('DDFACET_TEST_DATA_DIR','./')+"/"
        cls._outputDir =  getenv('DDFACET_TEST_OUTPUT_DIR','/tmp/')+"/"
        cls._refHDUList = []
        cls._outHDUList = []

        #Read and override default parset
        cls._inputParsetFilename = cls._inputDir + cls.__name__+ ".parset.cfg"
        cls._outputParsetFilename = cls._outputDir + cls.__name__+ ".run.parset.cfg"
        cls._inputLog = cls._inputDir + cls.__name__+ ".log"
        cls._outputLog = cls._outputDir + cls.__name__ + ".run.log"
        if not path.isfile(cls._inputParsetFilename):
            raise RuntimeError("Default parset file %s does not exist" % cls._inputParsetFilename)
        p = Parset(cls._inputParsetFilename)
        cls._defaultParset = p
        cls._imagePrefix = cls._outputDir+cls.__name__+".run"
        cls.setParsetOption("Output",
                            "Name",
                            cls._imagePrefix)
        # set up path to each ms relative to environment variable
        if type(p.DicoPars["Data"]["MS"]) is list:
            #MSdir can actually contain strange directives to select fields and DDIDs... so may appear invalid
            #for ms in p.DicoPars["Data"]["MS"]:
                #if path.dirname(ms) != "":
                #    raise RuntimeError("Expected only measurement set name, "
                #                       "not relative or absolute path in %s" % ms)
            abs_ms = [cls._inputDir+ms for ms in p.DicoPars["Data"]["MS"]]

            cls.setParsetOption("Data", "MS", "["+(",".join(abs_ms))+"]")
        else:
            ms = p.DicoPars["Data"]["MS"]
            #MSdir can actually contain strange directives to select fields and DDIDs... so may appear invalid
            #if path.dirname(ms) != "":
            #   raise RuntimeError("Expected only measurement set name, "
            #                        "not relative or absolute path in %s" % ms)
            abs_ms = cls._inputDir+ms
            cls.setParsetOption("Data", "MS", abs_ms)

        fOutputParset = open(cls._outputParsetFilename,mode='w')
        cls._defaultParset.write(fOutputParset)
        fOutputParset.close()

        #Build dictionary of HDUs
        for ref_id in cls.defineImageList():
            fname = cls._inputDir+cls.__name__+"."+ref_id+".fits"
            if not path.isfile(fname):
                raise RuntimeError("Reference image %s does not exist" % fname)
            fitsHDU = fits.open(fname)
            cls._refHDUList.append(fitsHDU)

        #Setup test constants
        cls._maxSqErr = cls.defineMaxSquaredError()
        cls._thresholdMSE = cls.defMeanSquaredErrorLevel()


        #Run DDFacet with desired setup. Crash the test if DDFacet gives a non-zero exit code:
        cls._stdoutLogFile = cls._outputDir+cls.__name__+".run.out.log"
        cls._stderrLogFile = cls._outputDir+cls.__name__+".run.err.log"

        args = ['DDF.py',
            cls._outputParsetFilename,
            '--Debug-APPVerbose=2',
            '--Output-Name=%s' % cls._imagePrefix]

        stdout_file = open(cls._stdoutLogFile, 'w')
        stderr_file = open(cls._stderrLogFile, 'w')

        with stdout_file, stderr_file:
            p = Popen(args, 
                      env=os.environ.copy(),
                      stdout=stdout_file, 
                      stderr=stderr_file)
            x = cls.timeoutsecs()
            delay = 1.0
            timeout = int(x / delay)
            while p.poll() is None and timeout > 0:
                time.sleep(delay)
                timeout -= delay
            #timeout reached, kill process if it is still rolling
            ret = p.poll()
            if ret is None:
                p.kill()
                ret = 99

            if ret == 99:
                raise RuntimeError("Test timeout reached. Killed process.")
            elif ret != 0:
                raise RuntimeError("DDF exited with non-zero return code %d" % ret)


        #Finally open up output FITS files for testing and build a dictionary of them
        for ref_id in cls.defineImageList():
            fname = cls._outputDir+cls.__name__+".run."+ref_id+".fits"
            if not path.isfile(fname):
                raise RuntimeError("Reference image %s does not exist" % fname)
            fitsHDU = fits.open(fname)
            cls._outHDUList.append(fitsHDU)

        #Save diffmaps for later use:
        for ref_id in cls.defineImageList():
            fname = cls._inputDir + cls.__name__ + "." + ref_id + ".fits"
            compfname = cls._outputDir + cls.__name__ + ".run." + ref_id + ".fits"
            difffname = cls._outputDir + cls.__name__ + ".diff." + ref_id + ".fits"
            args = ["fitstool.py", "-f", "--diff", "--output", difffname, fname, compfname]
            subprocess.check_call(args, env=os.environ.copy())

    @classmethod
    def tearDownClass(cls):
        unittest.TestCase.tearDownClass()
        for fitsfile in cls._refHDUList:
            fitsfile.close()
        for fitsfile in cls._outHDUList:
            fitsfile.close()

    def setUp(self):
        unittest.TestCase.setUp(self)

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    '''
    Test cases:
    '''
    def testMaxSquaredError(self):
        cls = self.__class__
        list_except = []
        for imgI, (ref, out) in enumerate(zip(cls._refHDUList, cls._outHDUList)):
            imgIdentity = cls.defineImageList()[imgI] + " image"
            for ref_hdu, out_hdu in zip(ref, out):
                try:
                    if ref_hdu.data is None:
                        assert out_hdu.data is None, "ref_hdu data is None, so out_hdu must be None in %s" % imgIdentity
                    else:
                        assert out_hdu.data.shape == ref_hdu.data.shape, "ref_hdu data shape doesn't match out_hdu"
                    assert np.all((ref_hdu.data - out_hdu.data)**2 <= cls._maxSqErr[imgI]), "FITS data not the same for %s" % \
                                                                                            imgIdentity
                except AssertionError, e:
                    list_except.append(str(e))
                    continue
        if len(list_except) != 0:
            msg = "\n".join(list_except)
            raise AssertionError("The following assertions failed:\n %s" % msg)

    def testMeanSquaredError(self):
        cls = self.__class__
        list_except = []
        for imgI, (ref, out) in enumerate(zip(cls._refHDUList, cls._outHDUList)):
                imgIdentity = cls.defineImageList()[imgI] + " image"
                for ref_hdu, out_hdu in zip(ref, out):
                    try:
                        if ref_hdu.data is None:
                            assert out_hdu.data is None, "ref_hdu data is None, so out_hdu must be None in %s" % imgIdentity
                        else:
                            assert out_hdu.data.shape == ref_hdu.data.shape, "ref_hdu data shape doesn't match out_hdu"
                        assert np.mean((ref_hdu.data - out_hdu.data)**2) <= cls._thresholdMSE[imgI], "MSE of FITS data not the same for %s" % \
                                                             imgIdentity
                    except AssertionError, e:
                        list_except.append(str(e))
                    continue
        if len(list_except) != 0:
            msg = "\n".join(list_except)
            raise AssertionError("The following assertions failed:\n %s" % msg)

    def testPerformanceRegression(self):
        cls = self.__class__
        assert path.isfile(cls._inputLog), "Reference log file %s does not exist" % cls._inputLog
        assert path.isfile(cls._outputLog), "Test run log file %s does not exist" % cls._inputLog
        with open(cls._inputLog) as f:
            vals = None
            logtext = f.readline()
            while logtext:
                vals = re.match(r".*DDFacet ended successfully after (?P<mins>[0-9]+)?m(?P<secs>[0-9]+.[0-9]+)?s",
                                logtext)
                if vals is not None:
                    break
                logtext = f.readline()
            assert vals is not None, "Could not find the successful termination string in reference log... " \
                                     "have you changed something?"
            assert vals.group("mins") is not None, "Minutes not found in reference log"
            assert vals.group("secs") is not None, "Seconds not found in reference log"
            reftime = float(vals.group("mins")) * 60.0 + float(vals.group("secs"))
        with open(cls._outputLog) as f:
            assert vals is not None, "Could not find the successful termination string in test run log... " \
                                     "have you changed something?"
            assert vals.group("mins") is not None, "Minutes not found in test run log"
            assert vals.group("secs") is not None, "Seconds not found in test run log"
            testruntime = float(vals.group("mins")) * 60.0 + float(vals.group("secs"))
        assert testruntime / reftime <= 1.0 + cls.defExecutionTime(), "Runtime for this test was significantly " \
                                                                      "longer than reference run. Check the logs."

    def testMajorMinorCycleCount(self):
        cls = self.__class__
        assert path.isfile(cls._inputLog), "Reference log file %s does not exist" % cls._inputLog
        assert path.isfile(cls._outputLog), "Test run log file %s does not exist" % cls._inputLog
        input_major = 0
        input_minor = 0
        output_major = 0
        output_minor = 0
        with open(cls._inputLog) as f:
            vals = None
            logtext = f.readline()
            while logtext:
                vals = re.match(r".*=== Running major cycle (?P<majors>[0-9]+)? ====.*",
                                logtext, re.IGNORECASE)
                if vals is not None:
                    input_major = max(input_major, int(vals.group("majors")))

                vals = None
                vals = re.match(r".*\[iter=(?P<minors>[0-9]+)?\].*",
                                logtext)
                if vals is not None:
                    input_minor = max(input_minor, int(vals.group("minors")))
                logtext = f.readline()

        with open(cls._outputLog) as f:
            vals = None
            logtext = f.readline()
            while logtext:
                vals = re.match(r".*=== Running major cycle (?P<majors>[0-9]+)? ====.*",
                                logtext, re.IGNORECASE)
                if vals is not None:
                    output_major = max(output_major, int(vals.group("majors")))

                vals = None
                vals = re.match(r".*\[iter=(?P<minors>[0-9]+)?\].*",
                                logtext)
                if vals is not None:
                    output_minor = max(output_minor, int(vals.group("minors")))
                logtext = f.readline()
        # OMS: check for output-input rather than abs(): fewer cycles is OK
        assert output_major - input_major <= cls.defMajorCycleTolerance(), "Number of major cycles used to reach termination " \
                                                                  "differs: Known good: %d, current %d" % (
                                                                  input_major, output_major)
        assert output_minor - input_minor <= cls.defMinorCycleTolerance(), "Number of minor cycles used to reach termination " \
                                                                  "differs: Known good: %d, current %d" % (
                                                                  input_minor, output_minor)


if __name__ == "__main__":
    pass # abstract class
