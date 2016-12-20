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
from DDFacet.Parset.ReadCFG import Parset
from pyrap.tables import table
from matplotlib import pyplot as plt
import numpy as np

def run_ddf(parset, image_prefix, stdout_filename, stderr_filename):
    """ Execute DDFacet """
    args = ['DDF.py', parset,
            '--ImageName=%s' % image_prefix]

    stdout_file = open(stdout_filename, 'w')
    stderr_file = open(stderr_filename, 'w')

    with stdout_file, stderr_file:
        subprocess.check_call(args, env=os.environ.copy(),
                              stdout=stdout_file, stderr=stderr_file)

class TestFacetPredict(unittest.TestCase):
    """
        Automated assurance test for ddfacet predict option
        Finds parset with the name TestFacetPredict.parset.cfg and executes a predict
        Compares the amplitude and phase differences between a DFT (Meqtrees) and a facet-based degridding
        with a source close to the facet centre
        Input:
            TestFacetPredict.parset.cfg in DDFACET_TEST_INPUT_DIR
        Output
            Amplitude plots of the visibilities for comparison
            [classname].run.parset into DDFACET_TEST_OUTPUT_DIR
    """
    @classmethod
    def getAmpThreshold(cls):
        return 0.10

    @classmethod
    def getPhaseThreshold(cls):
        return 0.05

    @classmethod
    def setParsetOption(cls, section, option, value):
        """
            Sets the default option read by the configuration parser
            args:
                section: Configuration [section] name
                option: Section option name
                value: Value for option (refer to default parset for documentation)
        """
        cls._defaultParsetConfig.set(section, option, value)

    @classmethod
    def setUpClass(cls):
        unittest.TestCase.setUpClass()
        cls._inputDir = getenv('DDFACET_TEST_DATA_DIR', './') + "/"
        cls._outputDir = getenv('DDFACET_TEST_OUTPUT_DIR', '/tmp/') + "/"
        cls._inputParsetFilename = cls._inputDir + cls.__name__ + ".parset.cfg"
        cls._outputParsetFilename = cls._outputDir + cls.__name__ + ".run.parset.cfg"
        if not path.isfile(cls._inputParsetFilename):
            raise RuntimeError("Default parset file %s does not exist" % cls._inputParsetFilename)
        p = Parset(File=cls._inputParsetFilename)
        cls._defaultParsetConfig = p.Config
        cls._imagePrefix = cls._outputDir + cls.__name__ + ".run"

        # set up path to each ms relative to environment variable
        if type(p.DicoPars["VisData"]["MSName"]) is list:
            for ms in p.DicoPars["VisData"]["MSName"]:
                if path.dirname(ms) != "":
                    raise RuntimeError("Expected only measurement set name, "
                                       "not relative or absolute path in %s" % ms)
            abs_ms = [cls._inputDir + ms for ms in p.DicoPars["VisData"]["MSName"]]
            cls._ms_list = abs_ms
            cls.setParsetOption("VisData", "MSName", "[" + (",".join(abs_ms)) + "]")
        else:
            ms = p.DicoPars["VisData"]["MSName"]
            if path.dirname(ms) != "":
                raise RuntimeError("Expected only measurement set name, "
                                   "not relative or absolute path in %s" % ms)
            abs_ms = cls._inputDir + ms
            cls._ms_list = [abs_ms]
            cls.setParsetOption("VisData", "MSName", abs_ms)

        ms = p.DicoPars["Images"]["PredictModelName"]
        abs_skymodel = cls._inputDir + ms
        cls.setParsetOption("Images", "PredictModelName", abs_skymodel)

        ms = p.DicoPars["Beam"]["FITSFile"]
        abs_beam = cls._inputDir + ms
        cls.setParsetOption("Beam", "FITSFile", abs_beam)

        # write out parset file to output directory
        fOutputParset = open(cls._outputParsetFilename, mode='w')
        cls._defaultParsetConfig.write(fOutputParset)
        fOutputParset.close()

        # run ddfacet
        cls._stdoutLogFile = cls._outputDir + cls.__name__ + ".run.out.log"
        cls._stderrLogFile = cls._outputDir + cls.__name__ + ".run.err.log"

        run_ddf(parset=cls._outputParsetFilename,
                image_prefix=cls.__class__.__name__,
                stdout_filename=cls._stdoutLogFile,
                stderr_filename=cls._stderrLogFile)

    @classmethod
    def tearDownClass(cls):
        unittest.TestCase.tearDownClass()
        pass

    def setUp(self):
        unittest.TestCase.setUp(self)

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def testDDFagainstMeqtreesAmplitude(self):

        list_except = []
        for ms_i, ms in enumerate(self._ms_list):
            with table(ms) as t:
                meqtrees = t.getcol("DATA")
                ddfacet = t.getcol("MODEL_DATA")
                diff = np.abs(meqtrees) / np.abs(ddfacet)
                for icorr in xrange(4):
                    gainsPlot = self._outputDir + \
                                self.__class__.__name__ + \
                                ".ms" + str(ms_i) + \
                                ".corr" + str(icorr) + \
                                ".amp" + \
                                ".gainsplot.png"
                    #Amplitude error
                    fig = plt.figure()
                    plt.plot(np.abs(meqtrees[:,0,icorr]), 'rx', label="Meqtrees")
                    plt.plot(np.abs(ddfacet[:,0,icorr]), 'bx', label="DDFacet")
                    plt.title(os.path.basename(ms) + " correlation" + str(icorr))
                    plt.xlabel("row")
                    plt.ylabel("Jy")
                    plt.legend()
                    fig.savefig(gainsPlot)
                    plt.close(fig)

                try:
                    assert np.max(np.abs(diff - 1.0)) <= TestFacetPredict.getAmpThreshold(), \
                        "Facet prediction != meqtrees for ms %s" % ms
                except AssertionError, e:
                    list_except.append(str(e))
                continue

        if len(list_except) != 0:
            msg = "\n".join(list_except)
            raise AssertionError("The following assertions failed:\n %s" % msg)

    def testDDFagainstMeqtreesPhase(self):

        list_except = []
        for ms_i, ms in enumerate(self._ms_list):
            with table(ms) as t:
                meqtrees = t.getcol("DATA")
                ddfacet = t.getcol("MODEL_DATA")
                diff_rel = np.angle(meqtrees) / np.angle(ddfacet)
                try:
                    assert np.max(np.abs(diff_rel - 1.0)) <= TestFacetPredict.getPhaseThreshold(), \
                        "Facet prediction != meqtrees for ms %s" % ms
                except AssertionError, e:
                    list_except.append(str(e))
                continue

        if len(list_except) != 0:
            msg = "\n".join(list_except)
            raise AssertionError("The following assertions failed:\n %s" % msg)

if __name__ == "__main__":
    unittest.main()