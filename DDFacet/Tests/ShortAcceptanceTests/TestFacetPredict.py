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

def run_ddf(parset, image_prefix, stdout_filename, stderr_filename, beam_model="FITS"):
    """ Execute DDFacet """
    args = ['DDF.py', parset,
            '--Output-Name=%s' % image_prefix,
            '--Beam-Model=%s' % beam_model]
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
        with a source close to the facet centre. With and without beam enabled.
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
        return 10 # degrees

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
    def setUpClass(cls):
        unittest.TestCase.setUpClass()
        cls._inputDir = getenv('DDFACET_TEST_DATA_DIR', './') + "/"
        cls._outputDir = getenv('DDFACET_TEST_OUTPUT_DIR', '/tmp/') + "/"
        cls._inputParsetFilename = cls._inputDir + cls.__name__ + ".parset.cfg"
        cls._outputParsetFilename = cls._outputDir + cls.__name__ + ".run.parset.cfg"
        if not path.isfile(cls._inputParsetFilename):
            raise RuntimeError("Default parset file %s does not exist" % cls._inputParsetFilename)
        p = Parset(cls._inputParsetFilename)
        cls._defaultParset = p
        cls._imagePrefix = cls._outputDir + cls.__name__ + ".run"

        # set up path to each ms relative to environment variable
        if type(p.DicoPars["Data"]["MS"]) is list:
            for ms in p.DicoPars["Data"]["MS"]:
                if path.dirname(ms) != "":
                    raise RuntimeError("Expected only measurement set name, "
                                       "not relative or absolute path in %s" % ms)
            abs_ms = [cls._inputDir + ms for ms in p.DicoPars["Data"]["MS"]]
            cls._ms_list = abs_ms
            cls.setParsetOption("Data", "MS", "[" + (",".join(abs_ms)) + "]")
        else:
            ms = p.DicoPars["Data"]["MS"]
            if path.dirname(ms) != "":
                raise RuntimeError("Expected only measurement set name, "
                                   "not relative or absolute path in %s" % ms)
            abs_ms = cls._inputDir + ms
            cls._ms_list = [abs_ms]
            cls.setParsetOption("Data", "MS", abs_ms)

        ms = p.DicoPars["Predict"]["FromImage"]
        abs_skymodel = cls._inputDir + ms
        cls.setParsetOption("Predict", "FromImage", abs_skymodel)

        ms = p.DicoPars["Beam"]["FITSFile"]
        abs_beam = cls._inputDir + ms
        cls.setParsetOption("Beam", "FITSFile", abs_beam)

        # write out parset file to output directory
        fOutputParset = open(cls._outputParsetFilename, mode='w')
        cls._defaultParset.write(fOutputParset)
        fOutputParset.close()

    @classmethod
    def tearDownClass(cls):
        unittest.TestCase.tearDownClass()
        pass

    def setUp(self):
        unittest.TestCase.setUp(self)

    def tearDown(self):
        unittest.TestCase.tearDown(self)

    def testDDFagainstMeqtreesWithBeam(self):
        # run ddfacet
        self._stdoutLogFile = self._outputDir + self.__class__.__name__ + ".run.withbeam.out.log"
        self._stderrLogFile = self._outputDir + self.__class__.__name__ + ".run.withbeam.err.log"

        run_ddf(parset=self._outputParsetFilename,
                image_prefix=self.__class__.__name__,
                stdout_filename=self._stdoutLogFile,
                stderr_filename=self._stderrLogFile,
                beam_model="FITS")

        list_except = []
        for ms_i, ms in enumerate(self._ms_list):
            with table(ms) as t:
                # first test amplitude
                meqtrees = t.getcol("CORRECTED_DATA")
                ddfacet = t.getcol("MODEL_DATA")
                diff = np.abs(meqtrees) / np.abs(ddfacet)
                for icorr in xrange(4):
                    gainsPlot = self._outputDir + \
                                self.__class__.__name__ + \
                                ".ms" + str(ms_i) + \
                                ".corr" + str(icorr) + \
                                ".amp" + \
                                ".with_beam" + \
                                ".gainsplot.png"
                    # Plot amplitudes
                    fig = plt.figure()
                    plt.plot(np.abs(meqtrees[:, 0, icorr]), 'rx', label="Meqtrees")
                    plt.plot(np.abs(ddfacet[:, 0, icorr]), 'bx', label="DDFacet")
                    plt.title(os.path.basename(ms) + " correlation" + str(icorr))
                    plt.xlabel("row")
                    plt.ylabel("Jy")
                    plt.legend()
                    fig.savefig(gainsPlot)
                    plt.close(fig)

                try:
                    # For the time being we cannot predict Q, U or V, so only check I and V prediction for a model
                    # with only I in it
                    assert np.max(np.abs(diff[:, :, 0] - 1.0)) <= TestFacetPredict.getAmpThreshold(), \
                        "Facet amplitude prediction != meqtrees for ms %s correlation %d" % (ms, 0)
                    assert np.max(np.abs(diff[:, :, 3] - 1.0)) <= TestFacetPredict.getAmpThreshold(), \
                        "Facet amplitude prediction != meqtrees for ms %s correlation %d" % (ms, 3)
                except AssertionError, e:
                    list_except.append(str(e))

                # next test phase
                diff_rel = np.angle(meqtrees) - np.angle(ddfacet)
                for icorr in xrange(4):
                    gainsPlot = self._outputDir + \
                                self.__class__.__name__ + \
                                ".ms" + str(ms_i) + \
                                ".corr" + str(icorr) + \
                                ".phase" + \
                                ".with_beam" + \
                                ".gainsplot.png"
                    # Plot phase differences
                    fig = plt.figure()
                    plt.plot(diff_rel[:, 0, icorr], 'mx', label="Phase diff")
                    plt.title(os.path.basename(ms) + " correlation" + str(icorr))
                    plt.xlabel("row")
                    plt.ylabel("Relative error (deg)")
                    plt.legend()
                    fig.savefig(gainsPlot)
                    plt.close(fig)

                try:
                    # For the time being we cannot predict Q, U or V, so only check I and V prediction for a model
                    # with only I in it
                    assert np.max(np.abs(diff_rel[:, :, 0] - 1.0)) <= TestFacetPredict.getPhaseThreshold(), \
                        "Facet phase prediction != meqtrees for ms %s correlation %d" % (ms, 0)
                    assert np.max(np.abs(diff_rel[:, :, 3] - 1.0)) <= TestFacetPredict.getPhaseThreshold(), \
                        "Facet phase prediction != meqtrees for ms %s correlation %d" % (ms, 3)
                except AssertionError, e:
                    list_except.append(str(e))

        if len(list_except) != 0:
            msg = "\n".join(list_except)
            raise AssertionError("The following assertions failed:\n %s" % msg)

    def testDDFagainstMeqtreesWithoutBeam(self):
        # run ddfacet
        self._stdoutLogFile = self._outputDir + self.__class__.__name__ + ".run.withoutbeam.out.log"
        self._stderrLogFile = self._outputDir + self.__class__.__name__ + ".run.withoutbeam.err.log"

        run_ddf(parset=self._outputParsetFilename,
                image_prefix=self.__class__.__name__,
                stdout_filename=self._stdoutLogFile,
                stderr_filename=self._stderrLogFile,
                beam_model="None")

        list_except = []
        for ms_i, ms in enumerate(self._ms_list):
            with table(ms) as t:
                # first test amplitude
                meqtrees = t.getcol("DATA")
                ddfacet = t.getcol("MODEL_DATA")
                diff = np.abs(meqtrees) / np.abs(ddfacet)
                for icorr in xrange(4):
                    gainsPlot = self._outputDir + \
                                self.__class__.__name__ + \
                                ".ms" + str(ms_i) + \
                                ".corr" + str(icorr) + \
                                ".amp" + \
                                ".without_beam" + \
                                ".gainsplot.png"
                    # Plot amplitudes
                    fig = plt.figure()
                    plt.plot(np.abs(meqtrees[:, 0, icorr]), 'rx', label="Meqtrees")
                    plt.plot(np.abs(ddfacet[:, 0, icorr]), 'bx', label="DDFacet")
                    plt.title(os.path.basename(ms) + " correlation" + str(icorr))
                    plt.xlabel("row")
                    plt.ylabel("Jy")
                    plt.legend()
                    fig.savefig(gainsPlot)
                    plt.close(fig)

                try:
                    # For the time being we cannot predict Q, U or V, so only check I and V prediction for a model
                    # with only I in it
                    assert np.max(np.abs(diff[:, :, 0] - 1.0)) <= TestFacetPredict.getAmpThreshold(), \
                        "Facet amplitude prediction != meqtrees for ms %s correlation %d" % (ms, 0)
                    assert np.max(np.abs(diff[:, :, 3] - 1.0)) <= TestFacetPredict.getAmpThreshold(), \
                        "Facet amplitude prediction != meqtrees for ms %s correlation %d" % (ms, 3)
                except AssertionError, e:
                    list_except.append(str(e))

                # next test phase
                diff_rel = np.angle(meqtrees) - np.angle(ddfacet)
                for icorr in xrange(4):
                    gainsPlot = self._outputDir + \
                                self.__class__.__name__ + \
                                ".ms" + str(ms_i) + \
                                ".corr" + str(icorr) + \
                                ".phase" + \
                                ".without_beam" + \
                                ".gainsplot.png"
                    # Plot phase differences
                    fig = plt.figure()
                    plt.plot(diff_rel[:, 0, icorr], 'mx', label="Phase diff")
                    plt.title(os.path.basename(ms) + " correlation" + str(icorr))
                    plt.xlabel("row")
                    plt.ylabel("Relative error (deg)")
                    plt.legend()
                    fig.savefig(gainsPlot)
                    plt.close(fig)

                try:
                    # For the time being we cannot predict Q, U or V, so only check I and V prediction for a model
                    # with only I in it
                    assert np.max(np.abs(diff_rel[:, :, 0] - 1.0)) <= TestFacetPredict.getPhaseThreshold(), \
                        "Facet phase prediction != meqtrees for ms %s correlation %d" % (ms, 0)
                    assert np.max(np.abs(diff_rel[:, :, 3] - 1.0)) <= TestFacetPredict.getPhaseThreshold(), \
                        "Facet phase prediction != meqtrees for ms %s correlation %d" % (ms, 3)
                except AssertionError, e:
                    list_except.append(str(e))

        if len(list_except) != 0:
            msg = "\n".join(list_except)
            raise AssertionError("The following assertions failed:\n %s" % msg)

if __name__ == "__main__":
    unittest.main()
