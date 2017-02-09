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
import unittest
import shutil
import subprocess

import numpy as np

from DDFacet.Parset.ReadCFG import Parset

def run_ddf(parset, image_prefix, stdout_filename, stderr_filename):
    """ Execute DDFacet """
    args = ['DDF.py', parset,
        '--Output-Name=%s' % image_prefix]

    stdout_file = open(stdout_filename, 'w')
    stderr_file = open(stderr_filename, 'w')

    with stdout_file, stderr_file:
        subprocess.check_call(args, env=os.environ.copy(),
            stdout=stdout_file, stderr=stderr_file)

class TestMontblancPredict(unittest.TestCase):

    def testMontblancPredict(self):
        pc = self._parset

        # Set the image name prefix
        pc.set("Output", "Name", self._image_prefix)

        # Configure DDFacet to predict using Montblanc
        pc.set("Image", "Mode", "Predict")
        pc.set("Image", "PredictMode", "Montblanc")

        # Predict from Predict.DicoModel
        pc.set("Predict", "InitDicoModel", os.path.join(self._input_dir,
            "sky_models", "Predict.DicoModel"))

        # Predict into MONTBLANC_DATA
        pc.set("Data", "MS", os.path.join(self._input_dir,
            "basicSSMFClean.MS_p0"))
        pc.set("Predict", "ColName", "MONTBLANC_DATA")

        # Write the parset config to the output file name
        with open(self._output_parset_filename, 'w') as op:
            pc.write(op)

        # Run DDFacet
        run_ddf(self._output_parset_filename, self._image_prefix,
            self._stdout_filename, self._stderr_filename)

    def testDDFacetPredict(self):
        pc = self._parset

        # Set the image name prefix
        pc.set("Output", "Name", self._image_prefix)

        # Configure DDFacet to predict using DDFacet's DeGridder
        pc.set("Image", "Mode", "Predict")
        pc.set("Image", "PredictMode", "BDA-degrid")

        # Predict from Predict.DicoModel
        pc.set("Predict", "InitDicoModel", os.path.join(self._input_dir,
            "sky_models", "Predict.DicoModel"))

        # Predict into DDFACET_DATA
        pc.set("Data", "MS", os.path.join(self._input_dir,
            "basicSSMFClean.MS_p0"))
        pc.set("Predict", "ColName", "DDFACET_DATA")

        # Write the parset config to the output file name
        with open(self._output_parset_filename, 'w') as op:
            pc.write(op)

        # Run DDFacet
        run_ddf(self._output_parset_filename, self._image_prefix,
            self._stdout_filename, self._stderr_filename)

    def setUp(self):
        cname = self.__class__.__name__
        self._input_dir = os.getenv('DDFACET_TEST_DATA_DIR', '/') + "/"
        self._output_dir = os.getenv('DDFACET_TEST_OUTPUT_DIR', '/tmp/') + "/"
        self._image_prefix = ''.join((self._output_dir, cname, ".run"))
        self._input_parset_filename = ''.join((self._input_dir, cname, '.parset.cfg'))
        self._output_parset_filename = ''.join((self._output_dir, cname, '.run.parset.cfg'))
        self._stdout_filename = ''.join((self._output_dir, cname, ".run.out.log"))
        self._stderr_filename = ''.join((self._output_dir, cname, ".run.err.log"))

        if not os.path.isfile(self._input_parset_filename):
            raise RuntimeError("Parset file %s does not exist" % self._input_parset_filename)

        self._parset = Parset(self._input_parset_filename)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
