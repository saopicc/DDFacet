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

# Tests DDF using LOFAR data and automasking with constrained uv coverage
# Tests SkyModel helper scripts

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range
import unittest
from subprocess import Popen
import os
from os import path, getenv
import subprocess
import time

import DDFacet.Tests.ShortAcceptanceTests.ClassCompareFITSImage

class TestLOFAR_J1329_p4729(DDFacet.Tests.ShortAcceptanceTests.ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def __run(cls, cmdargs, timeout=600):
        p = Popen(cmdargs, 
                    env=os.environ.copy())

        x = timeout
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
            raise RuntimeError("{} exited with non-zero return code {}".format(cmdargs[0], ret))

    @classmethod
    def defineImageList(cls):
        """ Method to define set of reference images to be tested.
            Can be overridden to add additional output products to the test.
            These must correspond to whatever is used in writing out the FITS files (eg. those in ClassDeconvMachine.py)
            Returns:
                List of image identifiers to reference and output products
        """
        return ['dirty', 'dirty.corr', 'psf', 'NormFacets', 'Norm',
                'app.residual', 'app.model',
                'app.convmodel', 'app.restored']

    @classmethod
    def defineMaxSquaredError(cls):
        """ Method defining maximum error tolerance between any pair of corresponding
            pixels in the output and corresponding reference FITS images.
            Should be overridden if another tolerance is desired
            Returns:
                constant for maximum tolerance used in test case setup
        """
        return [1e-6,1e-6,1e-6,1e-4,1e-4,
                5e-3,5e-3,
                5e-3,5e-3]

    @classmethod
    def defMeanSquaredErrorLevel(cls):
        """ Method defining maximum tolerance for the mean squared error between any
            pair of FITS images. Should be overridden if another tolerance is
            desired
            Returns:
            constant for tolerance on mean squared error
        """
        return [1e-7,1e-7,1e-7,1e-7,1e-7,
                1e-5,1e-5,
                1e-5,1e-5]

    @classmethod
    def timeoutsecs(cls):
        return 2100 * 4

    @classmethod
    def pre_imaging_hook(cls):
        pass

    @classmethod
    def post_imaging_hook(cls):
        basename = cls._outputDir+cls.__name__+".run"
        restoredname = basename+".app.restored.fits"
        # steps neeeded for mask
        args = ['MakeMask.py',
                '--RestoredIm={}'.format(restoredname),
                '--Th=7']
        cls.__run(args)

        args = ['MaskDicoModel.py',
                '--InDicoModel={}.DicoModel'.format(basename),
                '--OutDicoModel={}.DicoModel.Masked'.format(basename),
                '--MaskName={}.mask.fits'.format(restoredname)]
        cls.__run(args)

        # steps neeeded for autocluster
        args = ['MakeCatalog.py',
                '--RestoredIm={}'.format(restoredname)]
        cls.__run(args)

        args = ['ClusterCat.py',
                '--DoPlot=0',
                '--NGen=100',
                '--NCPU=56',
                '--NCluster=6',
                '--FluxMin=0.0001',
                '--SourceCat={}.pybdsm.srl.fits'.format(restoredname.replace(".fits", ""))]
        cls.__run(args)

        # steps needed for manual cluster
        tagfilename = '{}.tagged.reg'.format(restoredname)
        with open(tagfilename, 'w+') as f:
            f.write('# Region file format: DS9 version 4.1\n')
            f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
            f.write('fk5\n')
            f.write('circle(13:45:25.720,+49:46:10.957,832.169")\n')
            f.write('circle(13:31:32.759,+50:06:56.311,832.169")\n')
            f.write('circle(13:21:19.565,+48:37:58.258,832.169")\n')
            f.write('circle(13:37:36.370,+47:40:09.337,832.169")\n')
            f.write('circle(13:41:49.433,+46:55:39.438,832.169")\n')
            f.write('circle(13:36:05.311,+46:04:48.536,832.169")\n')
            f.write('circle(13:29:37.068,+45:39:16.561,832.169")\n')
            f.write('circle(13:19:55.809,+46:38:15.724,832.169")\n')
            f.write('circle(13:11:10.167,+47:37:33.579,832.169")\n')
            f.write('circle(13:46:18.333,+48:11:13.511,832.169")')

        args=["MakeModel.py",
              "--ds9PreClusterFile={}".format(tagfilename),
              "--NCluster=10",
              "--DoPlot=0",
              "--BaseImageName={}".format(basename)]
        cls.__run(args)


if __name__ == '__main__':
    unittest.main()
