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

import unittest

import DDFacet.Tests.ShortAcceptanceTests.ClassCompareFITSImage
import numpy as np
import path

class TestDeepCleanWithBeam(DDFacet.Tests.ShortAcceptanceTests.ClassCompareFITSImage.ClassCompareFITSImage):

    @classmethod
    def defineImageList(cls):
        """ Method to define set of reference images to be tested.
            Can be overridden to add additional output products to the test.
            These must correspond to whatever is used in writing out the FITS files (eg. those in ClassDeconvMachine.py)
            Returns:
                List of image identifiers to reference and output products
        """
        return ['dirty', 'app.residual']

    @classmethod
    def defineMaxSquaredError(cls):
        """ Method defining maximum error tolerance between any pair of corresponding
            pixels in the output and corresponding reference FITS images.
            Should be overridden if another tolerance is desired
            Returns:
                constant for maximum tolerance used in test case setup
        """
        return [1e+9, 1e+9]  # dont care about sidelobes DR test will check the one and only source

    @classmethod
    def defMeanSquaredErrorLevel(cls):
        """ Method defining maximum tolerance for the mean squared error between any
            pair of FITS images. Should be overridden if another tolerance is
            desired
            Returns:
            constant for tolerance on mean squared error
        """
        return [1e+9, 1e+9]  # dont care about sidelobes DR test will check the one and only source

    @classmethod
    def defDRTolerance(cls):
        """
        Relative tolerance of clean dynamic range
        """
        return 0.05 # +/- 5% drift

    def testMaxSquaredError(self):
        pass # skip: since there is only one source we don't care about verifying the components placed on the sidelobes

    def testMeanSquaredError(self):
        pass # skip: since there is only one source we don't care about verifying the components placed on the sidelobes

    def testDR(self):
        """
        Checks clean dynamic range against previous known good result
        """
        cls = self.__class__

        dirty_ref = cls._refHDUList[cls.defineImageList().index("dirty")][0].data[...]
        appresidue_ref = cls._refHDUList[cls.defineImageList().index("app.residual")][0].data[...]
        DR_ref = max(np.max(dirty_ref), abs(np.min(dirty_ref))) / \
                 max(np.max(appresidue_ref), abs(np.min(appresidue_ref)))
        dirty_out = cls._outHDUList[cls.defineImageList().index("dirty")][0].data[...]
        appresidue_out = cls._outHDUList[cls.defineImageList().index("app.residual")][0].data[...]
        DR_out = max(np.max(dirty_out), abs(np.min(dirty_out))) / \
                 max(np.max(appresidue_out), abs(np.min(appresidue_out)))
        assert abs(1.0 - DR_ref / DR_out) <= cls.defDRTolerance(), "DR value has regressed. " \
                                                                   "Known good: %f, current %f" % (DR_ref, DR_out)

class TestDeepCleanWithoutBeam(TestDeepCleanWithBeam):
    pass # also do a DR check for the deep clean without the beam (same as above)

if __name__ == '__main__':
    unittest.main()