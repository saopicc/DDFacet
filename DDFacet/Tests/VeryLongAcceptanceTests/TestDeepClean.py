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

class TestDeepClean(DDFacet.Tests.ShortAcceptanceTests.ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defDRTolerance(cls):
        """
        Relative tolerance of clean dynamic range
        """
        return 0.03 # 3% drift

    def testDR(self):
        """
        Checks clean dynamic range against previous known good result
        """
        cls = self.__class__

        dirty_ref = cls._refHDUList[cls.defineImageList().index("dirty")]
        appresidue_ref = cls._refHDUList[cls.defineImageList().index("app.residual")]
        DR_ref = max(np.max(dirty_ref), abs(np.min(dirty_ref))) / \
                 max(np.max(appresidue_ref), abs(np.min(appresidue_ref)))
        dirty_out = cls._outHDUList[cls.defineImageList().index("dirty")]
        appresidue_out = cls._outHDUList[cls.defineImageList().index("app.residual")]
        DR_out = max(np.max(dirty_out), abs(np.min(dirty_out))) / \
                 max(np.max(appresidue_out), abs(np.min(appresidue_out)))
        assert abs(DR_ref / DR_out) <= cls.defDRTolerance(), "DR value has regressed. " \
                                                             "Known good: %f, current %f" % (DR_ref, DR_out)
if __name__ == '__main__':
    unittest.main()