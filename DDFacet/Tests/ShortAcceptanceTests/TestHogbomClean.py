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


class TestHogbomClean(DDFacet.Tests.ShortAcceptanceTests.ClassCompareFITSImage.ClassCompareFITSImage):

    @classmethod
    def defExecutionTime(cls):
        """
        Relative tolerance for total execution time in comparison with reference runs
        """
        return 1.0  # Longer for shorter tests 100%

    @classmethod
    def defineImageList(cls):
        return ['dirty', 'dirty.corr', 'psf', 'NormFacets', 'Norm',
                'app.residual', 'app.model',
                'app.convmodel', 'app.restored']

    @classmethod
    def defineMaxSquaredError(cls):
        return [1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
                1e-3, 1e-3,
                1e-3, 1e-3]  # epsilons per image pair, as listed in defineImageList

    @classmethod
    def defMeanSquaredErrorLevel(cls):
        return [1e-6, 1e-6, 1e-6, 1e-6, 1e-6,
                1e-5, 1e-5,
                1e-5, 1e-5]  # epsilons per image pair, as listed in defineImageList

if __name__ == '__main__':
    unittest.main()