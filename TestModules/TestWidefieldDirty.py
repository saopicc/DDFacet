import unittest
import ClassCompareFITSImage


class TestWidefieldDirty(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['dirty', 'dirty.corr', 'Norm', 'NormFacets']


if __name__ == '__main__':
    unittest.main()
