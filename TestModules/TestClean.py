import unittest
import ClassCompareFITSImage


class TestSSMFClean(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['dirty', 'dirty.corr', 'psf', 'NormFacets', 'Norm',
                'app.residual', 'app.model',
                'app.convmodel', 'app.restored']

    @classmethod
    def defineMaxSquaredError(cls):
        return [1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
                1e-4, 1e-4,
                1e-4, 1e-4]  # epsilons per image pair, as listed in defineImageList

class TestSSSFClean(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['dirty', 'dirty.corr', 'psf', 'NormFacets', 'Norm',
                'app.residual', 'app.model',
                'app.convmodel', 'app.restored']

    @classmethod
    def defineMaxSquaredError(cls):
        return [1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
                1e-4, 1e-4,
                1e-4, 1e-4]  # epsilons per image pair, as listed in defineImageList


if __name__ == '__main__':
    unittest.main()