import unittest
import ClassCompareFITSImage


class TestWeightingBriggs1PSF(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['psf']

class TestWeightingBriggs1PSF(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['psf']

class TestWeightingBriggsMinus1PSF(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['psf']

class TestWeightingBriggs0PSF(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['psf']

class TestWeightingNaturalPSF(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['psf']

class TestWeightingUniformPSF(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['psf']

if __name__ == '__main__':
    unittest.main()