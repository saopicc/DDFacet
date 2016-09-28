import unittest
import ClassCompareFITSImage


class TestWeightingBriggs1PSF(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['psf']
    
    @classmethod
    def defineMaxSquaredError(cls):
        return [1e-7] #epsilons per image pair, as listed in defineImageList

    @classmethod
    def defMeanSquaredErrorLevel(cls):
        return [1e-7] #epsilons per image pair, as listed in defineImageList 

class TestWeightingBriggs1PSF(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['psf']
    
    @classmethod
    def defineMaxSquaredError(cls):
        return [1e-7] #epsilons per image pair, as listed in defineImageList

    @classmethod
    def defMeanSquaredErrorLevel(cls):
        return [1e-7] #epsilons per image pair, as listed in defineImageList 

class TestWeightingBriggsMinus1PSF(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['psf']
    
    @classmethod
    def defineMaxSquaredError(cls):
        return [1e-7] #epsilons per image pair, as listed in defineImageList

    @classmethod
    def defMeanSquaredErrorLevel(cls):
        return [1e-7] #epsilons per image pair, as listed in defineImageList 


class TestWeightingBriggs0PSF(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['psf']
    
    @classmethod
    def defineMaxSquaredError(cls):
        return [1e-7] #epsilons per image pair, as listed in defineImageList

    @classmethod
    def defMeanSquaredErrorLevel(cls):
        return [1e-7] #epsilons per image pair, as listed in defineImageList 

class TestWeightingNaturalPSF(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['psf']
    
    @classmethod
    def defineMaxSquaredError(cls):
        return [1e-7] #epsilons per image pair, as listed in defineImageList

    @classmethod
    def defMeanSquaredErrorLevel(cls):
        return [1e-7] #epsilons per image pair, as listed in defineImageList 

class TestWeightingUniformPSF(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['psf']
    
    @classmethod
    def defineMaxSquaredError(cls):
        return [1e-7] #epsilons per image pair, as listed in defineImageList

    @classmethod
    def defMeanSquaredErrorLevel(cls):
        return [1e-7] #epsilons per image pair, as listed in defineImageList 

if __name__ == '__main__':
    unittest.main()
