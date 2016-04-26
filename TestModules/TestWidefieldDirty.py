import unittest
import ClassCompareFITSImage


class TestWidefieldDirty(ClassCompareFITSImage.ClassCompareFITSImage):
    @classmethod
    def defineImageList(cls):
        return ['dirty', 'dirty.corr', 'Norm', 'NormFacets']
   
    @classmethod
    def defineMaxSquaredError(cls):
        """ Method defining maximum error tolerance between any pair of corresponding
            pixels in the output and corresponding reference FITS images.
            Should be overridden if another tolerance is desired
            Returns:
                constant for maximum tolerance used in test case setup
        """
        return [1e-7,1e-7,1e-7,1e-7] #epsilons per image pair, as listed in defineImageList

    @classmethod
    def defMeanSquaredErrorLevel(cls):
        """ Method defining maximum tolerance for the mean squared error between any
            pair of FITS images. Should be overridden if another tolerance is
            desired
            Returns:
                constant for tolerance on mean squared error
        """
        return [1e-7,1e-7,1e-7,1e-7] #epsilons per image pair, as listed in defineImageList

if __name__ == '__main__':
    unittest.main()
