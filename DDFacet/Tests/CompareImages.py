#!/usr/bin/env python
from pyrap.images import image
import sys
from astropy.io import fits

def test(File0,File1,OutFile="Diff.cube.psf.fits"):
    f0=fits.open(File0)
    f1=fits.open(File1)
    d=f0[0].data-f1[0].data
    f0[0].data[:]=d[:]
    print("Writting: %s"%OutFile)
    f0.writeto(OutFile,overwrite=1)

if __name__=="__main__":
    test(*sys.argv[1:])
