import numpy as np
from pyrap.images import image
import ClassImageDeconvMachine

def test():
    impsf=image("ImageTest2.psf")
    psf=impsf.getdata()
    imdirty=image("ImageTest2.dirty")
    dirty=imdirty.getdata()
    DC=ClassImageDeconvMachine.ClassImageDeconvMachine()
    DC.SetDirtyPSF(dirty,psf)
    DC.setSideLobeLevel(0.3)
    DC.Clean()
