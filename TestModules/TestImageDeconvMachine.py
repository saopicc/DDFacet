import numpy as np
from pyrap.images import image
import ClassImageDeconvMachine

def test():
    impsf=image("Continuous.psf")
    psf=impsf.getdata()
    imdirty=image("Continuous.dirty")
    dirty=imdirty.getdata()
    DC=ClassImageDeconvMachine.ClassImageDeconvMachine(Gain=0.051,MaxMinorIter=200,NCPU=30)
    DC.SetDirtyPSF(dirty,psf)
    DC.setSideLobeLevel(0.1)
    DC.Clean()
