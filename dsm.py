#!/usr/bin/python

import os
import sys
from pyrap.images import image
import glob
import numpy as np


if __name__=="__main__":


    S=sys.argv[1::]
    
    ll=glob.glob(S[-1])
    im=image(ll[-1])
    d=im.getdata()
    
    ind=np.int64(np.random.rand(10000)*d.size)
    std=np.std(d.flat[ind])
    vmin=-10*std
    vmax=40*std

    S=" ".join(S)

    ss="ds9 -view vertical -cmap bb -scalelims %f %f %s -lock frame wcs -lock scale yes -match scalelimits -match scale -match colorbar -lock colorbar yes"%(vmin,vmax,S)
    os.system(ss)
