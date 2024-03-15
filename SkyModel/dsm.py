#!/usr/bin/env python
#from __future__ import division, absolute_import, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pyrap

import os
import sys

from pyrap.tables import table
from pyrap.images import image
import glob
import numpy as np

def driver():
    S=sys.argv[1::]
    
    ll=glob.glob(S[-1])
    im=image(ll[-1])
    d=im.getdata()
    
    ind=np.int64(np.random.rand(10000)*d.size)
    A=d.flat[ind]
    A=A[np.isnan(A)==0]
    std=np.std(A)
    vmin=-10*std
    vmax=40*std

    S=" ".join(S)

    ss="ds9 -cmap bb -scalelims %f %f %s -lock slice image -lock frame wcs -lock scale yes -match scalelimits -match scale -match colorbar -lock colorbar yes -view vertical"%(vmin,vmax,S)
    print(ss)
    os.system(ss)

if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()