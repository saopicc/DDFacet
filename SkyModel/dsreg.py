#!/usr/bin/python
from __future__ import division, absolute_import, print_function

import os
import sys

if __name__=="__main__":
    

    
    os.system("ds9 %s -regions load all %s"%(sys.argv[1],sys.argv[2]))
