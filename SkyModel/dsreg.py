#!/usr/bin/python
from __future__ import division, absolute_import, print_function

import os
import sys

def driver():
    os.system("ds9 %s -regions load all %s"%(sys.argv[1],sys.argv[2]))

if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()