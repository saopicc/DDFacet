#!/usr/bin/python

import os
import sys


if __name__ == "__main__":
    name = sys.argv[1]
    typein = sys.argv[2]
    print sys.argv
    strin = 'grep -r "%s" --include=*.%s .' % (name, typein)
    print strin
    os.system(strin)
