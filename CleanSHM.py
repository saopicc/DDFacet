#!/usr/bin/env python

import optparse
import sys

from Other import MyLogger
from Array import NpShared
log=MyLogger.getLogger("ClearSHM")

def read_options():
    desc="""CohJones Questions and suggestions: cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* SHM")
    group.add_option('--ID',help='ID of ssared memory to be deleted, default is %default',default=None)
    opt.add_option_group(group)
    options, arguments = opt.parse_args()
    
    return options


if __name__=="__main__":
    options = read_options()
    print>>log, "Clear shared memory"
    if options.ID!=None:
        NpShared.DelAll(options.ID)
    else:
        NpShared.DelAll()

