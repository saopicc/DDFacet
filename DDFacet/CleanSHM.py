#!/usr/bin/env python

import optparse

from DDFacet.Array import NpShared
from DDFacet.Other import MyLogger

log= MyLogger.getLogger("ClearSHM")
from DDFacet.cbuild.Gridder import _pyGridderSmearPols as _pyGridderSmear
import glob

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
    if options.ID is not None:
        NpShared.DelAll(options.ID)
    else:
        NpShared.DelAll()

    ll=glob.glob("/dev/shm/sem.*")
        
    print>>log, "Clear Semaphores"
    
    ListSemaphores=[".".join(l.split(".")[1::]) for l in ll]

    _pyGridderSmear.pySetSemaphores(ListSemaphores)
    _pyGridderSmear.pyDeleteSemaphore(ListSemaphores)
