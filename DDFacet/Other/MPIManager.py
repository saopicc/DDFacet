from DDFacet.Other import ModColor

size = 1 # number of mpi processes
rank = 0 # local process ID
useMPI=False

try:
    from mpi4py.MPI import *
    size = COMM_WORLD.size
    rank = COMM_WORLD.rank
    if size>1:
        useMPI=True
    else:
        print(ModColor.Str(" mpi4py properly initialised, but size=1, not using MPI mode.",col="blue"))
        
except ModuleNotFoundError:
    print(ModColor.Str(" Could not initialise mpi4py ",col="blue"))
    pass

import os
DDF_FORCE_NOT_USE_MPI=int(os.environ.get("DDF_FORCE_NOT_USE_MPI", "0"))
W=60
if useMPI and DDF_FORCE_NOT_USE_MPI:
    useMPI=False
    print(ModColor.Str("="*W,col="blue"))
    print(ModColor.Str(" MPI mode disabled by DDF_FORCE_NOT_USE_MPI ".center(W,"="),col="blue"))
    print(ModColor.Str("="*W,col="blue"))
elif useMPI:
    print(ModColor.Str("="*W,col="blue"))
    print(ModColor.Str("  MPI mode enabled ".center(W,"="),col="blue"))
    print(ModColor.Str(("  size=%i "%size).center(W,"="),col="blue"))
    print(ModColor.Str("="*W,col="blue"))


    
try:
    from fasteners import InterProcessLock
except ModuleNotFoundError:
    if useMPI:
        raise ModuleNotFoundError("fasterners package is missing. Install DDFacet using "
                                  "pip install 'DDFacet[mpi-support]'")
    # define a dummy InterProcessLock
    class InterProcessLock(object):
        def __init__(self,
                     path,
                     sleep_func=None,
                     logger=None):
            pass
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
