size = 1 # number of mpi processes
rank = 0 # local process ID
useMPI=False

try:
    from mpi4py.MPI import *
    
    useMPI = True
    size = COMM_WORLD.size
    rank = COMM_WORLD.rank
except ModuleNotFoundError:
    pass

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