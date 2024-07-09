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
