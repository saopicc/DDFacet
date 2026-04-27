from DDFacet.Other import ModColor
import os

size = 1 # number of mpi processes
rank = 0 # local process ID
useMPI=False
DDF_FORCE_NOT_USE_MPI=int(os.environ.get("DDF_FORCE_NOT_USE_MPI", "0"))

if not DDF_FORCE_NOT_USE_MPI:
    try:
        from mpi4py.MPI import *
        size = COMM_WORLD.size
        rank = COMM_WORLD.rank
        comm=COMM_WORLD

        if size>1:
            useMPI=True
        else:
            print(ModColor.Str(" mpi4py properly initialised, but size=1, not using MPI mode.",col="blue"))
    except ModuleNotFoundError:
        print(ModColor.Str(" Could not initialise mpi4py ",col="blue"))
        pass

W=60
if useMPI and DDF_FORCE_NOT_USE_MPI:
    useMPI=False
    #print(ModColor.Str("="*W,col="blue"))
    print(ModColor.Str(" MPI mode disabled by DDF_FORCE_NOT_USE_MPI ".center(W,"="),col="blue"))
    #print(ModColor.Str("="*W,col="blue"))
elif useMPI:
    #print(ModColor.Str("="*W,col="blue"))
    #print(ModColor.Str("  MPI mode enabled ".center(W,"="),col="blue"))
    #print(ModColor.Str(("  size=%i "%size).center(W,"="),col="blue"))
    print(ModColor.Str(("  MPI mode enabled [n=%i]"%size).center(W,"="),col="blue"))
    #print(ModColor.Str("="*W,col="blue"))

# print("FLKSDLDFSLKSFDLJ",useMPI,size,rank,DDF_FORCE_NOT_USE_MPI)
    
    

    
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



import numpy as np




# ---------- large numpy broadcast ----------

def _bcast_large_array(arr, root=0, chunk_bytes=256*1024*1024):
    comm = COMM_WORLD
    rank = comm.rank

    if rank == root:
        arr = np.ascontiguousarray(arr)
        shape = arr.shape
        dtype = arr.dtype.str
        nbytes = arr.nbytes
    else:
        shape = dtype = nbytes = None

    shape = comm.bcast(shape, root)
    dtype = np.dtype(comm.bcast(dtype, root))
    nbytes = comm.bcast(nbytes, root)

    if rank != root:
        arr = np.empty(shape, dtype=dtype)
        arr = np.ascontiguousarray(arr)

    buf = memoryview(arr).cast('B')   # ✅ MPI-safe byte view

    offset = 0
    while offset < nbytes:
        size = min(chunk_bytes, nbytes - offset)
        chunk = buf[offset:offset+size]

        # explicit count form — most robust
        comm.Bcast([chunk, size, BYTE], root=root)

        offset += size

    return arr



# ---------- recursive broadcast ----------

def _bcast_any(obj, root=0):
    comm = COMM_WORLD

    rank = comm.rank

    # ---- detect type on root ----
    if rank == root:
        if isinstance(obj, np.ndarray):
            tag = "ndarray"
        elif isinstance(obj, dict):
            tag = "dict"
        elif isinstance(obj, (list, tuple)):
            tag = "list"
        else:
            tag = "scalar"
    else:
        tag = None

    tag = comm.bcast(tag, root=root)

    # ---- ndarray ----
    if tag == "ndarray":
        if rank != root:
            obj = None
        return _bcast_large_array(obj, root=root)

    # ---- dict ----
    elif tag == "dict":
        if rank == root:
            keys = list(obj.keys())
        else:
            keys = None

        keys = comm.bcast(keys, root=root)

        out = {}
        for k in keys:
            if rank == root:
                v = obj[k]
            else:
                v = None
            out[k] = _bcast_any(v, root=root)

        return out

    # ---- list / tuple ----
    elif tag == "list":
        if rank == root:
            n = len(obj)
        else:
            n = None

        n = comm.bcast(n, root=root)

        out = []
        for i in range(n):
            if rank == root:
                v = obj[i]
            else:
                v = None
            out.append(_bcast_any(v, root=root))

        return tuple(out) if isinstance(obj, tuple) else out

    # ---- small scalar ----
    else:
        return comm.bcast(obj, root=root)


# ---------- public function ----------

def bcast_chunk_dict(d, root=0):
    return _bcast_any(d, root=root)
