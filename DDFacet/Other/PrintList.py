import numpy as np

def normalize(x):

    # numpy scalar -> python scalar
    if isinstance(x, np.generic):
        return x.item()

    # list -> recurse
    if isinstance(x, list):
        return [normalize(v) for v in x]

    # tuple -> recurse
    if isinstance(x, tuple):
        return tuple(normalize(v) for v in x)

    # ndarray -> recurse
    if isinstance(x, np.ndarray):
        return normalize(x.tolist())

    return x

def ListToStr(L,SizeMax=50,Unit=None):
    if isinstance(L,np.ndarray) and len(L.shape)==1:
        L=L.tolist()
    if not isinstance(L, list):
        stop
    L = normalize(L)
    LEN=len(L)
    ss=", ".join(str(a) for a in L)
    if len(ss)>SizeMax:
        Ls=ss.split(", ")
        sMean=2+np.mean(np.float32(np.array([len(str(s)) for s in L])))
        Ns=int(SizeMax/sMean)
        dNs=Ns//2

        dNs=np.max([1,dNs])
        dNs=np.min([dNs,SizeMax])
        Ls1=Ls[0:dNs]+[" ...."]+Ls[-dNs:]
        ss=", ".join(s for s in Ls1)

    if Unit is None:
        ss="[%s], (len = %i)"%(ss,LEN)
    else:
        ss="[%s] %s, (len = %i)"%(ss,Unit,LEN)
        
    return ss
