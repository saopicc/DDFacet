import numpy as np

def ListToStr(L,SizeMax=50,Unit=None):
    if isinstance(L,np.ndarray) and len(L.shape)==1:
        L=L.tolist()
    if not isinstance(L, list):
        stop
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
