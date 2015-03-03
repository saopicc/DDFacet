import sharedarray.SharedArray as SharedArray
from Other import ModColor
import numpy as np
from Other import MyLogger
log=MyLogger.getLogger("NpShared")

def zeros(*args,**kwargs):
    return SharedArray.create(*args,**kwargs)

def ToShared(Name,A):

    try:
        a=SharedArray.create(Name,A.shape,dtype=A.dtype)
    except:
        print>>log, ModColor.Str("File %s exists, delete it..."%Name)
        DelArray(Name)
        a=SharedArray.create(Name,A.shape,dtype=A.dtype)

    a[:]=A[:]
    return a

def DelArray(Name):
    try:
        SharedArray.delete(Name)
    except:
        pass

def ListNames():
    ll=list(SharedArray.list())
    return [AR.name for AR in ll]
    
def DelAll(key=None):
    ll=ListNames()
    for name in ll:
        if key!=None:
            if key in name: DelArray(name)
        else:
            DelArray(name)

def GiveArray(Name):
    try:
        return SharedArray.attach(Name)
    except:
        return None


def DicoToShared(Prefix,Dico,DelInput=False):
    DicoOut={}
    print>>log, ModColor.Str("DicoToShared: start [prefix = %s]"%Prefix)
    for key in Dico.keys():
        if type(Dico[key])!=np.ndarray: continue
        #print "%s.%s"%(Prefix,key)
        ThisKeyPrefix="%s.%s"%(Prefix,key)
        print>>log, ModColor.Str("  %s -> %s"%(key,ThisKeyPrefix))
        ar=Dico[key]
        Shared=ToShared(ThisKeyPrefix,ar)
        DicoOut[key]=Shared
        if DelInput:
            del(Dico[key],ar)
            
    if DelInput:
        del(Dico)

    print>>log, ModColor.Str("DicoToShared: done")

    return DicoOut


def SharedToDico(Prefix):

    print>>log, ModColor.Str("SharedToDico: start [prefix = %s]"%Prefix)
    Lnames=ListNames()
    keys=[Name for Name in Lnames if Prefix in Name]
    if len(keys)==0: return None
    DicoOut={}
    for Sharedkey in keys:
        key=Sharedkey.split(".")[-1]
        print>>log, ModColor.Str("  %s -> %s"%(Sharedkey,key))
        Shared=GiveArray(Sharedkey)
        if Shared==None:
            print>>log, ModColor.Str("      None existing key"%(key))
            return None
        DicoOut[key]=Shared
    print>>log, ModColor.Str("SharedToDico: done")


    return DicoOut

def PackListSquareMatrix(Name,LArray):
    NArray=len(LArray)
    dtype=LArray[0].dtype
    TotSize=0
    for i in range(NArray):
        TotSize+=LArray[i].size


    # [N,shape0...shapeN,Arr0...ArrN]
    DelArray(Name)
    S=SharedArray.create(Name,(TotSize+NArray+1,),dtype=dtype)
    S[0]=NArray
    idx=1
    for i in range(NArray):
        A=LArray[i]
        S[idx]=A.shape[0]
        idx+=1

    for i in range(NArray):
        A=LArray[i]
        S[idx:idx+A.size]=A.ravel()
        idx+=A.size


def UnPackListSquareMatrix(Name):
    LArray=[]
    S=GiveArray(Name)

    NArray=int(S[0])
    idx=1

    ShapeArray=[]
    for i in range(NArray):
        ShapeArray.append(S[idx])
        idx+=1

    for i in range(NArray):
        shape=int(ShapeArray[i])
        size=shape**2
        A=S[idx:idx+size].reshape((shape,shape))
        LArray.append(A)
        idx+=A.size
    return LArray
  
    



# import SharedArray
# import ModColor

# def ToShared(Name,A):

#     try:
#         a=SharedArray.create(Name,A.shape,dtype=A.dtype)
#     except:
#         print ModColor.Str("File %s exists, delete it..."%Name)
#         DelArray(Name)
#         a=SharedArray.create(Name,A.shape,dtype=A.dtype)


#     a[:]=A[:]
#     return a

# def DelArray(Name):
#     SharedArray.delete(Name)
    
# def GiveArray(Name):
#     return SharedArray.attach(Name)
