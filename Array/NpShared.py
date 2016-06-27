#import sharedarray.SharedArray as SharedArray
import SharedArray
from DDFacet.Other import ModColor
import numpy as np
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("NpShared")

def zeros(Name,*args,**kwargs):
    try:
        return SharedArray.create(Name,*args,**kwargs)
    except:
        DelArray(Name)
        return SharedArray.create(Name,*args,**kwargs)

def SizeShm():
    L=ListNames()
    S=0
    for l in L:
        A=GiveArray(l)
        if A!=None:
            S+=A.nbytes
    return float(S)/(1024**2)


def ToShared(Name,A):

    #print "dtype, shape = %s, %s"%(str(A.dtype), str(A.shape))
    #print "Number of nan in %s: %i"%(Name,np.count_nonzero(np.isnan(A)))
    #print "Number of inf in %s: %i"%(Name,np.count_nonzero(np.isinf(A)))

    try:
        a=SharedArray.create(Name,A.shape,dtype=A.dtype)
    except:
        print>>log, ModColor.Str("File %s exists, delete it..."%Name)
        DelArray(Name)
        a=SharedArray.create(Name,A.shape,dtype=A.dtype)

    # Ex=Exists(Name)
    # if Ex:
    #     DelArray(Name)
    # a=SharedArray.create(Name,A.shape,dtype=A.dtype)



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

def Exists(Name):
    LNames=ListNames()
    Exists=False
    for ThisName in LNames:
        if Name==ThisName:
            Exists=True
    return Exists
    


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
        if type(Shared)==type(None):
            print>>log, ModColor.Str("      None existing key %s"%(key))
            return None
        DicoOut[key]=Shared
    print>>log, ModColor.Str("SharedToDico: done")


    return DicoOut

####################################################
####################################################

def PackListArray(Name,LArray):
    DelArray(Name)

    NArray=len(LArray)
    ListNDim=[len(LArray[i].shape) for i in range(len(LArray))]
    NDimTot=np.sum(ListNDim)
    # [NArray,NDim0...NDimN,shape0...shapeN,Arr0...ArrN]

    dS=LArray[0].dtype
    TotSize=0
    for i in range(NArray):
        TotSize+=LArray[i].size


    S=SharedArray.create(Name,(1+NArray+NDimTot+TotSize,),dtype=dS)
    S[0]=NArray
    idx=1
    # write ndims
    for i in range(NArray):
        S[idx]=ListNDim[i]
        idx+=1

    # write shapes
    for i in range(NArray):
        ndim=ListNDim[i]
        A=LArray[i]
        S[idx:idx+ndim]=A.shape
        idx+=ndim

    # write arrays
    for i in range(NArray):
        A=LArray[i]
        S[idx:idx+A.size]=A.ravel()
        idx+=A.size


def UnPackListArray(Name):
    S=GiveArray(Name)

    NArray=np.int32(S[0].real)
    idx=1

    # read ndims
    ListNDim=[]
    for i in range(NArray):
        ListNDim.append(np.int32(S[idx].real))
        idx+=1

    # read shapes
    ListShapes=[]
    for i in range(NArray):
        ndim=ListNDim[i]
        shape=np.int32(S[idx:idx+ndim].real)
        ListShapes.append(shape)
        idx+=ndim

    # read values
    ListArray=[]
    for i in range(NArray):
        shape=ListShapes[i]
        size=np.prod(shape)
        A=S[idx:idx+size].reshape(shape)
        ListArray.append(A)
        idx+=size
    return ListArray

####################################################
####################################################

def PackListSquareMatrix(Name,LArray):
    DelArray(Name)

    NArray=len(LArray)
    dtype=LArray[0].dtype
    TotSize=0
    for i in range(NArray):
        TotSize+=LArray[i].size


    # [N,shape0...shapeN,Arr0...ArrN]
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

    NArray=np.int32(S[0].real)
    idx=1

    ShapeArray=[]
    for i in range(NArray):
        ShapeArray.append(np.int32(S[idx].real))
        idx+=1

    print>>log, ShapeArray

    for i in range(NArray):
        shape=np.int32(ShapeArray[i].real)
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
