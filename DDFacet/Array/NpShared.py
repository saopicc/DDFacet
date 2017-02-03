'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

#import sharedarray.SharedArray as SharedArray
import SharedArray
from DDFacet.Other import ModColor
import numpy as np
from DDFacet.Other import MyLogger
import traceback
log = MyLogger.getLogger("NpShared")
import os.path


def zeros(Name, *args, **kwargs):
    try:
        return SharedArray.create(Name, *args, **kwargs)
    except:
        DelArray(Name)
        return SharedArray.create(Name, *args, **kwargs)


def SizeShm():
    L = ListNames()
    S = 0
    for l in L:
        A = GiveArray(l)
        if A is not None:
            S += A.nbytes
    return float(S)/(1024**2)


def CreateShared(Name, shape, dtype):
    try:
        a = SharedArray.create(Name, shape, dtype=dtype)
    except OSError:
        print>> log, ModColor.Str("File %s exists, deleting" % Name)
        DelArray(Name)
        a = SharedArray.create(Name, shape, dtype=dtype)
    return a

def ToShared(Name, A):

    a = CreateShared(Name, A.shape, A.dtype)
    a[:] = A[:]
    return a

def DelArray(Name):
    try:
        SharedArray.delete(Name)
    except:
        pass

_locking = True

def Lock (array):
    global _locking
    if _locking:
        try:
                SharedArray.mlock(array)
        except:
            print>> log, "Warning: Cannot lock memory. Try updating your kernel security settings."
            _locking = False

def Unlock (array):
    global _locking
    if _locking:
        try:
                SharedArray.munlock(array)
        except:
            print>> log, "Warning Cannot unlock memory. Try updating your kernel security settings."
            _locking = False


def ListNames():
    ll = list(SharedArray.list())
    return [AR.name for AR in ll]


def DelAll(key=None):
    ll = ListNames()
    for name in ll:
        if key is not None:
            if key in name:
                DelArray(name)
        else:
            DelArray(name)


def GiveArray(Name):
    # return SharedArray.attach(Name)
    try:
        return SharedArray.attach(Name)
    except Exception as e:  # as exception:
        # #print str(e)
        # print
        # print "Exception for key [%s]:"%Name
        # print "   %s"%(str(e))
        # print
        print "Error loading",Name
        traceback.print_exc()
        return None


def Exists(Name):
    if Name.startswith("file://"):
        return os.path.exists(Name[7:])
    if Name.startswith("shm://"):
        Name = Name[6:]
    return Name in ListNames()


def DicoToShared(Prefix, Dico, DelInput=False):
    DicoOut = {}
    print>>log, ModColor.Str("DicoToShared: start [prefix = %s]" % Prefix)
    for key in Dico.keys():
        if not isinstance(Dico[key], np.ndarray):
            continue
        # print "%s.%s"%(Prefix,key)
        ThisKeyPrefix = "%s.%s" % (Prefix, key)
        print>>log, ModColor.Str("  %s -> %s" % (key, ThisKeyPrefix))
        ar = Dico[key]
        Shared = ToShared(ThisKeyPrefix, ar)
        DicoOut[key] = Shared
        if DelInput:
            del(Dico[key], ar)

    if DelInput:
        del(Dico)

    print>>log, ModColor.Str("DicoToShared: done")
    #print ModColor.Str("DicoToShared: done")

    return DicoOut


def SharedToDico(Prefix):

    print>>log, ModColor.Str("SharedToDico: start [prefix = %s]" % Prefix)
    Lnames = ListNames()
    keys = [Name for Name in Lnames if Prefix in Name]
    if len(keys) == 0:
        return None
    DicoOut = {}
    for Sharedkey in keys:
        key = Sharedkey.split(".")[-1]
        print>>log, ModColor.Str("  %s -> %s" % (Sharedkey, key))
        Shared = GiveArray(Sharedkey)
        if isinstance(Shared, type(None)):
            print>>log, ModColor.Str("      None existing key %s" % (key))
            return None
        DicoOut[key] = Shared
    print>>log, ModColor.Str("SharedToDico: done")

    return DicoOut

####################################################
####################################################


def PackListArray(Name, LArray):
    DelArray(Name)

    NArray = len(LArray)
    ListNDim = [len(LArray[i].shape) for i in xrange(len(LArray))]
    NDimTot = np.sum(ListNDim)
    # [NArray,NDim0...NDimN,shape0...shapeN,Arr0...ArrN]

    dS = LArray[0].dtype
    TotSize = 0
    for i in xrange(NArray):
        TotSize += LArray[i].size

    S = SharedArray.create(Name, (1+NArray+NDimTot+TotSize,), dtype=dS)
    S[0] = NArray
    idx = 1
    # write ndims
    for i in xrange(NArray):
        S[idx] = ListNDim[i]
        idx += 1

    # write shapes
    for i in xrange(NArray):
        ndim = ListNDim[i]
        A = LArray[i]
        S[idx:idx+ndim] = A.shape
        idx += ndim

    # write arrays
    for i in xrange(NArray):
        A = LArray[i]
        S[idx:idx+A.size] = A.ravel()
        idx += A.size


def UnPackListArray(Name):
    S = GiveArray(Name)

    NArray = np.int32(S[0].real)
    idx = 1

    # read ndims
    ListNDim = []
    for i in xrange(NArray):
        ListNDim.append(np.int32(S[idx].real))
        idx += 1

    # read shapes
    ListShapes = []
    for i in xrange(NArray):
        ndim = ListNDim[i]
        shape = np.int32(S[idx:idx+ndim].real)
        ListShapes.append(shape)
        idx += ndim

    # read values
    ListArray = []
    for i in xrange(NArray):
        shape = ListShapes[i]
        size = np.prod(shape)
        A = S[idx:idx+size].reshape(shape)
        ListArray.append(A)
        idx += size
    return ListArray

####################################################
####################################################


def PackListSquareMatrix(shared_dict, Name, LArray):

    NArray = len(LArray)
    dtype = LArray[0].dtype
    TotSize = 0
    for i in xrange(NArray):
        TotSize += LArray[i].size

    # [N,shape0...shapeN,Arr0...ArrN]
    S = shared_dict.addSharedArray(Name, (TotSize+NArray+1,), dtype=dtype)
    S[0] = NArray
    idx = 1
    for i in xrange(NArray):
        A = LArray[i]
        S[idx] = A.shape[0]
        idx += 1

    for i in xrange(NArray):
        A = LArray[i]
        S[idx:idx+A.size] = A.ravel()
        idx += A.size


def UnPackListSquareMatrix(Array):
    LArray = []
    S = GiveArray(Array) if type(Array) is str else Array

    NArray = np.int32(S[0].real)
    idx = 1

    ShapeArray = []
    for i in xrange(NArray):
        ShapeArray.append(np.int32(S[idx].real))
        idx += 1

    print>>log, ShapeArray

    for i in xrange(NArray):
        shape = np.int32(ShapeArray[i].real)
        size = shape**2
        A = S[idx:idx+size].reshape((shape, shape))
        LArray.append(A)
        idx += A.size
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
