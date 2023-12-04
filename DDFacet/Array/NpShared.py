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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

#import sharedarray.SharedArray as SharedArray
import SharedArray
from DDFacet.Other import ModColor
import numpy as np
from DDFacet.Other import logger
import traceback
log = logger.getLogger("NpShared")
import os.path


def zeros(Name, *args, **kwargs):
    try:
        return SharedArray.create(Name, *args, **kwargs)
    except:
        DelArray(Name)
        return SharedArray.create(Name, *args, **kwargs)


def SizeShm():
    from subprocess import check_output
    import subprocess
    try:
        cmd = "df -h | grep shm"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        s=ps.communicate()[0]
        
        ss=s.split(" ")

        ss=[i for i in ss if i!=""][2]
        if "G" in ss:
            S=float(ss.replace("G",""))*1024
        elif "M" in ss:
            S=float(ss.replace("M",""))
        elif "K" in ss:
            S=float(ss.replace("K",""))/1024

        #S=float(check_output(["du", "-sc","/dev/shm/"]).split("\t")[0])/1024
    except:
        S=None
    return S
    # L = ListNames()
    # S = 0
    # for l in L:
    #     A = GiveArray(l)
    #     if A is not None:
    #         S += A.nbytes
    # return float(S)/(1024**2)


def CreateShared(Name, shape, dtype):
    try:
        a = SharedArray.create(str(Name), shape, dtype=dtype)
    except OSError:
        print(ModColor.Str("File %s exists, deleting" % Name), file=log)
        DelArray(str(Name))
        a = SharedArray.create(str(Name), shape, dtype=dtype)
    return a

def ToShared(Name, A):

    a = CreateShared(str(Name), A.shape, A.dtype)
    np.copyto(a,A)
    return a

def DelArray(Name):
    try:
        SharedArray.delete(str(Name))
    except:
        pass

_locking = True

def Lock (array):
    global _locking
    if _locking:
        try:
                SharedArray.mlock(array)
        except:
            print("Warning: Cannot lock memory. Try updating your kernel security settings.", file=log)
            _locking = False

def Unlock (array):
    global _locking
    if _locking:
        try:
                SharedArray.munlock(array)
        except:
            print("Warning Cannot unlock memory. Try updating your kernel security settings.", file=log)
            _locking = False


def ListNames():
    ll = list(SharedArray.list())
    return [str(AR.name.decode("ascii")) for AR in ll]


def DelAll(key=None):
    ll = ListNames()
    for name in ll:
        if key is not None:
            if str(key) in str(name):
                DelArray(str(name))
        else:
            DelArray(str(name))


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
        print("Error loading",Name)
        # why are we not raising this serious error
        #traceback.print_exc()
        #return None
        raise e


def Exists(Name):
    Name = str(Name)
    if Name.startswith("file://"):
        return os.path.exists(Name[7:])
    if Name.startswith("shm://"):
        Name = Name[6:]
    return Name in ListNames()


def DicoToShared(Prefix, Dico, DelInput=False):
    DicoOut = {}
    print(ModColor.Str("DicoToShared: start [prefix = %s]" % Prefix), file=log)
    for key in Dico.keys():
        if not isinstance(Dico[key], np.ndarray):
            continue
        # print "%s.%s"%(Prefix,key)
        ThisKeyPrefix = "%s.%s" % (Prefix, key)
        print(ModColor.Str("  %s -> %s" % (key, ThisKeyPrefix)), file=log)
        ar = Dico[key]
        Shared = ToShared(ThisKeyPrefix, ar)
        DicoOut[key] = Shared
        if DelInput:
            del(Dico[key], ar)

    if DelInput:
        del(Dico)

    print(ModColor.Str("DicoToShared: done"), file=log)
    #print ModColor.Str("DicoToShared: done")

    return DicoOut


def SharedToDico(Prefix):

    print(ModColor.Str("SharedToDico: start [prefix = %s]" % Prefix), file=log)
    Lnames = ListNames()
    keys = [Name for Name in Lnames if Prefix in Name]
    if len(keys) == 0:
        return None
    DicoOut = {}
    for Sharedkey in keys:
        key = Sharedkey.split(".")[-1]
        print(ModColor.Str("  %s -> %s" % (Sharedkey, key)), file=log)
        Shared = GiveArray(Sharedkey)
        if isinstance(Shared, type(None)):
            print(ModColor.Str("      None existing key %s" % (key)), file=log)
            return None
        DicoOut[key] = Shared
    print(ModColor.Str("SharedToDico: done"), file=log)

    return DicoOut

####################################################
####################################################


def PackListArray(Name, LArray):
    Name = str(Name)
    DimName = Name+'.dimensions'
    DatName = Name+'.data'

    DelArray(DimName)
    DelArray(DatName)

    NArray = len(LArray)
    ListNDim = [len(LArray[i].shape) for i in range(len(LArray))]
    NDimTot = np.sum(ListNDim)
    # [NArray,NDim0...NDimN,shape0...shapeN,Arr0...ArrN]

    dS = LArray[0].dtype
    TotSize = 0
    for i in range(NArray):
        TotSize += LArray[i].size

    Dim = SharedArray.create(DimName, 1+NArray+NDimTot, dtype=int)
    Dat = SharedArray.create(DatName, TotSize, dtype=dS)
    Dim[0] = NArray
    didx = 1
    # write ndims
    for i in range(NArray):
        Dim[didx] = ListNDim[i]
        didx += 1

    # write shapes
    for i in range(NArray):
        ndim = ListNDim[i]
        A = LArray[i]
        Dim[didx:didx+ndim] = A.shape
        didx += ndim

    # write arrays
    idx = 0
    for i in range(NArray):
        A = LArray[i]
        Dat[idx:idx+A.size] = A.ravel()
        idx += A.size


def UnPackListArray(Name):
    Name = str(Name)
    DimName = Name+'.dimensions'
    DatName = Name+'.data'
    Dim = GiveArray(DimName)
    Dat = GiveArray(DatName)

    NArray = Dim[0]
    idx = 1

    # read ndims
    ListNDim = []
    for i in range(NArray):
        ListNDim.append(Dim[idx])
        idx += 1

    # read shapes
    ListShapes = []
    for i in range(NArray):
        ndim = ListNDim[i]
        shape = Dim[idx:idx+ndim]
        ListShapes.append(shape)
        idx += ndim

    idx = 0
    # read values
    ListArray = []
    for i in range(NArray):
        shape = ListShapes[i]
        size = np.prod(shape)
        A = Dat[idx:idx+size].reshape(shape)
        ListArray.append(A)
        idx += size
    return ListArray

####################################################
####################################################


def PackListSquareMatrix(shared_dict, Name, LArray):
    Name = str(Name)
    NArray = len(LArray)
    dtype = LArray[0].dtype
    TotSize = 0
    for i in range(NArray):
        TotSize += LArray[i].size

    # [N,shape0...shapeN,Arr0...ArrN]
    S = shared_dict.addSharedArray(Name, (TotSize+NArray+1,), dtype=dtype)
    S[0] = NArray
    idx = 1
    for i in range(NArray):
        A = LArray[i]
        S[idx] = A.shape[0]
        idx += 1

    for i in range(NArray):
        A = LArray[i]
        S[idx:idx+A.size] = A.ravel()
        idx += A.size


def UnPackListSquareMatrix(Array):
    LArray = []
    S = GiveArray(Array) if type(Array) is str else Array

    NArray = np.int32(S[0].real)
    idx = 1

    ShapeArray = []
    for i in range(NArray):
        ShapeArray.append(np.int32(S[idx].real))
        idx += 1

    print(ShapeArray, file=log)

    for i in range(NArray):
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
