

from __future__ import division, absolute_import, print_function
import numpy.lib.recfunctions
import numpy as np

def AppendField(dataAll, FName, dataType):
    dataCol=np.zeros((dataAll.shape[0],),dtype=dataType)
    dataOut=numpy.lib.recfunctions.append_fields(dataAll, FName, dataCol, usemask=False)
    dataOut=dataOut.view(np.recarray)
    return dataOut

def RemoveField(a, FName):
    names = list(a.dtype.names)
    if FName not in names: return a
    new_names = [n for n in names if n!=FName]
    b = a[new_names]
    return b.copy()
