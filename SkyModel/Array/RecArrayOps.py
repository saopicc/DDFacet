import numpy.lib.recfunctions
import numpy as np

def AppendField(dataAll,(FName,dataType)):
    dataCol=np.zeros((dataAll.shape[0],),dtype=dataType)

    dataOut=numpy.lib.recfunctions.append_fields(dataAll, FName, dataCol, usemask=False)
    dataOut=dataOut.view(np.recarray)
    return dataOut
