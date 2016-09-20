import os
import pickle

def Save(Obj,fileout):
    #print "  Saving in %s ... "%fileout,
    pickle.dump(Obj, file(fileout,'w'))
    #print "  done"

def Load(filein):
    #print "  Loading from %s"%filein
    G= pickle.load( open( filein, "rb" ) )
    return G


import os
import numpy as np
ARRAY_GEN_NAME="np.array_"
def DicoNPToFile(Dico,FileOut):
    DicoSave={}
    FileOut=os.path.abspath(FileOut)
    DirArrays="%s.NpArray"%FileOut

    for key in Dico.keys():
        Obj=Dico[key]
        if type(Obj).__module__ == np.__name__:
            if not(os.path.isdir(DirArrays)):
                os.makedirs(DirArrays)  

            
            ThisArrayFile="%s/%s%s.npy"%(DirArrays,ARRAY_GEN_NAME,str(key))
            np.save(ThisArrayFile,Obj)
            DicoSave[key]=ThisArrayFile
        else:
            DicoSave[key]=Dico[key]

    Save(DicoSave,FileOut)

def FileToDicoNP(FileIn):
    
    D=Load(FileIn)
    
    DicoOut={}
    for key in D.keys():
        Obj=D[key]
        if type(Obj)==str:
            if ARRAY_GEN_NAME in Obj:
                A=np.load(Obj)
                DicoOut[key]=A
            else:
                DicoOut[key]=D[key]

        else:
            DicoOut[key]=D[key]
    return DicoOut
