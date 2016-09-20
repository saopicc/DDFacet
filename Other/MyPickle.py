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
DICO_GEN_NAME="DICO_"

def DicoNPToFile(Dico,FileOut):
    DicoSave={}
    FileOut=os.path.abspath(FileOut)
    DirArrays="%s.SubArray"%FileOut
    DirDico="%s.SubDico"%FileOut

    #    print "============================="

    for key in Dico.keys():
        Obj=Dico[key]
        #print "key %s has type %s"%(str(key),str(type(Obj)))
        if type(Obj).__module__ == np.__name__:
            if not(os.path.isdir(DirArrays)):
                #print "Create directory"
                os.makedirs(DirArrays)  
            
            ThisArrayFile="%s/%s%s.npy"%(DirArrays,ARRAY_GEN_NAME,str(key))
            #print "  Save %s"%ThisArrayFile
            np.save(ThisArrayFile,Obj)
            #print "    Save OK"
            DicoSave[key]=ThisArrayFile

        # ============================================
        # Need to be recursive for dictionnaries containinung nparrays 
        elif type(Obj)==dict:
            #print "  key=%s is a dico"%key
            if not(os.path.isdir(DirDico)):
                #print "Create directory %s"%DirDico
                os.makedirs(DirDico)  

            DicoFile="%s/%s%s"%(DirDico,DICO_GEN_NAME,str(key))
            #print "Saving Dico in %s"%DicoFile
            DicoNPToFile(Obj,DicoFile)
            DicoSave[key]=DicoFile

        else:
            print "  key=%s is not a numpy array"%key
            DicoSave[key]=Dico[key]
    

    #print "Pickled"
    #print DicoSave
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
            elif DICO_GEN_NAME in Obj:
                A=FileToDicoNP(Obj)
                DicoOut[key]=A
            else:
                DicoOut[key]=D[key]

        else:
            DicoOut[key]=D[key]



    return DicoOut
