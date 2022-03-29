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

from DDFacet.Other import logger
log = logger.getLogger("MyPickle")

import os
import six
if six.PY3:
    import pickle as cPickle
else:
    import cPickle
import pickle

def Save(Obj,fileout):
    #print "  Saving in %s ... "%fileout,
    cPickle.dump(Obj, open(fileout,'wb'), 2)
    #pickle.dump(Obj, open(fileout,'w'))
    #print "  done"

def Load(filein):
    
    try:
        G= cPickle.load( open( filein, "rb" ))
    except UnicodeDecodeError:
        log.print("Cannot read dicomodel: %s"%filein)
        D0 = cPickle.load( open( filein, "rb" ), encoding='bytes')
        log.print("  converting to Python3...")
        G = convert(D0)
        # NameOut="%s.Py3.DicoModel"%filein
        # log.print("Saving in %s"%NameOut)
        # Save(G,NameOut)
    return G

def convert(input):
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.items()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, bytes):
        return input.decode('ascii')
    else:
        return input

# def DicoPy2To3(DicoIn, DicoOut):
#     for key,value in DicoIn.items():
#         keyout=key
#         valout=value
#         if isinstance(key,bytes):
#             keyout=key.decode("ascii")
#         if isinstance(value,bytes):
#             DicoOut[keyout]=value.decode("ascii")
#         elif isinstance(valout,dict):
#             DicoOut[keyout] = {}
#             DicoOut[keyout] = DicoPy2To3(DicoIn[key], DicoOut[keyout])
#         elif isinstance(valout,list):
#             DicoOut[keyout] = ListPy2To3(value)
#         else:
#             DicoOut[keyout] = valout
#     return DicoOut

# def ListPy2To3(ListIn):
#     for v in ListIn:

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
        elif type(Obj)==dict or Obj.__class__.__name__=='SharedDict':
            #print "  key=%s is a dico"%key
            if not(os.path.isdir(DirDico)):
                #print "Create directory %s"%DirDico
                os.makedirs(DirDico)  

            DicoFile="%s/%s%s"%(DirDico,DICO_GEN_NAME,str(key))
            #print "Saving Dico in %s"%DicoFile
            DicoNPToFile(Obj,DicoFile)
            DicoSave[key]=DicoFile

        else:
            #print "  key=%s is not a numpy array"%key
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
