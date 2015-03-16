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


