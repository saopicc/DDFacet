from __future__ import division, absolute_import, print_function

import numpy as np

def ReadFromNp(Cat0):

    ns=Cat0.shape[0]
    Cat=np.zeros((ns,),dtype=[('Name','|S200'),('ra',float),('dec',float),('Sref',float),('I',float),('Q',float),\
                                ('U',float),('V',float),('RefFreq',float),('alpha',float),('ESref',float),\
                                ('Ealpha',float),('kill',int),('Cluster',int),('Type',int),('Gmin',float),\
                                ('Gmaj',float),('Gangle',float),("Select",int),('l',float),('m',float),("Exclude",bool)])
    Cat=Cat.view(np.recarray)
    Cat.RefFreq=1.
    Cat.ra[0:ns]=Cat0.ra
    Cat.dec[0:ns]=Cat0.dec
    Cat.I[0:ns]=Cat0.I
    if "Gmin" in list(Cat0.dtype.fields.keys()):
        Cat.Gmin[0:ns]=Cat0.Gmin
        Cat.Gmaj[0:ns]=Cat0.Gmaj
        Cat.Gangle[0:ns]=Cat0.Gangle


    Cat=Cat[Cat.ra!=0.]
    Cat.Type[Cat.Gmaj>0.]=1

    Cat.Sref=Cat.I
    return Cat
