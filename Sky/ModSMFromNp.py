
import numpy as np

def ReadFromNp(Cat0):

    ns=Cat0.shape[0]
    Cat=np.zeros((ns,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('Sref',np.float),('I',np.float),('Q',np.float),\
                                ('U',np.float),('V',np.float),('RefFreq',np.float),('alpha',np.float),('ESref',np.float),\
                                ('Ealpha',np.float),('kill',np.int),('Cluster',np.int),('Type',np.int),('Gmin',np.float),\
                                ('Gmaj',np.float),('Gangle',np.float),("Select",np.int),('l',np.float),('m',np.float),("Exclude",bool)])
    Cat=Cat.view(np.recarray)
    Cat.RefFreq=1.
    Cat.ra[0:ns]=Cat0.ra
    Cat.dec[0:ns]=Cat0.dec
    Cat.I[0:ns]=Cat0.I
    if "Gmin" in Cat0.dtype.fields.keys():
        Cat.Gmin[0:ns]=Cat0.Gmin
        Cat.Gmaj[0:ns]=Cat0.Gmaj
        Cat.Gangle[0:ns]=Cat0.Gangle


    Cat=Cat[Cat.ra!=0.]
    Cat.Type[Cat.Gmaj>0.]=1

    Cat.Sref=Cat.I
    return Cat
