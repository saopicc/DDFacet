
import numpy as np

def ReadFromNp(ra,dec,s):

    Cat=np.zeros((ra.shape[0],),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('Sref',np.float),('I',np.float),('Q',np.float),\
                                ('U',np.float),('V',np.float),('RefFreq',np.float),('alpha',np.float),('ESref',np.float),\
                                ('Ealpha',np.float),('kill',np.int),('Cluster',np.int),('Type',np.int),('Gmin',np.float),\
                                ('Gmaj',np.float),('Gangle',np.float),("Select",np.int),('l',np.float),('m',np.float),("Exclude",bool)])
    Cat=Cat.view(np.recarray)
    Cat.RefFreq=1.
    ns=ra.shape[0]
    Cat.ra[0:ns]=ra
    Cat.dec[0:ns]=dec
    Cat.I[0:ns]=s

    Cat=Cat[Cat.ra!=0.]
    Cat.Type[Cat.Gmaj>0.]=1

    Cat.Sref=Cat.I
    return Cat
