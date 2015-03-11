import numpy as np
from Gridder import _pyGridder

class ClassWeighting():
    def __init__(self,ImShape,CellSizeRad):
        self.ImShape=ImShape
        self.CellSizeRad=CellSizeRad
        
    def CalcWeights(self,uvw,w,Robust=0):
        u,v,_=uvw.T

        

        nch,npol,npixIm,_=self.ImShape
        FOV=self.CellSizeRad*npixIm#/2

        cell=2./(FOV)
        
        d=np.sqrt(u**2+v**2)
        w[d==0]=0
        uvmax=np.max(d)
        npix=2*(int(uvmax/cell)+1)

        #npix=npixIm
        xc,yc=npix/2,npix/2


        grid=np.zeros((npix,npix),dtype=np.float32)



        

        x,y=np.int64(u/cell)+xc,np.int64(v/cell)+yc

        condx=((x>0)&(x<npix))
        condy=((y>0)&(y<npix))
        ind=np.where(np.logical_not(condx & condy))[0]
        x[ind]=0
        y[ind]=0
        w[ind]=0
        
        w=_pyGridder.pyGridderPoints(grid.astype(np.float64),x.astype(np.int32),y.astype(np.int32),w.astype(np.float64),float(Robust))



        # import pylab
        # pylab.clf()
        # pylab.scatter(d,w)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop
        
        
        
        return w
