import numpy as np
from Gridder import _pyGridder
import MyLogger
import ModColor
log=MyLogger.getLogger("ClassWeighting")


#import ImagingWeights
import ClassMS
from pyrap.tables import table

def test():
    MS=ClassMS.ClassMS("/media/6B5E-87D0/killMS2/TEST/Simul/0000.MS")
    t=table(MS.MSName,ack=False)
    WEIGHT=t.getcol("WEIGHT")
    t.close()
    ImShape=(1, 1, 375, 375)
    CellSizeRad=(1./3600)*np.pi/180
    CW=ClassWeighting(ImShape,CellSizeRad)
    CW.CalcWeights(MS.uvw,WEIGHT,MS)
    

class ClassWeighting():
    def __init__(self,ImShape,CellSizeRad):
        self.ImShape=ImShape
        self.CellSizeRad=CellSizeRad
        
    def CalcWeights(self,uvw,VisWeights,Robust=0):


        #u,v,_=uvw.T
        


        #Robust=-2
        nch,npol,npixIm,_=self.ImShape
        FOV=self.CellSizeRad*npixIm#/2

        #cell=1.5*4./(FOV)
        cell=2./(FOV)
        cell=4./(FOV)

        #wave=6.
        u,v,_=uvw.T#/wave
        
        d=np.sqrt(u**2+v**2)
        VisWeights[d==0]=0
        uvmax=np.max(d)
        npix=2*(int(uvmax/cell)+1)
        if (npix%2)==0:
            npix+=1

        #npix=npixIm
        xc,yc=npix/2,npix/2


        grid=np.zeros((npix,npix),dtype=np.float32)

        print>>log, "Calculating imaging weights with Robust=%3.1f on an [%i,%i] grid"%(Robust,npix,npix)


        

        x,y=np.int64(np.round(u/cell))+xc,np.int64(np.round(v/cell))+yc

        condx=((x>0)&(x<npix))
        condy=((y>0)&(y<npix))
        ind=np.where(np.logical_not(condx & condy))[0]
        x[ind]=0
        y[ind]=0
        VisWeights[ind]=0
        Mode=0
        x,y=np.int64(np.round(u/cell)),np.int64(np.round(v/cell))
        w=_pyGridder.pyGridderPoints(grid.astype(np.float64),x.astype(np.int32),y.astype(np.int32),VisWeights.astype(np.float64),2*float(Robust),Mode)

        #IW=ImagingWeights.ImagingWeight(weighttype="robust",rmode="normal",robustness=-2)
        #IW=ImagingWeights.ImagingWeight(weighttype="uniform",rmode="normal",robustness=-2)
        #w=IW.density(self.CellSizeRad, self.ImShape,MS.ChanFreq.flatten(), MS.uvw,VisWeights)





        # import pylab
        # pylab.clf()
        # #pylab.scatter(d,w)
        # pylab.imshow(w,interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop

        #IW.set_density(w,self.CellSizeRad)
        #w=IW.weightDensityDependent(MS.uvw, MS.ChanFreq.flatten(), MS.flag_all, VisWeights)
        # freqs=np.array([3e8/6.00])

        # import pylab
        # pylab.clf()
        # pylab.scatter(d,w)
        # #pylab.imshow(w,interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop
        
        return w
