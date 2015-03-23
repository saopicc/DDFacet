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
    WEIGHT=t.getcol("IMAGING_WEIGHT")
    t.close()
    ImShape=(1, 1, 1025, 1025)
    CellSizeRad=(20./3600)*np.pi/180
    CW=ClassWeighting(ImShape,CellSizeRad)
    #CW.CalcWeights(MS.uvw[199:200],WEIGHT[199:200,0:3],MS.flag_all[199:200,0:3],MS.ChanFreq[0:3],Weighting="Uniform")

    MS.flag_all.fill(0)

    # for i in [206]:#range(200,211):
    #     r0,r1=i,i+10
    #     print r0,r1
    #     uvw=np.float64(MS.uvw[r0:r1].copy())
    #     flags=MS.flag_all[r0:r1,0:3].copy()
    #     W=WEIGHT[r0:r1,0:3].copy()
    #     W.fill(1)
    #     freqs=MS.ChanFreq[0:3].copy()
    #     CW.CalcWeights(uvw,W,flags,freqs,Weighting="Uniform")

    WEIGHT.fill(1)
    MS.flag_all[MS.A0==MS.A1,:,:]=1
    WEIGHT[MS.flag_all[:,:,0]==1]=0

    CW.CalcWeights(MS.uvw,WEIGHT,MS.flag_all,MS.ChanFreq,Robust=-1,Weighting="Briggs")
    

class ClassWeighting():
    def __init__(self,ImShape,CellSizeRad):
        self.ImShape=ImShape
        self.CellSizeRad=CellSizeRad
        
    def CalcWeights(self,uvw,VisWeights,flags,freqs,Robust=0,Weighting="Briggs"):


        #u,v,_=uvw.T

        #Robust=-2
        nch,npol,npixIm,_=self.ImShape
        FOV=self.CellSizeRad*npixIm#/2

        #cell=1.5*4./(FOV)
        cell=2./(FOV)
        cell=4./(FOV)

        #wave=6.

        u=uvw[:,0].copy()
        v=uvw[:,1].copy()

        d=np.sqrt(u**2+v**2)
        VisWeights[d==0]=0
        uvmax=np.max(d)
        npix=2*(int(uvmax/cell)+1)
        if (npix%2)==0:
            npix+=1

        #npix=npixIm
        xc,yc=npix/2,npix/2





        
        if Weighting=="Briggs":
            print>>log, "Weighting in Briggs mode"
            print>>log, "Calculating imaging weights with Robust=%3.1f on an [%i,%i] grid"%(Robust,npix,npix)
            Mode=0
        elif Weighting=="Uniform":
            print>>log, "Weighting in Uniform mode"
            Mode=1
        elif Weighting=="Natural":
            print>>log, "Weighting in Natural mode"
            return VisWeights
        else:
            stop

        grid=np.zeros((npix,npix),dtype=np.float64)


        VisWeights=np.float64(VisWeights)
        flags=np.int32(flags)

        #print "u=",u
        #print "v=",v
        w=_pyGridder.pyGridderPoints(grid,
                                     flags,
                                     u,
                                     v,
                                     VisWeights,
                                     2*float(Robust),
                                     Mode,
                                     np.float32(freqs.flatten()),
                                     np.array([cell,cell],np.float64))


        # C=299792458.
        # uf=u.reshape((u.size,1))*freqs.reshape((1,freqs.size))/C
        # vf=v.reshape((v.size,1))*freqs.reshape((1,freqs.size))/C

        # x,y=np.int32(np.round(uf/cell))+xc,np.int32(np.round(vf/cell))+yc
        # x,y=(uf/cell)+xc,(vf/cell)+yc
        # condx=((x>0)&(x<npix))
        # condy=((y>0)&(y<npix))
        # ind=np.where((condx & condy))[0]
        # X=x[ind]
        # Y=y[ind]

        
        # import pylab
        # pylab.clf()
        # pylab.scatter(uf.flatten(),vf.flatten(),c=w.flatten(),lw=0,alpha=0.3)#,w[ind,0])
        # #pylab.imshow(w,interpolation="nearest")
        # #pylab.scatter(X,Y)#,w[ind,0])
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        
        return w
