import numpy as np
import math
import DDFacet.cbuild.Gridder._pyGridder as _pyGridder
import DDFacet.cbuild.Gridder._pyWeightingCore as WeightingCore
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassWeighting")


#import ImagingWeights
from DDFacet.Data import ClassMS
from pyrap.tables import table

def test(field=0,weight="Uniform"):
    print>>log,"reading test MS"
#    MS=ClassMS.ClassMS("/media/6B5E-87D0/killMS2/TEST/Simul/0000.MS")
    MS=ClassMS.ClassMS("CYG-B-test.MS",DoReadData=False)
    t=table(MS.MSName,ack=False).query("FIELD_ID==%d"%field)
    WEIGHT=t.getcol("WEIGHT_SPECTRUM")
    flag_all=t.getcol("FLAG")
    uvw=t.getcol("UVW")
    t.close()
    ImShape=(1, 1, 6125, 6125)
    CellSizeRad=(0.06/3600)*np.pi/180
    CW=ClassWeighting(ImShape,CellSizeRad)
    #CW.CalcWeights(MS.uvw[199:200],WEIGHT[199:200,0:3],MS.flag_all[199:200,0:3],MS.ChanFreq[0:3],Weighting="Uniform")


    flag_all.fill(0)

    # for i in [206]:#range(200,211):
    #     r0,r1=i,i+10
    #     print r0,r1
    #     uvw=np.float64(MS.uvw[r0:r1].copy())
    #     flags=MS.flag_all[r0:r1,0:3].copy()
    #     W=WEIGHT[r0:r1,0:3].copy()
    #     W.fill(1)
    #     freqs=MS.ChanFreq[0:3].copy()
    #     CW.CalcWeights(uvw,W,flags,freqs,Weighting="Uniform")

    WEIGHT = WEIGHT.mean(axis=2)
    WEIGHT.fill(1)
    #flag_all[MS.A0==MS.A1]=1
    #WEIGHT[MS.flag_all[:,:,0]==1]=0
    print>>log,"calculating test weights for shape %s"%(flag_all.shape,)
    CW.CalcWeights(uvw,WEIGHT,flag_all,MS.ChanFreq,Robust=0,Weighting=weight)
    

_cc = 299792458

class ClassWeighting():
    def __init__(self,ImShape,CellSizeRad):
        self.ImShape=ImShape
        self.CellSizeRad=CellSizeRad

    def CalcWeights(self,uvw_weights_flags_freqs,Robust=0,Weighting="Briggs",Super=1):

        if Weighting=="Natural":
            print>>log, "Weighting in Natural mode"
            return [ x[1] for x in uvw_weights_flags_freqs ]

        nch,npol,npixIm,_ = self.ImShape
        FOV = self.CellSizeRad*npixIm
        cell =1./(Super*FOV)


        Weighting = Weighting.lower()
        if Weighting == "natural":
            print>>log, "Weighting in Natural mode"
            return VisWeights
        print>>log,"initializing weighting grid"

        # find max grid extent by consiering _unflagged_ UVs
        xymax = 0
        for uv, weights, flags, freqs in uvw_weights_flags_freqs:
            # max |u|,|v| in lambda 
            uvmax = abs(uv)[~flags,:].max() * freqs.max()/_cc
            xymax = max(xymax, int(math.floor(uvmax/cell)))
        xymax += 1
        # grid will be from [-xymax,xymax] in U and [0,xymax] in V
        npixx = xymax*2+1
        npixy = xymax+1
        npix = npixx*npixy

        print>>log, "Calculating imaging weights on an [%i,%i] grid with cellsize %g (method 3)"%(npixx,npixy,cell)
        grid = np.zeros(npix,np.float64)

        weights_index = [None]*len(uvw_weights_flags_freqs)

        for iMS, (uv, weights, flags, freqs) in enumerate(uvw_weights_flags_freqs):
            # flip sign of negative v values -- we'll only grid the top half of the plane
            uv[ uv[:,1]<0 ] *= -1
            # convert u/v to lambda, and then to pixel offset
            uv = uv[...,np.newaxis]*freqs[np.newaxis,np.newaxis,:]/_cc
            uv = np.floor(uv/cell).astype(int)
            # u is offset, v isn't since it's the top half
            x = uv[:,0,:]
            y = uv[:,1,:]
            x += xymax   # offset, since X grid starts at -xymax
            # convert to index array
            index = y*npixx + x
            weights_index[iMS] = weights, index
            del uv
            print>>log,"Accumulating weights (%d/%d)"%(iMS+1, len(uvw_weights_flags_freqs))
            # accumulate onto grid
            # print>>log,weights,index
            WeightingCore.accumulate_weights_onto_grid_1d(grid,weights.ravel(),index.ravel())

        # # this is the only slow part -- takes ~20 mins on CygA data (3 million rows x 256 channels so just under a billion uv-points)
        # print>>log, "Calculating imaging weights on an [%i,%i] grid with cellsize %g (method 1)"%(npixx,npixy,cell)
        # grid = np.zeros(npix,np.float64)
        # index_iter = zip(index.ravel(),VisWeights.ravel())
        # def gridinc (dum,arg):
        #    x,w = arg
        #    grid[x] += w
        # reduce(gridinc,index_iter)
        # print>>log,"weight grid computed"
        # # first attempt with cython using nditer. Not so effective: ~25 mins
        # print>>log, "Calculating imaging weights on an [%i,%i] grid with cellsize %g (method 2)"%(npixx,npixy,cell)
        # grid = np.zeros(npix,np.float64)
        # WeightingCore.accumulate_weights_onto_grid_using_nditer(grid,VisWeights,index)
        # print>>log,"weight grid computed"
        # second attept with cython using a simple for loop over 1D arrays: bingo, 3 seconds

        if Weighting == "uniform":
#            print>>log,"adjusting grid to uniform weight"
 #           grid[grid!=0] = 1/grid[grid!=0]
            print>>log,("applying uniform weighting (super=%.2f)"%Super)
            for weights, index in weights_index:
                weights /= grid[index]

        elif Weighting == "briggs" or Weighting == "robust":
            print>>log,("applying Briggs weighting (robust=%.2f, super=%.2f)"%(Robust, Super))
            avgW = (grid**2).sum() / grid.sum()
            numeratorSqrt = 5.0 * 10**(-Robust)
            sSq = numeratorSqrt**2 / avgW
            grid = 1/(1+grid*sSq)
            for weights, index in weights_index:
                weights *= grid[index]
                
        else:
            raise ValueError("unknown weighting \"%s\""%Weighting)

        print>>log,"weights computed"
        return [ weights for weights, index in weights_index ]



        
    def CalcWeightsOld(self,uvw,VisWeights,flags,freqs,Robust=0,Weighting="Briggs",Super=1):


        #u,v,_=uvw.T/*

        #Robust=-2
        nch,npol,npixIm,_=self.ImShape
        FOV=self.CellSizeRad*npixIm#/2

        #cell=1.5*4./(FOV)
        cell=1./(Super*FOV)
        #cell=4./(FOV)

        #wave=6.

        u=uvw[:,0].copy()
        v=uvw[:,1].copy()

        d=np.sqrt(u**2+v**2)
        VisWeights[d==0]=0
#        Lmean=3e8/np.mean(freqs)
        Lmin=3e8/np.max(freqs)

        uvmax=np.max(d)/Lmin
        #(1./self.CellSizeRad)#/2#np.max(d)
        npix=2*(int(uvmax/cell)+1)
        if (npix%2)==0:
            npix+=1

        #npix=npixIm
        xc,yc=npix/2,npix/2


        VisWeights=np.float64(VisWeights)
        #VisWeights.fill(1.)
        print>>log,"image grid cell is %g"%cell
        
        if Weighting=="Briggs":
            print>>log, "Weighting in Briggs mode (robust=%.1f, super=%.1f)"%(Robust,Super)
            print>>log, "Calculating imaging weights on an [%i,%i] grid"%(npix,npix)
            print>>log, ""
            Mode=0
        elif Weighting=="Uniform":
            print>>log, "Weighting in Uniform mode (super=%.1f)"%(Super)
            print>>log, "Calculating imaging weights on an [%i,%i] grid"%(npix,npix)
            Mode=1
        elif Weighting=="Natural":
            print>>log, "Weighting in Natural mode"
            return VisWeights
        else:
            stop

        grid=np.zeros((npix,npix),dtype=np.float64)


        flags=np.float32(flags)
        WW=np.mean(1.-flags,axis=2)
        VisWeights*=WW
        
        F=np.zeros(VisWeights.shape,np.int32)
        #print "u=",u
        #print "v=",v
        w=_pyGridder.pyGridderPoints(grid,
                                     F,
                                     u,
                                     v,
                                     VisWeights,
                                     float(Robust),
                                     Mode,
                                     np.float64(freqs.flatten()),
                                     np.array([cell,cell],np.float64))
        print>>log,"weights computed"


        # C=299792458.
        # uf=u.reshape((u.size,1))*freqs.reshape((1,freqs.size))/C
        # vf=v.reshape((v.size,1))*freqs.reshape((1,freqs.size))/C

        # x,y=np.int32(np.round(uf/cell))+xc,np.int32(np.round(vf/cell))+yc
        # x,y=(uf/cell)+xc,(vf/cell)+yc
        # condx=((x>0)&(x<npix))
        # condy=((y>0)&(y<npix))
        # ind=np.where((condx & condy))[0]
        # X=x#[ind]
        # Y=y#[ind]
        
        # w[w==0]=1e-10
        
        # import pylab
        # pylab.clf()
        # #pylab.scatter(uf.flatten(),vf.flatten(),c=w.flatten(),lw=0,alpha=0.3,vmin=0,vmax=1)#,w[ind,0])
        # grid[grid==0]=1e-10
        # pylab.imshow(np.log10(grid),interpolation="nearest")
        # incr=1
        # pylab.scatter(X.ravel()[::incr],Y.ravel()[::incr],c=np.log10(w.ravel())[::incr],lw=0)#,alpha=0.3)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop
        
        return w
