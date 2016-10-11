import os

import ClassTimeIt
import MyLogger
import ToolsDir.GiveMDC
import numpy as np

log=MyLogger.getLogger("ClassDDEGridMachine")

import ToolsDir
import pylab
import ClassData
from ClassDDEGridMachine import ClassDDEGridMachine

def GiveGM():
    GD=ClassData.ClassGlobalData("ParsetDDFacet.txt")
    MDC,GD=ToolsDir.GiveMDC.GiveMDC(GD=GD)
    MS=MDC.giveMS(0)
    MS.ReadData()

    GM=ClassDDEGridMachine(GD,MDC,DoDDE=False,WProj=True,lmShift=(0.,0.),#JonesDir=3,
                           Npix=GD.DicoConfig["Facet"]["MainFacetOptions"]["Npix"],
                           Cell=GD.DicoConfig["Facet"]["MainFacetOptions"]["Cell"],
                           Support=GD.DicoConfig["Facet"]["MainFacetOptions"]["Support"],
                           Nw=GD.DicoConfig["Facet"]["MainFacetOptions"]["Nw"],
                           OverS=GD.DicoConfig["Facet"]["MainFacetOptions"]["OverS"],
                           Padding=1.,ChanFreq=MS.ChanFreq.flatten())
    return GM


def testGrid(GM):
    MDC=GM.MDC
    MS=GM.MDC.giveMS(0)

    row0,row1=0,None

    ind=np.where((MS.A0==0)&(MS.A1==29))[0]
    uvw=np.float64(MS.uvw)[ind]#[row0:row1]
    times=np.float64(MS.times_all)[ind]#[row0:row1]
    data=np.complex64(MS.data)[ind]#[row0:row1]
    A0=np.int32(MS.A0)[ind]#[row0:row1]
    A1=np.int32(MS.A1)[ind]#[row0:row1]
    flag=np.bool8(MS.flag_all)[ind]#[row0:row1,:,:].copy()
    flag.fill(0)

    uvw=np.float64(MS.uvw)
    times=np.float64(MS.times_all)
    data=np.complex64(MS.data)
    A0=np.int32(MS.A0)
    A1=np.int32(MS.A1)
    flag=np.bool8(MS.flag_all)

    



    T=ClassTimeIt.ClassTimeIt("main")
    #Grid=GM.put(times,uvw,data,flag,(A0,A1),W=None,PointingID=0,DoNormWeights=True)
    #T.timeit("grid")



    Grid=np.zeros((1, 1, 1023, 1023),np.complex64)
    nch,npol,NpixFacet,NpixFacet=Grid.shape
    Grid[0,0,200,650]=100.
    ModelIm=np.zeros((nch,npol,NpixFacet,NpixFacet),dtype=np.float32)
    for ch in range(nch):
        for pol in range(npol):
            ModelIm[ch,pol]=Grid[ch,pol,:,:].T[::-1,:].real

    visPredict=np.zeros_like(data)
    _=GM.get(times,uvw,visPredict,flag,(A0,A1),ModelIm)
    T.timeit("degrid")

    ind=np.where((A0!=A1)&(np.abs(visPredict[:,0,0])>0.))[0]
    d0=data[ind,0,0]
    d1=visPredict[ind,0,0]
    diff=d0-d1
    u,v,w=uvw[ind].T
    incr=1
    pylab.clf()
    pylab.subplot(1,2,1)
    pylab.scatter(u[::incr],v[::incr],c=np.abs(diff[::incr]),lw=0,alpha=0.5)
    pylab.subplot(1,2,2)
    pylab.scatter(u[::incr],v[::incr],c=np.angle(diff[::incr]),lw=0,alpha=0.5)
    pylab.scatter(u[::incr],v[::incr],c=np.angle(d0[::incr])-np.angle(d1[::incr]),lw=0,alpha=0.5)
    pylab.colorbar()
    pylab.draw()
    pylab.show(False)


    fig=pylab.figure(1)
    os.system("rm -rf png/*.png")
    op0=np.real
    op1=np.angle
    for iAnt in range(36):
        for jAnt in range(36):
            
            ind=np.where((A0==iAnt)&(A1==jAnt))[0]
            if ind.size==0: continue
            d0=data[ind,0,0]
            if np.max(d0)<1e-6: continue
            
            d1=visPredict[ind,0,0]
            pylab.clf()
            pylab.subplot(2,1,1)
            pylab.plot(op0(d0))
            pylab.plot(op0(d1))
            pylab.plot(op0(d0-d1))
            pylab.plot(np.zeros(d0.size),ls=":",color="black")
            pylab.title("%s"%iAnt)
            pylab.subplot(2,1,2)
            pylab.plot(op1(d0))
            pylab.plot(op1(d1))
            pylab.plot(op1(d0-d1))
            pylab.plot(np.zeros(d0.size),ls=":",color="black")
            pylab.title("%s"%iAnt)
            pylab.draw()
            fig.savefig("png/resid_%2.2i_%2.2i.png"%(iAnt,jAnt))
            #pylab.show(False)
