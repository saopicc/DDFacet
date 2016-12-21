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

import numpy as np
import ClassMS
import pylab
import ClassImager
#import ClassImagerSphe as ClassImager
import MyImshow
import ClassGridMachine
import ClassSM


def testFacet():

    wmax=10000
    MS=ClassMS.ClassMS("MSTest.MS",Col="CORRECTED_DATA")
    SM=ClassSM.ClassSM("ModelRandom00.txt")
    l,m=MS.radec2lm_scalar(SM.SourceCat.ra,SM.SourceCat.dec)
    lmShift=(l[0],m[0])

    vis=MS.data
    Imager=ClassImager.ClassFacetImager(Support=11,NFacets=2,OverS=5,wmax=wmax,Nw=101,Npix=1024,Cell=40.,ChanFreq=MS.ChanFreq.flatten(),
                                        Padding=1.2)
    return
    # Imager.put(MS.uvw,vis,MS.flag_all)
    # Dirty=Imager.getDirtyIm()

    # # ImagerPSF=ClassGridMachine.ClassGridMachine(Support=11,OverS=5,wmax=wmax,Nw=101,Npix=1024,Cell=40.,ChanFreq=MS.ChanFreq.flatten(),RaDecRad=MS.radec,DoPSF=True,Padding=1.2)
    # # ImagerPSF.put(MS.uvw,vis,MS.flag_all)
    # # PSF=ImagerPSF.getDirtyIm()

    # pylab.figure(1)
    # pylab.clf()
    # ax=pylab.subplot(1,1,1)
    # MyImshow.imshow(ax,Dirty[0,0].real.T,interpolation="nearest",vmin=0,vmax=100)
    # pylab.draw()
    # pylab.show(False)
    

    # stop

def testGridWShiftGet():

    wmax=20000
    MS=ClassMS.ClassMS("MSTest.MS",Col="CORRECTED_DATA")
    SM=ClassSM.ClassSM("ModelRandom00.txt")
    l,m=MS.radec2lm_scalar(SM.SourceCat.ra,SM.SourceCat.dec)
    #l+=1.*np.pi/180
    #lmShift=(l[0],m[0])
    lmShift=(l[0]*5,m[0]*5)
    #lmShift=(0,0)
    
    RaDecRad=MS.radec
    xc,yc=SM.SourceCat.ra[0],SM.SourceCat.dec[0]#RaDecRad[0]+lmShift[0],RaDecRad[1]+lmShift[1]
    TransfRaDec=None#[RaDecRad,(xc,yc)]

    vis=MS.data

    # ind=np.where(np.abs(MS.uvw[:,2])>100)[0]
    # MS.uvw=MS.uvw[ind]
    # MS.data=MS.data[ind]
    # MS.flag_all=MS.flag_all[ind]
    
    Imager=ClassGridMachine.ClassGridMachine(Support=5,OverS=5,wmax=wmax,Nw=101,Npix=1024,Cell=40.,ChanFreq=MS.ChanFreq.flatten(),
                                             #TransfRaDec=None,
                                             #RaDecRad=MS.radec,
                                             WProj=True,
                                             #WProj=False,
                                             DoPSF=False,Padding=1.4,
                                             lmShift=lmShift,
                                             TransfRaDec=TransfRaDec)


    ModelIm=np.zeros(Imager.NonPaddedShape,dtype=np.float32)
    #ModelIm[0,0,300,400]=100

    ns=3
    _,_,n,n= Imager.NonPaddedShape
    indx=np.int64(np.random.rand(ns)*n)
    indy=np.int64(np.random.rand(ns)*n)

    #ModelIm[0,0,indx,indy]=100
    #ModelIm[0,3,indx,indy]=100
    
    # cx,cy=128,64
    # ModelIm[0,0,cx,cy]=100
    # cx,cy=128,128
    # ModelIm[0,0,cx,cy]=100
    # cx,cy=128+64,128+64
    # ModelIm[0,0,cx,cy]=100

    cx,cy=300,400
    ModelIm[0,0,cx,cy]=100

    Imager.setModelIm(ModelIm)
    uvw=MS.uvw.copy()

    vis=Imager.get(uvw,MS.data,MS.flag_all)

    uvw=MS.uvw.copy()
    Imager.put(uvw,vis,MS.flag_all)

    Dirty=Imager.getDirtyIm()

    pylab.figure(1)
    pylab.clf()
    ax=pylab.subplot(1,2,1)
    MyImshow.imshow(ax,Dirty[0,0].real.T,interpolation="nearest",vmin=0,vmax=100)
    ax=pylab.subplot(1,2,2,sharex=ax,sharey=ax)
    MyImshow.imshow(ax,ModelIm[0,0].real.T,interpolation="nearest",vmin=0,vmax=100)
    #pylab.xlim(260-50,320+50)
    #pylab.ylim(360-50,460+50)
    pylab.draw()
    pylab.show(False)
    

    stop

def testGridWShift():

    wmax=20000
    MS=ClassMS.ClassMS("MSTest.MS",Col="CORRECTED_DATA")
    SM=ClassSM.ClassSM("ModelRandom00.txt")
    l,m=MS.radec2lm_scalar(SM.SourceCat.ra,SM.SourceCat.dec)
    #l+=1.*np.pi/180
    lmShift=(l[0],m[0])
    #lmShift=(l[0],0)
    RaDecRad=MS.radec
    xc,yc=SM.SourceCat.ra[0],SM.SourceCat.dec[0]#RaDecRad[0]+lmShift[0],RaDecRad[1]+lmShift[1]
    TransfRaDec=None#[RaDecRad,(xc,yc)]

    vis=MS.data
    
    Imager=ClassGridMachine.ClassGridMachine(Support=5,OverS=5,wmax=wmax,Nw=101,Npix=1024,Cell=40.,ChanFreq=MS.ChanFreq.flatten(),
                                             #TransfRaDec=None,
                                             #RaDecRad=MS.radec,
                                             WProj=True,
                                             #WProj=False,
                                             DoPSF=False,Padding=1.2,
                                             lmShift=lmShift,
                                             TransfRaDec=TransfRaDec)
    Imager.put(MS.uvw,vis,MS.flag_all)

    # _,_,n,n=Imager.Grid.shape
    # uvs=(n/2)*1./(np.pi/180*((Imager.Cell)/3600))
    # u,v=np.mgrid[-uvs:uvs:n*1j,-uvs:uvs:n*1j]
    # corr=np.exp(-2.*np.pi*1j*(u*lmShift[0]+v*lmShift[1]))
    # Imager.Grid[0,0]*=corr

    Dirty=Imager.getDirtyIm()



    # ImagerPSF=ClassGridMachine.ClassGridMachine(Support=11,OverS=5,wmax=wmax,Nw=101,Npix=1024,Cell=40.,ChanFreq=MS.ChanFreq.flatten(),RaDecRad=MS.radec,DoPSF=True,Padding=1.2)
    # ImagerPSF.put(MS.uvw,vis,MS.flag_all)
    # PSF=ImagerPSF.getDirtyIm()

    pylab.figure(1)
    pylab.clf()
    ax=pylab.subplot(1,1,1)
    MyImshow.imshow(ax,Dirty[0,0].real.T,interpolation="nearest",vmin=0,vmax=100)
    #MyImshow.imshow(ax,np.angle(corr).T,interpolation="nearest")#,vmin=0,vmax=100)
    pylab.draw()
    pylab.show(False)
    

    stop

def testGridWPSF():

    wmax=10000
    MS=ClassMS.ClassMS("MSTest.MS",Col="CORRECTED_DATA")


    Imager=ClassGridMachine.ClassGridMachine(Support=11,OverS=5,wmax=wmax,Nw=101,Npix=1024,Cell=40.,ChanFreq=MS.ChanFreq.flatten(),RaDecRad=MS.radec,DoPSF=False,Padding=1.2,WProj=True)
    Imager.put(MS.uvw,MS.data,MS.flag_all)
    Dirty=Imager.getDirtyIm()

    #ImagerPSF=ClassGridMachine.ClassGridMachine(Support=11,OverS=5,wmax=wmax,Nw=101,Npix=1024,Cell=40.,ChanFreq=MS.ChanFreq.flatten(),RaDecRad=MS.radec,DoPSF=True,Padding=1.2)
    #ImagerPSF.put(MS.uvw,vis,MS.flag_all)
    #PSF=ImagerPSF.getDirtyIm()

    pylab.figure(1)
    pylab.clf()
    ax=pylab.subplot(1,2,1)
    MyImshow.imshow(ax,Dirty[0,0].real.T,interpolation="nearest",vmin=0,vmax=100)
    #ax=pylab.subplot(1,2,2)
    #MyImshow.imshow(ax,PSF[0,0].real.T,interpolation="nearest",vmin=0,vmax=1)
    pylab.draw()
    pylab.show(False)
    

    stop

def testGridW():

    wmax=5000
    MS=ClassMS.ClassMS("MSTest.MS",Col="CORRECTED_DATA")

    Imager=ClassImager.ClassGridMachine(Support=11,OverS=5,wmax=wmax,Nw=101,Npix=1024,Cell=40.,ChanFreq=MS.ChanFreq.flatten(),RaDecRad=MS.radec,DoPSF=False)

    #return
    vis=MS.data
    Imager.put(MS.uvw,vis,MS.flag_all)
    Imager.GridToIm()

    dx=200
    x0=Imager.Dirty.shape[0]/2

    im=Imager.Dirty[x0-dx:x0+dx,x0-dx:x0+dx]

    pylab.figure(1)
    pylab.clf()
    pylab.imshow(im[0,0].real.T,interpolation="nearest",vmin=0,vmax=100)
    pylab.draw()
    pylab.show(False)
    

    stop

def testDeGrid():

    wmax=10000
    MS=ClassMS.ClassMS("MSTest.MS",Col="DATA")

    Imager=ClassImager.ClassImager(Support=7,OverS=21,wmax=wmax,Nw=101,Npix=1024,Cell=40.,ChanFreq=MS.ChanFreq.flatten(),RaDecRad=MS.radec,DoPSF=False)

    MS.data.fill(1.)
    
    #u,v,w=MS.uvw.T

    #Imager.put(MS.uvw,MS.data,MS.flag_all)

    ModelIm=np.zeros_like(Imager.Grid)
    _,_,n,n=ModelIm.shape
    ModelIm[0,0,n/2+100,n/2+100]=1
    ns=100
    indx=np.int64(np.random.rand(ns)*n)
    indy=np.int64(np.random.rand(ns)*n)
    ModelIm[0,0,indx,indy]=1
    Imager.setModelIm(ModelIm)
    # pylab.figure(1)
    # pylab.clf()
    # pylab.imshow(Grid[0,0].real,interpolation="nearest")#,vmin=-1,vmax=2)
    # pylab.colorbar()
    # pylab.draw()
    # pylab.show()
    # stop

    # nbl=MS.nbl
    # uvw=MS.uvw[1::nbl,:].copy()
    # data=MS.data[1::nbl,:,:].copy()
    # flag_all=MS.flag_all[1::nbl,:,:].copy()

    MS.data.fill(0)
    
    vis=Imager.get(MS.uvw,MS.data,MS.flag_all)

    Imager.put(MS.uvw,vis,MS.flag_all)
    Imager.GridToIm()

    dx=200
    x0=Imager.Dirty.shape[0]/2

    im=Imager.Dirty[x0-dx:x0+dx,x0-dx:x0+dx]

    pylab.figure(1)
    pylab.clf()
    pylab.subplot(1,2,1)
    pylab.imshow(im[0,0].real.T,interpolation="nearest")#,vmin=0,vmax=1)
    pylab.plot(indx,indy,ls="",marker="+")#,vmin=0,vmax=1)
    pylab.subplot(1,2,2)
    pylab.imshow(ModelIm[0,0].real.T,interpolation="nearest")#,vmin=0,vmax=1)
    pylab.plot(indx,indy,ls="",marker="+")#,vmin=0,vmax=1)
    #pylab.colorbar()
    pylab.draw()
    pylab.show(False)


    # pylab.clf()
    # pylab.plot(vis[:,0,:].real)
    # pylab.plot(vis[:,0,:].imag)
    # pylab.draw()
    # pylab.show(False)
    

    stop



def testGrid():

    wmax=10000
    MS=ClassMS.ClassMS("MSTest.MS",Col="DATA")

    Imager=ClassImager.ClassImager(Support=9,wmax=wmax,Nw=101,Npix=512,Cell=8.,ChanFreq=MS.ChanFreq.flatten(),RaDecRad=MS.radec,DoPSF=False)

    MS.data.fill(1.)
    
    u,v,w=MS.uvw.T

    Imager.put(MS.uvw,MS.data,MS.flag_all)
    Imager.GridToIm()

    dx=200
    x0=Imager.Dirty.shape[0]/2

    im=Imager.Dirty[x0-dx:x0+dx,x0-dx:x0+dx]

    print "(min,max) = (%f, %f)"%(im.min(),np.max(im))

    
    pylab.figure(1)
    pylab.clf()
    pylab.imshow(im[0,0].real,interpolation="nearest")#,vmin=0,vmax=1)
    pylab.colorbar()
    pylab.draw()
    pylab.show()

    stop


if __name__=="__main__": testDeGrid()
