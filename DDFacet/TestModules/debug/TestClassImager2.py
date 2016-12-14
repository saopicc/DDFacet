'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA-SA, Rhodes University

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

import ClassFacetMachine
import numpy as np
import pylab
import ToolsDir
import MyPickle




def testFacet():

    wmax=20000
    ParsetFile="ParsetDDFacet.txt"
    MDC,GD=ToolsDir.GiveMDC.GiveMDC(ParsetFile)

    #FileSimulSols=GD.DicoConfig["Files"]["Simul"]["FileSols"]
    #SimulSols=MyPickle.Load(FileSimulSols)["Sols"]

    Imager=ClassFacetMachine.ClassFacetMachine(MDC,GD,Precision="S",Parallel=False)#,Sols=SimulSols)
    Imager.appendMainField(Support=9,NFacets=1,OverS=5,wmax=wmax,Nw=101,Npix=512,Cell=10.,Padding=1.4)
    Imager.Init()
        

    ########

    MS=Imager.MDC.giveMS(0)
    MS.ReadData()

    vis=MS.data


    #####
    ModelImage=Imager.GiveEmptyMainField()
    _,_,n,n= ModelImage.shape
    ns=10
    indx=np.int64(np.random.rand(ns)*n)
    indy=np.int64(np.random.rand(ns)*n)
    #ModelImage[0,:,indx,indy]=100
    #ModelImage[0,:,n/2,n/2]=100
    ModelImage[0,:,50,50]=100
    #Imager.Image[0,:,50,100]=100
    #Imager.setModelIm(Imager.Image)
    Imager.ModelOrig=ModelImage.copy()
    #vis=np.complex128(np.arange(vis.size).reshape(vis.shape))
    vis=np.complex128(np.zeros_like(vis))
    print vis
    print
    row0,row1=1,10
    #vis=Imager.GiveVis(MS.times_all[row0:row1],MS.uvw[row0:row1],vis[row0:row1],MS.flag_all[row0:row1],(MS.A0[row0:row1],MS.A1[row0:row1]),ModelImage)
    vis=Imager.GiveVis(MS.times_all,MS.uvw,vis,MS.flag_all,(MS.A0,MS.A1),ModelImage)
    #vis=Imager.GiveVisParallel(MS.times_all,MS.uvw,vis,MS.flag_all,(MS.A0,MS.A1),ModelImage)
    #####
    print vis
    print

    W=np.ones(MS.A0.shape,dtype=np.float32)
    Imager.putChunk(MS.times_all,MS.uvw,vis,MS.flag_all,(MS.A0,MS.A1),W,doStack=False)

    #Imager.CalcDirtyImagesParallel(MS.times_all,MS.uvw,vis,MS.flag_all,(MS.A0,MS.A1))

    Imager.Image=Imager.FacetsToIm()
    Imager.ToCasaImage()
    

    # import ClassSM
    # SM=Imager.MDC.giveSM(0)
    # l,m=MS.radec2lm_scalar(SM.SourceCat.ra,SM.SourceCat.dec)
    # Imager.lm=(l,m)

    return Imager



def PlotVis(Imager):
    pylab.clf()
    pylab.plot(Imager.DicoImager[0]["Predict"][100::666,0,0].real)
    pylab.draw()
    pylab.show(False)

def PlotResult(Imager):
    pylab.clf()
    #fig=pylab.figure(frameon=False)
    i=0
    #pylab.hold(True)
    ax=pylab.subplot(1,3,1)#2,2,i+1); i+=1
    extent=(0,1,0,1)
    v0,v1=0,100
    for key in sorted(Imager.DicoImager.keys()):
        #pylab.subplot(2,2,i+1); i+=1
        Dirty=Imager.DicoImager[key]["Dirty"]
        lmShift=Imager.DicoImager[key]["lmShift"]
        l=Imager.DicoImager[key]["lmDiam"]*2
        extent=(lmShift[0]-l/2.,lmShift[0]+l/2.,lmShift[1]-l/2.,lmShift[1]+l/2.)
        #MyImshow.imshow(ax,Dirty[0,0].real,interpolation="nearest",extent=extent,vmin=v0,vmax=v1)
        pylab.imshow(Dirty[0,0].real,interpolation="nearest",extent=extent)#,vmin=v0,vmax=v1)
        print np.max(Dirty[0,0].real)

#    l,m=Imager.lm
#    pylab.scatter(l,m)



    ax.autoscale()

    #v0,v1=-10,10
    ax=pylab.subplot(1,3,2)#2,2,i+1); i+=1
    #MyImshow.imshow(ax,Imager.Image[0,0].T[::-1],vmin=v0,vmax=v1,extent=Imager.ImageExtent,interpolation="nearest")
    #MyImshow.imshow(ax,Imager.Image[0,0].T[::-1,:],vmin=v0,vmax=v1,interpolation="nearest")
    #pylab.imshow(Imager.Image[0,0],vmin=v0,vmax=v1,interpolation="nearest",extent=Imager.ImageExtent)
    pylab.imshow(Imager.Image[0,0].T[::-1,:],interpolation="nearest",extent=Imager.ImageExtent,vmin=v0,vmax=v1)
   # pylab.scatter(l,m)

    ax=pylab.subplot(1,3,3,sharex=ax,sharey=ax)#2,2,i+1); i+=1
    # ax=pylab.subplot(1,3,3)#2,2,i+1); i+=1
    # #MyImshow.imshow(ax,Imager.Image[0,0].T[::-1],vmin=v0,vmax=v1,extent=Imager.ImageExtent,interpolation="nearest")
    pylab.imshow(Imager.ModelOrig[0,0].T[::-1,:],interpolation="nearest",extent=Imager.ImageExtent)#,vmin=v0,vmax=v1)
    
    pylab.draw()
    pylab.pause(0.1)
    pylab.show(False)


def PlotResult2(Imager):
    pylab.clf()
    pylab.imshow(Imager.Image[0,0].T[::-1,:],interpolation="nearest",extent=Imager.ImageExtent)#,vmin=v0,vmax=v1)
    pylab.draw()
    pylab.pause(0.1)
    pylab.show(False)
