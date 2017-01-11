#!/usr/bin/python
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

from pyrap.images import image
import glob
import numpy
#import pylab
import numpy as np
import os

def MakeEmpty():
    FileTemp="_temp"
    FileTempPB="_tempPB"
    OutFile="Stacked.image"
    os.system("rm -Rf %s"%OutFile)
    os.system("rm -Rf %s* %s*"%(FileTemp,FileTempPB))

    ll=sorted(glob.glob("*.corr"))
    llPB=sorted(glob.glob("*.avgpb"))

    # ############################"
    # # Take off beam 3
    # ll=ll[0:2]
    # llPB=llPB[0:2]
    # ############################"

    print "Files In: ", ll
    print "FilesPB In: ", llPB

    ############################"
    # That was meant to compute the average of RA and DEC
    # incrList=[]
    # ramean=[]
    # decmean=[]
    # for i in range(len(ll)):
    #     img=image(ll[i])
    #     c = img.coordinates()
    #     rac,decc=c.dict()['direction0']["crval"]
    #     incrx,incry=c.dict()['direction0']["cdelt"]
    #     print incrx,incry,rac,decc
    #     ramean.append(rac)
    #     decmean.append(decc)
    # Npix=abs(2.*int(((width*np.pi/180)/incrx)/2))
    # ramean=np.mean(ramean)
    # decmean=np.mean(decmean)
    # print Npix,Npix,ramean,decmean
    ############################"

    img=image(ll[0])
    cMain = img.coordinates()
    Npix=img.getdata().shape[2]
    ############################"
    # That's if you want to change the astrometry of the OutPut image
    #cMain.set_referencepixel([0.0, np.array([ 0.]), np.array([ Npix/2,Npix/2])])
    #cMain.set_referencevalue([0.0, np.array([ 0.]), np.array([ ramean,decmean])])
    #incr=cMain.get_increment()
    #incr[2]=[incrx,incry]
    #cMain.set_increment(incr)
    #img=image(ll[0])
    ############################"

    imOut=img.regrid( [2,3], cMain, outshape=(1,1,int(Npix),int(Npix)))
    Stack=imOut.getdata().copy()
    #return Stack
    Stack.fill(0)
    PBStack=np.zeros_like(Stack)

    

    for i in range(len(ll)):
        print "Read %s"%ll[i]
        FileTemp="_temp_%3.3i"%i
        FileTempPB="_tempPB_%3.3i"%i
        #os.system("rm -Rf %s"%FileTemp)
        #os.system("rm -Rf %s"%FileTempPB)

        img=image(ll[i])
        img.saveas(FileTemp)
        img=image(FileTemp)
        DataImg=img.getdata()
        dx=DataImg.shape[2]/2

        ImPB=image(llPB[i])
        ImPB.saveas(FileTempPB)
        ImPB=image(FileTempPB)
        PB=ImPB.getdata()
        #ImPB=ImPB.regrid( [2,3], img.coordinates(), outshape=DataImg.shape)

        PB[PB<0.7]=0.
        Center2=PB.shape[2]/2
        PB=PB[:,:,Center2-dx:Center2+dx,Center2-dx:Center2+dx]
        # pylab.figure(0)
        # pylab.clf()
        # pylab.imshow(PB[0,0,:,:],vmin=0,vmax=1)
        # pylab.draw()
        # pylab.show()
        PBsq=PB#*PB
        PBsq[np.isnan(PBsq)]=0.
        PBsq[np.isinf(PBsq)]=0.
        DataImg[np.isnan(DataImg)]
        DataImg[np.isinf(DataImg)]
        data=DataImg*PBsq

        img.putdata(data)
        ImStack=img.regrid( [2,3], cMain, outshape=(1,1,int(Npix),int(Npix)))
        DataAtStack=ImStack.getdata()
        Stack+=DataAtStack

        img.putdata(PBsq)
        ImPBStack=img.regrid( [2,3], cMain, outshape=(1,1,int(Npix),int(Npix)))
        PBAtStack=ImPBStack.getdata()
        PBStack+=PBAtStack

        # pylab.figure(1)
        # pylab.clf()
        # pylab.imshow(PBAtStack[0,0,:,:],vmin=-0.01,vmax=1)
        # pylab.draw()
        # pylab.show()

        # pylab.figure(2)
        # pylab.clf()
        # pylab.imshow(DataAtStack[0,0,:,:],vmin=-0.01,vmax=.1)
        # pylab.draw()
        # pylab.show()

        #del(img)
        #del(ImPB)
        #os.system("rm -Rf %s"%FileTemp)
        #os.system("rm -Rf %s"%FileTempPB)

    PBStack[PBStack<0.1]=1.
    imOut.putdata(Stack/PBStack)
    print "Saving mosaic in file: %s"%OutFile
    imOut.saveas(OutFile)
    #os.system("rm -Rf %s* %s*"%(FileTemp,FileTempPB))
