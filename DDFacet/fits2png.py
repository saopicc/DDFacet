#!/usr/bin/env python
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
#import matplotlib
#matplotlib.use('agg')

import time
import os
import numpy as np
from astropy.io import fits
import pylab
#import regrid

import matplotlib.pyplot as mpl

import aplpy

import findrms
import rad2hmsdms
from pyrap.images import image


def GiveNXNYPanels(Ns,ratio=800/500):
    nx=int(round(np.sqrt(Ns/ratio)))
    ny=int(nx*ratio)
    if nx*ny<Ns: ny+=1
    return nx,ny

def test():

    pylab.figure(1)
    CM=ConvertMachine("Test.KAFCA.NewSM.Nosmear.AP2.NoCorr.restored.fits",regFile="ds9.reg")
    CM.convertStamps()
    pylab.figure(2)
    CM=ConvertMachine("Test.KAFCA.NewSM.Nosmear.AP2.Cycle2.restored.fits",regFile="ds9.reg")
    CM.convertStamps()

class ConvertMachine():
    def __init__(self,FitsName,NameOut=None,regFile=None):
        self.FitsName=FitsName
        if NameOut is None:
            NameOut=self.FitsName+".png"
        self.NameOut=NameOut
        if regFile is not None:
            self.ReadReg(regFile)
        
    def ReadReg(self,infile):
        f  = open(infile, "rb")
        self.DicoStamps={}
        iStamp=0
        while True:
            F=f.readline()
            if F=="": break
            if not("circle" in F): continue
            sra,sdec,srad=F.split("(")[1].split(")")[0].split(",")
            rah,ram,ras=sra.split(":")
            decd,decm,decs=sdec.split(":")
            ra=15.*(float(rah)+float(ram)/60.+float(ras)/3600.)
            dec=(float(decd)+float(decm)/60.+float(decs)/3600.)
            self.DicoStamps[iStamp]={"radeg":ra,"decdeg":dec}
            iStamp+=1

    def convertStamps(self,BoxPix=200,SubPlots=(2,2),rms=0.0007):
        im=image(self.FitsName)
        data=im.getdata()[0,0]

        D=data
        Np=1000
        nx,ny=D.shape
        indx=np.int64(np.random.rand(Np)*nx)
        indy=np.int64(np.random.rand(Np)*ny)
        #rms=np.std(D[indx,indy])
        

        NX,NY=GiveNXNYPanels(len(self.DicoStamps.keys()),ratio=800/500)
        pol,freq,rac,decc=im.toworld((0,0,0,0))
        pylab.clf()
        for iStamp in self.DicoStamps.keys():
            ax=pylab.subplot(NX,NY,iStamp+1)
            
            outname=self.FitsName+".%2.2i"%iStamp
            rac=self.DicoStamps[iStamp]["radeg"]*np.pi/180
            decc=self.DicoStamps[iStamp]["decdeg"]*np.pi/180
            _,_,xc,yc= im.topixel((pol,freq,decc,rac))
            D=data[int(xc)-BoxPix:int(xc)+BoxPix,int(yc)-BoxPix:int(yc)+BoxPix]

            vmax=30.*rms#np.max([D.max(),10.*rms])
            pylab.imshow(D,vmin=-5.*rms,vmax=vmax,cmap="gray")
            ax.set_xticklabels([])
            ax.get_xaxis().set_visible(False)
            ax.set_yticklabels([])
            ax.get_yaxis().set_visible(False)
            
        pylab.tight_layout()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
            


    def convert(self,name,nameout,radecbox=None):
        
        
        a=fits.open(name)[0]
        #rac,decc,npix,dpix=a.header["CRVAL1"],a.header["CRVAL2"],a.header["NAXIS1"],a.header["CDELT1"]
        
        
        H=a.header
        freq=H["CRVAL4"]
        
        # raobs,decobs,freq=a.header["OBSRA"],a.header['OBSDEC'],a.header['RESTFRQ']
        # raobs=H["CRVAL1"]
        # raobs=H["CRVAL2"]
        # strra=rad2hmsdms.rad2hmsdms(raobs,Type="ra",deg=True).replace(" ",":")
        # strdec=rad2hmsdms.rad2hmsdms(decobs,deg=True).replace(" ",".")
        
        try:
            bmaj,bmin=a.header["BMAJ"],a.header['BMIN']
            bmaj*=3600.
            bmin*=3600.
        except:
            bmaj,bmin=-1.,-1.
            
        date=a.header['DATE-OBS']
        
        
        # radius=abs(float(npix)*dpix)/2.
        # radius/=padding
        
        #rms=findrms.findrms(a.data)
        #print "rms=%f"%rms
        fig = mpl.figure(0,figsize=(17,17), dpi=200)
        #cmap='gist_gray'
        cmap='Greys'
        cmap='gray'
            
            
        D=a.data
        Np=1000
        nx,ny,_,_=a.data.shape
        indx=np.int64(np.random.rand(Np)*nx)
        indy=np.int64(np.random.rand(Np)*ny)
        rms=np.std(D[indx,indy,0,0])
        fig.clf()
        
        dx=0.#4
        dy=0.#85
        c=0.0#7
        dd=0.95-c
        subplot=[c,c,dd,dd]
        lmc = aplpy.FITSFigure(name, 
                               slices=[0,0], 
                               #subplot=subplot,
                               dimensions=[0,1])#, 
                               #figure=fig , 
                               #auto_refresh=False)#,downsample=1)
        lmc.show_colorscale(cmap='gray',stretch='linear',vmin=-3.*rms,vmax=np.max(D))#cmin,vmax=cmax)

        lmc.show_grayscale()
        rac,decc,radius=radecbox
        print rac,decc,radius
        lmc.recenter(rac,decc,width=0.5,height=0.3)
        
        # if radecbox is not None:
        #     rac,decc,radius=radecbox
        #     lmc.recenter(rac,decc,radius)
            
        # lmc.show_colorscale(cmap=cmap,stretch='linear',vmin=-3.*rms,vmax=np.max(D))#,vmin=-5*rms,vmax=50.*rms)#cmin,vmax=cmax)
        # lmc.add_grid()
        # lmc.grid.set_alpha(1)
        # #lmc.grid.set_color('black')
        # lmc.set_tick_labels_format(xformat='hh:mm',yformat='dd:mm')
        # lmc.set_tick_color("black")

        mpl.draw() 
        mpl.show()
        
        # if "%s" in nameout:
        #     nameout="%s.png"%name
        nameout=name+".pdf"
        #fig.savefig(nameout)
        lmc.save(nameout)#,dpi=100)
        #dico={"beam":(bmaj,bmin),"ra":strra,"dec":strdec,"freq":freq/1.e6,"date":date,"rms":rms}
        #return nameout,dico
    
if __name__=="__main__":
    import sys
    print sys.argv

    fin=sys.argv[1]
    if len(sys.argv)==3:
        pad=sys.argv[2]
    else:
        pad=1.4

    convert(fin,pad)
