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

import time

import numpy as np
import psutil
import pylab

from collections import deque
import sys

#pylab.ion()

def GivePolygon(x,y):
    if isinstance(x,deque):
        x=list(x)
    if isinstance(y,deque):
        y=list(y)
    X=[0]+x+[np.max(x)]
    if type(y)==np.ndarray:
        y=y.tolist()

    Y=[0]+y+[0]
    return X,Y




class ClassMemMonitor():
    def __init__(self,dt=0.5,NMax=None):
        self.dt=dt
        self.NMax=NMax
        
        self.LMem=[]#deque(maxlen=NMax)
        self.LSMem=[]#deque(maxlen=NMax)
        self.LSMemAvail=[]#deque(maxlen=NMax)
        self.LMemAvail=[]#deque(maxlen=NMax)
        self.LMemTotal=[]#deque(maxlen=NMax)
        self.LShared=[]#deque(maxlen=NMax)
        self.LCPU=[]#deque(maxlen=NMax)
        self.LT=[]#deque(maxlen=NMax)
        self.t0=time.time()
        self.Swap0=None
        
    def update(self):
        vmem=psutil.virtual_memory()
        
        mem=vmem.used/float(2**20)/1024
        self.LMem.append(mem)
        
        memAvail=vmem.available/float(2**20)/1024
        self.LMemAvail.append(memAvail)
        
        memTotal=vmem.total/float(2**20)/1024
        self.LMemTotal.append(memTotal)

        smem=psutil.swap_memory()
        Smem=smem.used/float(2**20)/1024
        if self.Swap0 is None:
            self.Swap0=Smem
        self.LSMem.append(Smem-self.Swap0)

        SmemAvail=smem.total/float(2**20)/1024
        self.LSMemAvail.append(SmemAvail)

        TotSeen=np.array(self.LMemAvail)+np.array(self.LMem)
        Cache=TotSeen-np.array(self.LMemTotal)
        
        self.PureRAM=np.array(self.LMem)-Cache
        
        cpu=psutil.cpu_percent()
        self.LCPU.append(cpu)
        self.LT.append((time.time()-self.t0)/60)

    def plot(self):
        pylab.clf()
        ax = pylab.subplot(2,1,1)
        self.plotOne(N=self.NMax,ax=ax)
        ax = pylab.subplot(2,1,2)
        self.plotOne(N=None,ax=ax,NMax=1000)
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)
        
    def plotOne(self,N=None,ax=None,NMax=None):

        if N is not None:
            N=np.min([N,len(self.LMem)])
        else:
            N=len(self.LMem)

        incr=1
        if NMax is not None:
            incr=int(np.max([1,N//NMax]))
            
        LMem=self.LMem[-N:][::incr]
        LSMem=self.LSMem[-N:][::incr]
        LSMemAvail=self.LSMemAvail[-N:][::incr]
        LMemAvail=self.LMemAvail[-N:][::incr]
        LMemTotal=self.LMemTotal[-N:][::incr]
        LShared=self.LShared[-N:][::incr]
        LCPU=self.LCPU[-N:][::incr]
        PureRAM=self.PureRAM[-N:][::incr]
        LT=self.LT[-N:][::incr]

        t0=self.t0
        
        ax2 = ax.twinx()

        if len(LMem)<2: return

        ax.cla()
        
        # Total Available
        x,y=GivePolygon(LT,LMemTotal)
        ax.fill(x,y,'black', alpha=0.1, edgecolor='black')

        # Cache
        # ax.plot(LT,LMemTotal-np.array(LMem),lw=2,color="green")
        x,y=GivePolygon(LT,np.array(LMem))
        ax.fill(x,y,'black', alpha=0.1, edgecolor='blue',lw=2)
        
        # Total used excluding cache
        #x,y=GivePolygon(LT,np.array(LShared)+np.array(PureRAM))
        x,y=GivePolygon(LT,np.array(PureRAM))
        ax.fill(x,y,'black', alpha=0.3, edgecolor='blue',lw=2)
        
        # memory
        # ax.plot(LT,PureRAM,lw=2,color="blue")
        x,y=GivePolygon(LT,PureRAM)
        ax.fill(x,y,'green', alpha=0.3, edgecolor='green',lw=2,hatch="//")

        # swap
        x,y=GivePolygon(LT,LSMem)
        #ax.fill(x,y,'', alpha=1, edgecolor='red',lw=2,hatch="/")
        ax.fill(x,y,'gray', alpha=0.3, edgecolor='red',lw=1,hatch="*")
        # ax.plot(LT,np.array(LSMem),lw=2,ls=":",color="blue")
        # ax.plot(LT,np.array(LSMemAvail),lw=2,ls=":",color="red")
            
        # CPU
        ax2.plot(LT,LCPU, color="black",ls="--")
        
        #ax.legend(loc=0)
        ax.grid()
        
        
        ax.set_ylabel("Mem [GB]")
        ax2.set_ylabel("CPU [%]")
        ax.set_xlabel("Time [min.]")
        #ax2.set_ylabel(r"Temperature ($^\circ$C)")
        #ax2.set_ylim(0, 35)
        ax.set_xlim(np.min(LT),np.max(LT))
        ax.set_ylim(0,1.1*np.max(LMemTotal))
        ax2.set_ylim(0,100)
        
        #ax2.legend(loc=0)
        
        
    def start(self):
        while True:
            t0=time.time()
            while True:
                self.update()
                if time.time()-t0>5: break
                time.sleep(0.05)
            self.plot()


def test():
    MM=ClassMemMonitor()
    MM.start()
    

if __name__=="__main__":
    NMax=1000
    if len(sys.argv)>1:
        NMax=int(sys.argv[1])
        
    MM=ClassMemMonitor(NMax=NMax)
    MM.start()

