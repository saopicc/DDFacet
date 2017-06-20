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
from DDFacet.Array import NpShared


#pylab.ion()

def GivePolygon(x,y):
    X=[0]+x+[np.max(x)]
    if type(y)==np.ndarray:
        y=y.tolist()

    Y=[0]+y+[0]
    return X,Y

def monitorMem():
    LMem=[]
    LSMem=[]
    LSMemAvail=[]
    LMemAvail=[]
    LMemTotal=[]
    LShared=[]
    LCPU=[]
    t0=time.time()
    LT=[]
    Swap0=None


    while True:
        # process = psutil.Process(os.getpid())
        # mem = process.get_memory_info()[0] / float(2 ** 20) 
        vmem=psutil.virtual_memory()
        
        mem=vmem.used/float(2**20)
        LMem.append(mem)
        memAvail=vmem.available/float(2**20)
        LMemAvail.append(memAvail)
        
        memTotal=vmem.total/float(2**20)
        LMemTotal.append(memTotal)

        


        smem=psutil.swap_memory()
        Smem=smem.used/float(2**20)
        if Swap0 is None:
            Swap0=Smem
        LSMem.append(Smem-Swap0)

        SmemAvail=smem.total/float(2**20)
        LSMemAvail.append(SmemAvail)

        

        TotSeen=np.array(LMemAvail)+np.array(LMem)
        Cache=TotSeen-np.array(LMemTotal)
        
        PureRAM=np.array(LMem)-Cache
        
        Shared= NpShared.SizeShm()
        if not Shared: Shared=LShared[-1]
        LShared.append(Shared)


        cpu=psutil.cpu_percent()
        LCPU.append(cpu)
        LT.append((time.time()-t0)/60)

        ax = pylab.subplot(111)
        ax2 = ax.twinx()

        if len(LMem)>2:
            #pylab.clf()
            # pylab.subplot(1,2,1)
            # pylab.plot(LMem)
            # pylab.plot(LMemAvail)
            # pylab.plot(np.array(LMemAvail)+np.array(LMem))
            # pylab.subplot(1,2,2)
            # pylab.plot(LCPU)

            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.01)

            ax.cla()
            
            # Total Available
            #ax.plot(LT,LMemTotal,lw=2,color="black",ls=":")
            x,y=GivePolygon(LT,LMemTotal)
            ax.fill(x,y,'black', alpha=0.1, edgecolor='black')

            # Cache
            # ax.plot(LT,LMemTotal-np.array(LMem),lw=2,color="green")
            x,y=GivePolygon(LT,np.array(LMem))
            ax.fill(x,y,'black', alpha=0.1, edgecolor='blue',lw=2)

            # Total used excluding cache
            x,y=GivePolygon(LT,np.array(LShared)+np.array(PureRAM))
            ax.fill(x,y,'black', alpha=0.3, edgecolor='blue',lw=2)

            # memory
            # ax.plot(LT,PureRAM,lw=2,color="blue")
            x,y=GivePolygon(LT,PureRAM)
            ax.fill(x,y,'green', alpha=0.3, edgecolor='green',lw=2,hatch="//")


            # Shared
            #ax.plot(LT,LShared,lw=2,color="black")
            x,y=GivePolygon(LT,LShared)
            ax.fill(x,y,'red', alpha=0.3, edgecolor='red',lw=2,hatch="\\")
            #ax.plot(LT,TotSeen,lw=2,color="red")

            # # Total Used
            # ax.plot(LT,np.array(LMem),lw=2,color="blue",ls="--")
            

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

            
            ax.set_ylabel("Mem [MB]")
            ax2.set_ylabel("CPU [%]")
            ax.set_xlabel("Time [min.]")
            #ax2.set_ylabel(r"Temperature ($^\circ$C)")
            #ax2.set_ylim(0, 35)
            ax.set_xlim(np.min(LT),np.max(LT))
            ax.set_ylim(0,1.1*np.max(LMemTotal))
            
            #ax2.legend(loc=0)

            pylab.draw()
            #pylab.show()
            pylab.show(False)
            pylab.pause(0.5)


        time.sleep(0.5)



class ClassMemMonitor():

    def __init__(self,dt=0.5):
        self.dt=dt
        pass


    def start(self):

        #t = threading.Thread(target=monitorMem)
        #t.start()

        monitorMem()


def test():
    MM=ClassMemMonitor()
    MM.start()
    

if __name__=="__main__":
    MM=ClassMemMonitor()
    MM.start()
