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
import socket
import glob
from collections import deque
import sys
import os
#pylab.ion()
import optparse
import pickle

SaveFile="last_MakeMask.obj"

def read_options():
    desc=""" """
    opt = optparse.OptionParser(usage='Task to start a monitoring task, Usage: %prog <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* General options")
    group.add_option('--Mode',type=str,help="Plot/Dump",default="Plot")
    group.add_option('--Reset',type=int,help="Reset dump",default=0)
    

    opt.add_option_group(group)

    
    options, arguments = opt.parse_args()

    f = open(SaveFile,"wb")
    pickle.dump(options,f)


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
    def __init__(self,options=None):
        self.dt=0.5
        self.Mode=options.Mode
        self.HostName=socket.gethostname()
        self.FileDump=os.path.expanduser("~/monitor.%s.csv"%self.HostName)
        if options.Reset:
            os.system("rm %s"%self.FileDump)

        
        self.DicoProfile={}
        
        self.dtype=[("time",np.float64),
                    ("mem",np.float32),
                    ("memAvail",np.float32),
                    ("memTotal",np.float32),
                    ("Swap_mem",np.float32),
                    ("Swap_memAvail",np.float32),
                    ("cpu",np.float32)]
        
        if self.Mode=="ReadAndPlot":
            self.loadDumped()
        else:
            self.DicoProfile[self.HostName]={}
            for f,_ in self.dtype:
                self.DicoProfile[self.HostName][f]=[]
            self.t0=time.time()
            
    def loadDumped(self):
        ll=glob.glob(os.path.expanduser("~/monitor.*.csv"))
        t0=None
        for l in ll:
            if "pipeline" in l:
                self.Steps=np.genfromtxt(l,dtype=[("time",np.float64),
                                                  ("Name","|S200"),
                                                  ("Type","|S200")],
                                         delimiter=",")

            else:
                host=l.split("monitor.")[1].split(".csv")[0]
                A=np.genfromtxt(l,dtype=self.dtype,
                                delimiter=",")
                self.DicoProfile[host]={}
                for f in A.dtype.fields:
                    self.DicoProfile[host][f]=A[f].tolist()
                if t0 is None:
                    t0=self.DicoProfile[host]["time"][0]
                else:
                    t0=np.min([t0,self.DicoProfile[host]["time"][0]])
        self.t0=t0

                    
    def update(self):
        vmem=psutil.virtual_memory()
        mem=vmem.used/float(2**20)/1024
        memAvail=vmem.available/float(2**20)/1024
        memTotal=vmem.total/float(2**20)/1024
        swap_mem=psutil.swap_memory()
        Swap_mem=swap_mem.used/float(2**20)/1024
        # #print("!!!",smem)
        # if self.Swap0 is None:
        #     self.Swap0=Swap_mem
        # #self.LSMem.append(Smem-self.Swap0)
        Swap_memAvail=swap_mem.total/float(2**20)/1024
        
        cpu=psutil.cpu_percent()
        AbsTime=time.time()
        
        if self.Mode=="Plot":
            host=self.HostName
            self.DicoProfile[host]["mem"].append(mem)
            self.DicoProfile[host]["memAvail"].append(memAvail)
            self.DicoProfile[host]["memTotal"].append(memTotal)
            self.DicoProfile[host]["Swap_mem"].append(Swap_mem)
            self.DicoProfile[host]["Swap_memAvail"].append(Swap_memAvail)
            self.DicoProfile[host]["time"].append(AbsTime)
            self.DicoProfile[host]["cpu"].append(cpu)
            
        elif self.Mode=="Dump":
            with open(self.FileDump, 'a') as file:
                s=f"{AbsTime}, {mem}, {memAvail}, {memTotal}, {Swap_mem}, {Swap_memAvail}, {cpu}"
                file.write('%s\n'%s)
                

    def plotDumped(self):
        self.fig=pylab.figure("[%s]"%self.HostName)
        while True:
            self.loadDumped()
            self.plotDumpedStep()
            pylab.draw()
            pylab.show(block=False)
            pylab.pause(0.1)
            time.sleep(5)

                
    def plotDumpedStep(self):
        LHost=list(self.DicoProfile.keys())
        ny=len(LHost)
        pylab.clf()
        for iPlot,host in enumerate(LHost):
            ax = pylab.subplot(ny,1,iPlot+1)
            _,ax2=self.plotOne(N=None,ax=ax,host=host)
            for Step in self.Steps:
                tt=(Step["time"]-self.t0)/60
                Name=Step["Name"].decode("ascii").replace(" ","").replace("_"," ")
                Type=Step["Type"]
                ax2.plot([tt,tt],[0,100])
                ax2.text(tt, 0, "[%s]"%Name, fontsize=7,
                         rotation=90, rotation_mode='anchor',
                         transform_rotates_text=True,color="red", weight="bold")
                
    def plot(self):
        pylab.clf()
        ax = pylab.subplot(111)
        self.plotOne(N=None,ax=ax,NMax=1000)
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)
        
    def plotUpDown(self):
        pylab.clf()
        ax = pylab.subplot(2,1,1)
        self.plotOne(N=self.NMax,ax=ax)
        ax = pylab.subplot(2,1,2)
        self.plotOne(N=None,ax=ax,NMax=1000)
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)

        
    def plotOne(self,N=None,ax=None,NMax=None,host=None):
        if host==None:
            host=self.HostName

        
        LMem=self.DicoProfile[host]["mem"]
        LSMem=self.DicoProfile[host]["Swap_mem"]
        LSMemAvail=self.DicoProfile[host]["Swap_memAvail"]
        LMemAvail=self.DicoProfile[host]["memAvail"]
        LMemTotal=self.DicoProfile[host]["memTotal"]
        LCPU=self.DicoProfile[host]["cpu"]
        LT=np.array(self.DicoProfile[host]["time"])
        LT-=self.t0
        LT/=60
        LT=LT.tolist()
        
        TotSeen=np.array(LMemAvail)+np.array(LMem)
        Cache=TotSeen-np.array(LMemTotal)
        PureRAM=np.array(LMem)-Cache
        
        if N is not None:
            N=np.min([N,len(LMem)])
        else:
            N=len(LMem)

        # incr=1
        # if NMax is not None:
        #     incr=int(np.max([1,N//NMax]))

        
        
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
        LSMem=np.array(LSMem)/np.array(LSMemAvail)*np.max(LMemTotal)

        x,y=GivePolygon(LT,LSMem)
        #print(y)
        #ax.fill(x,y,'', alpha=1, edgecolor='red',lw=2,hatch="/")
        #ax.fill(x,y,'gray', alpha=0.3, edgecolor='red',lw=1,hatch="*")
        
        ax.plot(LT,np.array(LSMem),lw=2,ls=":",color="red")
        # ax.plot(LT,np.array(LSMemAvail),lw=2,ls=":",color="red")
        ax.set_title("[%s]"%host)
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
        ax2.set_ylim(0,110)
        return ax,ax2
        #ax2.legend(loc=0)
        
        
    def startMonitor(self):
        self.fig=pylab.figure("[%s]"%self.HostName)
        while True:
            t0=time.time()
            while True:
                self.update()
                if time.time()-t0>5: break
                time.sleep(0.2)
            if self.Mode=="Plot":
                self.plot()

                
def driver():
    read_options()
    f = open(SaveFile,'rb')
    options = pickle.load(f)
    NMax=1000

    MM=ClassMemMonitor(options=options)
    if options.Mode=="Plot" or options.Mode=="Dump":
        MM.startMonitor()
    elif options.Mode=="ReadAndPlot":
        
        MM.plotDumped()
    
if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()

