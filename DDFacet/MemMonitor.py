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
import copy
SaveFile="last_MemMonitor.obj"
from collections import OrderedDict
import pprint
import matplotlib.gridspec as gridspec

def read_options():
    desc=""" """
    opt = optparse.OptionParser(usage='Task to start a monitoring task, Usage: %prog <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* General options")
    group.add_option('--Mode',type=str,help="Plot/Dump/ReadAndPlot",default="Plot")
    group.add_option('--Reset',type=int,help="Reset dump",default=0)
    group.add_option('--SaveDir',type=str,help="Reset dump",default="")
    

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


class ClassRegister():
    def __init__(self,Reset=0):
        self.HostName=socket.gethostname()
        self.FileDump=os.path.expanduser("~/DDF_register.%s.csv"%self.HostName)
        if Reset:
            os.system("rm %s"%self.FileDump)
        self.Steps=None
        self.DicoRegister={}

    def register(self,ss,StepType=""):
        with open(self.FileDump, 'a') as file:
            # s=f"{time.time()},{ss.replace(' ','_')},{StepType}"
            s=f"{time.time()},{ss},{StepType}"
            file.write('%s\n'%s)
            
    def load(self,host=""):
        ll=glob.glob(os.path.expanduser("~/DDF_register.*.csv"))
        for l in ll:
            host=l.split("DDF_register.")[1].split(".csv")[0]
            self.DicoRegister[host]=np.genfromtxt(l,dtype=[("time",np.float64),
                                                           ("Name","|S200"),
                                                           ("Type","|S200")],
                                                  delimiter=",")


        self.buildDicoRegister()
    
    def buildDicoRegister(self):
        tt=self.DicoRegister[self.HostName]["time"].copy()
        Name=self.DicoRegister[self.HostName]["Name"].copy()
        DoneStep=set()
        LDeleteInterval=[]

        DicoBlock=OrderedDict()
        dt=0
        for iStep,TimeStep,NameStep,TypeStep in zip(range(tt.size),tt,Name,self.DicoRegister[self.HostName]["Type"]):
            TypeStep=TypeStep.decode("ascii")
            NameStep=NameStep.decode("ascii")
            NameStep=NameStep.replace("-start","").replace("-stop","").replace("-end","")
            print(TimeStep,NameStep,TypeStep)
            if TypeStep=="Start":
                continue
            elif TypeStep=="Stop":
                DicoThisBlock=DicoBlock.get(NameStep,None)
                if DicoThisBlock is None:
                    print("Stopped but not started")
                    stop
                DicoBlock[NameStep]["TRange"].append(TimeStep)
                DicoBlock[NameStep]["Status"]="Done"
            else: #  TypeStep=="Imaging" or TypeStep=="Calibration"
                DicoThisBlock=DicoBlock.get(NameStep,None)
                if DicoThisBlock is None:
                    DicoBlock[NameStep]={}
                else:
                    D=copy.deepcopy(DicoThisBlock)
                    D["TRange"].append(TimeStep)
                    D["Status"]="ToBeRemoved"
                    LDeleteInterval.append(D)
                DicoBlock[NameStep]["TRange"]=[TimeStep]
                DicoBlock[NameStep]["TypeStep"]=TypeStep
                DicoBlock[NameStep]["Status"]="Ongoing"
                    
        pprint.pp(DicoBlock)
        pprint.pp(LDeleteInterval)
        self.DicoBlock=DicoBlock
        self.LDeleteInterval=LDeleteInterval

        for Name in DicoBlock.keys():
            if DicoBlock[Name]["Status"]=="Ongoing":
                t0,=DicoBlock[Name]["TRange"]
                t1=time.time()
                DicoBlock[Name]["TRange"]=[t0,t1]
            
        
        # self.LDeleteInterval=[]
        
        self.DicoBlockCorr=copy.deepcopy(self.DicoBlock)
        
        L=[copy.deepcopy(self.DicoBlock[Name]["TRange"]) for Name in self.DicoBlock.keys()]
        LName=[]
        for iName,Name in enumerate(self.DicoBlock.keys()):
            LName.append(Name)
            t0t1=self.DicoBlock[Name]["TRange"]
            if len(t0t1)==2:
                t0,t1=t0t1
                t0b,t1b=L[iName]
                for DBlock in self.LDeleteInterval:
                    t0d,t1d=DBlock["TRange"]
                    dt=t1d-t0d
                    if t0>=t1d:
                        t0b-=dt
                        t1b-=dt
                        L[iName]=[t0b,t1b]
            # elif len(t0t1)==1:
            #     t0,=t0t1
            #     t0b,=L[iName]
            #     for DBlock in self.LDeleteInterval:
            #         t0d,t1d=DBlock["TRange"]
            #         dt=t1d-t0d
            #         if t0>=t1d:
            #             t0b-=dt
            #             L[iName]=[t0]
            #     else:
            #         stop
                    
        for iName,TRange in enumerate(L):
            Name=LName[iName]
            self.DicoBlockCorr[Name]["TRange"]=TRange
            
        

        
class ClassMemMonitor():
    def __init__(self,options=None):
        self.dt=0.5
        self.Mode=options.Mode
        self.HostName=socket.gethostname()
        self.FileDump=os.path.expanduser("~/DDF_monitor.%s.csv"%self.HostName)
        if options.Reset:
            os.system("rm %s"%self.FileDump)

        self.SavePNG=False
        if len(options.SaveDir)>0:
            self.SaveDir=os.path.expanduser(options.SaveDir)
            os.system("rm %s"%self.SaveDir)
            os.system("mkdir -p %s"%self.SaveDir)
            self.iFig=0
            self.SavePNG=True
            
        self.Steps=None
        self.DicoProfile={}
        self.DicoIOProfile={}
        
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
        self.Register=ClassRegister()
        self.Register.load()
        
        ll=glob.glob(os.path.expanduser("~/DDF_monitor.*.csv"))
        t0=None
        for l in ll:
            host=l.split("DDF_monitor.")[1].split(".csv")[0]
            A=np.genfromtxt(l,dtype=self.dtype,
                            delimiter=",")
            self.DicoProfile[host]=A # {}
            if t0 is None:
                t0=self.DicoProfile[host]["time"][0]
            else:
                t0=np.min([t0,self.DicoProfile[host]["time"][0]])
        self.t0=t0

        ll=glob.glob(os.path.expanduser("~/DDF_io_monitor.*.csv"))
        dtype=[("time",np.float64),
                    ("read",np.float32),
                    ("write",np.float32)]
        for l in ll:
            host=l.split("DDF_io_monitor.")[1].split(".csv")[0]
            A=np.genfromtxt(l,dtype=dtype,
                            delimiter=",")
            self.DicoIOProfile[host]=A # {}

        def MaskDicoProfile(DicoProfile):
            DicoProfile=copy.deepcopy(DicoProfile)
            for host in DicoProfile.keys():
                LTime=[]
                tt=DicoProfile[host]["time"]
                Sel=np.ones((tt.size,),bool)
                ttc=tt.copy()
                for DicoBlock in self.Register.LDeleteInterval:
                    t0,t1=DicoBlock["TRange"]
                    ind=np.where((tt>t0)&(tt<t1))[0]
                    print(t0,t1,ind.size)
                    Sel[ind]=0
                    ttc[ind[-1]:]=ttc[ind[-1]:]-(t1-t0)
                ind=np.where(Sel)[0]
                
                DicoProfile[host]=DicoProfile[host][ind]
                DicoProfile[host]["time"]=ttc[ind]
            return DicoProfile
        
        DicoProfile=MaskDicoProfile(self.DicoProfile)
        # print(self.DicoProfile['node081']['time'].shape,DicoProfile['node081']['time'].shape)
        DicoIOProfile=MaskDicoProfile(self.DicoIOProfile)
        
        def toList(D):
            DD={}
            for host in D.keys():
                A=D[host]
                DD[host]={}
                for f in A.dtype.fields:
                    DD[host][f]=A[f].tolist()
            return DD
        
        self.DicoProfile=toList(DicoProfile)
        self.DicoIOProfile=toList(DicoIOProfile)

        
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
        self.fig=pylab.figure("[%s]"%self.HostName,figsize=(15,8))
        ModePlotMPI=1
        if ":" in self.Mode:
            ModePlotMPI=int(self.Mode.split(":")[1])
        while True:
            
            self.loadDumped()
            if ModePlotMPI==1:
                self.plotDumpedStep()
            elif ModePlotMPI==2:
                self.plotSeparate()
                
            pylab.draw()
            pylab.show()
            # block=False)
            pylab.pause(0.1)
            if self.SavePNG:
                self.fig.savefig("%s/Monitor%5.5i.png"%(self.SaveDir,self.iFig))
                self.iFig+=1
            time.sleep(5)

                
    def plotDumpedStep(self):
        LHost=list(self.DicoProfile.keys())
        ny=len(LHost)
        pylab.clf()
        for iPlot,host in enumerate(LHost):
            ax = pylab.subplot(ny,1,iPlot+1)
            _,ax2=self.plotOne(N=None,ax=ax,host=host)
            if self.Steps is not None:
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
        
        
        
    def plotSeparate(self):

        gridsize = (3, 2)
        fig=self.fig
        fig.clf()
        # ax1 = pylab.subplot(4,1,1)
        # ax1b = pylab.subplot(4,1,2,sharex=ax1)
        # ax2 = pylab.subplot(4,1,3,sharex=ax1)
        # ax2b = pylab.subplot(4,1,4,sharex=ax1)
        
        # nx=4
        # ax1 = pylab.subplot(nx,1,1)
        # ax1b = pylab.subplot(nx,1,2)
        # ax2 = pylab.subplot(nx,1,3)
        # ax3 = pylab.subplot(nx,1,4)
        # ax3b = ax3.twinx()

        fig=self.fig
        gs = gridspec.GridSpec(4,4, figure=fig)
        gs.update(wspace=0.05, hspace=0.0, left=0.15, right=0.95, bottom=0.08, top=0.8)
        fig.clf()
        
        ax1 = fig.add_subplot(gs[0,:])
        ax1b = fig.add_subplot(gs[1,:])
        ax2 = fig.add_subplot(gs[2,:])
        ax3 = fig.add_subplot(gs[3,:])
        ax3b = ax3.twinx()

        
        fig.subplots_adjust(hspace=0.03)
        
        N=None
        NMax=None
        host=None
        if host==None:
            host=self.HostName

        Ngg=2000
        # ##################################
        def giveGridded(x,y,q=0.5):
            y=np.array(y)
            x0=x.min()
            x1=x.max()
            Ng=np.min([Ngg,x.size])
            xx=np.linspace(x0-1,x1+1,Ng)
            Lyy=[]
            Lyym=[]
            Lxx=[]
            for ii in range(xx.size-1):
                ind=np.where((x>=xx[ii])&(x<xx[ii+1]))[0]
                # Lyy.append(np.quantile(y[ind],[0.16,0.5,0.84]))
                if ind.size==0: continue
                Lyy.append(np.quantile(y[ind],[q]))
                Lyym.append(np.max(y[ind]))
                Lxx.append((xx[ii]+xx[ii+1])/2)
            # return np.array(Lxx),np.array(Lyym)
            return np.array(Lxx),np.array(Lyy)[:,0]
        # ##################################
        def PlotR(ax,x0,y0,x1,y1,c,alpha=0.1):
            rect = pylab.Rectangle((x0,y0),(x1-x0),(y1-y0),color=c,alpha=alpha,zorder=1)
            ax.add_patch(rect)

        def plotRegister(host,ax,yminmax=[0,100],Mode=None):
            host=self.HostName
            DicoBlock=self.Register.DicoBlockCorr
            # [host]
            y0,y1=yminmax
            for Name in DicoBlock.keys():
                Block=DicoBlock[Name]
                TRange=Block["TRange"]
                if len(TRange)==1:
                    stop
                    t0=(TRange[0]-self.t0)/3600
                    t1=(time.time()-self.t0)/3600
                    ax.plot([t0,t0],[y0,y1],color="black",lw=2  ,ls="--",zorder=3)
                elif len(TRange)==2:
                    t0,t1=TRange
                    t0=(t0-self.t0)/3600
                    t1=(t1-self.t0)/3600
                    ax.plot([t0,t0],[y0,y1],color="black",lw=2  ,ls="--",zorder=3)
                    if Block["Status"]!="Ongoing":
                        ax.plot([t1,t1],[y0,y1],color="black",lw=2  ,ls="--",zorder=3)
                if Mode!="Line":
                    Type=Block["TypeStep"]
                    if Type=="Imaging":
                        SupTitle="%s"%Name
                        c="red"
                        alpha=0.2
                    elif Type=="Calibration":
                        SupTitle="%s"%Name
                        c="blue"
                        alpha=0.3
                    PlotR(ax,t0,y0,t1,y1,c,alpha=alpha)
                
        # # ##################################
                
        ImCPU=np.array(self.DicoProfile[host]["time"]).size
        
        NHost=len(self.DicoProfile.keys())
        
        LHosts=sorted(list(set(list(self.DicoProfile.keys()))))
        LHosts1=[self.HostName]
        for host in LHosts:
            if host!=self.HostName: 
                LHosts1.append(host)
        LHosts=LHosts1
        LHosts=LHosts[::-1]
        for ihost,host in enumerate(LHosts):
            LMem=self.DicoProfile[host]["mem"]
            LSMem=self.DicoProfile[host]["Swap_mem"]
            LSMemAvail=self.DicoProfile[host]["Swap_memAvail"]
            LMemAvail=self.DicoProfile[host]["memAvail"]
            LMemTotal=self.DicoProfile[host]["memTotal"]
            LCPU=self.DicoProfile[host]["cpu"]
            LT=np.array(self.DicoProfile[host]["time"])
            
            _,LMem=giveGridded(LT,LMem)
            _,LSMem=giveGridded(LT,LSMem)
            _,LSMemAvail=giveGridded(LT,LSMemAvail)
            _,LMemAvail=giveGridded(LT,LMemAvail)
            _,LMemTotal=giveGridded(LT,LMemTotal)
            _,LCPU0=giveGridded(LT,LCPU,q=0.16)
            _,LCPUm=giveGridded(LT,LCPU,q=0.5)
            LT,LCPU1=giveGridded(LT,LCPU,q=0.84)
            
            LT-=self.t0
            LT/=3600
            LT=LT.tolist()

            
            TotSeen=np.array(LMemAvail)+np.array(LMem)
            Cache=TotSeen-np.array(LMemTotal)
            PureRAM=np.array(LMem)-Cache
            
            N=len(LMem)
            
            if len(LMem)<2: return
    
            plotRegister(host,ax1,yminmax=[0,100])
            
            plotRegister(host,ax2,yminmax=[0,1.1*np.max(LMemTotal)],Mode="Line")
            plotRegister(host,ax1b,yminmax=[0,NHost],Mode="Line")
            # Total Available
            x,y=GivePolygon(LT,LMemTotal)
            ax2.fill(x,y,'black', alpha=0.1, edgecolor='black')
    
            # Cache
            # ax.plot(LT,LMemTotal-np.array(LMem),lw=2,color="green")
            x,y=GivePolygon(LT,np.array(LMem))
            ax2.fill(x,y,'black', alpha=0.1, edgecolor='blue',lw=2)
            
            # Total used excluding cache
            #x,y=GivePolygon(LT,np.array(LShared)+np.array(PureRAM))
            x,y=GivePolygon(LT,np.array(PureRAM))
            ax2.fill(x,y,'black', alpha=0.3, edgecolor='blue',lw=2)
            
            # memory
            # ax.plot(LT,PureRAM,lw=2,color="blue")
            x,y=GivePolygon(LT,PureRAM)
            ax2.fill(x,y,'green', alpha=0.3, edgecolor='green',lw=2,hatch="//")
    
            # swap
            LSMem=np.array(LSMem)/np.array(LSMemAvail)*np.max(LMemTotal)
    
            x,y=GivePolygon(LT,LSMem)
            #print(y)
            #ax.fill(x,y,'', alpha=1, edgecolor='red',lw=2,hatch="/")
            #ax.fill(x,y,'gray', alpha=0.3, edgecolor='red',lw=1,hatch="*")
            
            ax2.plot(LT,np.array(LSMem),lw=2,ls=":",color="red")
            # ax.plot(LT,np.array(LSMemAvail),lw=2,ls=":",color="red")
            #ax.set_title("[%s]"%host)
            # CPU

            y0,ym,y1=LCPU0,LCPUm,LCPU1
            if host==self.HostName:
                ls="-"
                color="purple"
                ax1.fill_between(LT,y0,y1, color=color,ls=ls,alpha=0.5)
                # ax1.plot(LT,ym, color=color,ls=ls)
            else:
                ls=":"
                color="blue"
                ax1.fill_between(LT,y0,y1, color=color,ls=ls,alpha=0.25)
                
            
            
            
            

            LCPU=np.array(LCPU)
            vs=y1
            vs=ym
            normal = pylab.Normalize(0,3)
            from matplotlib import colors, cm

            def fMyScale(x,RMS=1.): return np.arcsinh((x/RMS)/2)/np.log(10)
            def inv_fMyScale(x,RMS=1.): return RMS*np.sinh(x*np.log(10))*2
            
            # normal = colors.SymLogNorm(linthresh=10, linscale=10,
            #                            vmin=0, vmax=5000, base=10)
            
            # if host==self.HostName:
            #     colors = pylab.cm.Purples(normal(vs))
            # else:
            #     colors = pylab.cm.Blues(normal(vs))
                
            # colors = pylab.cm.Purples(normal(vs))
            # colors = pylab.cm.PuRd(normal(vs))
            colors = pylab.cm.BuGn(normal(vs))
            colors = pylab.cm.BuGn(normal(fMyScale(vs)))
            colors = pylab.cm.PuBuGn(normal(fMyScale(vs)))
            normal = pylab.Normalize(0,100)
            colors = pylab.cm.PuBuGn(normal(vs))
            colors = pylab.cm.binary(normal(vs))
            colors = pylab.cm.BuPu(normal(vs))
            LT=np.array(LT)
            dt=np.median(LT[1:]-LT[:-1])
            for itime in range(LT.size-1):
                t0,t1=LT[itime],LT[itime+1]
                h=1
                w=t1-t0
                x=(t0+t1)/2
                y=ihost
                rect = pylab.Rectangle((x-w/2,y),w,h,color=colors[itime])
                ax1b.add_patch(rect)
                
            # import matplotlib.colorbar as cbar
            # cax, _ = cbar.make_axes(ax1b) 
            # cb2 = cbar.ColorbarBase(cax, cmap=pylab.cm.jet,norm=normal)
        for ihost in range(NHost):
            ax1b.plot([LT.min(),LT.max()],[ihost,ihost],lw=2,ls="-",color="black")
            
        LL=copy.deepcopy(LHosts)
        LL[-1]="%s\n%s"%(LL[-1],"[rank0]")
        
        ax1b.set_yticks(np.arange(NHost)+0.5,LL)
        ax2.set_ylabel("Mem [GB]")
        ax1.set_ylabel("CPU [%]")


        #ax2.set_ylabel(r"Temperature ($^\circ$C)")
        #ax2.set_ylim(0, 35)




        
        
        ax2.set_ylim(0,1.1*np.max(LMemTotal))
        ax1.grid()
        ax1b.grid(axis="x")
        
        # ax1b.set_title("")
        # ax2.set_title("")
        ax1.set_xticklabels([])
        ax1b.set_xticklabels([])
        ax2.set_xticklabels([])
        
        ax2.grid()
        ax1.set_ylim(0,110)
        ax1b.set_ylim(0,NHost)

        Max=0
        for ihost,host in enumerate(LHosts):
            Lio_write=self.DicoIOProfile[host]["write"]
            Lio_read=self.DicoIOProfile[host]["read"]
            Lio_T=np.array(self.DicoIOProfile[host]["time"])
            _,Lio_write=giveGridded(Lio_T,Lio_write,q=1.)
            Lio_T,Lio_read=giveGridded(Lio_T,Lio_read,q=1.)
            Lio_T-=self.t0
            Lio_T/=3600
            Lio_T=Lio_T.tolist()
            
            x,y=GivePolygon(Lio_T,Lio_read)
            l0=ax3.fill(x,y,'green', alpha=0.5, edgecolor='black',label="Read")
            x,y=GivePolygon(Lio_T,Lio_write)
            l1=ax3b.fill(x,y,'red', alpha=0.5, edgecolor='black',label="Write")
            
        Max=10000
        plotRegister(host,ax3,yminmax=[0,Max],Mode="Line")
        ax3.set_ylabel("Read [green, MB/s]")
        ax3b.set_ylabel("Write [red, MB/s]")
        ax3.set_xlabel("Time [hours]")
        ax3.set_ylim(0,Max)
        ax3b.set_ylim(0,Max)
        ax3.grid()

        Lx=[]
        LName=[]
        for Name in self.Register.DicoBlockCorr.keys():
            t0t1=self.Register.DicoBlockCorr[Name]["TRange"]
            if len(t0t1)==2:
                t0,t1=t0t1
            else:
                stop
                t0,=t0t1
                t1=time.time()
            t0=(t0-self.t0)/3600
            t1=(t1-self.t0)/3600
            
            tc=(t0+t1)/2
            print(Name,t0,t1,tc)
            Lx.append(tc)
            if "DDFacet" in Name:
                Name=Name.replace("] ","]\n")
            LName.append(Name)
        ax1top = ax1.twiny()
        ax1top.plot([])
        #ax1top.set_xlabel('epochs')
        ax1top.set_xticks(Lx)
        ax1top.set_xticklabels(LName, rotation=45,
                               rotation_mode="anchor",
                               ha='left',weight='bold')
        #ax1top.set_title("Step")

        t0,t1=np.min(LT),np.max(LT)
        ax1.set_xlim(t0,t1)
        ax1b.set_xlim(t0,t1)
        ax2.set_xlim(t0,t1)
        ax3.set_xlim(t0,t1)
        ax1top.set_xlim(t0,t1)
        
        # ax3.legend((l0,l1))
        
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

def main():
    f = open(SaveFile,'rb')
    options = pickle.load(f)
    NMax=1000

    MM=ClassMemMonitor(options=options)
    if options.Mode=="Plot" or options.Mode=="Dump":
        MM.startMonitor()
    elif options.Mode.startswith("ReadAndPlot"):
        MM.plotDumped()
                
                
def driver():
    read_options()
    main()
    
if __name__=="__main__":
    # do not place any other code here --- cannot be called as a package entrypoint otherwise, see:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    driver()

