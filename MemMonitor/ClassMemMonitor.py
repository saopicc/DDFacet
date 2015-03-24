import resource
import psutil
import time
import os
import numpy as np
import threading

import pylab

def monitorMem():
    LMem=[]
    LSMem=[]
    LSMemAvail=[]
    LMemAvail=[]
    LMemTotal=[]
    LCPU=[]
    t0=time.time()
    LT=[]
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
        LSMem.append(Smem)

        SmemAvail=smem.total/float(2**20)
        LSMemAvail.append(SmemAvail)

        

        TotSeen=np.array(LMemAvail)+np.array(LMem)
        Shared=TotSeen-np.array(LMemTotal)
        PureRAM=np.array(LMem)-Shared

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
            # CPU
            ax2.plot(LT,LCPU, color="gray")

            # memory
            ax.plot(LT,PureRAM,lw=2,color="blue")
            ax.plot(LT,LMemTotal-np.array(LMem),lw=2,color="green")
            ax.plot(LT,LMemTotal,lw=2,color="black")
            #ax.plot(LT,TotSeen,lw=2,color="red")
            ax.plot(LT,np.array(LMem),lw=2,color="blue",ls="--")


            # swap
            ax.plot(LT,np.array(LSMem),lw=2,ls=":",color="blue")
            ax.plot(LT,np.array(LSMemAvail),lw=2,ls=":",color="red")
            
            #ax.legend(loc=0)
            ax.grid()

            
            ax.set_ylabel("Mem [MB]")
            ax2.set_ylabel("CPU [%]")
            ax.set_xlabel("Time [min.]")
            #ax2.set_ylabel(r"Temperature ($^\circ$C)")
            #ax2.set_ylim(0, 35)
            ax.set_ylim(0,1.1*np.max(np.array(LMemAvail)+np.array(LMem)))
            #ax2.legend(loc=0)


            pylab.draw()
            pylab.show(False)
            pylab.pause(0.5)


        time.sleep(0.1)



class ClassMemMonitor():

    def __init__(self,dt=0.5):
        self.dt=dt
        pass


    def start(self):

        #t = threading.Thread(target=monitorMem)
        #t.start()

        monitorMem()


if __name__=="__main__":
    MM=ClassMemMonitor()
    MM.start()
