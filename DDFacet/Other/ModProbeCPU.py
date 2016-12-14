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
import socket

INTERVAL = 0.2



def getTimeList():
    """
    Fetches a list of time units the cpu has spent in various modes
    Detailed explanation at http://www.linuxhowtos.org/System/procstat.htm
    """
    cpuStats = file("/proc/stat", "r").readline()
    columns = cpuStats.replace("cpu", "").split(" ")
    return map(int, filter(None, columns))

def deltaTime(interval):
    """
    Returns the difference of the cpu statistics returned by getTimeList
    that occurred in the given time delta
    """
    timeList1 = getTimeList()
    time.sleep(interval)
    timeList2 = getTimeList()
    return [(t2-t1) for t1, t2 in zip(timeList1, timeList2)]

def getCpuLoad():
    """
    Returns the cpu load as a value from the interval [0.0, 1.0]
    """
    dt = list(deltaTime(INTERVAL))
    idle_time = float(dt[3])
    total_time = sum(dt)
    load = 1-(idle_time/total_time)
    return load

if __name__=="__main__":
    # while True:
    #     print "CPU usage=%.2f%%" % (getCpuLoad()*100.0)
    #     time.sleep(0.1)
    print "%s %7.2f"%(socket.gethostname(), getCpuLoad()*100.)





import numpy as np
import subprocess
import os
import sys
import socket

import psutil

def giveLoad():
    try:
        Plist=[psutil.Process(pnum) for pnum in psutil.get_pid_list()]
        loadList=[p.get_cpu_percent(interval=0.01) for p in Plist if p.username!="tasse"]
        load=np.sum(np.array(loadList))
    except:
        load=0.
    return load



import threading
import numpy as np
import time

class TrackCPU(threading.Thread):
     def __init__(self):
         threading.Thread.__init__(self)
         self._stop = threading.Event()
         self.ArrayLoads=np.zeros((10,),float)-1
         self.index=0
         self.AvgLoad=0

     def run(self):
         while True:
             
             self.ArrayLoads[self.index]=giveLoad()#getCpuLoad()*100.
             self.index+=1
             if self.index==self.ArrayLoads.size: self.index=0
             #print self.ArrayLoads
             self.AvgLoad=np.mean(self.ArrayLoads[self.ArrayLoads!=-1])
             time.sleep(1)
             if self.stopped(): break


     def stop(self):
         self._stop.set()

     def stopped(self):
         return self._stop.isSet()
