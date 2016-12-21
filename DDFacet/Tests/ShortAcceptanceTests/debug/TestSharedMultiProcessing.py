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

# http://stackoverflow.com/questions/25770971/how-to-modify-different-parts-of-a-numpy-array-of-complex-numbers-in-parallel-us
# http://briansimulator.org/sharing-numpy-arrays-between-processes/

from multiprocessing import sharedctypes
import numpy as np
import numpy
import multiprocessing
import ctypes
import ClassTimeIt
from numpy import ctypeslib

def NumpyToShared(A):

    size = A.size
    shape = A.shape
    sizeTot=size
    dtype=A.dtype
    if dtype==np.float32:
        DicoType={"ctype":"f"}
    elif dtype==np.float64:
        DicoType={"ctype":"d"}
    elif dtype==np.complex128:
        DicoType={"ctype":"d"}
    elif dtype==np.complex64:
        DicoType={"ctype":"f"}
    
    DicoType["nptype"]=dtype
        
    DicoType["ComplexMode"]=False
    if "complex" in str(A.dtype):
        sizeTot*=2
        DicoType["ComplexMode"]=True

    S_ctypes = sharedctypes.RawArray(DicoType["ctype"], sizeTot)
    #As = numpy.frombuffer(S_ctypes, dtype=DicoType["nptype"], count=size)
    As = ctypeslib.as_array(S_ctypes)

    if DicoType["ComplexMode"]:
        As[0::2]=A.reshape((size,)).real[:]
        As[1::2]=A.reshape((size,)).imag[:]
        #As[0:size]=A.reshape((size,)).real[:]
        #As[size::]=A.reshape((size,)).imag[:]
    else:
        As[:]=A.reshape((size,))[:]

    S_ctypes.shape=shape
    S_ctypes.DicoType=DicoType

    return S_ctypes

def SharedToNumpy(S_ctypes):
    if type(S_ctypes)==np.ndarray: return S_ctypes
    S = ctypeslib.as_array(S_ctypes)
    if S_ctypes.DicoType["ComplexMode"]:
        S = S.view(S_ctypes.DicoType["nptype"])
    S=S.reshape(S_ctypes.shape)
    #S.shape = S_ctypes.shape
    return S

def testComplex():
    At=np.random.randn(4,4)+1j*np.random.randn(4,4)

    #for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
    for dtype in [np.float64, np.complex128, np.float32, np.complex64]:
        A=dtype(At)
        As=NumpyToShared(A)
        npAs=SharedToNumpy(As)
        print A-npAs


# >>> shared_array_base = multiprocessing.Array(ctypes.c_double, 3*3*2)
# >>> shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
# >>> shared_array = shared_array.view(np.complex128).reshape(3, 3)


def test2(A=np.ones((5,5),dtype=np.float32)):

    #Actp=NumpyToShared(A)
    #start(A)
    Actp=NumpyToShared(A)
    start(Actp)


def start(Actp):
    ########### 
    work_queue = multiprocessing.Queue()
    jobs = range(50)

    NCPU=8
    for job in jobs:
        work_queue.put(job)
    result_queue = multiprocessing.Queue()

    workerlist=[]
    lock = multiprocessing.Lock()
    #S=None
    for ii in range(NCPU):
        ThisWorker=WorkerImager(work_queue, result_queue,lock)
        ThisWorker.setArray(Actp)
        workerlist.append(ThisWorker)
        workerlist[ii].start()
 
    results = []
    while len(results) < len(jobs):
        result = result_queue.get()
        results.append([])
            
    for ii in range(NCPU):
        workerlist[ii].shutdown()
        workerlist[ii].terminate()
        workerlist[ii].join()

    S = SharedToNumpy(Actp)
    print S[0,0]
    #print S_ctypes[:]
    



class WorkerImager(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,lock):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.lock=lock

    def setArray(self,S_ctypes):
        self.S = SharedToNumpy(S_ctypes)

    def shutdown(self):
        self.exit.set()
    def run(self):
        while not self.kill_received:
            #print "a"
            try:
                iFacet = self.work_queue.get()
            except:
                break

            #print np.mean(self.S)
            #self.lock.acquire()
            self.S+=1
            print self.S.__array_interface__["data"]
            print np.mean(self.S)
            
            #self.lock.release()
            
            self.result_queue.put([0])

