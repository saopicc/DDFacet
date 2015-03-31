from Other import MyLogger
log=MyLogger.getLogger("ClassSmearMapping")
import numpy as np
from Array import NpShared
import multiprocessing
from progressbar import ProgressBar

class ClassSmearMapping():
    def __init__(self,MS,radiusDeg=5.,Decorr=0.98,IdSharedMem="",NCPU=6):
        self.radiusDeg=radiusDeg
        self.radiusRad=radiusDeg*np.pi/180
        self.Decorr=Decorr
        self.NCPU=NCPU
        self.MS=MS
        self.IdSharedMem=IdSharedMem

    def BuildSmearMapping(self,DATA):
        print>>log, "Build smearing mapping ..."

        flags=DATA["flags"]
        uvw=DATA["uvw"]
        data=DATA["data"]
        A0=DATA["A0"]
        A1=DATA["A1"]
        
        DicoSmearMapping={}
        DicoSmearMapping["A0"]=A0
        DicoSmearMapping["A1"]=A1
        DicoSmearMapping["uvw"]=uvw
        NpShared.DicoToShared("%sSmearMapping"%self.IdSharedMem,DicoSmearMapping)

        times=DATA["times"]

        na=self.MS.na
        dFreq=self.MS.dFreq

        l=self.radiusRad
        dPhi=np.sqrt(6.*(1.-self.Decorr))
        NBlocksTot=0
        BlocksRowsList=[]

        NChan=self.MS.ChanFreq.size
        self.BlocksRowsList=[]

        InfoSmearMapping={}
        InfoSmearMapping["freqs"]=self.MS.ChanFreq
        InfoSmearMapping["dfreqs"]=self.MS.dFreq
        InfoSmearMapping["dPhi"]=dPhi
        InfoSmearMapping["l"]=l
        BlocksRowsList=[]

        for a0 in range(na):
            for a1 in range(na):
                if a0==a1: continue
                MapBL=GiveBlocksRowsListBL(a0,a1,InfoSmearMapping,IdSharedMem)
                if MapBL==None: continue
                BlocksRowsListBL,NBlocksTotBL=MapBL
                BlocksRowsList+=BlocksRowsListBL
                NBlocksTot+=NBlocksTotBL

    def BuildSmearMappingParallel(self,DATA):
        print>>log, "Build smearing mapping ..."

        flags=DATA["flags"]
        uvw=DATA["uvw"]
        data=DATA["data"]
        A0=DATA["A0"]
        A1=DATA["A1"]
        
        DicoSmearMapping={}
        DicoSmearMapping["A0"]=A0
        DicoSmearMapping["A1"]=A1
        DicoSmearMapping["uvw"]=uvw
        NpShared.DicoToShared("%sSmearMapping"%self.IdSharedMem,DicoSmearMapping)

        times=DATA["times"]

        na=self.MS.na
        dFreq=self.MS.dFreq

        l=self.radiusRad
        dPhi=np.sqrt(6.*(1.-self.Decorr))
        NBlocksTot=0
        BlocksRowsList=[]

        NChan=self.MS.ChanFreq.size
        self.BlocksRowsList=[]

        InfoSmearMapping={}
        InfoSmearMapping["freqs"]=self.MS.ChanFreq
        InfoSmearMapping["dfreqs"]=self.MS.dFreq
        InfoSmearMapping["dPhi"]=dPhi
        InfoSmearMapping["l"]=l
        BlocksRowsList=[]
        
        
        NCPU=self.NCPU
            

        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()
        for a0 in range(na):
            for a1 in range(na):
                if a0==a1: continue
                work_queue.put((a0,a1))

        NJobs=work_queue.qsize()
        workerlist=[]
        for ii in range(NCPU):
            W=WorkerMap(work_queue, 
                        result_queue,
                        self.IdSharedMem,
                        InfoSmearMapping)
            workerlist.append(W)
            workerlist[ii].start()

        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="Mapping ", HeaderSize=10,TitleSize=13)
        pBAR.render(0, '%4i/%i' % (0,NJobs))
        iResult=0
        while iResult < NJobs:
            DicoResult=result_queue.get()
            if DicoResult["Success"]:
                iResult+=1
            NDone=iResult
            intPercent=int(100*  NDone / float(NJobs))
            pBAR.render(intPercent, '%4i/%i' % (NDone,NJobs))



        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()

        NpShared.DelArray("%sModelImage"%self.IdSharedMem)
            
        return True





def GiveBlocksRowsListBL(a0,a1,InfoSmearMapping,IdSharedMem):
    DicoSmearMapping=NpShared.SharedToDico("%sSmearMapping"%IdSharedMem)

    A0=DicoSmearMapping["A0"]
    A1=DicoSmearMapping["A1"]
    ind=np.where((A0==a0)&(A1==a1))[0]
    if(ind.size==0): return
    C=3e8

    uvw=DicoSmearMapping["uvw"]
    dFreq=InfoSmearMapping["dfreqs"]
    dPhi=InfoSmearMapping["dPhi"]
    l=InfoSmearMapping["l"]
    freqs=InfoSmearMapping["freqs"]
    NChan=freqs.size
    nu0=np.mean(freqs)
    
    u,v,w=uvw[ind,:].T
    NChanBlockMax=1e3
    du=u[1::]-u[0:-1]
    dv=v[1::]-v[0:-1]
    dw=w[1::]-w[0:-1]
    
    du=np.concatenate((du,[du[-1]]))
    dv=np.concatenate((dv,[dv[-1]]))
    dw=np.concatenate((dw,[dw[-1]]))
    
    Duv=C*dPhi/(np.pi*l*nu0)
    duvtot=0
        
    CurrentRows=[]
    BlocksRowsListBL=[]
    NBlocksTotBL=0
    for iRowBL in range(ind.size):
        CurrentRows.append(ind[iRowBL])
        # Frequency Block
        uv=np.sqrt(u[iRowBL]**2+v[iRowBL]**2+w[iRowBL]**2)
        dnu=(C/np.pi)*dPhi/(uv*l)
        NChanBlock=dnu/dFreq
        if NChanBlock<NChanBlockMax:
            NChanBlockMax=NChanBlock

        # Time Block
        duvtot+=np.sqrt(du[iRowBL]**2+dv[iRowBL]**2+dw[iRowBL]**2)
        if duvtot>Duv:
            #BlocksRowsListBL.append(CurrentRows)
        
            NChanBlockMax=np.max([NChanBlockMax,1])
            
            ch=np.arange(0,NChan,NChanBlockMax).tolist()
            
            if not((NChan-1) in ch): ch.append((NChan-1))
            NChBlocks=len(ch)
            ChBlock=np.int32(np.linspace(0,NChan-1,NChBlocks))
            
            for iChBlock in range(ChBlock.size-1):
                ch0=ChBlock[iChBlock]
                ch1=ChBlock[iChBlock+1]
                ThiDesc=[ch0,ch1]
                ThiDesc+=CurrentRows
                BlocksRowsListBL.append(ThiDesc)
            NChanBlockMax=1e3
            CurrentRows=[]
            duvtot=0
            NBlocksTotBL+=1
            
    return BlocksRowsListBL,NBlocksTotBL


class WorkerMap(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 IdSharedMem,
                 InfoSmearMapping):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()

        self.IdSharedMem=IdSharedMem
        self.InfoSmearMapping=InfoSmearMapping

    def shutdown(self):
        self.exit.set()

    def run(self):
        #print multiprocessing.current_process()
        while not self.kill_received:
            #gc.enable()
            #a0,a1 = self.work_queue.get()
            try:
                a0,a1 = self.work_queue.get()
            except:
                break

            print a0,a1
            GiveBlocksRowsListBL(a0,a1,self.InfoSmearMapping,self.IdSharedMem)               
            
            self.result_queue.put({"Success":True})
