from Other import MyLogger
log=MyLogger.getLogger("ClassSmearMapping")
import numpy as np
from Array import NpShared
import multiprocessing
from progressbar import ProgressBar

class ClassSmearMapping():
    def __init__(self,MS,radiusDeg=1.,Decorr=0.98,IdSharedMem="",NCPU=6):
        self.radiusDeg=radiusDeg
        self.radiusRad=radiusDeg*np.pi/180
        self.Decorr=Decorr
        self.NCPU=NCPU
        self.MS=MS
        self.IdSharedMem=IdSharedMem

    def UnPackMapping(self):
        Map=NpShared.GiveArray("%sMappingSmearing"%(self.IdSharedMem))
        Nb=Map[0]
        NRowInBlocks=Map[1:Nb+1]
        StartRow=Map[Nb+1:2*Nb+1]
        print
        print StartRow

        for i in range(Nb)[0:10]:
            ii=StartRow[i]+2*Nb+1
            print "(iblock= %i , istart= %i), Nrow=%i"%(i,StartRow[i],NRowInBlocks[i]),Map[ii:ii+NRowInBlocks[i]]


    def BuildSmearMapping(self,DATA):
        print>>log, "Build decorrelation mapping ..."

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

        BlocksRowsListBLWorker=np.array([],np.int32)
        for a0 in range(na):
            for a1 in range(na):
                if a0==a1: continue
                MapBL=GiveBlocksRowsListBL(a0,a1,InfoSmearMapping,self.IdSharedMem)
                if MapBL==None: continue
                BlocksRowsListBL,BlocksSizesBL,NBlocksTotBL=MapBL
                BlocksRowsList+=BlocksRowsListBL
                NBlocksTot+=NBlocksTotBL
                BlocksRowsListBLWorker=np.concatenate((BlocksRowsListBLWorker,BlocksRowsListBL))

        NpShared.DelAll("%sSmearMapping"%self.IdSharedMem)


    def BuildSmearMappingParallel(self,DATA):
        print>>log, "Build decorrelation mapping ..."

        flags=DATA["flags"]
        uvw=DATA["uvw"]
        data=DATA["data"]
        A0=DATA["A0"]
        A1=DATA["A1"]

        # ind=np.where((A0==0))[0][0:36*10]
        # uvw=uvw[ind]
        # A0=A0[ind]
        # A1=A1[ind]

        
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
            
        ThisWorkerMapName="%sBlocksRowsList"%(self.IdSharedMem)
        NpShared.DelAll(ThisWorkerMapName)
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
                        InfoSmearMapping,
                        ii)
            workerlist.append(W)
            workerlist[ii].start()


        DicoWorkerResult={}
        for IdWorker in range(NCPU):
            DicoWorkerResult[IdWorker]={}
            DicoWorkerResult[IdWorker]["BlocksSizesBL"]={}

        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="Mapping ", HeaderSize=10,TitleSize=13)
        pBAR.render(0, '%4i/%i' % (0,NJobs))
        iResult=0
        NTotBlocks=0
        NTotRows=0
        while iResult < NJobs:
            DicoResult=result_queue.get()
            if DicoResult["Success"]:
                iResult+=1
                if DicoResult["Empty"]!=True:
                    IdWorker=DicoResult["IdWorker"]
                    AppendId=DicoResult["AppendId"]
                    DicoWorkerResult[IdWorker]["BlocksSizesBL"][AppendId]=DicoResult["BlocksSizesBL"]
                    NTotBlocks+=DicoResult["NBlocksTotBL"]
                    NTotRows+=np.sum(DicoResult["BlocksSizesBL"])
                    #print DicoResult["NBlocksTotBL"],len(DicoResult["BlocksSizesBL"])

            NDone=iResult
            intPercent=int(100*  NDone / float(NJobs))
            pBAR.render(intPercent, '%4i/%i' % (NDone,NJobs))

        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()

        NpShared.DelAll("%sSmearMapping"%self.IdSharedMem)
        



        FinalMappingHeader=np.zeros((2*NTotBlocks+1,),np.int32)
        FinalMappingHeader[0]=NTotBlocks
        
        NVis=np.where(A0!=A1)[0].size*NChan
        
        print>>log, "  Number of blocks:         %i"%NTotBlocks
        print>>log, "  Number of 4-Visibilities: %i"%NVis
        print>>log, "  Compression factor:       %.5f"%((NVis-NTotBlocks)/float(NVis))
        
        iStart=1
        MM=np.array([],np.int32)

        FinalMapping=np.zeros((NTotRows,),np.int32)
        iii=0
        for IdWorker in range(NCPU):
            print>>log, "  Worker: %i"%(IdWorker)
            ThisWorkerMapName="%sBlocksRowsList.Worker_%3.3i"%(self.IdSharedMem,IdWorker)
            BlocksRowsListBLWorker=NpShared.GiveArray(ThisWorkerMapName)
            if BlocksRowsListBLWorker==None: continue
            
            #FinalMapping=np.concatenate((FinalMapping,BlocksRowsListBLWorker))
            
            FinalMapping[iii:iii+BlocksRowsListBLWorker.size]=BlocksRowsListBLWorker[:]
            iii+=BlocksRowsListBLWorker.size

            N=0

            for AppendId in sorted(DicoWorkerResult[IdWorker]["BlocksSizesBL"].keys()):
                BlocksSizesBL=DicoWorkerResult[IdWorker]["BlocksSizesBL"][AppendId]
                #print "IdWorker,AppendId",IdWorker,AppendId,BlocksSizesBL
                MM=np.concatenate((MM,BlocksSizesBL))
                #print MM.shape,BlocksSizesBL
                N+=np.sum(BlocksSizesBL)
            #print N,BlocksRowsListBLWorker.size

        cumul=np.cumsum(MM)
        FinalMappingHeader[1:1+NTotBlocks]=MM
        FinalMappingHeader[NTotBlocks+1+1:2*NTotBlocks+1]=(cumul)[:-1]

        print>>log, "  Concat header"
        FinalMapping=np.concatenate((FinalMappingHeader,FinalMapping))
        NpShared.DelAll("%sBlocksRowsList"%(self.IdSharedMem))

        print>>log, "  Put in shared mem"
        Map=NpShared.ToShared("%sMappingSmearing"%(self.IdSharedMem),FinalMapping)
        self.UnPackMapping()

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
    BlocksSizesBL=[]
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
        if (duvtot>Duv)|(iRowBL==(ind.size-1)):
            #BlocksRowsListBL.append(CurrentRows)
        
            NChanBlockMax=np.max([NChanBlockMax,1])
            
            ch=np.arange(0,NChan,NChanBlockMax).tolist()
            
            if not((NChan) in ch): ch.append((NChan))
            NChBlocks=len(ch)
            ChBlock=np.int32(np.linspace(0,NChan,NChBlocks))
            
            for iChBlock in range(ChBlock.size-1):
                ch0=ChBlock[iChBlock]
                ch1=ChBlock[iChBlock+1]
                ThiDesc=[ch0,ch1]
                ThiDesc+=CurrentRows
                BlocksSizesBL.append(len(ThiDesc))
                BlocksRowsListBL+=(ThiDesc)
                NBlocksTotBL+=1
            NChanBlockMax=1e3
            CurrentRows=[]
            duvtot=0

    return BlocksRowsListBL,BlocksSizesBL,NBlocksTotBL


class WorkerMap(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 IdSharedMem,
                 InfoSmearMapping,
                 IdWorker):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()

        self.IdSharedMem=IdSharedMem
        self.InfoSmearMapping=InfoSmearMapping
        self.IdWorker=IdWorker
        self.AppendId=0

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

            rep=GiveBlocksRowsListBL(a0,a1,self.InfoSmearMapping,self.IdSharedMem)

            if rep!=None:
                ThisWorkerMapName="%sBlocksRowsList.Worker_%3.3i"%(self.IdSharedMem,self.IdWorker)
                BlocksRowsListBLWorker=NpShared.GiveArray(ThisWorkerMapName)
                if BlocksRowsListBLWorker==None:
                    BlocksRowsListBLWorker=np.array([],np.int32)

                BlocksRowsListBL,BlocksSizesBL,NBlocksTotBL=rep
                #print "AppendId:",self.AppendId,BlocksSizesBL,BlocksRowsListBL
                BlocksRowsListBLWorker=np.concatenate((BlocksRowsListBLWorker,BlocksRowsListBL))
                NpShared.ToShared(ThisWorkerMapName,BlocksRowsListBLWorker)
                self.result_queue.put({"Success":True,"bl":(a0,a1),"IdWorker":self.IdWorker,"AppendId":self.AppendId,"Empty":False,
                                       "BlocksSizesBL":BlocksSizesBL,"NBlocksTotBL":NBlocksTotBL})
                self.AppendId+=1
                
            else:
                self.result_queue.put({"Success":True,"bl":(a0,a1),"IdWorker":self.IdWorker,"AppendId":self.AppendId,"Empty":True})
