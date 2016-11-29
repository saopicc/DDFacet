import numpy as np
import multiprocessing
import Queue
import psutil
from multiprocessing import Process
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Other import MyLogger
from DDFacet.Array import NpShared
log = MyLogger.getLogger("ClassSmearMapping")

from DDFacet.Other import Multiprocessing


def _smearmapping_worker(jobitem, 
                         DATA, 
                         InfoSmearMapping, 
                         WorkerMapName, 
                         GridChanMapping):
    a0, a1 = jobitem
    rep = GiveBlocksRowsListBL(a0, a1, DATA, InfoSmearMapping, GridChanMapping)

    if rep is not None:
        # form up map name based on CPU ID
        ThisWorkerMapName = WorkerMapName % Multiprocessing.ProcessPool.getCPUId()

        BlocksRowsListBLWorker = NpShared.GiveArray(ThisWorkerMapName)
        if BlocksRowsListBLWorker is None:
            BlocksRowsListBLWorker = np.array([], np.int32)

        BlocksRowsListBL, BlocksSizesBL, NBlocksTotBL = rep
        BlocksRowsListBLWorker = np.concatenate((BlocksRowsListBLWorker, BlocksRowsListBL))
        NpShared.ToShared(ThisWorkerMapName, BlocksRowsListBLWorker)

        return {"bl": (a0, a1),
                    "MapName": ThisWorkerMapName,
                    "IdWorker": Multiprocessing.ProcessPool.getCPUId(),
                    "Empty": False,
                    "BlocksSizesBL": BlocksSizesBL,
                    "NBlocksTotBL": NBlocksTotBL}
    else:
        return {"bl": (a0, a1),
                    "IdWorker": Multiprocessing.ProcessPool.getCPUId(),
                    "Empty": True}


class ClassSmearMapping():

    def __init__(self, MS, radiusDeg=1., Decorr=0.98):
        self.radiusDeg = radiusDeg
        self.radiusRad = radiusDeg*np.pi/180
        self.Decorr = Decorr
        self.MS = MS

    def UnPackMapping(self):
        Map = NpShared.GiveArray("%sMappingSmearing" % (self.IdSharedMem))
        Nb = Map[0]
        NRowInBlocks = Map[1:Nb+1]
        StartRow = Map[Nb+1:2*Nb+1]
        # print
        # print NRowInBlocks.tolist()
        # print StartRow.tolist()
        MaxRow = 0

        for i in [3507]:  # range(Nb):
            ii = StartRow[i]
            MaxRow = np.max([MaxRow, np.max(Map[ii:ii+NRowInBlocks[i]])])
            print "(iblock= %i , istart= %i), Nrow=%i" % \
                  (i, StartRow[i], NRowInBlocks[i]), Map[ii:ii+NRowInBlocks[i]]
            #if MaxRow>=1080: stop

        print MaxRow
        # stop

    def BuildSmearMapping(self, DATA):
        print>>log, "Build decorrelation mapping ..."
        na = self.MS.na

        l = self.radiusRad
        dPhi = np.sqrt(6. * (1. - self.Decorr))
        NBlocksTot = 0
        self.BlocksRowsList = []

        InfoSmearMapping = {}
        InfoSmearMapping["freqs"] = self.MS.ChanFreq
        InfoSmearMapping["dfreqs"] = self.MS.dFreq
        InfoSmearMapping["dPhi"] = dPhi
        InfoSmearMapping["l"] = l
        BlocksRowsList = []

        BlocksRowsListBLWorker = np.array([], np. int32)
        for a0 in xrange(na):
            for a1 in xrange(na):
                if a0 == a1:
                    continue
                MapBL = GiveBlocksRowsListBL(a0, a1, DATA, InfoSmearMapping)
                if MapBL is None:
                    continue
                BlocksRowsListBL, BlocksSizesBL, NBlocksTotBL = MapBL
                BlocksRowsList += BlocksRowsListBL
                NBlocksTot += NBlocksTotBL
                BlocksRowsListBLWorker = np.concatenate((BlocksRowsListBLWorker,
                                                         BlocksRowsListBL))

    def BuildSmearMappingParallel(self, DATA, GridChanMapping):
        print>>log, "Build decorrelation mapping ..."

        na = self.MS.na

        l = self.radiusRad
        dPhi = np.sqrt(6. * (1. - self.Decorr))

        NChan = self.MS.ChanFreq.size
        self.BlocksRowsList = []

        InfoSmearMapping = {}
        InfoSmearMapping["freqs"] = self.MS.ChanFreq
        InfoSmearMapping["dfreqs"] = self.MS.dFreq
        InfoSmearMapping["dPhi"] = dPhi
        InfoSmearMapping["l"] = l
        BlocksRowsList = []

        joblist = [ (a0, a1) for a0 in xrange(na) for a1 in xrange(na) ]

        WorkerMapName = Multiprocessing.getShmURL("SmearWorker.%d")

        results = Multiprocessing.runjobs(joblist, title="Smear mapping", target=_smearmapping_worker,
                                            kwargs=dict(DATA=DATA,
                                                        InfoSmearMapping=InfoSmearMapping,
                                                        WorkerMapName=WorkerMapName,
                                                        GridChanMapping=GridChanMapping))

        # process worker results
        # for each map (each array resturned from worker), BlockSizes[MapName] will
        # contain a list of BlocksSizesBL entries returned from that worker
        BlockSizes = {}
        NTotBlocks = 0
        NTotRows = 0

        for DicoResult in results:
            if not DicoResult["Empty"]:
                MapName = DicoResult["MapName"]
                BlockSizes.setdefault(MapName,[]).append(np.array(DicoResult["BlocksSizesBL"]))
                NTotBlocks += DicoResult["NBlocksTotBL"]
                NTotRows += np.sum(DicoResult["BlocksSizesBL"])

        FinalMappingHeader = np.zeros((2*NTotBlocks+1, ), np.int32)
        FinalMappingHeader[0] = NTotBlocks

        iStart = 1
        # MM=np.array([],np.int32)
        MM = np.zeros((NTotBlocks, ), np.int32)

        FinalMapping = np.zeros((NTotRows, ), np.int32)
        iii = 0
        jjj = 0

        # now go through each per-worker mapping
        for MapName, block_sizes in BlockSizes.iteritems():
            #print>>log, "  Worker: %i"%(IdWorker)
            BlocksRowsListBLWorker = NpShared.GiveArray(MapName)
            if BlocksRowsListBLWorker is None:
                continue

            # FinalMapping=np.concatenate((FinalMapping,BlocksRowsListBLWorker))

            FinalMapping[
                iii:iii + BlocksRowsListBLWorker.size] = BlocksRowsListBLWorker[:]
            iii += BlocksRowsListBLWorker.size

            N = 0

            for BlocksSizesBL in block_sizes:
                # print "IdWorker,AppendId",IdWorker,AppendId,BlocksSizesBL
                # MM=np.concatenate((MM,BlocksSizesBL))
                MM[jjj:jjj+BlocksSizesBL.size] = BlocksSizesBL[:]
                jjj += BlocksSizesBL.size
                # print MM.shape,BlocksSizesBL
                N += np.sum(BlocksSizesBL)
            # print N,BlocksRowsListBLWorker.size

        cumul = np.cumsum(MM)
        FinalMappingHeader[1:1+NTotBlocks] = MM
        FinalMappingHeader[NTotBlocks+1+1::] = (cumul)[0:-1]
        FinalMappingHeader[NTotBlocks+1::] += 2*NTotBlocks+1

        #print>>log, "  Concat header"
        FinalMapping = np.concatenate((FinalMappingHeader, FinalMapping))
        for MapName in BlockSizes.iterkeys():
            NpShared.DelArray(MapName)

        #print>>log, "  Put in shared mem"

        NVis = np.where(DATA["A0"] != DATA["A1"])[0].size * NChan
        #print>>log, "  Number of blocks:         %i"%NTotBlocks
        #print>>log, "  Number of 4-Visibilities: %i"%NVis
        fact = (100.*(NVis-NTotBlocks)/float(NVis))

        # self.UnPackMapping()
        # print FinalMapping

        return FinalMapping, fact


def GiveBlocksRowsListBL(a0, 
                         a1, 
                         DATA, 
                         InfoSmearMapping, 
                         GridChanMapping):

    A0 = DATA["A0"]
    A1 = DATA["A1"]
    ind = np.where((A0 == a0) & (A1 == a1))[0]
    if(ind.size <= 1):
        return
    C = 3e8

    uvw = DATA["uvw"]
    dFreq = InfoSmearMapping["dfreqs"]
    dPhi = InfoSmearMapping["dPhi"]
    l = InfoSmearMapping["l"]
    freqs = InfoSmearMapping["freqs"]
    NChan = freqs.size
    nu0 = np.max(freqs)

    u, v, w = uvw[ind, :].T
    NChanBlockMax = 1e3
    du = u[1::] - u[0:-1]
    dv = v[1::] - v[0:-1]
    dw = w[1::] - w[0:-1]

    du = np.concatenate((du, [du[-1]]))
    dv = np.concatenate((dv, [dv[-1]]))
    dw = np.concatenate((dw, [dw[-1]]))

    Duv = C*(dPhi)/(np.pi*l*nu0)
    duvtot = 0

#    print Duv

    CurrentRows = []
    BlocksRowsListBL = []
    BlocksSizesBL = []
    NBlocksTotBL = 0
    for iRowBL in xrange(ind.size):
        CurrentRows.append(ind[iRowBL])
        # Frequency Block
        uv = np.sqrt(u[iRowBL]**2+v[iRowBL]**2+w[iRowBL]**2)
        dnu = (C/np.pi)*dPhi/(uv*l)
        NChanBlock = dnu/dFreq
        if NChanBlock < NChanBlockMax:
            NChanBlockMax = NChanBlock

        # Time Block
        duvtot += np.sqrt(du[iRowBL]**2+dv[iRowBL]**2+dw[iRowBL]**2)
        if (duvtot > Duv) | (iRowBL == (ind.size-1)):
            # BlocksRowsListBL.append(CurrentRows)

            NChanBlockMax = np.max([NChanBlockMax, 1])

            ch = np.arange(0, NChan, NChanBlockMax).tolist()

            if not((NChan) in ch):
                ch.append((NChan))
            NChBlocks = len(ch)
            ChBlock = np.int32(np.linspace(0, NChan, NChBlocks))

            # See if change in Grid ChannelMapping
            # GridChanMapping=np.array([0,0,0,1,1],np.int32)

            ChBlock_rampe = np.zeros((NChan, ), np.int32)
            for iChBlock in xrange(ChBlock.size-1):
                ch0 = ChBlock[iChBlock]
                ch1 = ChBlock[iChBlock+1]
                ChBlock_rampe[ch0:ch1] = iChBlock

            CH = (-1, -1)
            ChBlock_Cut_ChanGridMapping = []
            for iCh in xrange(NChan):
                CH0 = ChBlock_rampe[iCh]
                CH1 = GridChanMapping[iCh]
                CH_N = (CH0, CH1)
                if CH_N != CH:
                    ChBlock_Cut_ChanGridMapping.append(iCh)
                CH = CH_N
            if not((ChBlock[-1]) in ChBlock_Cut_ChanGridMapping):
                ChBlock_Cut_ChanGridMapping.append(ChBlock[-1])
            # print "%s ->
            # %s"%(str(ChBlock),str(np.array(ChBlock_Cut_ChanGridMapping,np.int32)))

            ChBlock = np.array(ChBlock_Cut_ChanGridMapping, np.int32)

            for iChBlock in xrange(ChBlock.size-1):
                ch0 = ChBlock[iChBlock]
                ch1 = ChBlock[iChBlock+1]
                ThiDesc = [ch0, ch1]
                ThiDesc += CurrentRows
                BlocksSizesBL.append(len(ThiDesc))
                BlocksRowsListBL += (ThiDesc)
                NBlocksTotBL += 1
            NChanBlockMax = 1e3
            CurrentRows = []
            duvtot = 0
    # stop
    return BlocksRowsListBL, BlocksSizesBL, NBlocksTotBL

