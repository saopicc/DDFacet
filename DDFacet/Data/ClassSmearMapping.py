import numpy as np
import itertools
from DDFacet.Other import MyLogger
from DDFacet.Array import NpShared
log = MyLogger.getLogger("ClassSmearMapping")

from DDFacet.Other import Multiprocessing


def _smearmapping_worker(jobitem, DATA, InfoSmearMapping, WorkerMapName, GridChanMapping):
    a0, a1 = jobitem
    rep = GiveBlocksRowsListBL_old(a0, a1, DATA, InfoSmearMapping, GridChanMapping)

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

    def BuildSmearMapping(self, DATA, GridChanMapping):
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
                stop
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

            FinalMapping[iii:iii+BlocksRowsListBLWorker.size] = BlocksRowsListBLWorker[:]
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


def GiveBlocksRowsListBL(a0, a1, DATA, InfoSmearMapping, GridChanMapping):

    A0 = DATA["A0"]
    A1 = DATA["A1"]
    row_index = np.where((A0 == a0) & (A1 == a1))[0]
    nrows = row_index.size
    if not nrows:
        return
    C = 3e8

    uvw = DATA["uvw"][row_index]
    dFreq = InfoSmearMapping["dfreqs"]   # channel width
    dPhi = InfoSmearMapping["dPhi"]      # delta-phi: max phase change allowed between rows
    l = InfoSmearMapping["l"]
    freqs = InfoSmearMapping["freqs"]
    NChan = freqs.size
    nu0 = np.max(freqs)

    # find delta-uv at each row: will have size of (Nrow,2)
    if True:  # compute delta-uv
        duv = uvw[:,:2].copy()
        duv[:-1,:2] = uvw[1:,:2] - uvw[:-1,:2]
    else:     # comnpute delta-uvw
        duv = uvw.copy()
        duv[:-1,:] = uvw[1:,:] - uvw[:-1,:]
    # last delta copied from previous row        
    duv[-1,:] = duv[-2,:]
    # convert to distance
    duv_array = np.sqrt((duv**2).sum(1))     # delta-uv-distance at each row

    # critical delta-uv interval for smearing
    Duv = C*(dPhi)/(np.pi*l*nu0)

    # uvw-distance at each row (size Nrow-1)
    uv = np.sqrt((uvw**2).sum(1))
    dnu = (C / np.pi) * dPhi / (uv * l)  # delta-nu for each row
    fracsizeChanBlock = dnu / dFreq  # max size of averaging block, in fractional channels, for each row

    duv_array = np.sqrt((duv**2).sum(1))     # delta-uv-distance at each row

    if True:  # fast-style, maybe not as precise
        # accumulate duv, and divide by Duv. Take the floor of that -- that gives us an integer, the time block number
        # for each row
        if Duv:
            rowblock = np.int32(duv_array.cumsum() / Duv)
            # find row numbers at which the time blocks are cut (adding nrows at end)
            blockcut = np.where(np.roll(rowblock,1) != rowblock)[0]
            nblocks = len(blockcut)
            blockcut = list(blockcut) + [nrows]
            # make list of rows slices for each time block
            block_slices = [ slice(blockcut[i],blockcut[i+1]) for i in xrange(nblocks) ]
        else:
            block_slices = [ slice(i,i+1) for i in xrange(nrows) ]
    else:   # slow-style, more like Cyril used to do it
        block_slices = []
        duvtot = row0 = 0
    # row0 is start of current block; duv[i] is distance of row i+1 wrt row i
        for row, duv in enumerate(duv_array):
            duvtot += duv
            if duvtot > Duv:  # if more than critical, then block is [row0,row+1)
                block_slices.append(slice(row0, row+1))
                duvtot = 0
                row0 = row+1
        # add last block
        block_slices.append(slice(row0,nrows))


    # now find the minimum (fractional) channel block size for each time block. If this is <1, set to 1
    fracsizeChanBlockMin = np.array([ max(fracsizeChanBlock[slc].min(), 1) for slc in block_slices ])

    # convert that into an integer number of channel blocks for each time block
    numChanBlocks = np.ceil(NChan/fracsizeChanBlockMin)

    # convert back into integer channel size (this will be smaller than the fractional size, and will tile the
    # channel space more evenly)
    sizeChanBlock = np.int32(np.ceil(NChan/numChanBlocks))  # per each time block

    # now, we only have a small set of possible sizeChanBlock values across all time blocks, and the split into channel
    # blocks needs to be computed separately for each such channelization
    uniqueChannelBlockSizes = np.array(list(set(sizeChanBlock)))
    num_bs = uniqueChannelBlockSizes.size
    # reverse mapping: from unique block size, to channelization number
    channelization_num = dict([ (sz,i) for i,sz in enumerate(uniqueChannelBlockSizes) ])

    # now make a mapping: for each possible block size, we have a list of integer (chanblock_number,grid_number) pairs, one per channel.
    # we make chanpairs: a (Nblocksize,Nchan+1,2) array to hold these. We include +1 at the end to form up cuts (below),
    # so the last pair is always (-1,-1)
    chanrange = np.arange(0,NChan,dtype=np.int32)
    chanpairs = np.zeros( (num_bs, NChan+1, 2), np.int32)
    chanpairs[:,:-1,0] = chanrange[np.newaxis,:] / uniqueChannelBlockSizes[:,np.newaxis]
    chanpairs[:,:-1,1] = GridChanMapping[np.newaxis,:]
    chanpairs[:,-1 ,:] = -1

    # now roll the chanpairs array (along the channel axis) to see where either one of (chanblock_number,grid_number) changes
    # hence, roll channels first, then compare, then collapse "pair" axis)
    # changes will be a (NBlocksize, NChan+1) array with True at every "cut block" position, including always at 0 and NChan
    changes = (chanpairs != np.roll(chanpairs,1,axis=1)).any(axis=2)

    # list of lists: for each blocksize, lists channels where to cut for that blocksize
    changes_where = [ np.where(changes[bs,:])[0] for bs in xrange(num_bs) ]

    # list of lists: for each blocksize, lists (ch0,ch1) pairs indicating the blocks
    channel_cuts = [ [ (chwh[i], chwh[i+1]) for i in xrange(len(chwh)-1) ] for chwh in changes_where ]

    # ok, now to form up the list in grand Cyril format: for each time block, for each channel block in that
    # time block, we need to make a list of [ch0,ch1,rows]

    blocklists = [ [ch0, ch1] + list(row_index[block_slices[iblock]]) for iblock, sz in enumerate(sizeChanBlock)
                                                                      for ch0, ch1 in channel_cuts[channelization_num[sz]] ]

    # list of blocklist sizes, per block
    BlocksSizesBL = [ len(bl) for bl in blocklists ]
    # total number of blocks
    NBlocksTotBL = len(BlocksSizesBL)
    # and concatenate all blocks into a single mega-list
    BlocksRowsListBL = list(itertools.chain(*blocklists))

    return BlocksRowsListBL, BlocksSizesBL, NBlocksTotBL


def GiveBlocksRowsListBL_old(a0, a1, DATA, InfoSmearMapping, GridChanMapping):
    A0 = DATA["A0"]
    A1 = DATA["A1"]
    ind = np.where((A0 == a0) & (A1 == a1))[0]
    if(ind.size <= 1):
        return
    C = 3e8

    uvw = DATA["uvw"]
    dFreq = InfoSmearMapping["dfreqs"]   # channel width
    dPhi = InfoSmearMapping["dPhi"]      # delta-phi: max phase change allowed between rows
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
    # loop over all rows for this baseline
    for iRowBL in xrange(ind.size):
        CurrentRows.append(ind[iRowBL])   # add to list of rows
        # Frequency Block
        uv = np.sqrt(u[iRowBL]**2+v[iRowBL]**2+w[iRowBL]**2)   # uvw distance for this row
        dnu = (C/np.pi)*dPhi/(uv*l)                            # delta-nu for this row
        NChanBlock = dnu/dFreq                                 # size of averaging block, in channels
        if NChanBlock < NChanBlockMax:                         # min size of channel averaging block for this time block
            NChanBlockMax = NChanBlock

        # Time Block
        duvtot += np.sqrt(du[iRowBL]**2+dv[iRowBL]**2+dw[iRowBL]**2)   # total delta-uv distance for this row
        # find block of rows where duvtot exceeds Duv
        if (duvtot > Duv) | (iRowBL == (ind.size-1)):
            # BlocksRowsListBL.append(CurrentRows)

            NChanBlockMax = np.max([NChanBlockMax, 1])

            ch = np.arange(0, NChan, NChanBlockMax).tolist()  # list of channels at which to average

            if not((NChan) in ch): # add NChan to list (trailing block)
                ch.append((NChan))

            NChBlocks = len(ch)     # number of channel blocks (confused, this is +1 now?)
            ChBlock = np.int32(np.linspace(0, NChan, NChBlocks))   # divide [0,NChan] into NChBlocks

            # See if change in Grid ChannelMapping
            # GridChanMapping=np.array([0,0,0,1,1],np.int32)

            # associate each channel with block number
            ChBlock_rampe = np.zeros((NChan, ), np.int32)
            for iChBlock in xrange(ChBlock.size-1):
                ch0 = ChBlock[iChBlock]
                ch1 = ChBlock[iChBlock+1]
                ChBlock_rampe[ch0:ch1] = iChBlock

            CH = (-1, -1)
            # find channels where (block_number, grid_number) changes
            # ChBlock_Cut_ChanGridMapping will be a list of channels at which to "cut"
            ChBlock_Cut_ChanGridMapping = []
            for iCh in xrange(NChan):
                CH0 = ChBlock_rampe[iCh]
                CH1 = GridChanMapping[iCh]
                CH_N = (CH0, CH1)
                if CH_N != CH:
                    ChBlock_Cut_ChanGridMapping.append(iCh)
                CH = CH_N
            # add last channel to list of "cuts"
            if not((ChBlock[-1]) in ChBlock_Cut_ChanGridMapping):
                ChBlock_Cut_ChanGridMapping.append(ChBlock[-1])
            # print "%s ->
            # %s"%(str(ChBlock),str(np.array(ChBlock_Cut_ChanGridMapping,np.int32)))

            ChBlock = np.array(ChBlock_Cut_ChanGridMapping, np.int32)
            # now go over cuts, and for each interval, add [ch0,ch1] to list of blocks
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

