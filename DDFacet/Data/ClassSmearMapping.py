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

import numpy as np
import math
import itertools
from DDFacet.Other import MyLogger
from DDFacet.Array import NpShared
log = MyLogger.getLogger("ClassSmearMapping")

from DDFacet.Other import Multiprocessing, ClassTimeIt
from DDFacet.Array import shared_dict
from DDFacet.Other.AsyncProcessPool import APP

bda_dicts = {}

class SmearMappingMachine (object):
    def __init__ (self, name=None, mode=2):
        self.name = name or "SMM.%x"%id(self)
        APP.registerJobHandlers(self)
        self._job_counter = APP.createJobCounter(self.name)
        self._data = self._blockdict = self._sizedict = None

    def _smearmapping_worker(self, DATA, blockdict, sizedict, a0, a1, dPhi, l, channel_mapping, mode):
        t = ClassTimeIt.ClassTimeIt()
        t.disable()
        if mode == 1:
            BlocksRowsListBL, BlocksSizesBL, _ = GiveBlocksRowsListBL_old(a0, a1, DATA, dPhi, l, channel_mapping)
        elif mode == 2:
            BlocksRowsListBL, BlocksSizesBL, _ = GiveBlocksRowsListBL(a0, a1, DATA, dPhi, l, channel_mapping)
        else:
            raise ValueError("unknown BDAMode setting %d"%mode)

        t.timeit('compute')
        if BlocksRowsListBL is not None:
            key = "%d:%d" % (a0,a1)
            sizedict[key]  = np.array(BlocksSizesBL)
            blockdict[key] = np.array(BlocksRowsListBL)
            t.timeit('store')

    def computeSmearMappingInBackground (self, base_job_id, MS, DATA, radiusDeg, Decorr, channel_mapping, mode):
        l = radiusDeg * np.pi / 180
        dPhi = np.sqrt(6. * (1. - Decorr))
        # create new empty shared dicts for results
        self._outdict = shared_dict.create("%s:%s:tmp" %(DATA.path, self.name))
        blockdict = self._outdict.addSubdict("blocks")
        sizedict  = self._outdict.addSubdict("sizes")
        self._nbl = 0
        for a0 in xrange(MS.na):
            for a1 in xrange(MS.na):
                if a0 != a1:
                    self._nbl += 1
                    APP.runJob("%s:%s:%d:%d" % (base_job_id, self.name, a0, a1), self._smearmapping_worker,
                               counter=self._job_counter, collect_result=False,
                               args=(DATA.readonly(), blockdict.writeonly(), sizedict.writeonly(), a0, a1, dPhi, l,
                                     channel_mapping, mode))



    def collectSmearMapping (self, DATA, field):
        APP.awaitJobCounter(self._job_counter, progress="Mapping %s"%self.name, total=self._nbl, timeout=1)
        self._outdict.reload()
        blockdict = self._outdict["blocks"]
        sizedict  = self._outdict["sizes"]
        # process worker results
        # for each map (each array returned from worker), BlockSizes[MapName] will
        # contain a list of BlocksSizesBL entries returned from that worker
        NTotBlocks = 0
        NTotRows = 0

        for key in sizedict.iterkeys():
            bsz = sizedict[key]
            NTotBlocks += len(bsz)
            NTotRows += bsz.sum()

        mapping = DATA.addSharedArray(field, (2 + NTotBlocks + NTotRows,), np.int32)

        mapping[0] = NTotBlocks
        mapping[1] = NTotBlocks>>32

        FinalMappingSizes = mapping[2:2+NTotBlocks]
        FinalMapping = mapping[2+NTotBlocks:]


        iii = 0
        jjj = 0

        # now go through each per-baseline mapping, sorted by baseline
        for key in sizedict.iterkeys():
            BlocksSizesBL = sizedict[key]
            BlocksRowsListBL = blockdict[key]

            FinalMapping[iii:iii+BlocksRowsListBL.size] = BlocksRowsListBL[:]
            iii += BlocksRowsListBL.size

            # print "IdWorker,AppendId",IdWorker,AppendId,BlocksSizesBL
            # MM=np.concatenate((MM,BlocksSizesBL))
            FinalMappingSizes[jjj:jjj+BlocksSizesBL.size] = BlocksSizesBL[:]
            jjj += BlocksSizesBL.size

        NVis = np.where(DATA["A0"] != DATA["A1"])[0].size * DATA["freqs"].size
        #print>>log, "  Number of blocks:         %i"%NTotBlocks
        #print>>log, "  Number of 4-Visibilities: %i"%NVis
        fact = (100.*(NVis-NTotBlocks)/float(NVis))

        # clear temp shared arrays/dicts
        del sizedict
        del blockdict
        self._outdict.delete()

        return mapping, fact






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
        print>> log, "Build decorrelation mapping ..."

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

        joblist = [(a0, a1) for a0 in xrange(na) for a1 in xrange(na)]

        # process worker results
        # for each map (each array resturned from worker), BlockSizes[MapName] will
        # contain a list of BlocksSizesBL entries returned from that worker
        BlockListsSizes = {}
        NTotBlocks = 0
        NTotRows = 0

        for (a0, a1) in joblist:
            if a0==a1: continue
            rep = GiveBlocksRowsListBL(a0, a1, DATA, InfoSmearMapping, GridChanMapping)
            if rep:
                BlocksRowsListBL, BlocksSizesBL, NBlocksTotBL = rep
                BlockListsSizes[a0,a1] = BlocksRowsListBL, BlocksSizesBL
                NTotBlocks += NBlocksTotBL
                NTotRows += np.sum(BlocksSizesBL)

        FinalMappingHeader = np.zeros((2 * NTotBlocks + 1,), np.int32)
        FinalMappingHeader[0] = NTotBlocks

        iStart = 1
        # MM=np.array([],np.int32)
        MM = np.zeros((NTotBlocks,), np.int32)

        FinalMapping = np.zeros((NTotRows,), np.int32)
        iii = 0
        jjj = 0

        # now go through each per-worker mapping
        for baseline, (BlocksRowsListBL, BlocksSizesBL) in BlockListsSizes.iteritems():
            # FinalMapping=np.concatenate((FinalMapping,BlocksRowsListBLWorker))

            FinalMapping[iii:iii + len(BlocksRowsListBL)] = BlocksRowsListBL
            iii += len(BlocksRowsListBL)

            N = 0

            # print "IdWorker,AppendId",IdWorker,AppendId,BlocksSizesBL
            # MM=np.concatenate((MM,BlocksSizesBL))
            MM[jjj:jjj + len(BlocksSizesBL)] = BlocksSizesBL
            jjj += len(BlocksSizesBL)
            # print MM.shape,BlocksSizesBL
            N += np.sum(BlocksSizesBL)
            # print N,BlocksRowsListBLWorker.size

        cumul = np.cumsum(MM)
        FinalMappingHeader[1:1 + NTotBlocks] = MM
        FinalMappingHeader[NTotBlocks + 1 + 1::] = (cumul)[0:-1]
        FinalMappingHeader[NTotBlocks + 1::] += 2 * NTotBlocks + 1

        # print>>log, "  Concat header"
        FinalMapping = np.concatenate((FinalMappingHeader, FinalMapping))

        # print>>log, "  Put in shared mem"

        NVis = np.where(DATA["A0"] != DATA["A1"])[0].size * NChan
        # print>>log, "  Number of blocks:         %i"%NTotBlocks
        # print>>log, "  Number of 4-Visibilities: %i"%NVis
        fact = (100. * (NVis - NTotBlocks) / float(NVis))

        # self.UnPackMapping()
        # print FinalMapping

        return FinalMapping, fact


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

        joblist = [ (a0, a1) for a0 in xrange(na) for a1 in xrange(na) if a0 != a1 ]


        WorkerMapName = Multiprocessing.getShmURL("SmearWorker.%d")

        results = Multiprocessing.runjobs(joblist, title="Smear mapping", target=_smearmapping_worker,
                                            kwargs=dict(DATA=DATA,
                                                        InfoSmearMapping=InfoSmearMapping,
                                                        WorkerMapName=WorkerMapName,
                                                        GridChanMapping=GridChanMapping))

        # process worker results
        # for each map (each array returned from worker), BlockSizes[MapName] will
        # contain a list of BlocksSizesBL entries returned from that worker
        RowsBlockSizes = {}
        NTotBlocks = 0
        NTotRows = 0
        worker_maps = {}

        for DicoResult in results:
            if not DicoResult["Empty"]:
                MapName = DicoResult["MapName"]
                map = worker_maps.get(MapName)
                if map is None:
                    map = worker_maps[MapName] = NpShared.GiveArray(MapName)
                bl = DicoResult["bl"]
                rowslice = DicoResult["Slice"]
                bsz = np.array(DicoResult["BlocksSizesBL"])
                RowsBlockSizes[bl] = map[rowslice], bsz
                NTotBlocks += DicoResult["NBlocksTotBL"]
                NTotRows += bsz.sum()

        # output mapping has 2 words for the total size, plus 2*NTotBlocks header, plus NTotRows blocklists
        OutputMapping = np.zeros((2 + 2*NTotBlocks + NTotRows, ), np.int32)

        # just in case NTotBlocks is over 2^31...
        # (don't want to use np.int32 for the whole mapping as that just wastes space, we may assume
        # that we have substantially fewer rows, so int32 is perfectly good as a row index etc.)
        OutputMapping[0] = NTotBlocks
        OutputMapping[1] = NTotBlocks >> 32

        BlockListSizes = OutputMapping[2:2+NTotBlocks]

        BlockLists = OutputMapping[2+NTotBlocks:]
        iii = 0
        jjj = 0

        # now go through each per-baseline mapping, sorted by baseline
        for _, (BlocksRowsListBL, BlocksSizesBL) in sorted(RowsBlockSizes.items()):
            #print>>log, "  Worker: %i"%(IdWorker)

            BlockLists[iii:iii+BlocksRowsListBL.size] = BlocksRowsListBL[:]
            iii += BlocksRowsListBL.size

            # print "IdWorker,AppendId",IdWorker,AppendId,BlocksSizesBL
            # MM=np.concatenate((MM,BlocksSizesBL))
            BlockListSizes[jjj:jjj+BlocksSizesBL.size] = BlocksSizesBL[:]
            jjj += BlocksSizesBL.size

        for MapName in worker_maps.iterkeys():
            NpShared.DelArray(MapName)

        #print>>log, "  Put in shared mem"

        NVis = np.where(DATA["A0"] != DATA["A1"])[0].size * NChan
        #print>>log, "  Number of blocks:         %i"%NTotBlocks
        #print>>log, "  Number of 4-Visibilities: %i"%NVis
        fact = (100.*(NVis-NTotBlocks)/float(NVis))

        # self.UnPackMapping()
        # print FinalMapping

        return OutputMapping, fact


def GiveBlocksRowsListBL(a0, a1, DATA, dPhi, l, GridChanMapping):

    A0 = DATA["A0"]
    A1 = DATA["A1"]
    row_index = np.where((A0 == a0) & (A1 == a1))[0]
    nrows = row_index.size
    if not nrows:
        return None, None, None
    C = 3e8

    uvw = DATA["uvw"][row_index]
    dFreq = DATA["dfreqs"]   # channel width
    freqs = DATA["freqs"]
    NChan = freqs.size
    nu0 = np.max(freqs)

    # critical delta-uv interval for smearing
    Duv = C*(dPhi)/(np.pi*nu0)

    ### old code -- computed delta-uv or delta-uvw and compared this to Duv
    # # find delta-uv at each row: will have size of (Nrow,2)
    # UVSLICE = slice(0,3)  # 0,2 for delta-uv-distance, or 0,3 for delta-uvw distance
    # duv = uvw[:,UVSLICE].copy()
    # duv[:-1,UVSLICE] = uvw[1:,UVSLICE] - uvw[:-1,UVSLICE]
    # # last delta copied from previous row
    # duv[-1,:] = duv[-2,:]
    # # convert to delta-phase: Cyril's old approximation of u*l+v*l+(w*l)
    # delta_phase = np.sqrt((duv**2).sum(1))*(l*np.pi*nu0/C)     # delta-phase at facet edge (approximate)

    ### better idea maybe: multiply uvw by l,m,n-1 at facet edge
    ### (Cyril's old code just took delta-uvw times l, which probably overweighted w!)

    # take l,m,n-1 at facet edge, compute ul,vm,w(n-1) vector
    lmn = np.array([l, l, math.sqrt(1-2*l*l)-1])
    uvwlmn = uvw * lmn[np.newaxis,:]

    duvw = uvwlmn.copy()
    duvw[:-1,:] = uvwlmn[1:,:] - uvwlmn[:-1,:]
    # last delta copied from previous row
    duvw[-1,:] = duvw[-2,:]
    # max delta phase is just the length of the delta-vector
    delta_phase = np.sqrt((duvw**2).sum(1))*(np.pi*nu0/C)

    ## same here: instead of uvw distance, multiply uvw by lmn on facet edge
    # # uvw-distance at each row (size Nrow-1)
    # uv = np.sqrt((uvw**2).sum(1))
    # dnu = (C / np.pi) * dPhi / (uv * l)  # delta-nu for each row
    uv = np.sqrt((uvwlmn**2).sum(1))
    dnu = (C / np.pi) * dPhi / uv  # delta-nu for each row
    fracsizeChanBlock = dnu / dFreq  # max size of averaging block, in fractional channels, for each row

    if True:  # fast-style, maybe not as precise
        # accumulate delta-phase, and divide by dPhi. Take the floor of that -- that gives us an integer, the time block number
        # for each row
        if Duv:
            rowblock = np.zeros(nrows+1)
            rowblock[:nrows] = np.int32(delta_phase.cumsum() / dPhi)
            rowblock[nrows] = -1
            # rowblock is now an nrows+1 vector of block numbers per each row, with an -1 at the end, e.g. [ 1 1 1 2 2 2 3 -1]
            # np.roll(rowblock,1) "rolls" this vector to the right, resulting in:                          [-1 1 1 1 2 2 2  3]
            # now, every position in the array where roll!=rowblock is a starting position for a block.
            # conveniently (and by construction, thanks to the -1 at the end), this always includes the 0 and the nrows position.
            blockcut = np.where(np.roll(rowblock,1) != rowblock)[0]
            nblocks = len(blockcut)
            # make list of rows slices for each time block. The where() statement above gives us a list of rows at
            # which the block index has changed, i.e. [0 N1 N2 Nrows]. Convert this into slice objects representing
            # [0,N1), [N1,N2), ..., [Nx,Nrows)
            block_slices = [ slice(blockcut[i],blockcut[i+1]) for i in xrange(nblocks-1) ]
        else:
            block_slices = [ slice(i,i+1) for i in xrange(nrows) ]
    else:   # slow-style, more like Cyril used to do it
        block_slices = []
        dphtot = row0 = 0
        # row0 is start of current block; delta_phase[i] is distance of row i+1 wrt row i
        for row, dph in enumerate(delta_phase):
            dphtot += dph
            if dphtot > dPhi:  # if more than critical, then block is [row0,row+1)
                block_slices.append(slice(row0, row+1))
                dphtot = 0
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

#    print>> log, "baseline %d:%d blocklists %s" % (a0, a1, " ".join([",".join(map(str,bl)) for bl in blocklists]))

    # list of blocklist sizes, per block
    BlocksSizesBL = [ len(bl) for bl in blocklists ]
    # total number of blocks
    NBlocksTotBL = len(BlocksSizesBL)
    # and concatenate all blocks into a single mega-list
    BlocksRowsListBL = list(itertools.chain(*blocklists))

    return BlocksRowsListBL, BlocksSizesBL, NBlocksTotBL

#BlocksRowsListBL, BlocksSizesBL, _ = GiveBlocksRowsListBL(a0, a1, DATA, dPhi, l, channel_mapping)

#def GiveBlocksRowsListBL_old(a0, a1, DATA, InfoSmearMapping, GridChanMapping):
def GiveBlocksRowsListBL_old(a0, a1, DATA, dPhi, l, channel_mapping):
    A0 = DATA["A0"]
    A1 = DATA["A1"]
    ind = np.where((A0 == a0) & (A1 == a1))[0]
    #if(ind.size <= 1):
    #    return
    nrows = ind.size
    if not nrows:
        return None, None, None
    C = 3e8

    GridChanMapping=channel_mapping
    uvw = DATA["uvw"]
    dFreq = DATA["dfreqs"]#InfoSmearMapping["dfreqs"]   # channel width
    #dPhi = InfoSmearMapping["dPhi"]      # delta-phi: max phase change allowed between rows
    #dFreq = InfoSmearMapping["dfreqs"]
    #dPhi = InfoSmearMapping["dPhi"]
    #l = InfoSmearMapping["l"]
    freqs = DATA["freqs"]#InfoSmearMapping["freqs"]
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
    blocklist = []
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
                blocklist.append(ThiDesc)
                BlocksRowsListBL += (ThiDesc)
                NBlocksTotBL += 1
            # import pdb; pdb.set_trace()
            NChanBlockMax = 1e3
            CurrentRows = []
            duvtot = 0


    #print BlocksRowsListBL, BlocksSizesBL, NBlocksTotBL
    return BlocksRowsListBL, BlocksSizesBL, NBlocksTotBL

    # stop
#    print>> log, "baseline %d:%d blocklists %s" % (a0, a1, " ".join([",".join(map(str,bl)) for bl in blocklist]))



