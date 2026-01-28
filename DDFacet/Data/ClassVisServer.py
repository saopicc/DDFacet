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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import numpy as np
import six
if six.PY3:
    import pickle as cPickle
else:
    import cPickle
import math, os, traceback
from DDFacet.Other import AsyncProcessPool

from DDFacet.Data import ClassMS
from DDFacet.Data import ClassWeightMachine
from DDFacet.Data import ClassMS, ClassDaskMS

from DDFacet.Data.ClassStokes import ClassStokes
from DDFacet.Other import ModColor
from DDFacet.Other import logger
from functools import reduce
logger.setSilent(["NpShared"])
from DDFacet.Data import ClassSmearMapping
from DDFacet.Data import ClassJones
from DDFacet.Array import shared_dict

from DDFacet.Other import reformat
import six
import DDFacet.Other.PrintList
if six.PY3:
    from DDFacet.cbuild.Gridder import _pyGridderSmearPols3x as _pyGridderSmearPols
else:
    from DDFacet.cbuild.Gridder import _pyGridderSmearPols27 as _pyGridderSmearPols
import copy
from DDFacet.Other import ClassGiveSolsFile
from astropy.io import fits
from astropy.time import Time as astropyTime
from SkyModel.Sky import ModVoronoi

from DDFacet.Other import MPIManager

log = logger.getLogger("ClassVisServer")

_cc = 299792458

SERIAL=True
SERIAL=False

def test():
    MSName = "/media/tasse/data/killMS_Pack/killMS2/Test/0000.MS"
    VS = ClassVisServer(MSName, TVisSizeMin=1e8, Weighting="Natural")
    VS.CalcWeights((1, 1, 1000, 1000), 20.*np.pi/180)
    VS.LoadNextVisChunk()


class ClassVisServer():
    def __init__(self, MSList, GD=None,
                 ColName=None,       # None if no data is read (only written)
                 TChunkSize=1,             # chunk size, in hours
                 LofarBeam=None,
                 AddNoiseJy=None,
                 DicoFields=None,
                 APP=None):
        self.GD = GD
        self.DicoFields=DicoFields
        if self.DicoFields is not None:
            self.NFields=len(self.DicoFields)

            
        self.APP=APP
        if self.APP is None:
            self.APP= AsyncProcessPool.initNew(Name="VS_DDFacet",
                                               ncpu=self.GD["Parallel"]["NCPU"],
                                               affinity=self.GD["Parallel"]["Affinity"],
                                               parent_affinity=self.GD["Parallel"]["MainProcessAffinity"],
                                               verbose=self.GD["Debug"]["APPVerbose"])
        self.APP.registerJobHandlers(self)
        self._app_id = "VS"
        
        
        self.MSList = [ MSList ] if isinstance(MSList, str) else MSList
        self.FacetMachine = None
        self.AddNoiseJy = AddNoiseJy
        self.TMemChunkSize = TChunkSize


        SubColName=None
        
        if ColName is not None and "-" in ColName:
            Spl=ColName.split("-")
            ColName,SubColName=Spl[0],Spl[1:]
        self.ColName = ColName
        self.SubColName = SubColName
        self.CountPickle = 0
        self.DicoSelectOptions = GD["Selection"]
        self.TaQL = self.DicoSelectOptions.get("TaQL", None)
        self.LofarBeam = LofarBeam
        self.ApplyBeam = False
        self.datashape = None
        self._use_data_cache = self.GD["Cache"]["VisData"]
        if self._use_data_cache == "off":
            self._use_data_cache = None
        self.DATA = None
        self._saved_data = None  # vis data saved here for single-chunk mode
        self.obs_detail = None
        self.Init()


        # smear mapping machines
        self._smm_grid = ClassSmearMapping.SmearMappingMachine("BDA.Grid",APP=self.APP)
        self._smm_degrid = ClassSmearMapping.SmearMappingMachine("BDA.Degrid",APP=self.APP)
        self._put_vis_column_job_id = self._put_vis_column_label = None
        
        self.WM=ClassWeightMachine.ClassWeightMachine(self)

    def startAPP(self):
        self.APP.startWorkers()
        self.APP.awaitWorkerStart()
        
    def stopAPP(self):
        if self.APP is None: return
        self.APP.terminate()
        self.APP.shutdown()
        del(self.APP)
        self.APP=None


    def Init(self, PointingID=0):
        self.ListMS = []
        global_freqs = set()
        NChanMax = 0
        ChanStart = self.DicoSelectOptions.get("ChanStart", 0)
        ChanEnd = self.DicoSelectOptions.get("ChanEnd", -1)
        ChanStep = self.DicoSelectOptions.get("ChanStep", 1)
        if (ChanStart, ChanEnd, ChanStep) == (0, -1, 1):
            chanslice = None
        else:
            chanslice = slice(
                ChanStart,
                ChanEnd if ChanEnd != -
                1 else None,
                ChanStep)

        min_freq_Data = 1e+999
        max_freq_Data = 0

        # max chunk shape accumulated here
        self._chunk_shape = [0, 0, 0]
        self.MSList = ClassMS.splitMSList(self.MSList)
        CMS = ClassDaskMS.ClassDaskMS if self.GD["Data"]["Dask"] else ClassMS.ClassMS 

        for iMS,msspec in enumerate(self.MSList):
            if not isinstance(msspec,str):
                msname, host, ddid, field, column = msspec
            else:
                msname, host, ddid, field, column = msspec, self.DicoSelectOptions["DDID"], self.DicoSelectOptions["Field"], self.ColName
                
            MS = CMS(
                msname,
                Col=column or self.ColName,
                SubCol=self.SubColName,
                DoReadData=False,
                AverageTimeFreq=(1, 3),
                Field=field, DDID=ddid, TaQL=self.TaQL,
                TimeChunkSize=self.TMemChunkSize, ChanSlice=chanslice,
                GD=self.GD, ResetCache=self.GD["Cache"]["Reset"],
                DicoSelectOptions = self.DicoSelectOptions,
                first_ms=self.ListMS[0] if self.ListMS else None,iMS=iMS)
            if MS.empty:
                continue
            self.ListMS.append(MS)
            # accumulate global set of frequencies, and min/max frequency
            global_freqs.update(MS.ChanFreq)
            min_freq_Data = min(min_freq_Data, (MS.ChanFreq-MS.ChanWidth/2).min())
            max_freq_Data = max(max_freq_Data, (MS.ChanFreq+MS.ChanWidth/2).max())

            # accumulate largest chunk shape
            for nrow in MS.getPerChunkRowCounts():
                shape = (nrow, len(MS.ChanFreq), MS.Ncorr)
                self._chunk_shape = [max(a, b)
                                     for a, b in zip(self._chunk_shape, shape)]

        if MPIManager.useMPI:
            freqs = MPIManager.COMM_WORLD.allgather(global_freqs)
            global_freqs = set()
            for freq in freqs:
                global_freqs.update(freq)
                
            min_freq_Data = MPIManager.COMM_WORLD.allreduce(min_freq_Data, MPIManager.MIN)
            max_freq_Data = MPIManager.COMM_WORLD.allreduce(max_freq_Data, MPIManager.MAX)

        size = reduce(lambda x, y: x * y, self._chunk_shape)
        print("shape of data/flag buffer will be %s (%.2f Gel)" % (
            self._chunk_shape, size / float(2 ** 30)), file=log)

        if not self.ListMS:
            print(ModColor.Str("--Data-MS does not specify any valid Measurement Set(s)"), file=log)
            raise RuntimeError("--Data-MS does not specify any valid Measurement Set(s)")

        self.obs_detail = self.ListMS[0].get_obs_details()

        # main cache is initialized from main cache of first MS
        # CF DEBUG : Modifying cache name to be MPI rank dependent
        if ".txt" in self.GD["Data"]["MS"]:
            # main cache is initialized from main cache of the MSList
            from DDFacet.Other.CacheManager import CacheManager
            if MPIManager.useMPI:
                self.maincache = self.cache = CacheManager("%s.rank_%d.ddfcache"%(self.GD["Data"]["MS"], MPIManager.rank), cachedir=self.GD["Cache"]["Dir"], reset=self.GD["Cache"]["Reset"])
            else:
                self.maincache = self.cache = CacheManager("%s.ddfcache"%self.GD["Data"]["MS"], cachedir=self.GD["Cache"]["Dir"], reset=self.GD["Cache"]["Reset"])

        else:
            # main cache is initialized from main cache of first MS
            self.maincache = self.cache = self.ListMS[0].maincache
        print("Main caching directory is %s"%self.maincache.dirname, file=log)



        # Assume the correlation layout of the first measurement set for now
        self.VisCorrelationLayout = self.ListMS[0].CorrelationIds
        self.StokesConverter = ClassStokes(
            self.VisCorrelationLayout,
            self.GD["RIME"]["PolMode"])
        for MS in self.ListMS:
            if not np.all(MS.CorrelationIds == self.VisCorrelationLayout):
                raise RuntimeError(
                    "Unsupported: Mixing Measurement Sets storing different correlation pairs are not supported at the moment")
                # TODO: it may be nice to have conversion code to deal with this

        self.nMS = len(self.ListMS)
        # make list of unique frequencies
        self.GlobalFreqs = np.array(sorted(global_freqs))

        
        self.RefFreq=np.mean(self.GlobalFreqs)


        # bandwidth = max_freq - min_freq
        # print("Total bandwidth is %g MHz (%g to %g MHz), with %d channels" % (
        #     bandwidth*1e-6, min_freq*1e-6, max_freq*1e-6, len(global_freqs)), file=log)
        
        max_freq_Cube, min_freq_Cube = max_freq_Data, min_freq_Data
        bandwidth_Cube = max_freq_Cube - min_freq_Cube
        log.print("Total bandwidth is %g MHz (%g to %g MHz), with %d channels" % (bandwidth_Cube*1e-6, min_freq_Data*1e-6, max_freq_Data*1e-6, len(global_freqs)))

        # print>>log,"GlobalFreqs: %d: %s"%(len(self.GlobalFreqs),repr(self.GlobalFreqs))

        # OMS: ok couldn't resist adding a bandwidth option since I need it for 3C147
        # if this is 0, then looks at NFreqBands parameter
        grid_bw = self.GD["Freq"]["BandMHz"]*1e+6
        
        if self.GD["Freq"].get("FMinMHz",None): min_freq_Cube=self.GD["Freq"]["FMinMHz"]*1e6
        if self.GD["Freq"].get("FMaxMHz",None): max_freq_Cube=self.GD["Freq"]["FMaxMHz"]*1e6
        bandwidth_Cube =  max_freq_Cube - min_freq_Cube
        
        if grid_bw:
            grid_bw = min(grid_bw, bandwidth_Cube)
            NFreqBands = self.GD["Freq"][
                "NBand"] = int(math.ceil(bandwidth_Cube/grid_bw))
        else:
            NFreqBands = np.min(
                [self.GD["Freq"]["NBand"],
                 len(self.GlobalFreqs)])  # self.nMS])
            grid_bw = bandwidth_Cube/NFreqBands

        self.NFreqBands = NFreqBands
        self.MultiFreqMode = NFreqBands > 1
        if self.MultiFreqMode:
            print(ModColor.Str(
                "MultiFrequency Mode: ON, %dx%g MHz bands" %
                (NFreqBands, grid_bw*1e-6)), file=log)

            # if not ("Alpha" in self.GD["SSDClean"]["SSDSolvePars"]):
            #     self.GD["SSDClean"]["SSDSolvePars"].append("Alpha")

        else:
            self.GD["Freq"]["NBand"] = 1
            self.GD["HMP"]["Alpha"] = [0., 0., 1.]
            if "Alpha" in self.GD["SSDClean"]["SSDSolvePars"]:
                self.GD["SSDClean"]["SSDSolvePars"].remove("Alpha")

            print(ModColor.Str("MultiFrequency Mode: OFF"), file=log)

        # Divide the global frequencies into frequency bands.
        # Somewhat of an open question how best to do it (equal bandwidth, or equal number of channels?), this is where
        # we can play various games with mapping the GlobalFreqs into a number of image bands.
        # For now, let's do it by equal bandwith as Cyril used to do:

        # these aren't used anywhere except the loop to construct the mapping below, so I'll remove them
        # FreqBands = np.linspace(self.GlobalFreqs.min(),self.GlobalFreqs.max(),NFreqBands+1)
        # self.FreqBandsMin = FreqBands[0:-1].copy()
        # self.FreqBandsMax = FreqBands[1::].copy()
        # self.FreqBandsMean = (self.FreqBandsMin + self.FreqBandsMax)/2

        # grid_band: array of ints, same size as self.GlobalFreqs, giving the
        # grid band number of each frequency channel
        grid_band = np.floor((self.GlobalFreqs - min_freq_Cube)/grid_bw).astype(int)
        # freq_to_grid_band: mapping from frequency to grid band number
        freq_to_grid_band = dict(zip(self.GlobalFreqs, grid_band))
        # print>>log,sorted(freq_to_grid_band.items())

        self.FreqBandCenters = np.linspace(min_freq_Cube+grid_bw/2, max_freq_Cube-grid_bw/2,self.NFreqBands)

        self.FreqBandChannels = []
        # freq_to_grid_band_chan: mapping from frequency to channel number
        # within its grid band
        freq_to_grid_band_chan = {}
        for iBand in range(self.NFreqBands):
            freqlist = sorted([freq for freq, band
                               in getattr(freq_to_grid_band, 
                                          "iteritems", 
                                          freq_to_grid_band.items)()
                               if band == iBand])
            self.FreqBandChannels.append(freqlist)
            freq_to_grid_band_chan.update(
                dict([(freq, chan) for chan, freq in enumerate(freqlist)]))
            print("Image band %d: %g to %g MHz contains %d MS channels from %g to %g MHz" % (iBand, (self.FreqBandCenters[iBand]-grid_bw/2)*1e-6, (
                self.FreqBandCenters[iBand]+grid_bw/2)*1e-6, len(freqlist), len(freqlist) and freqlist[0]*1e-6, len(freqlist) and freqlist[-1]*1e-6), file=log)

        self.FreqBandChannelsDegrid = {}
        self.DicoMSChanMapping = {}
        self.DicoMSChanMappingChan = {}
        self.DicoMSChanMappingDegridding = {}
        # When gridding, we make a dirty/residual image with N=NFreqBands output bands
        # When degridding, we make a model with M channels (M may depend on MS).
        # The structures initialized here map between MS channels and image channels as follows:
        # self.DicoMSChanMappingDegridding: a dict, indexed by MS number
        #       [iMS] = int array mapping MS channel numbers into model channel numbers (0...M-1)
        # self.FreqBandChannelsDegrid: a dict, indexed by MS number
        #       [iMS] = float32 array of M frequencies corresponding to M model channels for this MS
        # self.FreqBandChannels: a list, indexed by freq band number (N=NFreqBands items)
        #       [iband] = list of frequency channels that fall within that band
        # self.FreqBandCenters: a list of centre frequencies per each band (N=NFreqBands items)
        #       [iband] = centre frequency of that output band
        # self.DicoMSChanMapping: a dict, indexed by MS number
        #       [iMS] = int array mapping MS channel numbers to output band numbers
        # self.DicoMSChanMappingChan: a dict, indexed by MS number
        #       [iMS] = int array mapping MS channel numbers to channel number within the corresponding output band

        for iMS, MS in enumerate(self.ListMS):
            min_freq = (MS.ChanFreq - MS.ChanWidth/2).min()
            max_freq = (MS.ChanFreq + MS.ChanWidth/2).max()
            bw = max_freq - min_freq
            # print>>log,bw,min_freq,max_freq
            # map each channel to a gridding band
            bands = [freq_to_grid_band[freq] for freq in MS.ChanFreq]
            self.DicoMSChanMapping[iMS] = np.array(bands)
            self.DicoMSChanMappingChan[iMS] = np.array(
                [freq_to_grid_band_chan[freq] for freq in MS.ChanFreq])
            #stop
            # OMS: new option, DegridBandMHz specifies degridding band step. If
            # 0, fall back to NChanDegridPerMS
            degrid_bw = self.GD["Freq"]["DegridBandMHz"]*1e+6
            if degrid_bw:
                degrid_bw = min(degrid_bw, bw)
                degrid_bw = max(degrid_bw, MS.ChanWidth[0])
                NChanDegrid = min(
                    int(math.ceil(bw / degrid_bw)),
                    MS.ChanFreq.size)
            else:
                NChanDegrid = min(
                    self.GD["Freq"]["NDegridBand"]
                    or MS.ChanFreq.size, MS.ChanFreq.size)
                degrid_bw = bw/NChanDegrid

            # now map each channel to a degridding band
            self.DicoMSChanMappingDegridding[iMS] = np.floor(
                (MS.ChanFreq - min_freq)/degrid_bw).astype(int)

            # calculate center frequency of each degridding band
            
            # There was a problem with the above line for some MS, NDegridBand=3, DegridBandMHz=0 (due to rounding issues) was returning
            # edges with size 5, using linspace instead of arange 
            # edges = np.arange(min_freq, max_freq+degrid_bw, degrid_bw)
            edges = np.linspace(min_freq, max_freq, NChanDegrid+1) 
            
            self.FreqBandChannelsDegrid[iMS] = (edges[:-1] + edges[1:])/2
            
            print("%s   Bandwidth is %g MHz (%g to %g MHz), gridding bands are %s" % (
                MS, bw*1e-6, min_freq*1e-6, max_freq*1e-6, ", ".join(map(str, set(bands)))), file=log)

            
            
            # print("Grid band mapping: %s" % (" ".join(map(str, bands))), file=log)
            # print("Grid chan mapping: %s" % (
            #     " ".join(map(str, self.DicoMSChanMappingChan[iMS]))), file=log)
            # print("Degrid chan mapping: %s" % (
            #     " ".join(map(str, self.DicoMSChanMappingDegridding[iMS]))), file=log)
            # print("Degrid frequencies: %s" % (" ".join(
            #                                              ["%.2f" %
            #                                               (x * 1e-6)
            #                                               for x in self.FreqBandChannelsDegrid
            #                                               [iMS]])), file=log)

            print("Grid band mapping: %s" % DDFacet.Other.PrintList.ListToStr(bands), file=log)
            print("Grid chan mapping: %s" % DDFacet.Other.PrintList.ListToStr(self.DicoMSChanMappingChan[iMS]), file=log)
            print("Degrid chan mapping: %s" % DDFacet.Other.PrintList.ListToStr(self.DicoMSChanMappingDegridding[iMS]), file=log)
            s=" ".join(["%.2f" % (x * 1e-6) for x in self.FreqBandChannelsDegrid[iMS]])
            print("Degrid frequencies: %s" % DDFacet.Other.PrintList.ListToStr(s.split(" ")), file=log)


            
#            print>>log,MS

            # print>>log,"FreqBandChannelsDegrid %s"%repr(self.FreqBandChannelsDegrid[iMS])
            # print>>log,"self.DicoMSChanMappingDegriding %s"%repr(self.DicoMSChanMappingDegridding[iMS])
            # print>>log,"self.DicoMSChanMapping %s"%repr(self.DicoMSChanMapping[iMS])

        # print>>log,"FreqBandChannels %s"%repr(self.FreqBandChannels)

#        self.RefFreq=np.mean(self.ListFreqs)
        self.RefFreq = np.mean(self.GlobalFreqs)
        
        
        self.nTotalChunks = sum([ms.numChunks() for ms in self.ListMS])
        self.ReInitChunkCount()

        # TimesVisMin=np.arange(0,MS.DTh*60.,self.TVisSizeMin).tolist()
        #if not(MS.DTh*60. in TimesVisMin): TimesVisMin.append(MS.DTh*60.)
        # self.TimesVisMin=np.array(TimesVisMin)

    def SetImagingPars(self, OutImShape, CellSizeRad):
        self.OutImShape = OutImShape
        self.CellSizeRad_x,self.CellSizeRad_y=self.CellSizeRad = CellSizeRad

    def CalcMeanBeam(self):
        AverageBeamMachine = ClassBeamMean.ClassBeamMean(self)
        AverageBeamMachine.LoadData()
        AverageBeamMachine.CalcMeanBeam()

    def ReInitChunkCount(self):
        if self.nTotalChunks > 1 and self.DATA is not None:
            self.DATA.delete()
            self.DATA = None
        self.iCurrentMS = 0
        self.iCurrentChunk = -1


    def startVisPutColumnInBackground(self, DATA, field, column, likecol="DATA"):
        iMS, iChunk = DATA["iMS"], DATA["iChunk"]
        self._put_vis_column_label = "%d.%d" % (iMS+1, iChunk+1)
        self._put_vis_column_job_id = "PutData:%d:%d" % (iMS, iChunk)
        self.APP.runJob(self._put_vis_column_job_id, self.visPutColumnHandler, args=(DATA.readonly(), field, column, likecol), io=0,serial=SERIAL)

    def visPutColumnHandler (self, DATA, field, column, likecol):
        iMS, iChunk = DATA["iMS"], DATA["iChunk"]
        ms = self.ListMS[iMS]
        if ms.ToRADEC is not None:
            ms.Rotate(DATA,RotateType=["vis"],Sense="ToPhaseCenter",DataFieldName=field)
            

        #ms.PutVisColumn(column, DATA[field], row0, row1, likecol=likecol, sort_index=DATA["sort_index"],
        #                flags=DATA["flags"])
        ms.PutVisColumn(column, DATA[field], iChunk, likecol=likecol, sort_index=DATA["sort_index"])

    def collectPutColumnResults(self):
        if self._put_vis_column_job_id:
            self.APP.awaitJobResults(self._put_vis_column_job_id, progress="Writing %s" % self._put_vis_column_label)
            self._put_vis_column_job_id = None
            

    def startChunkLoadInBackground(self, last_cycle=False):
        """
        Called in main process. Increments chunk counter, initiates chunk load in background thread.
        Returns None if we get past the last chunk, else returns the chunk label.
        """
        while True:
            # advance chunk pointer
            self.iCurrentChunk += 1
            ms = self.ListMS[self.iCurrentMS]
            # go to next MS?
            if self.iCurrentChunk >= ms.numChunks():
                self.iCurrentMS += 1
                # no more MSs -- return None
                if self.iCurrentMS >= len(self.ListMS):
                    self._next_chunk_name = None
                    self.iCurrentMS = 0
                    self.iCurrentChunk = -1
                    return None
                # go back up to first chunk of next MS
                self.iCurrentChunk = -1
                continue
            self._next_chunk_name = "DATA:%d:%d" % (self.iCurrentMS, self.iCurrentChunk)
            self._next_chunk_label = "%d.%d" % (self.iCurrentMS + 1, self.iCurrentChunk + 1)
            # null chunk? skip to next chunk, unless we're in the last major cycle
            if not self.WM._ignore_vis_weights and not last_cycle:
                self.WM.awaitWeights()
                if self.WM.VisWeights[self.iCurrentMS][self.iCurrentChunk]["null"]:
                    print(ModColor.Str("chunk %s is null, skipping"%self._next_chunk_label), file=log)
                    continue
            # ok, now we're good to load
            print("scheduling loading of chunk %s" % self._next_chunk_label, file=log)
            # in single-chunk mode, DATA may already be loaded, in which case we do nothing
            if self.nTotalChunks > 1 or self.DATA is None:
                # tell the IO thread to start loading the chunk
                self.APP.runJob(self._next_chunk_name, self._handler_LoadVisChunk,
                           args=(self._next_chunk_name, self.iCurrentMS, self.iCurrentChunk), 
                           io=0,serial=SERIAL)
            return self._next_chunk_label

    def collectLoadedChunk(self, start_next=True, last_cycle=False):
        # previous data dict can now be discarded from shm
        if self.nTotalChunks > 1 and self.DATA is not None:
            print("Delete shared dict %s"%self.DATA.path, file=log)
            self.DATA.delete()
            self.DATA = None
        # if no next chunk scheduled, we're at end
        if not self._next_chunk_name:
            return "EndOfObservation"
        # in single-chunk mode, only read the MS once, then keep it forever,
        # but re-copy visibility data from original data
        if self.nTotalChunks == 1 and self.DATA is not None and "data" in self.DATA:
            np.copyto(self.DATA["data"], self._saved_data)
        else:
            # await completion of data loading jobs (which, presumably, includes smear mapping)
            self.APP.awaitJobResults(self._next_chunk_name, timing="Reading %s"%self._next_chunk_label )
            # reload the data dict -- background thread will now have populated it
            self.DATA = shared_dict.attach(self._next_chunk_name)
            self.DATA["label"] = self._next_chunk_label
            # in single-chunk mode, keep a copy of the data array
            if self.nTotalChunks == 1 and "data" in self.DATA and self._saved_data is None:
                self._saved_data = self.DATA["data"].copy()
        # schedule next event
        if start_next:
            self.startChunkLoadInBackground(last_cycle=last_cycle)
        # return the data dict
        return self.DATA

    def releaseLoadedChunk(self):
        """Releases memory associated with any saved data"""
        self._saved_data = None
        if self.DATA is not None:
            self.DATA.delete()
            self.DATA = None


    def _handler_LoadVisChunk(self, dictname, iMS, iChunk):
        """
        Called in IO thread to load a data chunk
        Args:
            null_data: if True, then we don't want to read the visibility data at all, but rather just want to make
                a null buffer of the same shape as the visibility data.
        """
        DATA = shared_dict.create(dictname)
        DATA["iMS"]    = iMS
        DATA["iChunk"] = iChunk
        ms = self.ListMS[iMS]

        print(ModColor.Str("loading ms %d of %d, chunk %d of %d" % (iMS+1, self.nMS, iChunk+1, ms.numChunks()), col="green"), file=log)

        ms.GiveChunk(DATA, iChunk, use_cache=self._use_data_cache,
                     read_data=bool(self.ColName), sort_by_baseline=self.GD["Data"]["Sort"])
        # update cache to match MSs current chunk cache
        self.cache = ms.cache


        times = DATA["times"]
        data = DATA.get("data")
        A0 = DATA["A0"]
        A1 = DATA["A1"]

        freqs = ms.ChanFreq.flatten()
        nbl = ms.nbl

        DecorrMode = self.GD["RIME"]["DecorrMode"]

        if 'F' in DecorrMode or "T" in DecorrMode:
            DATA["lm_PhaseCenter"] = ms.lm_PhaseCenter

        DATA["ChanMapping"] = self.DicoMSChanMapping[iMS]
        DATA["ChanMappingDegrid"] = self.DicoMSChanMappingDegridding[iMS]
        DATA["FreqMappingDegrid"] = self.FreqBandChannelsDegrid[iMS]

        print("  channel Mapping Gridding  : %s" % DDFacet.Other.PrintList.ListToStr(DATA["ChanMapping"]), file=log)
        print("  channel Mapping DeGridding: %s" % DDFacet.Other.PrintList.ListToStr(DATA["ChanMappingDegrid"]), file=log)

        if freqs.size > 1:
            DATA["freqs"] = np.float64(freqs)
        else:
            DATA["freqs"] = np.array([freqs[0]], dtype=np.float64)
        DATA["dfreqs"] = ms.dFreq

        DATA["nbl"] = nbl
        DATA["na"] = ms.na
        DATA["ROW0"] = ms.ROW0
        DATA["ROW1"] = ms.ROW1

        # get weights
        
        weights,sgnweights = self.WM.GetVisWeights(iMS, iChunk)
        DATA["Weights"] = weights

        if sgnweights is not None and -1 in sgnweights:
            if not np.allclose(self.VisCorrelationLayout,np.array([ 9, 10, 11, 12], dtype=np.int32)): stop
            sort_index=DATA["sort_index"]
            nrow,nch,npol= DATA["data"].shape
            d=DATA["data"].reshape((nrow*nch,npol))
            sgn=sgnweights[sort_index].reshape((nrow*nch,1))
            ind=np.where(sgn==-1)[0]
            #print(DATA["data"].reshape((nrow*nch,npol))[ind[0]])

            log.print("Taking the negative of %i visibilities..."%ind.size)
            d0=d.copy()
            XY=d[ind,1]
            YX=d[ind,2]
            XY1=YX.copy() # (-XY+YX)/2
            YX1=XY.copy() # (XY-YX)/2
            d[ind,1]=XY1[:]
            d[ind,2]=YX1[:]
            
            # # DATA["data"][...]=DATA["data"][...]*sgnweights.reshape((nrow,nch,1))
            # M=(1./np.sqrt(2))*np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,-1j,1j,0]],np.complex128)
            # nn=ind[0]
            # V0=d0[nn].reshape((2,2))
            # V1=d[nn].reshape((2,2))
            # # print("V0",np.dot(M,V0.reshape((-1,1))))
            # # print("V1",np.dot(M,V1.reshape((-1,1))))
            # # print(DATA["data"].reshape((nrow*nch,npol))[ind[0]])

            
            log.print(" ... done...")
        # DATA["data"][...]=-DATA["data"][...]

        if self.GD["Weight"]["OutColName"] and self.GD["Output"]["Mode"]!="Predict":
            # When the MS doesn't have an IMAGING_WEIGHT column
            ColDesc={'valueType': 'float',
                     'dataManagerType': 'StandardStMan',
                     'dataManagerGroup': 'SSMVar',
                     'option': 4,
                     'maxlen': 0,
                     'comment': '',
                     'ndim': 1,
                     'shape': np.array([DATA["freqs"].size]),
                     '_c_order': True,
                     'keywords': {}}
            
            ms.PutVisColumn(self.GD["Weight"]["OutColName"],
                            DATA["Weights"],
                            iChunk,
                            likecol="IMAGING_WEIGHT",
                            ColDesc=ColDesc,
                            sort_index=DATA["sort_index"])

        
        if weights is None:
            print(ModColor.Str("This chunk is all flagged or has zero weight."), file=log)
            return
        
        if DATA["sort_index"] is not None: # and DATA["Weights"] is not 1: # OMS 2023/12 they're not "1" ever and this seems a bug
            DATA["Weights"] = DATA["Weights"][DATA["sort_index"]]

        self.computeBDAInBackground(dictname, ms, DATA,
            ChanMappingGridding=DATA["ChanMapping"],
                                    ChanMappingDeGridding=DATA["ChanMappingDegrid"],iField=0)

        if self.DicoFields is not None:
            # for iField in range(self.NFields):
            #     JonesMachine = ClassJones.ClassJones(self.GD, ms, self.FacetMachine,
            #                                          iField=iField)
            #     JonesMachine.InitDDESols(DATA)
            JonesMachine = ClassJones.ClassJones(self.GD, ms, self.FacetMachine,
                                                 iField=None)
            JonesMachine.InitDDESols(DATA)
        else:
            JonesMachine = ClassJones.ClassJones(self.GD, ms, self.FacetMachine)
            JonesMachine.InitDDESols(DATA)
                

        if data is not None and self.AddNoiseJy is not None:
            data += (self.AddNoiseJy/np.sqrt(2.))*(np.random.randn(*data.shape)+1j*np.random.randn(*data.shape))

        # load results of smear mapping computation
        self.collectBDA(dictname, DATA)

    def setFacetMachine(self, FacetMachine):
        self.FacetMachine = FacetMachine
        self.WM.FacetMachine=FacetMachine
        # self.FullImShape = self.FacetMachine.OutImShape
        # self.PaddedFacetShape = self.FacetMachine.PaddedGridShape
        # self.FacetShape = self.FacetMachine.FacetShape
        # self.CellSizeRad_x,self.CellSizeRad_y=self.CellSizeRad = self.FacetMachine.CellSizeRad

    def setFOV(self, sh0, sh1, sh2, cell):
        self.FullImShape = sh0
        self.PaddedFacetShape = sh1
        self.FacetShape = sh2
        self.CellSizeRad_x,self.CellSizeRad_y = self.CellSizeRad = cell

    def collectBDA(self, base_job_id, DATA):
        """Called in I/O thread. Waits for BDA computation to complete (if any), then populates dict"""
        if "BDA.Grid" not in DATA:
            FinalMapping, fact = self._smm_grid.collectSmearMapping(DATA, "BDA.Grid")
            print(ModColor.Str("  Effective compression [grid]  :   %.2f%%" % fact, col="green"), file=log)
            np.save(open(self._bda_grid_cachename, 'wb'), FinalMapping)
            self.cache.saveCache("BDA.Grid")
        if "BDA.Degrid" not in DATA:
            FinalMapping, fact = self._smm_degrid.collectSmearMapping(DATA, "BDA.Degrid")
            print(ModColor.Str("  Effective compression [degrid]:   %.2f%%" % fact, col="green"), file=log)
            DATA["BDA.Degrid"] = FinalMapping
            np.save(open(self._bda_degrid_cachename, 'wb'), FinalMapping)
            self.cache.saveCache("BDA.Degrid")

    def computeBDAInBackground(self, base_job_id, ms, DATA, ChanMappingGridding=None, ChanMappingDeGridding=None, iField=None):

        GD=copy.deepcopy(self.GD)
        CriticalCacheParms=dict(Data=GD["Data"],
                                Compression=GD["Comp"],
                                Freq=GD["Freq"],
                                DataSelection=GD["Selection"],
                                Sorting=GD["Data"]["Sort"])
        del CriticalCacheParms["Data"]["ColName"],CriticalCacheParms["DataSelection"]["FlagAnts"]
        if iField is not None and self.DicoFields is not None:
            FacetMachine=self.FacetMachine.LFM[iField]
        else:
            FacetMachine=self.FacetMachine
        FullImShape = FacetMachine.OutImShape
        # self.PaddedFacetShape = self.FacetMachine.PaddedGridShape
        FacetShape = FacetMachine.FacetShape
        CellSizeRad_x,CellSizeRad_y=FacetMachine.CellSizeRad
            
        
        if True: # always True for now, non-BDA gridder is not maintained # if self.GD["Comp"]["CompGridMode"]:
            self._bda_grid_cachename, valid = self.cache.checkCache("BDA.Grid",CriticalCacheParms)
            if valid:
                print("  using cached BDA mapping %s" % self._bda_grid_cachename, file=log)
                DATA["BDA.Grid"] = np.load(self._bda_grid_cachename)
            else:
                if self.GD["Comp"]["GridFoV"] == "Facet":
                    _, _, nx, ny = FacetShape
                elif self.GD["Comp"]["GridFoV"] == "Full":
                    _, _, nx, ny = FullImShape
                mode = self.GD["Comp"]["BDAMode"]
                FOV =  np.sqrt((CellSizeRad_x*nx/2)**2+(CellSizeRad_y*ny/2)**2) * 180. / np.pi
                self._smm_grid.computeSmearMappingInBackground(base_job_id, ms, DATA, FOV,
                                                          (1. - self.GD["Comp"]["GridDecorr"]),
                                                          ChanMappingGridding, mode)

        if True: # always True for now, non-BDA gridder is not maintained # if self.GD["Comp"]["CompDeGridMode"]:
            self._bda_degrid_cachename, valid = self.cache.checkCache("BDA.Degrid",CriticalCacheParms)

            if valid:
                print("  using cached BDA mapping %s" % self._bda_degrid_cachename, file=log)
                DATA["BDA.Degrid"] = np.load(self._bda_degrid_cachename)
            else:
                if self.GD["Comp"]["DegridFoV"] == "Facet":
                    _, _, nx, ny = FacetShape
                elif self.GD["Comp"]["DegridFoV"] == "Full":
                    _, _, nx, ny = FullImShape
                mode = self.GD["Comp"]["BDAMode"]
                FOV =  np.sqrt((CellSizeRad_x*nx/2)**2+(CellSizeRad_y*ny/2)**2) * 180. / np.pi
                self._smm_degrid.computeSmearMappingInBackground(base_job_id, ms, DATA, FOV,
                                                          (1. - self.GD["Comp"]["DegridDecorr"]),
                                                          ChanMappingDeGridding, mode)


    def CalcWeightsBackground(self,iField=None):
        self.WM.CalcWeightsBackground(iField=iField)

    def getMaxUV(self):
        return self.WM.getMaxUV()
        
    def getMaxW(self):
        return self.WM.getMaxW()
