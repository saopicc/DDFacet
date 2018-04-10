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
import math, os, cPickle, traceback


import ClassMS
from DDFacet.Data.ClassStokes import ClassStokes
from DDFacet.Other import ModColor
from DDFacet.Other import MyLogger
from functools import reduce
MyLogger.setSilent(["NpShared"])
import ClassSmearMapping
import ClassJones
from DDFacet.Array import shared_dict
from DDFacet.Other.AsyncProcessPool import APP
import DDFacet.cbuild.Gridder._pyGridderSmearPols as _pyGridderSmearPols
import copy

log = MyLogger.getLogger("ClassVisServer")

_cc = 299792458


def test():
    MSName = "/media/tasse/data/killMS_Pack/killMS2/Test/0000.MS"
    VS = ClassVisServer(MSName, TVisSizeMin=1e8, Weighting="Natural")
    VS.CalcWeights((1, 1, 1000, 1000), 20.*np.pi/180)
    VS.LoadNextVisChunk()


class ClassVisServer():

    def __init__(self, MSList, GD=None,
                 ColName="DATA",           # if None, no data column is read
                 TChunkSize=1,             # chunk size, in hours
                 LofarBeam=None,
                 AddNoiseJy=None):
        self.GD = GD
        if APP is not None:
            APP.registerJobHandlers(self)
            self._weightjob_counter = APP.createJobCounter("VisWeights")
            self._calcweights_event = APP.createEvent("VisWeights")
            self._app_id = "VS"

        self.MSList = [ MSList ] if isinstance(MSList, str) else MSList
        self.FacetMachine = None
        self.AddNoiseJy = AddNoiseJy
        self.TMemChunkSize = TChunkSize

        self.Weighting = GD["Weight"]["Mode"].lower()

        if self.Weighting not in ("natural", "uniform", "briggs", "robust"):
            raise ValueError("unknown Weighting=%s" % self.Weighting)
        self.MFSWeighting = GD["Weight"]["MFS"]
        self.Robust = GD["Weight"]["Robust"]
        self.Super = GD["Weight"]["SuperUniform"]
        self.VisWeights = None

        self.CountPickle = 0
        self.ColName = ColName
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

        # if True, then skip weights calculation (but do load max-w!)
        self._ignore_vis_weights = False

        # smear mapping machines
        self._smm_grid = ClassSmearMapping.SmearMappingMachine("BDA.Grid")
        self._smm_degrid = ClassSmearMapping.SmearMappingMachine("BDA.Degrid")
        self._put_vis_column_job_id = self._put_vis_column_label = None



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

        min_freq = 1e+999
        max_freq = 0

        # max chunk shape accumulated here
        self._chunk_shape = [0, 0, 0]

        for msspec in self.MSList:
            if type(msspec) is not str:
                msname, ddid, field = msspec
            else:
                msname, ddid, field = msspec, self.DicoSelectOptions["DDID"], self.DicoSelectOptions["Field"]
            MS = ClassMS.ClassMS(
                msname, Col=self.ColName, DoReadData=False,
                AverageTimeFreq=(1, 3),
                Field=field, DDID=ddid, TaQL=self.TaQL,
                TimeChunkSize=self.TMemChunkSize, ChanSlice=chanslice,
                GD=self.GD, ResetCache=self.GD["Cache"]["Reset"],
                DicoSelectOptions = self.DicoSelectOptions)
            if MS.empty:
                continue
            self.ListMS.append(MS)
            # accumulate global set of frequencies, and min/max frequency
            global_freqs.update(MS.ChanFreq)
            min_freq = min(min_freq, (MS.ChanFreq-MS.ChanWidth/2).min())
            max_freq = max(max_freq, (MS.ChanFreq+MS.ChanWidth/2).max())

            # accumulate largest chunk shape
            for row0, row1 in MS.getChunkRow0Row1():
                shape = (row1-row0, len(MS.ChanFreq), MS.Ncorr)
                self._chunk_shape = [max(a, b)
                                     for a, b in zip(self._chunk_shape, shape)]

        size = reduce(lambda x, y: x * y, self._chunk_shape)
        print >>log, "shape of data/flag buffer will be %s (%.2f Gel)" % (
            self._chunk_shape, size / float(2 ** 30))

        if not self.ListMS:
            print>>log, ModColor.Str("--Data-MS does not specify any valid Measurement Set(s)")
            raise RuntimeError,"--Data-MS does not specify any valid Measurement Set(s)"

        self.obs_detail = self.ListMS[0].get_obs_details()

        # main cache is initialized from main cache of first MS
        if ".txt" in self.GD["Data"]["MS"]:
            # main cache is initialized from main cache of the MSList
            from DDFacet.Other.CacheManager import CacheManager
            self.maincache = self.cache = CacheManager("%s.ddfcache"%self.GD["Data"]["MS"], cachedir=self.GD["Cache"]["Dir"], reset=self.GD["Cache"]["Reset"])
        else:
            # main cache is initialized from main cache of first MS
            self.maincache = self.cache = self.ListMS[0].maincache

        print>>log,"Main caching directory is %s"%self.maincache.dirname



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


        bandwidth = max_freq - min_freq
        print>>log, "Total bandwidth is %g MHz (%g to %g MHz), with %d channels" % (
            bandwidth*1e-6, min_freq*1e-6, max_freq*1e-6, len(global_freqs))

        # print>>log,"GlobalFreqs: %d: %s"%(len(self.GlobalFreqs),repr(self.GlobalFreqs))

        # OMS: ok couldn't resist adding a bandwidth option since I need it for 3C147
        # if this is 0, then looks at NFreqBands parameter
        grid_bw = self.GD["Freq"]["BandMHz"]*1e+6

        if grid_bw:
            grid_bw = min(grid_bw, bandwidth)
            NFreqBands = self.GD["Freq"][
                "NBand"] = int(math.ceil(bandwidth/grid_bw))
        else:
            NFreqBands = np.min(
                [self.GD["Freq"]["NBand"],
                 len(self.GlobalFreqs)])  # self.nMS])
            grid_bw = bandwidth/NFreqBands

        self.NFreqBands = NFreqBands
        self.MultiFreqMode = NFreqBands > 1
        if self.MultiFreqMode:
            print>>log, ModColor.Str(
                "MultiFrequency Mode: ON, %dx%g MHz bands" %
                (NFreqBands, grid_bw*1e-6))

            if not ("Alpha" in self.GD["SSDClean"]["SSDSolvePars"]):
                self.GD["SSDClean"]["SSDSolvePars"].append("Alpha")

        else:
            self.GD["Freq"]["NBand"] = 1
            self.GD["HMP"]["Alpha"] = [0., 0., 1.]
            if "Alpha" in self.GD["SSDClean"]["SSDSolvePars"]:
                self.GD["SSDClean"]["SSDSolvePars"].remove("Alpha")

            print>>log, ModColor.Str("MultiFrequency Mode: OFF")

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
        grid_band = np.floor((self.GlobalFreqs - min_freq)/grid_bw).astype(int)
        # freq_to_grid_band: mapping from frequency to grid band number
        freq_to_grid_band = dict(zip(self.GlobalFreqs, grid_band))
        # print>>log,sorted(freq_to_grid_band.items())

### not sure why linspace is here and not arange?
# <<<<<<< HEAD
        self.FreqBandCenters = np.linspace(min_freq+grid_bw/2, max_freq-grid_bw/2,self.NFreqBands)
# =======
#        self.FreqBandCenters = min_freq+grid_bw/2 + np.arange(0, self.NFreqBands)*grid_bw
#>>>>>>> master

        self.FreqBandChannels = []
        # freq_to_grid_band_chan: mapping from frequency to channel number
        # within its grid band
        freq_to_grid_band_chan = {}
        for iBand in xrange(self.NFreqBands):
            freqlist = sorted([freq for freq, band
                               in freq_to_grid_band.iteritems()
                               if band == iBand])
            self.FreqBandChannels.append(freqlist)
            freq_to_grid_band_chan.update(
                dict([(freq, chan) for chan, freq in enumerate(freqlist)]))
            print>>log, "Image band %d: %g to %g MHz contains %d MS channels from %g to %g MHz" % (iBand, (self.FreqBandCenters[iBand]-grid_bw/2)*1e-6, (
                self.FreqBandCenters[iBand]+grid_bw/2)*1e-6, len(freqlist), len(freqlist) and freqlist[0]*1e-6, len(freqlist) and freqlist[-1]*1e-6)

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
            edges = np.arange(min_freq, max_freq+degrid_bw, degrid_bw)
            self.FreqBandChannelsDegrid[iMS] = (edges[:-1] + edges[1:])/2

            print>>log, "%s   Bandwidth is %g MHz (%g to %g MHz), gridding bands are %s" % (
                MS, bw*1e-6, min_freq*1e-6, max_freq*1e-6, ", ".join(map(str, set(bands))))
            print>>log, "Grid band mapping: %s" % (" ".join(map(str, bands)))
            print >>log, "Grid chan mapping: %s" % (
                " ".join(map(str, self.DicoMSChanMappingChan[iMS])))
            print >>log, "Degrid chan mapping: %s" % (
                " ".join(map(str, self.DicoMSChanMappingDegridding[iMS])))
            print >>log, "Degrid frequencies: %s" % (" ".join(
                                                         ["%.2f" %
                                                          (x * 1e-6)
                                                          for x in self.FreqBandChannelsDegrid
                                                          [iMS]]))

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
        self.CellSizeRad = CellSizeRad

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
        APP.runJob(self._put_vis_column_job_id, self. visPutColumnHandler, args=(DATA.readonly(), field, column, likecol), io=0)

    def visPutColumnHandler (self, DATA, field, column, likecol):
        iMS, iChunk = DATA["iMS"], DATA["iChunk"]
        ms = self.ListMS[iMS]
        row0, row1 = ms.getChunkRow0Row1()[iChunk]
        if ms.ToRADEC is not None:
            ms.Rotate(DATA,RotateType=["vis"],Sense="ToPhaseCenter",DataFieldName=field)
            

        ms.PutVisColumn(column, DATA[field], row0, row1, likecol=likecol, sort_index=DATA["sort_index"])

    def collectPutColumnResults(self):
        if self._put_vis_column_job_id:
            APP.awaitJobResults(self._put_vis_column_job_id, progress="Writing %s" % self._put_vis_column_label)
            self._put_vis_column_job_id = None

    def startChunkLoadInBackground(self):
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
            # null chunk? skip to next chunk
            if not self._ignore_vis_weights:
                self.awaitWeights()
                if self.VisWeights[self.iCurrentMS][self.iCurrentChunk]["null"]:
                    print>>log, ModColor.Str("chunk %s is null, skipping"%self._next_chunk_label)
                    continue
            # ok, now we're good to load
            print>>log, "scheduling loading of chunk %s" % self._next_chunk_label
            # in single-chunk mode, DATA may already be loaded, in which case we do nothing
            if self.nTotalChunks > 1 or self.DATA is None:
                # tell the IO thread to start loading the chunk
                APP.runJob(self._next_chunk_name, self._handler_LoadVisChunk,
                           args=(self._next_chunk_name, self.iCurrentMS, self.iCurrentChunk), 
                           io=0)#,serial=True)
            return self._next_chunk_label

    def collectLoadedChunk(self, start_next=True):
        # previous data dict can now be discarded from shm
        if self.nTotalChunks > 1 and self.DATA is not None:
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
            APP.awaitJobResults(self._next_chunk_name, timing="Reading %s"%self._next_chunk_label )
            # reload the data dict -- background thread will now have populated it
            self.DATA = shared_dict.attach(self._next_chunk_name)
            self.DATA["label"] = self._next_chunk_label
            # in single-chunk mode, keep a copy of the data array
            if self.nTotalChunks == 1 and "data" in self.DATA and self._saved_data is None:
                self._saved_data = self.DATA["data"].copy()
        # schedule next event
        if start_next:
            self.startChunkLoadInBackground()
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

        print>> log, ModColor.Str("loading ms %d of %d, chunk %d of %d" % (iMS+1, self.nMS, iChunk+1, ms.numChunks()), col="green")

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

        print>>log, "  channel Mapping Gridding  : %s" % str(DATA["ChanMapping"])
        print>>log, "  channel Mapping DeGridding: %s" % str(DATA["ChanMappingDegrid"])

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
        weights = self.GetVisWeights(iMS, iChunk)
        DATA["Weights"] = weights
        if weights is None:
            print>> log, ModColor.Str("This chunk is all flagged or has zero weight.")
            return
        if DATA["sort_index"] is not None and DATA["Weights"] is not 1:
            DATA["Weights"] = DATA["Weights"][DATA["sort_index"]]

        self.computeBDAInBackground(dictname, ms, DATA,
            ChanMappingGridding=DATA["ChanMapping"],
            ChanMappingDeGridding=DATA["ChanMappingDegrid"])

        JonesMachine = ClassJones.ClassJones(self.GD, ms, self.FacetMachine)
        JonesMachine.InitDDESols(DATA)

        if data is not None and self.AddNoiseJy is not None:
            data += (self.AddNoiseJy/np.sqrt(2.))*(np.random.randn(*data.shape)+1j*np.random.randn(*data.shape))

        # load results of smear mapping computation
        self.collectBDA(dictname, DATA)

    def setFacetMachine(self, FacetMachine):
        self.FacetMachine = FacetMachine
        self.FullImShape = self.FacetMachine.OutImShape
        self.PaddedFacetShape = self.FacetMachine.PaddedGridShape
        self.FacetShape = self.FacetMachine.FacetShape
        self.CellSizeRad = self.FacetMachine.CellSizeRad

    def setFOV(self, sh0, sh1, sh2, cell):
        self.FullImShape = sh0
        self.PaddedFacetShape = sh1
        self.FacetShape = sh2
        self.CellSizeRad = cell

    def collectBDA(self, base_job_id, DATA):
        """Called in I/O thread. Waits for BDA computation to complete (if any), then populates dict"""
        if "BDA.Grid" not in DATA:
            FinalMapping, fact = self._smm_grid.collectSmearMapping(DATA, "BDA.Grid")
            print>> log, ModColor.Str("  Effective compression [grid]  :   %.2f%%" % fact, col="green")
            np.save(file(self._bda_grid_cachename, 'w'), FinalMapping)
            self.cache.saveCache("BDA.Grid")
        if "BDA.Degrid" not in DATA:
            FinalMapping, fact = self._smm_degrid.collectSmearMapping(DATA, "BDA.Degrid")
            print>> log, ModColor.Str("  Effective compression [degrid]:   %.2f%%" % fact, col="green")
            DATA["BDA.Degrid"] = FinalMapping
            np.save(file(self._bda_degrid_cachename, 'w'), FinalMapping)
            self.cache.saveCache("BDA.Degrid")

    def computeBDAInBackground(self, base_job_id, ms, DATA, ChanMappingGridding=None, ChanMappingDeGridding=None):

        GD=copy.deepcopy(self.GD)
        CriticalCacheParms=dict(Data=GD["Data"],
                                Compression=GD["Comp"],
                                Freq=GD["Freq"],
                                DataSelection=GD["Selection"],
                                Sorting=GD["Data"]["Sort"])
        del CriticalCacheParms["Data"]["ColName"],CriticalCacheParms["DataSelection"]["FlagAnts"]
        

        if True: # always True for now, non-BDA gridder is not maintained # if self.GD["Comp"]["CompGridMode"]:
            self._bda_grid_cachename, valid = self.cache.checkCache("BDA.Grid",CriticalCacheParms)
            if valid:
                print>> log, "  using cached BDA mapping %s" % self._bda_grid_cachename
                DATA["BDA.Grid"] = np.load(self._bda_grid_cachename)
            else:
                if self.GD["Comp"]["GridFoV"] == "Facet":
                    _, _, nx, ny = self.FacetShape
                elif self.GD["Comp"]["GridFoV"] == "Full":
                    _, _, nx, ny = self.FullImShape
                mode = self.GD["Comp"]["BDAMode"]
                FOV = self.CellSizeRad * nx * (np.sqrt(2.) / 2.) * 180. / np.pi
                self._smm_grid.computeSmearMappingInBackground(base_job_id, ms, DATA, FOV,
                                                          (1. - self.GD["Comp"]["GridDecorr"]),
                                                          ChanMappingGridding, mode)

        if True: # always True for now, non-BDA gridder is not maintained # if self.GD["Comp"]["CompDeGridMode"]:
            self._bda_degrid_cachename, valid = self.cache.checkCache("BDA.Degrid",CriticalCacheParms)

            if valid:
                print>> log, "  using cached BDA mapping %s" % self._bda_degrid_cachename
                DATA["BDA.Degrid"] = np.load(self._bda_degrid_cachename)
            else:
                if self.GD["Comp"]["DegridFoV"] == "Facet":
                    _, _, nx, ny = self.FacetShape
                elif self.GD["Comp"]["DegridFoV"] == "Full":
                    _, _, nx, ny = self.FullImShape
                mode = self.GD["Comp"]["BDAMode"]
                FOV = self.CellSizeRad * nx * (np.sqrt(2.) / 2.) * 180. / np.pi
                self._smm_degrid.computeSmearMappingInBackground(base_job_id, ms, DATA, FOV,
                                                          (1. - self.GD["Comp"]["DegridDecorr"]),
                                                          ChanMappingDeGridding, mode)

    def GetVisWeights(self, iMS, iChunk):
        """
        Returns path to weights array for the given MS and chunk number.

        Waits for CalcWeights to complete (if running in background).
        """
        # wmax-only means weights not computed (i.e. predict-only mode)
        if self._ignore_vis_weights:
            return 1
        # otherwise make sure we get them
        self.awaitWeights()
        if self.VisWeights[iMS][iChunk]["null"]:
            return None
        path = self.VisWeights[iMS][iChunk]["cachepath"]
        if not os.path.getsize(path):
            return None
        return np.load(file(path))

    def getMaxW(self):
        """Returns the max W value. Since this is estimated as part of weights computation, 
        wait for that to finish firest"""
        self.awaitWeights()
        return self.VisWeights["wmax"]

    def awaitWeights(self):
        if self.VisWeights is None:
            # ensure the background calculation is complete
            APP.awaitEvents(self._calcweights_event)
            # load shared dict prepared in background thread
            self.VisWeights = shared_dict.attach("VisWeights")
            # check for errors
            for iMS, MS in enumerate(self.ListMS):
                for ichunk in xrange(len(MS.getChunkRow0Row1())):
                    msw = self.VisWeights[iMS][ichunk]
                    if "error" in msw:
                        print>>log,ModColor.Str("error computing weights for %s"%MS.MSName)
                        print>>log,ModColor.Str(msw["error"])
                        raise msw["error"]

    def IgnoreWeights(self):
        """
        Tells VisServer that visibility weights will not be needed (e.g. as in predict-only mode).
        Note that the background CalcWeights job is still run in this case, but just to get the wmax
        value from the MSs
        """
        print>>log,"visibility weights will not be computed"
        self._ignore_vis_weights = True

    def CalcWeightsBackground(self):
        """Starts parallel jobs to load weights in the background"""
        self.VisWeights = None
        if self.GD["Misc"]["ConserveMemory"]:
            APP.runJob("VisWeights", self._CalcWeights_serial, io=0, singleton=True, event=self._calcweights_event)
        else:
            APP.runJob("VisWeights", self._CalcWeights_handler, io=0, singleton=True, event=self._calcweights_event)
        # APP.awaitEvents(self._calcweights_event)

    def _CalcWeights_handler(self):
        self._weight_dict = shared_dict.create("VisWeights")
        # check for wmax in cache
        cache_keys = dict([(section, self.GD[section]) for section
              in ("Data", "Selection", "Freq", "Image", "Weight")])
        wmax_path, wmax_valid = self.maincache.checkCache("wmax", cache_keys)
        if wmax_valid:
            self._weight_dict["wmax"] = cPickle.load(open(wmax_path))
        # check cache first
        have_all_weights = wmax_valid
        for iMS, MS in enumerate(self.ListMS):
            msweights = self._weight_dict.addSubdict(iMS)
            for ichunk, (row0, row1) in enumerate(MS.getChunkRow0Row1()):
                msw = msweights.addSubdict(ichunk)
                path, valid = MS.getChunkCache(row0, row1).checkCache("ImagingWeights.npy", cache_keys)
                have_all_weights = have_all_weights and valid
                msw["cachepath"] = path
                if valid:
                    msw["null"] = not os.path.getsize(path)
        # if every weight is in cache, then we're done here
        if have_all_weights:
            print>> log, "all imaging weights, and wmax, are available in cache"
            return
        # spawn parallel jobs to load weights
        for ims,ms in enumerate(self.ListMS):
            msweights = self._weight_dict[ims]
            for ichunk in xrange(len(ms.getChunkRow0Row1())):
                msw = msweights[ichunk]
                APP.runJob("LoadWeights:%d:%d"%(ims,ichunk), self._loadWeights_handler,
                           args=(msw.writeonly(), ims, ichunk, self._ignore_vis_weights),
                           counter=self._weightjob_counter, collect_result=False)
        # wait for results
        APP.awaitJobCounter(self._weightjob_counter, progress="Load weights")
        self._weight_dict.reload()
        wmax = self._uvmax = 0
        # now work out weight grid sizes, etc.
        for ims, ms in enumerate(self.ListMS):
            msweights = self._weight_dict[ims]
            for ichunk in xrange(len(ms.getChunkRow0Row1())):
                msw = msweights[ichunk]
                if "error" in msw:
                    raise msw["error"]
                wmax = max(wmax, msw["wmax"])
                self._uvmax = max(self._uvmax, msw["uvmax_wavelengths"])
        # save wmax to cache
        cPickle.dump(wmax,open(wmax_path, "w"))
        self.maincache.saveCache("wmax")
        self._weight_dict["wmax"] = wmax
        if self._ignore_vis_weights:
            return
        if not self._uvmax:
            raise RuntimeError("data appears to be fully flagged: can't compute imaging weights")
        # in natural mode, leave the weights as is. In other modes, setup grid for calculations
        self._weight_grid = shared_dict.create("VisWeights.Grid")
        cell = npix = npixx = nbands = xymax = None
        if self.Weighting != "natural":
            nch, npol, npixIm, _ = self.FullImShape
            FOV = self.CellSizeRad * npixIm
            nbands = self.NFreqBands
            cell = 1. / (self.Super * FOV)
            if self.MFSWeighting or self.NFreqBands < 2:
                nbands = 1
                print>> log, "initializing weighting grid for single band (or MFS weighting)"
            else:
                print>> log, "initializing weighting grids for %d bands" % nbands
            # find max grid extent by considering _unflagged_ UVs
            xymax = int(math.floor(self._uvmax / cell)) + 1
            # grid will be from [-xymax,xymax] in U and [0,xymax] in V
            npixx = xymax * 2 + 1
            npixy = xymax + 1
            npix = npixx * npixy
            print>> log, "Calculating imaging weights on an [%i,%i]x%i grid with cellsize %g" % (npixx, npixy, nbands, cell)
            grid0 = self._weight_grid.addSharedArray("grid", (nbands, npix), np.float64)
            # now run parallel jobs to accumulate weights
            for ims, ms in enumerate(self.ListMS):
                for ichunk in xrange(len(ms.getChunkRow0Row1())):
                    if "weight" in self._weight_dict[ims][ichunk]:
                        APP.runJob("AccumWeights:%d:%d" % (ims, ichunk), self._accumulateWeights_handler,
                                   args=(self._weight_grid.readonly(),
                                         self._weight_dict[ims][ichunk].readwrite(),
                                         ims, ichunk, ms.ChanFreq, cell, npix, npixx, nbands, xymax),
                                   counter=self._weightjob_counter, collect_result=False)
            # wait for results
            APP.awaitJobCounter(self._weightjob_counter, progress="Accumulate weights")
            if self.Weighting == "briggs" or self.Weighting == "robust":
                numeratorSqrt = 5.0 * 10 ** (-self.Robust)
                grid0 = self._weight_grid["grid"]
                for band in range(nbands):
                    grid1 = grid0[band, :]
                    avgW = (grid1 ** 2).sum() / grid1.sum()
                    sSq = numeratorSqrt ** 2 / avgW
                    grid1[...] = 1 + grid1 * sSq
        # launch jobs to finalize weights and save them to the cache
        for ims, ms in enumerate(self.ListMS):
            for ichunk in xrange(len(ms.getChunkRow0Row1())):
                APP.runJob("FinalizeWeights:%d:%d" % (ims, ichunk), self._finalizeWeights_handler,
                           args=(self._weight_grid.readonly(),
                                 self._weight_dict[ims][ichunk].readwrite(),
                                 ims, ichunk, ms.ChanFreq, cell, npix, npixx, nbands, xymax),
                           counter=self._weightjob_counter, collect_result=False)
        APP.awaitJobCounter(self._weightjob_counter, progress="Finalize weights")
        # delete stuff
        if self._weight_grid is not None:
            self._weight_grid.delete()
        # check for errors
        self._weight_dict.reload()
        for ims, ms in enumerate(self.ListMS):
            for ichunk, (row0, row1) in enumerate(ms.getChunkRow0Row1()):
                if not self._weight_dict[ims][ichunk].get("success"):
                    raise RuntimeError("weight computation has failed, see error messages above")
        # mark cache as valid
        for ims, ms in enumerate(self.ListMS):
            for ichunk, (row0, row1) in enumerate(ms.getChunkRow0Row1()):
                ms.getChunkCache(row0, row1).saveCache("ImagingWeights.npy")

    def _loadWeights_handler(self, msw, ims, ichunk, wmax_only=False, reraise=False):
        """If wmax_only is True, then don't actually read or compute weighs -- only read UVWs
        and FLAGs to get wmax"""
        msname = "MS %d chunk %d"%(ims, ichunk)
        try:
            ms = self.ListMS[ims]
            msname = "%s chunk %d"%(ms.MSName, ichunk)
            row0, row1 = ms.getChunkRow0Row1()[ichunk]
            msfreqs = ms.ChanFreq
            nrows = row1 - row0
            chanslice = ms.ChanSlice
            if not nrows:
    #            print>> log, "  0 rows: empty chunk"
                return
            tab = ms.GiveMainTable()
    #        print>>log,"  %d.%d reading %s UVW" % (ims+1, ichunk+1, ms.MSName)
            uvw = tab.getcol("UVW", row0, nrows)
            flags = np.empty((nrows, len(ms.ChanFreq), len(ms.CorrelationIds)), np.bool)
            # print>>log,(ms.cs_tlc,ms.cs_brc,ms.cs_inc,flags.shape)
    #        print>>log,"  reading FLAG"
            tab.getcolslicenp("FLAG", flags, ms.cs_tlc, ms.cs_brc, ms.cs_inc, row0, nrows)
            if ms._reverse_channel_order:
                flags = flags[:,::-1,:]
            # if any polarization is flagged, flag all 4 correlations. Shape of flags becomes nrow,nchan
    #        print>>log,"  adjusting flags"
            # if any polarization is flagged, flag all 4 correlations. Shape
            # of flags becomes nrow,nchan
            flags = flags.max(axis=2)
            valid = ~flags
            # if all channels are flagged, flag whole row. Shape of flags becomes nrow
            rowflags = flags.min(axis=1)
            # if everything is flagged, skip this entry
            if rowflags.all():
    #            print>> log, "  all flagged: marking as null"
                msw["wmax"] = 0
                msw["uvmax_wavelengths"] = 0
                return
            # max of |u|, |v| in wavelengths
            uv = uvw[:, :2]
            uvmax_wavelengths = abs(uv[~rowflags,:]).max() * msfreqs.max() / _cc
            # adjust max uv (in wavelengths) and max w
            msw["wmax"] = abs(uvw[~rowflags,2]).max()
            msw["uvmax_wavelengths"] = uvmax_wavelengths
            del uvw
            if wmax_only:
                return
            msw["uv"] = uv
            msw["flags"] = rowflags
            # now read the weights
            weight = msw.addSharedArray("weight", (nrows, ms.Nchan), np.float32)
            weight_col = self.GD["Weight"]["ColName"]
            if weight_col == "WEIGHT_SPECTRUM":
                w = tab.getcol(weight_col, row0, nrows)[:, chanslice]
    #            print>> log, "  reading column %s for the weights, shape is %s" % (weight_col, w.shape)
                if ms._reverse_channel_order:
                    w = w[:, ::-1, :]
                # take mean weight across correlations and apply this to all
                weight[...] = w.mean(axis=2)
            elif weight_col == "None" or weight_col == None:
    #            print>> log, "  Selected weights columns is None, filling weights with ones"
                weight.fill(1)
            elif weight_col == "WEIGHT":
                w = tab.getcol(weight_col, row0, nrows)
    #            print>> log, "  reading column %s for the weights, shape is %s, will expand frequency axis" % (weight_col, w.shape)
                # take mean weight across correlations, and expand to have frequency axis
                weight[...] = w.mean(axis=1)[:, np.newaxis]
            else:
                # in all other cases (i.e. IMAGING_WEIGHT) assume a column
                # of shape NRow,NFreq to begin with, check for this:
                w = tab.getcol(weight_col, row0, nrows)[:, chanslice]
    #            print>> log, "  reading column %s for the weights, shape is %s" % (weight_col, w.shape)
                if w.shape != valid.shape:
                    raise TypeError("weights column expected to have shape of %s" %
                        (valid.shape,))
                weight[...] = w
            # flagged points get zero weight
            weight *= valid
            nullweight = (weight==0).all()
            if nullweight:
                msw.delete_item("weight")
                msw.delete_item("uv")
                msw.delete_item("flags")
            else:
                msw["bandmap"] = self.DicoMSChanMapping[ims]
        except Exception,exc:
            print>> log, ModColor.Str("Error loading weights from %s:"%msname)
            for line in traceback.format_exc().split("\n"):
                print>>log,ModColor.Str("  "+line)
            msw["error"] = exc
            msw.delete_item("weight")
            msw.delete_item("uv")
            msw.delete_item("flags")

    def _uv_to_index(self, ims, uv, weights, freqs, cell, npix, npixx, nbands, xymax):
        """Helper method: converts UV coordinates to indices into a UV-grid"""
        # flip sign of negative v values -- we'll only grid the top half of the plane
        uv[uv[:, 1] < 0] *= -1
        # convert u/v to lambda, and then to pixel offset
        uv = uv[..., np.newaxis] * freqs[np.newaxis, np.newaxis, :] / _cc
        uv = np.floor(uv / cell).astype(int)
        # u is offset, v isn't since it's the top half
        x = uv[:, 0, :]
        y = uv[:, 1, :]
        x += xymax  # offset, since X grid starts at -xymax
        # convert to index array -- this gives the number of the uv-bin on the grid
        #index = msw.addSharedArray("index", (uv.shape[0], len(freqs)), np.int64)
        index = np.zeros((uv.shape[0], len(freqs)), np.int64)
        index[...] = y * npixx + x
        # if we're in per-band weighting mode, then adjust the index to refer to each band's grid
        if nbands > 1:
            index += self.DicoMSChanMapping[ims][np.newaxis, :] * npix
        # zero weight refers to zero cell (otherwise it may end up outside the grid, since grid is
        # only big enough to accommodate the *unflagged* uv-points)
        index[weights == 0] = 0
        return index

    def _accumulateWeights_handler (self, wg, msw, ims, ichunk, freqs, cell, npix, npixx, nbands, xymax):
        msname = "MS %d chunk %d"%(ims, ichunk)
        try:
            ms = self.ListMS[ims]
            msname = "%s chunk %d"%(ms.MSName, ichunk)
            weights = msw["weight"]
            index = self._uv_to_index(ims, msw["uv"], weights, freqs, cell, npix, npixx, nbands, xymax)
            msw.delete_item("flags")
            _pyGridderSmearPols.pyAccumulateWeightsOntoGrid(wg["grid"], weights.ravel(), index.ravel())
        except Exception,exc:
            print>> log, ModColor.Str("Error accumulating weights from %s:"%msname)
            for line in traceback.format_exc().split("\n"):
                print>>log,ModColor.Str("  "+line)
            msw["error"] = exc
            os.unlink(msw["cachepath"])
            msw.delete_item("weight")
            msw.delete_item("uv")
            msw.delete_item("flags")

    def _finalizeWeights_handler(self, wg, msw, ims, ichunk, freqs, cell, npix, npixx, nbands, xymax):
        msname = "MS %d chunk %d"%(ims, ichunk)
        try:
            ms = self.ListMS[ims]
            msname = "%s chunk %d"%(ms.MSName, ichunk)
            msw["success"] = True
            if "weight" in msw:
                weight = msw["weight"]
                # renormalize to density, for uniform/briggs
                if self.Weighting != "natural":
                    index = self._uv_to_index(ims, msw["uv"], weight, freqs, cell, npix, npixx, nbands, xymax)
                    grid = wg["grid"].reshape((wg["grid"].size,))
                    #weight /= grid[msw["index"]]
                    index[index>=len(grid)]=0
                    weight /= grid[index]
    #                import pdb; pdb.set_trace()

                np.save(msw["cachepath"], weight)
                msw.delete_item("weight")
                msw.delete_item("uv")
                if "flags" in msw:
                    msw.delete_item("flags")
                if "index" in msw:
                    msw.delete_item("index")
                msw["null"] = False
            elif "error" in msw:
                msw["null"] = True
                msw["success"] = False
                os.unlink(msw["cachepath"])
            else:
                msw["null"] = True
                file(msw["cachepath"], 'w').truncate(0)
        except Exception,exc:
            print>> log, ModColor.Str("Error accumulating weights from %s:"%msname)
            for line in traceback.format_exc().split("\n"):
                print>>log,ModColor.Str("  "+line)
            msw["error"] = exc
            msw["success"] = False
            os.unlink(msw["cachepath"])

    def _CalcWeights_serial(self):
        self._weight_dict = shared_dict.create("VisWeights")
        # check for wmax in cache
        cache_keys = dict([(section, self.GD[section]) for section
                           in ("Data", "Selection", "Freq", "Image", "Weight")])
        wmax_path, wmax_valid = self.maincache.checkCache("wmax", cache_keys)
        if wmax_valid:
            self._weight_dict["wmax"] = cPickle.load(open(wmax_path))
        # check cache first
        have_all_weights = wmax_valid
        for iMS, MS in enumerate(self.ListMS):
            msweights = self._weight_dict.addSubdict(iMS)
            for ichunk, (row0, row1) in enumerate(MS.getChunkRow0Row1()):
                msw = msweights.addSubdict(ichunk)
                path, valid = MS.getChunkCache(row0, row1).checkCache("ImagingWeights.npy", cache_keys)
                have_all_weights = have_all_weights and valid
                msw["cachepath"] = path
                if valid:
                    msw["null"] = not os.path.getsize(path)
        # if every weight is in cache, then we're done here
        if have_all_weights:
            print>> log, "all imaging weights, and wmax, are available in cache"
            return

        wmax = self._uvmax = 0

        # scan through MSs to determine uv-max
        for ims, ms in enumerate(self.ListMS):
            ms = self.ListMS[ims]
            max_freq = ms.ChanFreq.max()
            for ichunk in xrange(len(ms.getChunkRow0Row1())):
                print>> log, "scanning UVWs %d.%d" % (ims, ichunk)
                row0, row1 = ms.getChunkRow0Row1()[ichunk]
                nrows = row1 - row0
                if not nrows:
                    continue
                tab = ms.GiveMainTable()
                uvw = tab.getcol("UVW", row0, nrows)
                rowflags = tab.getcol("FLAG_ROW", row0, nrows)
                # max of |u|, |v| in wavelengths
                uvmax_wavelengths = abs(uvw[~rowflags, :2]).max() * max_freq / _cc
                self._uvmax = max(self._uvmax, uvmax_wavelengths)

        # setup uv-grid for non-natural weights
        if self.Weighting != "natural":
            self._weight_grid = shared_dict.create("VisWeights.Grid")
            nch, npol, npixIm, _ = self.FullImShape
            FOV = self.CellSizeRad * npixIm
            nbands = self.NFreqBands
            cell = 1. / (self.Super * FOV)
            if self.MFSWeighting or self.NFreqBands < 2:
                nbands = 1
                print>> log, "initializing weighting grid for single band (or MFS weighting)"
            else:
                print>> log, "initializing weighting grids for %d bands" % nbands
            # find max grid extent by considering _unflagged_ UVs
            xymax = int(math.floor(self._uvmax / cell)) + 1
            # grid will be from [-xymax,xymax] in U and [0,xymax] in V
            npixx = xymax * 2 + 1
            npixy = xymax + 1
            npix = npixx * npixy
            print>> log, "Calculating imaging weights on an [%i,%i]x%i grid with cellsize %g" % (npixx, npixy, nbands, cell)
            self._weight_grid.addSharedArray("grid", (nbands, npix), np.float64)

        # scan through MSs one by one
        for ims, ms in enumerate(self.ListMS):
            msweights = self._weight_dict[ims]
            for ichunk in xrange(len(ms.getChunkRow0Row1())):
                msw = msweights[ichunk]
                print>>log,"loading weights %d.%d"%(ims, ichunk)
                self._loadWeights_handler(msw, ims, ichunk, self._ignore_vis_weights)

                # if nothing in MS, handler will not return a "weight" field. Mark this chunk as null then, and truncate the cache
                msw["null"] = "weight" not in msw
                if "error" in msw:
                    raise RuntimeError("weights computation failed for one or more MSs")
                if "weight" not in msw:
                    continue

                wmax = max(wmax, msw["wmax"])
                self._uvmax = max(self._uvmax, msw["uvmax_wavelengths"])

                # in Natural mode, we're done: dump weights out
                if self.Weighting == "natural":
                    self._finalizeWeights_handler(None, msw, ims, ichunk, 0, reraise=True)
                # else accumulate onto uv grid
                else:
                    self._accumulateWeights_handler(self._weight_grid, msw,
                                         ims, ichunk, ms.ChanFreq, cell, npix, npixx, nbands, xymax, reraise=True)
                    # delete to save memory
                    for field in "weight", "uv", "flags", "index":
                        if field in msw:
                            msw.delete_item(field)

        # save wmax to cache
        cPickle.dump(wmax, open(wmax_path, "w"))
        self.maincache.saveCache("wmax")
        self._weight_dict["wmax"] = wmax
        print>>log,"overall max W is %.2f meters"%wmax
        if self._ignore_vis_weights:
            return
        if not self._uvmax:
            raise RuntimeError("data appears to be fully flagged: can't compute imaging weights")

        if self.Weighting != "natural":
            # adjust uv-grid for robust weighting
            if self.Weighting == "briggs" or self.Weighting == "robust":
                numeratorSqrt = 5.0 * 10 ** (-self.Robust)
                grid0 = self._weight_grid["grid"]
                for band in range(nbands):
                    grid1 = grid0[band, :]
                    avgW = (grid1 ** 2).sum() / grid1.sum()
                    sSq = numeratorSqrt ** 2 / avgW
                    grid1[...] = 1 + grid1 * sSq

            # rescan through MSs one by one to re-adjust the weights
            for ims, ms in enumerate(self.ListMS):
                msweights = self._weight_dict[ims]
                for ichunk in xrange(len(ms.getChunkRow0Row1())):
                    msw = msweights[ichunk]
                    if msw["null"]:
                        print>> log, "skipping weights %d.%d (null)" % (ims, ichunk)
                        continue
                    print>> log, "reloading weights %d.%d" % (ims, ichunk)
                    self._loadWeights_handler(msw, ims, ichunk, self._ignore_vis_weights)
                    self._finalizeWeights_handler(self._weight_grid, msw,
                                                      ims, ichunk, ms.ChanFreq, cell, npix, npixx, nbands, xymax, reraise=True)

            if self._weight_grid is not None:
                self._weight_grid.delete()

        # mark caches as valid
        for ims, ms in enumerate(self.ListMS):
            for ichunk, (row0, row1) in enumerate(ms.getChunkRow0Row1()):
                ms.getChunkCache(row0, row1).saveCache("ImagingWeights.npy")

