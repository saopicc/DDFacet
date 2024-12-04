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
from DDFacet.Other import logger
log = logger.getLogger("ClassMS")

import os
import dask
BACKEND_AVAIL=True
try:
    from daskms import xds_from_storage_ms, xds_from_storage_table, xds_to_storage_table 
    from daskms.utils import assert_liveness
except ImportError:
    print('Dask-MS alternative backend not available. Install with alternative backends optional dependency', file=log)
    BACKEND_AVAIL=False
import dask.array as da
import math
from astropy.time import Time
# from pyrap.tables import table
from typing import List
from warnings import warn
import numpy as np
from DDFacet.Other import ModColor
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms

from DDFacet.Other import ClassTimeIt, reformat
from DDFacet.Other.CacheManager import CacheManager

import time
from astropy.time import Time

from .ClassMS import ClassMS


class ClassDaskMS(ClassMS):
    def __init__(self,MSname,
                 Col="DATA",
                 SubCol=None,
                 zero_flag=True,ReOrder=False,EqualizeFlag=False,DoPrint=True,DoReadData=True,
                 TimeChunkSize=None,GetBeam=False,RejectAutoCorr=False,SelectSPW=None,DelStationList=None,
                 AverageTimeFreq=None,
                 Field=0,DDID=0,TaQL=None,ChanSlice=None,GD=None,
                 DicoSelectOptions={},
                 ResetCache=False,
                 first_ms=None,
                 get_obs_detail=False):
        if not BACKEND_AVAIL:
            raise RuntimeError("Dask-MS backend not available. Reinstall with "
                               "pip install DDFacet[alternate-data-backends]")
        if TaQL:
            raise RuntimeError("TaQL support not available with dask-ms")
        self._chunking = None
        self._num_dataset = None
        self._dask_store_type = None
        ClassMS.__init__(self, MSname,
                 Col,
                 SubCol,
                 zero_flag,ReOrder,EqualizeFlag,DoPrint,DoReadData,
                 TimeChunkSize,GetBeam,RejectAutoCorr,SelectSPW,DelStationList,
                 AverageTimeFreq,
                 Field,DDID,TaQL,ChanSlice,GD,
                 DicoSelectOptions,
                 ResetCache,
                 first_ms,
                 get_obs_detail)


    @property
    def datasets(self):
        """Opens dataset if not already open"""
        if self._chunking:
            return xds_from_storage_ms(self.MSName, index_cols=("TIME",), chunks=self._chunking)
        else:
            return xds_from_storage_ms(self.MSName, index_cols=("TIME",))

    @property
    def dataset(self):
        """Opens dataset if not already open"""
        return self.datasets[self._num_dataset]

    def get_obs_details(self):
        """Gets observer details from MS, for FITS header mainly"""
        ## needs dask implementation

        results = {}
        object = ""

        try:
            to = xds_from_storage_table(f'{self.MSName}::OBSERVATION')[0].compute() 
            # to = table(self.MSName + '/OBSERVATION', readonly=True, ack=False)
        except RuntimeError:
            to = None
        try:
            tf = xds_from_storage_table(f'{self.MSName}::FIELD')[0].compute()
        except RuntimeError:
            tf = None
        if tf is not None and to is not None:
            print('Read observing details from %s'%self.MSName, file=log)
        else:
            print('Some observing details in %s missing'%self.MSName, file=log)

        # Stuff relying on an OBSERVATION table:
        if to is not None:
            # Time
            tm = Time(to.TIME_RANGE.values[0] / 86400.0,
                      scale="utc",
                      format='mjd')
            results['DATE-OBS'] = tm[0].iso.split()[0]

            # Object
            try:
                object = to.LOFAR_TARGET.values[0]
            except (AttributeError, IndexError):
                object = ""
                pass

            # Telescope
            telescope = to.TELESCOPE_NAME.values[0]
            results['TELESCOP'] = telescope

            # observer
            observer = to.OBSERVER.values[0]
            results['OBSERVER'] = observer

        if not object and tf is not None:
            object = tf.NAME.values[self.Field]

        if object:
            results['OBJECT'] = object

        # Time now
        tn = Time(time.time(), format='unix')
        results['DATE-MAP'] = tn.iso.split()[0]

        del to, tf
        assert_liveness(0, 0)
        return results

    def DelData(self):
        try:
            del(self.Weights)
        except:
            pass

        try:
            del(self.data,self.flag_all)
        except:
            pass

    def LoadLOFAR_ANTENNA_FIELD(self):
        # needs dask-ms implementation
        t = xds_from_storage_table(f"{self.MSName}::LOFAR_ANTENNA_FIELD")[0].compute()
        #print>>log, ModColor.Str(" ... Loading LOFAR_ANTENNA_FIELD table...")
        na,NTiles,_ = t.ELEMENT_OFFSET.shape

        try:
            _, nAntPerTiles, _ = t.TILE_ELEMENT_OFFSET.shape
            TileOffXYZ=t.TILE_ELEMENT_OFFSET.values.reshape(na,1,nAntPerTiles,3)
            RCU = t.ELEMENT_RCU.values
            RCUMask=(RCU!=-1)[:,:,0]
            Flagged = t.ELEMENT_FLAG.values[:,:,0]
        except:
            nAntPerTiles=1
            RCUMask=np.ones((na,96),bool)
            TileOffXYZ=np.zeros((na,1,nAntPerTiles,3),float)
            Flagged=t.ELEMENT_FLAG.values[:,:,0]
            #Flagged=Flagged.reshape(Flagged.shape[0],Flagged.shape[1],1,1)*np.ones((1,1,1,3),bool)
            
        StationXYZ=t.POSITION.values.reshape(na,1,1,3)
        ElementOffXYZ=t.ELEMENT_OFFSET.values
        ElementOffXYZ=ElementOffXYZ.reshape(na,NTiles,1,3)
        
        Dico={}
        Dico["FLAGED"]=Flagged
        Dico["StationXYZ"]=StationXYZ
        Dico["ElementOffXYZ"]=ElementOffXYZ
        Dico["TileOffXYZ"]=TileOffXYZ
        #Dico["RCU"]=RCU
        Dico["RCUMask"]=RCUMask
        Dico["nAntPerTiles"]=nAntPerTiles

        self.LOFAR_ANTENNA_FIELD=Dico

        del t  # close table
        assert_liveness(0, 0)
    
    def getChunkCache (self, ichunk):
        return self._chunk_caches[ichunk]

    def GiveChunk (self, DATA, ichunk, use_cache=None, read_data=True, sort_by_baseline=False):
        self.cache = self.getChunkCache(ichunk)
        return self.ReadData(DATA, ichunk, use_cache=use_cache, read_data=read_data, sort_by_baseline=sort_by_baseline)

    def numChunks (self):
        return len(self._chunk_rowcounts)

    def getPerChunkRowCounts(self):
        return self._chunk_rowcounts
        
    def ReadData(self,DATA,ichunk,
                 ReadWeight=False,
                 use_cache=False, read_data=True,
                 sort_by_baseline=True):
        """
        Args:
            ichunk: chunk number
            ReadWeight: True if weights need to be read
            use_cache: if True, reads data and flags from the chunk cache, if available
            read_data: if False, visibilities will not be read, only flags and other data
            sort_by_baseline: if True, sorts rows in baseline-time order
        Returns:
            DATA dictionary containing all read elements
        """
        row0, row1 = self._chunk_r0r1[ichunk]
        self.ROW0 = row0
        self.ROW1 = row1
        self.nRowRead = nRowRead = row1-row0

        if row0 >= self.F_nrows:# or nRowRead == 0:
            return "EndMS"
        if row1 > self.F_nrows:
            row1 = self.F_nrows
        
        # expected data column shape
        DATA["datashape"] = datashape = (nRowRead, len(self.ChanFreq), len(self.CorrelationNames))
        DATA["datatype"]  = np.complex64

        strMS = "%s" % (ModColor.Str(self.MSName, col="green"))
        print("%s: Reading next data chunk in [%i, %i] rows" % (
            strMS, row0, row1), file=log)

        # check cache for A0,A1,time,uvw
        if use_cache:
            # In force-cache mode, cache has no keys, so use it if it exists (i.e. if we have visibilities
            # cached from previous run)
            # In auto cache mode, cache key is the start time of the process. The cache is thus reset when first
            # touched, so we read the MS on the first major cycle, and cache subsequently.
            # cache_key = dict(time=self._start_time)

            # @o-smirnov: why not that?
            # cache_key = dict(data=self.GD["Data"])
            cache_key = dict(data=self.GD["Data"],
                             selection=self.GD["Selection"],
                             Comp=self.GD["Comp"])
            metadata_path, metadata_valid = self.cache.checkCache("A0A1UVWT.npz", cache_key, ignore_key=(use_cache=="force"))
        else:
            metadata_valid = False
        # if cache is valid, we're all good
        if metadata_valid:
            npz = np.load(metadata_path)
            A0, A1, uvw, time_all, time_uniq, sort_index, dot_uvw = \
                (npz["A0"], npz["A1"], npz["UVW"], npz["TIME"], npz["TIME_UNIQ"], npz["SORT_INDEX"], npz["DOT_UVW"])
            if not sort_index.size:
                sort_index = None
            if not dot_uvw.size:
                dot_uvw = None
        else:
            ds = self.dataset
            A0, A1, time_all, uvw = \
                dask.compute(ds.ANTENNA1.data.blocks[ichunk], ds.ANTENNA2.data.blocks[ichunk],
                                 ds.TIME.data.blocks[ichunk], ds.UVW.data.blocks[ichunk], scheduler="sync")
            if sort_by_baseline:
                # make sort index
                print("sorting by baseline-time", file=log)
                sort_index = np.lexsort((time_all, A1, A0))
                print("applying sort index to metadata rows", file=log)
                A0 = A0[sort_index]
                A1 = A1[sort_index]
                uvw = uvw[sort_index]
                time_all = time_all[sort_index]
            else:
                sort_index = None
            time_uniq = np.unique(time_all)
            dot_uvw = None

        if ReadWeight:
            weights = ds.WEIGHT.data.blocks[ichunk].compute(scheduler="sync")
            if sort_index is not None:
                weights = weights[sort_index]
            DATA["weights"] = weights

        # chan_index is "::" if not swapping, or "::-1" if swapping
        chan_index = slice(None, None, -1 if self._reverse_channel_order else None) 

        DATA["uvw"]   = uvw
        visdata = DATA.addSharedArray("data", shape=datashape, dtype=np.complex64)
        if read_data:
            # check cache for visibilities
            if use_cache:
                datapath, datavalid = self.cache.checkCache("Data.npy", dict(time=self._start_time), ignore_key=(use_cache=="force"))
            else:
                datavalid = False
            # read from cache if available, else from MS
            if datavalid:
                print("reading cached visibilities from %s" % datapath, file=log)
                visdata[...] = np.load(datapath)
                #self.RotateType=["uvw"]
            else:
                print("reading MS visibilities from column %s" % self.ColName, file=log)
          
                visdata1 = getattr(ds, self.ColName).data.blocks[ichunk]
                # subtract columns, if specfied
                if self.SubColName is not None:
                    for SubCol in self.SubColName:
                        print("  subtracting MS visibilities from column %s" % SubCol, file=log)
                        visdata1 -= getattr(ds, SubCol).data.blocks[ichunk]
                visdata1 = visdata1.compute(scheduler="sync")[:, self.ChanSlice] 
                        
                if sort_index is not None:
                    print("sorting visibilities", file=log)
                    t0 = time.time()
                    visdata[...] = visdata1[sort_index, chan_index, ...]
                    print("sorting took %.1fs"%(time.time()-t0), file=log)
                else:
                    visdata[...] = visdata1[:, chan_index, ...] 
                del visdata1
  
                if self.ToRADEC is not None:
                    self.Rotate(DATA,RotateType=["vis"])

                if use_cache:
                    print("caching visibilities to %s" % datapath, file=log)
                    np.save(datapath, visdata)
                    self.cache.saveCache("Data.npy")
        # create flag array (if flagbuf is not None, array uses memory of buffer)
        flags = DATA.addSharedArray("flags", shape=datashape, dtype=np.bool)
        # check cache for flags
        if use_cache:
            flagpath, flagvalid = self.cache.checkCache("Flags.npy", dict(time=self._start_time), ignore_key=(use_cache=="force"))
        else:
            flagvalid = False
        # read from cache if available, else from MS
        if flagvalid:
            print("reading cached flags from %s" % flagpath, file=log)
            flags[...] = np.load(flagpath)
        else:
            print("reading MS flags from column FLAG", file=log)
            flags1 = ds.FLAG.data.blocks[ichunk][:, self.ChanSlice].compute(scheduler="sync")
            if sort_index is not None:
                print("sorting flags", file=log)
                t0 = time.time()
                flags[...] = flags1[sort_index, chan_index, ...]
                print("sorting took %.1fs"%(time.time()-t0), file=log)
            else:
                flags[...] = flags1[:, chan_index, ...]
            del flags1
            self.UpdateFlags(flags, uvw, visdata, A0, A1, time_all)
            if use_cache:
                print("caching flags to %s" % flagpath, file=log)
                np.save(flagpath, flags)
                self.cache.saveCache("Flags.npy")

        DecorrMode=self.GD["RIME"]["DecorrMode"]
        if 'F' in DecorrMode or "T" in DecorrMode:
            if dot_uvw is None:
                dot_uvw = self.ComputeDotUVW(A0, A1, time_all, uvw)
            DATA["uvw_dt"] = dot_uvw

        DATA["lm_PhaseCenter"] = self.lm_PhaseCenter

        DATA["sort_index"] = sort_index

        DATA["times"] = time_all
        DATA["uniq_times"] = time_uniq   # vector of unique timestamps
        DATA["nrows"] = time_all.shape[0]
        DATA["A0"]  = A0
        DATA["A1"]  = A1
        DATA["dt"]  = self.dt
        DATA["dnu"] = self.ChanWidth

        if self.zero_flag and visdata is not None:
            visdata[flags] = 1e10  # OMS: hy is this not 0?
            visdata[np.isnan(visdata)] = 0.

        if self.ToRADEC is not None:
            self.Rotate(DATA,RotateType=["uvw"])

        # save cache
        if use_cache and not metadata_valid:
            np.savez(metadata_path,A0=A0,A1=A1,UVW=uvw,TIME=time_all,TIME_UNIQ=time_uniq,
                     SORT_INDEX=sort_index if sort_index is not None else np.array([]),
                     DOT_UVW=dot_uvw if dot_uvw is not None else np.array([]))
            self.cache.saveCache("A0A1UVWT.npz")

        return DATA
    
    def readUVWs(self, ichunk):
        ds = self.dataset
        uvws = ds.UVW.blocks[ichunk].compute()        
        del ds # close and check
        assert_liveness(0, 0)
        return uvws

    def readWeights(self, ichunk, weightcols: List[str], uvw_only=False):
        nrows = self._chunk_rowcounts[ichunk]
        if not nrows:
            return None, None, None, None
        ds = self.dataset
        uvw, flags = dask.compute(ds.UVW.data.blocks[ichunk], ds.FLAG.data.blocks[ichunk], scheduler="sync")
        flags = flags[:,self.ChanSlice,:]
        if self._reverse_channel_order:
            flags = flags[:,::-1,:]
        flags = flags.max(axis=2)
        valid = ~flags
        # if all channels are flagged, flag whole row. Shape of flags becomes nrow
        rowflags = flags.min(axis=1)
        # if everything is flagged, skip this entry
        if rowflags.all():
            del ds  # close and check
            assert_liveness(0, 0)
            return None, None, None, None
        # if only UVWs needed, return now
        if uvw_only:
            del ds  # close and check
            assert_liveness(0, 0)
            return uvw, None, None, None
        
        # now read the weights
        weights = None 

        for weight_col in weightcols:
            print("reading weighting column %s"%weight_col, file=log)
            if weight_col == "WEIGHT_SPECTRUM":
                w = ds.WEIGHT_SPECTRUM.data.blocks[ichunk].compute(scheduler="sync")[:, self.ChanSlice]
    #            print>> log, "  reading column %s for the weights, shape is %s" % (weight_col, w.shape)
                if self._reverse_channel_order:
                    w = w[:, ::-1, :]
                # take mean weight across correlations and apply this to all
                w = w.mean(axis=2)
            elif weight_col == "None" or weight_col == None:
                w = None
            elif weight_col == "Lucky_kMS" and self.GD["DDESolutions"]["DDSols"]:
                ID = self._chunk_r0r1[ichunk][0]
                SolsName=self.GD["DDESolutions"]["DDSols"]
                SolsDir=self.GD["DDESolutions"]["SolsDir"]
                if SolsDir is None:
                    FileName="%skillMS.%s.Weights.%i.npy"%(reformat.reformat(self.MSName),SolsName,ID)
                else:
                    _MSName=reformat.reformat(self.MSName).split("/")[-2]
                    DirName=os.path.abspath("%s%s"%(reformat.reformat(SolsDir),_MSName))
                    if not os.path.isdir(DirName):
                        os.makedirs(DirName)
                    FileName="%s/killMS.%s.Weights.%i.npy"%(DirName,SolsName,ID)
                log.print( "  loading weights from file: %s"%FileName)
                w = np.load(FileName)
            elif weight_col.endswith(".npy"):
                log.print("  loading weights from file: %s"%weight_col)
                w = np.load(weight_col)[slice(self._chunk_r0r1[ichunk]), :]
            elif weight_col == "WEIGHT":
                w = ds.WEIGHT.data.blocks[ichunk].compute(scheduler="sync")
                w = w.mean(axis=1)[:, np.newaxis]
                # broadcast will not work to initialize weights array below, so do init here
                if weights is None:
                    weights = np.empty((nrows, self.Nchan), np.float32) 
                    weights[...] = w
                else:
                    weights *= w
            else:
                # in all other cases (i.e. IMAGING_WEIGHT) assume a column
                # of shape NRow,NFreq to begin with, check for this:
                w = getattr(ds, weight_col).data.blocks[ichunk].compute(scheduler="sync")[:, self.ChanSlice]
                if w.shape != valid.shape:
                    raise ValueError("weights column expected to have shape of %s" %
                        (valid.shape,))
            # multiply into weights
            if weights is None:
                weights = w if w is not None else np.ones((nrows, self.Nchan), np.float32)
            elif w is not None:
                weights *= w

        del ds  # close and check
        assert_liveness(0, 0)
        return uvw, flags, rowflags, weights

    def ReadMSInfo(self,first_ms=None,DoPrint=True):
        """radec_first: ra/dec of first MS, if available"""
        T= ClassTimeIt.ClassTimeIt()
        T.enableIncr()
        T.disable()

        from daskms.fsspec_store import DaskMSStore

        datastore = DaskMSStore(self.MSName)
        self._dask_store_type = datastore.type()

        datasets = xds_from_storage_ms(datastore, index_cols=("TIME",))

        # map of DDIDs and FIELDs present in this MS
        ddid_fields = {(ds.DATA_DESC_ID, ds.FIELD_ID): nds for nds, ds in enumerate(datasets)}
        self._num_dataset = ddid_fields.get((self.DDID,self.Field))
        self.empty = self._num_dataset is None
        if self.empty:
            print(ModColor.Str("MS %s (field %d, ddid %d): no rows, skipping"%(self.MSName, self.Field, self.DDID)), file=log)
            return
        dataset = datasets[self._num_dataset]

        # open main table
        # table_all = self.GiveMainTable()

        self.F_nrows = dataset.dims['row']
        self.empty = not self.F_nrows
        if self.empty:
            print(ModColor.Str("MS %s (field %d, ddid %d): no rows, skipping"%(self.MSName, self.Field, self.DDID)), file=log)
            return
#            raise RuntimeError,"no rows in MS %s, check your Field/DDID/TaQL settings"%(self.MSName)

        #print MSname+'/ANTENNA'
        ta = xds_from_storage_table(f'{self.MSName}::ANTENNA')[0].compute()
        StationNames = ta.NAME.values

        self.StationPos = ta.POSITION.values
        na = self.StationPos.shape[0]
        nbl=(na*(na-1))/2+na
        del ta # close

        # get spectral window and polarization id
        ta_ddid = xds_from_storage_table(f'{self.MSName}::DATA_DESCRIPTION')[0].compute()
        self._spwid = ta_ddid.SPECTRAL_WINDOW_ID.values[self.DDID]
        self._polid = ta_ddid.POLARIZATION_ID.values[self.DDID]

        del ta_ddid # close

        # get polarizations
        tp = xds_from_storage_table(f'{self.MSName}::POLARIZATION')[0]
        # get list of corrype enums for first row of polarization table, and convert to strings via MS_STOKES_ENUMS. 
        # self.CorrelationNames will be a list of strings
        self.CorrelationIds = tp.CORR_TYPE.values[self._polid]

        self.CorrelationNames = [ (ctype >= 0 and ctype < len(self.MS_STOKES_ENUMS) and self.MS_STOKES_ENUMS[ctype]) or
                None for ctype in self.CorrelationIds ]
        self.Ncorr = len(self.CorrelationNames)
        del tp # close
        # NB: it is possible for the MS to have different polarization

        # get start and end times of the MS
        T0, T1 = dataset.TIME[[0,-1]].values

        # make mapping into chunks
        if self.GD["Data"]["ChunkRows"] < 0:
            # self.dataset and self.datasets will get chunking directly from the dataset
            self._chunking = None
        else:
            if self._dask_store_type == "zarr":
                raise RuntimeError("Zarr storage backend does not support rechunking. Please set --Data-ChunkRows -1 to enable auto chunking.")
            # chunk by given row counts
            if self.GD["Data"]["ChunkRows"]:    
                row_chunking = min(self.GD["Data"]["ChunkRows"], self.F_nrows)
            # else chunk by time
            elif self.TimeChunkSize:
                # guesstimate chunking 
                nchunks = max(1, math.ceil((T1-T0)/(self.TimeChunkSize*3600)))
                row_chunking = min(max(1, self.F_nrows // nchunks), self.F_nrows)
            # else single big chunk
            else:
                row_chunking = self.F_nrows
            self._chunking = dict(row=row_chunking)
            # reopen dataset with this chunking
            dataset = self.dataset

        self._chunk_rowcounts = dataset.chunks['row']
        r1 = np.cumsum(self._chunk_rowcounts)  # last row+1 of each chunk
        r0 = [0] + list(r1[:-1])               # first row of each chunk
        self._chunk_r0r1 = list(zip(r0, r1))   # (r0, r1) for each chunk

        # chunk_row0 gives the starting row of each chunk
        if len(self._chunk_rowcounts) == 1:
            print("MS %s DDID %d FIELD %d (%d rows) column %s will be processed as a single chunk"%(self.MSName, self.DDID, self.Field, self.F_nrows, self.ColName), file=log)
        else:
            print("MS %s DDID %d FIELD %d (%d rows) column %s will be split into %d chunks"%(self.MSName, self.DDID, self.Field,  self.F_nrows,
                                                                                    self.ColName, len(self._chunk_rowcounts)), file=log)
        self.Nchunk = len(self._chunk_rowcounts)

        # init the per-chunk caches
        for ichunk in range(self.Nchunk):
            # note that we don't need to reset the chunk cache -- the top-level MS cache would already have been reset,
            # being the parent directory
            self._chunk_caches[ichunk] = CacheManager(
                os.path.join(self.maincache.dirname, f"ch:{ichunk}"),
                reset=False)

        T.timeit()

        dt = float(dataset.INTERVAL[0])

        ta_spectral = xds_from_storage_table(f'{self.MSName}::SPECTRAL_WINDOW', group_cols="__row__")[self._spwid].compute()
        NSPW = ta_spectral.dims['row']
        reffreq = ta_spectral.REF_FREQUENCY.values
        orig_freq = ta_spectral.CHAN_FREQ.values.squeeze() 
        self.NchanOrig = len(orig_freq)
        chan_freq = orig_freq[self.ChanSlice]

        chan_width = ta_spectral.CHAN_WIDTH.values[self._spwid,self.ChanSlice]
        self.dFreq = chan_width.flatten()[0]
        self.ChanWidth = np.abs(chan_width)

        del ta_spectral # close

        T.timeit()

        self.ChanFreq=chan_freq
        self.ChanFreqOrig=self.ChanFreq.copy()
        self.Freq_Mean=np.mean(chan_freq)
        wavelength_chan=299792458./chan_freq

        self.Nchan = Nchan = len(wavelength_chan)

        tf = xds_from_storage_table(f'{self.MSName}::FIELD')[0].compute()
        rarad, decrad = tf.PHASE_DIR.values[self.Field][0]
        if np.abs(decrad)>=np.pi/2:
            log.print(ModColor.Str("BE CAREFUL SOME SOFTWARE HAVE BEEN SHOWN TO NOT PROPERLY MANAGE DEC=90 DEGREES"))
            log.print(ModColor.Str("   ... will slightly move the assumed phase center..."))
            U,V,W = dataset.UVW.values.T
            d=np.max(np.sqrt(U**2+V**2))
            wavelmin=3e8/self.ChanFreq.max()
            ddecrad=(wavelmin/d)/100
            decrad-=ddecrad
            ddec_arcsec=ddecrad*180/np.pi*3600
            log.print(ModColor.Str("   ... moving phase center by 1/100th of a resolution element... (%.3f arcsec))"%ddec_arcsec))
        del tf # close

        if rarad<0.: rarad+=2.*np.pi
        self.OriginalRadec = self.OldRadec = rarad,decrad

        if self.ToRADEC is not None:
            ranew, decnew = rarad, decrad
            # get RA/Dec from first MS, or else parse as coordinate string
            if self.ToRADEC == "align":
                if first_ms is not None:
                    ranew, decnew = first_ms.rarad, first_ms.decrad
                    # if (ranew, decnew)!=self.OriginalRadec:
                    #     self.GD["Image"]["PhaseCenterRADEC"]=[rad2hmsdms(first_ms.rarad,Type="ra").replace(" ",":"),rad2hmsdms(first_ms.decrad,Type="dec").replace(" ",":")]
                    #     log.print("PhaseCenterRADEC in 'align'")
                    #     log.print("   set the PhaseCenterRADEC in the parset to be: %s"%str(self.GD["Image"]["PhaseCenterRADEC"]))

                which = "the common phase centre"
            else:
                which = "%s %s"%tuple(self.ToRADEC)
                SRa,SDec=self.ToRADEC
                srah,sram,sras=SRa.split(":")
                sdecd,sdecm,sdecs=SDec.split(":")
                ranew=(np.pi/180)*15.*(float(srah)+float(sram)/60.+float(sras)/3600.)
                decnew=(np.pi/180)*np.sign(float(sdecd))*(abs(float(sdecd))+float(sdecm)/60.+float(sdecs)/3600.)
            # only enable rotation if coordinates actually change
            if ranew != rarad or decnew != decrad:
                print(ModColor.Str("MS %s will be rephased to %s"%(self.MSName,which)), file=log)
                self.OldRadec = rarad,decrad
                self.NewRadec = ranew,decnew
                rarad,decrad = ranew,decnew
            else:
                self.ToRADEC = None

        T.timeit()

        self.radeg=rarad*180./np.pi
        self.decdeg=decrad*180./np.pi
         
        self._reverse_channel_order = Nchan>1 and self.ChanFreq[0] > self.ChanFreq[-1]
        if self._reverse_channel_order:
            print(ModColor.Str("(NB: this MS has reverse channel order)",col="blue"), file=log)
            wavelength_chan = wavelength_chan[::-1]
            self.ChanFreq = self.ChanFreq[::-1]
            self.dFreq = np.abs(self.dFreq)

        T.timeit()

        self.na=na
        self.Nchan=Nchan
        self.NSPW=NSPW
        # self.NSPWChan=NSPWChan: removed this: each SPW is iterated over independently
        self.NSPWChan = Nchan
        self.F_tstart=T0
        #self.F_times_all=T1
        #self.F_times=F_time_slots_all
        #self.F_ntimes=F_time_slots_all.shape[0]
        self.dt=dt
        self.DTs=T1-T0
        self.DTh=self.DTs/3600.
        self.radec=(rarad,decrad)
        self.rarad=rarad
        self.decrad=decrad
        self.reffreq=reffreq
        self.StationNames=StationNames
        self.wavelength_chan=wavelength_chan
        self.rac=rarad
        self.decc=decrad
        self.nbl=nbl
        self.StrRA  = rad2hmsdms(self.rarad,Type="ra").replace(" ",":")
        self.StrDEC = rad2hmsdms(self.decrad,Type="dec").replace(" ",".")
        self.lm_PhaseCenter=self.radec2lm_scalar(self.OldRadec[0],self.OldRadec[1])
        T.timeit()

        del datasets, dataset
        # ensures all dask-ms table accessors are closed
        assert_liveness(0, 0)

    def PutVisColumn(self, colname, vis, ichunk, likecol="DATA", sort_index=None, ColDesc=None):
        
        row0, row1 = self._chunk_r0r1[ichunk]
        nrow = row1 - row0
        if self._reverse_channel_order:
            vis = vis[:,::-1,...]

        print("writing column %s rows %d:%d"%(colname,row0,row1), file=log)

        IsNan=np.isnan(vis)
        if np.count_nonzero(IsNan)>0:
            log.print(ModColor.Str("There are NaNs in the array to be put in %s, replacing by zeros..."%colname))
            vis[IsNan]=0

        datasets = self.datasets
        dataset = self.dataset

        # does the column exist?
        newcol = hasattr(dataset, colname)

        # if sorting rows, rearrange vis array back into MS order
        # if not sorting, then using slice(None) for row has no effect
        if sort_index is not None:
            reverse_index = np.empty(nrow,dtype=int)
            reverse_index[sort_index] = np.arange(0,nrow,dtype=int)
        else:
            reverse_index = slice(None)
        # form up full colum shape (column may be row-chan for weights or row-chan-corr for data)
        fullshape = list(vis.shape)
        fullshape[0] = nrow
        fullshape[1] = self.NchanOrig
        fullshape = tuple(fullshape)

        # if slicing, may need to read the original array and assign in
        if self.ChanSlice and self.ChanSlice != slice(None):
            if newcol:
                vis0 = np.zeros(fullshape, vis.dtype)
            else:
                vis0 = getattr(dataset, colname).values
            vis0[:, self.ChanSlice, ...] = vis[reverse_index, :, ...]
        # if not slicing, we write the full array
        else:
            if sort_index is None:
                vis0 = vis
            else:
                vis0 = np.zeros(fullshape, vis.dtype)
                vis0[sort_index,...] = vis
        # assign to dataset
        labels = ("row", "chan", "corr") if vis.ndim == 3 else ("row", "chan")

        if self._dask_store_type == "zarr":
            # create padded array and assign to chunk
            padded_data = da.empty([self.F_nrows] + list(fullshape[1:]), 
                                   chunks=tuple(dataset.chunks[axis] for axis in labels))
            padded_data[row0:row1] = vis0
            # assign to datasets
            datasets[self._num_dataset] = dataset.assign({colname: (labels, padded_data)})
            writes = xds_to_storage_table(datasets, self.MSName, columns=(colname,))
            dask.compute(getattr(writes[self._num_dataset], colname).data.blocks[ichunk], scheduler="sync")
        elif self._dask_store_type == "casa":
            # get subset of dataset
            cds = dataset.sel({"row": range(row0, row1)})
            # assign to it
            datasets[self._num_dataset] = cds.assign({colname: (labels, da.from_array(vis0))})
            writes = xds_to_storage_table(datasets, self.MSName, columns=(colname,))
            dask.compute(writes[self._num_dataset], scheduler="sync")
            del cds
        else:
            raise RuntimeError(f"The DDFacet dask-ms layer doesn't support writing to the '{self._dask_store_type}' backend yet")
        # write
            
        del writes
        del dataset, datasets
        assert_liveness(0, 0)


