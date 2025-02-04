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

from DDFacet.Data import ClassMS
from DDFacet.Data.ClassStokes import ClassStokes
from DDFacet.Other import ModColor
from DDFacet.Other import logger
from functools import reduce
logger.setSilent(["NpShared"])
from DDFacet.Data import ClassSmearMapping
from DDFacet.Data import ClassJones
from DDFacet.Array import shared_dict
from DDFacet.Other.AsyncProcessPool import APP
from DDFacet.Other import reformat
import six
import DDFacet.Other.PrintList
if six.PY3:
    from DDFacet.cbuild.Gridder import _pyGridderSmearPols3x as _pyGridderSmearPols
else:
    from DDFacet.cbuild.Gridder import _pyGridderSmearPols27 as _pyGridderSmearPols
import copy
from DDFacet.Other import ClassGiveSolsFile
import psutil
from astropy.io import fits
from astropy.time import Time as astropyTime

log = logger.getLogger("ClassWeightMachine")

_cc = 299792458

SERIAL=True
SERIAL=False

class ClassWeightMachine():
    def __init__(self,VS):
        self.VS=VS
        
        self._weightjob_counter = APP.createJobCounter("VisWeights")
        self._taperjob_counter = APP.createJobCounter("TaperWeights")
        self._calcweights_event = APP.createEvent("VisWeights")
        if APP is not None:
            APP.registerJobHandlers(self)
            self._app_id = "WeightMachine"

        GD=self.GD=VS.GD
        self.Weighting = GD["Weight"]["Mode"].lower()

        if self.Weighting not in ("natural", "uniform", "briggs", "robust"):
            raise ValueError("unknown Weighting=%s" % self.Weighting)
        self.MFSWeighting = GD["Weight"]["MFS"]
        self.Robust = GD["Weight"]["Robust"]
        self.Super = GD["Weight"]["SuperUniform"]
        self.VisWeights = None
        
        self.HasPrintedTaperingSettings = False
        self.EnableSigmoidTaper = GD["Weight"]["EnableSigmoidTaper"]
        self.SigmoidInCut = GD["Weight"]["SigmoidTaperInnerCutoff"]
        self.SigmoidOutCut = GD["Weight"]["SigmoidTaperOuterCutoff"]
        self.SigmoidInRoll = GD["Weight"]["SigmoidTaperInnerRolloffStrength"]
        self.SigmoidOutRoll = GD["Weight"]["SigmoidTaperOuterRolloffStrength"]
        
        # if True, then skip weights calculation (but do load max-w!)
        self._ignore_vis_weights = False
        self.ListMS=self.VS.ListMS
        self.maincache=self.VS.maincache
        self.DicoMSChanMapping=self.VS.DicoMSChanMapping
        self.FacetMachine=self.VS.FacetMachine
        self.NFreqBands=self.VS.NFreqBands
        

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
        return np.load(open(path, "rb")),np.load(open("%s.sgn.npy"%path, "rb"))

    def getMaxW(self):
        """Returns the max W value. Since this is estimated as part of weights computation, 
        wait for that to finish firest"""
        self.awaitWeights()
        return self.VisWeights["wmax"]
        
    def getMaxUV(self):
        """Returns the max UV value. Since this is estimated as part of weights computation, 
        wait for that to finish first"""
        self.awaitWeights()
        return self.VisWeights["uvmax"]

    def awaitWeights(self):
        if self.VisWeights is None:
            # ensure the background calculation is complete
            APP.awaitEvents(self._calcweights_event)
            # load shared dict prepared in background thread
            self.VisWeights = shared_dict.attach("VisWeights")
            # check for errors
            for iMS, MS in enumerate(self.ListMS):
                for ichunk in range(MS.numChunks()):
                    msw = self.VisWeights[iMS][ichunk]
                    if "error" in msw:
                        print(ModColor.Str("error computing weights for %s"%MS.MSName), file=log)
                        print(ModColor.Str(msw["error"]), file=log)
                        raise msw["error"]

    def IgnoreWeights(self):
        """
        Tells VisServer that visibility weights will not be needed (e.g. as in predict-only mode).
        Note that the background CalcWeights job is still run in this case, but just to get the wmax
        value from the MSs
        """
        print("visibility weights will not be computed", file=log)
        self._ignore_vis_weights = True

    def CalcWeightsBackground(self,iField=None):
        """Starts parallel jobs to load weights in the background"""
        self.VisWeights = None
        if MPIManager.size > 1:
            APP.runJob("VisWeights", self._CalcWeights_serial, io=0, singleton=True, event=self._calcweights_event, serial=True)
        else:
            if self.GD["Misc"]["ConserveMemory"]:
                #APP.runJob("VisWeights", self._CalcWeights_serial, io=0, singleton=True, event=self._calcweights_event)
                APP.runJob("VisWeights", self._CalcWeights_serial, io=0, singleton=True, event=self._calcweights_event, args=(iField,) ,serial=SERIAL)
            else:
                APP.runJob("VisWeights", self._CalcWeights_handler, io=0, singleton=True, event=self._calcweights_event, args=(iField,) ,serial=SERIAL)
            # APP.awaitEvents(self._calcweights_event)

    def _sigtaper(self, msw, chanfreq, inner_cut, outer_cut, outer_taper_strength, inner_taper_strength): 
        u = msw["uv"][:, 0]
        v = msw["uv"][:, 1]
        visweights = msw["weight"]
        if chanfreq.size != visweights.shape[1]:
            raise ValueError("Visibility weight channels don't match provided measurement frequencies")
        if visweights.ndim != 2:
            raise ValueError("Provided visibility weights must be of form nrow x nchan")
        uvdist = np.sqrt(u**2 + v**2)
        uvdistlda = np.outer(uvdist, 1 / (299792458.0 / chanfreq))
        if visweights.shape != uvdistlda.shape:
            raise ValueError("Visibility weights shape does not match number of rows in the UVW coords")
        inner_cut = np.abs(inner_cut)
        outer_cut = np.abs(outer_cut)
        if inner_cut >= outer_cut:
            raise ValueError("Taper inner cut exceeds outer cut or outer cut left unset")
        outer_taper_strength = max(1.0e-6, 1 - np.cos(min(np.abs(outer_taper_strength), 1.0) * np.pi / 2))
        inner_taper_strength = max(1.0e-6, 1 - np.cos(min(np.abs(inner_taper_strength), 1.0) * np.pi / 2))
        def __sigmoid(x): 
            d = (1 + np.exp(-x))
            d[d == 0] = 1.0e-10
            return 1 / d
        y = (__sigmoid(-uvdistlda * outer_taper_strength + outer_cut * outer_taper_strength) +
                __sigmoid(+uvdistlda * inner_taper_strength - inner_cut * inner_taper_strength) +
                __sigmoid(+uvdistlda * outer_taper_strength + outer_cut * outer_taper_strength) + 
                __sigmoid(-uvdistlda * inner_taper_strength - inner_cut * inner_taper_strength)) - 2.0
        visweights *= (y / y.max())

    def _CalcSigmoidTaper(self, ims, ms, ichunk):
        """
            Sets up the UV crafter using Sigmoids to taper inner and outer
            The cuts are specified in uvlambda and the rolloff tuning parameter will be clamped to
            0 < tuner <= 1.0.

            Provide visibility weights of the form nrow x nchan (by now they should be padded
            to that form either way by the visreader)
        """
        if self.EnableSigmoidTaper:
            if not self.HasPrintedTaperingSettings:
                print("Tapering visibilities with the uv-crafter:", file=log)
                print("\t Inner Cutoff {0:.2f} klda".format(self.SigmoidInCut * 1.0e-3), file=log)
                print("\t Outer Cutoff {0:.2f} klda".format(self.SigmoidOutCut * 1.0e-3), file=log)
                print("\t Inner Rolloff Strength {0:.2f}".format(self.SigmoidInRoll), file=log)
                print("\t Outer Rolloff Strength {0:.2f}".format(self.SigmoidOutRoll), file=log)
                self.HasPrintedTaperingSettings = True
            # APP will handle any serialization if NCPU == 1
            APP.runJob("SigmoidTaper:%d:%d" % (ims, ichunk), self._sigtaper,
                        args=(self._weight_dict[ims][ichunk].readwrite(),
                                ms.ChanFreq,
                                self.SigmoidInCut, self.SigmoidOutCut, 
                                self.SigmoidOutRoll, self.SigmoidInRoll),
                        counter=self._weightjob_counter, collect_result=False,serial=SERIAL)
            APP.awaitJobCounter(self._weightjob_counter, progress="Sigmoid Tapering")

    def _CalcWeights_handler(self,iField=None):
        StrField=""
        # if iField is not None:
        #     StrField="_Field%i"%iField
        self._weight_dict = shared_dict.create("VisWeights")
        # check for wmax in cache
        cache_keys = dict([(section, self.GD[section]) for section
              in ("Data", "Selection", "Freq", "Image", "Weight", "DDESolutions")])
        wmax_path, wmax_valid = self.maincache.checkCache("wmax", cache_keys)
        uvmax_path, uvmax_valid = self.maincache.checkCache("uvmax", cache_keys)
        if wmax_valid:
            self._weight_dict["wmax"] = cPickle.load(open(wmax_path,'rb'))
        if uvmax_valid:
            self._weight_dict["uvmax"] = cPickle.load(open(uvmax_path,'rb'))
        # check cache first
        have_all_weights = wmax_valid and uvmax_valid
        for iMS, MS in enumerate(self.ListMS):
            msweights = self._weight_dict.addSubdict(iMS)
            for ichunk in range(len(MS.getPerChunkRowCounts())):
                msw = msweights.addSubdict(ichunk)
                path, valid = MS.getChunkCache(ichunk).checkCache("ImagingWeights%s.npy"%StrField, cache_keys, reset=(self.GD["Cache"]["Weight"]=="reset"))
                have_all_weights = have_all_weights and valid
                msw["cachepath"] = path
                if valid:
                    msw["null"] = not os.path.getsize(path)

        # if every weight is in cache, then we're done here
        if have_all_weights:
            print("all imaging weights, wmax, and uvmax are available in cache", file=log)
            return
        # spawn parallel jobs to load weights
        for ims,ms in enumerate(self.ListMS):
            msweights = self._weight_dict[ims]
            for ichunk in range(len(ms.getPerChunkRowCounts())):
                msw = msweights[ichunk]
                APP.runJob("LoadWeights:%d:%d%s"%(ims,ichunk,StrField), self._loadWeights_handler,
                           args=(msw.writeonly(), ims, ichunk, self._ignore_vis_weights),
                           counter=self._weightjob_counter, collect_result=False,serial=SERIAL)
        # wait for results
        APP.awaitJobCounter(self._weightjob_counter, progress="Load weights")
        self._weight_dict.reload()
        wmax = self._uvmax = 0
        num_valid_chunks = 0
        # now work out weight grid sizes, etc.
        for ims, ms in enumerate(self.ListMS):
            msweights = self._weight_dict[ims]
            for ichunk in range(len(ms.getPerChunkRowCounts())):
                msw = msweights[ichunk]
                if "error" in msw:
                    raise msw["error"]
                if "weight" in msw:
                    num_valid_chunks += 1
                    wmax = max(wmax, msw["wmax"])
                    self._uvmax = max(self._uvmax, msw["uvmax_wavelengths"])

        if MPIManager.useMPI:
            self._uvmax = MPIManager.COMM_WORLD.allreduce(self._uvmax, MPIManager.MAX)
            wmax = MPIManager.COMM_WORLD.allreduce(wmax, MPIManager.MAX)

                    
        # save wmax to cache
        cPickle.dump(wmax,open(wmax_path, "wb"))
        self.maincache.saveCache("wmax")
        self._weight_dict["wmax"] = wmax
        # LB - Need to cache this to set scales in ScaleMachine
        cPickle.dump(self._uvmax, open(uvmax_path, "wb"))
        self.maincache.saveCache("uvmax")
        self._weight_dict["uvmax"] = self._uvmax
        if self._ignore_vis_weights:
            return
        if not self._uvmax:
            UserWarning("data appears to be fully flagged: can't compute imaging weights")

        # in natural mode, leave the weights as is. In other modes, setup grid for calculations
        self._weight_grid = shared_dict.create("VisWeights.Grid")
        cell = npix = npixx = nbands = xymax = None
        
        if iField is not None:
            CellSizeRad_x,CellSizeRad_y=self.FacetMachine.LFM[iField].CellSizeRad
            nch, npol, npixIm_x, npixIm_y = self.FacetMachine.LFM[iField].OutImShape
        else:
            CellSizeRad_x,CellSizeRad_y=self.FacetMachine.CellSizeRad
            nch, npol, npixIm_x, npixIm_y = self.FacetMachine.OutImShape
            
        if self.Weighting != "natural":
            FOV_x = CellSizeRad_x * npixIm_x
            FOV_y = CellSizeRad_y * npixIm_y
            nbands = self.NFreqBands
            
            cell_u = 1. / (self.Super * FOV_x)
            cell_v = 1. / (self.Super * FOV_y)
            if self.MFSWeighting or self.NFreqBands < 2:
                nbands = 1
                print("initializing weighting grid for single band (or MFS weighting)", file=log)
            else:
                print("initializing weighting grids for %d bands" % nbands, file=log)
            # find max grid extent by considering _unflagged_ UVs
            xymax = int(math.floor(self._uvmax / np.min([cell_u,cell_v]))) + 1
            cell=cell_u,cell_v
            # grid will be from [-xymax,xymax] in U and [0,xymax] in V
            npixx = xymax * 2 + 1
            npixy = xymax + 1
            npix = npixx * npixy

            GridSizeGB=(nbands* npix)*8/1024**3
            AvailableGB= psutil.virtual_memory().available/1024**3
            NJobs=0
            for ims, ms in enumerate(self.ListMS):
                for ichunk in range(len(ms.getPerChunkRowCounts())):
                    if "weight" in self._weight_dict[ims][ichunk]:
                        NJobs+=1
            NGrids=np.min([int(AvailableGB/GridSizeGB),NJobs])
            NGrids=np.max([1,NGrids])

            
            # If the grid size is higher that the uint32 index the cell cannot be adressed
            if nbands*npix>4294967290:
                stop
            
            if NGrids!=NJobs:
                NGrids=1
                useSems = num_valid_chunks > 1
                OneGridPerJob=False
                print("Not enough space in the RAM: using one single grid and semaphores on it", file=log)
            else:
                OneGridPerJob=True
                useSems = 0
                print("Enough space in the RAM: using one grids per accumulateWeights job and no semaphores", file=log)
            gridJobs = self._weight_grid.addSharedArray("gridJobs", (NGrids,nbands, npix), np.float64)
            print("Calculating imaging weights on %i [%i,%i]x%i grids with cellsize %g,%g" % (NGrids,npixx, npixy, nbands, cell[0],cell[1]), file=log)
                
                
            # now run parallel jobs to accumulate weights
            iGrid=0
            for ims, ms in enumerate(self.ListMS):
                for ichunk in range(len(ms.getPerChunkRowCounts())):
                    if "weight" in self._weight_dict[ims][ichunk]:
                        APP.runJob("AccumWeights:%d:%d%s" % (ims, ichunk,StrField), self._accumulateWeights_handler,
                                   args=(self._weight_grid.readonly(),
                                         self._weight_dict[ims][ichunk].readwrite(),
                                         ims, ichunk, ms.ChanFreq, cell, npix, npixx, nbands, xymax, useSems,iGrid),
                                   counter=self._weightjob_counter, collect_result=False,serial=SERIAL)
                        if OneGridPerJob:
                            iGrid+=1
                            
            # wait for results
            APP.awaitJobCounter(self._weightjob_counter, progress="Accumulate weights")
            self._weight_grid["grid"]=np.sum(gridJobs,axis=0)
            #self._weight_grid["gridJobs"].delete()
            grid0=self._weight_grid["grid"]
            
            if self.Weighting == "briggs" or self.Weighting == "robust":
                numeratorSqrt = 5.0 * 10 ** (-self.Robust)
                grid0 = self._weight_grid["grid"]
                for band in range(nbands):
                    grid1 = grid0[band, :]
                    avgW = (grid1 ** 2).sum() / grid1.sum()
                    sSq = numeratorSqrt ** 2 / avgW
                    grid1[...] = 1 + grid1 * sSq

        if MPIManager.useMPI:
            self._weight_grid["grid"] = MPIManager.COMM_WORLD.allreduce(self._weight_grid["grid"], MPIManager.SUM)
            
        # launch jobs to finalize weights and save them to the cache
        for ims, ms in enumerate(self.ListMS):
            for ichunk in range(len(ms.getPerChunkRowCounts())):
                self._CalcSigmoidTaper(ims, ms, ichunk)
                APP.runJob("FinalizeWeights:%d:%d%s" % (ims, ichunk,StrField), self._finalizeWeights_handler,
                           args=(self._weight_grid.readonly(),
                                 self._weight_dict[ims][ichunk].readwrite(),
                                 ims, ichunk, ms.ChanFreq, cell, npix, npixx, nbands, xymax),
                           counter=self._weightjob_counter, collect_result=False,serial=SERIAL)
        APP.awaitJobCounter(self._weightjob_counter, progress="Finalize weights")
        # delete stuff
        if self._weight_grid is not None:
            self._weight_grid.delete()
        # check for errors
        self._weight_dict.reload()
        for ims, ms in enumerate(self.ListMS):
            for ichunk in range(len(ms.getPerChunkRowCounts())):
                if not self._weight_dict[ims][ichunk].get("success"):
                    raise RuntimeError("weight computation has failed, see error messages above")
        # mark cache as valid
        for ims, ms in enumerate(self.ListMS):
            for ichunk in range(len(ms.getPerChunkRowCounts())):
                ms.getChunkCache(ichunk).saveCache("ImagingWeights%s.npy"%StrField)

    def _loadWeights_handler(self, msw, ims, ichunk, wmax_only=False):
        """If wmax_only is True, then don't actually read or compute weighs -- only read UVWs
        and FLAGs to get wmax"""
        msname = "MS %d chunk %d"%(ims, ichunk)
        #if True:
        try:
            
            ms = self.ListMS[ims]
            List_weight_col = self.GD["Weight"]["ColName"]
            if not isinstance(List_weight_col,list):
                List_weight_col=[List_weight_col]
            uvw, flags, rowflags, weights, sgnweights = ms.readWeights(ichunk, uvw_only=wmax_only, weightcols=List_weight_col)
            
            # skip empty or fully flagged chunks
            if uvw is None:
                msw["wmax"] = 0
                msw["uvmax_wavelengths"] = 0
                return
            msname = "%s chunk %d"%(ms.MSName, ichunk)
            msfreqs = ms.ChanFreq
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
            weight = msw.addSharedArray("weight", flags.shape, np.float32)
            weight.fill(1)
            weight[...] = weights
            weight[flags] = 0
            
            sgnweight = msw.addSharedArray("sgnweight", flags.shape[0:2], np.int8)
            sgnweight.fill(1)
            sgnweight[...] = sgnweights
            sgnweight[flags] = 0
            
            # np.savez("SingleFacet_%i%i.npz"%(ims, ichunk),uv=uv,flags=flags,rowflags=rowflags,weight=weight,weights=weights)
            # stop
            
            # check for null weights
            nullweight = (weight==0).all()
            if nullweight:
                msw.delete_item("sgnweight")
                msw.delete_item("weight")
                msw.delete_item("uv")
                msw.delete_item("flags")
            else:
                msw["bandmap"] = self.DicoMSChanMapping[ims]
        except Exception as exc:
            print(ModColor.Str("Error loading weights from %s:"%msname), file=log)
            for line in traceback.format_exc().split("\n"):
                print(ModColor.Str("  "+line), file=log)
            msw["error"] = exc
            msw.delete_item("weight")
            msw.delete_item("uv")
            msw.delete_item("flags")
            msw.delete_item("sgnweight")
            
            chanslice = ms.ChanSlice
            if not nrows:
    #            print>> log, "  0 rows: empty chunk"
                return
            tab = ms.GiveMainTable()
    #        print>>log,"  %d.%d reading %s UVW" % (ims+1, ichunk+1, ms.MSName)
            uvw = tab.getcol("UVW", row0, nrows)
            flags = np.empty((nrows, len(ms.ChanFreq), len(ms.CorrelationIds)), bool)

            
            # print>>log,(ms.cs_tlc,ms.cs_brc,ms.cs_inc,flags.shape)
    #        print>>log,"  reading FLAG"
            tab.getcolslicenp("FLAG", flags, ms.cs_tlc, ms.cs_brc, ms.cs_inc, row0, nrows)
            # if any polarization is flagged, flag all 4 correlations. Shape of flags becomes nrow,nchan
    #        print>>log,"  adjusting flags"
            # if any polarization is flagged, flag all 4 correlations. Shape
            # of flags becomes nrow,nchan

            
            
            d0, d1 = self.GD["Selection"]["UVRangeKm"]
            d0 = d0**2*1e6
            d1 = d1**2*1e6
            duv = (uvw[:,:2]**2).sum(1)  # u^2+v^2... and we already squared d0 and d1
            flags[(duv < d0) | (duv > d1),:,:] = True
            #flags = flags.any(axis=2)
            #rowflags = flags.all(axis=1)
            



            
            

    def _uv_to_index_Cheap(self, ims, uv, weights, freqsAll, cell, npix, npixx, nbands, xymax):
        """Helper method: converts UV coordinates to indices into a UV-grid"""
        # flip sign of negative v values -- we'll only grid the top half of the plane
        cell=np.array(cell,np.float64) 
        cell_u,cell_v=cell
        index = np.zeros((uv.shape[0], len(freqsAll)), np.uint32)

        uv[uv[:, 1] < 0] *= -1
        uvs=uv
        for iChan in range(freqsAll.size):
            freqs=freqsAll[iChan:iChan+1]
            # convert u/v to lambda, and then to pixel offset
            uv = uvs[..., np.newaxis] * freqs[np.newaxis, np.newaxis, :] / _cc
        
            #print("JHYGJHY uv",uv.nbytes/1024**3)
            #uv2 = np.floor(uv / cell.reshape((1,2,1))).astype(int)
            u = uv[:, 0, :]
            v = uv[:, 1, :]
            x = np.floor(u / cell[0]).astype(int)
            y = np.floor(v / cell[1]).astype(int)
            # # u is offset, v isn't since it's the top half
            # x = uv2[:, 0, :]
            # y = uv2[:, 1, :]
            #np.savez("indexIn.new.npz",x=x,y=y,xymax=xymax,DicoMSChanMapping=self.DicoMSChanMapping[ims])
            x += xymax  # offset, since X grid starts at -xymax
            # convert to index array -- this gives the number of the uv-bin on the grid
            #index = msw.addSharedArray("index", (uv.shape[0], len(freqs)), np.int64)
            
            
            
            #print("JHYGJHY index",index.nbytes/1024**3)
            index[:,iChan].flat[:] = y.flat[:] * npixx + x.flat[:]
            
            
        # np.savez("indexIn.new.npz",uv=uv,uv2=uv2,index=index,x=x,y=y,
        #          xymax=xymax,cell=cell,
        #          DicoMSChanMapping=self.DicoMSChanMapping[ims])
        # stop
        # if we're in per-band weighting mode, then adjust the index to refer to each band's grid
        if nbands > 1:
            index += self.DicoMSChanMapping[ims][np.newaxis, :] * npix
        # zero weight refers to zero cell (otherwise it may end up outside the grid, since grid is
        # only big enough to accommodate the *unflagged* uv-points)
        index[weights == 0] = 0
        #np.savez("indexIn2.new.npz",index=index,x=x,y=y,xymax=xymax,DicoMSChanMapping=self.DicoMSChanMapping[ims])
        return index


    def _uv_to_index(self, ims, uv, weights, freqs, cell, npix, npixx, nbands, xymax):
        """Helper method: converts UV coordinates to indices into a UV-grid"""
        # flip sign of negative v values -- we'll only grid the top half of the plane
        cell=np.array(cell,np.float64) 
        cell_u,cell_v=cell

        uv[uv[:, 1] < 0] *= -1
        # convert u/v to lambda, and then to pixel offset
        uv = uv[..., np.newaxis] * freqs[np.newaxis, np.newaxis, :] / _cc
        
        #uv2 = np.floor(uv / cell.reshape((1,2,1))).astype(int)
        u = uv[:, 0, :]
        v = uv[:, 1, :]
        x = np.floor(u / cell[0]).astype(int)
        y = np.floor(v / cell[1]).astype(int)
        # # u is offset, v isn't since it's the top half
        # x = uv2[:, 0, :]
        # y = uv2[:, 1, :]
        #np.savez("indexIn.new.npz",x=x,y=y,xymax=xymax,DicoMSChanMapping=self.DicoMSChanMapping[ims])
        x += xymax  # offset, since X grid starts at -xymax
        # convert to index array -- this gives the number of the uv-bin on the grid
        #index = msw.addSharedArray("index", (uv.shape[0], len(freqs)), np.int64)
        index = np.zeros((uv.shape[0], len(freqs)), np.uint32)
        index[...] = y * npixx + x
        # np.savez("indexIn.new.npz",uv=uv,uv2=uv2,index=index,x=x,y=y,
        #          xymax=xymax,cell=cell,
        #          DicoMSChanMapping=self.DicoMSChanMapping[ims])
        # stop
        # if we're in per-band weighting mode, then adjust the index to refer to each band's grid
        if nbands > 1:
            index += self.DicoMSChanMapping[ims][np.newaxis, :] * npix
        # zero weight refers to zero cell (otherwise it may end up outside the grid, since grid is
        # only big enough to accommodate the *unflagged* uv-points)
        index[weights == 0] = 0
        #np.savez("indexIn2.new.npz",index=index,x=x,y=y,xymax=xymax,DicoMSChanMapping=self.DicoMSChanMapping[ims])
        return index

    
    def _accumulateWeights_handler (self, wg, msw, ims, ichunk, freqs, cell, npix, npixx, nbands, xymax, useSems=False,iJob=None):
        msname = "MS %d chunk %d"%(ims, ichunk)
        
        try:
            ms = self.ListMS[ims]
            if iJob is not None:
                grid=wg["gridJobs"][iJob]
            else:
                grid=wg["grid"]
            msname = "%s chunk %d"%(ms.MSName, ichunk)
            weights = msw["weight"]
            #wg["grid"].fill(1)
            #print("JHYG",msw["uv"].nbytes/1024**3, weights.nbytes/1024**3)
            index = self._uv_to_index_Cheap(ims, msw["uv"], weights, freqs, cell, npix, npixx, nbands, xymax)
            #np.savez("index.new.npz",ims=ims,msw=msw["uv"], weights=weights, freqs=freqs, cell=cell, npix=npix, npixx=npixx, nbands=nbands, xymax=xymax)
            msw.delete_item("flags")
            #np.savez("accumulateWeights_handler.new.npz",grid=wg["grid"], weights=weights.ravel(), index=index.ravel())
            #print("JHKKYG",grid.nbytes/1024**3, weights.nbytes/1024**3, index.nbytes/1024**3)
            #wg["grid"].fill(0)
            #print("JHYGH dtype",grid.dtype,weights.dtype,index.dtype)
            #print("JHYGH shape",grid.shape,weights.shape,index.shape)
            
            if useSems:
                #print("pyAccumulateWeightsOntoGrid")
                _pyGridderSmearPols.pyAccumulateWeightsOntoGrid(grid, weights.ravel(), index.ravel())
                #print("pyAccumulateWeightsOntoGrid:done")
            else:
                #print("pyAccumulateWeightsOntoGridNoSem")
                _pyGridderSmearPols.pyAccumulateWeightsOntoGridNoSem(grid, weights.ravel(), index.ravel())
                #print("pyAccumulateWeightsOntoGridNoSem:done")
            #print("JHKKYG111",grid.nbytes/1024**3, weights.nbytes/1024**3, index.nbytes/1024**3)
            
        except Exception as exc:
            print(ModColor.Str("Error accumulating weights from %s:"%msname), file=log)
            for line in traceback.format_exc().split("\n"):
                print(ModColor.Str("  "+line), file=log)
            msw["error"] = exc
            os.unlink(msw["cachepath"])
            msw.delete_item("weight")
            msw.delete_item("sgnweight")
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
                    index = self._uv_to_index_Cheap(ims, msw["uv"], weight, freqs, cell, npix, npixx, nbands, xymax)
                    grid = wg["grid"].reshape((wg["grid"].size,))
                    #weight /= grid[msw["index"]]
                    index[index>=len(grid)]=0
                    weight /= grid[index]
    #                import pdb; pdb.set_trace()

                np.save(msw["cachepath"], weight)
                np.save("%s.sgn.npy"%msw["cachepath"], msw["sgnweight"])
                msw.delete_item("weight")
                msw.delete_item("sgnweight")
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
                open(msw["cachepath"], 'w').truncate(0)
        except Exception as exc:
            print(ModColor.Str("Error accumulating weights from %s:"%msname), file=log)
            for line in traceback.format_exc().split("\n"):
                print(ModColor.Str("  "+line), file=log)
            msw["error"] = exc
            msw["success"] = False
            os.unlink(msw["cachepath"])

    def _CalcWeights_serial(self,iField=None):
        if iField is not None and self.VS.DicoFields is not None:
            FacetMachine=self.FacetMachine.LFM[iField]
        else:
            FacetMachine=self.FacetMachine
        FullImShape = FacetMachine.OutImShape
        # self.PaddedFacetShape = self.FacetMachine.PaddedGridShape
        FacetShape = FacetMachine.FacetShape
        CellSizeRad_x,CellSizeRad_y=FacetMachine.CellSizeRad
        
        self._weight_dict = shared_dict.create("VisWeights")
        # check for wmax in cache
        cache_keys = dict([(section, self.GD[section]) for section
                           in ("Data", "Selection", "Freq", "Image", "Weight")])
        wmax_path, wmax_valid = self.maincache.checkCache("wmax", cache_keys)
        uvmax_path, uvmax_valid = self.maincache.checkCache("uvmax", cache_keys)
        if wmax_valid:
            self._weight_dict["wmax"] = cPickle.load(open(wmax_path, "rb"))
        if uvmax_valid:
            self._weight_dict["uvmax"] = cPickle.load(open(uvmax_path, "rb"))
        # check cache first
        have_all_weights = wmax_valid and uvmax_valid
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
            print("all imaging weights, wmax, and uvmax are available in cache", file=log)
            return

        wmax = self._uvmax = 0
        # scan through MSs to determine uv-max
        for ims, ms in enumerate(self.ListMS):
            ms = self.ListMS[ims]
            max_freq = ms.ChanFreq.max()
            for ichunk in range(len(ms.getChunkRow0Row1())):
                print("scanning UVWs %d.%d" % (ims, ichunk), file=log)
                row0, row1 = ms.getChunkRow0Row1()[ichunk]
                nrows = row1 - row0
                if not nrows:
                    continue
                tab = ms.GiveMainTable()
                uvw = tab.getcol("UVW", row0, nrows)
                # rowflags = tab.getcol("FLAG_ROW", row0, nrows)
                
                # if all channels are flagged, flag whole row. Shape of flags becomes nrow
                flags = tab.getcol("FLAG", row0, nrows)
                d0, d1 = self.GD["Selection"]["UVRangeKm"]
                d0 = d0**2*1e6
                d1 = d1**2*1e6
                duv = (uvw[:,:2]**2).sum(1)  # u^2+v^2... and we already squared d0 and d1
                flags[(duv < d0) | (duv > d1),:,:] = True
                if ms._reverse_channel_order:
                    flags = flags[:,::-1,:]
                flags = flags.any(axis=2)
                rowflags = flags.all(axis=1)
                
                # max of |u|, |v| in wavelengths
                if not rowflags.all():
                    uvmax_wavelengths = abs(uvw[~rowflags, :2]).max() * max_freq / _cc
                    self._uvmax = max(self._uvmax, uvmax_wavelengths)
                    wmax = max(wmax, abs(uvw[~rowflags, 2]).max())
                    

        if MPIManager.useMPI:
            self._uvmax = MPIManager.COMM_WORLD.allreduce(self._uvmax, MPIManager.MAX)
            wmax = MPIManager.COMM_WORLD.allreduce(wmax, MPIManager.MAX)
            
        # setup uv-grid for non-natural weights
        if self.Weighting != "natural":
            self._weight_grid = shared_dict.create("VisWeights.Grid")
            nch, npol, npixIm_x, npixIm_y = FullImShape
            FOV_x = CellSizeRad_x * npixIm_x
            FOV_y = CellSizeRad_y * npixIm_y
            nbands = self.NFreqBands
            cell_u = 1. / (self.Super * FOV_x)
            cell_v = 1. / (self.Super * FOV_y)
            if self.MFSWeighting or self.NFreqBands < 2:
                nbands = 1
                print("initializing weighting grid for single band (or MFS weighting)", file=log)
            else:
                print("initializing weighting grids for %d bands" % nbands, file=log)
            # find max grid extent by considering _unflagged_ UVs
            xymax = int(math.floor(self._uvmax / np.min([cell_u,cell_v]))) + 1
            # grid will be from [-xymax,xymax] in U and [0,xymax] in V
            npixx = xymax * 2 + 1
            npixy = xymax + 1
            npix = npixx * npixy
            print("Calculating imaging weights on an [%i,%i]x%i grid with cellsize [%g,%g]" % (npixx, npixy, nbands, cell_u,cell_v), file=log)
            self._weight_grid.addSharedArray("grid", (nbands, npix), np.float64)
        else:
            nbands = self.NFreqBands
            nch, npol, npixIm_x, npixIm_y = FullImShape
            FOV_x = CellSizeRad_x * npixIm_x
            FOV_y = CellSizeRad_y * npixIm_y
            cell_u = 1. / (self.Super * FOV_x)
            cell_v = 1. / (self.Super * FOV_y)
            xymax = int(math.floor(self._uvmax / np.min([cell_u,cell_v]))) + 1
            npixx = xymax * 2 + 1
            npixy = xymax + 1
            npix = npixx * npixy
        cell=np.array([cell_u,cell_v])
        # scan through MSs one by one
        for ims, ms in enumerate(self.ListMS):
            msweights = self._weight_dict[ims]
            for ichunk in range(len(ms.getChunkRow0Row1())):
                msw = msweights[ichunk]
                print("loading weights %d.%d"%(ims, ichunk), file=log)
                self._loadWeights_handler(msw, ims, ichunk, self._ignore_vis_weights)

                # if nothing in MS, handler will not return a "weight" field. Mark this chunk as null then, and truncate the cache
                msw["null"] = "weight" not in msw
                if "error" in msw:
                    raise RuntimeError("weights computation failed for one or more MSs")
                if "weight" not in msw:
                    open(msw["cachepath"], 'wb').truncate(0)
                    continue

                # in Natural mode, we're done: dump weights out
                if self.Weighting == "natural":
                    self._CalcSigmoidTaper(ims, ms, ichunk)
                    self._finalizeWeights_handler(None, msw,
                                                  ims, ichunk, ms.ChanFreq, cell, 
                                                  npix, npixx, nbands, xymax)
                # else accumulate onto uv grid
                else:
                    self._accumulateWeights_handler(self._weight_grid, msw,
                                                    ims, ichunk, ms.ChanFreq, cell,
                                                    npix, npixx, nbands, xymax)
                    #np.savez("msw.new.npz",**msw)
                                    
        if MPIManager.useMPI:
            self._weight_grid["grid"] = MPIManager.COMM_WORLD.allreduce(self._weight_grid["grid"], MPIManager.SUM)
            
        # save wmax to cache
        cPickle.dump(wmax, open(wmax_path, "wb"))
        self.maincache.saveCache("wmax")
        self._weight_dict["wmax"] = wmax
        # LB - Need to cache this to set scales in ScaleMachine
        cPickle.dump(self._uvmax, open(uvmax_path, "wb"))
        self.maincache.saveCache("uvmax")
        self._weight_dict["uvmax"] = self._uvmax
        print("overall max W is %.2f meters"%wmax, file=log)
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
                    #np.savez("grids.new.npz",grid0=grid0,numeratorSqrt=numeratorSqrt,grid1=grid1,avgW=avgW,sSq=sSq)
                    grid1[...] = 1 + grid1 * sSq

                    
            # rescan through MSs one by one to re-adjust the weights
            for ims, ms in enumerate(self.ListMS):
                msweights = self._weight_dict[ims]
                for ichunk in range(len(ms.getChunkRow0Row1())):
                    msw = msweights[ichunk]
                    if msw["null"]:
                        print("skipping weights %d.%d (null)" % (ims, ichunk), file=log)
                        open(msw["cachepath"], 'w').truncate(0)
                        continue
                    print("reloading weights %d.%d" % (ims, ichunk), file=log)
                    self._loadWeights_handler(msw, ims, ichunk, self._ignore_vis_weights)
                    self._CalcSigmoidTaper(ims, ms, ichunk)
                    self._finalizeWeights_handler(self._weight_grid, msw,
                                                      ims, ichunk, ms.ChanFreq, cell, npix, npixx, nbands, xymax)

            if self._weight_grid is not None:
                self._weight_grid.delete()

        # free memory
        for ims, ms in enumerate(self.ListMS):
            msweights = self._weight_dict[ims]
            for ichunk in range(len(ms.getChunkRow0Row1())):
                msw = msweights[ichunk]
                # delete to save memory
                if self.Weighting != "natural":
                    for field in "weight", "uv", "flags", "index":
                        if field in msw:
                            msw.delete_item(field)

        # mark caches as valid
        for ims, ms in enumerate(self.ListMS):
            for ichunk, (row0, row1) in enumerate(ms.getChunkRow0Row1()):
                ms.getChunkCache(row0, row1).saveCache("ImagingWeights.npy")
        
