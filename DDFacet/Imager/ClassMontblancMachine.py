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

import montblanc
import montblanc.util as mbu
import montblanc.impl.rime.tensorflow.ms.ms_manager as MS

from montblanc.impl.rime.tensorflow.sources import (SourceProvider,
    FitsBeamSourceProvider)
from montblanc.impl.rime.tensorflow.sinks import SinkProvider
from DDFacet.Data.ClassStokes import StokesTypes
import logging

from DDFacet.Other import MyLogger
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Data.ClassStokes import ClassStokes

from astropy import wcs as pywcs

log=MyLogger.getLogger("ClassMontblancMachine")

DEBUG = False

class ClassMontblancMachine(object):
    def __init__(self, GD, npix, cell_size_rad, polarization_type="linear"):
        shndlrs = filter(lambda x: isinstance(x, logging.StreamHandler),
                         montblanc.log.handlers)
        montblanc.log.propagate = False
        log_levels = {"NOTSET": logging.NOTSET,
                      "DEBUG": logging.DEBUG,
                      "INFO": logging.INFO,
                      "WARNING": logging.WARNING,
                      "ERROR": logging.ERROR,
                      "CRITICAL": logging.CRITICAL}
        for s in shndlrs:
            s.level = log_levels.get(GD["Montblanc"]["LogLevel"], logging.WARNING)

        apnd = "a" if GD["Log"]["Append"] else "w"
        lgname = GD["Output"]["Name"] + ".montblanc.log" \
                if GD["Montblanc"]["LogFile"] is None else GD["Montblanc"]["LogFile"]
        fhndlr = logging.FileHandler(lgname, mode=apnd)
        fhndlr.level = logging.DEBUG
        montblanc.log.addHandler(fhndlr)

        # configure solver
        self._slvr_cfg = slvr_cfg = montblanc.rime_solver_cfg(
            data_source="default",
            polarisation_type=polarization_type,
            mem_budget=int(np.ceil(GD["Montblanc"]["MemoryBudget"]*1024*1024*1024)),
            dtype=GD["Montblanc"]["SolverDType"],
            auto_correlations=True,
            version=GD["Montblanc"]["DriverVersion"]
        )

        self._solver = montblanc.rime_solver(slvr_cfg)

        self._cell_size_rad = cell_size_rad
        self._npix = npix
        self._mgr = DataDictionaryManager(polarization_type)

        # Configure the Beam upfront
        if GD["Beam"]["Model"] == "FITS":
            fits_file_spec = GD["Beam"]["FITSFile"]
            l_axis = GD["Beam"]["FITSLAxis"]
            m_axis = GD["Beam"]["FITSMAxis"]
            self._beam_prov = FitsBeamSourceProvider(fits_file_spec,
                l_axis=l_axis, m_axis=m_axis)
        else:
            self._beam_prov = None

    def getChunk(self, data, residuals, model, MS):
        # Configure the data dictionary manager
        self._mgr.cfg_from_data_dict(data, model, residuals, MS, self._npix, self._cell_size_rad)

        # Configure source providers
        source_provs = []  if self._beam_prov is None else [self._beam_prov]
        source_provs.append(DDFacetSourceProvider(self._mgr))

        # Configure sink providers
        sink_provs = [DDFacetSinkProvider(self._mgr)]

        self._solver.solve(source_providers=source_provs, sink_providers=sink_provs)

    def close(self):
        self._solver.close()

class DataDictionaryManager(object):
    """
    Given a dictionary of MS arrays, infers the number of:
        - timesteps
        - baselines
        - antenna
        - channels

    ASSUMPTIONS:
        1. We're dealing with a single band
        2. We're dealing with a single field
        3. We're dealing with a single observation
    """
    def __init__(self, solver_polarization_type="linear"):
        self._solver_polarization_type = solver_polarization_type
        pass

    def _transform(self, c):
        coord = np.deg2rad(self._wcs.wcs_pix2world(np.asarray([c]), 1))
        delta_ang = coord[0] - self._phase_dir

        if DEBUG:
            montblanc.log.debug("WCS src coordinates [deg] '{c}'".format(c=np.rad2deg(coord)))

        # assume SIN Projection
        l = np.cos(coord[0, 1])*np.sin(delta_ang[0])
        m = np.sin(coord[0, 1])*np.cos(self._phase_dir[1]) - \
            np.cos(coord[0, 1])*np.sin(self._phase_dir[1])*np.cos(delta_ang[0])

        return (l, m)

    def cfg_from_data_dict(self, data, models, residuals, MS, npix, cell_size_rad):
        time, uvw, antenna1, antenna2, flag = (data[i] for i in  ('times', 'uvw', 'A0', 'A1', 'flags'))

        self._npix = npix
        self._cell_size_rad = cell_size_rad

        # Get point and gaussian sources
        self._point_sources = []
        self._gaussian_sources = []

        for model in models:
            if model[0] == 'Delta':
                self._point_sources.append(model)
            elif model[0] == 'Gaussian':
                self._gaussian_sources.append(model)
            else:
                raise ValueError("Montblanc does not predict "
                    "'{mt}' model types.".format(mt=model[0]))

        self._npsrc = npsrc = len(self._point_sources)
        self._ngsrc = ngsrc = len(self._gaussian_sources)

        montblanc.log.info("{n} point sources".format(n=npsrc))
        montblanc.log.info("{n} gaussian sources".format(n=ngsrc))

        # Extract the antenna positions and
        # phase direction from the measurement set
        self._antenna_positions = MS.StationPos
        self._phase_dir = np.array((MS.rarad, MS.decrad))
        self._data_feed_labels = ClassStokes(MS.CorrelationIds, ["I"]).AvailableCorrelationProducts()

        # Extract the required prediction feeds from the MS
        if set(self._data_feed_labels) <= set(["XX", "XY", "YX", "YY"]):
            self._montblanc_feed_labels = ["XX", "XY", "YX", "YY"]
            if self._solver_polarization_type != "linear":
                raise RuntimeError("Solver configured for linear polarizations. "
                                   "Does not support combining measurement sets of different feed types")
        elif set(self._data_feed_labels) <= set(["RR", "RL", "LR", "LL"]):
            self._montblanc_feed_labels = ["RR", "RL", "LR", "LL"]
            if self._solver_polarization_type != "circular":
                raise RuntimeError("Solver configured for circular polarizations. "
                                   "Does not support combining measurement sets of different feed types")
        else:
            raise RuntimeError("Montblanc only supports linear or circular feed measurements.")

        self._predict_feed_map = [self._montblanc_feed_labels.index(x) for x in self._data_feed_labels]
        montblanc.log.info("Montblanc to MS feed mapping: %s" % ",".join([str(x) for x in self._predict_feed_map]))

        montblanc.log.info("Phase centre of {pc}".format(pc=np.rad2deg(self._phase_dir)))
        self._frequency = MS.ChanFreq
        self._ref_frequency = MS.reffreq
        self._nchan = nchan = self._frequency.size

        # Merge antenna ID's and count their frequencies
        ants = np.concatenate((antenna1, antenna2))
        unique_ants = np.unique(ants)
        ant_counts = np.bincount(ants)
        self._na = na = unique_ants.size

        # can compute the time indexes using a scan operator
        # assuming the dataset is ordered and the time column
        # contains the integration centroid of the correlator dumps
        # note: time first, then antenna1 slow varying then antenna2
        # fast varying
        self._sort_index = np.lexsort(np.array((time, antenna1, antenna2))[::-1])

        tfilter = np.zeros(time[self._sort_index].shape, dtype=np.bool)
        tfilter[1:] = time[self._sort_index][1:] != time[self._sort_index][:-1]
        self._tindx = np.cumsum(tfilter)
        self._ntime = self._tindx[-1] + 1

        # include the autocorrelations to safely pad the array to a maximum possible size
        self._nbl = self._na * (self._na - 1) // 2 + self._na
        self._blindx = self.baseline_index(antenna1[self._sort_index],
                                           antenna2[self._sort_index],
                                           self._na)

        # Sanity check antenna and row dimensions
        if self._antenna_positions.shape[0] < na:
            raise ValueError("Number of antenna positions {aps} "
                "is less than the number of antenna '{na}' "
                "found in antenna1/antenna2".format(
                    aps=self._antenna_positions.shape, na=na))
        self._nrow = self._nbl * self._ntime
        if self._nrow < uvw.shape[0]:
            raise ValueError("Number of rows in input chunk exceeds "
                             "nbl * ntime. Please ensure that only one field, "
                             "one observation and one data descriptor is present in the chunk you're "
                             "trying to predict.")

        # construct a mask to indicate values sparse matrix
        self._datamask = np.zeros((self._ntime, self._nbl), dtype=np.bool)
        self._datamask[self._tindx, self._blindx] = True
        self._datamask = self._datamask.reshape((self._nrow))

        # padded row numbers of the sparce data matrix
        self._sparceindx = np.cumsum(self._datamask) - 1
        self._sparceindx

        # pad the antenna array
        self._padded_a1 = np.empty((self._nbl))
        self._padded_a2 = np.empty((self._nbl))
        lmat = np.triu((np.cumsum(np.arange(self._na)[None, :] >=
                                  np.arange(self._na)[:, None]) - 1).reshape([self._na, self._na]))
        for bl in xrange(self._nbl):
            blants = np.argwhere(lmat == bl)[0]
            self._padded_a1[bl] = blants[0]
            self._padded_a2[bl] = blants[1]
        self._padded_a1 = np.tile(self._padded_a1, self._ntime)
        self._padded_a2 = np.tile(self._padded_a2, self._ntime)
        assert np.all(self._padded_a1[self._datamask] == antenna1[self._sort_index])
        assert np.all(self._padded_a2[self._datamask] == antenna2[self._sort_index])

        # pad the time array
        self._padded_time = np.unique(time).repeat(self._nbl)
        assert np.all(self._padded_time[self._datamask] == time[self._sort_index])

        # Pad the uvw array to contain nbl * ntime entries (including the autocorrs)
        # with zeros where baselines may be missing
        self._residuals = residuals.view()

        self._padded_uvw = np.zeros((self._ntime, self._nbl, 3), dtype=uvw.dtype)
        self._padded_uvw[self._tindx, self._blindx, :] = uvw[self._sort_index, :]
        self._padded_uvw = self._padded_uvw.reshape((self._nrow, 3))
        assert np.all(self._padded_uvw[self._datamask] == uvw[self._sort_index])

        self._flag = flag

        # progress...
        self._numtotal = None
        self._numcomplete = 0
        self._pBAR = ProgressBar(Title="  montblanc predict")
        self.render_progressbar()

        # Initialize WCS frame
        wcs = pywcs.WCS(naxis=2)
        # note half a pixel will correspond to even sized image projection poles
        montblanc.log.debug("NPIX: %d" % self._npix)
        assert self._npix % 2 == 1, "Currently only supports odd-sized maps"
        l0m0 = [self._npix // 2, self._npix // 2]
        wcs.wcs.crpix = l0m0
        # remember that the WCS frame uses degrees
        wcs.wcs.cdelt = [np.rad2deg(self._cell_size_rad),
                         np.rad2deg(self._cell_size_rad)]
        # assume SIN image projection
        wcs.wcs.ctype = ["RA---SIN","DEC--SIN"]

        wcs.wcs.crval = [np.rad2deg(self._phase_dir[0]),
                         np.rad2deg(self._phase_dir[1])]
        self._wcs = wcs

        # Finally output some info
        montblanc.log.info('Chunk contains:' % unique_ants)
        montblanc.log.info('\tntime: %s' % self._ntime)
        montblanc.log.info('\tnbl: %s' % self._nbl)
        montblanc.log.info('\tnchan: %s' % self._nchan)
        montblanc.log.info('\tna: %s' % self._na)

    def baseline_index(self, a1, a2, no_antennae):
        """
         Computes unique index of a baseline given antenna 1 and antenna 2
         (zero indexed) as input. The arrays may or may not contain
         auto-correlations.

         There is a quadratic series expression relating a1 and a2
         to a unique baseline index(can be found by the double difference
                                    method)

         Let slow_varying_index be S = min(a1, a2). The goal is to find
         the number of fast varying terms. As the slow
         varying terms increase these get fewer and fewer, because
         we only consider unique baselines and not the conjugate
         baselines)
         B = (-S ^ 2 + 2 * S *  # Ant + S) / 2 + diff between the
         slowest and fastest varying antenna

         :param a1: array of ANTENNA_1 ids
         :param a2: array of ANTENNA_2 ids
         :param no_antennae: number of antennae in the array
         :return: array of baseline ids
        """
        if a1.shape != a2.shape:
            raise ValueError("a1 and a2 must have the same shape!")

        slow_index = np.min(np.array([a1, a2]), axis=0)

        return (slow_index * (-slow_index + (2 * no_antennae + 1))) // 2 + \
                np.abs(a1 - a2)

    def updated_dimensions(self):
        return [('ntime', self._ntime), ('nbl', self._nbl),
            ('na', self._na), ('nbands', 1), ('nchan', self._nchan),
            ('npsrc', self._npsrc), ('ngsrc', self._ngsrc)]

    def render_progressbar(self):
        if self._numtotal is not None:
            self._pBAR.render(self._numcomplete, self._numtotal)


class DDFacetSourceProvider(SourceProvider):
    def __init__(self, manager):
        self._manager = manager

    def name(self):
        return "DDFacet Source Provider"

    def update_nchunks(self, context):
        # @sjperkins please expose the solver's iterdims
        # https://github.com/ska-sa/montblanc/blob/master/montblanc/impl/rime/tensorflow/RimeSolver.py#L139
        dim_names = ["nbl", "ntime"]
        global_sizes = context.dim_global_size(*dim_names)
        ext_sizes = context.dim_extent_size(*dim_names)
        ntotal = reduce(lambda x, y: x * y if y != 0 else 1,
                        global_sizes)
        self._manager._numtotal = max(ntotal, self._manager._numtotal)
        self._manager.render_progressbar()

    def point_lm(self, context):
        self.update_nchunks(context)
        (lp, up) = context.dim_extents('npsrc')

        mgr = self._manager
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        pt_slice = mgr._point_sources[lp:up]
        # Assign coordinate tuples
        sky_coords_rad = np.array([mgr._transform(p[1]) for p in pt_slice],
            dtype=context.dtype)

        return sky_coords_rad.reshape(context.shape)

    def point_stokes(self, context):
        self.update_nchunks(context)
        (lp, up) = context.dim_extents('npsrc')
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        pt_slice = self._manager._point_sources[lp:up]
        # Assign I stokes, zero everything else
        stokes = np.zeros(context.shape, context.dtype)
        stokes[:,:,0] = np.array([p[2].reshape((1)) for p in pt_slice])
        if DEBUG:
            montblanc.log.debug("Point stokes parameters {ps}".format(
                ps=stokes[:16,0,0]))

        return stokes

    def point_alpha(self, context):
        self.update_nchunks(context)
        (lp, up) = context.dim_extents('npsrc')
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        pt_slice = self._manager._point_sources[lp:up]
        # Assign alpha, broadcasting into the time dimension
        alpha = np.zeros(context.shape, context.dtype)
        alpha[:,:] = np.array([p[4] for p in pt_slice])[:,np.newaxis]
        if DEBUG:
            montblanc.log.debug("Alpha parameters {ps}".format(
                ps=alpha[:16,0]))
        return alpha

    def gaussian_lm(self, context):
        self.update_nchunks(context)
        (lg, ug) = context.dim_extents('ngsrc')

        mgr = self._manager
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        g_slice = mgr._gaussian_sources[lg:ug]
        # Assign coordinate tuples
        sky_coords_rad = np.array([mgr._transform(g[1]) for g in g_slice],
            dtype=context.dtype)

        return sky_coords_rad.reshape(context.shape)

    def gaussian_stokes(self, context):
        self.update_nchunks(context)
        (lg, ug) = context.dim_extents('ngsrc')
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        g_slice = self._manager._gaussian_sources[lg:ug]
        # Assign I stokes, zero everything else
        stokes = np.zeros(context.shape, context.dtype)
        stokes[:,:,0] = np.array([g[2] for g in g_slice])
        return stokes

    def gaussian_alpha(self, context):
        self.update_nchunks(context)
        (lg, ug) = context.dim_extents('ngsrc')
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        g_slice = self._manager._gaussian_sources[lg:ug]
        # Assign alpha, broadcasting into the time dimension
        alpha = np.empty(context.shape, context.dtype)
        alpha[:,:] = np.array([g[4] for g in g_slice])[:,np.newaxis]
        return alpha

    def gaussian_shape(self, context):
        self.update_nchunks(context)
        (lg, ug) = context.dim_extents('ngsrc')
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        g_slice = self._manager._gaussian_sources[lg:ug]

        # Extract major, minor and theta parameters
        major, minor, theta = np.array([g[5] for g in g_slice]).T

        # Convert to lproj, mproj, ratio system
        gauss_shape = np.empty(context.shape, context.dtype)
        gauss_shape[0,:] = major * np.sin(theta)
        gauss_shape[1,:] = minor * np.cos(theta)
        major[major == 0.0] = 1.0
        gauss_shape[2,:] = minor / major

        return gauss_shape

    def frequency(self, context):
        self.update_nchunks(context)
        lc, uc = context.dim_extents('nchan')

        return self._manager._frequency[lc:uc]

    def parallactic_angles(self, context):
        self.update_nchunks(context)
        # Time extents
        (lt, ut) = context.dim_extents('ntime')
        mgr = self._manager

        return mbu.parallactic_angles(mgr._padded_time[lt:ut],
            mgr._antenna_positions,
            mgr._phase_dir).astype(context.dtype)

    def model_vis(self, context):
        self.update_nchunks(context)

        return np.zeros(shape=context.shape, dtype=context.dtype)

    def uvw(self, context):
        self.update_nchunks(context)

        (lt, ut) = context.dim_extents('ntime')
        na, nbl = context.dim_global_size('na', 'nbl')
        ddf_uvw = self._manager._padded_uvw

        # Create per antenna UVW coordinates.
        # u_01 = u_1 - u_0
        # u_02 = u_2 - u_0
        # ...
        # u_0N = u_N - U_0
        # where N = na - 1

        # Choosing u_0 = 0 we have:
        # u_1 = u_01
        # u_2 = u_02
        # ...
        # u_N = u_0N

        # Then, other baseline values can be derived as
        # u_21 = u_1 - u_2

        # Allocate space for per-antenna UVW, zeroing first antenna at each timestep
        ant_uvw = np.zeros(shape=context.shape, dtype=context.dtype)
        ant_uvw[:, 0, :] = 0

        # Read in uvw[1:na] row at each timestep
        for ti, t in enumerate(xrange(lt, ut)):
            lrow = t*nbl + 1 # skip autocorr (padding ensures it is there though)
            urow = lrow + na - 1
            ant_uvw[ti, 1:na, :] = ddf_uvw[lrow:urow, :]

        return ant_uvw

    def antenna1(self, context):
        self.update_nchunks(context)

        lrow, urow = MS.row_extents(context)
        view = self._manager._padded_a1[lrow:urow]
        return view.reshape(context.shape).astype(context.dtype)

    def antenna2(self, context):
        self.update_nchunks(context)

        lrow, urow = MS.row_extents(context)
        view = self._manager._padded_a2[lrow:urow]
        return view.reshape(context.shape).astype(context.dtype)

    def updated_dimensions(self):
        """ Defer to the manager """
        return self._manager.updated_dimensions()

class DDFacetSinkProvider(SinkProvider):
    def __init__(self, manager):
        self._manager = manager

    def name(self):
        return "DDFacet Sink Provider"

    def model_vis(self, context):
        lrow, urow = MS.row_extents(context)

        # Get the sparce representation (sorted) selection for this chunk
        datamask = self._manager._datamask[lrow:urow]
        sparceindx = np.unique(self._manager._sparceindx[lrow:urow][datamask])
        sort_indx = self._manager._sort_index[sparceindx]
        nrow_sparce = sparceindx.size
        nrow = urow - lrow
        nchan = self._manager._nchan
        ncorr_mb = 4 #montblanc always predict four correlations
        sparce_flags = self._manager._flag[sort_indx, :, :]

        # Compute residuals
        chunk_nbl, chunk_ntime, chunk_nchan, chunk_ncorr = context.data.shape
        chunk_nrow = chunk_nbl * chunk_ntime
        datamask_tile = datamask.repeat(chunk_nchan * chunk_ncorr).reshape((chunk_nrow, chunk_nchan, chunk_ncorr))
        mod = context.data.reshape((chunk_nrow, chunk_nchan, chunk_ncorr))
        mod_sel = mod[datamask_tile].reshape(nrow_sparce, chunk_nchan, chunk_ncorr)

        def __print_model_stats(mod_sel):
            mod_sel_pow = np.abs(mod_sel)
            montblanc.log.debug("\t MEAN: %s" % ",".join(
                ["%.3f" % x for x in np.nanmean(np.nanmean(mod_sel_pow,
                                                           axis=0),
                                                axis=0)]))
            montblanc.log.debug("\t MAX: %s" % ",".join(
                ["%.3f" % x for x in np.nanmax(np.nanmax(mod_sel_pow,
                                                         axis=0),
                                               axis=0)]))
            montblanc.log.debug("\t MIN: %s" % ",".join(
                ["%.3f" % x for x in np.nanmin(np.nanmin(mod_sel_pow,
                                                         axis=0),
                                               axis=0)]))
            montblanc.log.debug("\t STD: %s" % ",".join(
                ["%.3f" % x for x in np.nanstd(np.nanstd(mod_sel_pow,
                                                         axis=0),
                                               axis=0)]))
        if DEBUG:
            montblanc.log.debug("Model stats:")
            __print_model_stats(mod_sel)

        def __print_residual_stats(prev, sparce_flags):
            prev_fl = np.abs(prev)
            prev_fl[sparce_flags] = np.nan
            montblanc.log.debug("\t MEAN: %s" % ",".join(
                ["%.3f" % x for x in np.nanmean(np.nanmean(prev_fl,
                                                           axis=0),
                                                axis=0)]))
            montblanc.log.debug("\t MAX: %s" % ",".join(
                ["%.3f" % x for x in np.nanmax(np.nanmax(prev_fl,
                                                         axis=0),
                                               axis=0)]))
            montblanc.log.debug("\t MIN: %s" % ",".join(
                ["%.3f" % x for x in np.nanmin(np.nanmin(prev_fl,
                                                         axis=0),
                                               axis=0)]))
            montblanc.log.debug("\t STD: %s" % ",".join(
                ["%.3f" % x for x in np.nanstd(np.nanstd(prev_fl,
                                                         axis=0),
                                               axis=0)]))
        if DEBUG:
            montblanc.log.debug("Old residual stats:")
            __print_residual_stats(self._manager._residuals[sort_indx, :, :], sparce_flags)

        if self._manager._data_feed_labels == self._manager._montblanc_feed_labels:
            self._manager._residuals[sort_indx, :, :] -= mod_sel
        else:
            for ci, c in enumerate(self._manager._data_feed_labels):
                self._manager._residuals[sort_indx, :, ci] -= mod_sel[:, :, self._manager._predict_feed_map[ci]]

        if DEBUG:
            montblanc.log.debug("New residual stats:")
            __print_residual_stats(self._manager._residuals[sort_indx, :, :], sparce_flags)

        # update progress
        self._manager._numcomplete += nrow

        self._manager.render_progressbar()

class DDFacetSinkPredict(SinkProvider):
    def __init__(self, manager):
        self._manager = manager

    def name(self):
        return "DDFacet Sink Predict"

    def model_vis(self, context):
        lrow, urow = MS.row_extents(context)
        view = self._manager._residuals[lrow:urow,:,:]
        view[:] = context.data.reshape(view.shape)

def _rle(array):
    """
    Return Run-length encoding (length, starts, values) of supplied array
    """
    assert array.ndim == 1

    # Number of elements
    n = array.shape[0]
    # Indicate differing consecutive elements
    diffs = np.array(array[1:] != array[:-1])
    # Get indices of differing elements, appending
    # the last by default
    idx = np.append(np.where(diffs), n-1)
    length = np.diff(np.append(-1, idx))
    starts = np.cumsum(np.append(0, length))[:-1]

    return (length, starts, array[idx])

if __name__ == '__main__':
    test()
