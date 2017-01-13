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

from montblanc.config import RimeSolverConfig as Options
from montblanc.impl.rime.tensorflow.sources import (SourceProvider,
    FitsBeamSourceProvider)
from montblanc.impl.rime.tensorflow.sinks import SinkProvider

class ClassMontblancMachine(object):
    def __init__(self, GD, npix, cell_size_rad):
        self._slvr_cfg = slvr_cfg = montblanc.rime_solver_cfg(
            data_source=Options.DATA_SOURCE_DEFAULT,
            tf_server_target=GD["Montblanc"]["TensorflowServerTarget"],
            mem_budget=2*1024*1024*1024,
            dtype='double',
            auto_correlations=False,
            version='tf')

        self._solver = montblanc.rime_solver(slvr_cfg)

        self._cell_size_rad = cell_size_rad
        self._npix = npix
        self._mgr = DataDictionaryManager()

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
    """
    def __init__(self):
        pass

    def cfg_from_data_dict(self, data, models, residuals, MS, npix, cell_size_rad):
        time, uvw, antenna1, antenna2 = (data[i] for i in  ('times', 'uvw', 'A0', 'A1'))

        # Transform pixel to lm coordinates
        l0m0 = np.floor((npix / 2.0, npix / 2.0))
        self._transform = lambda c: (np.asarray(c) - l0m0)*cell_size_rad

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

        # Store the data dictionary
        self._data = data

        self._residuals = residuals

        # Get the indices that sort the time array
        time_idx = np.argsort(time)
        # Get baselines per timestep, the starting positions
        # of each timestep and the times at this timestep
        rle = (bls_per_tstep, tstep_starts, tstep_times) = _rle(time[time_idx])

        # Extract the antenna positions and
        # phase direction from the measurement set
        self._antenna_positions = MS.StationPos
        self._phase_dir = np.array((MS.rarad, MS.decrad))

        montblanc.log.info("Phase centre of {pc}".format(pc=self._phase_dir))

        self._frequency = MS.ChanFreq
        self._ref_frequency = MS.reffreq

        self._nchan = nchan = self._frequency.size

        self._bls_per_tstep = bls_per_tstep
        self._tstep_starts = tstep_starts
        self._tstep_times = tstep_times

        # Number of timesteps
        self._ntime = ntime = bls_per_tstep.shape[0]

        # Take the maximum baseline count per timestep as
        # the number of baselines
        max_bl_timestep_idx = np.argmax(bls_per_tstep)
        self._nbl = nbl = bls_per_tstep[max_bl_timestep_idx]

        # for bls, tstart, time in np.array(rle).T:
        #     montblanc.log.info('ANT1\n{a}'.format(a=antenna1[tstart:tstart+bls]))
        #     montblanc.log.info('ANT2\n{a}'.format(a=antenna2[tstart:tstart+bls]))

        # Merge antenna ID's and count their frequencies
        ants = np.concatenate((antenna1, antenna2))
        unique_ants = np.unique(ants)
        ant_counts = np.bincount(ants)
        self._na = na = unique_ants.size

        # Sanity check antenna dimensions
        if self._antenna_positions.shape[0] < na:
            raise ValueError("Number of antenna positions {aps} "
                "is less than the number of antenna '{na}' "
                "found in antenna1/antenna2".format(
                    aps=self._antenna_positions.shape, na=na))

        # Index of most commonly occurring antenna
        max_ant_count_idx = np.argmax(ant_counts)

        #print 'rle', rle
        montblanc.log.info('ants %s' % unique_ants)
        montblanc.log.info('ant_counts %s' % ant_counts)
        montblanc.log.info('ntime %s' % ntime)
        montblanc.log.info('nbl %s' % nbl)
        montblanc.log.info('nchan %s' % nchan)
        montblanc.log.info('na %s' % na)

    def updated_dimensions(self):
        return [('ntime', self._ntime), ('nbl', self._nbl),
            ('na', self._na), ('nbands', 1), ('nchan', self._nchan),
            ('npsrc', self._npsrc), ('ngsrc', self._ngsrc)]

class DDFacetSourceProvider(SourceProvider):
    def __init__(self, manager):
        self._manager = manager

    def name(self):
        return "DDFacet Source Provider"


    def point_lm(self, context):
        (lp, up) = context.dim_extents('npsrc')
        mgr = self._manager
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        pt_slice = mgr._point_sources[lp:up]
        # Assign coordinate tuples
        sky_coords_rad = np.array([mgr._transform(p[1]) for p in pt_slice],
            dtype=context.dtype)

        montblanc.log.debug("Radian Coordinates '{c}'".format(
            c=sky_coords_rad[:16,:]))

        return sky_coords_rad.reshape(context.shape)

    def point_stokes(self, context):
        (lp, up) = context.dim_extents('npsrc')
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        pt_slice = self._manager._point_sources[lp:up]
        # Assign I stokes, zero everything else
        stokes = np.zeros(context.shape, context.dtype)
        stokes[:,:,0] = np.array([p[2] for p in pt_slice])
        montblanc.log.debug("Point stokes parameters {ps}".format(
            ps=stokes[:16,0,0]))

        return stokes

    def point_alpha(self, context):
        (lp, up) = context.dim_extents('npsrc')
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        pt_slice = self._manager._point_sources[lp:up]
        # Assign alpha, broadcasting into the time dimension
        alpha = np.zeros(context.shape, context.dtype)
        alpha[:,:] = np.array([p[4] for p in pt_slice])[:,np.newaxis]
        montblanc.log.debug("Alpha parameters {ps}".format(
            ps=alpha[:16,0]))
        return alpha

    def gaussian_lm(self, context):
        (lg, ug) = context.dim_extents('ngsrc')
        mgr = self._manager
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        g_slice = mgr._gaussian_sources[lg:ug]
        # Assign coordinate tuples
        sky_coords_rad = np.array([mgr._transform(g[1]) for g in g_slice],
            dtype=context.dtype)

        montblanc.log.debug("Radian Coordinates '{c}'".format(
            c=sky_coords_rad[:16,:]))

        return sky_coords_rad.reshape(context.shape)

    def gaussian_stokes(self, context):
        (lg, ug) = context.dim_extents('ngsrc')
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        g_slice = self._manager._gaussian_sources[lg:ug]
        # Assign I stokes, zero everything else
        stokes = np.zeros(context.shape, context.dtype)
        stokes[:,:,0] = np.array([g[2] for g in g_slice])
        return stokes

    def gaussian_alpha(self, context):
        (lg, ug) = context.dim_extents('ngsrc')
        # ModelType, lm coordinate, I flux, ref_frequency, Alpha, Model Parameters
        g_slice = self._manager._gaussian_sources[lg:ug]
        # Assign alpha, broadcasting into the time dimension
        alpha = np.empty(context.shape, context.dtype)
        alpha[:,:] = np.array([g[4] for g in g_slice])[:,np.newaxis]
        return alpha

    def gaussian_shape(self, context):
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
        lc, uc = context.dim_extents('nchan')
        return self._manager._frequency[lc:uc]

    def ref_frequency(self, context):
        # Assumes a single band
        return np.full(context.shape,
            self._manager._ref_frequency,
            dtype=context.dtype)

    def parallactic_angles(self, context):
        # Time extents
        (lt, ut) = context.dim_extents('ntime')
        mgr = self._manager

        return mbu.parallactic_angles(mgr._phase_dir,
            mgr._antenna_positions,
            mgr._tstep_times[lt:ut]).astype(context.dtype)

    def model_vis(self, context):
        return np.zeros(shape=context.shape, dtype=context.dtype)

    def uvw(self, context):
        (lt, ut) = context.dim_extents('ntime')
        na, nbl = context.dim_global_size('na', 'nbl')
        ddf_uvw = self._manager._data['uvw']

        # Create per antenna UVW coordinates.
        # u_01 = u_1 - u_0
        # u_02 = u_2 - u_0
        # ...
        # u_0N = u_N - U_0
        # where N = na - 1.

        # Choosing u_0 = 0 we have:
        # u_1 = u_01
        # u_2 = u_02
        # ...
        # u_N = u_0N

        # Then, other baseline values can be derived as
        # u_21 = u_1 - u_2

        # Allocate space for per-antenna UVW, zeroing antenna 0 at each timestep
        ant_uvw = np.empty(shape=context.shape, dtype=context.dtype)
        ant_uvw[:,0,:] = 0

        # Read in uvw[1:na] row at each timestep
        for ti, t in enumerate(xrange(lt, ut)):
            lrow = t*nbl
            urow = lrow + na - 1
            ant_uvw[ti,1:na,:] = ddf_uvw[lrow:urow,:]

        return ant_uvw

    def antenna1(self, context):
        lrow, urow = MS.row_extents(context)
        view = self._manager._data['A0'][lrow:urow]
        return view.reshape(context.shape).astype(context.dtype)

    def antenna2(self, context):
        lrow, urow = MS.row_extents(context)
        view = self._manager._data['A1'][lrow:urow]
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
        montblanc.log.debug("Model vis mean {m} sum {s}".format(
            m=np.abs(context.data[:,:,:,0]).mean(),
            s=np.abs(context.data[:,:,:,0]).sum()))

        lrow, urow = MS.row_extents(context)
        view = self._manager._residuals[lrow:urow,:,:]

        montblanc.log.debug("Observed vis mean {m} sum {s}".format(
            m=np.abs(view[:,:,0]).mean(),
            s=np.abs(view[:,:,0]).sum()))

        # Compute residuals
        view[:,:,:] -= context.data.reshape(view.shape)

        montblanc.log.debug("Residual vis mean {m} sum {s}".format(
            m=np.abs(view[:,:,0]).mean(),
            s=np.abs(view[:,:,0]).sum()))

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