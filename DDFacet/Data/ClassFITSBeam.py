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

import numpy
import os
import os.path
import sys

from DDFacet.Other import logger
log = logger.getLogger("ClassFITSBeam")

import pyrap.tables

import numpy as np


dm = pyrap.measures.measures()
dq = pyrap.quanta

# This a list of the Stokes enums (as defined in casacore header measures/Stokes.h)
# These are referenced by the CORR_TYPE column of the MS POLARIZATION subtable.
# E.g. 5,6,7,8 corresponds to RR,RL,LR,LL
MS_STOKES_ENUMS = [
    "Undefined", "I", "Q", "U", "V", "RR", "RL", "LR", "LL", "XX", "XY", "YX", "YY", "RX", "RY", "LX", "LY", "XR", "XL", "YR", "YL", "PP", "PQ", "QP", "QQ", "RCircular", "LCircular", "Linear", "Ptotal", "Plinear", "PFtotal", "PFlinear", "Pangle"
  ];
# set of circular correlations
CIRCULAR_CORRS = set(["RR", "RL", "LR", "LL"]);
# set of linear correlations
LINEAR_CORRS = set(["XX", "XY", "YX", "YY"]);


class ClassFITSBeam (object):
    def __init__ (self, ms, opts):
        self.ms = ms
        # filename is potentially a list (frequencies will be matched)
        self.beamsets = opts["FITSFile"]
        if type(self.beamsets) is not list:
            self.beamsets = self.beamsets.split(',')
        self.pa_inc = opts["FITSParAngleIncDeg"]
        self.time_inc = opts["DtBeamMin"]
        self.nchan = opts["NBand"]
        self.feedangle = opts["FeedAngle"]
        self.applyrotation = (opts["FITSParAngleIncDeg"] or opts["DtBeamMin"]) and opts["ApplyPJones"]
        self.applyantidiagonal = opts["FlipVisibilityHands"]

        # make masure for zenith
        self.zenith = dm.direction('AZEL','0deg','90deg')
        # make position measure from antenna 0
        # NB: in the future we may want to treat position of each antenna separately. For
        # a large enough array, the PA w.r.t. each antenna may change! But for now, use
        # the PA of the first antenna for all calculations
        self.pos0 = dm.position('itrf',*[ dq.quantity(x,'m') for x in self.ms.StationPos[0] ]) 

        # make direction measure from field centre
        ra,dec = self.ms.OriginalRadec
        self.field_centre = dm.direction('J2000',dq.quantity(ra,"rad"),dq.quantity(dec,"rad"))

        # get channel frequencies from MS
        self.freqs = self.ms.ChanFreq.ravel()
        if not self.nchan:
            self.nchan = len(self.freqs)
        else:
            cw = self.ms.ChanWidth.ravel()          
            fq = np.linspace(self.freqs[0]-cw[0]/2, self.freqs[-1]+cw[-1]/2, self.nchan+1)
            self.freqs = (fq[:-1] + fq[1:])/2

        feed = opts["FITSFeed"]
        if feed:
            if len(feed) != 2:
                raise ValueError("FITSFeed parameter must be two characters (e.g. 'xy')")
            feed = feed.lower()
            if "x" in feed:
                self.feedbasis = "linear"
            else:
                self.feedbasis = "circular"
            self.corrs = [ a+b for a in feed for b in feed ]
            print("polarization basis specified by FITSFeed parameter: %s"%" ".join(self.corrs), file=log)
        else:
            # NB: need to check correlation names better. This assumes four correlations in that order!
            if "x" in self.ms.CorrelationNames[0].lower():
                self.corrs = "xx","xy","yx","yy"
                self.feedbasis = "linear"
                print("polarization basis is linear (MS corrs: %s)"%" ".join(self.ms.CorrelationNames), file=log)
            else:
                self.corrs = "rr","rl","lr","ll"
                self.feedbasis = "circular"
                print("polarization basis is circular (MS corrs: %s)"%" ".join(self.ms.CorrelationNames), file=log)
        if opts["FITSFeedSwap"]:
            print("swapping feeds as per FITSFeedSwap setting", file=log)
            self._feed_swap_map = dict(x="y", y="x", r="l", l="r")
        else:
            self._feed_swap_map = None
        # Following code is nicked from Cattery/Siamese/OMS/pybeams_fits.py
        REALIMAG = dict(re="real",im="imag");

        # get the Cattery: if an explicit path to Cattery set, use this and import Siamese directly
        explicit_cattery = False
        for varname in "CATTERY_PATH","MEQTREES_CATTERY_PATH":
            if varname in os.environ:
                sys.path.append(os.environ[varname])
                explicit_cattery = True

        if explicit_cattery:
            import Siamese.OMS.Utils as Utils
            import Siamese
            import Siamese.OMS.InterpolatedBeams as InterpolatedBeams
            print("explicit Cattery path set: using custom Siamese module from %s"%os.path.dirname(Siamese.__file__), file=log)
        else:
            import Cattery.Siamese.OMS.Utils as Utils
            import Cattery.Siamese as Siamese
            import Cattery.Siamese.OMS.InterpolatedBeams as InterpolatedBeams
            print("using standard Cattery.Siamese module from %s"%os.path.dirname(Siamese.__file__), file=log)

        def make_beam_filename (filename_pattern,corr,reim):
            """Makes beam filename for the given correlation and real/imaginary component (one of "re" or "im")"""
            return Utils.substitute_pattern(filename_pattern,
                      corr=corr.lower(),xy=corr.lower(),CORR=corr.upper(),XY=corr.upper(),
                      reim=reim.lower(),REIM=reim.upper(),ReIm=reim.title(),
                      realimag=REALIMAG[reim].lower(),REALIMAG=REALIMAG[reim].upper(),
                      RealImag=REALIMAG[reim].title());

        self.vbs = {}

        # now, self.beamsets specifies a list of filename patterns. We need to find the one with the closest
        # frequency coverage

        if isinstance(opts["FITSFile"], str) and opts["FITSFile"].upper() == "UNITY":
            self.use_unity_ejones = True
        else:
            self.use_unity_ejones = False
            for corr in self.corrs:
                beamlist = []
                corr1 = "".join([self._feed_swap_map[x] for x in corr])if self._feed_swap_map else corr
                for beamset in self.beamsets:
                    filenames = make_beam_filename(beamset, corr1, 're'), make_beam_filename(beamset, corr1, 'im')
                    # get interpolator from cache, or create object
                    vb = ClassFITSBeam._vb_cache.get(filenames)
                    if vb is None:
                        print("loading beam patterns %s %s" % filenames, file=log)
                        ClassFITSBeam._vb_cache[filenames] = vb = InterpolatedBeams.LMVoltageBeam(
                            verbose=opts["FITSVerbosity"],
                            l_axis=opts["FITSLAxis"], m_axis=opts["FITSMAxis"]
                        )  # verbose, XY must come from options
                        vb.read(*filenames)
                    else:
                        print("beam patterns %s %s already in memory" % filenames, file=log)
                    # find frequency "distance". If beam frequency range completely overlaps MS frequency range,
                    # this is 0, otherwise a positive number
                    distance = max(vb._freqgrid[0] - self.freqs[0], 0) + \
                               max(self.freqs[-1] - vb._freqgrid[-1], 0)
                    beamlist.append((distance, vb, filenames))
                # select beams with smallest distance
                dist0, vb, filenames = sorted(beamlist)[0]
                if len(beamlist) > 1:
                    if dist0 == 0:
                        print("beam patterns %s %s overlap the frequency coverage" % filenames, file=log)
                    else:
                        print("beam patterns %s %s are closest to the frequency coverage (%.1f MHz max separation)" % (
                                        filenames[0], filenames[1], dist0*1e-6), file=log)
                    print("  MS coverage is %.1f to %.1f GHz, beams are %.1f to %.1f MHz"%(
                        self.freqs[0]*1e-6, self.freqs[-1]*1e-6, vb._freqgrid[0]*1e-6, vb._freqgrid[-1]*1e-6), file=log)
                self.vbs[corr] = vb


    _vb_cache = {}

    def getBeamSampleTimes (self, times, quiet=False):
        """For a given list of timeslots, returns times at which the beam must be sampled"""
        if not quiet:
            print("computing beam sample times for %d timeslots"%len(times), file=log)
        dt = self.time_inc*60
        beam_times = [ times[0] ]
        for t in times[1:]:
            if t - beam_times[-1] >= dt:
                beam_times.append(t)
        if not quiet:
            print("  DtBeamMin=%.2f min results in %d samples"%(self.time_inc, len(beam_times)), file=log)
        if self.pa_inc:
            pas = [ 
                # put antenna0 position as reference frame. NB: in the future may want to do it per antenna
                dm.do_frame(self.pos0) and 
                # put time into reference frame
                dm.do_frame(dm.epoch("UTC",dq.quantity(t0,"s"))) and
                # compute PA 
                dm.posangle(self.field_centre,self.zenith).get_value("deg") for t0 in beam_times ]
            pa0 = pas[0]
            beam_times1 = [ beam_times[0] ]
            for t, pa in zip(beam_times[1:], pas[1:]):
                if abs(pa-pa0) >= self.pa_inc:
                    beam_times1.append(t)
                    pa0 = pa
            if not quiet:
                print("  FITSParAngleIncrement=%.2f deg results in %d samples"%(self.pa_inc, len(beam_times1)), file=log)
            beam_times = beam_times1
        beam_times.append(times[-1]+1)
        return beam_times

    def getFreqDomains (self):
        domains = np.zeros((len(self.freqs),2),np.float64)
        df = (self.freqs[1]-self.freqs[0])/2 if len(self.freqs)>1 else self.freqs[0]
        domains[:,0] = self.freqs-df
        domains[:,1] = self.freqs+df
#        import pdb; pdb.set_trace()
        return domains

    def evaluateBeam (self, t0, ra, dec):
        """Evaluates beam at time t0, in directions ra, dec.
        Inputs: t0 is a single time. ra, dec are Ndir vectors of directions.
        Output: a complex array of shape [Ndir,Nant,Nfreq,2,2] giving the Jones matrix per antenna, direction and frequency
        """

        # put antenna0 position as reference frame. NB: in the future may want to do it per antenna
        dm.do_frame(self.pos0);
        # put time into reference frame
        dm.do_frame(dm.epoch("UTC",dq.quantity(t0,"s")))
        # compute PA 
        parad = dm.posangle(self.field_centre,self.zenith).get_value("rad")
        import math
        # print("time %f, position angle %f"%(t0, parad*180/math.pi), file=log)

        # compute l,m per direction
        ndir = len(ra)
        l0 = numpy.zeros(ndir,float)
        m0 = numpy.zeros(ndir,float)
        for i,(r1,d1) in enumerate(zip(ra,dec)):
          l0[i], m0[i] = self.ms.radec2lm_scalar(r1,d1,original=True)
        # print(ra*180/np.pi,dec*180/np.pi, file=log)
        # print(l0*180/np.pi,m0*180/np.pi, file=log)
        # rotate each by parallactic angle
        r = numpy.sqrt(l0*l0+m0*m0)
        angle = numpy.arctan2(m0,l0)
        l = r*numpy.cos(angle + parad + np.deg2rad(self.feedangle))
        m = r*numpy.sin(angle + parad + np.deg2rad(self.feedangle))



        # # print("Beam evaluated for l,m", file=log)
        # print(l, file=log)
        # print(m, file=log)
        # print("time %f, position angle %f"%(t0, parad*180/math.pi), file=log)

        # get interpolated values. Output shape will be [ndir,nfreq]
        if self.use_unity_ejones:
            beamjones = [ np.ones([ndir, len(self.freqs)], dtype=numpy.complex64),
                          np.zeros([ndir, len(self.freqs)], dtype=numpy.complex64),
                          np.zeros([ndir, len(self.freqs)], dtype=numpy.complex64),
                          np.ones([ndir, len(self.freqs)], dtype=numpy.complex64) ]
        else:
            beamjones = [ self.vbs[corr].interpolate(l,m,freq=self.freqs,freqaxis=1) for corr in self.corrs ]

        # now make output matrix

        jones = numpy.zeros((ndir,self.ms.na,len(self.freqs),2,2),dtype=numpy.complex64)
        # populate it with values
        # NB: here we copy the same Jones to every antenna. In principle we could compute
        # a parangle per antenna. When we have pointing error, it's also going to be per
        # antenna
        for iant in range(self.ms.na):
            for ijones,(ix,iy) in enumerate(((0,0),(0,1),(1,0),(1,1))):
                bj = beamjones[ijones]
                jones[:,iant,:,ix,iy] = beamjones[ijones].reshape((len(bj),1)) if bj.ndim == 1 else bj

        feedswap_jones = np.array([[1., 0.], [0., 1.]], dtype=numpy.complex64)
        if self.applyantidiagonal:
            feedswap_jones = np.array([[0., 1.], [1., 0.]], dtype=numpy.complex64)

        Pjones = np.array([[1., 0.], [0., 1.]], dtype=numpy.complex64)
        if self.applyrotation:
            print("Applying derotation to data, since beam is sampled in time. "
                  "If you have equatorial mounts this is not what you should be doing!", file=log)
            if self.feedbasis == "linear":
                """ 2D rotation matrix according to Hales, 2017: 
                Calibration Errors in Interferometric Radio Polarimetry """
                c1, s1 = np.cos(parad + np.deg2rad(self.feedangle)), np.sin(parad + np.deg2rad(self.feedangle))
                # assume all stations has same parallactic angle
                Pjones[0, 0] = c1
                Pjones[0, 1] = s1
                Pjones[1, 0] = -s1
                Pjones[1, 1] = c1
            elif self.feedbasis == "circular":
                """ phase rotation matrix according to Hales, 2017: 
                Calibration Errors in Interferometric Radio Polarimetry """
                e1 = np.exp(1.0j * -(parad + np.deg2rad(self.feedangle)))
                e2 = np.exp(1.0j * (parad + np.deg2rad(self.feedangle)))
                # assume all stations has same parallactic angle
                Pjones[0, 0] = e1
                Pjones[0, 1] = 0
                Pjones[1, 0] = 0
                Pjones[1, 1] = e2
            else:
                raise RuntimeError("Feed basis not supported")

        # dot diagonal block matrix of P with E diagonal block vector 
        # again assuming constant P matrix across all stations
        E_vec = jones.reshape(ndir * self.ms.na * len(self.freqs), 2, 2)
        for i in range(E_vec.shape[0]):
            E_vec[i,:,:] = np.dot(Pjones, np.dot(E_vec[i, :, :], feedswap_jones))

        return E_vec.reshape(ndir, self.ms.na, len(self.freqs), 2, 2)







