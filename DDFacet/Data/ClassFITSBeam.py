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

import numpy
import os
import os.path
import sys

from DDFacet.Other import MyLogger
log = MyLogger.getLogger("ClassFITSBeam")

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

        # make masure for zenith
        self.zenith = dm.direction('AZEL','0deg','90deg')
        # make position measure from antenna 0
        # NB: in the future we may want to treat position of each antenna separately. For
        # a large enough array, the PA w.r.t. each antenna may change! But for now, use
        # the PA of the first antenna for all calculations
        self.pos0 = dm.position('itrf',*[ dq.quantity(x,'m') for x in self.ms.StationPos[0] ]) 

        # make direction measure from field centre
        self.field_centre = dm.direction('J2000',dq.quantity(self.ms.rarad,"rad"),dq.quantity(self.ms.decrad,"rad"))

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
                raise ValueError,"FITSFeed parameter must be two characters (e.g. 'xy')"
            feed = feed.lower()
            self.corrs = [ a+b for a in feed for b in feed ]
            print>>log,"polarization basis specified by FITSFeed parameter: %s"%" ".join(self.corrs)
        else:
            # NB: need to check correlation names better. This assumes four correlations in that order!
            if "x" in self.ms.CorrelationNames[0].lower():
                self.corrs = "xx","xy","yx","yy"
                print>>log,"polarization basis is linear (MS corrs: %s)"%" ".join(self.ms.CorrelationNames)
            else:
                self.corrs = "rr","rl","lr","ll"
                print>>log,"polarization basis is circular (MS corrs: %s)"%" ".join(self.ms.CorrelationNames)
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
            print>>log,"explicit Cattery path set: using custom Siamese module from %s"%os.path.dirname(Siamese.__file__)
        else:
            import Cattery.Siamese.OMS.Utils as Utils
            import Cattery.Siamese as Siamese
            import Cattery.Siamese.OMS.InterpolatedBeams as InterpolatedBeams
            print>>log,"using standard Cattery.Siamese module from %s"%os.path.dirname(Siamese.__file__)

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

        for corr in self.corrs:
            beamlist = []
            for beamset in self.beamsets:
                filenames = make_beam_filename(beamset, corr, 're'), make_beam_filename(beamset, corr, 'im')
                # get interpolator from cache, or create object
                vb = ClassFITSBeam._vb_cache.get(filenames)
                if vb is None:
                    print>> log, "loading beam patterns %s %s" % filenames
                    ClassFITSBeam._vb_cache[filenames] = vb = InterpolatedBeams.LMVoltageBeam(
                        verbose=opts["FITSVerbosity"],
                        l_axis=opts["FITSLAxis"], m_axis=opts["FITSMAxis"]
                    )  # verbose, XY must come from options
                    vb.read(*filenames)
                else:
                    print>> log, "beam patterns %s %s already in memory" % filenames
                # find frequency "distance". If beam frequency range completely overlaps MS frequency range,
                # this is 0, otherwise a positive number
                distance = max(vb._freqgrid[0] - self.freqs[0], 0) + \
                           max(self.freqs[-1] - vb._freqgrid[-1], 0)
                beamlist.append((distance, vb, filenames))
            # select beams with smallest distance
            dist0, vb, filenames = sorted(beamlist)[0]
            if len(beamlist) > 1:
                if dist0 == 0:
                    print>> log, "beam patterns %s %s overlap the frequency coverage" % filenames
                else:
                    print>> log, "beam patterns %s %s are closest to the frequency coverage (%.1f MHz max separation)" % (
                                    filenames[0], filenames[1], dist0*1e-6)
                print>>log,"  MS coverage is %.1f to %.1f GHz, beams are %.1f to %.1f MHz"%(
                    self.freqs[0]*1e-6, self.freqs[-1]*1e-6, vb._freqgrid[0]*1e-6, vb._freqgrid[-1]*1e-6)
            self.vbs[corr] = vb


    _vb_cache = {}

    def getBeamSampleTimes (self, times, quiet=False):
        """For a given list of timeslots, returns times at which the beam must be sampled"""
        if not quiet:
            print>>log,"computing beam sample times for %d timeslots"%len(times)
        dt = self.time_inc*60
        beam_times = [ times[0] ]
        for t in times[1:]:
            if t - beam_times[-1] >= dt:
                beam_times.append(t)
        if not quiet:
            print>>log,"  DtBeamMin=%.2f min results in %d samples"%(self.time_inc, len(beam_times))
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
                print>>log,"  FITSParAngleIncrement=%.2f deg results in %d samples"%(self.pa_inc, len(beam_times1))
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
        # print>>log,"time %f, position angle %f"%(t0, parad*180/math.pi)

        # compute l,m per direction
        ndir = len(ra)
        l0 = numpy.zeros(ndir,float)
        m0 = numpy.zeros(ndir,float)
        for i,(r1,d1) in enumerate(zip(ra,dec)):
          l0[i], m0[i] = self.ms.radec2lm_scalar(r1,d1)
        # print>>log,ra*180/np.pi,dec*180/np.pi
        # print>>log,l0*180/np.pi,m0*180/np.pi
        # rotate each by parallactic angle
        r = numpy.sqrt(l0*l0+m0*m0)
        angle = numpy.arctan2(m0,l0)
        l = r*numpy.cos(angle+parad)
        m = r*numpy.sin(angle+parad)  

        # print>>log,"Beam evaluated for l,m"
        # print>>log,l
        # print>>log,m

        # get interpolated values. Output shape will be [ndir,nfreq]
        beamjones = [ self.vbs[corr].interpolate(l,m,freq=self.freqs,freqaxis=1) for corr in self.corrs ]

        # now make output matrix
        jones = numpy.zeros((ndir,self.ms.na,len(self.freqs),2,2),dtype=numpy.complex64)

        # populate it with values
        # NB: here we copy the same Jones to every antenna. In principle we could compute
        # a parangle per antenna. When we have pointing error, it's also going to be per
        # antenna
        for iant in xrange(self.ms.na):
            for ijones,(ix,iy) in enumerate(((0,0),(0,1),(1,0),(1,1))):
                bj = beamjones[ijones]
                jones[:,iant,:,ix,iy] = beamjones[ijones].reshape((len(bj),1)) if bj.ndim == 1 else bj
        return jones







