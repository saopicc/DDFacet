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
from argparse import ArgumentError
from lib2to3.pgen2.parse import ParseError

from DDFacet.compatibility import range

import numpy
import os
import os.path
import sys

from DDFacet.Other import logger
log = logger.getLogger("ClassFITSBeam")

import pyrap.tables

import numpy as np
import json
import re


dm = pyrap.measures.measures()
dq = pyrap.quanta

# get the Cattery: if an explicit path to Cattery set, use this and import Siamese directly
explicit_cattery = False
for varname in "CATTERY_PATH","MEQTREES_CATTERY_PATH":
    if varname in os.environ:
        sys.path.append(os.environ[varname])
        explicit_cattery = True

cattery_import_failure = False

if explicit_cattery:
    try:
        import Siamese.OMS.Utils as Utils
        import Siamese
        import Siamese.OMS.InterpolatedBeams as InterpolatedBeams
        print("explicit Cattery path set: using custom Siamese module from %s"%os.path.dirname(Siamese.__file__), file=log)
    except ImportError:
        cattery_import_failure = True
else:
    try:
        import Cattery.Siamese.OMS.Utils as Utils
        import Cattery.Siamese as Siamese
        import Cattery.Siamese.OMS.InterpolatedBeams as InterpolatedBeams
        print("using standard Cattery.Siamese module from %s"%os.path.dirname(Siamese.__file__), file=log)
    except ImportError:
        cattery_import_failure = True

if cattery_import_failure:
    print("WARNING! Failure to import Meqtrees Cattery. FITS-based primary beam correction disabled", file=log)

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

# Following code is nicked from Cattery/Siamese/OMS/pybeams_fits.py
REALIMAG = dict(re="real",im="imag")

def make_beam_filename (filename_pattern,corr,reim,stationtype):
    """Makes beam filename for the given correlation and real/imaginary component (one of "re" or "im")"""
    return Utils.substitute_pattern(filename_pattern,
                corr=corr.lower(),xy=corr.lower(),CORR=corr.upper(),XY=corr.upper(),
                reim=reim.lower(),REIM=reim.upper(),ReIm=reim.title(),
                realimag=REALIMAG[reim].lower(),REALIMAG=REALIMAG[reim].upper(),
                RealImag=REALIMAG[reim].title(), 
                stype=stationtype.lower(), STYPE=stationtype.upper())

def is_pattern(bs):
    return (any(map(lambda x: "$(" + x + ")" in bs,
                    ["corr","CORR","xy","XY"])) and \
            any(map(lambda x: "$(" + x + ")" in bs,
                    ["reim", "REIM", "ReIm","realimag",
                     "REALIMAG","RealImag"]))) or \
            bs.upper() == "UNITY"

class ClassFITSBeam (object):
    _vb_cache = {} # cached filenames
    def __get_stationtype(self, station):
        return self.station_types["define-stationtypes"].get(
                    station, 
                    self.station_types["define-stationtypes"]["cmd::default"]
        )
    def __get_beamsets(self, station):
        return self.station_types["patterns"].get(
            self.__get_stationtype(station),
            self.station_types["patterns"]["cmd::default"]
        )

    def __load_patterns(self):

        _example_use = \
               "Patterns should resemble 'prefix$(stype)infix$(corr)$infix$(reim).fits' to specify " \
               "a set of fits files containing real and imaginary parts for each correlation hand, e.g. " \
               "mybeam_$(stype)_$(corr)_$(reim).fits specifying mybeam_meerkat_xx_re.fits, mybeam_meerkat_xx_im.fits, " \
               "mybeam_meerkat_xy_re.fits... Lists of such patterns with different frequency coverage may be specified.\n"\
               "Station types can be specified using Json configuration files, containing: \n"\
               "{'lband': {\n" \
               "   'patterns': {\n" \
               "       'cmd::default': ['$(stype)_$(corr)_$(reim).fits',...],\n" \
               "   },\n" \
               "  'define-stationtypes': {\n" \
               "      'cmd::default': 'meerkat',\n" \
               "      'ska000': 'ska'\n" \
               "  },\n" \
               "  ...\n" \
               "}\n" \
               "This will substitute 'meerkat' for all antennas but ska000, with 'meerkat_$(corr)_$(reim).fits' " \
               "whereas beams for ska000 will be loaded from 'ska_$(corr)_$(reim).fits' in this example.\n" \
               "The station name may be specified as regex by adding a '~' infront of the pattern to match, e.g " \
               "'~ska[0-9]{3}': 'ska' will assgign all the 'ska' type to all matching names such as ska000, ska001, ..., skaNNN.\n" \
               "Each station type in the pattern section may specify a list of patterns for different frequency ranges.\n" \
               "Multiple keyed dictionaries such as this may be specified within one file. They will be treated as chained " \
               "configurations, adding more patterns and station-types to the first such block.\n" \
               "Warning: Once a station is type-specialized the type applies to **ALL** chained blocks!\n" \
               "Blocks from more than one config file can be loaded by comma separation, e.g. " \
               "'--Beam-FITSFile conf1.json,conf2.json,...', however no block may define multiple types for any station.\n" \
               "If patterns for a particular station type already exists more patterns are just appended to the existing list.\n" \
               "Warning: where multiple patterns specify the same frequency range the first such pattern closest to the MS " \
               "SPW frequency coverage will be loaded.\n" \
               "If no configuration file is provided the pattern may not contain $(stype) -- station independence is assumed. " \
               "This is the same as specifing the following config: \n" \
               "{'lband': {\n" \
               "   'patterns': {\n" \
               "       'cmd::default': ['$(corr)_$(reim).fits',...],\n" \
               "   },\n" \
               "  'define-stationtypes': {\n" \
               "      'cmd::default': 'cmd::default',\n" \
               "  }\n" \
               "}\n" \
               "'corr' above may be one of 'corr', 'CORR', 'xy', 'XY' to cast the hands to lower or UPPER case.\n" \
               "'reim' above may be one of 'reim', 'REIM', 'ReIm', 'realimag', 'REALIMAG', 'RealImag' to cast the " \
               "real or imaginary substitutions to lower case, CamelCase or UPPER case respectively.\n" \
               "'stype' above may be one of 'stype' or 'STYPE' to cast the station/antenna type id to the given name " \
               "as defined in the Json file. This is optional depending on whether station types are being used.\n" \
               "You may specify 'UNITY' as a pattern to apply a unitarian response across the entire FoV in case you " \
               "just want derotation of the visibilities. If 'UNITY' is specified for a station, you may not load additional "\
               "beam patterns for it - it is assumed that unitarian response will be applied at all frequencies."
        for bs in self.beamsets:
            if os.path.splitext(bs)[1] == ".json":
                if os.path.exists(bs):
                    with open(bs) as fbs:
                        vels = json.loads(fbs.read())
                    if not isinstance(vels, dict):
                        raise ValueError("Station type config should contain dictionary blocks. " + 
                                         _example_use)
                    for key in vels:
                        if not set(vels[key].keys()).issubset(["patterns", "define-stationtypes"]):
                            raise ValueError("Station type config blocks may only contain keys "
                                            "'patterns' and 'define-stationtypes'. " + _example_use)
                        for st_t, patterns in vels[key].get("patterns", {}).items():
                            if not isinstance(patterns, list):
                                patterns = [patterns]
                            self.station_types["patterns"][st_t] = \
                                self.station_types["patterns"].get(st_t, []) + patterns

                        for st in filter(lambda key: key.find("~") < 0,
                                        vels[key].get("define-stationtypes", {}).keys()):
                            st_t = vels[key]["define-stationtypes"][st]
                            if st in self.station_types["define-stationtypes"] and \
                                self.station_types["define-stationtypes"][st] != st_t:
                                raise ValueError(f"Ambiguous redefinition of station type for '{st}' while "
                                                f"parsing '{bs}'. Check your configuration chain. " + _example_use)
                            self.station_types["define-stationtypes"][st] = st_t

                        for wildcard in map(lambda x: x[1:], 
                                            filter(lambda key: key.find("~") == 0,
                                                vels[key].get("define-stationtypes", {}).keys())):
                            fcmatches = list(filter(lambda st: re.match(wildcard, st), self.ms.StationNames))
                            if len(fcmatches) == 0:
                                raise ValueError(f"No station name matches regex '{wildcard}' while parsing '{bs}'. "
                                                 f"Check your station types config!")
                            for st in fcmatches:
                                st_t = vels[key]["define-stationtypes"]["~" + wildcard]
                                if st in self.station_types["define-stationtypes"] and \
                                    self.station_types["define-stationtypes"][st] != st_t:
                                    raise ValueError(f"Ambigous redefinition of station type for '{st}' while "
                                                     f"parsing wildcard station name '{wildcard}' in '{bs}'. "
                                                     f"Check your configuration chain. " + _example_use)
                                self.station_types["define-stationtypes"][st] = st_t
                else:
                    raise FileNotFoundError(f"Station beam pattern map config file '{bs}' does not exist")
            elif is_pattern(bs):
                self.station_types["patterns"]["cmd::default"] = \
                    self.station_types["patterns"].get("cmd::default", []) + [bs]
            else:
                raise ValueError(f"'{bs} specified for Beam-FITS is neither a pattern nor json "
                                 f"config file. " + _example_use)
        self.station_types["define-stationtypes"].setdefault("cmd::default", "cmd::default")
        self.station_types["patterns"].setdefault("cmd::default", [])

        for st in self.ms.StationNames:
            beamsets = self.__get_beamsets(st)
            this_st_t = self.__get_stationtype(st)
            if beamsets == []:
                raise ValueError(f"EJones via FITS beams are enabled in your parset, "
                                 f"but no beam patterns are specified for station {st}. "
                                 f"Please check your config")

            if any(map(lambda x: x.upper() == "UNITY", beamsets)) and \
                not all(map(lambda x: x.upper() == "UNITY", beamsets)):
                raise ValueError(f"Ambiguous use of UNITY beamset for {st}. If UNITY "
                                 f"is specified no other beamset patterns may be specified for the station. "
                                 f"Unitarian response will be assumed at all frequencies")

            if this_st_t == "cmd::default" and \
                any(map(lambda x: "$(" + x + ")" in bs,
                       ["stype","STYPE"])):
                raise ValueError(f"One or more patterns for station {st} requests substition on type "
                                 f"but you have not specified a type for this station via types configuration file. "
                                 f"Did you want to load a configuration file as well? " + _example_use)

        # now we may have many patterns in our little database, but what really determines
        # if we should use station dependent beams is if more than just the default group of
        # stations have been assigned a type to use other than the default
        if len(self.station_types["define-stationtypes"]) > 1 and \
            any(map(lambda st: self.station_types["define-stationtypes"][st] != 
                               self.station_types["define-stationtypes"]["cmd::default"],
                    self.station_types["define-stationtypes"].keys())):
            self.station_dependent_beams = True
        
        if self.station_dependent_beams:
            print("Using station-dependent E Jones for the array - this "
                  "may take longer to interpolate depending on how many "
                  "unique elements are in the array", file=log)
        else:
            print("Using station-independent E Jones for the array", file=log)

    def __expand_beamsets(self, opts):
        # now, self.beamsets specifies a list of filename patterns per station. 
        # We need to find the one with the closest frequency coverage if this list is longer than 1
        for station in self.ms.StationNames if self.station_dependent_beams else [self.ms.StationNames[0]]:
            # guarranteed not to be empty
            this_beamsets = self.__get_beamsets(station)
            # guarranteed to be one type or the default type
            this_st_t = self.__get_stationtype(station)
            printname = (station + f" (type: {this_st_t})" if this_st_t != "cmd::default" else "") \
                if self.station_dependent_beams else "All stations"

            self.vbs[station] = {}
            for corr in self.corrs:
                beamlist = []
                corr1 = "".join([self._feed_swap_map[x] for x in corr]) if self._feed_swap_map \
                    else corr
                for beamset in this_beamsets:    
                    if beamset.upper() == "UNITY":
                        self.vbs[station][corr] = ("UNITYDIAG", "")
                        if corr == self.corrs[0]:
                            print(printname + ": using unitarian beams", file=log)
                    else:
                        filenames = make_beam_filename(beamset, corr1, 're', this_st_t), \
                                    make_beam_filename(beamset, corr1, 'im', this_st_t)
                        assert len(filenames) == 2
                        # get interpolator from cache, or create object
                        vb = ClassFITSBeam._vb_cache.get(filenames)
                        if vb is None:
                            print(printname + ": loading beam patterns %s %s" % filenames, file=log)
                            vb = \
                                InterpolatedBeams.LMVoltageBeam(
                                    verbose=opts["FITSVerbosity"],
                                    l_axis=opts["FITSLAxis"], 
                                    m_axis=opts["FITSMAxis"]
                            )  # verbose, XY must come from options
                            vb.read(*filenames)
                            ClassFITSBeam._vb_cache[filenames] = vb
                        else:
                            print(printname + ": beam patterns %s %s already in memory" % filenames, file=log)
                        # find frequency "distance". If beam frequency range completely overlaps MS frequency range,
                        # this is 0, otherwise a positive number
                        distance = max(vb._freqgrid[0] - self.freqs[0], 0) + \
                                max(self.freqs[-1] - vb._freqgrid[-1], 0)
                        beamlist.append((distance, vb, filenames))
                        # select beams with smallest distance in frequency from our band
                        dist0, vb, filenames = sorted(beamlist, key=lambda beam: beam[0])[0]
                        if len(beamlist) > 1:
                            if dist0 == 0:
                                print(printname + ": beam patterns %s %s overlap the frequency coverage" % filenames, file=log)
                            else:
                                print(printname + ": beam patterns %s %s are closest to the frequency coverage (%.1f MHz max separation)" % (
                                                filenames[0], filenames[1], dist0*1e-6), file=log)
                            print("  MS coverage is %.1f to %.1f GHz, beams are %.1f to %.1f MHz"%(
                                self.freqs[0]*1e-6, self.freqs[-1]*1e-6, vb._freqgrid[0]*1e-6, vb._freqgrid[-1]*1e-6), file=log)
                        self.vbs[station][corr] = (vb, filenames)


    def __init__ (self, ms, opts):
        if cattery_import_failure:
            raise ImportError("The MeqTrees Cattery was not found in your path. Run installation with e.g. "
                              "the optional extras specifier: 'pip install ddfacet[fits-beam-support]'. "
                              "Brackets may need to be escaped depending on your shell of choice (e.g. with zsh)")
        self.ms = ms
        # filename is potentially a list (frequencies will be matched)
        self.beamsets = opts["FITSFile"]
        if not isinstance(self.beamsets,list):
            self.beamsets = self.beamsets.split(',')
        self.station_types = {
            "patterns": {},
            "define-stationtypes": {}
        }
        self.station_dependent_beams = False
        self.__load_patterns()
    
        self.pa_inc = opts["FITSParAngleIncDeg"]
        self.time_inc = opts["DtBeamMin"]
        self.nchan = opts["NBand"]
        self.feedangle = opts["FeedAngle"]
        self.applyrotation = (opts["FITSParAngleIncDeg"] or opts["DtBeamMin"]) and opts["ApplyPJones"]
        self.applyantidiagonal = opts["FlipVisibilityHands"]
        self._frame = opts["FITSFrame"]

        # make measure for zenith
        if self._frame == "altaz":
            self.zenith = dm.direction('AZEL','0deg','90deg')
        else: # for azelgeo frames or the newly incorporated unstearable zenith mode
            self.zenith = dm.direction('AZELGEO','0deg','90deg')
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
        
        self.vbs = {} 
        self.__expand_beamsets(opts)
        
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

        # setup reference frame and compute PA
        if self._frame != "equatorial":
            # put antenna0 position as reference frame. NB: in the future may want to do it per antenna
            dm.do_frame(self.pos0)
            # put time into reference frame
            dm.do_frame(dm.epoch("UTC",dq.quantity(t0,"s")))
            # compute PA
            parad = dm.posangle(self.field_centre,self.zenith).get_value("rad")
        else:
            parad = 0
        # print("time %f, position angle %f"%(t0, parad*180/math.pi), file=log)

        # compute l,m per direction
        ndir = len(ra)
        l = numpy.zeros(ndir,float)
        m = numpy.zeros(ndir,float)

        if self._frame == "altaz" or self._frame == "equatorial" or self._frame == "altazgeo":
            # convert each ra/dec to l/m
            for i,(r1,d1) in enumerate(zip(ra,dec)):
                l[i], m[i] = self.ms.radec2lm_scalar(r1,d1,original=True)
            # for alt-az mounts, rotate by PA
            if self._frame == "altaz" or self._frame == "altazgeo":
                # rotate each by parallactic angle
                r = numpy.sqrt(l*l+m*m)
                angle = numpy.arctan2(m, l)
                l = r*numpy.cos(angle + parad + np.deg2rad(self.feedangle))
                m = r*numpy.sin(angle + parad + np.deg2rad(self.feedangle))
        elif self._frame == "zenith":
            az = numpy.zeros(ndir, float)
            el = numpy.zeros(ndir, float)
            for i, (r1,d1) in enumerate(zip(ra,dec)):
                dir_j2000 = dm.direction('J2000', dq.quantity(r1, "rad"), dq.quantity(d1, "rad"))
                dir_azel = dm.measure(dir_j2000, "AZELGEO")
                dir_azel_val = dm.get_value(dir_azel)
                az[i], el[i] = dir_azel_val[0].get_value(), dir_azel_val[1].get_value()

            r = numpy.cos(el)
            l = r*numpy.sin(az)   # az=0 is North, l=0, M>0
            m = r*numpy.cos(az)   # az=90 is East, m=0, l>0
        else:
            raise RuntimeError("unknown FITSFrame {}".format(self._frame))

        log(2).print("Beam evaluated for l,m {}, {}".format(l, m))

        # get interpolated values. Output shape will be [ndir,nfreq]
        # Note: going to cache the interpolated values for this evaluation of the beam
        # so that they can just be broadcast if we have homogenious antennae
        __bj_cache = dict(zip(self.corrs, [{}] * len(self.corrs)))

        def __compute_beam_jones(station):
            if any(map(lambda corr: isinstance(self.vbs[station][corr][0], str) and
                                    self.vbs[station][corr][0] == "UNITYDIAG", self.corrs)):
                return [np.ones([ndir, len(self.freqs)], dtype=numpy.complex64),
                        np.zeros([ndir, len(self.freqs)], dtype=numpy.complex64),
                        np.zeros([ndir, len(self.freqs)], dtype=numpy.complex64),
                        np.ones([ndir, len(self.freqs)], dtype=numpy.complex64)]
            else:
                jones = []
                for corr in self.corrs:
                    vb, filenames = self.vbs[station][corr]
                    cachekey = "&&".join(filenames)
                    if cachekey in __bj_cache[corr]:
                        jones.append(__bj_cache[corr][cachekey])
                    else:
                        bj = vb.interpolate(l,m,freq=self.freqs,freqaxis=1)
                        __bj_cache[corr][cachekey] = bj
                        jones.append(bj)  
                return jones
                            

        # now make output matrix
        jones = numpy.zeros((ndir, self.ms.na, len(self.freqs), 2, 2),
                            dtype=numpy.complex64)
        # populate it with values
        # NB: here we copy the same P Jones to every antenna. In principle we could compute
        # a parangle per antenna. When we have pointing error, it's also going to be per
        # antenna
        if not self.station_dependent_beams:
            sn = self.ms.StationNames[0] # no station dependence in the beams, same beam for all
            arraybeamjones = __compute_beam_jones(sn)

        for iant in range(self.ms.na):
            for ijones,(ix,iy) in enumerate(((0,0),(0,1),(1,0),(1,1))):
                if self.station_dependent_beams:
                    sn = self.ms.StationNames[iant]
                    beamjones = __compute_beam_jones(sn)
                    jones[:,iant,:,ix,iy] = beamjones[ijones].reshape((len(beamjones[ijones]), 1)) if beamjones[ijones].ndim == 1 \
                        else beamjones[ijones]
                else:
                    beamjones = arraybeamjones
                    jones[:,iant,:,ix,iy] = beamjones[ijones].reshape((len(beamjones[ijones]), 1)) if beamjones[ijones].ndim == 1 \
                        else beamjones[ijones]

        feedswap_jones = np.array([[1., 0.], [0., 1.]], dtype=numpy.complex64)
        if self.applyantidiagonal:
            feedswap_jones = np.array([[0., 1.], [1., 0.]], dtype=numpy.complex64)

        Pjones = np.array([[1., 0.], [0., 1.]], dtype=numpy.complex64)
        if self.applyrotation:
            if self._frame == "equatorial" or self._frame == "zenith":
                print("Applying derotation to data, since beam is sampled in time. "
                      "If you have equatorial or zenithal mounts this is not what you should be doing!", file=log)
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






