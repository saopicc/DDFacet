'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2018  Cyril Tasse, l'Observatoire de Paris,
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
from scipy.interpolate import interp1d
import pandas as pd
from pyrap import quanta as qa
import datetime
from DDFacet.Data.ClassStokes import ClassStokes
from DDFacet.Other import MyLogger
log= MyLogger.getLogger("PointingProvider")

class InvalidPointingSolutions(Exception):
    pass
class InvalidFeedType(Exception):
    pass

class PointingProvider(object):
    __COMPULSORY_HEADER = ["#ANT", "POL", "TIME", "RA_ERR(deg)", "DEC_ERR(deg)"]
    __AVAILABLE_INTERPOLATORS = {
            "LERP": "linear",
    }
    def __init__(self, MS, solutions_file="", interp_mode="LERP"):
        """
        Pointing Solutions Reader
        
        Reads pointing solutions solved for externally in as deliminated text and interpolates
        in time per antenna feed. Currently only linear interpolation ('LERP') is supported, where
        the boundaries are clamped to the nearest neighbour in instances where solutions need to be 
        extrapolated.
        
        Args:
        MS: initialized DDFacet.Data.ClassMS instance containing at least StationNames and feed information
        solutions_file: is the path to space deliminated solutions with the following compulsory header:
                        #ANT POL TIME RA_ERR(deg) DEC_ERR(deg)
                        If left as None or empty string poiting errors of 0 is assumed for all stations in 
                        MS.StationNames
        interp_mode: interpolation method to use, currently only LERP is supported
        """
        self._MS_hndl = MS
        self._data_feed_labels = ClassStokes(MS.CorrelationIds, ["I"]).AvailableCorrelationProducts()
        if set(self._data_feed_labels) <= set(["XX", "XY", "YX", "YY"]):
            self._feed_type = "linear"
            self._ptcorr_labels = ["XX", "YY"]
        elif set(self._data_feed_labels) <= set(["RR", "RL", "LR", "LL"]):
            self._feed_type = "circular"
            self._ptcorr_labels = ["RR", "LL"]
        else:
            raise InvalidFeedType("Imager pointing errors module only supports linear or circular feed measurements.")
        self._raw_offsets = None
        self._read_raw(solutions_file)
        self._solutions_file = solutions_file
        self._interp_mode = None
        self._interp_offsets = None
        self._min_time_ant = {}
        self._max_time_ant = {}
        self.interpolator = interp_mode
        
    def _read_raw(self, pointing_errs_file):
        """ Reads raw data """
        if pointing_errs_file == "" or not pointing_errs_file:
            na = len(self._MS_hndl.StationNames)
            # initialize dataframe used to initialize interpolators 2 * ntime * 2 correlations per antenna
            # if no pointing file is specified
            self._raw_offsets = pd.DataFrame(zip([sn for sn in self._MS_hndl.StationNames for i in range(4)], 
                                                 self._ptcorr_labels * (na * 2),
                                                 np.zeros(na * 4, dtype=np.float64),
                                                 np.zeros(na * 4, dtype=np.float64),
                                                 np.zeros(na * 4, dtype=np.float64)),
                                             columns=PointingProvider.__COMPULSORY_HEADER)
        else:
            self._raw_offsets = pd.read_csv(pointing_errs_file, sep="\s+")
            if not (set(PointingProvider.__COMPULSORY_HEADER) <= set([l for l in self._raw_offsets.columns])):
                raise InvalidPointingSolutions("Requires at least %s columns present" % ",".join(PointingProvider.__COMPULSORY_HEADER))

            if np.any(np.logical_and(self._raw_offsets["POL"] != self._ptcorr_labels[0],
                                     self._raw_offsets["POL"] != self._ptcorr_labels[1])):
                raise InvalidPointingSolutions("%s feed in measurement set, but found pointing error solutions for another feed type" % self._feed_type)
            
            print>> log, "Read pointing error solutions from %s" % pointing_errs_file

    def _init_interpolator(self):
        """ Initializes dictionary keyed on antenna, correlation and RA, DEC with solution interpolators """
        self._interp_offsets = {}
        mjd2utc = lambda x: datetime.datetime.utcfromtimestamp(qa.quantity(str(x)+'s').to_unix_time())
        if self.sols_filename == "" or not self.sols_filename:
            print>>log, "Initializing pointing solutions to 0.0, 0.0 for stations %s" % ",".join(set(self._raw_offsets["#ANT"]))
        else:
            print>>log, "Pointing solutions span %s to %s UTC" % (mjd2utc(self._raw_offsets.min()["TIME"]),
                                                                  mjd2utc(self._raw_offsets.max()["TIME"]))
            print>>log, "Pointing solutions contain solutions for stations %s" % ",".join(set(self._raw_offsets["#ANT"]))
        for a in set(self._raw_offsets["#ANT"]):
            self._interp_offsets[a] = {c: {} for c in self._ptcorr_labels}
            self._min_time_ant[a] = self._raw_offsets.where(self._raw_offsets["#ANT"] == a).min()["TIME"]
            self._max_time_ant[a] = self._raw_offsets.where(self._raw_offsets["#ANT"] == a).max()["TIME"]

            for f in ["XX", "YY"] if self._feed_type == "linear" else ["RR", "LL"]:
                sel = self._raw_offsets.where(np.logical_and(self._raw_offsets["#ANT"] == a,
                                                             self._raw_offsets["POL"] == f)).dropna()
                sel_sorted = sel.sort_values("TIME")
                self._interp_offsets[a][f]["RA"] = interp1d(sel["TIME"],
                                                            sel["RA_ERR(deg)"],
                                                            kind=PointingProvider.__AVAILABLE_INTERPOLATORS[self._interp_mode],
                                                            bounds_error=False,
                                                            fill_value=(sel_sorted["RA_ERR(deg)"].iloc[0], sel_sorted["RA_ERR(deg)"].iloc[-1]))
                self._interp_offsets[a][f]["DEC"] = interp1d(sel["TIME"],
                                                             sel["DEC_ERR(deg)"],
                                                             kind=PointingProvider.__AVAILABLE_INTERPOLATORS[self._interp_mode],
                                                             bounds_error=False,
                                                             fill_value=(sel_sorted["DEC_ERR(deg)"].iloc[0], sel_sorted["DEC_ERR(deg)"].iloc[-1]))
                
                print>> log, "Station %s feed %s has interquartile pointing spread of (%.2f, %.2f) deg in RA and (%.2f, %.2f) deg in DECL" % \
                     (a, f, sel.quantile(.25)["RA_ERR(deg)"], sel.quantile(.75)["RA_ERR(deg)"], 
                      sel.quantile(.25)["DEC_ERR(deg)"], sel.quantile(.75)["DEC_ERR(deg)"])
    
    @property
    def sols_filename(self):
        """ Returns filename of solutions file """
        return self._solutions_file
    
    @property
    def interpolator(self):
        """ Returns type of interpolation being used """
        return self._interp_mode
    
    @interpolator.setter
    def interpolator(self, value):
        """ Sets type of interpolation being used. Currently only LERP is supported """
        if value not in PointingProvider.__AVAILABLE_INTERPOLATORS.keys():
            raise ValueError("Only supports the following interpolators: %s" % ",".join(PointingProvider.__AVAILABLE_INTERPOLATORS.keys()))
        self._interp_mode = value
        self._init_interpolator()
        
    def __get_offset(self, antenna_name, time, corr):
        """ 
        Gets interpolated pointing offset 
        
        Arguments:
        antenna_name: antenna/station name as it appears in the ::ANTENNA subtable
        time: float / ndarray of mean Julian date time values (UTC) as read from TIME
        corr: XX, YY or RR, LL depending on feed type of measurement
        """
        if antenna_name not in self._interp_offsets.keys():
            print>>log, "No pointing solutions for station %s, assuming 0.0" % antenna_name
            return np.array([np.zeros_like(time), np.zeros_like(time)])
        num_extrap = np.sum(np.logical_or(time < self._min_time_ant[antenna_name],
                                          time > self._max_time_ant[antenna_name]))

        if num_extrap > 0:
            print>>log, "Warning: extrapolating %.2f%% pointing errors for antenna %s." % \
                    ((num_extrap / float(time.size) * 100.0), antenna_name)
        if corr not in self._interp_offsets[antenna_name].keys():
            raise KeyError("No interpolated solutions for correlation %s. This is a bug." % corr)
        ra = self._interp_offsets[antenna_name][corr]["RA"](time)
        dec = self._interp_offsets[antenna_name][corr]["DEC"](time) 
        return np.array([ra, dec])
    
    def offset_XX(self, antenna_name, time):
        """
        Gets XX offset (for linear feed data)
        
        Arguments:
        antenna_name: antenna/station name as it appears in the ::ANTENNA subtable
        time: float / ndarray of mean Julian date time values (UTC) as read from TIME
        """
        if self._feed_type != "linear":
            raise ValueError("Feed type is not linear.")
        return self.__get_offset(antenna_name, time, "XX")
    
    def offset_YY(self, antenna_name, time):
        """
        Gets YY offset (for linear feed data)
        
        Arguments:
        antenna_name: antenna/station name as it appears in the ::ANTENNA subtable
        time: float / ndarray of mean Julian date time values (UTC) as read from TIME
        """
        if self._feed_type != "linear":
            raise ValueError("Feed type is not linear.")
        return self.__get_offset(antenna_name, time, "YY")
    
    def offset_RR(self, antenna_name, time):
        """
        Gets RR offset (for circular feed data)
        
        Arguments:
        antenna_name: antenna/station name as it appears in the ::ANTENNA subtable
        time: float / ndarray of mean Julian date time values (UTC) as read from TIME
        """
        if self._feed_type != "circular":
            raise ValueError("Feed type is not circular.")
        return self.__get_offset(antenna_name, time, "RR")
    
    def offset_LL(self, antenna_name, time):
        """
        Gets LL offset (for circular feed data)
        
        Arguments:
        antenna_name: antenna/station name as it appears in the ::ANTENNA subtable
        time: float / ndarray of mean Julian date time values (UTC) as read from TIME
        """
        if self._feed_type != "circular":
            raise ValueError("Feed type is not circular.")
        return self.__get_offset(antenna_name, time, "LL")
