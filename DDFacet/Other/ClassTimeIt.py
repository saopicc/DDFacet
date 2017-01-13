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

import time as timemod
import MyLogger
log = MyLogger.getLogger("ClassTimeIt")
DoLog = False


class ClassTimeIt():

    def __init__(self, name=""):
        self.t0 = timemod.time()
        self.IsEnable = True
        if name == "":
            self.name = name
        else:
            self.name = name+": "
        self.IsEnableIncr = False
        self.Counter = ""

    def reinit(self):
        self.t0 = timemod.time()

    def timestr(self, hms=False):
        t1 = timemod.time()
        dt = t1-self.t0
        self.t0 = t1
        if not hms:
            return "%7.5fs" % dt
        else:
            ss = dt/60.
            m = int(ss)
            s = (ss-m)*60.
            return "%im%.1fs" % (m, s)

    def seconds(self):
        t1 = timemod.time()
        dt = t1 - self.t0
        self.t0 = t1
        return dt

    def timehms(self):
        return self.timestr(hms=True)

    def timeit(self, stri=" Time", hms=False):
        if not self.IsEnable:
            return
        ts = self.timestr(hms=hms)
        if not hms:
            Sout = "  * %s%s %s : %s" % (self.name, stri, str(self.Counter), ts)
            if self.IsEnableIncr:
                self.Counter += 1
        else:
            Sout = "  * %s computation time: %s" % (stri, ts)
        if DoLog:
            print>>log, Sout
        else:
            print Sout

    def disable(self):
        self.IsEnable = False

    def enableIncr(self, incr=1):
        self.IsEnableIncr = True
        self.Counter = 0

    def AddDt(self, Var):
        t1 = timemod.time()
        dt = t1-self.t0
        Var = Var+dt
        self.t0 = t1
        return Var
