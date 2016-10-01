
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
