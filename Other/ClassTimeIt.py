
import time as timemod
import MyLogger
log=MyLogger.getLogger("ClassTimeIt")
DoLog=False


class ClassTimeIt():
    def __init__(self,name=""):
        self.t0=timemod.time()
        self.IsEnable=True
        if name=="":
            self.name=name
        else:
            self.name=name+": "
        self.IsEnableIncr=False
        self.Counter=""

    def reinit(self):
        self.t0=timemod.time()

    def timeit(self,stri=" Time",hms=False):
        if self.IsEnable==False: return
        t1=timemod.time()
        dt=t1-self.t0
        if not(hms):
            Sout= "  * %s%s %s : %7.5fs"%(self.name,stri,str(self.Counter),dt)
            if self.IsEnableIncr: self.Counter+=1
        else:
            ss=(dt)/60.
            m=int(ss)
            s=(ss-m)*60.
            Sout= "  * %s computation time: %i min. %4.1f sec."%(stri,m,s)
        self.t0=t1
        if DoLog:
            print>>log, Sout
        else:
            print Sout

        return dt

    def timeitHMS(self,stri=" Time"):
        t1=timemod.time()
        self.t0=t1

    def disable(self):
        self.IsEnable=False

    def enableIncr(self,incr=1):
        self.IsEnableIncr=True
        self.Counter=0

    def AddDt(self,Var):
        t1=timemod.time()
        dt=t1-self.t0
        Var=Var+dt
        self.t0=t1
        return Var
