import logging
import sys
import os

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message != '\n':
            self.logger.log(self.level, message)

import ModColor

_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey,statusfile=None):
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(statusfile or _proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]


def _shmem_size (since=0.0):
    '''Return shared memory usage in bytes.
    '''
    return _VmB('Shmem:','/proc/meminfo') - since

def _memory(since=0.0):
    '''Return memory usage in bytes.
    '''
    return _VmB('VmSize:') - since


def _resident(since=0.0):
    '''Return resident memory usage in bytes.
    '''
    return _VmB('VmRSS:') - since


def _stacksize(since=0.0):
    '''Return stack size in bytes.
    '''
    return _VmB('VmStk:') - since

log_memory = False

def enableMemoryLogging (enable=True):
    global log_memory
    log_memory = enable

class LoggerMemoryFilter (logging.Filter):
    def filter(self, event):
        vss = float(_memory()/(1024**3))
        rss = float(_resident()/(1024**3))
        shm = float(_shmem_size()/(1024**3))
        setattr(event,"virtual_memory_gb",vss)
        setattr(event,"resident_memory_gb",rss)
        setattr(event,"shared_memory_gb",shm)
        if log_memory and hasattr(event,"msg"):
            event.msg = "[r:%.1f v:%.1f sh:%.1fGb] "%(rss,vss,shm) + event.msg
        return True


class MyLogger():
    def __init__(self):
#fmt="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
        fmt=" - %(asctime)s - %(name)-25.25s | %(message)s"
        datefmt='%H:%M:%S'#'%H:%M:%S.%f'
        logging.basicConfig(level=logging.DEBUG,format=fmt,datefmt=datefmt)
        self.Dico={}
        self.Silent=False
        self._myfilter = LoggerMemoryFilter()


    def getLogger(self,name,disable=False):

        if not(name in self.Dico.keys()):
            logger = logging.getLogger(name)
            logger.addFilter(self._myfilter)
            fp = LoggerWriter(logger, logging.INFO)
            self.Dico[name]=fp
            
        #self.Dico[name].logger.log(logging.DEBUG, "Get Logger for: %s"%name)
        log=self.Dico[name]

            


        return log




    #logger2 = logging.getLogger("demo.X")
    #debug_fp = LoggerWriter(logger2, logging.DEBUG)
    #print>>fp, ModColor.Str("An INFO message")
    #print >> debug_fp, "A DEBUG message"
    #print >> debug_fp, 1

M=MyLogger()

getLogger=M.getLogger

itsLog=getLogger("MyLogger")
import ModColor
def setSilent(Lname):
    print>>itsLog, ModColor.Str("Set silent: %s"%Lname,col="red")
    if type(Lname)==str:
        log=getLogger(Lname)
        log.logger.setLevel(logging.CRITICAL)
    elif type(Lname)==list:
        for name in Lname:
            log=getLogger(name)
            log.logger.setLevel(logging.CRITICAL)


def setLoud(Lname):
    print>>itsLog, ModColor.Str("Set loud: %s"%Lname,col="green")
    if type(Lname)==str:
        log=getLogger(Lname)
        log.logger.setLevel(logging.DEBUG)
    elif type(Lname)==list:
        for name in Lname:
            log=getLogger(name)
            log.logger.setLevel(logging.DEBUG)


if __name__=="__main__":
    log=getLogger("a.x")
    print>>log, "a.x"
