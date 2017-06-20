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

import logging
import os
import ModColor

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message != '\n':
            self.logger.log(self.level, message)


_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey,statusfile=None):
    global _scale
    _proc_status = '/proc/%d/status' % os.getpid()
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
    '''Return shared memory usage in bytes.'''
    return _VmB('Shmem:','/proc/meminfo') - since

def _memory(since=0.0):
    '''Return memory usage in bytes.'''
    return _VmB('VmSize:') - since

def _memory_peak(since=0.0):
    '''Return memory usage in bytes.'''
    return _VmB('VmPeak:') - since

def _resident(since=0.0):
    '''Return resident memory usage in bytes.'''
    return _VmB('VmRSS:') - since

def _resident_peak(since=0.0):
    '''Return resident memory usage in bytes.'''
    return _VmB('VmHWM:') - since

def _stacksize(since=0.0):
    '''Return stack size in bytes.'''
    return _VmB('VmStk:') - since

log_memory = False
file_handler = None

# if set, ID will be included in log messages
subprocess_id = None

def enableMemoryLogging (enable=True):
    global log_memory
    log_memory = enable

def logToFile (filename,append=False):
    global file_handler
    if not file_handler:
        file_handler = logging.FileHandler(filename,mode='a' if append else 'w')
    logging.getLogger('').addHandler(file_handler)

def getLogFilename():
    '''Returns log filename if logToFile has been called previously, None otherwise'''
    global file_handler
    if not file_handler:
        return None
    return file_handler.baseFilename

class LoggerMemoryFilter (logging.Filter):
    def filter(self, event):
        vss = float(_memory()/(1024**3))
        vss_peak = float(_memory_peak()/(1024**3))
        rss = float(_resident()/(1024**3))
        rss_peak = float(_resident_peak()/(1024**3))
        shm = float(_shmem_size()/(1024**3))
        setattr(event,"virtual_memory_gb",vss)
        setattr(event,"resident_memory_gb",rss)
        setattr(event,"shared_memory_gb",shm)
        if log_memory and hasattr(event,"msg"):
            event.msg = "[%.1f/%.1f %.1f/%.1f %.1fGb] "%(rss,rss_peak,vss,vss_peak,shm) + event.msg
        if subprocess_id:
            event.msg = ModColor.Str("[%s] "%subprocess_id, col="blue") + event.msg
        return True


class MyLogger():
    def __init__(self):
#       fmt="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
        fmt = " - %(asctime)s - %(name)-25.25s | %(message)s"
#        fmt = "%(asctime)s %(name)-25.25s | %(message)s"
        datefmt = '%H:%M:%S'#'%H:%M:%S.%f'
        logging.basicConfig(level=logging.DEBUG,format=fmt,datefmt=datefmt)
        self.Dico={}
        self.Silent=False
        self._myfilter = LoggerMemoryFilter()
        self._formatter = logging.Formatter(fmt, datefmt)


    def getLogger(self,name,disable=False):

        if not(name in self.Dico.keys()):
            logger = logging.getLogger(name)
            logger.addFilter(self._myfilter)
            if file_handler:
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(self._formatter)
                # logger.addHandler(file_handler)
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
    # print>>itsLog, ModColor.Str("Set silent: %s" % Lname, col="red")
    if type(Lname)==str:
        log=getLogger(Lname)
        log.logger.setLevel(logging.CRITICAL)
    elif type(Lname)==list:
        for name in Lname:
            log=getLogger(name)
            log.logger.setLevel(logging.CRITICAL)


def setLoud(Lname):
    print>>itsLog, ModColor.Str("Set loud: %s" % Lname, col="green")
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
