# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

# This module has been adapted from the DDFacet package,
# (c) Cyril Tasse et al., see http://github.com/saopicc/DDFacet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

from six import string_types
import logging, logging.handlers, os, re, sys, multiprocessing
from . import ModColor

# dict of logger wrappers created by the application
_loggers = {}

# global verbosity levels (used for loggers for which an explicit level is not set)
_global_verbosity = 0
_global_log_verbosity = None

# this will be the handler for the log file
_file_handler = None
# this will be a null handler
_null_handler = logging.NullHandler()

def logToFile(filename, append=False):
    global _file_handler
    if not _file_handler:
        _file_handler = logging.FileHandler(filename, mode='a' if append else 'w')
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(_logfile_formatter)
        # set it as the target for the existing wrappers' handlers
        for wrapper in _loggers.values():
            wrapper.logfile_handler.setTarget(_file_handler)

def getLogFilename():
    '''Returns log filename if logToFile has been called previously, None otherwise'''
    global _file_handler
    if not _file_handler:
        return None
    return _file_handler.baseFilename

class _DefaultWriter(object):
    """A default writer logs messages to a logger"""
    def __init__(self, logger, level, color=None, bold=None):
        self.logger = logger
        self.level = level
        self.color = (color or "red") if bold else color
        self.bold = bool(color) if bold is None else bold

    def write(self, message):
        message = message.rstrip()
        if self.color and message:  # do not colorize empty messages, else "\n" is issued independently
            message = ModColor.Str(message, col=self.color, Bold=self.bold)
        self.logger.log(self.level, message)

    print = write

class LoggerWrapper(object):
    def __init__(self, logger, verbose=None, log_verbose=None):
        self.logger = logger
        logger.propagate = False

        # initialize handlers for console and logfile

        self.console_handler = logging.StreamHandler(sys.stderr)
        self.console_handler.setFormatter(_console_formatter)

        self.logfile_handler = logging.handlers.MemoryHandler(1, logging.DEBUG, _file_handler or _null_handler)
        self.logfile_handler.setFormatter(_logfile_formatter)

        # set verbosity levels
        self._verbose = self._log_verbose = None
        self.verbosity(verbose if verbose is not None else _global_verbosity)
        self.log_verbosity(log_verbose if log_verbose is not None else _global_log_verbosity)

        # other init

        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.logfile_handler)
        self.logger.addFilter(_log_filter)

        # make logger methods available
        for method in "log", "debug", "info", "warning", "error", "critical", "exception":
            setattr(self,method, getattr(self.logger, method))

    def verbosity(self, set_verb=None):
        if set_verb is not None:
            self._verbose = set_verb
            self.console_handler.setLevel(logging.INFO - set_verb)
            if self._log_verbose is None:
                self.logfile_handler.setLevel(logging.INFO - set_verb)
        return self._verbose

    def log_verbosity(self, set_verb=None):
        if set_verb is not None:
            self._log_verbose = set_verb
            self.logfile_handler.setLevel(logging.INFO - set_verb)
        return self._log_verbose if self._log_verbose is not None else self._verbose

    def __call__(self, level, color=None):
        """
        Function call operator on logger. Use to issue messages at different verbosity levels.
        E.g. log(2).print("message" will issue a message at level logging.INFO - 2.)
        An optional color argument will colorize the message.

        Returns:
            A writer object (to which a message may be sent with "<<")
        """
        # effective verbosity level is either set explicitly when the writer is created, or else use global level
        return _DefaultWriter(self.logger, logging.INFO - level, color=color)
    
    def warn(self, msg, color=None):
        """
        Wrapper for log.warn
        """
        _DefaultWriter(self.logger, logging.WARN, color=color).write(msg)
    
    def error(self, msg, color="red"):
        """
        Wrapper for log.error
        """
        _DefaultWriter(self.logger, logging.ERROR, color=color).write(msg)
        
    def info(self, msg, color=None):
        """
        Wrapper for log.info
        """
        _DefaultWriter(self.logger, logging.INFO, color=color).write(msg)
    
    print = info

    def write(self, message):
        return self.logger.info(message.rstrip())


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

_log_memory = False
def enableMemoryLogging(enable=True):
    global _log_memory
    _log_memory = enable

_subprocess_label = None

def set_subprocess_label(label):
    """
    Sets the subprocess label explicitly
    """
    global _subprocess_label
    _subprocess_label = label


def get_subprocess_label():
    """
    Returns the subprocess ID. For the main process, this is empty. For subprocesses
    (started by multiprocessing), this is "Pn" by default, where n is the process number.
    """
    global _subprocess_label
    if _subprocess_label is None:
        name = multiprocessing.current_process().name
        if name == "MainProcess":
            _subprocess_label = ""
        else:
            _subprocess_label = name.replace("Process-", "P")
    return _subprocess_label


class LogFilter(logging.Filter):
    """LogFilter augments the event by a few new attributes used by our formatter"""
    def filter(self, event):
        if not event.getMessage().strip():
            return False
        # short logger name (without app_name in front of it)
        setattr(event, 'shortname', event.name.split('.',1)[1] if '.' in event.name else event.name)
        setattr(event, 'separator', '| ')
        # memory usage info
        vss = float(_memory()/(1024**3))
        vss_peak = float(_memory_peak()/(1024**3))
        rss = float(_resident()/(1024**3))
        rss_peak = float(_resident_peak()/(1024**3))
        shm = float(_shmem_size()/(1024**3))
        setattr(event,"virtual_memory_gb",vss)
        setattr(event,"resident_memory_gb",rss)
        setattr(event,"shared_memory_gb",shm)
        if _log_memory:
            setattr(event, "memory", "[%.1f/%.1f %.1f/%.1f %.1fGb] "%(rss,rss_peak,vss,vss_peak,shm))
            setattr(event, 'separator', '')
        else:
            setattr(event, "memory", "")
        # subprocess info
        subprocess_id = get_subprocess_label()
        if subprocess_id:
            setattr(event, "subprocess", ModColor.Str("[%s] "%subprocess_id, col="blue"))
            setattr(event, 'separator', '')
        else:
            setattr(event, "subprocess", "")
        return True

_log_filter = LogFilter()

class ColorStrippingFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt, strip=True):
        logging.Formatter.__init__(self, fmt, datefmt)
        self.strip = strip

    def format(self, record):
        """Uses parent class to format record, then strips off colors"""
        msg = logging.Formatter.format(self, record)
        if self.strip:
            return re.sub("\033\\[[0-9]+m", "", msg, 0)
        else:
            return msg

#_fmt = " - %(asctime)s - %(shortname)-18.18s %(subprocess)s%(memory)s%(separator)s%(message)s"
_fmt = " - %(asctime)s - %(shortname)-28.28s %(subprocess)s%(memory)s%(separator)s%(message)s"
#        _fmt = "%(asctime)s %(name)-25.25s | %(message)s"
_datefmt = '%H:%M:%S'#'%H:%M:%S.%f'
_logfile_formatter = ColorStrippingFormatter(_fmt, _datefmt, strip=True)
_console_formatter = ColorStrippingFormatter(_fmt, _datefmt, strip=False)

_root_logger = None
log = None

def init(app_name):
    global _app_name
    global _root_logger
    if _root_logger is None:
        logging.basicConfig(datefmt=_datefmt)
        # logging.basicConfig(level=logging.DEBUG, datefmt=_datefmt)
        _app_name = app_name
        _root_logger = logging.getLogger(app_name)
        _root_logger.setLevel(logging.DEBUG)
        global log
        log = _loggers[''] = LoggerWrapper(_root_logger)

def getLogger(name, verbose=None, log_verbose=None, disable=False):
    """Creates a new logger (or returns one, if already created)

        The disable argument is no longer relevant
    """
    init("app")
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger("{}.{}".format(_app_name, name))
    lw = _loggers[name] = LoggerWrapper(logger, verbose, log_verbose)
    lw(2).print("logger initialized")

    return lw

def setBoring(boring=True):
    _console_formatter.strip = boring

def setGlobalVerbosity(verbosity):
    global _global_verbosity
    if verbosity is None:
        _global_verbosity = 0
        return
    # ensure verbosity is turned into a list.
    if type(verbosity) is int:
        verbosity = [verbosity]
    elif isinstance(verbosity, string_types):
        verbosity = verbosity.split(",")
    elif not isinstance(verbosity, (list, tuple)):
        raise TypeError("can't parse verbosity specification of type '{}'".format(type(verbosity)))
    for element in verbosity:
        if type(element) is int or re.match("^[0-9]+$", element):
            _global_verbosity = int(element)
            log(0, "green").print("set global console verbosity level {}".format(_global_verbosity))
        else:
            m = re.match("^(.+)=([0-9]+)$", element)
            if not m:
                raise ValueError("can't parse verbosity specification '{}'".format(element))
            logger = getLogger(m.group(1))
            level = int(m.group(2))
            logger.verbosity(level)
            logger(0,"green").print("set console verbosity level {}={}".format(m.group(1), level))

def setGlobalLogVerbosity(verbosity):
    global _global_log_verbosity
    if verbosity is None:
        _global_log_verbosity = None  # None means follow console default
        return
    # ensure verbosity is turned into a list.
    if type(verbosity) is int:
        verbosity = [verbosity]
    elif isinstance(verbosity, string_types):
        verbosity = verbosity.split(",")
    elif not isinstance(verbosity, (list, tuple)):
        raise TypeError("can't parse verbosity specification of type '{}'".format(type(verbosity)))
    for element in verbosity:
        if type(element) is int or re.match("^[0-9]+$", element):
            _global_log_verbosity = int(element)
            if _global_log_verbosity is not None:
                log(0, "green").print("set global log verbosity level {}".format(_global_log_verbosity))
        else:
            m = re.match("^(.+)=([0-9]+)$", element)
            if not m:
                raise ValueError("can't parse verbosity specification '{}'".format(element))
            logger = getLogger(m.group(1))
            level = int(m.group(2))
            logger.log_verbosity(level)
            logger(0,"green").print("set log verbosity level {}={}".format(m.group(1), level))


def setSilent(Lname):
    """Silences the specified sublogger(s)"""
    log(2).print(ModColor.Str("set silent: %s" % Lname, col="red"))
    if isinstance(Lname, string_types):
        getLogger(Lname).logger.setLevel(logging.CRITICAL)
    elif type(Lname) is list:
        for name in Lname:
            getLogger(name).logger.setLevel(logging.CRITICAL)


def setLoud(Lname):
    """Un-silences the specified sublogger(s)"""
    log(2).print(ModColor.Str("set loud: %s" % Lname, col="green"))
    if isinstance(Lname, string_types):
        getLogger(Lname).logger.setLevel(logging.DEBUG)
    elif type(Lname) is list:
        for name in Lname:
            getLogger(name).logger.setLevel(logging.DEBUG)


if __name__=="__main__":
    log=getLogger("a.x")
    log.print("a.x")
