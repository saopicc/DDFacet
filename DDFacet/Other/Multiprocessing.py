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
RuntimeWarning("Deprecated class")

import psutil
import os, re, errno
import Queue
import multiprocessing
import numpy as np
import glob
import re

from DDFacet.Other import MyLogger
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Array import NpShared
from DDFacet.Array import shared_dict

log = MyLogger.getLogger("Multiprocessing")
#MyLogger.setSilent("Multiprocessing")

# store PID here, so that it corresponds to the PID of the parent thread (when we fork off child processes)
_pid = os.getpid()

def getShmPrefix():
    """Returns prefix used for shared memory arrays. ddf.PID is the convention"""
    return "ddf.%d" % _pid

# init SharedDict with the same base name
shared_dict.SharedDict.setBaseName(getShmPrefix())

def getShmName(name, **kw):
    """
    Forms up a name for a shm-backed shared element. This takes the form of "ddf.PID.", where PID is the
    pid of the process where the cache manager was created (so the parent process, presumably), followed
    by a filename of the form "NAME:KEY1_VALUE1:...", as returned by getElementName(). See getElementName()
    for usage.
    """
    # join keyword args into "key=value:key=value:..."
    kws = ":".join([name] + ["%s_%s" % (key, value) for key, value in sorted(kw.items())])
    return "%s.%s" % (getShmPrefix(), kws)

def cleanupShm ():
    """
    Deletes all shared arrays for this process
    """
    NpShared.DelAll(getShmPrefix())
    # above statement don't work for directories and subdirectories
    os.system("rm -rf /dev/shm/%s"%getShmPrefix())

def cleanupStaleShm ():
    """
    Cleans up "stale" shared memory from previous runs of DDF
    """
    # check for stale shared memory
    uid = os.getuid()
    # list of all files in /dev/shm/ matching ddf.PID.* and belonging to us
    shmlist = [ ("/dev/shm/"+filename, re.match('(sem\.)?ddf\.([0-9]+)(\..*)?$',filename)) for filename in os.listdir("/dev/shm/")
                if os.stat("/dev/shm/"+filename).st_uid == uid ]
    # convert to list of filename,pid tuples
    shmlist = [ (filename, int(match.group(2))) for filename, match in shmlist if match ]
    # now check all PIDs to find dead ones
    # if we get ESRC error from sending signal 0 to the process, it's not running, so we mark it as dead
    dead_pids = set()
    for pid in set([x[1] for x in shmlist]):
        try:
            os.kill(pid, 0)
        except OSError, err:
            if err.errno == errno.ESRCH:
                dead_pids.add(pid)
    # ok, make list of candidates for deletion
    victims = [ filename for filename,pid in shmlist if pid in dead_pids ]
    if victims:
        print>>log, "reaping %d shared memory objects associated with %d dead DDFacet processes"%(len(victims), len(dead_pids))
        dirs = [ v for v in victims if os.path.isdir(v) ]
        files = [ v for v in victims if not os.path.isdir(v) ]
        # rm -fr only works for a limited number of arguments (which the semaphore list can easily exceed)
        # so use os.unlink() to remove files, and rm -fr for directories
        for path in files:
            os.unlink(path)
        os.system("rm -fr " + " ".join(dirs))
        # print "rm -fr " + " ".join(victims)


def getShmURL(name, **kw):
    """
    Forms up a URL for a shm-backed shared element. This takes the form of "shm://" plus getShmName()
    """
    return "shm://" + getShmName(name, **kw)

# This instantiates threads that are never used
#
# class ProcessPool (object):
#     """
#     ProcessPool implements a pool of forked child processes which can execute jobs (defined by a worker
#     function) in paralellel.
#     """
#     def __init__ (self, GD=None, ncpu=None, affinity=None):
#         self.GD = GD
#         self.affinity = self.GD["Parallel"]["Affinity"] if affinity is None else affinity
#         if isinstance(self.affinity, int):
#             self.cpustep = abs(self.affinity) or 1
#             self.ncpu = self.GD["Parallel"]["NCPU"] if ncpu is None else ncpu
#             maxcpu = psutil.cpu_count() / self.cpustep
#         elif isinstance(self.affinity, list):
#             if any(map(lambda x: x < 0, self.affinity)):
#                 raise RuntimeError("Affinities must be list of positive numbers")
#             if psutil.cpu_count() < max(self.affinity):
#                 raise RuntimeError("There are %d virtual threads on this system. Some elements of the affinity map are "
#                                    "higher than this. Check parset." % psutil.cpu_count())
#             self.ncpu = self.GD["Parallel"]["NCPU"] if ncpu is None else ncpu
#             if self.ncpu != len(self.affinity):
#                 print>> log, ModColor.Str("Warning: NCPU does not match affinity list length. Falling back to "
#                                           "NCPU=%d" % len(self.affinity))
#             self.ncpu = self.ncpu if self.ncpu == len(self.affinity) else len(self.affinity)
#             maxcpu = max(self.affinity) + 1 # zero indexed list
#
#         elif isinstance(self.affinity, str) and str(self.affinity) == "autodetect":
#             # this works on Ubuntu so possibly Debian-like systems, no guarantees for the rest
#             # the mapping on hyperthread-enabled NUMA machines with multiple processors can get very tricky
#             # /sys/devices/system/cpu/cpu*/topology/thread_siblings_list should give us a list of siblings
#             # whereas core_siblings_list will give the list of sibling threads on the same physical processor
#             # for now lets not worry about assigning different jobs to physical cpus but for now keep things
#             # simple
#             hyperthread_sibling_lists = map(lambda x: x + "/topology/thread_siblings_list",
#                                             filter(lambda x: re.match(r"cpu[0-9]+", os.path.basename(x)),
#                                                    glob.glob("/sys/devices/system/cpu/cpu*")))
#             left_set = set([])
#             right_set = set([])
#             for siblings_file in hyperthread_sibling_lists:
#                 with open(siblings_file) as f:
#                     siblings = map(int, f.readline().split(","))
#                     if len(siblings) == 2:
#                         l, r = siblings
#                         # cannot be sure the indices don't swap around at some point
#                         # since there are two copies of the siblings we need to
#                         # check that the items are not in the other sets before adding
#                         # them to the left and right sets respectively
#                         if l not in right_set:
#                             left_set.add(l)
#                         if r not in left_set:
#                             right_set.add(r)
#                     elif len(siblings) == 1:
#                         left_set.add(siblings[0])
#                     else:
#                         raise RuntimeError("Don't know how to handle this architecture. It seems there are more than "
#                                            "two threads per core? Try setting things manually by specifying a list "
#                                            "to the affinity option")
#
#             self.affinity = list(left_set) # only consider 1 thread per core
#             self.ncpu = self.GD["Parallel"]["NCPU"] if ncpu is None else ncpu
#             if self.ncpu >= len(self.affinity):
#                 print>> log, ModColor.Str("Warning: NCPU is more than the number of physical cores on "
#                                           "the system. I will only use %d cores." % len(self.affinity))
#             self.ncpu = self.ncpu if self.ncpu <= len(self.affinity) else len(self.affinity)
#             maxcpu = max(self.affinity) + 1 # zero indexed list
#         else:
#             raise RuntimeError("Invalid option for Parallel.Affinity. Expected cpu step (int), list or "
#                                "'autodetect'")
#         # if NCPU is 0, use the maximum number of CPUs
#         if not self.ncpu:
#             self.ncpu = maxcpu
#         elif self.ncpu > maxcpu:
#             print>>log,ModColor.Str("NCPU=%d is too high for this setup (%d cores, affinity %s)" %
#                                     (self.ncpu, psutil.cpu_count(),
#                                      self.affinity if isinstance(self.affinity, int)
#                                         else ",".join(map(str,self.affinity))))
#             print>>log,ModColor.Str("Falling back to NCPU=%d" % (maxcpu))
#             self.ncpu = maxcpu
#         self.procinfo = psutil.Process()  # this will be used to control CPU affinity
#         # not convinced by the work producer pattern so this flag can enable/disable it
#         self._create_work_producer = False
#
#
#     def runjobs (self, joblist, target, args=(), kwargs={}, result_callback=None, title=None, pause_on_start=False):
#         parallel = self.ncpu > 1
#         procs = list()  # list of processes that are running
#
#         # set up queues
#         if parallel:
#             if len(joblist) > 10000000:   # Raz had it at 1000, but I don't buy it
#                 qlimit = self.ncpu * 8
#             else:
#                 qlimit = 0
#             m_work_queue = multiprocessing.Queue(maxsize=qlimit)
#         else:
#             m_work_queue = multiprocessing.Queue()
#         m_result_queue = multiprocessing.JoinableQueue()
#
#         # create work producer process
#         if self._create_work_producer:
#             work_p = multiprocessing.Process(target=self._work_producer, args=(m_work_queue, joblist,))
#         else:
#             work_p = None
#             for item in joblist:
#                 m_work_queue.put(item)  # possible issue if no space for 60 seconds
#
#         # generate list of CPU cores for workers to run on
#         if isinstance(self.affinity, int) and (not self.affinity or self.affinity == 1):
#             cores = range(self.ncpu)
#         elif isinstance(self.affinity, int) and self.affinity == 2:
#             cores = range(0, self.ncpu*2, 2)
#         elif isinstance(self.affinity, int) and self.affinity == -2:
#             cores = range(0, self.ncpu*2, 4) + range(1, self.ncpu*2, 4)
#         elif isinstance(self.affinity, list):
#             cores = self.affinity[:self.ncpu]
#         else:
#             raise ValueError, "unknown affinity setting"
#
#         # if fewer jobs than workers, reduce number of cores
#         cores = cores[:len(joblist)]
#         # create worker processes
#         for cpu in cores:
#             p = multiprocessing.Process(target=self._work_consumer,
#                 kwargs=dict(m_work_queue=m_work_queue, m_result_queue=m_result_queue,
#                 cpu=cpu, affinity=[cpu] if self.affinity else None,
#                 target=target, args=args, kwargs=kwargs))
#             procs.append(p)
#
#         print>> log, "%s: starting %d workers for %d jobs%s" % (title or "", len(cores), len(joblist),
#                         (", CPU cores " + " ".join(map(str,cores)) if self.affinity else ""))
#         # fork off child processes
#         if parallel:
#             for p in procs:
#                 p.start()
#
#         njobs = len(joblist)
#         iResult = 0
#
#         timer = ClassTimeIt.ClassTimeIt()
#         # create progress bar
#         if title:
#             pBAR = ProgressBar(Title="  %s "%title)
#             # pBAR.disable()
#             pBAR.render(0, '%4i/%i' % (0, njobs))
#         else:
#             pBAR = None
#
#         # in serial mode, just run both things to completion
#         if not parallel:
#             work_p and work_p.run()
#             for p in procs:
#                 m_work_queue.put("POISON-E")
#                 p.run()  # just run until all work is completed
#
#         # process results
#         results = []
#         while iResult < njobs:
#             result = None
#             try:
#                 result = m_result_queue.get(True, 10)
#             except Queue.Empty:
#                 # print>> log, "checking for dead workers"
#                 # shoot the zombie process
#                 multiprocessing.active_children()
#                 # check for dead workers
#                 pids_to_restart = []
#                 for w in procs:
#                     if not w.is_alive():
#                         pids_to_restart.append(w)
#                         raise RuntimeError("a worker process has died on us \
#                             with exit code %d. This is probably a bug." %
#                                            w.exitcode)
#                         #
#                         # for id in pids_to_restart:
#                         #     print>> log, "need to restart worker %d." % id
#                         #     pass
#
#             if result is not None and result["Success"]:
#                 iResult += 1
#                 m_result_queue.task_done()  # call task_done on the queue
#                 if result_callback:
#                     result_callback(iResult, result)
#                 results.append(result)
#             ## nothing returned? That's fine, it means the workers are just busy and we
#             ## timed out (10s timeout to update progress bar)
#             # else:
#             #     print>> log, "work_consumer: returned no result"
#             # update progress bar
#             if pBAR:
#                 intPercent = int(100 * iResult / float(njobs))
#                 pBAR.render(intPercent, '%4i/%i' % (iResult, njobs))
#
#         # if all work is done, send poison pill to workers
#         if parallel:
#             for p in procs:
#                 m_work_queue.put("POISON-E")
#
#             # join and close queues
#             m_result_queue.join()
#             m_work_queue.close()
#             m_result_queue.close()
#
#             # join producer process
#             work_p and work_p.join()
#
#             # join consumerr processes
#             for p in procs:
#                 p.join()
#
#         # compute average processing time
#         average_time = np.array([ r["Time"] for r in results ]).mean()
#
#         print>> log, "%s finished in %s, average (single core) time %.2fs per job" % (title or "", timer.timehms(), average_time)
#
#         # extract list of result objects
#         return [ r["Result"] for r in results ]
#
#     @staticmethod
#     def _work_producer(queue, data):
#         """Producer worker for ProcessPool"""
#         for item in data:
#             try:
#                 queue.put(item, 60)  # possible issue if no space for 60 seconds
#             except Queue.Full:
#                 print>> log, "work_producer: queue full"
#                 pass
#
#
#     # CPU id. This will be None in the parent process, and a unique number in each worker process
#     cpu_id = None
#
#     @staticmethod
#     def getCPUId ():
#         return ProcessPool.cpu_id
#
#     @staticmethod
#     def _work_consumer(m_work_queue, m_result_queue, cpu, affinity, target, args=(), kwargs={}):
#         """Consumer worker for ProcessPool"""
#         ProcessPool.cpu_id = cpu
#         if affinity:
#             psutil.Process().cpu_affinity(affinity)
#         timer = ClassTimeIt.ClassTimeIt()
#         pill = True
#         # While no poisoned pill has been given grab items from the queue.
#         while pill:
#             try:
#                 # Get queue item, or timeout and check if pill perscribed.
#                 jobitem = m_work_queue.get(True, 5)
#             except Queue.Empty:
#                 pass
#             else:
#                 if jobitem == "POISON-E":
#                     break
#                 elif jobitem is not None:
#                     result = target(jobitem, *args, **kwargs)
#                     # Send result back
#                     m_result_queue.put(dict(Success=True, Time=timer.seconds(), Result=result))
#
#
# # init default process pool
# def initDefaultPool(GD=None, ncpu=None, affinity=None):
#     global default_pool
#     global runjobs
#     default_pool = ProcessPool(GD, ncpu, affinity)
#     runjobs = default_pool.runjobs
#
#
# initDefaultPool(ncpu=0, affinity=0)



