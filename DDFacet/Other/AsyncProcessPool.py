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

import psutil
import os
import fnmatch
import Queue
import multiprocessing
import numpy as np
import traceback
from collections import OrderedDict

from DDFacet.Other import MyLogger
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Array import SharedDict

log = MyLogger.getLogger("AsyncProcessPool")

# PID of parent process
parent_pid = os.getpid()

class Job(object):
    def __init__ (self, job_id, jobitem, singleton=False, event=None, when_complete=None):
        self.job_id, self.jobitem, self.singleton, self.event, self.when_complete = \
            job_id, jobitem, singleton, event, (when_complete or (lambda:None))
        self.result = None
        self.complete = False

    def setResult (self, result):
        self.result = result
        self.complete = True
        self.when_complete()

class JobCounterPool(object):
    """Implements a condition variable that is a counter. Typically used to keep track of the number of pending jobs
    of a particular type, and to block until all are complete"""

    def __init__(self):
        """Creates a named counter, stores it in global map"""
        self._counters = OrderedDict()
        self._counters_array = None

    def start(self, shared_dict):
        """Called in parent process to complete initialization of all counters"""
        if os.getpid() != parent_pid:
            raise RuntimeError("This method can only be called in the parent process. This is a bug.")
        if self._counters_array is not None:
            raise RuntimeError("Workers already started. This is a bug.")
        self._counters_array = shared_dict.addSharedArray("Counters", (len(self._counters),), np.int32)

    def add(self, name):
        """Adds a named counter to the pool"""
        if name in self._counters:
            raise RuntimeError,"job counter %s already exists. This is a bug."%name
        if os.getpid() != parent_pid:
            raise RuntimeError("This method can only be called in the parent process. This is a bug.")
        if self._counters_array is not None:
            raise RuntimeError("Workers already started. This is a bug.")
        index = len(self._counters)
        self._counters[name] = multiprocessing.Condition(), index

    def increment(self, name):
        """Increments the named counter"""
        condition, index = self._counters[name]
        with condition:  # acquire lock
            self._counters_array[index] += 1

    def decrement(self, name):
        """Decrements the named counter. When it gets to zero, notifies any waiting processes."""
        condition, index = self._counters[name]
        with condition:  # acquire lock
            self._counters_array[index] -= 1
            # if decremented to 0, notify callers
            if self._counters_array[index] <= 0:
                condition.notify_all()

    def await(self, name):
        condition, index = self._counters[name]
        with condition:  # acquire lock
            while self._counters_array[index] > 0:
                condition.wait()


class AsyncProcessPool (object):
    """
    """
    def __init__ (self, ncpu=None, affinity=None, num_io_processes=1, verbose=0, job_handlers={}, event_list={}):
        self._shared_state = SharedDict.create("APP")
        self.affinity = affinity
        self.cpustep = abs(self.affinity) or 1
        self.ncpu = ncpu
        self.verbose = verbose
        maxcpu = psutil.cpu_count() / self.cpustep
        # if NCPU is 0, set to number of CPUs on system
        if not self.ncpu:
            self.ncpu = maxcpu
        elif self.ncpu > maxcpu:
            print>>log,ModColor.Str("NCPU=%d is too high for this setup (%d cores, affinity %d)" %
                                    (self.ncpu, psutil.cpu_count(), self.affinity))
            print>>log,ModColor.Str("Falling back to NCPU=%d" % (maxcpu))
            self.ncpu = maxcpu
        self.procinfo = psutil.Process()  # this will be used to control CPU affinity

        # create a queue for compute-bound tasks
        # generate list of CPU cores for workers to run on
        if not self.affinity or self.affinity == 1:
            cores = range(self.ncpu)
        elif self.affinity == 2:
            cores = range(0, self.ncpu*2, 2)
        elif self.affinity == -2:
            cores = range(0, self.ncpu*2, 4) + range(1, self.ncpu*2, 4)
        else:
            raise ValueError,"unknown affinity setting %d" % self.affinity
        self._compute_workers = []
        self._io_workers = []
        self._compute_queue   = multiprocessing.Queue()
        self._io_queues       = [ multiprocessing.Queue() for x in xrange(num_io_processes) ]
        self._result_queue    = multiprocessing.Queue()
        self._job_handlers   = job_handlers

        # start the workers
        for i, core in enumerate(cores):
            proc_id = "compute%02d"%i
            self._compute_workers.append( multiprocessing.Process(target=self._start_worker, args=(self, proc_id, [core], self._compute_queue)) )
        for i, queue in enumerate(self._io_queues):
            proc_id = "io%02d"%i
            self._io_workers.append( multiprocessing.Process(target=self._start_worker, args=(self, proc_id, None, queue)) )
        self._results_map = {}
        self._events = dict([(name,multiprocessing.Event()) for name in event_list])
        self._job_counters = JobCounterPool()
        self._started = False

    def registerJobHandlers (self, **kw):
        if os.getpid() != parent_pid:
            raise RuntimeError("This method can only be called in the parent process. This is a bug.")
        if self._started:
            raise RuntimeError("Workers already started. This is a bug.")
        self._job_handlers.update(kw)

    def registerEvents (self, *args):
        if os.getpid() != parent_pid:
            raise RuntimeError("This method can only be called in the parent process. This is a bug.")
        if self._started:
            raise RuntimeError("Workers already started. This is a bug.")
        self._events.update(dict([(name,multiprocessing.Event()) for name in args]))

    def registerJobCounters (self, *args):
        if os.getpid() != parent_pid:
            raise RuntimeError("This method can only be called in the parent process. This is a bug.")
        if self._started:
            raise RuntimeError("Workers already started. This is a bug.")
        for name in args:
            self._job_counters.add(name)

    def startWorkers(self):
        """Starts worker threads. All job handlers and events must be registered *BEFORE*"""
        self._job_counters.start(self._shared_state)
        for proc in self._compute_workers + self._io_workers:
            proc.start()
        self._started = True

    def runJob (self, job_id, function=None, io=None, args=(), kwargs={},
                event=None, counter=None,
                singleton=False, collect_result=True):
        """
        Puts a job on a processing queue.

        Args:
            job_id:  string job identifier
            function: object.method or function to call. Must be an entry in the job handlers dict.
            io:     if None, job is placed on compute queues. If 0/1/..., job is placed on an I/O queue of the given level
            event:  if not None, then the named event will be raised when the job is complete.
                    Otherwise, the job is a singleton, handled via the events directory.
            counter: if not None, the job will be associated with a job counter, which will be incremented upon runJob(),
                    and decremented when the job is complete.
            collect_result: if True, job's result will be collected and returned via awaitJobResults().
                    This mode is only available in the parent process.
            singleton: if True, then job is a one-off. If collect_result=True, then when complete, its result will remain
                    in the results map forever, so that subsequent calls to awaitJobResults() on it return that result.
                    A singleton job can't be run again.
                    If False, job result will be collected by awaitJobResults() and removed from the map: the job can be
                    run again.
            when_complete: called in parent thread when the job is complete
        """
        if collect_result and os.getpid() != parent_pid:
            raise RuntimeError("runJob() with collect_result can only be called in the parent process. This is a bug.")
        if collect_result and job_id in self._results_map:
            raise KeyError("Job '%s' has an uncollected result, or is a singleton. This is a bug."%job_id)
        # form up job item
        if '.' in function:
            object, method = function.split('.',1)
        else:
            object, method = function, None
        if object not in self._job_handlers:
            raise KeyError("Unknown job handler '%s'" % objname)
        # resolve event object
        if event:
            eventobj = self._events[event]
            eventobj.clear()
        else:
            eventobj = None
        # increment counter object
        if counter:
            self._job_counters.increment(counter)
        jobitem = dict(job_id=job_id, object=object, method=method, event=event, counter=counter,
                       collect_result=collect_result,
                       args=args, kwargs=kwargs)
        if self.verbose > 2:
            print>>log, "enqueueing job %s: %s"%(job_id, function)
        # place it on appropriate queue
        if io is None:
            self._compute_queue.put(jobitem)
        else:
            io = max(len(self._io_queues)-1, io)
            self._io_queues[io].put(jobitem)
        # insert entry into dict of pending jobs
        if collect_result:
            self._results_map[job_id] = Job(job_id, jobitem, singleton=singleton)

    def awaitJobCompletion (self, *counters):
        if self.verbose > 2:
            print>>log, "checking for job completion counters %s" % ", ".join(counters)
        for name in counters:
            self._job_counters.await(name)
            if self.verbose > 2:
                print>> log, "  %s is complete" % name

    def awaitEvents (self, *events):
        """
        Waits for events indicated by the given names to be set. This can be called from the parent process, or
        from any of the background processes.
        """
        if self.verbose > 2:
            print>>log, "checking for completion events on %s" % " ".join(events)
        for name in events:
            event = self._events.get(name)
            if event is None:
                raise KeyError("Unknown event '%s'" % name)
            while not event.is_set():
                if self.verbose > 2:
                    print>> log, "  %s not yet complete, waiting" % name
                event.wait()
            if self.verbose > 2:
                print>> log, "  %s is complete" % name

    def awaitJobResults (self, *jobspecs):
        """
        Waits for job(s) given by arguments to complete, and returns their results.
        Note that this only works for jobs scheduled by the same process, since each process has its own results map.
        A process will block indefinitely is asked to await on jobs scheduled by another process.

        Args:
            *jobs: list of job IDs. Each ID can contain a wildcard e.g. "job*", to wait for multiple jobs.

        Returns:
            a list of results. Each entry is the result returned by the job (if no wildcard), or a list
            of results from each job (if has a wildcard)
        """
        if os.getpid() != parent_pid:
            raise RuntimeError("This method can only be called in the parent process. This is a bug.")
        # make a dict of all jobs still outstanding
        awaiting_jobs = {}  # this maps job_id to a set of jobspecs (if multiple) that it matches
        job_results = OrderedDict()   # this maps jobspec to a list of results
        for jobspec in jobspecs:
            matching_jobs = [job_id for job_id in self._results_map.iterkeys() if fnmatch.fnmatch(job_id, jobspec)]
            for job_id in matching_jobs:
                awaiting_jobs.setdefault(job_id, set()).add(jobspec)
            job_results[jobspec] = len(matching_jobs), []
        # check dict of already returned results (perhaps from previous calls to awaitJobs). Remove
        # matching results, and assign them to appropriate jobspec lists
        for job_id, job in self._results_map.items():
            if job_id in awaiting_jobs and job.complete:
                for jobspec in awaiting_jobs[job_id]:
                    job_results[jobspec][1].append(job.result)
                if not job.singleton:
                    del self._results_map[job_id]
                del awaiting_jobs[job_id]
        if self.verbose > 1:
            print>>log, "checking job results: %s (%d still pending)"%(
                ", ".join(["%s %d/%d"%(jobspec, len(results), njobs) for jobspec, (njobs, results) in job_results.iteritems()]),
                len(awaiting_jobs))
        # sit here while any pending jobs remain
        while awaiting_jobs:
            try:
                result = self._result_queue.get(True, 10)
            except Queue.Empty:
                # print>> log, "checking for dead workers"
                # shoot the zombie process, if any
                multiprocessing.active_children()
                # check for dead workers
                pids_to_restart = []
                for w in self._compute_workers + self._io_workers:
                    if not w.is_alive():
                        pids_to_restart.append(w)
                        raise RuntimeError("a worker process has died on us \
                            with exit code %d. This is probably a bug." % w.exitcode)
            # ok, dispatch the result
            job_id = result["job_id"]
            job = self._results_map.get(job_id)
            if job is None:
                raise KeyError("Job '%s' was not enqueued. This is a logic error." % job_id)
            job.setResult(result)
            # if being awaited, dispatch appropriately
            if job_id in awaiting_jobs:
                for jobspec in awaiting_jobs[job_id]:
                    job_results[jobspec][1].append(result)
                if not job.singleton:
                    del self._results_map[job_id]
                del awaiting_jobs[job_id]
            # print status update
            if self.verbose > 1:
                print>>log,"received job results %s" % " ".join(["%s:%d"%(jobspec, len(results)) for jobspec, (_, results)
                                                             in job_results.iteritems()])
        # process list of results for each jobspec to check for errors
        for jobspec, (njobs, results) in job_results.iteritems():
            times = np.array([ res['time'] for res in results ])
            num_errors = len([result for res in results if not res['success']])
            if self.verbose > 0:
                print>> log, "%s: %d jobs complete, average single-core time %.2fs per job" % (jobspec, len(results), times.mean())
            if num_errors:
                print>>log, ModColor.Str("%s: %d jobs returned an error. Aborting."%(jobspec, num_errors), col="red")
                raise RuntimeError("some distributed jobs have failed")
        # return list of results
        return [ [results if '*' in jobspec else results[0]] for jobspec, results in job_results.iteritems() ]

    @staticmethod
    def _start_worker (object, proc_id, affinity, worker_queue):
        """
            Helper method for worker process startup. ets up affinity, and calls _run_worker method on
            object with the specified work queue.

        Args:
            object:
            proc_id:
            affinity:
            work_queue:

        Returns:

        """
        AsyncProcessPool.proc_id = proc_id
        MyLogger.subprocess_id = proc_id
        if affinity:
            psutil.Process().cpu_affinity(affinity)
        object._run_worker(worker_queue)

    def _run_worker (self, queue):
        """
            Runs worker loop on given queue. Waits on queue, picks off job items, looks them up in context table,
            calls them, and returns results in the work queue.
        """
        try:
            pill = True
            # While no poisoned pill has been given grab items from the queue.
            while pill:
                try:
                    # Get queue item, or timeout and check if pill perscribed.
                    # print>>log,"%s: calling queue.get()"%AsyncProcessPool.proc_id
                    jobitem = queue.get(True, 10)
                    # print>>log,"%s: queue.get() returns %s"%(AsyncProcessPool.proc_id, jobitem)
                except Queue.Empty:
                    continue
                timer = ClassTimeIt.ClassTimeIt()
                if jobitem == "POISON-E":
                    break
                elif jobitem is not None:
                    event = None
                    job_id, objname, method, eventname, counter, args, kwargs = [ jobitem.get(attr)
                                            for attr in "job_id", "object", "method", "event", "counter", "args", "kwargs" ]
                    obj = self._job_handlers.get(objname)
                    try:
                        if obj is None:
                            raise KeyError("Unknown jobitem object '%s'"%objname)
                        event = self._events[eventname] if eventname else None
                        if self.verbose > 1:
                            print>> log, "job %s: calling %s.%s" % (job_id, objname, method)
                        if method is None:
                            # call object directly
                            result = obj(*args, **kwargs)
                        else:
                            call = getattr(obj, method, None)
                            if call is None:
                                raise KeyError("Unknown method '%s' of object '%s'"%(method, objname))
                            result = call(*args, **kwargs)
                        if self.verbose > 3:
                            print>> log, "%s.%s result is %s" % (objname, method, result)
                        # Send result back
                        if jobitem['collect_result']:
                            self._result_queue.put(dict(job_id=job_id, proc_id=self.proc_id, success=True, result=result, time=timer.seconds()))
                    except KeyboardInterrupt:
                        raise
                    except Exception, exc:
                        print>> log, ModColor.Str("process %s: exception raised processing job %s: %s" % (
                            AsyncProcessPool.proc_id, job_id, traceback.format_exc()))
                        if jobitem['collect_result']:
                            self._result_queue.put(dict(job_id=job_id, proc_id=self.proc_id, success=False, error=exc, time=timer.seconds()))
                    finally:
                        # Raise event
                        if event is not None:
                            event.set()
                        if counter is not None:
                            self._job_counters.decrement(counter)

        except KeyboardInterrupt:
            print>>log, ModColor.Str("Ctrl+C caught, exiting", col="red")
            return
    # CPU id. This will be None in the parent process, and a unique number in each worker process
    proc_id = None
