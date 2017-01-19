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
import os, re, errno
import Queue
import multiprocessing
import numpy as np
import traceback

from DDFacet.Other import MyLogger
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Array import NpShared

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


class AsyncProcessPool (object):
    """
    """
    def __init__ (self, ncpu=None, affinity=None, num_io_processes=1,
                  job_handlers={}, event_list={}):

        self.affinity = self.GD["Parallel"]["Affinity"] if affinity is None else affinity
        self.cpustep = abs(self.affinity) or 1
        self.ncpu = self.GD["Parallel"]["NCPU"] if ncpu is None else ncpu
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
        self._jobs = {}
        self._events = dict([(name,multiprocessing.Event()) for name in event_list])
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

    def startWorkers(self):
        """Starts worker threads. All job handlers and events must be registered *BEFORE*"""
        if self._started:
            raise RuntimeError("Workers already started. This is a bug.")
        for proc in self._compute_workers + self._io_workers:
            proc.start()
        self._started = True

    def runJob (self, job_id, function=None, io=None, args=(), kwargs={}, event=None, when_complete=None):
        """
        Puts a job on a processing queue.

        Args:
            job_id:  string job identifier
            function: object.method or function to call. Must be an entry in the job handlers dict.
            io:     if None, job is placed on compute queues. If 0/1/..., job is placed on an I/O queue of the given level
            event:  if None, then job will be handled via the result queue.
                    Otherwise, the job is a singleton, handled via the events directory.
            when_complete:
        """
        if job_id in self._jobs:
            raise KeyError("Job '%s' already (or still) enqueued. This is a logic error."%job_id)
        # form up job item
        if '.' in function:
            object, method = function.split('.',1)
        else:
            object, method = function, None
        if object not in self._job_handlers:
            raise KeyError("Unknown jobitem object '%s'" % objname)
        if event:
            if event in self._events:
                eventobj = self._events[event]
                eventobj.clear()
            else:
                raise KeyError("Unknown job completion event '%s'" % event)
        else:
            eventobj = None
        jobitem = dict(job_id=job_id, object=object, method=method, event=event, args=args, kwargs=kwargs)
        print>>log, "enqueueing job %s"%function
        # place it on appropriate queue
        if io is None:
            self._compute_queue.put(jobitem)
        else:
            io = max(len(self._io_queues)-1, io)
            self._io_queues[io].put(jobitem)
        # insert entry into dict of pending jobs
        self._jobs[job_id] = Job(job_id, jobitem, singleton=not event, event=eventobj, when_complete=when_complete)

    def awaitEvents (self, *events):
        """
        Waits for events indicated by the given names to be set. This can be called from the parent process, or
        from any of the background processes.
        """
        print>>log, "checking for completion events on %s" % " ".join(events)
        for name in events:
            event = self._events.get(name)
            if event is None:
                raise KeyError("Unknown event '%s'" % name)
            while not event.is_set():
                print>> log, "  %s not yet complete, waiting" % name
                event.wait()
            print>> log, "  %s is complete" % name

    def awaitJobs (self, *jobspecs):
        """
        Waits for job(s) given by arguments to complete, and returns their results. This can only be called from
        the parent process.

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
            matching_jobs = [job_id for job_id in self._jobs.iterkeys() if fnmatch.fnmatch(job_id, jobspec)]
            for job_id in matching_jobs:
                awaiting_jobs.setdefault(job_id, set()).add(jobspec)
            job_results[jobspec] = len(matching_jobs), []
        # check dict of already returned results (perhaps from previous calls to awaitJobs). Remove
        # matching results, and assign them to appropriate jobspec lists
        for job_id, job in self._jobs.items():
            if job_id in awaiting_jobs and job.complete:
                for jobspec in pending_jobs[job_id]:
                    job_results[jobspec][1].append(result)
                if not job.singular:
                    del self._jobs[job_id]
                del awaiting_jobs[job_id]
        print>>log, "checking job results: %s (%d still pending)"%(
            ", ".join(["%s %d/%d"%(len(results, njobs)) for jobspec, (njobs, results) in pending_jobs.iteritems()]),
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
            job = self._jobs.get(result["job_id"])
            if job is None:
                raise KeyError("Job '%s' was not enqueued. This is a logic error." % job_id)
            job.setResult(result)
            # if being awaited, dispatch appropriately
            if job_id in awaiting_jobs:
                for jobspec in awaiting_jobs[job_id]:
                    job_results[jobspec].append(job.result)
                if not job.singular:
                    del self._jobs[job_id]
                del awaiting_jobs[job_id]
            # print status update
            print>>log,"received job results %s" % " ".join(["%s:%d"%(jobspec, len(results)) for jobspec, results in job_results])
        # process list of results for each jobspec to check for errors
        for jobspec, results in job_results.iteritems():
            times = [ result['time'] for results in results ]
            num_errors = len([result for result in results if not result['success']])
            print>> log, "%s: %d jobs complete, average single-core time %.2fs per job" % (jobspec, len(results), np.array(times).mean())
            if num_errors:
                print>>log, ModColor.Str("%d jobes returned an error. Aborting.", col="red")
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
            calls them, and returns resuslts in the work queue.
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
                    job_id, objname, method, eventname, args, kwargs = [ jobitem.get(attr)
                                            for attr in "job_id", "object", "method", "event", "args", "kwargs" ]
                    obj = self._job_handlers.get(objname)
                    try:
                        if obj is None:
                            raise KeyError("Unknown jobitem object '%s'"%objname)
                        if eventname:
                            event = self._events.get(eventname)
                            if event is None:
                                raise KeyError("Unknown event '%s'"%eventname)
                        else:
                            event = None
                        print>> log, "calling %s.%s" % (objname, method)
                        if method is None:
                            # call object directly
                            result = obj(*args, **kwargs)
                        else:
                            call = getattr(obj, method, None)
                            if call is None:
                                raise KeyError("Unknown method '%s' of object '%s'"%(method, objname))
                            result = call(*args, **kwargs)
                        print>> log, "%s.%s result is %s" % (objname, method, result)
                        # Raise event
                        if event:
                            event.set()
                        # Send result back
                        self._result_queue.put(dict(job_id=job_id, proc_id=self.proc_id, success=True, result=result, time=timer.seconds()))
                    except KeyboardInterrupt, exc:
                        raise
                    except Exception, exc:
                        print>> log, ModColor.Str("process %s: exception raised processing job %s: %s" % (
                            AsyncProcessPool.proc_id, job_id, traceback.format_exc()))
                        self._result_queue.put(dict(job_id=job_id, proc_id=self.proc_id, success=False, error=exc, time=timer.seconds()))
        except KeyboardInterrupt:
            print>>log, "Ctrl+C caught, exiting"
            return
    # CPU id. This will be None in the parent process, and a unique number in each worker process
    proc_id = None
