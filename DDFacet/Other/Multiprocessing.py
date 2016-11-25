import psutil
import Queue
import multiprocessing
import numpy as np

from DDFacet.Other import MyLogger
from DDFacet.Other import ClassTimeIt
from DDFacet.Other.progressbar import ProgressBar

log = MyLogger.getLogger("Multiprocessing")
#MyLogger.setSilent("Multiprocessing")


# A generic producer that places data items from a list into a queue
def _work_producer(queue, data):
    for item in data:
        try:
            queue.put(item, 60)  # possible issue if no space for 60 seconds
        except Queue.Full:
            print>> log, "work_producer: queue full"
            pass


# Init W worker that is called by Multiprocessing.Process
def _work_consumer(m_work_queue, m_result_queue, target, args=(), kwargs={}):
    timer = ClassTimeIt.ClassTimeIt()
    pill = True
    # While no poisoned pill has been given grab items from the queue.
    while pill:
        try:
            # Get queue item, or timeout and check if pill perscribed.
            jobitem = m_work_queue.get(True, 5)
        except Queue.Empty:
            print>> log, "work_consumer: empty worker queue"
            pass
        else:
            if jobitem == "POISON-E":
                break
            elif jobitem is not None:
                result = target(jobitem, *args, **kwargs)
                # Send result back
                m_result_queue.put(dict(Success=True, Time=timer.seconds(), Result=result))


class ProcessPool (object):
    def __init__ (self, GD):
        self.GD = GD
        self.affinity = self.GD["Parallel"]["Affinity"]
        self.cpustep = self.affinity or 1
        self.ncpu = self.GD["Parallel"]["NCPU"] or psutil.cpu_count() / cpustep
        self.procinfo = psutil.Process()  # this will be used to control CPU affinity


    def runjobs (self, joblist, target, args=(), kwargs={}, result_callback=None, title=None):
        parallel = self.ncpu > 1
        procs = list()  # list of processes that are running

        # set up queues
        if parallel:
            if len(joblist) > 1000:
                qlimit = self.ncpu * 8
            else:
                qlimit = 0
            m_work_queue = multiprocessing.Queue(maxsize=qlimit)
        else:
            m_work_queue = multiprocessing.Queue()
        m_result_queue = multiprocessing.JoinableQueue()

        # create work producer process
        work_p = multiprocessing.Process(target=_work_producer, args=(m_work_queue, joblist,))

        # create worker processes
        for cpu in xrange(self.ncpu):
            p = multiprocessing.Process(target=_work_consumer, args=(m_work_queue, m_result_queue, target, args, kwargs))
            procs.append(p)

        # fork off child processes
        if parallel:
            parent_affinity = self.procinfo.cpu_affinity()
            try:
                # work producer process does nothing much, so I won't pin it
                work_p.start()
                main_core = 0  # CPU affinity placement
                # start all processes and pin them each to a core
                for p in procs:
                    if self.affinity:
                        self.procinfo.cpu_affinity([main_core])
                        main_core += self.cpustep
                    p.start()
            finally:
                self.procinfo.cpu_affinity(parent_affinity)
        print>> log, "%s: starting %d workers for %d jobs, affinity %d" % (title or "", self.ncpu, len(joblist), self.affinity)

        njobs = len(joblist)
        iResult = 0

        timer = ClassTimeIt.ClassTimeIt()
        # create progress bar
        if title:
            pBAR = ProgressBar('white', width=50, block='=', empty=' ',
                               Title="  %s "%title, HeaderSize=10, TitleSize=13)
            # pBAR.disable()
            pBAR.render(0, '%4i/%i' % (0, njobs))
        else:
            pBAR = None

        # in serial mode, just run both things to completion
        if not parallel:
            work_p.run()
            for p in procs:
                m_work_queue.put("POISON-E")
                p.run()  # just run until all work is completed

        # process results
        results = []
        while iResult < njobs:
            result = None
            try:
                result = m_result_queue.get(True, 10)
            except Queue.Empty:
                print>> log, "checking for dead workers"
                # shoot the zombie process
                multiprocessing.active_children()
                # check for dead workers
                pids_to_restart = []
                for w in procs:
                    if not w.is_alive():
                        pids_to_restart.append(w)
                        raise RuntimeError("a worker process has died on us \
                            with exit code %d. This is probably a bug." %
                                           w.exitcode)
                        #
                        # for id in pids_to_restart:
                        #     print>> log, "need to restart worker %d." % id
                        #     pass

            if result is not None and result["Success"]:
                iResult += 1
                m_result_queue.task_done()  # call task_done on the queue
                if result_callback:
                    result_callback(iResult, result)
                results.append(result)
            else:
                print>> log, "work_consumer: returned no result"
            # update progress bar
            if pBAR:
                intPercent = int(100 * iResult / float(njobs))
                pBAR.render(intPercent, '%4i/%i' % (iResult, njobs))

        # if all work is done, send poison pill to workers
        if parallel:
            for p in procs:
                m_work_queue.put("POISON-E")

            # join and close queues
            m_result_queue.join()
            m_work_queue.close()
            m_result_queue.close()

            # join producer process
            work_p.join()

            # join consumerr processes
            for p in procs:
                p.join()

        # compute average processing time
        average_time = np.array([ r["Time"] for r in results ]).mean()

        print>> log, "%s finished in %s, average time %.2fs per job" % (title or "", timer.timehms(), average_time)

        # extract list of result objects
        return [ r["Result"] for r in results ]