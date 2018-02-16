# Multiprocessing with AsyncProcessPool and SharedDict

AsyncProcessPool, or APP for short, is a new multiprocessing scheme that I introduced in https://github.com/cyriltasse/DDFacet/pull/306. The idea is, you dispatch parallel jobs "in the background" as soon as the data to run them is available, carry on doing something else, and then collect the jobs' results when needed, but not before. This allows us to rather simply parallelize I/O, gridding, and anything else really.

APP uses the Python ``multiprocessing`` module. The main (parent) process forks a bunch of worker processes. Workers are persistent, i.e. stay around for the duration of the program's run.

Workers are grouped into multiple *compute workers* (the ``--Parallel-NCPU`` option determines how many), and one (or few) *I/O workers*. There are separate *compute job* and *I/O job* queues. Many compute workers service the same compute queue; each I/O worker is associated with a separate I/O queue. The idea is that I/O jobs, i.e. ones that require little CPU but mostly wait on I/O, are treated a little separately (and we can have different CPU affinity settings for I/O processes and compute processes), but this is mostly a logistical distinction.

In a nutshell, you start jobs like this:

```
for i in xrange(10):
   APP.runJob("my_job_id:%d" % i, self.job_function, args=(i, "foo", "bar"))
```

and you collect their results like this:

```
results = APP.awaitJobResults("my_job_id:*")
```

The first call places 10 jobs on the compute queue. Each job consists of calling ``self.job_function(i, "foo", "bar")``, for ``i`` from 0 to 9. This is done by the worker processes. ``self.job_function`` is referred to as a job handler.

The second call waits for all jobs matching the pattern to finish, and returns their results (if any) in list. (Note that passing large results this way is not very efficient -- far better to use SharedDict for large data! -- so typically these will just be return codes or Nones.)

## Worker init and worker state

Workers are forked from the parent process. This means that, at time of forking (when ``APP.startWorkers()`` is called), they are exact clones of the parent, and thus contain exactly the same object state etc. We'll call this *original initialized state* (OIS). Thus every worker will have its own copy of every object in the program (for example, ``self.VS`` and ``self.FacetMachine`` in ``ClassDeconvMachine``). But, from that that point on, the internal state of the workers is decoupled from the parent. And each worker may find itself requested to process any kind of job in the future! This can create all sorts of confusion, so follow some simple rules:

* keep job handlers simple. Basically, job handlers should be largely stateless, and should assume that, in any given object, they can be called in any given order.

* job handlers can only assume that their ``self`` object is in OIS. A new clone of a worker process may be created at any time, and in this clone everything is reset back to OIS. Any job handler must therefore be able to run from OIS.

* therefore, job handlers should not try to modify their object's internal state (i.e. assign to ``self.attr``), with the exception of creating and reloading SharedDicts to get/set shared data.

* all data relevant to the job must be passed in via the job handler's arguments, or (if large) via a SharedDict. More on that later.

For example, ``ClassDeconvMachine``, which is the main loop of DDFacet basically, now proceeds as follows:

* It creates a ``ClassVisServer`` (``self.VS``) and initializes it with the MS list (and the VisServer reads and sets up all necessary MS information)

* It creates and initializes FacetMachine(s)

* It calls ``APP.startWorkers()``

For this point on, each worker process has its own copy of ``self.VS`` and ``self.FacetMachine`` inside it, initialized and ready to run any job handler (e.g. read given chunk of data, grid chunk of data, etc. -- in any order.)

## Jobs

Jobs are initiated with a call to ``APP.runJob``. This is defined as:
```
    def runJob (self, job_id, handler=None, io=None, args=(), kwargs={},
                event=None, counter=None,
                singleton=False, collect_result=True,
                serial=False):
```
``job_id`` is a unique string identifier for the job. You can't have two pending jobs with the same name. Handler is either a function, or a bound method of an object. (NB: all job handler functions, and objects containing job handlers, must be registered with ``APP.registerJobHandlers()`` during initialization, before the workers are started). ``io`` is None for compute jobs, or the number of an I/O queue for I/O jobs. ``args`` and ``kwargs`` will be passed to the job handler in the worker process.

Jobs are *singletons* (``singleton=True``) if they only ever need to run once. An example is VisServer.CalcWeights. Most jobs aren't singletons.

The other three arguments determine how a job notifies us of its completion. The ``collect_result=True`` argument indicates that the result of the job (i.e. the return value of the handler) must be returned (via a result queue) to the parent process. Note that jobs may be initiated by any process, not just the parent (for example, when the I/O thread loads an MS chunk, it may initiate a bunch of compute jobs to make the BDA mappings). However, only the parent process is allowed to run jobs with ``collect_results=True``, and to collect the results. Worker processes that want to run their own jobs (currently, this is only the case in the VisServer, which initiates BDA mapping jobs) can still do it, but they can only use *events* or *counters* to notify themselves of job completion. The parent process can use all three mechanisms.

### Collecting job results

As described above, ``APP.awaitJobResults()`` waits for a bunch of jobs (as specified by a job id, or a wildcard-containing pattern) to complete, and collects their results. If the jobs are already complete, their results are returned immediately. An optional ``progress="Title"`` argument renders a ProgressBar while the process waits.

Any exceptions in the job handlers are caught, and reported back to the parent process. ``awaitJobResults()`` throws an exception in such cases.

### Events

An event is simply a flag. The event must be registered with ``APP.registerEvents()`` at initialization. Running a job with ``event=`` tells it to *raise* the named event when completed. Any other process can wait for the event by calling ``APP.awaitEvents()``.

### Job counters

A counter is, ahem, a counter (strictly speaking, a *condition variable* with an associated counter). Counters are created with ``APP.createJobCounter()`` (take a look inside ``ClassSmearMapping.py`` for an example). When a job is run with ``counter=`` (see ``computeSmearMappingInBackground()``), the counter is incremented. When the job has been completed, the counter is decremented. Any process can call ``APP.awaitJobCounter()`` on the counter to sit and wait for the counter to return to zero. This is a good way to start N jobs (all on the same counter), and wait for them to complete.

Note that events or counters do not provide exception reporting (though this could, and perhaps should, be implemented).

### Debugging jobs

Passing ``serial=True`` to a ``runJob()`` call causes that job to be run in serial mode, i.e. immediately and in the same process. This is useful if you want to debug a job handler (i.e. attach pdb and stop on an exception).

Running DDF with ``--Parallel-NCPU 1`` makes all jobs run serially.

## SharedDict

The ``SharedDict`` class implements a dict-like object that resides in shared memory. This makes for very efficient means of communication between processes. Most invocations of APP job handlers in DDF are, semantically, either something like "do this computation and put the results into this SharedDict", or "take this SharedDict, do some long computations, put results into this other SharedDict".

SharedDicts are implemented as directory hierarchies under ``/dev/shm``. A SharedDict has a short name (e.g. "foo"), and a corresponding path (e.g. ``/dev/shm/ddf.PID/foo``, where PID is the pid of the parent process).

To create and populate a SharedDict, one would do

```
import SharedDict
dd = SharedDict.create("foo")
dd["a"] = 1
dd["b"] = numpy.zeros((10,10), numpy.int)       # dd["b"] becomes a SharedArray
c = dd.addSharedArray("c", (10,10), numpy.int)  # faster: no intermediate array is created
subdd = dd.addSubdict("x")
```

SharedDict keys can only be of types ``str``, ``int`` or ``bool``. Values can be of any type that supports pickling (they are stored as pickles in shared memory). Two types of values are particularly efficient: numpy arrays are stored as SharedArrays, and dicts are stored as SharedDicts.

### Reading A SharedDict

In another process, if you know the name or path to a SharedDict, you can simply

```
import SharedDict
dd = SharedDict.attach(name)
```

and presto, you have a dict-like object to work with. Note that attaching is very lightweight (objects are only actually loaded on-demand, when you start accessing them by key).

### SHM content versus Python objects

It's important to appreciate the distinction between the object ``dd`` above, which exists in the address space of that particular Python interpreter, and the SHM content itself (in ``/dev/shm/ddf.PID/foo/*``). The former is only a representation of the latter. After the ``attach()`` call, the two are in sync. If the process inserts new keys into ``dd``, these immediately reflect in SHM, so the two stay in sync. However if **another** process attaches to the SHM version and proceeds to insert new keys, those keys will show up in SHM, but will not implicitly propagate into the ``dd`` object of the first process.

Calling ``dd.reload()`` makes the object sync up with the SHM content again. In general, however, this scenario requires some care (as in the case of any shared resource accessed by multiple processes). See more below.

Deleting a SharedDict object merely destroys its Python representation, the SHM contents stay undisturbed. If you want to delete the SHM objects as well, you must call ``dd.clear()`` or ``dd.delete()`` explicitly.

### Synchronization

SharedDict has no explicit synchronization support (at present). This means that if processes read and write to the same SharedDict unabashedly, they are liable to confuse each other, unless they exercise a certain amount of discipline.

It's always safe to:

* get items from a SharedDict, once it's been attached or reloaded

* attach to any SharedDict with ``attach(load=False)``. This attaches to SHM, but does not load content/

* insert **new** keys into a SharedDict

It is **not** safe to ``reload()`` or ``attach(load=True)`` (load=True is the default) to a SharedDict while another process is busy populating it. The contents seen by Python may then be somewhat unpredictable.

### Safe access patterns

The following SharedDict access patterns are safe:

#### "Read-only"

Job handlers in child processes attach to an existing SharedDict, and do not write to it. Simple and safe.

#### "Loader"

* parent process calls runJob() to schedule e.g. an I/O job

* job handler in worker process creates a new shared dict and populates it

* job handler completes

* parent process attaches to the shared dict with ``attach(load=True)``. The parent is now free to modify the dict (until such time as

#### "Populators"

This pattern applies when you have many jobs, and each job results in a different key being inserted into the SharedDict:

* the parent process creates an empty SharedDict

* parent process calls runJob() many times. Job handlers in worker processes attach to this dict with ``attach(load=False)``

* each worker inserts a different key into the dict

* all job handlers complete

* the parent process reloads the dict with ``reload()``, or ``attach(load=True)``


