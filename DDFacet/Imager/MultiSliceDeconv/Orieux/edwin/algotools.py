#/bin/env python3
# -*- coding: utf-8 -*-
# sampling.py --- Stochastic sampling algorithms

# Copyright (c) 2011, 2012, 2013  François Orieux <orieux@iap.fr>

# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Commentary:

"""
References
----------
"""

# code:

from __future__ import division  # Then 10/3 provide 3.3333 instead of 3
import warnings
import time
import tempfile
import collections
import functools
import math

#import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import tqdm
except:
    pass

try:
    import notify2
except:
    pass
else:
    if not notify2.initted:
        notify2.init('Edwin iterative')

__author__ = "François Orieux"
__copyright__ = "2018 F. Orieux <orieux@l2s.centralesupelec.fr>"
__credits__ = ["François Orieux"]
__license__ = "mit"
__version__ = "0.1.0"
__maintainer__ = "François Orieux"
__email__ = "orieux@l2s.centralesupelec.fr"
__status__ = "development"
__url__ = ""
__keywords__ = "tools, algorithmes"


#%% Traces
class Trace(collections.abc.Sequence):
    def __init__(self, burnin=1, init=None, name=''):
        self.burnin = burnin
        self.name = name
        if init is None:
            self._hist = None
        else:
            self._hist = np.asarray(init)[np.newaxis]

    @property
    def burned(self):
        return True if len(self) > self.burnin else False

    @property
    def last(self):
        if self._hist is None:
            return None
        else:
            return self._hist[-1]

    @last.setter
    def last(self, value):
        if self._hist is None:
            self._hist = np.asarray(value)[np.newaxis]
        else:
            self._hist = np.append(self._hist,
                                   np.asarray(value)[np.newaxis],
                                   axis=0)

    def __ilshift__(self, value):
        """Use <<= as a affectation or Trace gets value meaning"""
        self.last = value
        return self

    #%% Arithmetic
    def __pos__(self):
        return +self.last

    def __neg__(self):
        return -self.last

    def __abs__(self):
        return abs(self.last)

    def __round__(self, n):
        return round(self.last)

    def __floor__(self):
        return math.floor(self.last)

    def __ceil__(self):
        return math.ceil(self.last)

    def __trunc__(self):
        return math.trunc(self.last)

    def __add__(self, value):
        return self.last + value

    def __sub__(self, value):
        return self.last - value

    def __mul__(self, value):
        return self.last * value

    def __floordiv__(self, value):
        return self.last // value

    def __truediv__(self, value):
        return self.last / value

    def __mod__(self, value):
        return self.last % value

    def __divmod__(self, value):
        return divmod(self.last, value)

    def __pow__(self, value):
        return pow(self.last, value)

    def __radd__(self, value):
        return value + self.last

    def __rsub__(self, value):
        return value - self.last

    def __rmul__(self, value):
        return value * self.last

    def __rfloordiv__(self, value):
        return value // self.last

    def __rtruediv__(self, value):
        return value / self.last

    def __rmod__(self, value):
        return value % self.last

    def __rdivmod__(self, value):
        return divmod(value, self.last)

    def __rpow__(self, value):
        return pow(value, self.last)

    def sum(self, burnin=None):
        if burnin is None:
            return np.sum(self._hist[self.burnin:], axis=0)
        else:
            return np.sum(self._hist[burnin:], axis=0)

    def mean(self, burnin=None):
        start = self.burnin if burnin is None else burnin
        if len(self) > start:
            return np.mean(self._hist[start:], axis=0)
        else:
            return np.array([])

    def std(self, burnin=None):
        start = self.burnin if burnin is None else burnin
        if len(self) > start:
            return np.std(self._hist[start:], axis=0)
        else:
            return np.array([])

    def cum_mean(self, burnin=None):
        start = self.burnin if burnin is None else burnin
        if len(self) > start:
            return np.cumsum(self._hist[start:]) / (
                np.arange(len(self._hist[start:])) + 1)
        else:
            return np.array([])

    def cum_std(self, burnin=None):
        start = self.burnin if burnin is None else burnin
        if len(self) > start:
            samples = self._hist[start:]
            cum_mean = self.cum_mean(start)
            return np.sqrt(
                np.cumsum(samples**2) / (np.arange(len(samples)) + 1) -
                cum_mean**2)
        else:
            return np.array([])

    @property
    def delta(self):
        if self._hist is None:
            return np.inf
        elif len(self._hist) >= 2:
            return np.sum((self._hist[-1] -
                           self._hist[-2])**2) / np.sum(self._hist[-1]**2)
        else:
            return np.inf

    @property
    def mean_delta(self):
        """ Variation of mean
        If
        μ_N = s_N / N,  with s_N = ∑_i x_i
        then
        μ_N - μ_(N-1) = (1/N - 1/(N-1))s_N + 1/(N-1) x_N

        Variation is |μ_N - μ_(N-1)|^2 / |μ_N|^2
        """
        if len(self) > self.burnin + 1:
            num = len(self) - self.burnin
            diff = (1 / num - 1 / (num - 1)) * self.sum() + self.last / (num - 1)
            return np.sum(diff**2) / np.sum(self.mean()**2)
        else:
            return np.nan

    @property
    def std_delta(self):
        if len(self) > self.burnin + 1:
            last_std = np.std(self._hist[self.burnin:], axis=0)
            prev_std = np.std(self._hist[self.burnin:-2], axis=0)
            return np.sum(np.abs(last_std - prev_std)**2) / \
                np.sum(np.abs(last_std)**2)
        else:
            return np.nan

    @property
    def shape(self):
        return self._hist.shape[1:]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def full_shape(self):
        return self._hist.shape

    @property
    def size(self):
        return np.prod(self._hist.shape[1:])

    def __len__(self):
        return len(self._hist)

    def __getitem__(self, key):
        if key not in range(-len(self), len(self)):
            raise IndexError
        else:
            return self._hist[key]

    def export(self, path, name):
        f = h5py.File(path)
        f.create_dataset(name, data=np.asarray(self._hist))
        f.close()

    def __repr__(self):
        return "{} ({}) / len: {} [burnin: {}] × shape: {}".format(
            self.name, type(self).__name__, len(self), self.burnin,
            self.shape)


class FileTrace(Trace):
    def __init__(self, burnin=1, init=None, maxitem=5000, name=''):
        super().__init__(burnin=burnin, name=name)
        self._temp_file = tempfile.NamedTemporaryFile()
        self._h5file = h5py.File(self._temp_file.name)
        self.maxitem = maxitem
        if init is None:
            self._hist = None
        else:
            init = np.asarray(init)
            self._hist = self._h5file.create_dataset(
                'trace',
                data=init[np.newaxis],
                maxshape=(maxitem, ) + init.shape)

    @Trace.last.setter
    def last(self, value):
        if self._hist is None:
            self._hist = self._h5file.create_dataset(
                'trace',
                data=value[np.newaxis],
                maxshape=(self.maxitem, ) + value.shape)
        else:
            self._hist.resize(len(self._hist) + 1, axis=0)
            self._hist[-1] = np.asarray(value)[np.newaxis]

    def __del__(self):
        self._h5file.close()


class DummyTrace(Trace):
    def __init__(self, burnin=1, init=None, name=''):
        super().__init__(burnin=burnin, name=name)
        self.sum_count = 0
        if init is None:
            self._val = None
            self._sum = None
            self._sum2 = None
            self.count = 0
        else:
            self._val = np.asarray(init)
            self._sum = np.zeros_like(init)
            self._sum2 = np.zeros_like(init)
            self.count = 1
        self._delta = np.inf

    @property
    def burned(self):
        return True if len(self) > self.burnin else False

    @property
    def last(self):
        return self._val

    @last.setter
    def last(self, value):
        self.count += 1

        if self._sum is None and self.burned:
            self.sum_count += 1
            self._sum = np.asarray(value)
            self._sum2 = np.asarray(value)**2
        elif self.burned:
            self.sum_count += 1
            self._sum = self._sum + np.asarray(value)
            self._sum2 = self._sum2 + np.asarray(value)**2

        if self._val is not None:
            self._delta = np.sum((self._val - value)**2) / np.sum(self._val**2)
        self._val = np.asarray(value)

    def sum(self, burnin=None):
        return self._sum

    def mean(self, burnin=None):
        return self._sum / self.sum_count

    def std(self, burnin=None):
        return np.sqrt(self._sum2 / self.sum_count - self.mean()**2)

    def cum_mean(self, burnin=None):
        return self.mean()

    def cum_std(self, burnin=None):
        return self.std()

    @property
    def delta(self):
        return self._delta

    @property
    def std_delta(self):
        return np.nan

    @property
    def shape(self):
        return self._val.shape

    @property
    def size(self):
        return np.prod(self._val.shape)

    def __len__(self):
        return self.count

    def __getitem__(self, key):
        if key not in range(len(self)):
            raise IndexError
        else:
            return self._val


#%% Traces plotter
class MplScalarTracePlot:
    def __init__(self, axe, name=''):
        self.axe = axe
        self.fig = self.axe.figure
        self.name = name

        axe.set_title('{}'.format(self.name))
        axe.set_xlabel('Iteration')

    def update(self, trace):
        if not hasattr(self, 'line'):
            self.line, = self.axe.plot(trace)
        else:
            self.line.set_ydata(trace)
            self.line.set_xdata(np.arange(len(trace)))
        self.axe.set_title('{}: {:.2g}'.format(self.name, trace.last))
        self.axe.set_xlim(0, max(len(trace), trace.burnin + 10))
        # self.axe.set_ylim(0.9 * min(trace) - 0.1, 1.1 * max(trace) + 0.1)
        self.axe.set_ylim(0.9 * min(trace), 1.1 * max(trace))

    def update_stoch(self, trace):
        if not hasattr(self, 'burnin_line'):
            self.burnin_line = self.axe.axvline(
                trace.burnin, ls='--', alpha=0.2)

        if trace.burned:
            if hasattr(self, 'mean_line'):
                self.mean_line.set_ydata(trace.cum_mean())
                self.mean_line.set_xdata(
                    np.arange(len(trace.cum_mean())) +
                    trace.burnin)
                self.noise_fill.remove()
            else:
                self.mean_line, = self.axe.plot(trace.cum_mean(),
                                                color='red',
                                                lw=2)
            self.noise_fill = self.axe.fill_between(
                np.arange(trace.burnin, len(trace)),
                trace.cum_mean() - trace.cum_std(),
                trace.cum_mean() + trace.cum_std(),
                facecolor='blue', alpha=0.2)

            self.axe.set_title('{}: {:.2g} / {:.2g} +- {:.2g}'.format(
                self.name, trace.last,
                trace.mean(), trace.std()))
        else:
            self.axe.set_title('{}: {:.2g}'.format(self.name, trace.last))


class Mpl1DTracePlot:
    def __init__(self, axe, name=''):
        self.axe = axe
        self.fig = self.axe.figure
        self.name = name

        self.axe.set_title('{}'.format(self.name))
        self.axe.set_xlabel('\'t\'')

    def update(self, trace):
        if not hasattr(self, 'line'):
            self.line, = self.axe.plot(trace.last)
        else:
            self.line.set_ydata(trace.last)
            self.line.set_xdata(np.arange(len(trace.last)))
        self.axe.set_ylim(1.1 * min(trace.last),
                          1.1 * max(trace.last))
        self.axe.set_title('{} / Δ: {:.1g}'.format(self.name, trace.delta))
        self.axe.set_xlabel('\'t\'')

    def update_stoch(self, trace):
        if trace.burned:
            if hasattr(self, 'mean_line'):
                self.mean_line.set_ydata(trace.mean())
                self.noise_fill.remove()
            else:
                self.mean_line, = self.axe.plot(trace.mean(),
                                                color='red',
                                                alpha=1)
            self.noise_fill = self.axe.fill_between(
                np.arange(len(trace.last)),
                trace.mean() - trace.std(),
                trace.mean() + trace.std(),
                facecolor='blue', alpha=0.2)
            self.axe.set_ylim(1.1 * min(trace.mean()),
                              1.1 * max(trace.mean())),
            # self.axe.set_ylim(min(1.1 * min(trace.mean()),
            #                       self.axe.get_ylim()[0]),
            #                   max(1.1 * max(trace.mean()),
            #                       self.axe.get_ylim()[1]))
        self.axe.set_title('{} / Δμ: {:.1g} / Δσ: {:.1g}'.format(
            self.name, trace.mean_delta, trace.std_delta))
        self.axe.set_xlabel('\'t\'')


class Mpl2DTracePlot:
    def __init__(self, axe, name=''):
        self.axe = axe
        self.fig = self.axe.figure
        self.name = name

        self.axe.set_title('{}'.format(self.name))
        self.axe.set_xlabel('\'t\'')

    def update(self, trace):
        if not hasattr(self, 'im'):
            self.im = self.axe.imshow(abs(trace.last), cmap=cm.gray)
            self.axe.set_axis_off()
        else:
            self.im.set_data(abs(trace.last))
            self.im.autoscale()
        self.axe.set_title('{} / Δ: {:.1g}'.format(self.name, trace.delta))

    def update_stoch(self, trace):
        if trace.burned:
            mean = abs(trace.mean())
            std = abs(trace.std())
            self.im.set_data(np.concatenate(
                (mean, std / np.max(std) * np.max(mean)), axis=1))
            self.axe.set_aspect('auto')
            self.im.autoscale()
        self.axe.set_title('{} / Δμ: {:.1g} / Δσ: {:.1g}'.format(
            self.name, trace.mean_delta, trace.std_delta))


class Mpl2DTraceLogPlot:
    def __init__(self, axe, name=''):
        self.axe = axe
        self.fig = self.axe.figure
        self.name = name

        self.axe.set_title('{}'.format(self.name))
        self.axe.set_xlabel('\'t\'')

    def update(self, trace):
        if not hasattr(self, 'im'):
            self.im = self.axe.imshow(np.log(trace.last), cmap=cm.gray)
            plt.axis('off')
        else:
            self.im.set_data(np.log(trace.last))
        self.axe.set_title('{} / Δ: {:.1g}'.format(self.name, trace.delta))

    def update_stoch(self, trace):
        if trace.burned:
            self.im.set_data(np.concatenate(
                (np.log(trace.mean()), trace.std()), axis=1))
            self.axe.set_aspect('auto')
        self.axe.set_title('{} / Δμ: {:.1g} / Δσ: {:.1g}'.format(
            self.name, trace.mean_delta, trace.std_delta))



#%% Feedbacks
class Feedback:
    def __init__(self, name):
        self.name = name

    def init(self):
        pass

    def show(self, iteration, min_iter, max_iter):
        print('Iter {} [{}] / {}'.format(iteration, min_iter, max_iter))

    def close(self):
        pass


class Notification(Feedback):
    def __init__(self, name):
        if not notify2.initted:
            notify2.init('Sampling')

        self.name = name
        self.notif = notify2.Notification(
            self.name, "Iteration ? / ?")

    @staticmethod
    def filled_bar_str(iteration, min_iter, max_iter):
        return "{}{}".format(
            round(iteration / max_iter * 10) * "■",
            (10 - round(iteration / max_iter * 10)) * "□")

    def show(self, iteration, min_iter, max_iter):
        self.notif.update(
            self.name,
            "Iteration {} [{}] / {} {}".format(
                iteration,
                min_iter,
                max_iter,
                Notification.filled_bar_str(iteration, min_iter, max_iter)))
        self.notif.show()

    def close(self):
        self.notif.close()


class Bar(Feedback):
    def __init__(self, name):
        self.name = name

    def show(self, iteration, min_iter, max_iter):
        if hasattr(self, 'bar'):
            self.bar.update(n=1)
        else:
            self.bar = tqdm.tqdm(
                desc=self.name, total=max_iter,
                dynamic_ncols=True)

    def close(self):
        self.bar.close()


class Figure(Feedback):
    def __init__(self, fig, name='', plotters=None, traces=None,
                 stochastic=True):
        super().__init__(name)
        self.fig = fig
        self.plotters_list = [] if plotters is None else plotters
        self.traces_list = [] if traces is None else traces
        plt.show(block=False)
        self.stochastic = stochastic
        self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        for axe in self.fig.get_axes():
            axe.cla()
        # self.init()

    def init(self):
        pass
        # for axe in self.fig.get_axes():
        #     axe.cla()

    def show(self, iteration, min_iter, max_iter):
        for plotter, trace in zip(self.plotters_list, self.traces_list):
            plotter.update(trace)
            if self.stochastic:
                plotter.update_stoch(trace)

        self.fig.suptitle('{} : [{}] <= {} / {}'.format(
            self.name, min_iter, iteration, max_iter))

        if iteration == 1:
            self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


#%% Looper
class Looper(collections.abc.Iterator):
    def __init__(self, max_iter, min_iter, feedbacks=None,
                 stop_fct=None, callback=None, speedrun=True):
        self.max_iter = max_iter
        self.min_iter = min_iter
        if self.min_iter > self.max_iter:
            warnings.warn(
                "Max iteration ({0}) is lower than min iteration ({1}). "
                "Max is set to min".format(self.max_iter, self.min_iter))
            self.max_iter = self.min_iter
        self.nit = 0
        self.status = 2
        self.message = 'Running'
        self.succes = False
        self.timestamp = []
        self.feedbacks_duration = []

        self.speedrun = speedrun
        self.callback = callback

        if stop_fct is not None:
            self.stop = stop_fct
        else:
            def always_false():
                return False
            self.stop = always_false

        if isinstance(feedbacks, collections.Iterable):
            self.feedbacks = feedbacks
        elif feedbacks is None:
            self.feedbacks = []
        else:
            self.feedbacks = [feedbacks]


    @property
    def time(self):
        """Return effective time of controlled iteration"""
        arr = np.array(self.timestamp) - self.timestamp[0]
        return arr - np.cumsum(self.feedbacks_duration)

    @property
    def mean_time(self):
        """Return effective mean time of controlled iteration"""
        return np.mean(np.diff(self.time))

    def __iter__(self):
        self.nit = 0
        self.status = 2
        self.message = 'Running'
        self.succes = False
        self.timestamp = [time.time()]
        self.feedbacks_duration = [0]
        for feedback in self.feedbacks:
            feedback.init()
        return self

    def __next__(self):
        if (self.nit >= self.min_iter) and self.stop():
            for feedback in self.feedbacks:
                feedback.show(self.nit, self.min_iter, self.max_iter)
                feedback.close()
            self.status = 0
            self.message = 'Condition reached'
            self.succes = True
            raise StopIteration()
        elif self.nit < self.max_iter:
            if not self.speedrun:
                tic = time.time()
                for feedback in self.feedbacks:
                    feedback.show(self.nit, self.min_iter, self.max_iter)
                self.feedbacks_duration.append(time.time() - tic)
            else:
                self.feedbacks_duration.append(0)

            self.timestamp.append(time.time())
            self.nit += 1
            return self.nit
        else:
            for feedback in self.feedbacks:
                feedback.close()
            self.status = 1
            self.message = 'Maximum iteration reached'
            self.succes = False
            raise StopIteration()

    def __repr__(self):
        fb = np.mean(self.feedbacks_duration)
        return ("Succes : {}; {}\n".format(self.succes, self.message) +
                "[{}] <= {} / {}\n".format(
                    self.min_iter, self.nit, self.max_iter) +
                "Total time {:.2g} / mean time {:.2g} / FB time {:.2g} ({:.1f}%)".format(
                    self.time[-1], self.mean_time,
                    fb, 100 * fb * (fb + self.mean_time)))


class IterativeAlg:
    def __init__(self, max_iter, min_iter, stochastic, threshold=1e-6,
                 speedrun=True, feedbacks=None, trace=DummyTrace,
                 name=''):

        if isinstance(feedbacks, collections.Iterable):
            for fb in feedbacks:
                fb.name = name
                fb.stochastic = stochastic
                if isinstance(fb, Figure):
                    self._figure = fb
                else:
                    self._figure = None
        elif isinstance(feedbacks, Figure):
            feedbacks.name = name
            feedbacks.stochastic = stochastic
            if isinstance(feedbacks, Figure):
                self._figure = feedbacks
            else:
                self._figure = None

        self.looper = Looper(
            max_iter,
            min_iter,
            speedrun=speedrun,
            feedbacks=feedbacks)

        self.stochastic = stochastic
        self.threshold = threshold
        self.name = name
        self._trace = trace

    @property
    def max_iter(self):
        return self.looper.max_iter

    @property
    def min_iter(self):
        return self.looper.min_iter

    def stop(self, obj):
        if self.stochastic:
            if obj.mean_delta < self.threshold:
                return True
            else:
                return False
        else:
            if obj.delta < self.threshold:
                return True
            else:
                return False

    def fig_register_trace(self, *args):
        if hasattr(self, '_figure'):
            axes = self._figure.fig.get_axes()
            self._figure.traces_list = args
            for axe, trace in zip(axes, args):
                if trace.ndim == 0:
                    self._figure.plotters_list.append(MplScalarTracePlot(axe, trace.name))
                elif trace.ndim == 1:
                    self._figure.plotters_list.append(Mpl1DTracePlot(axe, trace.name))
                elif trace.ndim == 2:
                    self._figure.plotters_list.append(Mpl2DTracePlot(axe, trace.name))
                else:
                    print('Trace DataType not supported for plotting')

    def watch_for_stop(self, trace):
        self.looper.stop = functools.partial(self.stop, trace)


class IterRes:
    def __init__(self, looper, **kwargs):
        self.looper = looper
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.status = looper.status
        self.message = looper.message
        self.succes = looper.succes
        self.nit = self.looper.nit
        self.max_iter = looper.max_iter
        self.min_iter = looper.min_iter
        self.time = list(looper.time)
        self.mean_time = looper.mean_time
        self.fb_time = np.mean(looper.feedbacks_duration)
        self.fun = 0
        self.jac = 0
        self.hess = 0
        self.hess_inv = 0
        self.nfev = 0
        self.njev = 0
        self.maxcv = 0

    @property
    def x(self):
        return self.minimizer

    def __repr__(self):
        return self.looper.__repr__()
