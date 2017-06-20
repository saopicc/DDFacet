#!/usr/bin/python

import pylab
import re
import numpy as np
import os.path
import sys

class DDFLog(object):
    def __init__ (self, filename):
        self.read(file(filename))
        self.name = os.path.basename(filename)

    def read(self,fobj):
        miter = 1
        self.major_start_flux = []
        self.major_start_iter = []
        self.major_end_flux = []
        self.major_end_iter = []
        self.minor_iters = []
        self.minor_peaks = []
        self.minor_pnrs = []
        for line in fobj.readlines():
            line = line.strip()
            m = re.search('Running minor cycle \[MinorIter = (\d+)/', line)
            if m:
                miter = int(m.group(1))+1
            m = re.search('\[iter=(\d+)\] (peak of|Reached.*peak flux)\s+(\d[^\s,]+)\s+', line)
            if m:
                print line
                self.major_end_iter.append(int(m.group(1))+1)
                self.major_end_flux.append(float(m.group(3)))
            m = re.search('Dirty image peak flux\s+=\s+(\d[^\s]+)\s', line)
            if m:
                print line
                self.major_start_flux.append(float(m.group(1)))
                self.major_start_iter.append(miter)
            m = re.search('\[iter=(\d+)\] peak residual\s+(\d[^\s]+),\s+.*PNR\s+(\d[^\s]*)', line)
            if m:
                print line
                self.minor_iters.append(int(m.group(1))+1)
                self.minor_peaks.append(float(m.group(2)))
                self.minor_pnrs.append(float(m.group(3)))


    def plot(self,col='black'):
        label = "%s (%d m/c)" % (self.name, len(self.major_start_iter))
        pylab.plot(self.minor_iters, self.minor_peaks,'.', mec=col, mfc='None', label=label)
        pylab.plot(self.major_start_iter, self.major_start_flux, '^', mec=col, mfc='None')
        pylab.plot(self.major_end_iter, self.major_end_flux, 'v', mec=col, mfc='None')
        pylab.xscale('log')
        pylab.yscale('log')
        pylab.xlabel('Minor loop iteration')
        pylab.ylabel('Peak residual')

colors = ('red', 'blue', 'green', 'black', 'purple')

if __name__ == '__main__':
    print sys.argv
    logs = [ DDFLog(x) for x in sys.argv[1:] ]
    for i, log in enumerate(logs):
        log.plot(col=colors[i])
    pylab.legend()
    pylab.show()