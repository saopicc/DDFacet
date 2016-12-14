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

import numpy as np
import fftw3
import pyfftw
import multiprocessing
import matplotlib
import matplotlib.pyplot as pl
import time
 
 
def fft_comparison_tests(size=2048, dtype=np.complex128, byte_align=False):
    """ Compare speed and test the API of pyFFTW3 and PyFFTW
    which are, somewhat surprisingly, completely different independent modules"""
 
    
 
    test_array = np.ones( (size,size), dtype=dtype)
    test_array[size*3/8:size*5/8, size*3/8:size*5/8] = 1 # square aperture oversampling 2...
 
    ncores = multiprocessing.cpu_count()
 
    pl.clf()
    for FFT_direction in ['forward']: #,'backward']:
 
        print "Array size: {1} x {1}\nDirection: {2}\ndtype: {3}\nncores: {4}".format(0, size, FFT_direction, dtype, ncores)
        print ""
        # Let's first deal with some planning to make sure wisdom is generated ahead of time.
        for i, fft_type in enumerate(['numpy','pyfftw3','pyfftw']):
                # planning using PyFFTW3
                p0 = time.time()
                print "Now planning "+fft_type+" in the "+FFT_direction+" direction"
                #if (test_array.shape, FFT_direction) not in _FFTW3_INIT.keys():
                if fft_type=='numpy':
                    print "\tno planning required"
                elif fft_type=='pyfftw3':
                    fftplan = fftw3.Plan(test_array.copy(), None, nthreads = ncores,direction=FFT_direction, flags=['measure'])
                else:
                    pyfftw.interfaces.cache.enable()
                    pyfftw.interfaces.cache.set_keepalive_time(30)
 
                    if byte_align: test_array = pyfftw.n_byte_align_empty( (size,size), 16, dtype=dtype)
                    test_array = pyfftw.interfaces.numpy_fft.fft2(test_array, overwrite_input=True, planner_effort='FFTW_MEASURE', threads=ncores)
 
                p1 = time.time()
                print "\tTime elapsed planning: {0:.4f} s".format(p1-p0)
 
        print ""
 
        # Now let's run some FFTs
        for i, fft_type in enumerate(['numpy','pyfftw3','pyfftw']):
            
            # display
            if fft_type == 'pyfftw' and byte_align:
                test_array = pyfftw.n_byte_align_empty( (size,size), 16, dtype=dtype)
                output_array = pyfftw.n_byte_align_empty( (size,size), 16, dtype=dtype)
                test_array[:,:] = 0
 
            else:
                test_array = np.zeros( (size,size), dtype=np.complex128)
            test_array[size*3/8:size*5/8, size*3/8:size*5/8] = 1 # square aperture oversampling 2...
            pl.subplot(2,3, 1 + i)
            pl.imshow(np.abs(test_array), vmin=0, vmax=1)
            pl.title( "FFT type: {0:10s} input array".format(fft_type))
            pl.draw()
 
 
            # actual timed FFT section starts here:
            t0 = time.time()
 
            if fft_type=='numpy':
                test_array = np.fft.fft2(test_array)
            elif fft_type=='pyfftw3':
 
                fftplan = fftw3.Plan(test_array, None, nthreads = multiprocessing.cpu_count(),direction=FFT_direction, flags=['measure'])
                fftplan.execute() # execute the plan
            elif fft_type=='pyfftw':
                test_array = pyfftw.interfaces.numpy_fft.fft2(test_array, overwrite_input=True, planner_effort='FFTW_MEASURE', threads=ncores)
 
    
            t1 = time.time()
 
            if FFT_direction=='forward': test_array = np.fft.fftshift(test_array)
 
            # display
            t_elapsed = t1-t0
            summarytext = "FFT type: {0:10s}\tTime Elapsed: {3:.4f} s".format(fft_type, size, FFT_direction, t_elapsed)
            print summarytext
 
            pl.subplot(2,3, 1+i+3)
 
 
 
            psf = np.real( test_array * np.conjugate(test_array))
            norm=matplotlib.colors.LogNorm(vmin=psf.max()*1e-6, vmax=psf.max())
 
            cmap = matplotlib.cm.jet
            cmap.set_bad((0,0,0.5))
 
            pl.imshow(psf[size*3/8:size*5/8, size*3/8:size*5/8], norm=norm)
            pl.title(summarytext)
            pl.draw()
            pl.show(False)
            pl.pause(0.1)
 
            #stop()

def main():
    for size in [1024, 2048, 4096]:
        fft_comparison_tests(size=size)
        print "--------------------------------------------------------"
 
if __name__ == '__main__':
    main()
