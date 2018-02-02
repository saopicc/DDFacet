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

import DDFacet.ToolsDir.ModFitPSF as fitter
import numpy as np
from matplotlib import pyplot as plt
from DDFacet.ToolsDir.gaussfitter2 import twodgaussian as gauss2d
import DDFacet.ToolsDir.ModFFTW as fftconvolve

def testFitWONoise():
    def fitRotatedPSF(rotAngle):
        nhalf = 512
        xx, yy = np.meshgrid(np.linspace(-nhalf, nhalf, nhalf * 2 + 1),
                             np.linspace(-nhalf, nhalf, nhalf * 2 + 1))
        inpars = np.array([1., 0., 0., 50., 3., rotAngle])
        psf = gauss2d(inpars, circle=0, rotate=1, vheight=0)(xx,yy)
        maxAt = np.argmax(psf)
        maxAtCrd = np.array([maxAt % psf.shape[0], maxAt / psf.shape[0]])

        wnd = 50
        psfWnd = psf[(maxAtCrd[0] - wnd):(maxAtCrd[0] + wnd + 1),
                    (maxAtCrd[1] - wnd):(maxAtCrd[1] + wnd + 1)]
        outpars = fitter.FitCleanBeam(psfWnd) * np.array([1,1,180/np.pi])
        inpars = [1, 0, 0, outpars[1], outpars[0], outpars[2]]
        cleanBeam = gauss2d(inpars, circle=0, rotate=1, vheight=0)(xx,yy)
        assert np.allclose(psf,cleanBeam)
    for r in np.linspace(0,360,100):
        fitRotatedPSF(r)

def testFitWNoise():
    def fitRotatedPSF(rotAngle):
        nhalf = 512
        xx, yy = np.meshgrid(np.linspace(-nhalf, nhalf, nhalf * 2 + 1),
                             np.linspace(-nhalf, nhalf, nhalf * 2 + 1))
        inpars = np.array([1., 0., 0., 50., 3., rotAngle])
        psf = gauss2d(inpars, circle=0, rotate=1, vheight=0)(xx, yy)
        psf += np.random.randn(psf.shape[0],psf.shape[1])*1e-7
        maxAt = np.argmax(psf)
        maxAtCrd = np.array([maxAt % psf.shape[0], maxAt / psf.shape[0]])

        wnd = 50
        psfWnd = psf[(maxAtCrd[0] - wnd):(maxAtCrd[0] + wnd + 1),
                 (maxAtCrd[1] - wnd):(maxAtCrd[1] + wnd + 1)]
        outpars = fitter.FitCleanBeam(psfWnd) * np.array([1, 1, 180 / np.pi])
        inpars = [1, 0, 0, outpars[1], outpars[0], outpars[2]]
        cleanBeam = gauss2d(inpars, circle=0, rotate=1, vheight=0)(xx, yy)
        assert np.allclose(psf, cleanBeam, rtol=1e-6, atol=1e-6)
    for r in np.linspace(0,360,100):
        fitRotatedPSF(r)

def testRestore():
    def restoreFittedBeam(rot):
        imgSize = 256
        cellSize = np.deg2rad(4./3600.)
        params = (10, 5, rot) #maj, min, theta
        #create input with code borrowed from Tigger:
        xx,yy = np.meshgrid(np.arange(0,imgSize),np.arange(0,imgSize))
        inp = gauss2d([1,imgSize/2,imgSize/2,params[1],params[0],params[2]],circle=0,rotate=1,vheight=0)(xx,yy)
        inp = inp.reshape(1,1,imgSize,imgSize)
        #fit
        fittedParams = tuple((fitter.FitCleanBeam(inp[0, 0, :, :]) *
                              np.array([cellSize, cellSize, 1])).tolist())
        #restore fitted clean beam with an FFT convolution:
        delta = np.zeros([1, 1, imgSize, imgSize])
        delta[0, 0, imgSize / 2, imgSize / 2] = 1
        rest = fftconvolve.ConvolveGaussian(shareddict={"in":delta,
                                                        "out":delta},
                                            field_in="in",
                                            field_out="out",
                                            ch=0,
                                            CellSizeRad=cellSize,
                                            GaussPars_ch=fittedParams,
                                            Normalise=False)
        assert np.allclose(inp, rest, rtol=1e-2, atol=1e-2)
    for r in np.linspace(0,360,100):
        restoreFittedBeam(r)

def testFitSinc():
     #make the psf a sinc function
     nhalf = 512
     sinc1d = np.sinc(np.linspace(-nhalf,nhalf,nhalf*2+1)/(nhalf*2+1)*np.pi*6)
     psf = np.outer(sinc1d,sinc1d)
     maxAt = np.argmax(psf)
     maxAtCrd = np.array([maxAt % psf.shape[0], maxAt / psf.shape[0]])
     outpars = fitter.FitCleanBeam(psf) * np.array([1, 1, 180 / np.pi])
     xx, yy = np.meshgrid(np.linspace(-nhalf, nhalf, nhalf * 2 + 1), np.linspace(-nhalf, nhalf, nhalf * 2 + 1))
     inpars = [1, 0, 0, outpars[0], outpars[1], outpars[2]]
     cleanBeam = gauss2d(inpars, circle=0, rotate=1, vheight=0)(xx, yy)
     wnd = 200
     psfWnd = psf[(maxAtCrd[0] - wnd):(maxAtCrd[0] + wnd + 1),(maxAtCrd[1] - wnd):(maxAtCrd[1] + wnd + 1)]
     lev, fnull = fitter.FindSidelobe(psfWnd)
     assert np.isclose(lev,np.sinc(5/2.0),rtol=1e-2,atol=1e-8) # just check if it found the first sidelobe level



