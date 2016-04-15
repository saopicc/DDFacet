import DDFacet.ToolsDir.ModFitPSF as fitter
import numpy as np
from matplotlib import pyplot as plt
from Tigger.Tools.gaussfitter2 import twodgaussian as gauss2d

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
        inpars = [1, 0, 0, outpars[0], outpars[1], outpars[2]]
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
        psf += np.random.randn(psf.shape[0],psf.shape[1])*1e-6
        maxAt = np.argmax(psf)
        maxAtCrd = np.array([maxAt % psf.shape[0], maxAt / psf.shape[0]])

        wnd = 50
        psfWnd = psf[(maxAtCrd[0] - wnd):(maxAtCrd[0] + wnd + 1),
                 (maxAtCrd[1] - wnd):(maxAtCrd[1] + wnd + 1)]
        outpars = fitter.FitCleanBeam(psfWnd) * np.array([1, 1, 180 / np.pi])
        inpars = [1, 0, 0, outpars[0], outpars[1], outpars[2]]
        cleanBeam = gauss2d(inpars, circle=0, rotate=1, vheight=0)(xx, yy)
        assert np.allclose(psf, cleanBeam)

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



