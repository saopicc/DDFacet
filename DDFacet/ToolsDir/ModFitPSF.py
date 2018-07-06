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
import gaussfitter2
import scipy.ndimage.measurements

from DDFacet.Other import MyLogger
log= MyLogger.getLogger("FitPSF")


def FitCleanBeam(PSF):
    """
    Does a least squares fit of an eliptical gaussian to the PSF to determine the
    parameters for the clean beam.

    Returns:
        sigma_x of gaussian in pixels
        sigma_y of gaussian in pixels
        theta: The rotation angle (radians) of the gaussian from the y axis counter-clockwise (beware of the periodicity)
    """
    # zero everything except the main lobe
    PSF1 = PSF.copy()
    labels = scipy.ndimage.measurements.label(PSF1>=0)[0]
    amax = np.unravel_index(np.argmax(PSF), PSF.shape)
    PSF1[labels != labels[amax]] = 0
    popt=gaussfitter2.gaussfit(PSF1, vheight=0, return_all=0, rotate=1)
    amp, xo, yo, sigma_x, sigma_y, theta=popt
    theta = np.deg2rad(theta)
    gmaj, gmin = (0, 0)
    '''
    Because two gaussians with parameters (||sigma_x||,||sigma_y||,theta) and
    (||sigma_y||,||sigma_x||,theta-PI/2) looks exactly the same the fitted parameters may
    swap between runs, so take this into account
    '''
    if np.abs(sigma_y) >= np.abs(sigma_x):
        gmaj = np.abs(sigma_y)
        gmin = np.abs(sigma_x)
        theta += np.pi / 2
    else:
        gmaj = np.abs(sigma_x)
        gmin = np.abs(sigma_y)
    #ensure the angle is 0 <= th < PI
    theta = 2*np.pi - theta
    theta = theta - (int(theta / (2 * np.pi)) * 2 * np.pi)
    theta = theta if theta > 0 else 2*np.pi + theta
    theta = theta if theta <= np.pi else theta - np.pi

    #print gmaj, gmin, theta

    return gmaj, gmin, theta

def FindSidelobe(PSF):
    """
    Finds the maximum sidelobe level in the PSF outside the fit region and the
    position of this maximum. This will very likely be the level of the second
    (absolute) lobe.

    Args:
        PSF, a 2D array containing the dirty beam (can be a cropped/windowed PSF)
    Returns:
        Tupple:
            level: PSF sidelobe level
            null_px: position of the first sidelobe from the the centre of the PSF window in pixels
        Tupple:
            cb_maj: major axis sigma of clean beam
            cb_min: minor axis sigma of clean beam
            th: rotation angle of clean beam (radians)
    """
    # Peak of PSF (may be less than unity because of DD terms)
    # Assume there is only one peak
    amax = np.unravel_index(np.argmax(PSF), PSF.shape)
    nx,ny=PSF.shape
    
    PSF1 = PSF.copy()
    labels = scipy.ndimage.measurements.label(PSF1>=0)[0]
    PSF1[labels == labels[amax]] = 0
    
    amax_sl = np.unravel_index(np.argmax(PSF1), PSF1.shape)
    sidelobe_dist = np.sqrt((amax_sl[0] - amax[0])**2 + (amax_sl[1] - amax[1])**2)
    sidelobe_level = PSF1[amax_sl]
    
    return sidelobe_level, sidelobe_dist

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np
    import gaussfitter2

    xx,yy = np.meshgrid(np.linspace(-500,500,1000),np.linspace(-500,500,1000))
    g = gaussfitter2.twodgaussian([1,0,0,50,100,10],0,1,0)(xx,yy)
    params = FitCleanBeam(g)
    print params * np.array([1,1,180/np.pi])
    g2 = gaussfitter2.twodgaussian([1,0,0,params[1],params[0],np.rad2deg(params[2])],0,1,0)(xx,yy)
    plt.figure()
    plt.imshow(g)
    plt.figure()
    plt.imshow(g2)
    plt.show()

''' Old code
def gauss(x0,y0,SigMaj,SigMin,ang,x,y):
    # SigMaj,SigMin,ang=GaussPars
    CT=np.cos(ang)
    ST=np.sin(ang)
    C2T=np.cos(2*ang)
    S2T=np.sin(2*ang)
    sx2=SigMaj**2
    sy2=SigMin**2
    a=(CT**2/(2.*sx2))+(ST**2/(2.*sy2))
    b=-(S2T/(4.*sx2))+(S2T/(4.*sy2))
    c=(ST**2/(2.*sx2))+(CT**2/(2.*sy2))

    k=a*x**2+2.*b*x*y+c*y**2
    Gauss=np.exp(-k)
    #Gauss/=np.sum(Gauss)
    return Gauss

def residuals(x0,y0,SigMaj,SigMin,ang,x,y,PSF):
    return (PSF-gauss(x0,y0,SigMaj,SigMin,ang,x,y)).flatten()

def FitGauss(PSF):
    npix,_=PSF.shape
    x0,y0=npix/2,npix/2
    #SigMaj,SigMin,ang
    
    PSFSlice=np.max(PSF,axis=0)
    SigMaj,SigMin,ang=1.,1.,0
    StartSol=np.array([x0,y0,SigMaj,SigMin,ang])
    N=npix
    X,Y=np.mgrid[-npix/2:npix/2:N*1j,-npix/2:npix/2:N*1j]
    print scipy.optimize.leastsq(residuals, StartSol, args=(X,Y,PSF))

def test():
    npix=20
    N=npix

    x,y=np.mgrid[-npix/2:npix/2:N*1j,-npix/2:npix/2:N*1j]
    ang=30.*np.pi/180
    PSF=gauss(10,10,1,2,ang,x,y)
    FitGauss2(PSF)

def findIntersection(fun1,fun2,x0):
    return fsolve(lambda x : fun1(x) - fun2(x),x0)

def FitGauss2(PSF):
    npix=int(np.sqrt(PSF.shape[0]))-1
    x0,y0=npix/2,npix/2
    #SigMaj,SigMin,ang

    PSFSlice=np.max(PSF.reshape(npix+1, npix+1),axis=0)

    # import pylab
    # pylab.clf()
    # pylab.plot(PSFSlice)
    # pylab.draw()
    # pylab.show(False)

    # print np.interp(np.arange(PSFSlice.size),PSFSlice, np.max(PSFSlice)/2.)
    # stop

    SigMaj,SigMin,ang=3.,3.,0
    StartSol=x0, y0, SigMaj,SigMin,ang
    #N=npix
    #X,Y=np.mgrid[-npix/2:npix/2:N*1j,-npix/2:npix/2:N*1j]

    x = np.linspace(0, npix, npix+1)
    y = np.linspace(0, npix, npix+1)
    x, y = np.meshgrid(x, y)

    amplitude, offset=1.,0.
    popt, pcov = scipy.optimize.curve_fit(twoD_Gaussian, (x, y, amplitude, offset), PSF, p0=StartSol)
    #check the covariance matrix diagonal for variation on fitted parameters:
    fitStdDev = np.sqrt(np.diag(pcov))
    fitStdDev[4] *= np.max([fitStdDev[2],fitStdDev[3]]) #pixel offset at the edge of the major axis (delta arc length)
    if np.any(3*fitStdDev >= 1): #99.7% of the time the fitted PSF will look the same between runs
        print>>log, ModColor.Str("Your fitted PSF parameters may be off by 1 or more pixels. The resulting restored image may be incorrect.","red")
    else:
        print>>log, ModColor.Str("Your PSF should be fitted correctly to within "
                                 "(x0,y0,Sx,Sy,theta) px = (%.3f,%.3f,%.3f,%.3f,%.3f) px" %
                                 tuple(fitStdDev.tolist()),"green")
    vmin,vmax=0,1
    N=npix
    data_noisy=PSF
    data_fitted = twoD_Gaussian((x, y, 1, 0), *popt)
    #print popt
    # pylab.clf()
    # pylab.subplot(1,2,1)
    # pylab.imshow(data_noisy.reshape(N+1, N+1), cmap=pylab.cm.jet, origin='bottom',
    #              extent=(x.min(), x.max(), y.min(), y.max()),vmin=vmin,vmax=vmax,interpolation="nearest")
    # pylab.subplot(1,2,2)
    # pylab.imshow(data_fitted.reshape(N+1, N+1), cmap=pylab.cm.jet, origin='bottom',
    #              extent=(x.min(), x.max(), y.min(), y.max()),vmin=vmin,vmax=vmax,interpolation="nearest")
    # print np.min(data_noisy),np.max(data_noisy)
    # print np.min(data_fitted),np.max(data_fitted)
    # pylab.draw()
    # pylab.show(False)

    return popt


def twoD_Gaussian((x, y, amplitude, offset), xo, yo, sigma_x, sigma_y, theta):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def testFindSidelobe():
    from pyrap.images import image
    im=image("ImageTest2.psf.fits")
    PSF=im.getdata()[0,0]
    FindSidelobe(PSF)

def test2():
    import pylab
    # Create x and y indices
    N=200
    x = np.linspace(0, N, N+1)
    y = np.linspace(0, N, N+1)
    x, y = np.meshgrid(x, y)

    #create data
    data = twoD_Gaussian((x, y, 1, 0), 100, 100, 20, 40, 30*np.pi/180)

    # plot twoD_Gaussian data generated above
    #pylab.figure()
    #pylab.imshow(data.reshape(N+1, N+1))
    #pylab.colorbar()




    data_noisy = data + 0.2*np.random.normal(size=data.shape)

    popt=FitGauss2(data_noisy)


    data_fitted = twoD_Gaussian((x, y, 1, 0), *popt)
'''

