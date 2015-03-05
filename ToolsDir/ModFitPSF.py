import scipy.optimize
import numpy as np

def gauss(x0,y0,SigMaj,SigMin,ang,x,y):

    #SigMaj,SigMin,ang=GaussPars

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


########################
def twoD_Gaussian((x, y, amplitude, offset), xo, yo, sigma_x, sigma_y, theta):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

from scipy.optimize import fsolve
import pylab
import numpy

def findIntersection(fun1,fun2,x0):
 return fsolve(lambda x : fun1(x) - fun2(x),x0)

def DoFit(PSF):
    popt=FitGauss2(PSF.flatten())
    xo, yo, sigma_x, sigma_y, theta=popt
    npi=int(theta/np.pi)
    theta-=npi*np.pi
    return np.abs(sigma_x), np.abs(sigma_y), theta

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

def FindSidelobe(PSF):
    x,y=np.where(PSF==np.max(PSF))
    x0=x[0]
    y0=y[0]
    profile=PSF[x0,:]


    PSFhalf=profile[y0::]
    dx=np.where(PSFhalf<0)[0][0]
    PSFsmall=PSF[x0-dx:x0+dx,y0-dx:y0+dx]
    
    popt=FitGauss2(PSFsmall.ravel())

    npix=int(np.sqrt(PSFsmall.ravel().shape[0]))-1
    x = np.linspace(0, npix, npix+1)
    y = np.linspace(0, npix, npix+1)
    x, y = np.meshgrid(x, y)
    data_fitted = twoD_Gaussian((x, y, 1, 0), *popt)
    N=npix
    D=data_fitted.reshape(N+1, N+1)
    PSFnew=PSF.copy()
    PSFnew[x0-dx:x0+dx,y0-dx:y0+dx]=PSFnew[x0-dx:x0+dx,y0-dx:y0+dx]-D[:,:]
    profile0=PSFnew[x0,:]

    return np.max(PSFnew)

    

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

