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

def Gaussian1D(extend,n,sig):
    xx=extend
    x=np.mgrid[-xx:xx:1j*n]
    rsq=x**2
    z=np.exp(-rsq/(2.*sig**2))
    return x,z




def Gaussian(extend,n,sig):
    xx=extend
    x,y=np.mgrid[-xx:xx:1j*n,-xx:xx:1j*n]
    rsq=x**2+y**2
    z=np.exp(-rsq/(2.*sig**2))
    return x,y,z

def GaussianSymetric(sig,n):
    x,y=np.mgrid[-n:n:1j*(2*n+1),-n:n:1j*(2*n+1)]
    rsq=x**2+y**2
    z=np.exp(-rsq/(2.*sig**2))
    return x,y,z

def GaussianXY(xin,yin,sin,off=(0.,0.),sig=(1.,1.),pa=0.):
    SigMin,SigMaj=1./(np.sqrt(2.)*sig[0]),1./(np.sqrt(2.)*sig[1])
    ang=pa
    SminCos=SigMin*np.cos(ang)
    SminSin=SigMin*np.sin(ang)
    SmajCos=SigMaj*np.cos(ang)
    SmajSin=SigMaj*np.sin(ang)
    x=xin-off[0]
    y=yin-off[1]
    up=x*SminCos-y*SminSin
    vp=x*SmajSin+y*SmajCos
    uvp=((x*SminCos-y*SminSin)**2+(x*SmajSin+y*SmajCos)**2)
    return sin*np.exp(-uvp)

def Gaussian2D(xin,yin,GaussPar=(1.,1.,0.)):
    
    S0,S1,PA=GaussPar
    SMaj=np.max([S0,S1])
    SMin=np.min([S0,S1])
    A=np.array([[1./SMaj**2,0],
                [0,1./SMin**2]])

    c,s,t=np.cos,np.sin,PA
    R=np.array([[c(t),-s(t)],
                [s(t),c(t)]])
    A=np.dot(np.dot(R.T,A),R)
    sOut=xin.shape
    x=np.array([xin.ravel(),yin.ravel()])
    R=[np.dot(np.dot(x[:,iPix].T,A),x[:,iPix]) for iPix in range(x.shape[-1])]
    return np.exp(-np.array(R)).reshape(sOut)

def testGaussian2D():
    xin,yin=np.mgrid[-10:10:101*1j,-10:10:101*1j]
    
    PA=10*np.pi/180.
    G=Gaussian2D(xin,yin,GaussPar=(1.,4.,PA))
    import pylab
    pylab.clf()
    pylab.imshow(G,interpolation="nearest")
    pylab.draw()
    pylab.show(False)
