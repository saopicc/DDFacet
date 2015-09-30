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

def GaussianXY(xin,yin,sin,off=(0.,0.),sig=(1.,1.),pa=0.):
    s0,s1=sig[0],sig[1]
    if s0==0.: s0=1e-6
    if s1==0.: s1=1e-6
    SigMin,SigMaj=1./(np.sqrt(2.)*s0),1./(np.sqrt(2.)*s1)
    ang=pa
    SminCos=SigMin*np.cos(ang)
    SminSin=SigMin*np.sin(ang)
    SmajCos=SigMaj*np.cos(ang)
    SmajSin=SigMaj*np.sin(ang)
    x=xin-off[0]
    y=yin-off[1]
    #up=x*SminCos-y*SminSin
    #vp=x*SmajSin+y*SmajCos
    uvp=((x*SminCos-y*SminSin)**2+(x*SmajSin+y*SmajCos)**2)
    return sin*np.exp(-uvp)
