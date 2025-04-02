
from __future__ import division, absolute_import, print_function
import numpy as np
import scipy.signal

def test():
    S=np.load("ListPix.npz")
    ListPix=S["ListPix"]
    Mask=S["Mask"]
    CII=ClassIncreaseIsland(Mask=Mask)
    L=CII.IncreaseIslandFFT(ListPix,dx=10)
    

def Gaussian2D(x,y,GaussPar=(1.,1.,0)):
    d=np.sqrt(x**2+y**2)
    sx,sy,_=GaussPar
    if sx==0: sx=1e-6
    if sy==0: sy=1e-6
    return np.exp(-x**2/(2.*sx**2)-y**2/(2.*sy**2))
  
class ClassIncreaseIsland():
    def __init__(self,Mask=None):
        self.Mask=Mask # is inverted ie 0 within ilands
        pass

    def IncreaseIsland(self,*args,**kwargs):
        return self.IncreaseIslandFFT(*args,**kwargs)

    def IncreaseIslandFFT(self,ListPix,dx=2,AllowMasked=True):
        
        sx,sy=dx,dx
        GaussPar=(sx,sy,0)
        dxy=6.
        Nsx,Nsy=dxy*sx,dxy*sy
        xin,yin=np.mgrid[-Nsx:Nsx:(2*Nsx+1)*1j,-Nsy:Nsy:(2*Nsy+1)*1j]
        G=Gaussian2D(xin,yin,GaussPar=GaussPar)
        G/=np.sum(G)

        sx,sy=dx,dx
        GaussPar=(sx,sy,0)
        dxy=6.
        Nsx,Nsy=sx,sy
        xin,yin=np.mgrid[-Nsx:Nsx:(2*Nsx+1)*1j,-Nsy:Nsy:(2*Nsy+1)*1j]
        G=np.float32(np.sqrt(xin**2+yin**2)<dx)
        G/=np.sum(G)
        
        x,y=np.array(ListPix).T

        x0=x.min()
        y0=y.min()
        x1=x.max()
        y1=y.max()
        nx=x1-x0+1
        ny=y1-y0+1

        nxG,nyG=G.shape
        dx0=nxG#//2
        dy0=nyG#//2
        #dx0=dy0=0
        
        A=np.zeros((nx+2*dx0,ny+2*dy0),np.float32)
        
        xx=x-x0+dx0
        yy=y-y0+dy0
        A[xx,yy]=1

        
        Ac=scipy.signal.fftconvolve(A,G, mode='same')

        indx,indy=np.where(Ac>(1e-3*Ac.max()))
        W=Ac[indx,indy]
        xx=indx+x0-dx0
        yy=indy+y0-dy0

        if not AllowMasked:
            MM=self.Mask[0,0,xx,yy]
            ind=np.where((MM==0))[0]
            xx=xx[ind]
            yy=yy[ind]
            W=W[ind]

        OutListPix2=np.array([xx,yy]).T
        
        # import pylab
        # ax=pylab.subplot(1,3,1)
        # pylab.imshow(G)
        # ax=pylab.subplot(1,3,2)
        # pylab.imshow(A)
        # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
        # pylab.imshow(Ac)
        # pylab.draw()
        # pylab.show()

        return OutListPix2.tolist(),W.tolist()

    
    def IncreaseIslandSlow(self,ListPix,dx=2,AllowMasked=True):
        #np.savez("ListPix.npz",ListPix=ListPix,Mask=self.Mask)
        #stop
        _,_,Nx,Ny=self.Mask.shape
        nx=dx*2+1
        xg,yg=np.mgrid[-dx:dx:1j*nx,-dx:dx:1j*nx]
        xg=np.int64(xg.flatten())
        yg=np.int64(yg.flatten())
        OutListPix=[]
        #AA=np.zeros((Nx,Ny),bool)
        #BB=np.zeros((Nx,Ny),bool)
        for Pix in ListPix:
            x0,y0=Pix
            #AA[x0,y0]=1
            x1=xg+x0
            y1=yg+y0
            for iPixAdd in range(xg.size):
                i,j=x1[iPixAdd],y1[iPixAdd]
                try:
                    if AllowMasked:
                        OutListPix.append((i,j))
                    else:
                        if self.Mask[0,0,i,j]==0:
                            OutListPix.append((i,j))
                except:
                    print("failed",i,j)
                    pass
        OutListPix=list(set(OutListPix))
        OutListPix2=[[x,y] for x,y in OutListPix]

        # np.savez("MM.npz",M=np.logical_not(self.Mask[0,0]),A=AA)
        # import pylab
        # ax=pylab.subplot(1,3,1)
        # pylab.imshow(np.logical_not(self.Mask[0,0])[:,:])
        # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
        # pylab.imshow(AA)
        # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
        # pylab.imshow(BB)
        # pylab.draw()
        # pylab.show()
        # stop
        return OutListPix2

