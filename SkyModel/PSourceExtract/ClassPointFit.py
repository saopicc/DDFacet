
import numpy as np
import Gaussian
import pylab
import scipy.optimize
import time
import ClassIslands
import ModColor
from progressbar import ProgressBar
from pyrap.images import image


class ClassPointFit():
    def __init__(self,x,y,z,psf=(1,1,0),noplot=False,noise=1.):
        self.psf=psf
        self.itera=0
        self.noise=noise
        self.noplot=False#True#noplot
        self.x=np.array(x)
        self.y=np.array(y)
        self.z=np.array(z)
        self.data=np.array([self.x.flatten(),self.y.flatten(),self.z.flatten()])
        self.itera=0
        self.xmin=None



    
    def PutFittedArray(self,A):
        if self.xmin==None: return
        x,y,z=self.data

        data2=self.func(self.xmin,self.data)
        A[(x.astype(np.int64),y.astype(np.int64))]+=data2
        return


        dx=5
        xxx,yyy=[],[]
        for i in range(x.shape[0]):
            x0,y0=round(x[i]),round(y[i])
            xx,yy=np.mgrid[x0-dx:x0+dx,y0-dx:y0+dx]
            xxx+=xx.flatten().tolist()
            yyy+=yy.flatten().tolist()
        LPix=list(set([(xxx[i],yyy[i]) for i in range(len(xxx))]))
        xxx=np.array([LPix[i][0] for i in range(len(LPix))])
        yyy=np.array([LPix[i][1] for i in range(len(LPix))])
        data2=self.func(self.xmin,(xxx,yyy,yyy))
        A[(xxx.astype(np.int64),yyy.astype(np.int64))]+=data2


    def DoAllFit(self,Nstart=1,Nend=10):

        bic_keep=1e10
        #print 
        for i in range(Nstart,Nend):
            xmin,bic=self.DoFit(i)
            #print xmin
            if xmin==None: break
            if bic<bic_keep:
                xmin_keep=xmin
                bic_keep=bic
            else:
                break
        if xmin==None:
            return []
        
        self.xmin=xmin_keep
        #print xmin_keep
        l,m,s=self.GetPars(xmin_keep)
        out=[(l[i],m[i],s[i]) for i in range(s.shape[0])]
        #print out
        return out

    def DoFit(self,Nsources=10):
        self.Nsources=Nsources

        self.itera=0
        #self.noplot=False
        guess=self.GiveGuess(Nsources)
        self.noplot=True
        lms=np.array(guess).T.flatten()
        self.itera=0

        try:
            xmin,retval=scipy.optimize.leastsq(self.funcResid, lms, args=(self.data,),xtol=1e-3,ftol=1e-3,gtol=1e-3)
            predict=self.func(xmin,self.data)
            x,y,Data=self.data
            #self.noplot=False
            #self.plotIter2(x,y,z,data2)
            #self.plotIter(z,data2)
            #self.noplot=True
            #l,m,s=self.GetPars(xmin)
            #pylab.plot(l,m,ls="",marker="+")
            #pylab.draw()
            #time.sleep(1)
            w=Data/np.sum(Data)
            #w=1
            bic=np.sum(w**2*(Data-predict)**2/(self.noise)**2)+Nsources*3*np.log(x.shape[0])
            #print "Number of parameters: %i, BIC=%f"%(Nsources,bic)
            return xmin,bic
        except:
            #print ModColor.Str("problem: ")+"Number of parameters: %i, Nmeas: %i,"%(Nsources,self.data.shape[1])
            return None,None


    def GetPars(self,pars):
        ns=pars.shape[0]/3
        l,m,s=pars[0:ns],pars[ns:2*ns],pars[2*ns::]
        s=np.abs(s)
        return l,m,s

    def func(self,pars,xyz):
        x,y,z=xyz
        l,m,s=self.GetPars(pars)
        G=np.zeros_like(x)
        for i in range(l.shape[0]):
            G+=Gaussian.GaussianXY(l[i]-x,m[i]-y,s[i],sig=(self.psf[0],self.psf[1]),pa=self.psf[2])
        return G
    
    def funcResid(self,pars,xyz):
            
        x,y,z=xyz
        G=self.func(pars,xyz)
        #print np.sum((G-z)**2)
        #self.plotIter(z,G)
        #Gn=G/np.max(G)
        #return (Gn*(G-z)).flatten()
        #val=np.max(1e-5,np.max(G))
        #Gn=G/val
        #return (Gn*(G-z)).flatten()
        w=z/np.max(z)
        w/=np.sum(w)
        #w=1
        return (w*(G-z)).flatten()

    def GiveGuess(self,Nsources):
        x,y,z=self.data.copy()
        S=[]
        
        #self.plotIter(self.data[2],z)
        for i in range(Nsources):
            ind=np.argmax(z)
            x0,y0,s0=x[ind],y[ind],z[ind]
            #print s0,np.max(z)
            z-=Gaussian.GaussianXY(x0-x,y0-y,s0,sig=(self.psf[0],self.psf[1]),pa=self.psf[2])
            S.append([x0,y0,s0])
            #self.plotIter2(x,y,self.data[2],z)
            #self.plotIter(self.data[2],z)
        return S
        
    def plotIter(self,z,G):
        if self.noplot: return
        nn=int(np.sqrt(z.shape[0]))
        vmin,vmax=np.min(z),np.max(z)
        pylab.clf()
        pylab.subplot(1,2,1)
        pylab.imshow(z.reshape(nn,nn),interpolation="nearest",vmin=vmin,vmax=vmax)
        pylab.subplot(1,2,2)
        pylab.imshow(G.reshape(nn,nn),interpolation="nearest",vmin=vmin,vmax=vmax)
        #pylab.imshow(G.reshape(nn,nn)-z.reshape(nn,nn),interpolation="nearest",vmin=vmin,vmax=vmax)
        pylab.title("iter=%i"%self.itera)
        pylab.draw()
        pylab.show()
        self.itera+=1

    def plotIter2(self,x,y,z,G):
        if self.noplot: return
        nn=int(np.sqrt(z.shape[0]))
        vmin,vmax=np.min(z),np.max(z)
        pylab.clf()
        pylab.subplot(1,2,1)
        pylab.scatter(x,y,c=z.tolist(),vmin=vmin,vmax=vmax)
        pylab.subplot(1,2,2)
        pylab.scatter(x,y,c=G.tolist(),vmin=vmin,vmax=vmax)
        pylab.title("iter=%i"%self.itera)
        pylab.draw()
        pylab.show()
        self.itera+=1
