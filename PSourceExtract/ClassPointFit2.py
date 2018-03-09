
import numpy as np
import Gaussian
import pylab
import scipy.optimize
import time
import ClassIslands
from Other import ModColor
from Other.progressbar import ProgressBar

def init():
    nn=101.
    x,y=np.mgrid[0:nn,0:nn]
    xx=sorted(list(set(x.flatten().tolist())))
    dx=xx[1]-xx[0]
    dx=1.5
    z=Gaussian.GaussianXY(x,y,1.,off=(50,50),sig=(1.2*dx,1.2*dx),pa=20.*np.pi/180)
    z+=Gaussian.GaussianXY(x,y,1.,off=(55,50),sig=(1.2*dx,1.2*dx),pa=20.*np.pi/180)
    z+=Gaussian.GaussianXY(x,y,.5,off=(25,25),sig=(1.2*dx,1.2*dx),pa=20.*np.pi/180)
    z+=Gaussian.GaussianXY(x,y,.5,off=(75,25),sig=(1.2*dx,1.2*dx),pa=20.*np.pi/180)
    noise=0.01
    #z+=np.random.randn(nn,nn)*noise
    # z+=Gaussian.GaussianXY(x,y,1.,off=(50,50),sig=(1*dx,1*dx),pa=20.*np.pi/180)
    #pylab.clf()
    dx*=1.5
    pylab.imshow(z,interpolation="nearest")
    pylab.show()
    Fit=ClassPointFit(x,y,z,psf=(dx,dx,0.),noise=noise)
    Fit.DoAllFit()

def init2():
    from pyrap.images import image
    im=image("/home/tasse/Desktop/FITS/image_049_073.img.restored.fits")
    PMaj=(im.imageinfo()["restoringbeam"]["major"]["value"])
    PMin=(im.imageinfo()["restoringbeam"]["minor"]["value"])
    PPA=(im.imageinfo()["restoringbeam"]["positionangle"]["value"])
    print ModColor.Str(" - Using psf (maj,min,pa)=(%6.2f, %6.2f, %6.2f)"
                           %(PMaj,PMin,PPA),col='green',Bold=False)


    ToSig=(1./3600.)*(np.pi/180.)/(2.*np.sqrt(2.*np.log(2)))
    PMaj*=ToSig
    PMin*=ToSig
    PPA*=np.pi/180

    b=im.getdata()[0,0,:,:]
    b=b[3000:4000,3000:4000]#[100:250,200:350]
    c=im.coordinates()
    incr=np.abs(c.dict()["direction0"]["cdelt"][0])
    print ModColor.Str("   - Psf Size Sigma_(Maj,Min) = (%5.1f,%5.1f) pixels"%(PMaj/incr,PMin/incr),col="green",Bold=False)
        
    Islands=ClassIslands.ClassIslands(b,10.,Boost=1,DoPlot=1)
    #Islands.Noise=30e-3
    Islands.FindAllIslands()
    sourceList=[]
    ImOut=np.zeros_like(b)
    pylab.ion()
    for i in range(len(Islands.ListX)):
        #comment='Isl %i/%i' % (i+1,len(Islands.ListX))
        #pBAR.render(int(100* float(i+1) / len(Islands.ListX)), comment)
        xin,yin,zin=np.array(Islands.ListX[i]),np.array(Islands.ListY[i]),np.array(Islands.ListS[i])
        xm=int(np.sum(xin*zin)/np.sum(zin))
        ym=int(np.sum(yin*zin)/np.sum(zin))
        #Fit=ClassPointFit(xin,yin,zin,psf=(PMaj/incr,PMin/incr,PPA),noise=Islands.Noise)
        Fit=ClassPointFit(xin,yin,zin,psf=(PMaj/incr,PMin/incr,PPA),noise=Islands.Noise[xm,ym])
        sourceList+=Fit.DoAllFit()
        Fit.PutFittedArray(ImOut)
    xlist=[]; ylist=[]; slist=[]
    for ijs in sourceList:
        i,j,s=ijs
        xlist.append(i); ylist.append(j); slist.append(s)

    Islands.FittedComps=(xlist,ylist,slist)
    Islands.FitIm=ImOut
    Islands.plot()


class ClassPointFit():
    def __init__(self,x,y,z,psf=(1,1,0),noplot=False,noise=1.):
        self.psf=psf#(0.1,0.1,0.)#psf
        self.itera=0
        self.noise=noise
        self.noplot=False#True#noplot
        self.x=np.array(x)
        self.y=np.array(y)
        self.z=np.array(z)
        self.data=np.array([self.x.flatten(),self.y.flatten(),self.z.flatten()])
        self.itera=0
        self.xmin=None
        self.xy_set=self.GiveTupleSet(x,y)

    
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
        l,m,s,dp=self.GetPars(xmin_keep)

        a0,b0=self.psf[0],self.psf[1]
        psf=self.givePsf(dp)
        a1,b1=psf[0],psf[1]

        ratio=a1*b1/(a0*b0)
        #print a1,b1,ratio
        out=[(l[i],m[i],s[i]*ratio) for i in range(s.shape[0])]
        # for i in range(l.shape[0]):
        #     print "l: %5.1f, s: %5.1f, s: %5.1f, psf: %5.1f"%(l[i],m[i],s[i],dp)

        return out

    def DoFit(self,Nsources=2):
        self.Nsources=Nsources

        self.itera=0
        #self.noplot=False
        lms=self.GiveGuess(Nsources)
        #self.noplot=True
        self.itera=0

        #try:
        if True:
            xmin,retval=scipy.optimize.leastsq(self.funcResid, lms, args=(self.data,),gtol=0)#maxfev=10000)#,xtol=1e-4)#,ftol=1e-4)#,gtol=1e-5)
            predict=self.func(xmin,self.data)
            x,y,Data=self.data
            w=Data/np.max(Data)
            #w=Data/np.sum(Data)
            w=1.
            #w=np.sqrt(w)

            l,m,s,dp=self.GetPars(xmin)
            psf=self.givePsf(dp)
            a1,b1=psf[0],psf[1]
            #a1,b1=self.psf[0],self.psf[1]
            npix=a1*b1*np.pi/4.

            sigma=self.noise*npix

            k=(Nsources*3+1)
            n=x.shape[0]

            G=self.func(xmin,self.data)
            #self.plotIter(Data,G)


            chi2=np.sum((Data-predict)**2/(2*sigma**2))

            aic=chi2+2*k#-2.*logL
            aicc=aic+(2.*k*(k+1)/(n-k-1.))
            # self.residual=np.sqrt(np.sum((Data-predict)**2))
            # St,err=self.GiveStErr(xmin)
            # #chi2=(err/St)**2
            bic=chi2+k*np.log(n)
            # print "Number of parameters: %i, bic=%f"%(Nsources,bic)
            #print "St=%f, errSt=%f"%(St,err)
            return xmin,bic#err
#        except:
#            return None,None

    def GetPars(self,pars):
        ns=pars.shape[0]/3
        l,m,s,dp=pars[0:ns],pars[ns:2*ns],pars[2*ns:3*ns],pars[-1]
        s=np.abs(s)
        # print
        # for i in range(l.shape[0]):
        #     print "C%i (l,m,s)=(%f, %f, %f)"%(i,l[i],m[i],s[i]*1000)
        # print "  with dp=%f"%dp
        return l,m,s,dp

    def givePsf(self,dp):
        #return (np.sqrt(self.psf[0]**2+dp**2),np.sqrt(self.psf[1]**2+dp**2),self.psf[2])
        return (self.psf[0]*(1.+dp),self.psf[1]*(1.+dp),self.psf[2])

    def func(self,pars,xyz):
        x,y,z=xyz
        l,m,s,dp=self.GetPars(pars)
        psf=self.givePsf(dp)
        # print "psf0: %s"%str(self.psf)
        # print "psf1: %s"%str(psf)

        G=np.zeros_like(x)
        for i in range(l.shape[0]):
            G+=Gaussian.GaussianXY(l[i]-x,m[i]-y,s[i],sig=(psf[0],psf[1]),pa=self.psf[2])
        return G

    def GiveTupleSet(self,l,m):
        return [(int(round(l[i])),int(round(m[i]))) for i in range(l.shape[0])]
    
    def funcResid(self,pars,xyz):
        

        x,y,z=xyz
        G=self.func(pars,xyz)
        #print np.sum((G-z)**2)
        #self.plotIter(z,G)
        #self.plotIter2(x,y,z,G,pars=pars)
        #Gn=G/np.max(G)
        #return (Gn*(G-z)).flatten()
        #val=np.max(1e-5,np.max(G))
        #Gn=G/val
        #return (Gn*(G-z)).flatten()
        w=z/np.max(z)
        #w/=np.sum(w)
        w=1
        #w=np.sqrt(w)
        l,m,s,dp=self.GetPars(pars)
        ans=(w*(G-z)).flatten()#*(1+0.01*np.exp(np.abs(x*2)))#(1+1e-5*np.exp(np.abs(x*2)))#*(1+np.exp(np.abs(dp)*4-13))
        #ans+=np.random.randn(ans.shape[0])*self.noise
        thr=.5
        if np.abs(dp)>thr:
            ans*=5.*np.abs(dp)/thr
        lm_set=self.GiveTupleSet(l,m)
        for i in range(len(lm_set)):
            if not(lm_set[i] in self.xy_set):
                ans*=5.

        return ans

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
        S=np.array(S).T.flatten()
        S=np.array(S.tolist()+[.1])

        return S
        
    def plotIter(self,z,G):
        #if self.noplot: return
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

    def plotIter2(self,x,y,z,G,pars=None):
        #return
        #if self.noplot: return

        nn=int(np.sqrt(z.shape[0]))
        vmin,vmax=np.min(z),np.max(z)
        pylab.clf()
        pylab.subplot(1,2,1)
        pylab.scatter(x,y,c=z.tolist(),vmin=vmin,vmax=vmax)
        pylab.subplot(1,2,2)
        pylab.scatter(x,y,c=G.tolist(),vmin=vmin,vmax=vmax)
        if pars!=None:
            l,m,s,dp=self.GetPars(pars)
            pylab.scatter(l,m,marker="+")
        pylab.title("iter=%i"%self.itera)
        pylab.draw()
        pylab.show()
        self.itera+=1


    def GiveStErr(self,pars):
        l,m,s,dp=self.GetPars(pars)
        psf=self.givePsf(dp)
        SThMsq=(psf[0]**2)*2./self.GiveRhoSq(pars,mode='M')
        SThmsq=(psf[1]**2)*2./self.GiveRhoSq(pars,mode='m')
        ThM=psf[0]
        Thm=psf[1]
        ThN=(self.psf[0]+self.psf[1])/2.
        
        a0,b0=self.psf[0],self.psf[1]
        a1,b1=psf[0],psf[1]
        ratio=a1*b1/(a0*b0)
        St=s*ratio
        SErrsq=(s**2)*2./self.GiveRhoSq(pars,mode='P')
        StErr=(St**2)*((SErrsq/St**2)+(ThN**2/(ThM*Thm))*(SThMsq/ThM**2 + SThmsq/Thm**2))
        return np.sum(St),np.sqrt(np.sum(StErr))

    def GiveRhoSq(self,pars,mode='P'):
        l,m,s,dp=self.GetPars(pars)
        psf=self.givePsf(dp)
        ThM=psf[0]
        Thm=psf[1]
        ThN=(self.psf[0]+self.psf[1])/2.
        if mode=="P":
            aM=am=3./2
        elif mode=="M":
            aM,am=5./2,1/2.
        elif mode=="m":
            aM,am=1./2,5./2
        m0=ThM*Thm/(4.*ThN**2)
        m1=(1.+(ThN/ThM)**2)**aM
        m2=(1.+(ThN/Thm)**2)**am
        m3=(s/np.max(self.residual,self.noise))**2
        rhosq=m0*m1*m2*m3
        return rhosq

