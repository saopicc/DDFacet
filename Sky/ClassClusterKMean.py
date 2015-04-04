import numpy as np
import pylab

def test():
    
    N0=1000
    x=np.random.randn(N0)
    y=np.random.randn(N0)
    s=np.random.rand(N0)**2
    #s.fill(1)
    s=(s-np.min(s))/(np.max(s)-np.min(s))
    s+=1
    s*=10
    
    s[0]=50
    s[1]=30


    Nk=10

    CM=ClassClusterKMean(x,y,s,Nk)
    CM.Cluster()

class ClassClusterKMean():
    def __init__(self,x,y,s,NCluster=10,DoPlot=True):
        self.X=x
        self.Y=y
        self.S=s
        self.NCluster=NCluster
        self.DoPlot=DoPlot

    def Cluster(self):
        x=self.X
        y=self.Y
        s=self.S

        nk=Nk=self.NCluster
        
        indC=np.int32(np.random.rand(Nk)*x.size)
        xc,yc=x[indC],y[indC]
        DicoSources={}
    
        ns=x.size
        while True:
            #d=s.reshape((ns,1))*np.sqrt((x.reshape((ns,1))-xc.reshape((1,Nk)))**2+(y.reshape((ns,1))-yc.reshape((1,Nk)))**2)
            d=np.sqrt((x.reshape((ns,1))-xc.reshape((1,Nk)))**2+(y.reshape((ns,1))-yc.reshape((1,Nk)))**2)
            indk=np.argmin(d,axis=1)
            xc0=xc.copy()
            yc0=yc.copy()
            #pylab.clf()
            for iK in range(Nk):
                ind=np.where(indk==iK)[0]
                if ind.size==0: continue
                xx=x[ind]
                yy=y[ind]
                ss=s[ind]
                
                xc[iK]=np.sum(ss*xx)/np.sum(ss)
                yc[iK]=np.sum(ss*yy)/np.sum(ss)
                c=np.ones(xx.size)*iK
                
                if self.DoPlot:
                    pylab.scatter(xx,yy,c=c,s=ss,vmin=0,vmax=Nk,lw=0)
                    pylab.scatter(xc[iK],yc[iK],c="black",marker="s")
            if self.DoPlot:
                pylab.draw()
                pylab.show(False)
                pylab.pause(0.1)
            condx=np.allclose(xc,xc0)
            condy=np.allclose(yc,yc0)
            if condx&condy: break

        d=np.sqrt((x.reshape((ns,1))-xc.reshape((1,Nk)))**2+(y.reshape((ns,1))-yc.reshape((1,Nk)))**2)
        indk=np.argmin(d,axis=1)
        
        if self.DoPlot:
            pylab.clf()
            pylab.scatter(x,y,c=indk,s=ss,vmin=0,vmax=Nk,lw=0)
            pylab.scatter(xc,yc,c="black",marker="s")
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)
    

        KK={}
        keys=[]
        for i in range(nk):
            key="%3.3i"%i
            keys.append(key)
            KK[key]={"ListCluster":[]}
    
        xnode=xc
        ynode=yc

        for i in range(x.shape[0]):
            d=np.sqrt((x[i]-xnode)**2+(y[i]-ynode)**2)
            ind=np.where(d==np.min(d))[0][0]
            (KK[keys[ind]])["ListCluster"].append(i)

        
        return KK
    
