import numpy as np
import pylab


def test():
    
    N0=1000
    x=np.random.randn(N0)
    y=np.random.randn(N0)
    s=np.random.rand(N0)**2
    #s.fill(1)
    s[0]=50
    s[1]=30
    s-=np.min(s)
    s*=10
    s+=1
    Nk=20
    
    indC=np.int32(np.random.rand(Nk)*x.size)
    xc,yc=x[indC],y[indC]
    DicoSources={}

    ns=x.size
    while True:
        d=np.sqrt((x.reshape((ns,1))-xc.reshape((1,Nk)))**2+(y.reshape((ns,1))-yc.reshape((1,Nk)))**2)
        indk=np.argmin(d,axis=1)
        pylab.clf()
        for iK in range(Nk):
            ind=np.where(indk==iK)[0]
            xx=x[ind]
            yy=y[ind]
            ss=s[ind]

            xc[iK]=np.sum(ss*xx)/np.sum(ss)
            yc[iK]=np.sum(ss*yy)/np.sum(ss)
            c=np.ones(xx.size)*iK
            
            pylab.scatter(xx,yy,c=c,s=ss,vmin=0,vmax=Nk,lw=0)
            pylab.scatter(xc[iK],yc[iK],c="black",marker="s")
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        print "caca"
