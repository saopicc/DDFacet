import numpy as np
from SkyModel.Other import ModColor

def init(n=1000):
    global x,y,s,ns
    ns=n
    x=np.random.randn(ns)#-0.5
    y=np.random.randn(ns)#-0.5
    s=np.random.rand(ns)
    s[0]=10
    s[1]=100

def test():
    RadialCluster(x,y,s,3)


class ClassClusterRadial():
    def __init__(self,x,y,s,NCluster=10,DoPlot=True):
        self.X=x
        self.Y=y
        self.S=s
        self.DoPlot=DoPlot
        self.NCluster=NCluster



    def Cluster(self):
        DoPlot=self.DoPlot
        NRing=self.NCluster
        x,y,s=self.X,self.Y,self.S
        

        r_s=np.sqrt(x**2+y**2)
        th_s=np.angle(x+1j*y)
        rmax=np.max(r_s)
        Ns=x.shape[0]
        Col=np.zeros_like(x)
        
        rspace=(np.linspace(0.,1.,NRing+1)**2)*rmax*1.01
        RegDef=[]
        nreg=np.array([1,4,6,8,12,16,20,24])[0:NRing]

        DictNode={}
        indr=0
        if DoPlot:
            import pylab
            pylab.clf()

        for i in range(NRing):
            r0,r1=rspace[i],rspace[i+1]
            th=np.linspace(0.,2.*np.pi,nreg[i]+1)-np.pi
            for j in range(nreg[i]):
                th0,th1=th[j],th[j+1]
                cond_r =(r_s>r0)&(r_s<r1)
                cond_th=(th_s>th0)&(th_s<th1)
                ind=np.where(cond_r&cond_th)[0]

                thline=np.linspace(th0,th1,100.)
                #print "ts: ",r0,r1,th0,th1
                rr=np.array([r0,r1])
                if DoPlot:
                    l0,l1=r0*np.cos(thline),r0*np.sin(thline); pylab.plot(l0,l1,color="black")#,c=indr)
                    l0,l1=r1*np.cos(thline),r1*np.sin(thline); pylab.plot(l0,l1,color="black")#,c=indr)
                    l0,l1=rr*np.cos(th0),rr*np.sin(th0); pylab.plot(l0,l1,color="black")#,c=indr)
                    l0,l1=rr*np.cos(th1),rr*np.sin(th1); pylab.plot(l0,l1,color="black")#,c=indr)

                if ind.shape[0]>0:
                    DictNode[indr]={"ListCluster":ind.tolist()}
                    indr+=1
                    #print th0,th1
                    Col[ind]=indr
                
    
        if DoPlot:
            pylab.scatter(x,y,c=Col)
            pylab.draw()
            pylab.show()
        return DictNode
