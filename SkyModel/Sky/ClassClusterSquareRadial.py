from __future__ import division, absolute_import, print_function
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


class ClassClusterSquareRadial():
    def __init__(self,x,y,s,
                 NCluster=10,
                 DoPlot=True,
                 D_FITS=None):

        
        self.X=x
        self.Y=y
        self.S=s
        self.WSq=D_FITS["NPix"]*D_FITS["dPix"]/3
        
        self.DoPlot=DoPlot
        self.NCluster=NCluster



    def Cluster(self):
        DoPlot=self.DoPlot
        
        x,y,s=self.X,self.Y,self.S
        

        th_s=np.angle(x+1j*y)
        
        Col=np.zeros_like(x)
        
        RegDef=[]

        DictNode={}
        indr=0
        if DoPlot:
            import pylab
            pylab.clf()

        nreg=self.NCluster-1
        th=np.linspace(0.,2.*np.pi,nreg+1)-np.pi
        Cluster=np.zeros((x.size,),np.int)
        
        for j in range(th.size-1):
            th0,th1=th[j],th[j+1]
            cond_th=(th_s>th0)&(th_s<=th1)
            ind=np.where(cond_th)[0]
            Cluster[ind]=j+1

        Cx=((x>-self.WSq/2)&(x<self.WSq/2))
        Cy=((y>-self.WSq/2)&(y<self.WSq/2))
        
        Cluster[Cx&Cy]=0
        print(np.unique(Cluster))
        
        for j in range(self.NCluster):
            ind=np.where(Cluster==j)[0]
            
            if ind.shape[0]>0:
                DictNode[indr]={"ListCluster":ind.tolist()}
                indr+=1
                #print th0,th1
                Col[ind]=indr
                
        if DoPlot:
            pylab.scatter(x,y,c=Col)
            pylab.draw()
            pylab.show()#block=False)
            #pylab.pause(0.1)


        return DictNode
