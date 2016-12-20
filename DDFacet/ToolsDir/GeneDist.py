import numpy as np

def test():
    X=np.random.randn(100)
    W=np.ones_like(X)
    W[0]=50
    DM=ClassDistMachine()

    DM.setRefSample(X,W)
    DM.GiveSample(1000)



class ClassDistMachine():
    def __init__(self):
        pass
        
    def setCumulDist(self,x,y):
        self.xyCumulDist=x,y

    
    def setRefSample(self,X,W=None,Ns=100,xmm=None):
        x,D=self.giveCumulDist(X,W=W,Ns=Ns,xmm=xmm)
        self.xyCumulD=(x,D)
        
        
    def giveCumulDist(self,X,W=None,Ns=10,Norm=True,xmm=None):
        xmin=X.min()
        xmax=X.max()
        if xmm is None:
            xr=xmax-xmin
            x=np.linspace(xmin-0.1*xr,xmax+0.1*xr,Ns)
        else:
            x=np.linspace(xmm[0]-1e-6,xmm[1]+1e-6,Ns)
            

        if W is None:
            W=np.ones((X.size,),np.float32)

        D=(x.reshape((Ns,1))>(X).reshape((1,X.size)))*W.reshape((1,X.size))
        D=np.float32(np.sum(D,axis=1))

        if Norm:
            D/=D[-1]
        
        return x,D

    #def InterpDist(self):
        
    def giveDist(self,X,W=None,Ns=10,Norm=True,xmm=None):
        xmin=X.min()
        xmax=X.max()
        if xmm is None:
            xr=xmax-xmin
            x=np.linspace(xmin-0.1*xr,xmax+0.1*xr,Ns)
        else:
            x=np.linspace(xmm[0]-1e-6,xmm[1]+1e-6,Ns)
            

        if W is None:
            W=np.ones((X.size,),np.float32)

        D=(x.reshape((Ns,1))>(X).reshape((1,X.size)))*W.reshape((1,X.size))
        D0=(x[0:-1].reshape((Ns-1,1))<(X).reshape((1,X.size)))
        D1=(x[1::].reshape((Ns-1,1))>(X).reshape((1,X.size)))
        D2=(D0&D1)

        D=np.float32(np.sum(D2,axis=1))

        if Norm:
            D/=np.sum(D)
        xm=(x[0:-1]+x[1::])/2.
        return xm,D

    #def InterpDist(self):
        


    def GiveSample(self,N):
        ys=np.random.rand(N)
        xd,yd=self.xyCumulD
        xp=np.interp(ys, yd, xd, left=None, right=None)

        # x,y=self.giveCumulDist(xp,Ns=10)
        # pylab.clf()
        # pylab.plot(xd,yd)
        # pylab.plot(x,y)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        
        return xp
