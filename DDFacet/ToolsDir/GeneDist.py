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
