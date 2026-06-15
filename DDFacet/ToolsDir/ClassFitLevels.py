import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize

def test():
    
    S=np.load("PNG/AB_Major1_14080_12812.npz",allow_pickle=1)
    ich=0
    _,A,B,model,indx,indy,Bapp=S["LAB"][ich]
    CF=ClassFitLevels(A,B,model)
    CF.solve()
    
    
class ClassFitLevels():
    def __init__(self,Dirty,PSF,Model,NegMask):
        self.Dirty=Dirty
        self.PSF=PSF
        self.Model=Model
        self.NegMask=NegMask
        if len(self.Dirty.shape)!=2: stop
        if len(self.PSF.shape)!=2: stop
        if len(self.Model.shape)!=2: stop
        W=np.ones(self.NegMask.shape,np.float32)
        indx,indy=np.where(self.NegMask==1)
        nx,ny=W.shape
        W.flat[indx*ny+indy]=0
        self.W=W
        
    def solve(self):
        Dirty=self.Dirty
        PSF=self.PSF
        Model=self.Model
        W=self.W

        Model1=Model.copy()
        Model1[Model1!=0]=1
        Mc=fftconvolve(Model,PSF,mode="same")
        Mc1=fftconvolve(Model1,PSF,mode="same")
        
        def giveChi2(x):
            a,b=x.ravel()
            R=Dirty-(a*Mc+b*Mc1)
            Chi2=(np.sqrt(np.sum((R*W)**2)))
            return Chi2

        x0=np.array([1,0],np.float32)
        res = minimize(giveChi2, x0)#, constraints=cons)
        x = res.x
        a,b=x.ravel()
        M=(a*Model+b*Model1)
        

        return M
        
