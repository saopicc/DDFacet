
import numpy as np

def MyCumulHist(x,NBins=100,Norm=True):
    xmin,xmax=x.min(),x.max()
    
    X=np.linspace(xmin,xmax,NBins)
    dx=-(x.reshape((x.size,1))-X.reshape((1,X.size)))
    N=np.float32(np.sum(dx>0,axis=0))
    if np.isnan(N[-1])==True: stop
    if N[-1]==0: stop

    if Norm:
        N/=N[-1]
        #print N[-1]
    return X,N
