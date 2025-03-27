import numpy as np
from scipy.spatial.distance import pdist, squareform
import pylab

class SVGD():

    def __init__(self,LikelyhoodModel=None,
                 ArrayMethodsMachine=None):
        self.ArrayMethodsMachine=ArrayMethodsMachine
        self.LikelyhoodModel=LikelyhoodModel
        pass
    
    def svgd_kernel(self, theta, h = -1, n=2):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**n
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))

        # compute the rbf kernel
        Kxy = np.exp( -pairwise_dists / h**n / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)
    
 
    def update(self, x0, n_iter = 1000, stepsize = 1e-3, bandwidth = -1, alpha = 0.90, debug = False,DoPlot=False):
        # Check input
        lnprob=self.LikelyhoodModel.dlnprob
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')
        x00=x0[0]
        theta = np.copy(x0) 
        
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0

        LdL=[]
        LdChi2_Chi2=[]
        Chi2_0=1e100
        sig_dChi2_Chi2=0.1
        mean_dChi2_Chi2=1.
        max_dChi2_Chi2=1.
        for ii in range(n_iter):

            if DoPlot and ii%1==0:
                self.ArrayMethodsMachine._PlotIndiv(theta,iChannel=0,Mode="Rand",Title="Iter=%i"%ii)
                #self.ArrayMethodsMachine._PlotIndiv(theta,iChannel=0,Mode="MeanIm")

            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta, h = -1, n=2)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]  
            # adagrad 
            if ii == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad
            theta[0]=x00

            Chi2,_=self.LikelyhoodModel.lnprob(theta)
            if DoPlot:
                print("[%.4i]@%.4i (alpha=%.5f)"%(self.ArrayMethodsMachine.iIsland,ii,stepsize),Chi2)
            
            dChi2_Chi2=(Chi2-Chi2_0)/np.abs(Chi2_0)
            LdL.append(Chi2)
            LdChi2_Chi2.append(dChi2_Chi2)
            if np.array(LdChi2_Chi2).size>10:
                sig_dChi2_Chi2=np.std(np.array(LdChi2_Chi2)[-10:])
                max_dChi2_Chi2=np.max(np.abs(np.array(LdChi2_Chi2)[-10:]))
            if dChi2_Chi2>sig_dChi2_Chi2:
                print("[%.4i]@%.4i (alpha=%.5f, Sig_dChi2_Chi2=%.5f)"%(self.ArrayMethodsMachine.iIsland,ii,stepsize,sig_dChi2_Chi2),"Reduce Stepsize by half")
                stepsize=stepsize/2
            if dChi2_Chi2<0 and max_dChi2_Chi2<1e-2:
                print("[%.4i]@%.4i (alpha=%.5f, Sig_dChi2_Chi2=%.5f)"%(self.ArrayMethodsMachine.iIsland,ii,stepsize,sig_dChi2_Chi2),"STOP")
                return theta
            Chi2_0=Chi2
            print("[%.4i]@%.4i (alpha=%.5f, Sig_dChi2_Chi2=%.5f)"%(self.ArrayMethodsMachine.iIsland,ii,stepsize,sig_dChi2_Chi2),Chi2,dChi2_Chi2)
            
        return theta
    
