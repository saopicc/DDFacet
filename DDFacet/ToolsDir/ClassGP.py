'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA-SA, Rhodes University

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
import scipy as scp
from scipy.linalg import solve_triangular as soltri
from scipy import optimize as opt

class ClassGP(object):
    def __int__(self,x,xp):
        self.x = x
        self.xp = xp
        self.XX = self.abs_diff(self.x,self.x)
        self.XXp = self.abs_diff(self.x, self.xp)
        self.XXpp = self.abs_diff(self.xp, self.xp)

    def abs_diff(self,x, xp):
        """
        Creates matrix of differences (x_i - x_j) for vectorising.
        """
        N = x.size
        Np = xp.size
        return np.tile(x, (Np, 1)).T - np.tile(xp, (N, 1))

    def diag_dot(self,A,B):
        D = np.zeros(A.shape[0])
        for i in xrange(A.shape[0]):
            D[i] = np.dot(A[i,:],B[:,i])
        return D

    def logp_and_gradlogp(self,theta,y,N):
        """
        Returns the negative log (marginal) likelihood (the function to be optimised) and its gradient
        """
        #tmp is Ky
        tmp = self.cov_func(theta,self.XX)
        #tmp is L
        tmp = scp.linalg.cholesky(tmp)
        detK = 2.0*np.sum(np.log(np.diag(tmp)))
        #tmp is Linv
        tmp = soltri(tmp.T,np.eye(N).T)
        #tmp2 is Linvy
        tmp2 = np.dot(tmp,y)
        logp = np.dot(tmp2.T,tmp2)/2.0 + detK/2.0 + N*np.log(2*np.pi)/2.0
        nhypers = theta.size
        dlogp = np.zeros(nhypers)
        #tmp is Kinv
        tmp = np.dot(tmp.T,tmp)
        #tmp2 becomes Kinvy
        tmp2 = np.reshape(np.dot(tmp,y),(N,1))
        #tmp2 becomes aaT
        tmp2 = np.dot(tmp2,tmp2.T)
        #tmp2 becomes Kinv - aaT
        tmp2 = tmp - tmp2
        dKdtheta = self.dcov_func(theta,self.XX,mode=0)
        dlogp[0] = np.sum(self.diag_dot(tmp2,dKdtheta))/2.0
        dKdtheta = self.dcov_func(theta,self.XX,mode=1)
        dlogp[1] = np.sum(self.diag_dot(tmp2,dKdtheta))/2.0
        dKdtheta = self.dcov_func(theta,self.XX,mode=2)
        dlogp[2] = np.sum(self.diag_dot(tmp2,dKdtheta))/2.0
        return logp,dlogp

    def cov_func(self,theta,x,mode="Noise"):
        """
        Covariance function including noise variance
        """
        if mode != "Noise":
            #Squared exponential
            return theta[0]**2.0*np.exp(-x**2.0/(2.0*theta[1]**2.0))
        else:
            #Squared exponential
            return theta[0]**2*np.exp(-x**2.0/(2.0*theta[1]**2.0)) + theta[2]**2.0*np.eye(x.shape[0])

    def dcov_func(self,theta,x,mode=0):
        if mode == 0:
            return 2*theta[0]*np.exp(-x**2/(2*theta[1]**2))
        elif mode == 1:
            return x**2*theta[0]**2*np.exp(-x**2/(2*theta[1]**2))/theta[1]**3
        elif mode == 2:
            return 2*theta[2]*np.eye(x.shape[0])

    def meanf(self,theta,XX,XXp,y):
        """
        Posterior mean function
        """
        Kp = self.cov_func(theta,self.XXp,mode="nn")
        Ky = self.cov_func(theta,self.XX)
        L = np.linalg.cholesky(Ky)
        Linv = soltri(L.T,np.eye(y.size)).T
        LinvKp = np.dot(Linv,Kp)
        return np.dot(LinvKp.T,np.dot(Linv,y))

    def covf(self,theta,XX,XXp,XXpp):
        """
        Posterior covariance matrix
        """
        Kp = self.cov_func(theta,self.XXp,mode="nn")
        Kpp = self.cov_func(theta,self.XXpp,mode="nn")
        Ky = self.cov_func(theta,self.XX)
        L = np.linalg.cholesky(Ky)
        Linv = np.linalg.inv(L)
        LinvKp = np.dot(Linv,Kp)
        return Kpp - np.dot(LinvKp.T,LinvKp)

    def trainGP(self,y,theta0):
        # Set bounds for hypers (they must be strictly positive)
        bnds = ((1e-5, None), (1e-5, None), (1e-5, None))

        # Do optimisation
        thetap = opt.fmin_l_bfgs_b(self.logp_and_gradlogp, theta0, fprime=None, args=(y, n), bounds=bnds)

        #Check for convergence
        if thetap[2]["warnflag"]:
            print "Warning flag raised"
        # Return optimised value of theta
        return thetap[0]

    def EvalGP(self,theta0,y):
        theta = self.trainGP(theta0,y)

        return self.meanf(theta,self.XX,self.XXp,y), theta
