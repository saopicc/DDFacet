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
import scipy as scp
from scipy.linalg import solve_triangular as soltri
from scipy import optimize as opt
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassGP")
class ClassGP(object):
    """
    Why the
    """
    def __init__(self, x, xp, covariance_function='sqexp', mean_func=None):
        self.x = x
        self.xp = xp
        self.N = self.x.size
        self.Np = self.xp.size
        self.set_covariance(covariance_function=covariance_function)
        # if mean_func is None:
        #     self.fm = np.zeros(self.N)
        #     self.fmp = np.zeros(self.Np)
        # else:
        #     try:
        #         self.fm = mean_func(self.x)
        #         self.fmp = mean_func(self.xp)
        #     except:
        #         print>>log, "Please pass a callable mean function to ClassGP"


    def set_abs_diff(self):
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
        """
        Computes the diagonal of C = AB where A and B are square matrices
        """
        D = np.zeros(A.shape[0])
        for i in xrange(A.shape[0]):
            D[i] = np.dot(A[i,:],B[:,i])
        return D

    def set_covariance(self,covariance_function='sqexp'):
        if covariance_function == "sqexp":
            self.cov_func = lambda theta, x, mode : self.cov_func_sqexp(theta, x, mode=mode)
            self.dcov_func = lambda theta, x, mode : self.dcov_func_sqexp(theta, x, mode= mode)
        elif covariance_function == 'mat52':
            self.cov_func = lambda theta, x, mode : self.cov_func_mat52(theta, x, mode=mode)
            self.dcov_func = lambda theta, x, mode : self.dcov_func_mat52(theta, x, mode= mode)
        elif covariance_function == 'mat72':
            self.cov_func = lambda theta, x, mode: self.cov_func_mat72(theta, x, mode=mode)
            self.dcov_func = lambda theta, x, mode: self.dcov_func_mat72(theta, x, mode=mode)

    def logp_and_gradlogp(self, theta, y):
        """
        Returns the negative log (marginal) likelihood (the function to be optimised) and its gradient
        """
        #tmp is Ky
        tmp = self.cov_func(theta, self.XX, mode="Noise")
        #tmp is L
        try:
            tmp = np.linalg.cholesky(tmp)
        except:
            logp = 1.0e8
            dlogp = np.ones(theta.size)*1.0e8
            return logp, dlogp
        detK = 2.0*np.sum(np.log(np.diag(tmp)))
        #tmp is Linv
        tmp = np.linalg.inv(tmp)
        #tmp2 is Linvy
        tmp2 = np.dot(tmp,y)
        logp = np.dot(tmp2.T,tmp2)/2.0 + detK/2.0 + self.N*np.log(2*np.pi)/2.0
        nhypers = theta.size
        dlogp = np.zeros(nhypers)
        #tmp is Kinv
        tmp = np.dot(tmp.T,tmp)
        #tmp2 becomes Kinvy
        tmp2 = np.reshape(np.dot(tmp,y),(self.N,1))
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
        return logp, dlogp

    def cov_func_mat52(self, theta, x, mode="Noise"):
        if mode != "Noise":
            return theta[0] ** 2 * np.exp(-np.sqrt(5) * np.abs(x) / theta[1]) * (1 + np.sqrt(5) * np.abs(x) / theta[1] + 5 * np.abs(x) ** 2 / (3 * theta[1] ** 2))
        else:
            return theta[0] ** 2 * np.exp(-np.sqrt(5) * np.abs(x) / theta[1]) * (
            1 + np.sqrt(5) * np.abs(x) / theta[1] + 5 * np.abs(x) ** 2 / (3 * theta[1] ** 2)) + theta[2]**2.0*np.eye(x.shape[0])

    def dcov_func_mat52(self, theta, x, mode=0):
        """
        Derivates of the covariance function w.r.t. the hyperparameters
        """
        if mode == 0:
            return 2*self.cov_func_mat52(theta, x, mode='nn')/theta[0]
        elif mode == 1:
            return np.sqrt(5)*np.abs(x)*self.cov_func_mat52(theta, x, mode='nn')/theta[1]**2 + theta[0] ** 2 * \
                        np.exp(-np.sqrt(5) * np.abs(x) / theta[1])*(-np.sqrt(5) * np.abs(x) / theta[1]**2 - 10 * np.abs(x) ** 2 / (3 * theta[1] ** 3))
        elif mode == 2:
            return 2*theta[2]*np.eye(x.shape[0])

    def cov_func_mat72(self, theta, x, mode="Noise"):
        if mode != "Noise":
            return theta[0]**2 * np.exp(-np.sqrt(7) * np.abs(x) / theta[1]) * (1 + np.sqrt(7) * np.abs(x) / theta[1] +
                                14 * np.abs(x)**2/(5 * theta[1]**2) + 7*np.sqrt(7)*np.abs(x)**3/(15*theta[1]**3))
        else:
            return theta[0]**2 * np.exp(-np.sqrt(7) * np.abs(x) / theta[1]) * (1 + np.sqrt(7) * np.abs(x) / theta[1]
                            + 14 * np.abs(x) ** 2 / (5 * theta[1] ** 2) + 7*np.sqrt(7)*np.abs(x)**3/(15*theta[1]**3)) +\
                            theta[2]**2.0*np.eye(x.shape[0])

    def dcov_func_mat72(self, theta, x, mode=0):
        """
        Derivates of the covariance function w.r.t. the hyperparameters
        """
        if mode == 0:
            return 2*self.cov_func_mat72(theta, x, mode='nn')/theta[0]
        elif mode == 1:
            return np.sqrt(7)*np.abs(x)*self.cov_func_mat72(theta, x, mode='nn')/theta[1]**2 + theta[0] ** 2 * \
                        np.exp(-np.sqrt(7) * np.abs(x) / theta[1])*(-np.sqrt(7) * np.abs(x) / theta[1]**2 - 28 *
                        np.abs(x) ** 2 / theta[1] ** 3 - 21*np.sqrt(7)*np.abs(x)**3 / theta[1]**4)
        elif mode == 2:
            return 2*theta[2]*np.eye(x.shape[0])

    def cov_func_sqexp(self, theta, x, mode="Noise"):
        """
        Covariance function including noise variance
        """
        if mode != "Noise":
            #Squared exponential
            return theta[0]**2.0*np.exp(-x**2.0/(2.0*theta[1]**2.0))
        else:
            #Squared exponential
            return theta[0]**2*np.exp(-x**2.0/(2.0*theta[1]**2.0)) + theta[2]**2.0*np.eye(x.shape[0])

    def dcov_func_sqexp(self, theta, x, mode=0):
        """
        Derivates of the covariance function w.r.t. the hyperparameters
        """
        if mode == 0:
            return 2*theta[0]*np.exp(-x**2/(2*theta[1]**2))
        elif mode == 1:
            return x**2*theta[0]**2*np.exp(-x**2/(2*theta[1]**2))/theta[1]**3
        elif mode == 2:
            return 2*theta[2]*np.eye(x.shape[0])

    def meanf(self, theta, y):
        """
        Posterior mean function
        """
        Kp = self.cov_func(theta,self.XXp, mode="nn")
        Ky = self.cov_func(theta,self.XX, mode="Noise")
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

    def trainGP(self, theta0, y):
        # Set bounds for hypers (they must be strictly positive)
        bnds = ((1e-5, None), (1e-5, None), (1e-3, None))

        # Do optimisation
        thetap = opt.fmin_l_bfgs_b(self.logp_and_gradlogp, theta0, fprime=None, args=(y,), bounds=bnds) #, factr=1e10, pgtol=0.1)

        #Check for convergence
        if thetap[2]["warnflag"]:
            print "Warning flag raised"
        # Return optimised value of theta
        return thetap[0]

    def EvalGP(self, theta0, y):
        theta = self.trainGP(theta0, y)

        return self.meanf(theta, y), theta
