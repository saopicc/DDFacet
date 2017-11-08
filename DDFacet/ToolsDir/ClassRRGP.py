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
"""
Created on Wed Jan 18 10:18:27 2017

@author: landman

A class to implement reduced rank GPR as outlined in arxiv-1401.5508

"""

import numpy as np
import scipy as scp
from scipy.linalg import solve_triangular as soltri
from scipy import optimize as opt
from DDFacet.ToolsDir import ClassGP
#import matplotlib.pyplot as plt
import time


class RR_GP(ClassGP.ClassGP):
    def __init__(self, x, xp, L, m, covariance_function='sqexp'):
        """
        A class to implement reduced rank GPR
        Args:
            x: input (locations where we have data)
            xp: targets (locations where we want to evaluate the function
            L: the boundary of the domain on which we evaluate the eigenfunctions of the Laplacian
            m: the number of basis function to use
            covariance_function: which covariance function to use (currently only squared exponential 'sqexp' supported)
        """
        # Initialise base class
        ClassGP.ClassGP.__init__(self, x, xp, covariance_function=covariance_function)
        self.L = L #args[2]
        self.m = m #args[3]

        # Calculate basis functions and eigenvals
        self.Eigvals = np.zeros(self.m)
        self.Phi = np.zeros([self.N, self.m])
        self.Phip_Degrid = np.zeros([self.Np, self.m])
        for j in xrange(m):
            self.Eigvals[j] = self.eigenvals(j+1)
            self.Phi[:, j] = self.eigenfuncs(j+1, x)
            self.Phip_Degrid[:, j] = self.eigenfuncs(j+1, xp)
        self.Phip = self.Phi
        # Compute dot(Phi.T,Phi) # This doesn't change, doesn't depend on theta
        self.PhiTPhi = np.dot(self.Phi.T, self.Phi)
        self.s = np.sqrt(self.Eigvals)

        # Set the covariance function on spectral density
        self.set_spectral_density(covariance_function=covariance_function)

        # Set bounds for hypers (they must be strictly positive)
        lmin = self.L/(2*self.m)
        self.bnds = ((1e-5, 1.0e2), (lmin, 0.95*self.L), (1.0e-8, None))

    def set_spectral_density(self, covariance_function='sqexp'):
        if covariance_function == "sqexp":
            self.spectral_density = lambda theta : self.spectral_density_sqexp(theta, self.s)
            self.dspectral_density = lambda theta, S, mode: self.dspectral_density_sqexp(theta, S, self.s, mode=mode)
        elif covariance_function == 'mat52':
            self.spectral_density = lambda theta : self.spectral_density_mat52(theta, self.s)
            self.dspectral_density = lambda theta, S, mode: self.dspectral_density_mat52(theta, S, self.s, mode=mode)
        elif covariance_function == 'mat72':
            self.spectral_density = lambda theta: self.spectral_density_mat72(theta, self.s)
            self.dspectral_density = lambda theta, S, mode: self.dspectral_density_mat72(theta, S, self.s, mode=mode)

    def RR_logp_and_gradlogp(self, theta, y, grid_mode="regular"):
        S = self.spectral_density(theta)
        Lambdainv = np.diag(1.0 / S)
        Z = self.PhiTPhi + theta[2] ** 2 * Lambdainv
        if grid_mode=="regular":
            Zdiag = 1/np.diag(Z)
            L = np.sqrt(Zdiag)
            logdetZ = 2.0 * np.sum(np.log(L))
            Zinv = np.diag(1/Zdiag)
        else:
            try:
                L = np.linalg.cholesky(Z)
            except:
                print "Had to add jitter, theta = ", theta
                L = np.linalg.cholesky(Z + 1.0e-6)
                #
                # logp = 1.0e2
                # dlogp = np.ones(theta.size)*1.0e8
                # return logp, dlogp
            Linv = np.linalg.inv(L)
            Zinv = np.dot(Linv.T,Linv)
            logdetZ = 2.0 * np.sum(np.log(np.diag(L)))
        # Get the log term
        logQ = (self.N - self.m) * np.log(theta[2] ** 2) + logdetZ + np.sum(np.log(S))
        # Get the quadratic term
        PhiTy = np.dot(self.Phi.T, y)
        ZinvPhiTy = np.dot(Zinv, PhiTy)
        yTy = np.dot(y.T, y)
        yTQinvy = (yTy - np.dot(PhiTy.T, ZinvPhiTy)) / theta[2] ** 2
        # Get their derivatives
        dlogQdtheta = np.zeros(theta.size)
        dyTQinvydtheta = np.zeros(theta.size)
        for i in xrange(theta.size-1):
            dSdtheta = self.dspectral_density(theta, S, mode=i)
            dlogQdtheta[i] = np.sum(dSdtheta/S) - theta[2]**2*np.sum(dSdtheta/S*np.diag(Zinv)/S)
            dyTQinvydtheta[i] = -np.dot(ZinvPhiTy.T, np.dot(np.diag(dSdtheta/S **2), ZinvPhiTy))
        # Get derivatives w.r.t. sigma_n
        dlogQdtheta[2] = 2 * theta[2] * ((self.N - self.m) / theta[2] ** 2 + np.sum(np.diag(Zinv) / S))
        dyTQinvydtheta[2] = 2 * (np.dot(ZinvPhiTy.T, np.dot(Lambdainv,ZinvPhiTy)) - yTQinvy)/theta[2]

        logp = (yTQinvy + logQ + self.N * np.log(2 * np.pi)) / 2.0
        dlogp = (dlogQdtheta + dyTQinvydtheta)/2
        return logp, dlogp

    def eigenvals(self, j):
        return (np.pi * j / (2.0 * self.L)) ** 2

    def eigenfuncs(self, j, x):
        return np.sin(np.pi * j * (x + self.L) / (2.0 * self.L)) / np.sqrt(self.L)

    # def cov_func52(self, x, theta):
    #     return theta[0] ** 2 * np.exp(-np.sqrt(5) * np.abs(x) / theta[1]) * (
    #     1 + np.sqrt(5) * np.abs(x) / theta[1] + 5 * np.abs(x) ** 2 / (3 * theta[1] ** 2))
    #
    def spectral_density_mat52(self, theta, s):  # This is for Mattern with v=5/2
        return theta[0] ** 2 * 2 * 149.07119849998597 / (theta[1] ** 5 * (5.0 / theta[1] ** 2 + s ** 2) ** 3)

    def dspectral_density_mat52(self, theta, S, s, mode=0):
        if mode == 0:
            return 2 * S / theta[0]
        elif mode == 1:
            return S * (-5 / theta[1] + 30 / (theta[1] ** 3 * (5.0 / theta[1] ** 2 + s ** 2)))

    def spectral_density_mat72(self, theta, s):  # This is for Mattern with v=5/2
        return theta[0] ** 2 * 2 * 5807.95327805 / (theta[1] ** 7 * (7.0 / theta[1] ** 2 + s ** 2) ** 3)

    def dspectral_density_mat72(self, theta, S, s, mode=0):
        if mode == 0:
            return 2 * S / theta[0]
        elif mode == 1:
            return S * (-7 / theta[1] + 56 / (theta[1] ** 3 * (7.0 / theta[1] ** 2 + s ** 2)))

    def spectral_density_sqexp(self, theta, s):
        return theta[0]**2.0*np.sqrt(2.0*np.pi*theta[1]**2)*np.exp(-theta[1]**2*s**2/2)

    def dspectral_density_sqexp(self, theta, S, s, mode=0):
        if mode == 0:
            return 2 * S / theta[0]
        elif mode == 1:
            return S/theta[1] - s**2*theta[1]*S


    def RR_meanf(self, theta, y):
        S = self.spectral_density(theta)
        Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / S)
        L = np.linalg.cholesky(Z)
        Linv = np.linalg.inv(L)
        fcoeffs = np.dot(Linv.T, np.dot(Linv, np.dot(self.Phi.T, y)))
        fbar = np.dot(self.Phip, fcoeffs)
        return fbar, fcoeffs

    def RR_Give_Coeffs(self,theta, y):
        S = self.spectral_density(theta)
        Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / S)
        try:
            L = np.linalg.cholesky(Z)
        except:
            print "Had to add jitter. Theta = ", theta
            L = np.linalg.cholesky(Z + 1.0e-6*np.eye(Z.shape[0]))
        Linv = np.linalg.inv(L)
        fcoeffs = np.dot(Linv.T, np.dot(Linv, np.dot(self.Phi.T, y)))
        return fcoeffs

    def RR_Reset_Targets(self, xp):
        self.xp = xp
        self.Phip = np.zeros([xp.size, self.m])
        for j in xrange(self.m):
            self.Phip[:, j] = self.eigenfuncs(j+1, xp)

    def RR_From_Coeffs(self, coeffs):
        return np.dot(self.Phip, coeffs)

    def RR_From_Coeffs_Degrid(self, coeffs):
        return np.dot(self.Phip_Degrid, coeffs)

    def RR_From_Coeffs_Degrid_ref(self, coeffs):
        Phip = np.zeros(self.m)
        for j in xrange(self.m):
            Phip[j] = self.eigenfuncs(j+1, 1.0)
        return np.sum(Phip*coeffs)

    def RR_covf(self, theta):
        S = self.dspectral_density(theta)
        Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / S)
        L = np.linalg.cholesky(Z)
        Linv = np.linalg.inv(L)
        fcovcoeffs = theta[2] ** 2 * np.dot(Linv.T, Linv)
        covf = theta[2] ** 2 * np.dot(self.Phip, np.dot(Linv.T, np.dot(Linv, self.Phip.T)))
        return covf, fcovcoeffs

    def RR_trainGP(self, theta0, y):
        # Do optimisation
        self.SolverFlag = 0
        thetap = opt.fmin_l_bfgs_b(self.RR_logp_and_gradlogp, theta0, fprime=None, args=(y,), bounds=self.bnds) #, factr=1e10, pgtol=0.1)

        if np.any(np.isnan(thetap[0])):
            raise Exception('Solver crashed error. Are you trying a noise free simulation? Use FreqMode = Poly instead.')

        #Check for convergence
        if thetap[2]["warnflag"]:
            self.SolverFlag = 1
            #print "Warning flag raised", thetap[2]
            #print thetap[0]
        # Return optimised value of theta
        return thetap[0]

    def RR_EvalGP(self, theta0, y):
        theta = self.RR_trainGP(theta0, y)
        coeffs = self.RR_Give_Coeffs(theta, y)
        return coeffs, theta

    def mean_and_cov(self, theta):
        Z = self.PhiTPhi + theta[2] ** 2 * np.diag(1.0 / self.S)
        L = np.linalg.cholesky(Z)
        Linv = np.linalg.inv(L)
        self.fcoeffs = np.dot(Linv.T, np.dot(Linv, np.dot(self.Phi.T, self.y)))
        fbar = np.dot(self.Phip, np.dot(Linv.T, np.dot(Linv, np.dot(self.Phi.T, self.y))))
        self.fcovcoeffs = theta[2] ** 2 * np.dot(Linv.T, Linv)
        covf = theta[2] ** 2 * np.dot(self.Phip, np.dot(Linv.T, np.dot(Linv, self.Phip.T)))
        return fbar, covf
