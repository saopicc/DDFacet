#! /usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt

from .edwin import udft
from .edwin import optim, operators, criterions
from .edwin.criterions import Huber
# from edwin import udft
# from edwin import optim, operators, criterions
# from edwin.criterions import Huber
# #from edwin.udft import urdft2 as dft
# #from edwin.udft import uirdft2 as idft

from numpy.fft import rfftn
from numpy.fft import irfftn
def dft(var):
    return rfftn(var, norm="ortho")
def idft(var, shape=None):
    return irfftn(var, s=shape, norm="ortho")

import astropy.io.fits as pyfits
from tqdm import tqdm





def test():
    dirty = np.squeeze(pyfits.open("../test.dirty.fits")[0].data)
    psf = np.squeeze(pyfits.open("../test.psf.fits")[0].data)

    
    CO=ClassOrieux(dirty,psf)
    
    #%% Plot
    gauss = (slice(450, 550), slice(450, 550))
    star = (slice(150, 250), slice(700, 800))
    deconv=CO.Deconv()
    
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.imshow(dirty)
    plt.subplot(2, 2, 2)
    plt.imshow(deconv)
    # plt.subplot(2, 2, 1)
    # plt.imshow(dirty[gauss])
    # plt.subplot(2, 2, 2)
    # plt.imshow(deconv[gauss])
    # plt.subplot(2, 2, 3)
    # plt.imshow(dirty[star])
    # plt.subplot(2, 2, 4)
    # plt.imshow(deconv[star])
    plt.draw()
    plt.show()
    
class ClassOrieux():
    def __init__(self,dirty,psf):
        

        self.dirty=dirty
        #%% Load
        self.reg_lapl = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 8
        self.reg_laplf = udft.ir2fr(self.reg_lapl, dirty.shape)
    
        self.reg_r = udft.ir2fr(np.array([[0], [1], [-1]]) / 2, dirty.shape)
        self.reg_c = udft.ir2fr(np.array([[0, 1, -1]]) / 2, dirty.shape)

        self.reg_indep = np.ones((1, 1))
        self.reg_indepf = udft.ir2fr(self.reg_indep, dirty.shape)
        
        self.fr = udft.ir2fr(psf, dirty.shape)
        # fr = udft.ir2fr(idft(np.abs(dft(psf))**2), dirty.shape)
        
    def Deconv(self,niter=20):
        #%% Run pos
        deconv, aux_r, aux_c = self.deconv_huber_rc_pos(self.dirty,
                                                        self.fr,
                                                        hyper=5.0,
                                                        sq=0.001,
                                                        c=10,
                                                        n_iter=niter)
        return deconv

    #%% CvxDiff deconv
    def wiener(self,dataf, aux, fr, hyper, slack=0, lagrangians=0, c=0):
        """Wiener deconvolution with Identity as regularization.  Can be used
        for positivity with ADMM
        """
        return idft(
            (np.conj(fr) * dataf + dft(hyper * aux + lagrangians + c * slack)) /
            (np.abs(fr)**2 + hyper + c),
            aux.shape)


    def deconv_huber(self,data, fr, hyper, sq, n_iter=50):
        """Circulante deconvolution with independant huber penalization on the
        pixel, optimizerd with a GY algorithm
        """
        aux = np.zeros_like(data)
        dirtyf = dft(data)
        for it in range(n_iter):#tqdm(range(n_iter)):
            im = wiener(dirtyf, aux, fr, hyper)
            aux = sq.min_gy(im)

        return im, aux


    def deconv_huber_pos(self,data, fr, hyper, sq, c=1, n_iter=50):
        """Circulante deconvolution with independant huber penalization and
        positivity on the pixel, optimizerd with a GY+ADMM algorithm
        """
        aux = np.zeros_like(data)
        slack = np.zeros_like(data)
        lagrangians = np.zeros_like(data)
        dirtyf = dft(data)
        for it in range(n_iter):#tqdm(range(n_iter)):
            im = wiener(dirtyf, aux, fr, hyper, slack, lagrangians, c)
            aux = sq.min_gy(im)
            slack = np.fmax(0, (c * im - lagrangians) / c)
            lagrangians = np.fmax(0, lagrangians - c * im)
            c = 0.9 * c

        return im, aux


    def wiener_rc(self,dataf, aux_r, aux_c, fr, reg_r, reg_c, hyper, slack=0,
                  lagrangians=0, c=0):
        """Wiener deconvolution with regularization in link and column"""
        return ((np.conj(fr) * dataf +
                 hyper * np.conj(reg_r) * dft(aux_r) +
                 hyper * np.conj(reg_c) * dft(aux_c) +
                 dft(lagrangians + c * slack)) /
                (np.abs(fr)**2 +
                 hyper * (np.abs(reg_r)**2 + np.abs(reg_c)**2) +
                 c))


    def deconv_huber_rc_pos(self,data, fr, hyper, sq, c=1, n_iter=50):
        """Circulante deconvolution with independant huber penalization and
        positivity on the pixel, optimizerd with a GY+ADMM algorithm
        """
        aux_r = np.zeros_like(data)
        aux_c = np.zeros_like(data)
        slacks = np.zeros_like(data)
        lagrangians = np.zeros_like(data)
        dirtyf = dft(data)
        hub = Huber(sq)
        reg_r=self.reg_r
        reg_c=self.reg_c
        for it in range(n_iter):#tqdm(range(n_iter)):
            imf = self.wiener_rc(dirtyf, aux_r, aux_c, fr, reg_r, reg_c, hyper,
                                 slacks, lagrangians, c)
            aux_r = hub.min_gy(idft(reg_r * imf))
            aux_c = hub.min_gy(idft(reg_c * imf))
            im = idft(imf, self.dirty.shape)
            slacks = np.fmax(0, (c * im - lagrangians) / c)
            lagrangians = np.fmax(0, lagrangians - c * im)
            
        return im, aux_r, aux_c



    #%% Bimodel w. Fourier
    def biwiener(self,dataf, aux, fr, reg, hyper_d, hyper_p, slacks=0, lagrangians=0,
                 c=0):
        """Wiener deconvolution with Identity as regularization.  Can be used
        for positivity with ADMM
        """
        # Lemme of block inversion matrix with matrix [[m11, m12], [m21, m22]]
        m11 = np.abs(fr)**2 + hyper_d * np.abs(reg)**2 + c
        m12 = np.abs(fr)**2
        m21 = m12
        m22 = np.abs(fr)**2 + hyper_p + c
        inv_11 = 1 / (m11 - m12 * m21 / m22)
        inv_12 = - inv_11 * m12 / m22
        inv_21 = - m21 / (m11 - m12 * m21 / m22) / m22
        inv_22 = (1 - inv_21 * m12) / m22
        
        transp_d = np.conj(fr) * dataf + dft(lagrangians[0] + c * slacks[0])
        transp_p = np.conj(fr) * dataf + dft(aux + lagrangians[1] + c * slacks[1])
        
        return idft(np.concatenate(((inv_11 * transp_d + inv_12 * transp_p)[np.newaxis],
                                    (inv_21 * transp_d + inv_22 * transp_p)[np.newaxis]), axis=0), slacks.shape[1:])


    def bideconv_huber_pos(self,data, fr, hyper_d, hyper_p, sq=0.001, c=1, n_iter=50):
        """Circulante deconvolution with independant huber penalization and
        positivity on the pixel, optimizerd with a GY+ADMM algorithm
        """
        slacks = np.zeros((2,) + data.shape)
        lagrangians = np.zeros_like(slacks)
        aux = np.zeros_like(data)
        dirtyf = dft(data)
        hub = Huber(sq)
        for it in range(n_iter):#tqdm(range(n_iter)):
            ims = biwiener(dirtyf, aux, fr, reg_laplf,
                           hyper_d, hyper_p,
                           slacks, lagrangians, c)
            aux = hub.min_gy(ims[1])
            slacks = np.fmax(0, (c * ims - lagrangians) / c)
            lagrangians = np.fmax(0, lagrangians - c * ims)
            # c = 0.9 * c

        return ims, aux, slacks, lagrangians

        

# #%% Run bipos
# deconv, aux, slacks, lagrangians = bideconv_huber_pos(
#     dirty, fr, hyper_d=10, hyper_p=0.1, sq=0.001, c=10, n_iter=20)

# #%% Plot
# rec = np.sum(deconv, axis=0)
# gauss = (slice(450, 550), slice(450, 550))
# star = (slice(150, 250), slice(700, 800))

# repro = idft(fr * dft(rec), dirty.shape)
# residual = dirty - idft(fr * dft(rec), dirty.shape)

# plt.figure(1)
# plt.clf()
# plt.subplot(2, 3, 1)
# plt.imshow(dirty[gauss])
# plt.title('Dirty')
# plt.subplot(2, 3, 2)
# plt.imshow(deconv[0][gauss])
# plt.title('D')
# plt.subplot(2, 3, 3)
# plt.imshow(deconv[1][gauss])
# plt.title('P')
# plt.subplot(2, 3, 4)
# plt.imshow(dirty[star])
# plt.subplot(2, 3, 5)
# plt.imshow(deconv[0][star])
# plt.subplot(2, 3, 6)
# plt.imshow(deconv[1][star])


# #%% Bimodel if Fourier is not possible
# sys.exit(0)


# #%% Bimodel if Fourier is not possible
# class ForwardModel(operators.LinearOperator):
#     def __init__(self, shape, fr):
#         """shape: data shape in image space, ie (Nr, Nc) with the number in
# row and column"""
#         super().__init__(
#             in_shape=(2,) + shape, out_shape=shape, name='BiModel')
#         self.fr = fr.reshape((1,) + fr.shape)

#     def forward(self, obj):
#         return idft(np.sum(self.fr * dft(obj), axis=0), self.out_shape)

#     def reverse(self, residual):
#         return np.tile(idft(np.conj(self.fr) * dft(residual), self.out_shape),
#                        (2,) + residual.ndim * (1, ))

#     def fwrev(self, obj):
#         return np.tile(idft(np.sum(np.abs(self.fr)**2 * dft(obj), axis=0),
#                             self.out_shape),
#                        (2,) + (obj.ndim - 1) * (1, ))


# class SmoothReg(operators.LinearOperator):
#     def __init__(self, shape):
#         """shape: data shape in image space, ie (Nr, Nc) with the number in
# row and column"""
#         super().__init__((2,) + shape, (2,) + shape, name='Smooth reg')
#         self.shape_im = shape
#         self.reg = operators.CircularConvolution(
#             np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 8,
#             self.shape_im)

#     def _concat(self, im1, im2):
#         return np.concatenate((im1[np.newaxis], im2[np.newaxis]), axis=0)

#     def forward(self, obj):
#         return self._concat(self.reg(obj[0]), np.zeros(self.shape_im))

#     def reverse(self, obj):
#         return self._concat(self.reg.t(obj[0]), np.zeros(self.shape_im))


# class IdentityReg(operators.LinearOperator):
#     def __init__(self, shape):
#         super().__init__(shape, shape)
#         self.shape_im = shape

#     def _concat(self, im1, im2):
#         return np.concatenate((im1[np.newaxis], im2[np.newaxis]), axis=0)

#     def forward(self, obj):
#         return self._concat(np.zeros_like(obj[0]), obj[1])

#     def reverse(self, obj):
#         return self._concat(np.zeros_like(obj[0]), obj[1])


# def deconv_bimodel_huber_pos(data, instr, hyper, sq=0.001, mu=1, n_iter=50):
#     """Circulante deconvolution with independant huber penalization and
# positivity on the pixel, optimizerd with a GY+ADMM algorithm
#     """
#     nx, ny = data.shape
#     sreg = SmoothReg(data.shape)
#     ireg = IdentityReg(data.shape)

#     slacks = np.zeros(instr.in_shape)
#     multipliers = np.zeros(instr.in_shape)

#     opt = optim.ConjGrad(max_iter=100, min_iter=20, restart=50,
#                          threshold=1e-6, speedrun=True, feedbacks=None)

#     aux = np.zeros(instr.in_shape)

#     hub = Huber(sq)

#     for iteration in tqdm(range(n_iter)):
#         crit = (criterions.LinearMeanSquare(instr, data, precision=1) +
#                 criterions.LinearMeanSquare(sreg, precision=hyper) +
#                 criterions.LinearMeanSquare(ireg, aux, precision=hyper) +
#                 criterions.AugumentedLagragian(
#                     operators.Identity(instr.in_shape),
#                     slacks,
#                     multipliers,
#                     mu=mu))

#         obj, info = opt.run(crit)

#         slacks = np.fmax(0, (mu * obj - multipliers) / mu)
#         multipliers = np.fmax(0, multipliers - mu * obj)

#         aux = hub.min_gy(obj)

#     return obj, aux, slacks, multipliers


# #%% Test bimodel
# instr = ForwardModel(dirty.shape, fr)
# sreg = SmoothReg(dirty.shape)
# ireg = IdentityReg(dirty.shape)

# obj, aux, slacks, multipliers = deconv_bimodel_huber_pos(
#     dirty, instr, 0.1, sq=0.001, mu=1, n_iter=10)
