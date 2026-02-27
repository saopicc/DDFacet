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
from scipy.signal import fftconvolve

from numpy.fft import rfftn
from numpy.fft import irfftn
def dft(var):
    return rfftn(var, norm="ortho")
def idft(var, shape=None):
    return irfftn(var, s=shape, norm="ortho")

import astropy.io.fits as pyfits
from tqdm import tqdm
import scipy.stats
import pylab
from DDFacet.ToolsDir import GeneDist

def doFFT(A,dx=1.):
    # 2D FFT
    Nx,Ny=A.shape
    F = np.fft.fft2(A)/ (Nx * Ny)
    dy=dx=dx/3600*np.pi/180 # not imporant
    # frequency axes (cycles per unit of x and y)
    fx = np.fft.fftfreq(Nx, d=dx)
    fy = np.fft.fftfreq(Ny, d=dy)
    
    # center zero frequency
    fAs = np.fft.fftshift(F)
    fx_shifted = np.fft.fftshift(fx)
    fy_shifted = np.fft.fftshift(fy)

    return fAs,fx_shifted,fy_shifted

def test2():
    S=np.load("PNG2/AB_Major1_14342_12358.npz",allow_pickle=1)

    ich=1
    _,A,B,model,indx,indy,Bapp=S["LAB"][ich]
    # A=B.copy()
    
    # Asave=S["Asave"]#[0]
    # Bsave=S["Bsave"]#[0]
    # Asave=Bsave.copy()

    Model=S["Model"]
    nch=S["nch"][()]
    FreqBandToTaylor_ParmVec=S["FreqBandToTaylor_ParmVec"]
    FreqBandToTaylor_FluxVec=S["FreqBandToTaylor_FluxVec"]
    #ModelMachine=S["ModelMachine"][()]
    #ModelFit=S["ModelFit"]
    s_dirty_cut=S["s_dirty_cut"][()]
    s_psf_cut=S["s_psf_cut"][()]
    ARMS=S["ARMS"]
    RMS=ARMS[ich]
    SNR=A.max()/RMS

    
    dx=dy=1.5/3600*np.pi/180
    signal = A


    # 2D frequency grids
    # magnitude (optional)
    #magnitude = np.abs(F_shifted)

    
    # nx,ny=Model.shape[-2:]
    # MM=Model.copy().reshape((nch,nx*ny)).T
    # meanMM=MM.mean(axis=-1).reshape((-1,1))
    # indZero=np.where(meanMM==0)[0]
    # meanMM[indZero]=1
    # MM=MM/meanMM
        
    # MM=MM.reshape((1,nx*ny,nch))
    # FF=FreqBandToTaylor_FluxVec
    # NComb,NParm=FreqBandToTaylor_ParmVec.shape
    # FF=FF.reshape((NComb,1,nch))
    # indComb=np.argmin(np.sum((MM-FF)**2,axis=-1),axis=0)
        
        
    # CoefImage2=FreqBandToTaylor_ParmVec[indComb].copy()
    # CoefImage2[:,0]=meanMM.ravel()
    # CoefImage2[indZero,:]=0
    # CoefImage2=CoefImage2.T.reshape((NParm,1,nx,ny))
    
    # CoefImage=CoefImage2

    #CoefImage=S["CoefImage"]

    
    #Model=CoefImage#-Model

    #A=A+A.T
    #A=B+B.T
    
    LScaleModel=[]
    Model.fill(0)
    CO=ClassOrieux(A.copy(),B.copy(),
                   RMS=RMS)
    C=CO.Deconv(hyper="auto",
                sq=0.1,
                c=10,
                niter=20,
                Mode="HuberPos")
    
    Model[ich,0,s_dirty_cut,s_dirty_cut]=C[:,:]
    #print(Model.max())


    nx,ny=3,4
    pylab.figure("test",figsize=(20,10))
    iPlot=1
    pylab.clf()
    LScaleDirty=[]
    
            
    STD = scipy.stats.median_abs_deviation(A[A!=0],axis=None,scale="normal")
    v0,v1=-STD,A.max()#50*STD
    ax=pylab.subplot(nx, ny, iPlot); iPlot+=1
    LScaleDirty.append((v0,v1))
    pylab.imshow(A,vmin=v0,vmax=v1)
    pylab.title("Dirty [ch#%i]"%ich)
    
    if CO.fAs is not None:
        ax2=pylab.subplot(nx,ny, iPlot); iPlot+=1
        pylab.imshow(np.abs(CO.fAs))#,extent=(fx_shifted.min(),fx_shifted.max(),fy_shifted.min(),fy_shifted.max()))
        pylab.title("fAs [ch#%i]"%ich)
        ax2=pylab.subplot(nx,ny, iPlot,sharex=ax2,sharey=ax2); iPlot+=1
        pylab.imshow(np.abs(CO.fAsc))#,extent=(fx_shifted.min(),fx_shifted.max(),fy_shifted.min(),fy_shifted.max()))
        pylab.title("fAsc [ch#%i]"%ich)
    
        pylab.subplot(nx,ny, iPlot,sharex=ax2,sharey=ax2); iPlot+=1
        pylab.imshow(np.abs(CO.fBs))#,extent=(fx_shifted.min(),fx_shifted.max(),fy_shifted.min(),fy_shifted.max()))
        pylab.title("fBs [ch#%i]"%ich)
    
    
    
    ax=pylab.subplot(nx, ny, iPlot); iPlot+=1
    pylab.imshow(Model[ich][0])#,vmin=v0,vmax=v1)
    pylab.title("Model [ch#%i]"%ich)

    

    ModelConv=fftconvolve(Model[ich][0][s_dirty_cut,s_dirty_cut],B, mode='same')
    Resid=A-ModelConv
    pylab.subplot(nx, ny, iPlot,sharex=ax,sharey=ax); iPlot+=1
    pylab.imshow(Resid,vmin=v0,vmax=v1)
    pylab.colorbar()
    #pylab.imshow(Asave[ich,0],vmin=v0,vmax=v1)
    pylab.title("D-P*Model [ch#%i]"%ich)
    
    DM=GeneDist.ClassDistMachine()
    DM.setRefSample(Resid,Ns=1000,
                    #xmm=(model_rms.min(),model_rms.max()),
                    )
    x_I,y_I=DM.xyCumulD
    pylab.subplot(nx, ny, iPlot); iPlot+=1
    pylab.plot(x_I,y_I)#,vmin=v0,vmax=v1)
    pylab.title("hist resid")
                    
    pylab.tight_layout()
    pylab.draw()
    pylab.show(block=False)
    pylab.pause(0.1)
    
    







    
class ClassOrieux():
    def __init__(self,dirty,psf,RMS=None):
        self.RMS=RMS
        self.psf=psf
        self.dirty=dirty
        self.fAs=None
        self.fBs=None
        #%% Load
        self.reg_lapl = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 8
        self.reg_laplf = udft.ir2fr(self.reg_lapl, dirty.shape)
    
        self.reg_r = udft.ir2fr(np.array([[0], [1], [-1]]) / 2, dirty.shape)
        self.reg_c = udft.ir2fr(np.array([[0, 1, -1]]) / 2, dirty.shape)

        self.reg_indep = np.ones((1, 1))
        self.reg_indepf = udft.ir2fr(self.reg_indep, dirty.shape)
        
        self.fr = udft.ir2fr(psf, dirty.shape)
        # fr = udft.ir2fr(idft(np.abs(dft(psf))**2), dirty.shape)

    def giveHyper(self):
        A=self.dirty
        B=self.psf
        fAs,fx_shifted, fy_shifted=doFFT(A)
        nx,ny=fAs.shape
        fAs[nx//2,:]=0
        fAs[:,ny//2]=0
        fBs,_,_=doFFT(B/B.max())
        M0=(np.abs(fBs)<1e-2*np.abs(fBs).max())
        fBs[M0]=1
        fAs/=fBs
        
        fAs[M0]=0
        FX, FY = np.meshgrid(fx_shifted, fy_shifted)
        
        fcutx=FX.max()/4
        fcuty=FY.max()/4
        Cx=(FX>-fcutx)&(FX<fcutx)
        Cy=(FY>-fcuty)&(FY<fcuty)
        indx,indy=np.where(Cx & Cy)
        fAsc=fAs.copy()
        fAsc[indx,indy]=0
        Max=np.abs(fAsc).max()
        self.fAsc=fAsc
        RMS=self.RMS
        if True:#RMS is None:
            RMS=scipy.stats.median_abs_deviation(fAsc[fAsc!=0],axis=None,scale="normal")
        
        ThisSNR=Max/RMS
        Flux=np.median(np.abs(fAsc[fAsc!=0]))
        ThisSNR=Flux/RMS
        
        self.fAs=fAs
        self.fBs=fBs
        
        SNR0=1
        SNR1=20
        hyper0=1000
        hyper1=5
        a=(hyper0-hyper1)/(SNR0-SNR1)
        b=hyper0-a*SNR0
        hyp=lambda SNR: a*SNR+b
        hyper=hyp(ThisSNR)
        hyper=np.max([hyper1,hyper])
        hyper=np.min([hyper0,hyper])
        print(ThisSNR,Flux,hyper)
        return hyper
        
    def Deconv(self,hyper="auto",
               sq=0.001,
               c=10,
               niter=20,
               Mode="HuberRcPos"):
        #%% Run pos

        if hyper=="auto":
            hyper=self.giveHyper()
        
        if Mode=="HuberRcPos":
            deconv, aux_r, aux_c = self.deconv_huber_rc_pos(self.dirty,
                                                            self.fr,
                                                            hyper=hyper,
                                                            sq=sq,
                                                            c=c,
                                                            n_iter=niter)
        elif Mode=="Huber":
            deconv,_ = self.deconv_huber(self.dirty, self.fr, hyper, sq, niter)
        elif Mode=="HuberPos":
            deconv,_ = self.deconv_huber_pos(self.dirty, self.fr, hyper, sq, c=c, n_iter=niter)
            
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
        
        hub = Huber(sq)
        for it in range(n_iter):#tqdm(range(n_iter)):
            im = self.wiener(dirtyf, aux, fr, hyper)
            aux = hub.min_gy(im)

        return im, aux


    def deconv_huber_pos(self,data, fr, hyper, sq, c=1, n_iter=50):
        """Circulante deconvolution with independant huber penalization and
        positivity on the pixel, optimizerd with a GY+ADMM algorithm
        """
        aux = np.zeros_like(data)
        slack = np.zeros_like(data)
        lagrangians = np.zeros_like(data)
        dirtyf = dft(data)
        hub = Huber(sq)
        for it in range(n_iter):#tqdm(range(n_iter)):
            im = self.wiener(dirtyf, aux, fr, hyper, slack, lagrangians, c)
            aux = hub.min_gy(im)
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
            ims = self.biwiener(dirtyf, aux, fr, reg_laplf,
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
# plt.subplot(2, ny, ny)
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
