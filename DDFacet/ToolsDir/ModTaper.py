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

def TaperGauss(A):

    x=np.arange(A.size)
    x0=A.size/2
    sigx=(0.5*A.size)/3
    g=np.exp(-(x-x0)**2/(2*sigx**2))
    return A*g


def Sphe2D(Npix,factor=1.):
    y=np.zeros((Npix,),float)
    x,y=np.mgrid[-1:1:1j*Npix,-1:1:1j*Npix]

    R=np.sqrt(x**2+y**2)
    z=np.zeros_like(R)
    zf=z.ravel()
    for (i,r) in zip(range(Npix**2),R.ravel()):
        zf[i]=EvalSphe(np.abs(r)*factor)
    return z

    #
    #
    # x=np.abs(np.mgrid[-1:1:1j*Npix])
    # zf1D=np.zeros((Npix,),np.float64)
    # for i in range(Npix):
    #     zf1D[i]=EvalSphe(np.abs(x[i])*factor)
    #
    # zf=zf1D.reshape((Npix,1))*zf1D.reshape((1,Npix))
    return zf

def Sphe1D(Npix,factor=1.):
    y=np.zeros((Npix,),float)
    if Npix%2==0:
        xx0=Npix/2
        dx=(Npix)/2.
        xx=(np.arange(Npix)-xx0)/dx
    else:
        dx=(Npix-1)/2.
        xx0=(Npix-1)/2
        xx=(np.arange(Npix)-xx0)/dx

    for (i,x) in zip(range(Npix),xx):
        y[i]=EvalSphe(np.abs(x)*factor)
    return y



def EvalSphe(nu):
    P = np.array([[ 8.203343e-2, -3.644705e-1, 6.278660e-1,-5.335581e-1,  2.312756e-1],\
                      [ 4.028559e-3, -3.697768e-2, 1.021332e-1,-1.201436e-1, 6.412774e-2]])
    Q=np.array([[1.0000000e0, 8.212018e-1, 2.078043e-1],\
                    [1.0000000e0, 9.599102e-1, 2.918724e-1]])

    part = 0
    end = 0.0
    if ((nu >= 0.0) & (nu < 0.75)):
        part = 0
        end = 0.75
    elif ((nu >= 0.75) & (nu <= 1.00)):
        part = 1
        end = 1.00
    else:
        return 0.0

    nusq = nu **2
    delnusq = nusq - end * end
    delnusqPow = delnusq
    top = P[part][0]
    for k in range(1,5):
        top += P[part][k] * delnusqPow
        delnusqPow *= delnusq


    bot = Q[part][0]
    delnusqPow = delnusq
    for k in range(1,3):
        bot += Q[part][k] * delnusqPow
        delnusqPow *= delnusq


    result = (1.0 - nusq) * (top / bot)

    return result


