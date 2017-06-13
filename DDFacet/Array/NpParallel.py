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

import DDFacet.cbuild.Gridder._pyArrays as _pyArrays
from DDFacet.Other import ClassTimeIt
import numpy as np
import psutil

NCPU_global = 0 #psutil.cpu_count()

def A_add_B_prod_factor(
    A,
    B,
    Aedge=None,
    Bedge=None,
    factor=1.,
     NCPU=0):

    NCPU = NCPU or NCPU_global

    NDimsA = len(A.shape)
    NDimsB = len(B.shape)
    if NDimsB != 2:
        stop
    ShapeOrig = A.shape

    if Aedge is None:
        Aedge = np.array([0, A.shape[-2], 0, A.shape[-1]])
    if Bedge is None:
        Bedge = np.array([0, B.shape[-2], 0, B.shape[-1]])

    Aedge = np.int32(Aedge)
    Bedge = np.int32(Bedge)

    factor = float(factor)

    Blocks = np.int32(np.linspace(Aedge[0], Aedge[1], NCPU+1))

    NX, NY = A.shape[-2], A.shape[-1]
    nz = A.size/(NX*NY)
    A = A.reshape((nz, NX, NY))

    for iz in xrange(nz):
        ThisA = A[iz]
        _pyArrays.pyAddArray(ThisA, Aedge, B, Bedge, float(factor), Blocks)

    A = A.reshape(ShapeOrig)
    return A

def A_whereMax(A,NCPU=0,DoAbs=1,Mask=None):
    NCPU = NCPU or NCPU_global
    if NCPU==1:
        NX,NY=A.shape[-2],A.shape[-1]
        nz=A.size/(NX*NY)
        A=A.reshape((nz,NX,NY))
        if Mask is not None:
            Mask=Mask.reshape((nz,NX,NY))

        if DoAbs:
            AbsA=np.abs(A)
            if Mask is not None:
                M=np.max(AbsA[Mask==0])
                _,x,y=np.where((AbsA == M)&(Mask==0))
                return x[0],y[0],M
            else:
                M=np.max(AbsA)
                _,x,y=np.where((AbsA == M))
                return x[0],y[0],M
        else:
            AbsA=A
            if Mask is not None:
                M=np.max(AbsA[Mask==0])
                _,x,y=np.where((AbsA == M)&(Mask==0))
                return x[0],y[0],M
            else:
                M=np.max(AbsA)
                _,x,y=np.where((AbsA == M))
                return x[0],y[0],M
                
    else:
        return A_whereMaxParallel(A,NCPU=NCPU,DoAbs=DoAbs,Mask=Mask)


def A_whereMaxParallel(A,NCPU=0,DoAbs=1,Mask=None):
    NCPU = NCPU or NCPU_global

    NDimsA=len(A.shape)
    ShapeOrig=A.shape
    
    NX,NY=A.shape[-2],A.shape[-1]
    Blocks = np.int32(np.linspace(0,NX,NCPU+1))

    if not (A.flags.c_contiguous):
        raise TypeError("Expected contiguous array as input")
    if A.dtype!=np.float32:
        raise TypeError("Expected input array dtype: float32")
    
    nz=A.size/(NX*NY)
    A=A.reshape((nz,NX,NY))

    Ans=np.zeros((nz,3),np.float32)
    
    for iz in range(nz):
        ThisA=A[iz]
        if Mask is None:
            _pyArrays.pyWhereMax(ThisA,Blocks,Ans[iz],DoAbs)
        else:
            _pyArrays.pyWhereMaxMask(ThisA,Mask,Blocks,Ans[iz],DoAbs)
            

    chMaxAns=np.argmax(Ans[:,2])
    Ans=Ans[chMaxAns]
    i,j,V=Ans
    i=int(i)
    j=int(j)
    
        

    A=A.reshape(ShapeOrig)
    return i,j,V

# def A_whereMax(A, NCPU=0, DoAbs=1, Mask=None):
#     """
#     Args:
#         A:
#         NCPU: if 0, use all CPUs
#         DoAbs:
#         Mask:

#     Returns:

#     """
#     NCPU = NCPU or NCPU_global
#     NDimsA = len(A.shape)
#     ShapeOrig = A.shape

#     NX, NY = A.shape[-2], A.shape[-1]
#     Blocks = np.int32(np.linspace(0, NX, NCPU+1))

#     if not (A.flags.c_contiguous):
#         raise TypeError("Expected contiguous array as input")
#     if A.dtype != np.float32:
#         raise TypeError("Expected input array dtype: float32")

#     nz = A.size/(NX*NY)
#     A = A.reshape((nz, NX, NY))

#     Ans = np.zeros((nz, 3), np.float32)

#     for iz in xrange(nz):
#         ThisA = A[iz]

#         if Mask is None:
#             _pyArrays.pyWhereMax(ThisA, Blocks, Ans[iz], DoAbs)
#         else:
#             _pyArrays.pyWhereMaxMask(ThisA, Mask, Blocks, Ans[iz], DoAbs)

#     chMaxAns = np.argmax(Ans[:, 2])
#     Ans = Ans[chMaxAns]
#     i, j, V = Ans
#     i = int(i)
#     j = int(j)

#     A = A.reshape(ShapeOrig)
#     return i, j, V


def testWhereMax():
    
    Npx=1030
    Npy=500
    nch=1
    np.random.seed(8)
    A=np.zeros((nch,Npx,Npy),dtype=np.float32)
    A=np.float32(np.random.randn(*(nch,Npx,Npy)))
    
    Mask=np.zeros((Npx,Npy),np.bool8)
    #A[0,100,200]=30.

    T= ClassTimeIt.ClassTimeIt()
    #Ans=A_whereMax(A,NCPU=1,Mask=Mask)
    #print Ans
    T.timeit("1")
    f=np.abs
    #f=lambda x: x
    ind= np.where(f(A)==np.max(f(A)))
    T.timeit("2")

    print ind,A[ind]
    print "==========================="
    print A_whereMax(A,NCPU=1,DoAbs=0,Mask=None)
    print A_whereMax(A,NCPU=1,DoAbs=0,Mask=np.zeros(A.shape,np.bool8))
    print "==========================="
    print A_whereMax(A,NCPU=6,DoAbs=0,Mask=None)
    print A_whereMax(A,NCPU=6,DoAbs=0,Mask=np.zeros(A.shape,np.bool8))
    print "==========================="
    print A_whereMax(A,NCPU=1,DoAbs=1,Mask=None)
    print A_whereMax(A,NCPU=1,DoAbs=1,Mask=np.zeros(A.shape,np.bool8))
    print "==========================="
    print A_whereMax(A,NCPU=6,DoAbs=1,Mask=None)
    print A_whereMax(A,NCPU=6,DoAbs=1,Mask=np.zeros(A.shape,np.bool8))


# testWhereMax()

def testAdd():

    Np = 16000

    x0 = 1000
    x1 = 11000
    y0 = 5000
    y1 = 15000
    nch = 1

    _a0 = np.ones((nch, Np, Np), dtype=np.float32)
    _a1 = np.ones((nch, Np, Np), dtype=np.float32)
    # b=np.float32(np.arange(Np**2).reshape((Np,Np)))
    b = np.ones((Np, Np), dtype=np.float32)

    Aedge = np.array([x0, x1, y0, y1], np.int32)
    Bedge = np.array([x0+1, x1+1, y0, y1], np.int32)

    T = ClassTimeIt.ClassTimeIt()

    N = 10
    NBlocks = 6
    factor = -1.

    for i in xrange(N):
        # b=np.float32(np.random.randn(Np,Np))
        A_add_B_prod_factor(
            _a0,
            b,
            Aedge,
            Bedge,
            factor=float(factor),
            NCPU=NBlocks)
        # A_add_B_prod_factor(_a0,b,Aedge,Bedge,factor=float(factor),NCPU=NBlocks)
        # for ch in range(nch):
        #     a0=_a0[ch]
        #     a1=_a1[ch]
        #     a_x0,a_x1,a_y0,a_y1=Aedge
        #     b_x0,b_x1,b_y0,b_y1=Bedge
        #     a1[a_x0:a_x1,a_y0:a_y1]+=b[b_x0:b_x1,b_y0:b_y1]*factor
        # # print a1[a_x0:a_x1,a_y0:a_y1].shape
        # # print b[b_x0:b_x1,b_y0:b_y1].shape
        # # print a_x0,a_x1,a_y0,a_y1
        # # print b_x0,b_x1,b_y0,b_y1
        # print "done"
        # print np.max(_a0-_a1)
        # # print

    # print _a0
    # print _a1

    T.timeit("1")
    for i in xrange(N):
        for ch in xrange(nch):
            a0 = _a0[ch]
            a1 = _a1[ch]
            a_x0, a_x1, a_y0, a_y1 = Aedge
            b_x0, b_x1, b_y0, b_y1 = Bedge
            a1[a_x0:a_x1, a_y0:a_y1] += b[b_x0:b_x1, b_y0:b_y1]*factor
            # a1=a0+b*factor
    T.timeit("2")


# test()
