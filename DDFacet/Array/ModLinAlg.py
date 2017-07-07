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

import scipy.linalg
import numpy as np
from DDFacet.Other import ModColor

def invertChol(A):
    L=np.linalg.cholesky(A)
    Linv=np.linalg.inv(L)
    Ainv=np.dot(Linv.T,Linv)
    return Ainv

def invertLU(A):
    lu,piv=scipy.linalg.lu_factor(A)
    return scipy.linalg.lu_solve((lu,piv),np.eye(A.shape[0],A.shape[0]))

def sqrtSVD(A):
    #u,s,v=np.linalg.svd(A+np.random.randn(*A.shape)*(1e-6*A.max()))
    A=(A+A.T)/2.
    thr=1e-8
    u,s,v=np.linalg.svd(A+np.random.randn(*A.shape)*(thr*A.max()))
    s[s<0.]=0.
    ssq=np.diag(np.sqrt(s))
    Asq=np.dot(np.dot(u,ssq),v)
    return Asq

def BatchInverse(A,H=False):
    shapeOut=A.shape
    A=A.reshape((A.size/4,2,2))
    #A.shape=N,2,2
    N,dum,dum=A.shape
    Ainv=np.zeros_like(A)
    if not(H):
        a0=A[:,0,0]
        d0=A[:,1,1]
        b0=A[:,0,1]
        c0=A[:,1,0]
    else:
        a0=A[:,0,0].conj()
        d0=A[:,1,1].conj()
        b0=A[:,1,0].conj()
        c0=A[:,0,1].conj()
        
    det=1./(a0*d0-b0*c0)
    Ainv[:,0,0]=d0*det
    Ainv[:,0,1]=-b0*det
    Ainv[:,1,0]=-c0*det
    Ainv[:,1,1]=a0*det
    Ainv=Ainv.reshape(shapeOut)
    return Ainv
    
def BatchH(A):
    shapeOut=A.shape
    A=A.reshape((A.size/4,2,2))

    N,dum,dum=A.shape
    AH=np.zeros_like(A)

    a0=A[:,0,0].conj()
    d0=A[:,1,1].conj()
    b0=A[:,1,0].conj()
    c0=A[:,0,1].conj()
    AH[:,0,0]=a0
    AH[:,1,1]=d0
    AH[:,0,1]=b0
    AH[:,1,0]=c0

    AH=AH.reshape(shapeOut)
    return AH
    
def BatchDot(A,B):
    shapeOut=A.shape
    A=A.reshape((A.size/4,2,2))
    B=B.reshape((B.size/4,2,2))

    C=np.zeros_like(A)
    # if A.size>=B.size:
    #     C=np.zeros_like(A)
    #     shapeOut=A.shape
    # else:
    #     C=np.zeros_like(B)
    #     shapeOut=B.shape

    # print "A:",A.shape
    # print "B:",B.shape
    # print "C:",C.shape
    
    a0=A[:,0,0]
    b0=A[:,1,0]
    c0=A[:,0,1]
    d0=A[:,1,1]

    a1=B[:,0,0]
    b1=B[:,1,0]
    c1=B[:,0,1]
    d1=B[:,1,1]

    C00=C[:,0,0]
    C01=C[:,0,1]
    C10=C[:,1,0]
    C11=C[:,1,1]

    C00[:]=a0*a1+c0*b1
    C01[:]=a0*c1+c0*d1
    C10[:]=b0*a1+d0*b1
    C11[:]=b0*c1+d0*d1


    C=C.reshape(shapeOut)

    return C
    
def BatchDot2(A,B):
    #A=A.reshape((A.size/4,2,2))
    #B=B.reshape((B.size/4,2,2))

    shapeOut=A.shape

    NDir_a,nf,na,_=shapeOut
    A=A.reshape((NDir_a,nf,na,2,2))
    NDir_b,nf,na,_=B.shape
    B=B.reshape((NDir_b,nf,na,2,2))
    C=np.zeros_like(A)

    # if B.shape[0]==1:
    #     NDir=A.shape[0]
    #     #print "a"
    #     #B=B*np.ones((NDir,1,1,1,1))
    #     #print "b"
    #     #return BatchDot(A,B)
    #     #B=B.reshape((1,B.size/(4*NDir),2,2))
    #     C=np.zeros_like(A)
    # else:
    #     C=np.zeros_like(B)
    #     shapeOut=B.shape

    # print "A:",A.shape
    # print "B:",B.shape
    # print "C:",C.shape
    
    a0=A[:,:,:,0,0]
    b0=A[:,:,:,1,0]
    c0=A[:,:,:,0,1]
    d0=A[:,:,:,1,1]

    a1=B[:,:,:,0,0]
    b1=B[:,:,:,1,0]
    c1=B[:,:,:,0,1]
    d1=B[:,:,:,1,1]

    C00=C[:,:,:,0,0]
    C01=C[:,:,:,0,1]
    C10=C[:,:,:,1,0]
    C11=C[:,:,:,1,1]

    C00[:,:,:]=a0*a1+c0*b1
    C01[:,:,:]=a0*c1+c0*d1
    C10[:,:,:]=b0*a1+d0*b1
    C11[:,:,:]=b0*c1+d0*d1

    C=C.reshape(shapeOut)

    return C

def testInvertSVD():
    import pylab
    A=np.random.randn(10,10)+1j*np.random.randn(10,10)
    A0=np.linalg.inv(A)
    A1=invSVD(A)
    A2=A0-A1
    pylab.clf()
    pylab.subplot(3,2,1)
    pylab.imshow(A0.real,interpolation="nearest")
    pylab.colorbar()
    pylab.subplot(3,2,2)
    pylab.imshow(A0.imag,interpolation="nearest")
    pylab.colorbar()
    pylab.subplot(3,2,3)
    pylab.imshow(A1.real,interpolation="nearest")
    pylab.colorbar()
    pylab.subplot(3,2,4)
    pylab.imshow(A1.imag,interpolation="nearest")
    pylab.colorbar()

    pylab.subplot(3,2,5)
    pylab.imshow(A2.real,interpolation="nearest")
    pylab.colorbar()
    pylab.subplot(3,2,6)
    pylab.imshow(A2.imag,interpolation="nearest")
    pylab.colorbar()
    pylab.draw()
    pylab.show(False)
    pylab.pause(0.1)        
    

# def testSVD():
#     a=np.


def invSVD(A,Cut=1e-6):
    #print "rand"
    Ar=A  # +np.random.randn(*A.shape)*(1e-6*A.max())
    #print "stard",Ar.shape
    
    try:
        u,s,v=np.linalg.svd(Ar)
    except:
        Name="errSVDArray_%i"%int(np.random.rand(1)[0]*10000)
        print ModColor.Str("Problem inverting Matrix, saving as %s"%Name)
        print ModColor.Str("  will make it svd-able")
        np.save(Name,Ar)
        # weird - I found a matrix I cannot do svd on... - that works
        Cut=1e-20
        #Ar=np.complex64(Ar)
        u,s,v=np.linalg.svd(np.complex64(Ar))#+np.random.randn(*Ar.shape)*(1e-10*np.abs(Ar).max()))
        #u,s,v=np.linalg.svd()
        u=np.real(u)
        s=np.real(s)
        v=np.real(v)
        

    #u,s,v=np.linalg.svd(np.complex128(Ar))

    #print "ok"
    
    s[s<0.]=Cut

    s[s<Cut*s.max()]=Cut*s.max()

    ssq=(1./s)
    #Asq=np.conj(np.dot(np.dot(v.T,ssq),u.T))
    v0=v.T*ssq.reshape(1,ssq.size)
    Asq=np.conj(np.dot(v0,u.T))
    return Asq


import scipy.sparse.linalg
def invSVD_Lanczos(A):
    u,s,v=scipy.sparse.linalg.svds(A+np.random.randn(*A.shape)*(1e-6*A.max()))
    #s[s<0.]=1.e-6
    s[s<1.e-6*s.max()]=1.e-6*s.max()
    ssq=(1./s)
    #Asq=np.conj(np.dot(np.dot(v.T,ssq),u.T))
    v0=v.T*ssq.reshape(1,ssq.size)
    Asq=np.conj(np.dot(v0,u.T))
    return Asq



def SVDw(A):
    #A=(A+A.T)/2.
    u,s,v=np.linalg.svd(A)
    s[s<0.]=0.
    ssq=np.diag(np.sqrt(s))
    Asq=np.dot(np.dot(u,ssq),u.T)
    return Asq

def EigClean(A):
    
    Lq,Uq=np.linalg.eig(A.copy())
    ind =np.where(Lq<0.)[0]
    if ind.shape[0]>0:
        Lq[ind]=1e-3
    #UqInv=np.linalg.inv(Uq)
    Anew=np.real(np.dot(np.dot(Uq,np.diag(Lq)),Uq.T))
    Lq,Uq=np.linalg.eig(Anew)
#    print Lq
    return Anew


def Dot_ListBlockMat_Mat(ListBlocks,Mat):
    n=ListBlocks[0].shape[0]
    m=Mat.shape[1]
    nblock=len(ListBlocks)
    WorkMat=Mat.reshape(nblock,n,m)
    OutMat=np.zeros_like(WorkMat)
    
    for iblock in range(nblock):
        ThisBlock=ListBlocks[iblock]
        OutMat[iblock]=np.dot(ThisBlock.astype(np.float64),WorkMat[iblock].astype(np.float64))

    OutMat=OutMat.reshape(nblock*n,m)
    return OutMat
        
def Dot_ListBlockMat_Mat_Iregular(ListBlocks,Mat):
    m=Mat.shape[1]
    nblock=len(ListBlocks)
    OutMat=np.zeros_like(Mat)
    
    i0=0
    for iblock in range(nblock):
        ThisBlock=ListBlocks[iblock]
        xb,yb=ThisBlock.shape
        i1=i0+xb
        WorkMat=Mat[i0:i1,:]
        OutMat[i0:i1,:]=np.dot(ThisBlock.astype(np.float64),WorkMat.astype(np.float64))
        i0+=xb

    return OutMat

def test_Dot_ListBlockMat_Mat():
    nblocks=50
    n=100
    m=200
    B=np.random.randn(nblocks*n,m)
    ListBlocks=[]
    BlocksMat=np.zeros((nblocks*n,nblocks*n),float)
    for iblock in range(nblocks):
        ThisBlock=np.random.randn(n,n)
        ListBlocks.append(ThisBlock)
        istart=iblock*n
        BlocksMat[istart:istart+n,istart:istart+n]=ThisBlock
        
    import ClassTimeIt
    T=ClassTimeIt.ClassTimeIt()


    print "Dimensions A[%s], B[%s]"%(BlocksMat.shape,B.shape)
    R0=Dot_ListBlockMat_Mat(ListBlocks,B)
    T.timeit("ListProd")
    R1=np.dot(BlocksMat,B)
    T.timeit("NpProd")
    R2=Dot_ListBlockMat_Mat_Iregular(ListBlocks,B)
    T.timeit("ListProdIrregular")

    print np.allclose(R0,R1)
    print np.allclose(R2,R1)

    
def test_Dot_ListBlockMat_Mat_Big():
    nblocks=10*50
    n=100
    m=200
    B=np.random.randn(nblocks*n,m)
    ListBlocks=[]

    for iblock in range(nblocks):
        ThisBlock=np.random.randn(n,n)
        ListBlocks.append(ThisBlock)

        
    import ClassTimeIt
    T=ClassTimeIt.ClassTimeIt()


    print "Dimensions A[%ix%s -> %s], B[%s]"%(nblocks,ThisBlock.shape,(nblocks*n,nblocks*n),B.shape)
    R0=Dot_ListBlockMat_Mat(ListBlocks,B)
    T.timeit("ListProd")

    
