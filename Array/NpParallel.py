import _pyGridder
import ClassTimeIt
import numpy as np

def A_add_B_prod_factor(A,B,Aedge=None,Bedge=None,factor=1.,NCPU=6):

    NDimsA=len(A.shape)
    NDimsB=len(B.shape)
    if NDimsB!=2: stop
    ShapeOrig=A.shape

    if Aedge==None:
        Aedge=np.array([0,A.shape[-2],0,A.shape[-1]])
    if Bedge==None:
        Bedge=np.array([0,B.shape[-2],0,B.shape[-1]])

    Aedge=np.int32(Aedge)
    Bedge=np.int32(Bedge)

    factor=float(factor)
    
    Blocks = np.int32(np.linspace(Aedge[0],Aedge[1],NCPU+1))
    

    
    NX,NY=A.shape[-2],A.shape[-1]
    nz=A.size/(NX*NY)
    A=A.reshape((nz,NX,NY))

    for iz in range(nz):
        ThisA=A[iz]
        _pyGridder.pyAddArray(ThisA,Aedge,B,Bedge,float(factor),Blocks)

    A=A.reshape(ShapeOrig)
    return A

def test():

    Np=16000
    
    x0=1000
    x1=11000
    y0=5000
    y1=15000
    nch=1

    _a0=np.ones((nch,Np,Np),dtype=np.float32)
    _a1=np.ones((nch,Np,Np),dtype=np.float32)
    #b=np.float32(np.arange(Np**2).reshape((Np,Np)))
    b=np.ones((Np,Np),dtype=np.float32)

    
    Aedge=np.array([x0,x1,y0,y1],np.int32)
    Bedge=np.array([x0+1,x1+1,y0,y1],np.int32)


    T=ClassTimeIt.ClassTimeIt()

    N=10
    NBlocks=6
    factor=-1.
    
    for i in range(N):
        #b=np.float32(np.random.randn(Np,Np))
        A_add_B_prod_factor(_a0,b,Aedge,Bedge,factor=float(factor),NCPU=NBlocks)
        #A_add_B_prod_factor(_a0,b,Aedge,Bedge,factor=float(factor),NCPU=NBlocks)
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
    for i in range(N):
        for ch in range(nch):
            a0=_a0[ch]
            a1=_a1[ch]
            a_x0,a_x1,a_y0,a_y1=Aedge
            b_x0,b_x1,b_y0,b_y1=Bedge
            a1[a_x0:a_x1,a_y0:a_y1]+=b[b_x0:b_x1,b_y0:b_y1]*factor
            #a1=a0+b*factor
    T.timeit("2")
    
    
#test()
