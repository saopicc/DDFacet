import numpy as np
from DDFacet.Gridder import _pyGridder
#from DDFacet.Gridder import _pyGridderSmear
from DDFacet.Gridder import _pyGridderPseudoPSFSmear#_pyDeGridderWPolPseudoPSF
#import pylab
from pyrap.images import image
#import MyPickle
import os
from DDFacet.Other import MyLogger
from DDFacet.Other import ClassTimeIt
import ModCF
from DDFacet.ToolsDir import ModToolBox

from DDFacet.Other import ModColor
import ClassCasaImage
#import ClassApplyJones
#import ToolsDir.GiveMDC
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
from DDFacet.Array import ModLinAlg
import copy
import time
from DDFacet.ToolsDir import ModFFTW

#import ClassApplyJones
#from ClassME import MeasurementEquation
log=MyLogger.getLogger("ClassDDEGridMachinePseudoPSF")

#from DDFacet.Other import ToolsDir
import pylab
#import ClassData
from DDFacet.Array import NpShared


from DDFacet.Other import ClassTimeIt

from DDFacet.Parset import ReadCFG
#import ReadCFG
#import MyOptParse

#from DDFacet.Data import ClassMS


from DDFacet.Data import ClassVisServer

def testGrid():
    Parset=ReadCFG.Parset("%s/DDFacet/Parset/DefaultParset.cfg"%os.environ["DDFACET_DIR"])
    DC=Parset.DicoPars
    DC["VisData"]["MSName"]="/home/atemkeng/PSF_CYRIL_METHOD/VLAC-lowresolution.MS"#"/home/atemkeng/DDFacet/Test/vlac-hires-1.0s-1.0MHz.MS"#"/home/atemkeng/DDFacet/Test/vlac-hires-1.0s-1.0MHz.MS"#/media/tasse/data/killMS_Pack/killMS2/Test/0000.MS"
    DC["ImagerCF"]["wmax"]=2000    
    DC["ImagerCF"]["Nw"]=400    

    
    DC["ImagerMainFacet"]["NFacets"]= 3
    DC["ImagerMainFacet"]["Npix"]= 513
    DC["ImagerMainFacet"]["Cell"]= 1.250#30 #1.25
    DC["ImagerMainFacet"]["Padding"]= 1.7
    DC["ImagerCF"]["OverS"]  = 25#21
    DC["ImagerCF"]["Support"]= 9
    DC["Compression"]["CompressModeDeGrid"] =False 
    DC["Compression"]["CompDeGridMode"] = True   #GD["Compression"]["CompDeGridMode"]==0
    VS=ClassVisServer.ClassVisServer(DC["VisData"]["MSName"],
                                     ColName=DC["VisData"]["ColName"],
                                     TVisSizeMin=DC["VisData"]["TChunkSize"]*60*1.1,
                                     #DicoSelectOptions=DicoSelectOptions,
                                     TChunkSize=DC["VisData"]["TChunkSize"],
                                     IdSharedMem="caca",
                                     Robust=DC["ImagerGlobal"]["Robust"],
                                     Weighting="Natural",
                                     DicoSelectOptions=dict(DC["DataSelection"]),
                                     NCPU=DC["Parallel"]["NCPU"],GD=DC)
    Cell=(DC["ImagerMainFacet"]["Cell"]/3600.)*np.pi/180.
    npix=DC["ImagerMainFacet"]["Npix"]
    Padding=DC["ImagerMainFacet"]["Padding"]
    #_,npix=EstimateNpix(npix,Padding)
    offsetSourceX=100
    offsetSourceY=200
    M=-Cell*(offsetSourceX)
    L=Cell*offsetSourceY
    print "position", np.sqrt(M**2+L**2)*180./np.pi
    #stop
    N=np.sqrt(1.-L**2-M**2)
    print M,L
    
    sh=[1,1,npix,npix]
    VS.setFOV(sh,sh,sh,Cell)
    VS.CalcWeigths()
    Load=VS.LoadNextVisChunk()
    
    DATA=VS.ThisDataChunk

    from pyrap.tables import table
    tabhires1 = table("/home/atemkeng/PSF_CYRIL_METHOD/VLAC-lowresolution.MS")
    UVWhires1 = tabhires1.getcol("UVW")
    A0hires1 = tabhires1.getcol("ANTENNA1")
    A1hires1= tabhires1.getcol("ANTENNA2")
    datahires1 = tabhires1.getcol("DATA")
    
    tabhires1.close()

    #DATA["data"]=datahires1.copy()
    
    # DicoConfigGM={"Npix":NpixFacet,
    #               "Cell":Cell,
    #               "ChanFreq":ChanFreq,
    #               "DoPSF":False,
    #               "Support":Support,
    #               "OverS":OverS,
    #               "wmax":wmax,
    #               "Nw":Nw,
    #               "WProj":True,
    #               "DoDDE":self.DoDDE,
    #               "Padding":Padding}
    # GM=ClassDDEGridMachine(Parset.DicoPars,DoDDE=False,WProj=True,lmShift=(0.,0.),JonesDir=3,SpheNorm=True,IdSharedMem="caca")
    # GM=ClassDDEGridMachine(Parset.DicoPars,
    #                        IdSharedMem="caca",
    #                        **DicoConfigGMself.DicoImager[iFacet]["DicoConfigGM"])

    ChanFreq=VS.MS.ChanFreq.flatten()
    GM=ClassDDEGridMachine(DC,
                           ChanFreq,
                           npix,
                           lmShift=(0.,0.),#self.DicoImager[iFacet]["lmShift"],
                           IdSharedMem="caca")





    row0=0
    row1=DATA["uvw"].shape[0]#-1
    A0=np.int32(DATA["A0"]).copy()#[row0:row1]
    A1=np.int32(DATA["A1"]).copy()#[row0:row1]
    ind=np.where((A0==17)&(A1==26))[0]
    uvw=np.float64(DATA["uvw"][ind]).copy()#[row0:row1]
    duvw = np.float64(DATA["duvw_dt"][ind]).copy();
    listsemar = [duvw[:,0],duvw[:,1],duvw[:,2], 400,51, 1400000000.0]
    print 'duvw',duvw.shape
    print 'uvw',uvw.shape
    
    c1=duvw[:,0];c4=duvw[:,1]
    print uvw.shape, duvw.shape
    
    c2=uvw[:,0];c3=uvw[:,1]
    #pylab.plot(c1,c4,'o',label='evaluateuvw')
    #pylab.plot(c2,c3,label='trueevaluateuvw')
    #pylab.legend()
    #pylab.show()
    #stop
    

    ## from uvw fine duvw_dt
    ##duvw = np.diff(uvw, axis=0);
   
    #uvw[:,2]=0
    times=np.float64(DATA["times"][ind]).copy()#[row0:row1]
    data=np.complex64(DATA["data"][ind]).copy()#[row0:row1]
    #data.fill(1.)
    #data[:,:,0]=1
    #data[:,:,3]=1
    A0=np.int32(DATA["A0"][ind]).copy()#[row0:row1]
    A1=np.int32(DATA["A1"][ind]).copy()#[row0:row1]
    
    C=2.99792458e8
    U,V,W=uvw.T
    
    
    
    from pyrap.tables import table
    tabhires = table("/home/atemkeng/PSF_CYRIL_METHOD/VLAC-hiresolution.MS")
    UVWhires = tabhires.getcol("UVW")
    
    A0hires = tabhires.getcol("ANTENNA1")
    A1hires= tabhires.getcol("ANTENNA2")
    uvwhires=UVWhires[(A0hires==17)&(A1hires==26)]
    uh=uvwhires[:,0].reshape(144,100)
    vh=uvwhires[:,1].reshape(144,100)
    wh=uvwhires[:,2].reshape(144,100)
    datahires = tabhires.getcol("DATA")
    print"duvw", duvw
    dt=50000
    
    v3g = 2*np.pi*(ChanFreq[0]/C)*(uh[:, 99]*L + vh[:, 99]*M + wh[:,99]*(N-1.))
    v4g = 2*np.pi*(ChanFreq[0]/C)*(uh[:,0]*L + vh[:,0]*M + wh[:, 0]*(N-1.))
    sine= (v3g- v4g)/2.
    sine[sine==0]=1e-9
    factsine=np.sin(sine)/sine
    #K=factsine*10.*np.exp(2.*np.pi*1j*(ChanFreq[0]/C)*(uh[:,50]*L+vh[:,50]*M+wh[:,50]*(N-1)))
    #print'****************', factsine
   
    dA_time=np.pi*(duvw[:,0].copy()*L+duvw[:,1].copy()*M+duvw[:,2].copy()*(N-1.))*dt;
    dA_time[dA_time==0]=1e-9
    K=(np.sin(dA_time)/dA_time)*10.*np.exp(2.*np.pi*1j*(ChanFreq[0]/C)*(U*L+V*M+W*(N-1)))
    print'****************', (np.sin(dA_time)/dA_time)
    
    #K = datahires[(A0hires==1)&(A1hires==26)][:,0,0]
    #data.fill(0.)
    tabhires.close()
    
    #K = np.complex64(DATA["data"][ind]).copy()[:,0,0]
   # print "K",K.shape
    
    # pylab.clf()
    # pylab.plot(uvw)
    # pylab.draw()
    # pylab.show(False)
    # pylab.pause(0.1)

    #uvw.fill(0)
    
    #print data
    flag=np.bool8(DATA["flags"][ind]).copy()#[row0:row1,:,:].copy()
    #ind=np.where(np.logical_not((A0==12)&(A1==14)))[0]
    #flag[ind,:,:]=1
    flag.fill(0)

    #MapSmear=NpShared.GiveArray("%sMappingSmearing"%("caca"))
    #stop
    #row=19550
    #print A0[row],A1[row],flag[row]
    #stop

    T=ClassTimeIt.ClassTimeIt("main")
    #Grid=GM.put(times,uvw,data,flag,(A0,A1),W=None,PointingID=0,DoNormWeights=True)#, DicoJonesMatrices=DicoJonesMatrices)
    Grid=GM.put(times,uvw,data,flag,(A0,A1),W=None,PointingID=0,DoNormWeights=True,DicoJonesMatrices=None,freqs=ChanFreq)
    # pylab.clf()
    # pylab.imshow(np.real(Grid[0,0]))
    # #pylab.imshow(np.random.rand(50,50))
    # pylab.colorbar()
    # pylab.draw()
    # pylab.show(False)

    #print Grid
    
    # Grid=np.zeros(sh,np.complex64)
    T.timeit("grid")
    # Grid[np.isnan(Grid)]=-1

    #Grid[0,0,100,100]=10.

    
    Grid.fill(0.)
   
    _,_,n,n=Grid.shape
    #print Grid.shape
    
    Grid[:,:,n/2+offsetSourceX,n/2+offsetSourceY]=10.
    
    data.fill(0)

    #GM.GD["Compression"]["CompressModeDeGrid"] = True
    #data=GM.get(times,uvw,data,flag,(A0,A1),Grid)#, DicoJonesMatrices=DicoJonesMatrices)
    #data=GM.get(times,uvw,data,flag,(A0,A1),Grid,DicoJonesMatrices=None,freqs=ChanFreq,ImToGrid=True)


    
    #data0=data.copy()
    #data.fill(0)
    flag.fill(0)
    #np.save('du',duvw[:,0])
    du=duvw[:,0].copy()
    dv=duvw[:,1].copy()
    dw=duvw[:,2].copy()
    data1=GM.get(times,uvw,data,flag,(A0,A1),Grid,DicoJonesMatrices=None,freqs=ChanFreq,ImToGrid=True,du=du,dv=dv,dw=dw, dt=dt,df=1, freq=1400000000.0, lcor=L, mcor=M)
    #data1=GM.get(times,uvw,data,flag,(A0,A1),Grid)#, DicoJonesMatrices=DicoJonesMatrices)

    #ind=np.where(((A0==12)&(A1==14)))[0]
    #data0=data0[ind]
    #data1=data1[ind]
    #print data0-data1
    op0=np.abs
    op1=np.angle
    #op0=np.real
    #op1=np.imag
    nbl=VS.MS.nbl

    #ind=np.where((A0==0)&(A1==21))[0]

    #d0=data0[ind,0,0].ravel()
    d1=-data1[:,0,0]#.ravel()-data1[:,0,0]
    print "herererererere",d1==K
    
    #ind=np.where((d0-d1)[:]!=0)

    print "****",op0(d1)
    
    pylab.clf()
    pylab.subplot(1,2,1)
    #pylab.plot(op0(d0))
    pylab.plot(op0(d1), 'r',label='amplitude of the degridded data')
    pylab.plot(op0(K),'b', label='amplitude of the ungridded data')
    pylab.ylabel("Amplitude")
    pylab.xlabel("uv bins")
   # pylab.plot(op0(K)-op0(d1), 'k',label='residual')
    pylab.legend(loc='lower center')
    pylab.title("10Jy source, data from the shortest  baseline")
    #pylab.plot(uvw[ind,0])
    #pylab.plot(op0(d0-d1))
    pylab.ylim(-11,11)
    pylab.grid()
    pylab.subplot(1,2,2)
    #pylab.plot(op1(d0))
    pylab.plot(op1(d1),'r',label='phase of the degridded data')
    pylab.plot(op1(K), 'b',label='phase of the ungridded data')
    pylab.ylabel("Phase")
    pylab.xlabel("uv bins")
    print"***********max real",op0(K).max()
    print"***********max grid",op0(d1).max()
    #pylab.plot(op1(K)-op1(d1), 'k',label='residual')
    
    #pylab.plot(op1(d0-d1))
    pylab.ylim(-11,11)
    pylab.grid()
    pylab.draw()
    pylab.legend()
    pylab.title("10Jy source, data from the shortest baseline")
    pylab.show(False)
    pylab.pause(0.1)
    
#     for ibl in [122]:#range(1,nbl)[::11]:
#         d0=data0[ibl::nbl,:,0].ravel()
#         d1=data1[ibl::nbl,:,0].ravel()
#         pylab.clf()
#         pylab.subplot(1,2,1)
#         pylab.plot(op0(d0))
#         pylab.plot(op0(d1))
#         pylab.plot(op0(d0-d1))
#         pylab.title(ibl)
#         pylab.subplot(1,2,2)
#         pylab.plot(op1(d0))
#         pylab.plot(op1(d1))
#         pylab.plot(op1(d0-d1))
#         pylab.draw()
#         pylab.show(False)
#         pylab.pause(0.1)
# #        time.sleep(0.2)


class ClassDDEGridMachine():
    def __init__(self,GD,
                 ChanFreq,
                 Npix,
                 lmShift=(0.,0.),
                 IdSharedMem="",
                 IDFacet=0,
                 SpheNorm=True):
        T=ClassTimeIt.ClassTimeIt("Init_ClassDDEGridMachine")
        T.disable()
        self.GD=GD
        self.IDFacet=IDFacet
        self.SpheNorm=SpheNorm

        self.IdSharedMem=IdSharedMem

        #self.DoPSF=DoPSF
        self.DoPSF=False
        # if DoPSF:
        #     self.DoPSF=True
        #     Npix=Npix*2

        Precision=GD["ImagerGlobal"]["Precision"]
        PolMode=GD["ImagerGlobal"]["PolMode"]

        if Precision=="S":
            self.dtype=np.complex64
        elif Precision=="D":
            self.dtype=np.complex128

        self.dtype=np.complex64
        T.timeit("0")
        Padding=GD["ImagerMainFacet"]["Padding"]
        self.NonPaddedNpix,Npix=EstimateNpix(Npix,Padding)
        self.Padding=Npix/float(self.NonPaddedNpix)
        #self.Padding=Padding
        

        self.PolMode=PolMode
        if PolMode=="I":
            self.npol=1
            self.PolMap=np.array([0,5,5,0],np.int32)
        elif PolMode=="IQUV":
            self.npol=4
            self.PolMap=np.array([0,1,2,3],np.int32)

        self.Npix=Npix
        self.NonPaddedShape=(1,self.npol,self.NonPaddedNpix,self.NonPaddedNpix)
        self.GridShape=(1,self.npol,self.Npix,self.Npix)
        x0=(self.Npix-self.NonPaddedNpix)/2#+1
        self.PaddingInnerCoord=(x0,x0+self.NonPaddedNpix)

        T.timeit("1")

        OverS=GD["ImagerCF"]["OverS"]
        Support=GD["ImagerCF"]["Support"]
        Nw=GD["ImagerCF"]["Nw"]
        wmax=GD["ImagerCF"]["wmax"]
        Cell=GD["ImagerMainFacet"]["Cell"]

        
        #T=ClassTimeIt.ClassTimeIt("ClassImager")
        #T.disable()

        self.Cell=Cell
        self.incr=(np.array([-Cell,Cell],dtype=np.float64)/3600.)*(np.pi/180)
        #CF.fill(1.)
        ChanFreq=ChanFreq.flatten()
        self.ChanFreq=ChanFreq
        if self.ChanFreq.size>1:
            df=self.ChanFreq[1::]-self.ChanFreq[0:-1]
            ddf=np.abs(df-np.mean(df))
            self.ChanEquidistant=int(np.max(ddf)<1.)
        else:
            self.ChanEquidistant=1

        #print self.ChanEquidistant
        self.FullScalarMode=int(GD["DDESolutions"]["FullScalarMode"])

        T.timeit("3")

        self.ChanWave=2.99792458e8/self.ChanFreq
        self.UVNorm=2.*1j*np.pi/self.ChanWave
        self.UVNorm.reshape(1,self.UVNorm.size)
        self.Sup=Support
        self.WProj=True
        self.wmax=wmax
        self.Nw=Nw
        self.OverS=OverS
        self.lmShift=lmShift

        T.timeit("4")
        self.CalcCF()

        self.reinitGrid()
        self.CasaImage=None
        self.DicoATerm=None
        T.timeit("5")

    def CalcCF(self):
        #Grid=np.zeros(self.GridShape,dtype=self.dtype)
        #self.FFTWMachine=ModFFTW.FFTW_2Donly(Grid, ncores = 1)
        #self.FFTWMachine=ModFFTW.FFTW_2Donly_np(Grid, ncores = 1)
        #self.FFTWMachine=ModFFTW.FFTW_2Donly_np(Grid, ncores = 1)

        self.FFTWMachine=ModFFTW.FFTW_2Donly_np(self.GridShape,self.dtype, ncores = 1)

        #SharedName="%sFFTW.%i"%(self.IdSharedMem,self.IDFacet)
        #self.FFTWMachine=ModFFTW.FFTW_2Donly(self.GridShape,self.dtype, ncores = 1, FromSharedId=SharedName)

        if self.WProj:
            self.WTerm=ModCF.ClassWTermModified(Cell=self.Cell,
                                                Sup=self.Sup,
                                                Npix=self.Npix,
                                                Freqs=self.ChanFreq,
                                                wmax=self.wmax,
                                                Nw=self.Nw,
                                                OverS=self.OverS,
                                                lmShift=self.lmShift,
                                                IdSharedMem=self.IdSharedMem,
                                                IDFacet=self.IDFacet)
        else:
            self.WTerm=ModCF.ClassSTerm(Cell=self.Cell,
                                        Sup=self.Support,
                                        Npix=self.Npix,
                                        Freqs=self.ChanFreq,
                                        wmax=self.wmax,
                                        Nw=self.Nw,
                                        OverS=self.OverS)

        self.ifzfCF= self.WTerm.ifzfCF
 

    def setSols(self,times,xi):
        self.Sols={"times":times,"xi":xi}


    def ShiftVis(self,uvw,vis,reverse=False):
        #if self.lmShift==None: return uvw,vis
        l0,m0=self.lmShift
        u,v,w=uvw.T
        U=u.reshape((u.size,1))
        V=v.reshape((v.size,1))
        W=w.reshape((w.size,1))
        n0=np.sqrt(1-l0**2-m0**2)-1
        if reverse: 
            corr=np.exp(-self.UVNorm*(U*l0+V*m0+W*n0))
        else:
            corr=np.exp(self.UVNorm*(U*l0+V*m0+W*n0))
        
        U+=W*self.WTerm.Cu
        V+=W*self.WTerm.Cv

        corr=corr.reshape((U.size,self.UVNorm.size,1))
        vis*=corr

        U=U.reshape((U.size,))
        V=V.reshape((V.size,))
        W=W.reshape((W.size,))
        uvw=np.array((U,V,W)).T.copy()

        return uvw,vis



    def reinitGrid(self):
        #self.Grid.fill(0)
        self.NChan, self.npol, _,_=self.GridShape
        self.SumWeigths=np.zeros((self.NChan,self.npol),np.float64)



    def GiveParamJonesList(self,DicoJonesMatrices,times,A0,A1,uvw):
        JonesMatrices=DicoJonesMatrices["Jones"]
        MapJones=DicoJonesMatrices["MapJones"]
        l0,m0=self.lmShift
        DicoClusterDirs=DicoJonesMatrices["DicoClusterDirs"]
        lc=DicoClusterDirs["l"]
        mc=DicoClusterDirs["m"]
        sI=DicoClusterDirs["I"]
        
        #lc,mc=np.random.randn(100)*np.pi/180,np.random.randn(100)*np.pi/180
        
        
        InterpMode=self.GD["DDESolutions"]["Type"]
        d0=self.GD["DDESolutions"]["Scale"]*np.pi/180
        gamma=self.GD["DDESolutions"]["gamma"]
        
        #d=np.sqrt((l0-lc)**2+(m0-mc)**2)
        #idir=np.argmin(d)
        #w=sI/(1.+d/d0)**gamma
        #w/=np.sum(w)
        
        
        d=np.sqrt((l0-lc)**2+(m0-mc)**2)
        idir=np.argmin(d)
        w=sI/(1.+d/d0)**gamma
        w/=np.sum(w)
        w[w<(0.2*w.max())]=0
        ind=np.argsort(w)[::-1]
        w[ind[3::]]=0
        w/=np.sum(w)


        # pylab.clf()
        # pylab.scatter(lc,mc,c=w)
        # pylab.scatter([l0],[m0],marker="+")
        # pylab.draw()
        # pylab.show(False)
        
        if InterpMode=="Nearest":
            InterpMode=0
        elif InterpMode=="Krigging":
            InterpMode=1

                
        #ParamJonesList=[MapJones,A0.astype(np.int32),A1.astype(np.int32),JonesMatrices.astype(np.complex64),idir]
        if A0.size!=uvw.shape[0]: stop
        self.CheckTypes(A0=A0,A1=A1,Jones=JonesMatrices)
        
        ParamJonesList=[MapJones,A0,A1,JonesMatrices,np.array([idir],np.int32),np.float32(w),np.array([InterpMode],np.int32)]
        #print "idir %i"%idir
        return ParamJonesList


    def put(self,times,uvw,visIn,flag,A0A1,W=None,PointingID=0,DoNormWeights=True,DicoJonesMatrices=None,freqs=None):#,doStack=False):
        #log=MyLogger.getLogger("ClassImager.addChunk")
        vis=visIn#.copy()

        T=ClassTimeIt.ClassTimeIt("put")
        T.disable()
        self.DoNormWeights=DoNormWeights
        if not(self.DoNormWeights):
            self.reinitGrid()

        #isleep=0
        #print "sleeping DDE... %i"%isleep; time.sleep(5); isleep+=1

        #LTimes=sorted(list(set(times.tolist())))
        #NTimes=len(LTimes)
        A0,A1=A0A1

        # if self.DicoATerm==None:
        #     self.CalcAterm(times,A0A1,PointingID=PointingID)
        # if self.DoDDE:
        #     for ThisTime,itime0 in zip(LTimes,range(NTimes)):
        #         Jones,JonesH=self.DicoATerm[ThisTime]
        #         JonesInv=ModLinAlg.BatchInverse(Jones)
        #         JonesHInv=ModLinAlg.BatchInverse(JonesH)
        #         indThisTime=np.where(times==ThisTime)[0]
        #         ThisA0=A0[indThisTime]
        #         ThisA1=A1[indThisTime]
        #         P0=ModLinAlg.BatchDot(JonesInv[ThisA0,:,:],vis[indThisTime])
        #         vis[indThisTime]=ModLinAlg.BatchDot(P0,JonesHInv[ThisA1,:,:])
        #     vis/=self.norm
                
        T.timeit("1")
        # uvw,vis=self.ShiftVis(uvw,vis,reverse=True)


        #if not(doStack):
        #    self.reinitGrid()
        #self.reinitGrid()
        npol=self.npol
        NChan=self.NChan

        NVisChan=vis.shape[1]
        if W==None:
            W=np.ones((uvw.shape[0],NVisChan),dtype=np.float64)
            
        #else:
        #    W=W.reshape((uvw.shape[0],1))*np.ones((1,NVisChan))

        #print "sleeping DDE... %i"%isleep; time.sleep(5); isleep+=1
        SumWeigths=self.SumWeigths
        if vis.shape!=flag.shape:
            raise Exception('vis[%s] and flag[%s] should have the same shape'%(str(vis.shape),str(flag.shape)))
        
        u,v,w=uvw.T
        #vis[u==0,:,:]=0
        #flag[u==0,:,:]=True
        # if self.DoPSF:
        #     vis.fill(0)
        #     vis[:,:,0]=1
        #     vis[:,:,3]=1

        T.timeit("2")

        Grid=np.zeros(self.GridShape,dtype=self.dtype)
        #print "sleeping DDE... %i"%isleep; time.sleep(5); isleep+=1

        l0,m0=self.lmShift
        FacetInfos=np.float64(np.array([self.WTerm.Cu,self.WTerm.Cv,l0,m0]))

        # if not(vis.dtype==np.complex64):
        #     print "vis should be of type complex128 (and has type %s)"%str(vis.dtype)
        #     stop

        #print "sleeping DDE... %i"%isleep; time.sleep(5); isleep+=1

        #print vis.dtype
        #vis.fill(1)

        self.CheckTypes(Grid=Grid,vis=vis,uvw=uvw,flag=flag,ListWTerm=self.WTerm.Wplanes,W=W)
        ParamJonesList=[]
        if DicoJonesMatrices!=None:
            ApplyAmp=0
            ApplyPhase=0
            ScaleAmplitude=0
            CalibError=0.
            
            if "A" in self.GD["DDESolutions"]["DDModeGrid"]:
                ApplyAmp=1
            if "P" in self.GD["DDESolutions"]["DDModeGrid"]:
                ApplyPhase=1
            if self.GD["DDESolutions"]["ScaleAmpGrid"]:
                ScaleAmplitude=1
                CalibError=(self.GD["DDESolutions"]["CalibErr"]/3600.)*np.pi/180
            LApplySol=[ApplyAmp,ApplyPhase,ScaleAmplitude,CalibError]
            ParamJonesList=self.GiveParamJonesList(DicoJonesMatrices,times,A0,A1,uvw)
            ParamJonesList=ParamJonesList+LApplySol


        T.timeit("3")
        #print "sleeping DDE..."; time.sleep(5)


        if freqs==None:
            freqs=np.float64(self.ChanFreq)
        
        T2=ClassTimeIt.ClassTimeIt("Gridder")
        T2.disable()
        #print "vis",vis.min(),vis.max()

        if self.GD["Compression"]["CompGridMode"]==0:
            
            Grid=_pyGridder.pyGridderWPol(Grid,
                                              vis,
                                              uvw,
                                              flag,
                                              W,
                                              SumWeigths,
                                              0,
                                              self.WTerm.Wplanes,
                                              self.WTerm.WplanesConj,
                                              np.array([self.WTerm.RefWave,self.WTerm.wmax,len(self.WTerm.Wplanes),self.WTerm.OverS],dtype=np.float64),
                                              self.incr.astype(np.float64),
                                              freqs,
                                              [self.PolMap,FacetInfos],
                                              ParamJonesList) # Input the jones matrices
        else:
            MapSmear=NpShared.GiveArray("%sMappingSmearing.Grid"%(self.IdSharedMem))
            
            _pyGridderPseudoPSFSmear.pyGridderWPol(Grid,
                                               vis,
                                               uvw,
                                               flag,
                                               W,
                                               SumWeigths,
                                               0,
                                               self.WTerm.Wplanes,
                                               self.WTerm.WplanesConj,
                                               np.array([self.WTerm.RefWave,self.WTerm.wmax,len(self.WTerm.Wplanes),self.WTerm.OverS],dtype=np.float64),
                                               self.incr.astype(np.float64),
                                               freqs,
                                               [self.PolMap,FacetInfos],
                                               ParamJonesList,
                                               MapSmear,
                                               [self.FullScalarMode,self.ChanEquidistant])
        
        
        
        #return Grid
        T2.timeit("gridder")
        # print SumWeigths
        # return
        # del(Grid)
        T.timeit("4 (grid)")


        #print "minmax grid=",Grid.min(),Grid.max()
        Dirty= self.GridToIm(Grid)
        #print "minmax dirty=",Dirty.min(),Dirty.max()
        #Dirty=Grid
        #print Grid.max()
        del(Grid)
        T.timeit("5 (grid)")
        #print "sleeping DDE... %i"%isleep; time.sleep(5); isleep+=1
        if self.SpheNorm:
            Dirty = self.cutImPadded(Dirty)
        

        #print "sleeping DDE... %i"%isleep; time.sleep(5); isleep+=1
        T.timeit("6")
        # Grid[:,:,:,:]=Grid.real
        # import pylab
        # pylab.clf()
        # pylab.imshow(np.abs(Grid[0,0]))
        # pylab.draw()
        # pylab.show(False)
        # stop

        import gc
        gc.enable()
        gc.collect()

        return Dirty

    def CheckTypes(self,Grid=None,vis=None,uvw=None,flag=None,ListWTerm=None,W=None,A0=None,A1=None,Jones=None):
        if Grid!=None:
            if not(Grid.dtype==np.complex64):
                raise NameError('Grid.dtype %s %s'%(str(Grid.dtype),str(self.dtype)))
            if not(Grid.flags.c_contiguous):
                raise NameError("Has to be contiuous")
        if vis!=None:
            if not(vis.dtype==np.complex64):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
            if not(vis.flags.c_contiguous):
                raise NameError("Has to be contiuous")
        if uvw!=None:
            if not(uvw.dtype==np.float64):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
            if not(uvw.flags.c_contiguous):
                raise NameError("Has to be contiuous")
        if flag!=None:
            if not(flag.dtype==np.bool8):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
            if not(flag.flags.c_contiguous):
                raise NameError("Has to be contiuous")
        if ListWTerm!=None:
            if not(ListWTerm[0].dtype==np.complex64):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
        if W!=None:
            if not(W.dtype==np.float64):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
            if not(W.flags.c_contiguous):
                raise NameError("Has to be contiuous")
        if A0!=None:
            if not(A0.dtype==np.int32):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
            if not(A0.flags.c_contiguous):
                raise NameError("Has to be contiuous")
        if A1!=None:
            if not(A1.dtype==np.int32):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
            if not(A1.flags.c_contiguous):
                raise NameError("Has to be contiuous")
        if Jones!=None:
            if not(Jones.dtype==np.complex64):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
            if not(Jones.flags.c_contiguous):
                raise NameError("Has to be contiuous")


    def get(self,times,uvw,visIn,flag,A0A1,ModelImage,PointingID=0,Row0Row1=(0,-1),DicoJonesMatrices=None,freqs=None,ImToGrid=True,du=None,dv=None,dw=None, dt=None,df=None, freq=None, lcor=None, mcor=None):
        #log=MyLogger.getLogger("ClassImager.addChunk")
        T=ClassTimeIt.ClassTimeIt("get")
        T.disable()
        vis=visIn#.copy()



        #LTimes=sorted(list(set(times.tolist())))
        #NTimes=len(LTimes)
        A0,A1=A0A1

        T.timeit("0")
        if ImToGrid:
            if np.max(np.abs(ModelImage))==0: return vis
            Grid=self.dtype(self.setModelIm(ModelImage))
        else:
            Grid=ModelImage
        #np.save("Grid",Grid)
        

        T.timeit("1")
        #dummy=np.abs(vis).astype(np.float32)




        npol=self.npol
        NChan=self.NChan
        SumWeigths=self.SumWeigths
        if vis.shape!=flag.shape:
            raise Exception('vis[%s] and flag[%s] should have the same shape'%(str(vis.shape),str(flag.shape)))

        
        #u,v,w=uvw.T
        #vis[u==0,:,:]=0
        #flag[u==0,:,:]=True
      
        #uvwOrig=uvw.copy()
        
        # uvw,vis=self.ShiftVis(uvw,vis,reverse=False)
        
        # vis.fill(0)
        
        l0,m0=self.lmShift
        FacetInfos=np.float64(np.array([self.WTerm.Cu,self.WTerm.Cv,l0,m0]))
        Row0,Row1=Row0Row1
        if Row1==-1:
            Row1=uvw.shape[0]
        RowInfos=np.array([Row0,Row1]).astype(np.int32)

        T.timeit("2")
            
        self.CheckTypes(Grid=Grid,vis=vis,uvw=uvw,flag=flag,ListWTerm=self.WTerm.Wplanes)

        ParamJonesList=[]
        # if DicoJonesMatrices!=None:
        #     ApplyAmp=0
        #     ApplyPhase=0
        #     if "A" in self.GD["DDESolutions"]["DDModeDeGrid"]:
        #         ApplyAmp=1
        #     if "P" in self.GD["DDESolutions"]["DDModeDeGrid"]:
        #         ApplyPhase=1
        #     LApplySol=[ApplyAmp,ApplyPhase]
        #     ParamJonesList=self.GiveParamJonesList(DicoJonesMatrices,times,A0,A1,uvw)
        #     ParamJonesList=ParamJonesList+LApplySol

        if DicoJonesMatrices!=None:
            ApplyAmp=0
            ApplyPhase=0
            ScaleAmplitude=0
            CalibError=0.
            
            if "A" in self.GD["DDESolutions"]["DDModeGrid"]:
                ApplyAmp=1
            if "P" in self.GD["DDESolutions"]["DDModeGrid"]:
                ApplyPhase=1
            if self.GD["DDESolutions"]["ScaleAmpDeGrid"]:
                ScaleAmplitude=1
                CalibError=(self.GD["DDESolutions"]["CalibErr"]/3600.)*np.pi/180
            LApplySol=[ApplyAmp,ApplyPhase,ScaleAmplitude,CalibError]
            ParamJonesList=self.GiveParamJonesList(DicoJonesMatrices,times,A0,A1,uvw)
            ParamJonesList=ParamJonesList+LApplySol

        print freqs
        if freqs==None:
            freqs=np.float64(self.ChanFreq)
        print freqs
        

        T.timeit("3")
        #print vis

        if self.GD["Compression"]["CompDeGridMode"]==0:
            print "CompDeGridMode]==0:"
            
            _ = _pyGridder.pyDeGridderWPol(Grid,
                                             vis,
                                             uvw,
                                             flag,
                                             SumWeigths,
                                             0,
                                             self.WTerm.WplanesConj,
                                             self.WTerm.Wplanes,
                                             np.array([self.WTerm.RefWave,self.WTerm.wmax,len(self.WTerm.Wplanes),self.WTerm.OverS],dtype=np.float64),
                                             self.incr.astype(np.float64),
                                             freqs,
                                             [self.PolMap,FacetInfos,RowInfos],
                                             ParamJonesList)
        else:
            print "CompDeGridMode]==1:"
             
            MapSmear=NpShared.GiveArray("%sMappingSmearing.DeGrid"%(self.IdSharedMem))
            aray = [1]#np.float64(np.array([]))
            print du
            
            vis = _pyGridderPseudoPSFSmear.pyDeGridderWPolPseudoPSF(Grid, #pyDeGridderWPol
                                                  vis,
                                                  uvw,
                                                  flag,
                                                  SumWeigths,
                                                  0,
                                                  self.WTerm.WplanesConj,
                                                  self.WTerm.Wplanes,
                                                  np.array([self.WTerm.RefWave,self.WTerm.wmax,len(self.WTerm.Wplanes),self.WTerm.OverS],dtype=np.float64),
                                                  self.incr.astype(np.float64),
                                                  freqs,
                                                  [self.PolMap,FacetInfos,RowInfos],
                                                  ParamJonesList,
                                                  MapSmear,
                                                                    [self.FullScalarMode,self.ChanEquidistant],
                                                                    du,dv,dw, dt,df, np.float64(freq),lcor,mcor )
            

        T.timeit("4 (degrid)")
        #print vis
        
        # uvw,vis=self.ShiftVis(uvwOrig,vis,reverse=False)

        #T.timeit("5")
        return vis


    #########################################################
    ########### ADDITIONALS
    #########################################################

    def setModelIm(self,ModelIm):
        _,_,n,n=ModelIm.shape
        x0,x1=self.PaddingInnerCoord
        # self.ModelIm[:,:,x0:x1,x0:x1]=ModelIm
        ModelImPadded=np.zeros(self.GridShape,dtype=self.dtype)
        ModelImPadded[:,:,x0:x1,x0:x1]=ModelIm
        
        Grid=self.ImToGrid(ModelImPadded)*n**2
        return Grid

    def ImToGrid(self,ModelIm):
        
        npol=self.npol
        ModelImCorr=ModelIm*(self.WTerm.OverS*self.Padding)**2

        nchan,npol,_,_=ModelImCorr.shape
        for ichan in range(nchan):
            for ipol in range(npol):
                ModelImCorr[ichan,ipol][:,:]=ModelImCorr[ichan,ipol][:,:].real/self.ifzfCF


        ModelUVCorr=self.FFTWMachine.fft(ModelImCorr)

        return ModelUVCorr
        
    def cutImPadded(self,Dirty):
        x0,x1=self.PaddingInnerCoord
        Dirty=Dirty[:,:,x0:x1,x0:x1]
        # if self.CasaImage!=None:
        #     self.CasaImage.im.putdata(Dirty[0,0].real)
        return Dirty
        

    def getDirtyIm(self):
        Dirty= self.GridToIm()
        x0,x1=self.PaddingInnerCoord
        Dirty=Dirty[:,:,x0:x1,x0:x1]
        # if self.CasaImage!=None:
        #     self.CasaImage.im.putdata(Dirty[0,0].real)
        return Dirty

    def GridToIm(self,Grid):
        #log=MyLogger.getLogger("ClassImager.GridToIm")

        npol=self.npol
        T=ClassTimeIt.ClassTimeIt("GridToIm")
        T.disable()

        if self.DoNormWeights:
            Grid/=self.SumWeigths.reshape((self.NChan,npol,1,1))

        Grid*=(self.WTerm.OverS)**2
        T.timeit("norm")
        Dirty=np.real(self.FFTWMachine.ifft(Grid))
        nchan,npol,_,_=Grid.shape
        del(Grid)
        #Dirty=GridCorr
        T.timeit("fft")
        for ichan in range(nchan):
            for ipol in range(npol):
                #Dirty[ichan,ipol][:,:]=Dirty[ichan,ipol][:,:]#.real
                if self.SpheNorm:
                    Dirty[ichan,ipol][:,:]/=self.ifzfCF

        T.timeit("sphenorm")

        return Dirty

        

