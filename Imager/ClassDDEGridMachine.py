import numpy as np
from Gridder import _pyGridder
#import pylab
from pyrap.images import image
import MyPickle
import os
import MyLogger
import ClassTimeIt
import ModCF
import ModToolBox

import ModColor
import ClassCasaImage
import ClassApplyJones
import ToolsDir.GiveMDC
from ModToolBox import EstimateNpix
from Array import ModLinAlg
import copy
import time

import ClassApplyJones
from ClassME import MeasurementEquation
import MyLogger
log=MyLogger.getLogger("ClassDDEGridMachine")

import ToolsDir
import pylab
import ClassData

def GiveGM():
    GD=ClassData.ClassGlobalData("ParsetDDFacet.txt")
    MDC,GD=ToolsDir.GiveMDC.GiveMDC(GD=GD)
    MS=MDC.giveMS(0)
    MS.ReadData()
    GM=ClassDDEGridMachine(GD,DoDDE=False,WProj=True,lmShift=(0.,0.),JonesDir=3)
    return GM

import ClassTimeIt

def testGrid():
    GD=ClassData.ClassGlobalData("ParsetDDFacet.txt")
    MDC,GD=ToolsDir.GiveMDC.GiveMDC(GD=GD)
    MS=MDC.giveMS(0)
    MS.ReadData()

    GM=ClassDDEGridMachine(GD,DoDDE=False,WProj=True,lmShift=(0.,0.),JonesDir=3,SpheNorm=True)


    row0,row1=0,-1

    uvw=np.float64(MS.uvw)[row0:row1]
    times=np.float64(MS.times_all)[row0:row1]
    data=np.complex64(MS.data)[row0:row1]
    data.fill(1.)
    data[:,:,0]=1
    data[:,:,3]=1
    A0=np.int32(MS.A0)[row0:row1]
    A1=np.int32(MS.A1)[row0:row1]
    
    #uvw.fill(0)
    
    flag=np.bool8(MS.flag_all)[row0:row1,:,:].copy()
    flag.fill(0)
    # print uvw
    # print data
    # print flag

    print
    print "================"
    print

    # #############################
    # nt,nd,na=10,1,MS.na
    # JonesMatrices=np.random.randn(nt,nd,na,1,2,2)+1j*np.random.randn(nt,nd,na,1,2,2)
    # # JonesMatrices.fill(0)
    # # JonesMatrices[:,:,:,:,0,0]=1
    # # JonesMatrices[:,:,:,:,1,1]=1
    # tt=np.linspace(times.min(),times.max(),nt+1)
    # t0=tt[0:-1]
    # t1=tt[1::]
    # DicoJonesMatrices={}
    # DicoJonesMatrices["t0"]=t0
    # DicoJonesMatrices["t1"]=t1
    # DicoJonesMatrices["Jones"]=np.complex64(JonesMatrices)
    
    # DicoClusterDirs={}
    # DicoClusterDirs["l"]=np.array([0.])
    # DicoClusterDirs["m"]=np.array([0.])
    # DicoClusterDirs["I"]=np.array([1.])
    # DicoClusterDirs["Cluster"]=np.array([0.])
    # DicoJonesMatrices["DicoClusterDirs"]=DicoClusterDirs
    # ##############################

    # # SolsFile="/media/6B5E-87D0/killMS2/TEST/Simul/Simul.npz"

    # # DicoSolsFile=np.load(SolsFile)
    # # DicoSols={}
    # # DicoSols["t0"]=DicoSolsFile["Sols"]["t0"]
    # # DicoSols["t1"]=DicoSolsFile["Sols"]["t1"]
    # # nt,na,nd,_,_=DicoSolsFile["Sols"]["G"].shape
    # # G=np.swapaxes(DicoSolsFile["Sols"]["G"],1,2).reshape((nt,nd,na,1,2,2))
    # # G.fill(0)
    # # G[:,:,:,:,0,0]=1
    # # G[:,:,:,:,1,1]=1
    # # DicoSols["Jones"]=G
    
    # # ClusterCat=DicoSolsFile["ClusterCat"]
    # # ClusterCat=ClusterCat.view(np.recarray)
    # # DicoClusterDirs={}
    # # DicoClusterDirs["l"]=ClusterCat.l
    # # DicoClusterDirs["m"]=ClusterCat.m
    # # DicoClusterDirs["I"]=ClusterCat.SumI
    # # DicoClusterDirs["Cluster"]=ClusterCat.Cluster

    # # DicoSols["DicoClusterDirs"]=DicoClusterDirs
    # # DicoJonesMatrices=DicoSols
    # # # return DicoJonesMatrices

    # # import NpShared
    # # DicoJonesMatrices=NpShared.SharedToDico("killMSSolutionFile")
    # # DicoClusterDirs=NpShared.SharedToDico("DicoClusterDirs")
    # # DicoJonesMatrices["DicoClusterDirs"]=DicoClusterDirs
    
 


    T=ClassTimeIt.ClassTimeIt("main")
    Grid=GM.put(times,uvw,data,flag,(A0,A1),W=None,PointingID=0,DoNormWeights=True)#, DicoJonesMatrices=DicoJonesMatrices)
    T.timeit("grid")



    # Grid.fill(0)
    # _,_,n,n=Grid.shape
    # Grid[:,:,n/4,n/5]=1
    # data.fill(0)
    # data=GM.get(times,uvw,data,flag,(A0,A1),Grid, DicoJonesMatrices=DicoJonesMatrices)
    # Grid=GM.put(times,uvw,data,flag,(A0,A1),W=None,PointingID=0,DoNormWeights=True, DicoJonesMatrices=DicoJonesMatrices)
    # T.timeit("degrid")
    # import pylab
    pylab.clf()
    pylab.imshow(np.real(Grid[0,0]))
    pylab.draw()
    pylab.show(False)




class ClassDDEGridMachine():
    def __init__(self,GD,
                 Npix=1023,Cell=10.,Support=11,
                 ChanFreq=np.array([6.23047e7],dtype=np.float64),
                 wmax=10000,Nw=11,DoPSF=False,
                 RaDec=None,ImageName="Image",OverS=5,
                 Padding=1.4,WProj=False,lmShift=None,Precision="S",PolMode="I",DoDDE=True,
                 JonesDir=None,
                 IdSharedMem="",
                 IDFacet=0,
                 SpheNorm=True):

        self.GD=GD
        self.IDFacet=IDFacet
        self.SpheNorm=SpheNorm
        self.DoDDE=DoDDE
        self.JonesDir=JonesDir
        self.IdSharedMem=IdSharedMem

        #self.DoPSF=DoPSF
        self.DoPSF=False
        # if DoPSF:
        #     self.DoPSF=True
        #     Npix=Npix*2

        if Precision=="S":
            self.dtype=np.complex64
        elif Precision=="D":
            self.dtype=np.complex128

        self.dtype=np.complex64
        self.ImageName=ImageName

        
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



        
        T=ClassTimeIt.ClassTimeIt("ClassImager")
        T.disable()

        self.Cell=Cell
        self.incr=(np.array([-Cell,Cell],dtype=np.float64)/3600.)*(np.pi/180)
        #CF.fill(1.)
        ChanFreq=ChanFreq.flatten()
        self.ChanFreq=ChanFreq
        self.ChanWave=2.99792458e8/self.ChanFreq
        self.UVNorm=2.*1j*np.pi/self.ChanWave
        self.UVNorm.reshape(1,self.UVNorm.size)
        self.Sup=Support
        self.WProj=WProj
        self.wmax=wmax
        self.Nw=Nw
        self.OverS=OverS
        self.lmShift=lmShift

        self.CalcCF()

        self.reinitGrid()
        self.CasaImage=None
        self.DicoATerm=None

    def CalcCF(self):
        Grid=np.zeros(self.GridShape,dtype=self.dtype)
        #self.FFTWMachine=ModFFTW.FFTW_2Donly(Grid, ncores = 1)
        import ModFFTW
        #self.FFTWMachine=ModFFTW.FFTW_2Donly_np(Grid, ncores = 1)
        #self.FFTWMachine=ModFFTW.FFTW_2Donly_np(Grid, ncores = 1)

        #self.FFTWMachine=ModFFTW.FFTW_2Donly_np(self.GridShape,self.dtype, ncores = 1)

        SharedName="%sFFTW.%i"%(self.IdSharedMem,self.IDFacet)
        self.FFTWMachine=ModFFTW.FFTW_2Donly(self.GridShape,self.dtype, ncores = 1, FromSharedId=SharedName)

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


    def setCasaImage(self):
        self.CasaImage=ClassCasaImage.ClassCasaimage(self.ImageName,self.NonPaddedNpix,self.Cell,self.radec1)

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
        
        #lc,mc=np.random.randn(100)*np.pi/180,np.random.randn(100)*np.pi/180
        
        
        InterpMode=self.GD["DDESolutions"]["Type"]
        d0=self.GD["DDESolutions"]["Scale"]*np.pi/180
        gamma=self.GD["DDESolutions"]["gamma"]
        
        d=np.sqrt((l0-lc)**2+(m0-mc)**2)
        idir=np.argmin(d)
        w=1./(1.+d/d0)**gamma
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


    def put(self,times,uvw,visIn,flag,A0A1,W=None,PointingID=0,DoNormWeights=True,DicoJonesMatrices=None):#,doStack=False):
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
            if "A" in self.GD["DDESolutions"]["DDModeGrid"]:
                ApplyAmp=1
            if "P" in self.GD["DDESolutions"]["DDModeGrid"]:
                ApplyPhase=1
            LApplySol=[ApplyAmp,ApplyPhase]
            ParamJonesList=self.GiveParamJonesList(DicoJonesMatrices,times,A0,A1,uvw)
            ParamJonesList=ParamJonesList+LApplySol


        T.timeit("3")
        #print "sleeping DDE..."; time.sleep(5)



        
        T2=ClassTimeIt.ClassTimeIt("Gridder")
        T2.disable()

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
                                      self.ChanFreq.astype(np.float64),
                                      [self.PolMap,FacetInfos],
                                      ParamJonesList) # Input the jones matrices

        T2.timeit("gridder")
        # print SumWeigths
        # return
        # del(Grid)
        T.timeit("4 (grid)")
        ImPadded= self.GridToIm(Grid)
        #print "sleeping DDE... %i"%isleep; time.sleep(5); isleep+=1
        Dirty =ImPadded
        if self.SpheNorm:
            Dirty = self.cutImPadded(ImPadded)

        #print "sleeping DDE... %i"%isleep; time.sleep(5); isleep+=1
        T.timeit("5")
        # Grid[:,:,:,:]=Grid.real
        # import pylab
        # pylab.clf()
        # pylab.imshow(np.abs(Grid[0,0]))
        # pylab.draw()
        # pylab.show(False)
        # stop

        return Dirty

    def CheckTypes(self,Grid=None,vis=None,uvw=None,flag=None,ListWTerm=None,W=None,A0=None,A1=None,Jones=None):
        if Grid!=None:
            if not(Grid.dtype==np.complex64):
                raise NameError('Grid.dtype %s %s'%(str(Grid.dtype),str(self.dtype)))
        if vis!=None:
            if not(vis.dtype==np.complex64):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
        if uvw!=None:
            if not(uvw.dtype==np.float64):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
        if flag!=None:
            if not(flag.dtype==np.bool8):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
        if ListWTerm!=None:
            if not(ListWTerm[0].dtype==np.complex64):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
        if W!=None:
            if not(W.dtype==np.float64):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
        if A0!=None:
            if not(A0.dtype==np.int32):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
        if A1!=None:
            if not(A1.dtype==np.int32):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))
        if Jones!=None:
            if not(Jones.dtype==np.complex64):
                raise NameError('Grid.dtype %s'%(str(Grid.dtype)))


    def get(self,times,uvw,visIn,flag,A0A1,ModelImage,PointingID=0,Row0Row1=(0,-1),DicoJonesMatrices=None):
        #log=MyLogger.getLogger("ClassImager.addChunk")
        T=ClassTimeIt.ClassTimeIt("get")
        T.disable()
        vis=visIn#.copy()



        LTimes=sorted(list(set(times.tolist())))
        NTimes=len(LTimes)
        A0,A1=A0A1

        Grid=self.dtype(self.setModelIm(ModelImage))
        np.save("Grid",Grid)
        
        if np.max(np.abs(Grid))==0: return vis
        T.timeit("1")
        #dummy=np.abs(vis).astype(np.float32)




        npol=self.npol
        NChan=self.NChan
        SumWeigths=self.SumWeigths
        if vis.shape!=flag.shape:
            raise Exception('vis[%s] and flag[%s] should have the same shape'%(str(vis.shape),str(flag.shape)))

        
        u,v,w=uvw.T
        vis[u==0,:,:]=0
        flag[u==0,:,:]=True
      
        uvwOrig=uvw.copy()
        
        # uvw,vis=self.ShiftVis(uvw,vis,reverse=False)
        
        # vis.fill(0)
        
        l0,m0=self.lmShift
        FacetInfos=np.float64(np.array([self.WTerm.Cu,self.WTerm.Cv,l0,m0]))
        Row0,Row1=Row0Row1
        if Row1==-1:
            Row1=u.shape[0]
        RowInfos=np.array([Row0,Row1]).astype(np.int32)

        T.timeit("2")
            
        self.CheckTypes(Grid=Grid,vis=vis,uvw=uvw,flag=flag,ListWTerm=self.WTerm.Wplanes)

        ParamJonesList=[]
        if DicoJonesMatrices!=None:
            ApplyAmp=0
            ApplyPhase=0
            if "A" in self.GD["DDESolutions"]["DDModeDeGrid"]:
                ApplyAmp=1
            if "P" in self.GD["DDESolutions"]["DDModeDeGrid"]:
                ApplyPhase=1
            LApplySol=[ApplyAmp,ApplyPhase]
            ParamJonesList=self.GiveParamJonesList(DicoJonesMatrices,times,A0,A1,uvw)
            ParamJonesList=ParamJonesList+LApplySol


        T.timeit("3")
        #print vis
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
                                       self.ChanFreq.astype(np.float64),
                                       [self.PolMap,FacetInfos,RowInfos],
                                       ParamJonesList)

        T.timeit("4 (degrid)")
        #print vis
        
        # uvw,vis=self.ShiftVis(uvwOrig,vis,reverse=False)
        
        if self.DoDDE:
            for ThisTime,itime0 in zip(LTimes,range(NTimes)):
                Jones,JonesH=self.DicoATerm[ThisTime]
                indThisTime=np.where(times==ThisTime)[0]
                ThisA0=A0[indThisTime]
                ThisA1=A1[indThisTime]
                P0=ModLinAlg.BatchDot(Jones[ThisA0,:,:],vis[indThisTime])
                vis[indThisTime]=ModLinAlg.BatchDot(P0,JonesH[ThisA1,:,:])
            #vis*=self.norm

        T.timeit("5")
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
        import ClassTimeIt
        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        GridCorr=Grid

        if self.DoNormWeights:
            GridCorr=Grid/self.SumWeigths.reshape((self.NChan,npol,1,1))

        GridCorr*=(self.WTerm.OverS)**2
        T.timeit("norm")
        Dirty=self.FFTWMachine.ifft(GridCorr)
        #Dirty=GridCorr
        T.timeit("fft")
        nchan,npol,_,_=Grid.shape
        for ichan in range(nchan):
            for ipol in range(npol):
                Dirty[ichan,ipol][:,:]=Dirty[ichan,ipol][:,:].real
                if self.SpheNorm:
                    Dirty[ichan,ipol][:,:]/=self.ifzfCF

        T.timeit("sphenorm")

        return Dirty

        
    def ImToGrid(self,ModelIm):
        
        npol=self.npol
        ModelImCorr=ModelIm*(self.WTerm.OverS*self.Padding)**2

        nchan,npol,_,_=ModelImCorr.shape
        for ichan in range(nchan):
            for ipol in range(npol):
                ModelImCorr[ichan,ipol][:,:]=ModelImCorr[ichan,ipol][:,:].real/self.ifzfCF


        ModelUVCorr=self.FFTWMachine.fft(ModelImCorr)

        return ModelUVCorr
        

