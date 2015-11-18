
import numpy as np
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassJones")
from DDFacet.Other import reformat
from DDFacet.Array import NpShared
import os
from DDFacet.Array import ModLinAlg

import ClassFITSBeam

class ClassJones():
    def __init__(self,GD,MS,FacetMachine=None,IdSharedMem=""):
        self.GD=GD
        self.FacetMachine=FacetMachine
        self.IdSharedMem=IdSharedMem
        self.MS=MS
        ThisMSName=reformat.reformat(os.path.abspath(self.MS.MSName),LastSlash=False)
        self.DicoClusterDirs=None
        self.JonesNormSolsFile_killMS="%s/JonesNorm_killMS.npz"%ThisMSName
        self.JonesNormSolsFile_Beam="%s/JonesNorm_Beam.npz"%ThisMSName




    def InitDDESols(self,DATA):
        GD=self.GD
        self.DATA=DATA
        SolsFile=GD["DDESolutions"]["DDSols"]
        self.ApplyCal=False
        if SolsFile!="":
            self.ApplyCal=True
            try:
                DicoSols,TimeMapping,DicoClusterDirs=self.DiskToSols(self.JonesNormSolsFile_killMS)
            except:
                DicoSols,TimeMapping,DicoClusterDirs=self.MakeSols("killMS")
            self.ToShared("killMS",DicoSols,TimeMapping,DicoClusterDirs)

        ApplyBeam=(GD["Beam"]["BeamModel"]!=None)
        if ApplyBeam:
            self.ApplyCal=True
            try:
                DicoSols,TimeMapping,DicoClusterDirs=self.DiskToSols(self.JonesNormSolsFile_Beam)
            except:
                DicoSols,TimeMapping,DicoClusterDirs=self.MakeSols("Beam")
            self.ToShared("Beam",DicoSols,TimeMapping,DicoClusterDirs)
            



    def ToShared(self,StrType,DicoSols,TimeMapping,DicoClusterDirs):
        print>>log, "  Putting %s Jones in shm"%StrType
        NpShared.DicoToShared("%sDicoClusterDirs_%s"%(self.IdSharedMem,StrType),DicoClusterDirs)
        NpShared.DicoToShared("%sJonesFile_%s"%(self.IdSharedMem,StrType),DicoSols)
        NpShared.ToShared("%sMapJones_%s"%(self.IdSharedMem,StrType),TimeMapping)

    def SolsToDisk(self,OutName,DicoSols,DicoClusterDirs,TimeMapping):
        
        print>>log, "  Saving %s"%OutName
        l=DicoClusterDirs["l"]
        m=DicoClusterDirs["m"]
        I=DicoClusterDirs["I"]
        Cluster=DicoClusterDirs["Cluster"]
        t0=DicoSols["t0"]
        t1=DicoSols["t1"]
        tm=DicoSols["tm"]
        Jones=DicoSols["Jones"]
        TimeMapping=TimeMapping

        #np.savez(self.JonesNorm_killMS,l=l,m=m,I=I,Cluster=Cluster,t0=t0,t1=t1,tm=tm,Jones=Jones,TimeMapping=TimeMapping)
        np.savez(OutName,l=l,m=m,I=I,Cluster=Cluster,t0=t0,t1=t1,tm=tm,Jones=Jones,TimeMapping=TimeMapping)

    def DiskToSols(self,InName):
        #SolsFile_killMS=np.load(self.JonesNorm_killMS)
        #print>>log, "  Loading %s"%InName
        SolsFile=np.load(InName)
        print>>log, "  %s loaded"%InName
        
        DicoClusterDirs={}
        DicoClusterDirs["l"]=SolsFile["l"]
        DicoClusterDirs["m"]=SolsFile["m"]
        DicoClusterDirs["I"]=SolsFile["I"]
        DicoClusterDirs["Cluster"]=SolsFile["Cluster"]
        DicoSols={}
        DicoSols["t0"]=SolsFile["t0"]
        DicoSols["t1"]=SolsFile["t1"]
        DicoSols["tm"]=SolsFile["tm"]
        DicoSols["Jones"]=SolsFile["Jones"]
        TimeMapping=SolsFile["TimeMapping"]
        return DicoSols,TimeMapping,DicoClusterDirs

    def MakeSols(self,StrType):
        GD=self.GD
        DicoSols_killMS=None
        TimeMapping_killMS=None
        DicoSols_Beam=None
        TimeMapping_Beam=None
        
        print>>log, "Build solution Dico for %s"%StrType
        
        if StrType=="killMS":
            DicoClusterDirs_killMS,DicoSols=self.GiveKillMSSols()
            DicoClusterDirs=DicoClusterDirs_killMS
            print>>log, "  Build VisTime-to-Solution mapping"
            TimeMapping=self.GiveTimeMapping(DicoSols)
            self.SolsToDisk(self.JonesNormSolsFile_killMS,DicoSols,DicoClusterDirs_killMS,TimeMapping)

        BeamJones=None
        if StrType=="Beam":
            print>>log,"  Getting Jones directions from Facets"

            if self.FacetMachine!=None:
                DicoImager=self.FacetMachine.DicoImager
                NFacets=len(DicoImager)
                self.ClusterCatBeam=self.FacetMachine.FacetCat
                DicoClusterDirs={}
                DicoClusterDirs["l"]=self.ClusterCatBeam.l
                DicoClusterDirs["m"]=self.ClusterCatBeam.m
                DicoClusterDirs["ra"]=self.ClusterCatBeam.ra
                DicoClusterDirs["dec"]=self.ClusterCatBeam.dec
                DicoClusterDirs["I"]=self.ClusterCatBeam.I
                DicoClusterDirs["Cluster"]=self.ClusterCatBeam.Cluster
            else:
                
                self.ClusterCatBeam=np.zeros((1,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('SumI',np.float),
                                                     ("Cluster",int),
                                                     ("l",np.float),("m",np.float),
                                                     ("I",np.float)])
                self.ClusterCatBeam=self.ClusterCatBeam.view(np.recarray)
                self.ClusterCatBeam.I=1
                self.ClusterCatBeam.SumI=1
                self.ClusterCatBeam.ra[0]=self.MS.rac
                self.ClusterCatBeam.dec[0]=self.MS.decc
                DicoClusterDirs={}
                DicoClusterDirs["l"]=np.array([0.],np.float32)
                DicoClusterDirs["m"]=np.array([0.],np.float32)
                DicoClusterDirs["ra"]=self.MS.rac
                DicoClusterDirs["dec"]=self.MS.decc
                DicoClusterDirs["I"]=np.array([1.],np.float32)
                DicoClusterDirs["Cluster"]=np.array([0],np.int32)
                
            DicoClusterDirs_Beam=DicoClusterDirs
            DicoSols=self.GiveBeam()
            print>>log, "  Build VisTime-to-Beam mapping"
            TimeMapping=self.GiveTimeMapping(DicoSols)
            self.SolsToDisk(self.JonesNormSolsFile_Beam,DicoSols,DicoClusterDirs_Beam,TimeMapping)

        # if (BeamJones!=None)&(KillMSSols!=None):
        #     print>>log,"  Merging killMS and Beam Jones matrices"
        #     DicoSols=self.MergeJones(KillMSSols,BeamJones)
        # elif BeamJones!=None:
        #     DicoSols=BeamJones
        # elif KillMSSols!=None:
        #     DicoSols=KillMSSols

        DicoSols["Jones"]=np.require(DicoSols["Jones"], dtype=np.complex64, requirements="C")

        # ThisMSName=reformat.reformat(os.path.abspath(self.CurrentMS.MSName),LastSlash=False)
        # TimeMapName="%s/Mapping.DDESolsTime.npy"%ThisMSName

        


        return DicoSols,TimeMapping,DicoClusterDirs


    def GiveTimeMapping(self,DicoSols):
        print>>log, "  Build Time Mapping"
        DicoJonesMatrices=DicoSols
        times=self.DATA["times"]
        ind=np.zeros((times.size,),np.int32)
        nt,na,nd,_,_,_=DicoJonesMatrices["Jones"].shape
        ii=0
        for it in range(nt):
            t0=DicoJonesMatrices["t0"][it]
            t1=DicoJonesMatrices["t1"][it]
            indMStime=np.where((times>=t0)&(times<t1))[0]
            indMStime=np.ones((indMStime.size,),np.int32)*it
            ind[ii:ii+indMStime.size]=indMStime[:]
            ii+=indMStime.size
        TimeMapping=ind
        return TimeMapping


    def GiveKillMSSols(self):
        GD=self.GD
        SolsFile=GD["DDESolutions"]["DDSols"]
        if type(SolsFile)==list:
            SolsFileList=SolsFile
        else:
            SolsFileList=[SolsFile]

        if GD["DDESolutions"]["GlobalNorm"]==None:
            GD["DDESolutions"]["GlobalNorm"]=""

        GlobalNormList=GD["DDESolutions"]["GlobalNorm"]
        if type(GlobalNormList)!=list:
            GlobalNormList=[GD["DDESolutions"]["GlobalNorm"]]*len(GD["DDESolutions"]["DDSols"])

        if GD["DDESolutions"]["JonesNormList"]==None:
            GD["DDESolutions"]["JonesNormList"]="AP"

        JonesNormList=GD["DDESolutions"]["JonesNormList"]
        if type(JonesNormList)!=list:
            JonesNormList=[GD["DDESolutions"]["JonesNormList"]]*len(GD["DDESolutions"]["DDSols"])
        



        ListDicoSols=[]


        for File,ThisGlobalMode,ThisJonesMode in zip(SolsFileList,GlobalNormList,JonesNormList):

            DicoClusterDirs,DicoSols=self.GiveKillMSSols_SingleFile(File,GlobalMode=ThisGlobalMode,JonesMode=ThisJonesMode)
            ListDicoSols.append(DicoSols)

        DicoJones=ListDicoSols[0]
        for DicoJones1 in ListDicoSols[1::]:
            DicoJones=self.MergeJones(DicoJones1,DicoJones)

        return DicoClusterDirs,DicoJones

    def GiveKillMSSols_SingleFile(self,SolsFile,JonesMode="AP",GlobalMode=""):

        print>>log, "  Loading solution file %s"%(SolsFile)
        if not(".npz" in SolsFile):
            Method=SolsFile
            ThisMSName=reformat.reformat(os.path.abspath(self.MS.MSName),LastSlash=False)
            SolsFile="%s/killMS.%s.sols.npz"%(ThisMSName,Method)
            
        self.ApplyCal=True
        DicoSolsFile=np.load(SolsFile)
        
        ClusterCat=DicoSolsFile["SkyModel"]
        ClusterCat=ClusterCat.view(np.recarray)
        self.ClusterCat=ClusterCat
        DicoClusterDirs={}
        DicoClusterDirs["l"]=ClusterCat.l
        DicoClusterDirs["m"]=ClusterCat.m
        #DicoClusterDirs["l"]=ClusterCat.l
        #DicoClusterDirs["m"]=ClusterCat.m
        DicoClusterDirs["I"]=ClusterCat.SumI
        DicoClusterDirs["Cluster"]=ClusterCat.Cluster
        
        Sols=DicoSolsFile["Sols"]
        Sols=Sols.view(np.recarray)
        DicoSols={}
        DicoSols["t0"]=Sols.t0
        DicoSols["t1"]=Sols.t1
        DicoSols["tm"]=(Sols.t1+Sols.t0)/2.
        nt,na,nd,_,_=Sols.G.shape
        G=np.swapaxes(Sols.G,1,2).reshape((nt,nd,na,1,2,2))
        if GlobalMode=="MeanAbs":
            print>>log, "  Normalising by the mean of the amplitude"
            gmean_abs=np.mean(np.abs(G[:,:,:,:,0,0]),axis=0)
            gmean_abs=gmean_abs.reshape((1,nd,na,1))
            G[:,:,:,:,0,0]/=gmean_abs
            G[:,:,:,:,1,1]/=gmean_abs
        
        if not("A" in JonesMode):
            print>>log, "  Normalising by the amplitude"
            G[G!=0.]/=np.abs(G[G!=0.])
        if not("P" in JonesMode):
            print>>log, "  Zero-ing the phases"
            dtype=G.dtype
            G=(np.abs(G).astype(dtype)).copy()

        G=self.NormDirMatrices(G)
        DicoSols["Jones"]=G

        return DicoClusterDirs,DicoSols

    def NormDirMatrices(self,G):
        RefAnt=0
        print>>log,"  Normalising Jones Matrices with reference Antenna %i ..."%RefAnt
        nt,nd,na,nf,_,_=G.shape
        for iDir in range(nd):
            for it in range(nt):
                for iF in range(nf):
                    Gt=G[it,iDir,:,iF,:,:]
                    u,s,v=np.linalg.svd(Gt[RefAnt])
                    U=np.dot(u,v)
                    for iAnt in range(0,na):
                        G[it,iDir,iAnt,iF,:,:]=np.dot(U.T.conj(),Gt[iAnt,:,:])

        return G


    #######################################################
    ######################## BEAM #########################
    #######################################################

    def GiveBeam(self):
        GD=self.GD
        DtBeamMin=GD["Beam"]["DtBeamMin"]
        self.DtBeamMin=DtBeamMin
        if GD["Beam"]["BeamModel"]=="LOFAR":
            self.InitLOFARBeam()
            LOFARBeamMode=GD["Beam"]["LOFARBeamMode"]
            print>>log, "  Estimating LOFAR beam model in %s mode every %5.1f min."%(LOFARBeamMode,DtBeamMin)
            self.GiveInstrumentBeam=self.MS.GiveBeam
        elif GD["Beam"]["BeamModel"]=="FITS":
            self.FITSBeam = ClassFITSBeam.ClassFITSBeam(self.MS,GD["Beam"])
            self.GiveInstrumentBeam = self.FITSBeam.evaluateBeam
            print>>log, "  Estimating FITS beam model every %5.1f min."%DtBeamMin

        RAs=self.ClusterCatBeam.ra
        DECs=self.ClusterCatBeam.dec
        t0=self.DATA["times"][0]
        t1=self.DATA["times"][-1]
        DicoBeam=self.EstimateBeam(t0,t1,RAs,DECs)
        return DicoBeam

    def InitLOFARBeam(self):
        GD=self.GD
        LOFARBeamMode=GD["Beam"]["LOFARBeamMode"]
        #self.BeamMode,self.DtBeamMin,self.BeamRAs,self.BeamDECs = LofarBeam
        useArrayFactor=("A" in LOFARBeamMode)
        useElementBeam=("E" in LOFARBeamMode)
        self.MS.LoadSR(useElementBeam=useElementBeam,useArrayFactor=useArrayFactor)
        self.ApplyBeam=True

    def EstimateBeam(self,t0,t1,RA,DEC):
        DtBeamSec=self.DtBeamMin*60
        tmin,tmax=t0,t1
        TimesBeam=np.arange(tmin,tmax,DtBeamSec).tolist()
        if not(tmax in TimesBeam): TimesBeam.append(tmax)
        TimesBeam=np.float64(np.array(TimesBeam))
        T0s=TimesBeam[:-1].copy()
        T1s=TimesBeam[1:].copy()
        Tm=(T0s+T1s)/2.
        #RA,DEC=self.BeamRAs,self.BeamDECs
        NDir=RA.size
        DicoBeam={}
        DicoBeam["Jones"]=np.zeros((Tm.size,NDir,self.MS.na,self.MS.NSPWChan,2,2),dtype=np.complex64)
        DicoBeam["t0"]=np.zeros((Tm.size,),np.float64)
        DicoBeam["t1"]=np.zeros((Tm.size,),np.float64)
        DicoBeam["tm"]=np.zeros((Tm.size,),np.float64)
        rac,decc=self.MS.radec
        for itime in range(Tm.size):
            DicoBeam["t0"][itime]=T0s[itime]
            DicoBeam["t1"][itime]=T1s[itime]
            DicoBeam["tm"][itime]=Tm[itime]
            ThisTime=Tm[itime]

            Beam=self.GiveInstrumentBeam(ThisTime,RA,DEC)

            if self.GD["Beam"]["CenterNorm"]==1:
                Beam0=self.GiveInstrumentBeam(ThisTime,np.array([rac]),np.array([decc]))
                Beam0inv=ModLinAlg.BatchInverse(Beam0)
                nd,_,_,_,_=Beam.shape
                Ones=np.ones((nd, 1, 1, 1, 1),np.float32)
                Beam0inv=Beam0inv*Ones
                Beam=ModLinAlg.BatchDot(Beam0inv,Beam)
                
 
            DicoBeam["Jones"][itime]=Beam

        nt,nd,na,nch,_,_= DicoBeam["Jones"].shape
        DicoBeam["Jones"]=np.mean(DicoBeam["Jones"],axis=3).reshape((nt,nd,na,1,2,2))
        del(self.MS.SR)
        # print TimesBeam-TimesBeam[0]
        # print t0-t1
        # print DicoBeam["t1"][-1]-DicoBeam["t0"][0]

        return DicoBeam

    def MergeJones(self,DicoJ0,DicoJ1):
        T0=DicoJ0["t0"][0]
        DicoOut={}
        DicoOut["t0"]=[]
        DicoOut["t1"]=[]
        DicoOut["tm"]=[]
        it=0
        CurrentT0=T0


        while True:
            DicoOut["t0"].append(CurrentT0)
            T0=DicoOut["t0"][it]

            dT0=DicoJ0["t1"]-T0
            dT0=dT0[dT0>0]
            dT1=DicoJ1["t1"]-T0
            dT1=dT1[dT1>0]
            if(dT0.size==0)&(dT1.size==0):
                break
            elif dT0.size==0:
                dT=dT1[0]
            elif dT1.size==0:
                dT=dT0[0]
            else:
                dT=np.min([dT0[0],dT1[0]])

            T1=T0+dT
            DicoOut["t1"].append(T1)
            Tm=(T0+T1)/2.
            DicoOut["tm"].append(Tm)
            CurrentT0=T1
            it+=1

        
        DicoOut["t0"]=np.array(DicoOut["t0"])
        DicoOut["t1"]=np.array(DicoOut["t1"])
        DicoOut["tm"]=np.array(DicoOut["tm"])

        _,nd,na,nch,_,_=DicoJ0["Jones"].shape
        nt=DicoOut["tm"].size
        DicoOut["Jones"]=np.zeros((nt,nd,na,1,2,2),np.complex64)

        nt0=DicoJ0["t0"].size
        nt1=DicoJ1["t0"].size

        iG0=np.argmin(np.abs(DicoOut["tm"].reshape((nt,1))-DicoJ0["tm"].reshape((1,nt0))),axis=1)
        iG1=np.argmin(np.abs(DicoOut["tm"].reshape((nt,1))-DicoJ1["tm"].reshape((1,nt1))),axis=1)
        

        for itime in range(nt):
            G0=DicoJ0["Jones"][iG0[itime]]
            G1=DicoJ1["Jones"][iG1[itime]]
            DicoOut["Jones"][itime]=ModLinAlg.BatchDot(G0,G1)
            

        return DicoOut


        



