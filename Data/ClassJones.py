
import numpy as np
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassJones")
from DDFacet.Other import reformat
from DDFacet.Array import NpShared
import os
from DDFacet.Array import ModLinAlg

class ClassJones():
    def __init__(self,GD,FacetMachine,MS,IdSharedMem=""):
        self.GD=GD
        self.FacetMachine=FacetMachine
        self.IdSharedMem=IdSharedMem
        self.MS=MS
        ThisMSName=reformat.reformat(os.path.abspath(self.MS.MSName),LastSlash=False)
        self.JonesNormSolsFile="%s/JonesNorm.npz"%ThisMSName

    def InitDDESols(self,DATA):
        GD=self.GD
        self.DATA=DATA
        SolsFile=GD["DDESolutions"]["DDSols"]
        ApplyBeam=(GD["Beam"]["BeamModel"]!=None)
        self.ApplyCal=False
        if (SolsFile!="")|(ApplyBeam):
            try:
                DicoSols,DicoClusterDirs,TimeMapping=self.DiskToSols()
            except:
                DicoSols,DicoClusterDirs,TimeMapping=self.MakeSols()

#            DicoSols,DicoClusterDirs,TimeMapping=self.MakeSols()

            self.ToShared(DicoSols,DicoClusterDirs,TimeMapping)

    def ToShared(self,DicoSols,DicoClusterDirs,TimeMapping):
        NpShared.DicoToShared("%sDicoClusterDirs"%self.IdSharedMem,DicoClusterDirs)
        NpShared.DicoToShared("%skillMSSolutionFile"%self.IdSharedMem,DicoSols)
        NpShared.ToShared("%sMapJones"%self.IdSharedMem,TimeMapping)

    def DiskToSols(self):
        SolsFile=np.load(self.JonesNormSolsFile)
        
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
        return DicoSols,DicoClusterDirs,TimeMapping

    def SolsToDisk(self,DicoSols,DicoClusterDirs,TimeMapping):
        
        l=DicoClusterDirs["l"]
        m=DicoClusterDirs["m"]
        I=DicoClusterDirs["I"]
        Cluster=DicoClusterDirs["Cluster"]
        t0=DicoSols["t0"]
        t1=DicoSols["t1"]
        tm=DicoSols["tm"]
        Jones=DicoSols["Jones"]
        TimeMapping=TimeMapping

        np.savez(self.JonesNormSolsFile,l=l,m=m,I=I,Cluster=Cluster,t0=t0,t1=t1,tm=tm,Jones=Jones,TimeMapping=TimeMapping)

    def MakeSols(self):
        GD=self.GD
        KillMSSols=None
        DicoClusterDirs=None
        if GD["DDESolutions"]["DDSols"]!="":
            DicoClusterDirs,KillMSSols=self.GiveKillMSSols()
            

        BeamJones=None
        if GD["Beam"]["BeamModel"]!=None:
            if DicoClusterDirs==None:
                print>>log,"  Getting Jones directions from Facets"
                DicoImager=self.FacetMachine.DicoImager
                NFacets=len(DicoImager)
                self.ClusterCat=self.FacetMachine.FacetCat
                DicoClusterDirs={}
                DicoClusterDirs["l"]=self.ClusterCat.l
                DicoClusterDirs["m"]=self.ClusterCat.m
                DicoClusterDirs["ra"]=self.ClusterCat.ra
                DicoClusterDirs["dec"]=self.ClusterCat.dec
                DicoClusterDirs["I"]=self.ClusterCat.I
                DicoClusterDirs["Cluster"]=self.ClusterCat.Cluster

            BeamJones=self.GiveBeam()

        if (BeamJones!=None)&(KillMSSols!=None):
            print>>log,"  Merging killMS and Beam Jones matrices"
            DicoSols=self.MergeJones(KillMSSols,BeamJones)
        elif BeamJones!=None:
            DicoSols=BeamJones
        elif KillMSSols!=None:
            DicoSols=KillMSSols

        
        DicoSols["Jones"]=np.require(DicoSols["Jones"], dtype=np.complex64, requirements="C")

        # ThisMSName=reformat.reformat(os.path.abspath(self.CurrentMS.MSName),LastSlash=False)
        # TimeMapName="%s/Mapping.DDESolsTime.npy"%ThisMSName

        print>>log, "  Build VisTime-to-solution mapping"
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


        print>>log, "Done"
        self.SolsToDisk(DicoSols,DicoClusterDirs,TimeMapping)


        return DicoSols,DicoClusterDirs,TimeMapping

    def GiveKillMSSols(self):
        GD=self.GD
        SolsFile=GD["DDESolutions"]["DDSols"]
        print>>log, "Loading solution file: %s"%SolsFile
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
        if GD["DDESolutions"]["GlobalNorm"]=="MeanAbs":
            print>>log, "  Normalising by the mean of the amplitude"
            gmean_abs=np.mean(np.abs(G[:,:,:,:,0,0]),axis=0)
            gmean_abs=gmean_abs.reshape((1,nd,na,1))
            G[:,:,:,:,0,0]/=gmean_abs
            G[:,:,:,:,1,1]/=gmean_abs
            
        DicoSols["Jones"]=G

        return DicoClusterDirs,DicoSols


    #######################################################
    ######################## BEAM #########################
    #######################################################

    def GiveBeam(self):
        GD=self.GD
        if GD["Beam"]["BeamModel"]=="LOFAR":
            self.InitLOFARBeam()
            DtBeamMin=GD["Beam"]["DtBeamMin"]
            self.DtBeamMin=DtBeamMin
            LOFARBeamMode=GD["Beam"]["LOFARBeamMode"]
            print>>log, "  Estimating LOFAR beam model in %s mode every %5.1f min."%(LOFARBeamMode,DtBeamMin)
            
            RAs=self.ClusterCat.ra
            DECs=self.ClusterCat.dec
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

        for itime in range(Tm.size):
            DicoBeam["t0"][itime]=T0s[itime]
            DicoBeam["t1"][itime]=T1s[itime]
            DicoBeam["tm"][itime]=Tm[itime]
            ThisTime=Tm[itime]
            DicoBeam["Jones"][itime]=self.MS.GiveBeam(ThisTime,RA,DEC)

        nt,nd,na,nch,_,_= DicoBeam["Jones"].shape
        DicoBeam["Jones"]=np.mean(DicoBeam["Jones"],axis=3).reshape((nt,nd,na,1,2,2))

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


        



