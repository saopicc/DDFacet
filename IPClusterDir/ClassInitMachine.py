import ClassJonesMachine
import numpy as np
import glob
import IPClusterDir.ClassIPClusterNew2 as ClassIPCluster
import IPClusterDir.ClassEngine as ClassEngine
import ModColor
import MyLogger
import socket
import ClassSM
import ClassMS
from ToolsDir import ModParset
import MyPickle
import ModColor
from IPClusterDir.ClassClusterInterface import ClassClusterInterface
from ToolsDir import CatToFreqs
#import ClassDistributedH
import os
#HYPERCAL_DIR=os.environ["HYPERCAL_DIR"]
#execfile("%s/HyperCal/GetScriptName.py"%HYPERCAL_DIR)#; Name=
log=MyLogger.getLogger("ClassInitMachine")#"SetDico")

ParsetFile="ParsetNew.txt"
from ClassData import ClassMultiPointingData,ClassSinglePointingData,ClassGlobalData
from ClassME import MeasurementEquation
import MyPickle
#import ClassIterateMachine

def GiveMDC():
    ParsetFile="ParsetNew.txt"
    MS0=ClassMS.ClassMS("/media/tasse/data/MS/SimulTec/many/0000.MS/")
    SM0=ClassSM.ClassSM("/media/tasse/data/HyperCal2/test/ModelRandom0.txt.npy")
    MS1=ClassMS.ClassMS("/media/tasse/data/MS/SimulTec/many/0000.MS/")
    SM1=ClassSM.ClassSM("/media/tasse/data/HyperCal2/test/ModelRandom1.txt.npy")
    freqs=np.linspace(100e6,200e6,10)
    MS0.DelData()
    MS1.DelData()

    GD=ClassGlobalData(ParsetFile)
    MDC=ClassMultiPointingData(GD)

    MS=[MS0,MS1]
    SM=[SM0,SM1]

    for ID in range(2):
        MDC.setMS(MS[ID],PointingID=ID)
        MDC.setSM(SM[ID],PointingID=ID)
        MDC.setFreqs(freqs,PointingID=ID)
        MDC.setMappingBL(PointingID=ID)
    MME=MeasurementEquation()
    HYPERCAL_DIR=os.environ["HYPERCAL_DIR"]
    execfile("%s/HyperCal/Scripts/ScriptSetMultiRIME.py"%HYPERCAL_DIR)
    return MDC,GD,MME

def GiveInitMachine(ParsetFile):
    #ParsetFile="ParsetNew.txt"
    GD=ClassGlobalData(ParsetFile)
    InitMachine=ClassInitMachine(GD)
    InitMachine.InitCluster()

    return InitMachine

def test():
 
    # MDC,GD,MME=GiveMDC()
    ParsetFile="ParsetNew.txt"
    GD=ClassGlobalData(ParsetFile)
    InitMachine=ClassInitMachine(GD)
    InitMachine.InitCluster()

    return InitMachine
    #InitMachine.GetCluster()
    #InitMachine.C.VAll["ParsetFile"]=ParsetFile
    #InitMachine.C.BuildRemoteRIME(MDC)

#    SetMachine.updateXpParset(ParsetFile)
#    SetMachine.MountSolsSimul()

def IterMachine():
    IM=test()
    IterMachine=ClassIterateMachine.ClassIterateMachine(IM)
    IterMachine.setXDataTime((0,1))
    return IterMachine

def testIonKrig(IterMachine):
    IM=IterMachine.IM
    JM=IM.H.MME.giveJM(0)
    T=JM.DicoJones["TEC"]["Jones"].DicoModelTerm[0]
    Xp=IterMachine.IM.Xp


    # xpixS,ypixS,predict2=T.Krigging(IM.Xp.ToVec(["TEC"]))
    # Xk,Yk=T.XY_Plane
    # TECk=T.TECk
    # extent=(xpixS.min(),xpixS.max(),ypixS.min(),ypixS.max())

    #####
    i0,i1=Xp.giveIndxPar("TEC",NoHidePar=True)
    MDC=IM.MDC
    TrueSols=MDC.getCurrentData("SimulSols")['Sols']
    xi=TrueSols.SolsCat.xi[0][i0:i1]
    ShIon=np.sqrt(i1-i0)
    coefSim=xi[i0:i1]
    #S_TECk=self.I.GiveTEC(coefSim)
    S_xpixS,S_ypixS,S_TEC=T.Krigging(coefSim)
    S_TECk=T.TECk
    
    
    
    coef=Xp.ToVec(["TEC"])
    #TECk=self.I.GiveTEC(coef)
    xpixS,ypixS,TEC=T.Krigging(coef)
    Xk,Yk=T.XY_Plane
    TECk=T.TECk

    S_TEC_0=np.median(S_TEC)
    S_TEC-=S_TEC_0
    S_TECk-=S_TEC_0
    TEC_0=np.median(TEC)
    TEC-=TEC_0
    TECk-=TEC_0
    
    xlims,ylims=(np.min(xpixS),np.max(xpixS)),(np.min(ypixS),np.max(ypixS))
    
    tec0=S_TEC
    tec=TEC
    vmin,vmax=np.min(tec0),np.max(tec0)
    
    #vmin,vmax=-0.1,0.1
    
    extent=(xpixS.min(),xpixS.max(),ypixS.min(),ypixS.max())
    
    lp,mp=Xk,Yk



    import pylab
    pylab.clf()
    pylab.imshow(TEC.T[::-1,:],extent=extent)
    pylab.scatter(Xk,Yk,c=TECk)
    pylab.draw()
    pylab.show(False)


def testIterate(IM):
    import ClassIterateMachine
    Sols=IM.MDC.getCurrentData("SimulSols")["Sols"]
    x=Sols.SolsCat.xi[0]
    IM.Xp.FromVec(x)
    IterMachine=ClassIterateMachine.ClassIterateMachine(IM)
    
    IterMachine.setXDataTime((0,1))


    #testPredict
    it0,it1=IM.MDC.getCurrentData("itimes")
    D0bef=IM.H.LocalH.HR(x,it0,it1)
    D1bef=IM.H.DistributedHR([x],it0,it1)
    import pylab
    n=101
    pylab.ion()
    pylab.clf()

    pylab.plot(D0bef[0::n])
    pylab.plot(D1bef[0::n],ls="--",lw=2)
    zMS=IM.MDC.getCurrentData("zMS")
    pylab.plot(zMS[0::n])
    pylab.ylim(-200,200)
    pylab.pause(0.1)
    pylab.show(False)

    JM=IM.H.MME.giveJM(0)
    I=JM.DicoJones["TEC"]["Jones"]
    T=JM.DicoJones["TEC"]["Jones"].DicoModelTerm[0]

    # DpSimul=np.load("ionmat/Dp.1409821636.25.npy")
    # pylab.clf()
    # pylab.subplot(1,2,1)
    # pylab.imshow(DpSimul,interpolation="nearest")
    # pylab.subplot(1,2,2)
    # pylab.imshow(T.Dp,interpolation="nearest")
    # pylab.draw()
    # pylab.pause(0.1)
    # pylab.show(False)


    stop


def testNoiseMachine(IM):
    import ClassNoiseMachine
    stdin=1
    tbin=1
    IM.MDC.setCurrentData("itimes",(0,1))
    IM.MDC.setCurrentData("rmsData",stdin)
    NM=ClassNoiseMachine.ClassNoiseMachine(IM)
    #res=NM.GetReprNoise()

def testPredictDistSimul(IM):
    x=IM.Xp.ToVec()
    it0,it1=0,1




    DicoData=IM.CI.GiveDataChunk(it0,it1)
    # ###########
    D1=IM.H.DistributedHR([x],it0,it1)
    RandomMapRepr=np.random.rand(*(D1.flatten().shape))>0.9
    IM.H.setDistributedRandomMapRepr(RandomMapRepr)


    # test PredictDistSimul
    D0bef=IM.H.LocalH.HR(x,it0,it1)
    D1bef=IM.H.DistributedHR([x],it0,it1)
    for i in range(IM.CI.NPointings):
        flags=DicoData[i]["flags"].copy()
        DicoData[i]["flags"]=np.random.rand(*(flags.shape))>0.5
    IM.H.setDistributedCurrentFlags(DicoData)
    D0=IM.H.LocalH.HR(x,it0,it1)
    D1=IM.H.DistributedHR([x],it0,it1)

    print np.allclose(D0bef.flatten(),D1bef.flatten())
    print np.allclose(D0.flatten(),D1.flatten())
    # 

    # Test ApplyJones
    #import ClassApplyJones
    #AJ=ClassApplyJones.ClassApplyJones(IM.H.MME)
    #DicoDataOut=AJ.ApplyJones(DicoData,Code="Left.inv,Right.inv")
    # # 




    stop

class ClassInitMachine():
    def __init__(self,GD):
        self.GD=GD
        self.DicoConfig=GD.DicoConfig
        self.E=None
        self.SelSubFreqCat()

    def InitCluster(self,Mode="KAFCA"):
        self.StartClusterInterface()
        #print "1"
        #self.CI.E.clear()
        self.BuildMDC()
        #print "2"
        #self.CI.E.clear()


        if Mode=="KAFCA":
            self.BuildFullCorrMME()
            self.BuildDistributedH()
        
            # update MaskHDF5
            self.CI.SetMaskHDF5(self.Xp)

            FileSimulSols=self.GD.DicoConfig["Files"]["Simul"]["FileSols"]
            if FileSimulSols!=None:
                SimulSols=MyPickle.Load(FileSimulSols)
                self.MDC.setCurrentData("SimulSols",SimulSols)
                self.MDC.setCurrentData("SimulMode",True)

            

    def BuildFullCorrMME(self):
        MME=MeasurementEquation(NameParsetME="RIME_FullCorr")
        MDC=self.MDC
        GD=self.GD
        HYPERCAL_DIR=GD.HYPERCAL_DIR

        execfile("%s/HyperCal/Scripts/ScriptSetMultiRIME.py"%HYPERCAL_DIR)
        self.FullCorrMME=MME
        #self.LocalH=ClassHyperH.ClassHyperH(MME,GD)

    def SelSubFreqCat(self):

        DicoConfig=self.DicoConfig
        ll=DicoConfig["Files"]["FileMSCat"]["Name"]
        chStart,chEnd,chIncr=DicoConfig["Files"]["FileMSCat"]["StartEndNf"]
        fmin,fmax,nf=float(chStart),float(chEnd),int(chIncr)

        CondCat=False
        for f in ll:
            if ".npy" in f: CondCat=True

        LSM=[]
        LCat=[]

        self.LCatMSFull=None
        if not(CondCat):
            ColName=DicoConfig["Files"]["ColName"]
            #nodes=["igor"]*len(ll)
            nodes=[self.GD.LocalHostName]*len(ll)
            
            Cat=np.zeros((len(ll),),dtype=[("node","|S200"),("dirMSname","|S200"),("PointingID",np.int64)])
            Cat=Cat.view(np.recarray)
            Cat.node=np.array(nodes)
            Cat.dirMSname=np.array(ll)
            LCat.append(Cat)
            if "FileSourceCat" in DicoConfig["Files"].keys():
                SkyModel=DicoConfig["Files"]["FileSourceCat"]
                LSM.append(SkyModel)
            self.NPointings=len(DicoConfig["Files"]["FileMSCat"]["Name"])
        else:
            self.LCatMSFull=[]
            self.NPointings=len(DicoConfig["Files"]["FileMSCat"]["Name"])
            for SkyModel,CatName,PointingID in zip(DicoConfig["Files"]["FileSourceCat"],DicoConfig["Files"]["FileMSCat"]["Name"],range(self.NPointings)):

                ColName=DicoConfig["Files"]["ColName"]
                Cat=np.load(CatName).copy()
                Cat=Cat.view(np.recarray)
                self.LCatMSFull.append(Cat.copy())
                if "Freq" in Cat.dtype.fields.keys():
                    Cat=CatToFreqs.CatToFreqs(Cat,fmin,fmax,nf)
                import numpy.lib.recfunctions as rec
                if not("PointingID" in Cat.dtype.fields.keys()):
                    Cat=rec.append_fields(Cat,'PointingID',PointingID*np.ones((Cat.shape[0],),dtype=np.int64),usemask=False,asrecarray=True)
                LCat.append(Cat)
                LSM.append(SkyModel)


        # np.save("Cat.npy",Cat)
        # stop
        # get a properly spaced catalog with holes for missing sb
        self.GD.LCatMSNodes=LCat
        self.GD.CatMSNodes=np.concatenate(LCat)
        self.GD.CatMSNodes=self.GD.CatMSNodes.view(np.recarray)
        self.GD.LSkyModelNames=LSM
        self.ColName=ColName



    def StartClusterInterface(self):
        self.E=ClassEngine.Engine(self.GD)
        self.CI=ClassClusterInterface(self.E)
        self.CI.E.InitProberCPU()
        self.CI.E.GiveLoads()
        self.CI.LoadRemoteMS()

    def BuildMDC(self):

        GD=self.GD
        NPointing=self.NPointings
        if "FileSourceCat" in GD.DicoConfig["Files"].keys():
            ListSM=GD.DicoConfig["Files"]["FileSourceCat"]
            ThereIsSM=True
        else:
            ListSM=[None for i in range(NPointing)]
            ThereIsSM=False

        MDC=ClassMultiPointingData(GD)
        for ID in range(self.NPointings):
            DMS=self.CI.GiveDMS(ID)
            MDC.setMS(DMS.MS,PointingID=ID)
            if ThereIsSM:
                SM=ClassSM.ClassSM(self.GD.LSkyModelNames[ID])
                SM.AppendRefSource((DMS.MS.rac,DMS.MS.decc))
                MDC.setSM(SM,PointingID=ID)
            freqs=DMS.freqs
            MDC.setFreqs(freqs,PointingID=ID)
            MDC.setMappingBL(PointingID=ID)
        #print "1a"
        #self.CI.E.clear()
        if ThereIsSM: self.CI.LoadRemoteSM(MDC)
        #print "1b"
        #self.CI.E.clear()
        self.CI.AttachMDC_dMSLocal(MDC)
        #print "1c"
        #self.CI.E.clear()
        self.MDC=MDC


    def BuildDistributedH(self):
        # Build both (Local/Remote MME) and (Local/Remote H)
        self.H=ClassDistributedH.ClassDistributedH(self.E,self.MDC)
        self.Xp=self.H.MME.Xp







        


