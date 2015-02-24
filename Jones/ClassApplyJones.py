import ModColor
import numpy as np
import ModLinAlg

class ClassApplyJones():
    def __init__(self,MME):
        self.MME=MME
        self.MDC=MME.Xp.MDC
        self.DicoNormJones={}
        self.MMEConfig=MME.Xp.GD.DicoConfig[MME.NameParsetME]
        for PointingID in self.MDC.ListID:
            self.DicoNormJones[PointingID]={}
            ListDo=["Left","Right"]

            for Desc in ListDo:
                for inv in ["inv","noinv"]:
                    Description="%s.%s"%(Desc,inv)
                    self.DicoNormJones[PointingID][Description]={}
                    self.DicoNormJones[PointingID][Description]["time"]=0
                
            if "LeftRightTerm" in self.MMEConfig.keys():
                self.LeftRightTerm=self.MMEConfig["LeftRightTerm"]
                if self.LeftRightTerm!=False:
                    
                    self.DicoNormJones[PointingID][self.LeftRightTerm]={}
                    self.DicoNormJones[PointingID][self.LeftRightTerm]["time"]=0

        self.radec=None

    def CheckKeyDico(self,key):
        for PointingID in self.MDC.ListID:
            if not(key in self.DicoNormJones[PointingID].keys()):
                self.DicoNormJones[PointingID][key]={}
                self.DicoNormJones[PointingID][key]["time"]=0

    def setRaDec(self,ra,dec):
        self.radec=(ra,dec)


    def BuildNormJones(self,Description="Left.inv",itimes=(0,1),SkipJones="",ForceUpdate=False):

        for PointingID in self.MDC.ListID:
            DC=self.MDC.giveSinglePointingData(PointingID)
            JM=self.MME.giveJM(PointingID)
            SMachine=self.MME.giveSM(PointingID)
            lms,radecsa=SMachine.GetSky_RA_LM()
            ClusterCat=DC.SM.ClusterCat
            RAJones=radecsa[:,0]
            DECJones=radecsa[:,1]
            RACluster,DECCluster,SumI=ClusterCat.ra,ClusterCat.dec,ClusterCat.SumI
            it0,it1=itimes
            Row0,Row1=it0*DC.MS.nbl,it1*DC.MS.nbl
            tt=DC.MS.times_all[Row0:Row1]


            indp=0

            ThisTime=np.mean(tt)

            if (self.DicoNormJones[PointingID][Description]["time"]==ThisTime)&(not(ForceUpdate)):
                return

            # print ModColor.Str("%s: Build NormJones for %s, itimes=%s, PointingID=%s"%(self.MME.NameParsetME,Description,str(itimes),str(PointingID)))
            Desc,inv=Description.split(".")

            Jones=JM.GetJones(tt,DC.freqs,Descriptive=Desc,SkipJones=SkipJones,radec=self.radec)[0] # take the Jones Matrix in the 0th direction (reference)
            

            Jones=np.swapaxes(Jones,0,1)
            JonesH=ModLinAlg.BatchH(Jones)

            if inv=="inv":
                Jones=ModLinAlg.BatchInverse(Jones)
                JonesH=ModLinAlg.BatchInverse(JonesH)

            self.DicoNormJones[PointingID][Description]["M,MH"]=(Jones,JonesH)
            self.DicoNormJones[PointingID][Description]["time"]=ThisTime

    def ApplyJones(self,DicoData,Code="Left.inv,Right.inv",SelPointingID=None,SkipJones="",ForceUpdate=False):
        #Code="Left.inv"
        # log=MyLogger.getLogger("ClassParamX.CorrectData")

        ListTerm=Code.split(",")[::-1]
        import copy
        DicoDataOut=copy.deepcopy(DicoData)
        for Term in ListTerm:
            self.CheckKeyDico(Term)
            for PointingID in self.MDC.ListID:
                if SelPointingID!=None:
                    if PointingID!=SelPointingID: continue
                itimes=DicoData[PointingID]["itimes"]
                self.BuildNormJones(Description=Term,itimes=itimes,SkipJones=SkipJones,ForceUpdate=ForceUpdate)
                # print ModColor.Str("%s : Code=%s : Apply NormJones for %s, PointingID=%s"%(self.MME.NameParsetME,Code,Term,PointingID),col="blue")
                A0,A1=DicoData[PointingID]["A0A1"]
                dataCorr=DicoDataOut[PointingID]['data']
                Jones,JonesH=self.DicoNormJones[PointingID][Term]["M,MH"]
                P0=ModLinAlg.BatchDot(Jones[A0,:,:],dataCorr)
                dataCorr=ModLinAlg.BatchDot(P0,JonesH[A1,:,:])
                DicoDataOut[PointingID]['data']=dataCorr
                #print DicoDataOut[PointingID]["data"][0,0,:]
                #if Term=="Right.inv": stop


        # import pylab
        # import time
        # nf=DicoData[0]["data"][0,:,:].shape[0]
        # for bl in range(100)[::10]:
        #     v0=DicoData[0]["data"][bl,:,:]
        #     v1=DicoDataOut[0]["data"][bl,:,:]

        #     pylab.clf()
        #     pylab.plot(v0.real)
        #     pylab.plot(v1.real,ls=":")
        #     pylab.title("bl=%i"%bl)
        #     pylab.pause(0.1)
        #     pylab.draw()
        #     pylab.show(False)
        #     time.sleep(.3)
        return DicoDataOut


