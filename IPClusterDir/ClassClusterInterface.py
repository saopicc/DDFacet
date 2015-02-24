

import os
import time
from IPython.parallel import Client
import IPython.parallel.controller.heartmonitor
#IPython.parallel.controller.heartmonitor.HeartMonitor.period=100000
import glob
import numpy as np
import ModColor
from progressbar import ProgressBar
import ClassTimeIt
import ModDicoFiles
import subprocess
import MyLogger
import PrintRecArray
#import sys
#for i in sys.path: print i
from ClassDistributedMS import ClassDistributedSinglePointingMS



class ClassClusterInterface():
    def __init__(self,E):
       self.E=E
       self.GD=E.GD
       self.DicoMS={}
       self.NPointings=np.max(self.E.GD.CatEngines.PointingID)+1

    def GiveDMS(self,PointingID):
        return self.DicoMS[PointingID]

    def GiveDataChunk(self,t0=0,t1=1):
        out={}
        for PointingID in sorted(self.DicoMS.keys()):
            DMS=self.GiveDMS(PointingID)
            out[PointingID]= DMS.GiveViSTchunk(t0=t0,t1=t1)
        return out

    def SetMaskHDF5(self,Xp):
        if Xp.HasHDF5:
            CHDF5=Xp.giveModelTerm(Xp.HDF5Key)
            for PointingID in range(self.NPointings):
                dMS=self.GiveDMS(PointingID)
                CHDF5.SetMaskHDF5(dMS.freqs,PointingID)


    def LoadRemoteMS(self):
        for PointingID in range(self.NPointings):
            dMS=ClassDistributedSinglePointingMS(self.E,PointingID)
            #t0,t1=self.GD.DicoConfig["Files"]["TimeSel"]

            dMS.LoadMS()
            self.DicoMS[PointingID]=dMS

    def LoadRemoteSM(self,MDC):
        for PointingID in range(self.NPointings):
            dMS=self.GiveDMS(PointingID)
            dMS.SendSM(MDC.giveSM(PointingID))
        PrintRecArray.Print(self.E.GD.CatEngines)


    def AttachMDC_dMSLocal(self,MDC):
        for PointingID in range(self.NPointings):
            dMS=self.GiveDMS(PointingID)
            dMS.MDC=MDC

    def update_MaskFreq_reachable(self):
        log=MyLogger.getLogger("IPCluster.updateReachable")
        
        CatEngines=self.GD.CatEngines
        CatEngines.available=True
        self.E.GiveLoads()
        CondReject=(CatEngines.LoadCPU>400)
        ListNonReachable=sorted(list(set((CatEngines.node[CondReject]).tolist())))
        for i in range(len(ListNonReachable)):
            host=ListNonReachable[i]
            CatEngines.available[CatEngines.node==host]=False
            print>>log, "host %s is non reachable"%host

        for PointingID in range(self.NPointings):
            dMS=self.DicoMS[PointingID]
            dMS.updateMaskFreqs()


        # CondMS=((self.CatEngines.used==True)&(self.CatEngines.isProberCPU==False))
        # CondAv=(self.CatEngines.available==True)&(self.CatEngines.isProberCPU==False)
        # #indMSrc=(np.where(CondMS & CondAv)[0])
        # indMSrc=(np.where(CondMS)[0])
        # FreqSel=self.CatEngines.Freq[indMSrc]
        
        # indSortFreq=np.argsort(FreqSel)
        # indMSrc=indMSrc[indSortFreq]

        # #print ListNonReachable
        # #print indMSrc,indMSrc

        # self.indMSrc=indMSrc
        # self.VMS=self.rc[indMSrc.tolist()]

        # indrc=(np.where(CondAv)[0]).tolist()
        # self.indrc=indrc
        # self.NEng=len(self.indrc)
        # self.EngIds=(np.array(self.C.EngIds)[indrc]).tolist()
        # self.V=self.rc[indrc]

        # CondAll=(self.CatEngines.isProberCPU==False)
        # indrcAll=(np.where(CondAll)[0]).tolist()
        # self.VAll=self.rc[indrcAll]









    def Reload(self,name="PredictDir.ClassHyperH"):
        log=MyLogger.getLogger("IPCluster.Reload")
        #self.V.execute("reload(ClassParam)")
        #r=self.V.execute("import __builtin__"); r.wait(); print>>log, r.get()
        #r=self.V.execute("__builtin__.reload = dreload"); r.wait(); print>>log, r.get()
        r=self.V.execute("dreload.func_defaults=(['numpy', 'sys', '__builtin__', 'scipy', 'matplotlib', 'pylab', '__main__'],)"); r.wait(); print>>log, r.get()
        #r=self.V.execute("%load_ext autoreload"); r.wait(); print>>log, r.get()
        #r=self.V.execute("%autoreload 2"); r.wait(); print>>log, r.get()

        #self.V.execute("reload(PredictDir.ClassNLOper_C)")
        #self.V.execute("reload(ClassParam)")
        if name!=None:
            #r=self.V.execute("reload(%s)"%name); r.wait()
            #r=self.V.execute("dreload(%s, exclude=['numpy', 'sys', '__builtin__', '__main__'])"%name); r.wait()
            r=self.V.execute("dreload(%s, exclude=['numpy', 'sys', '__builtin__', 'scipy', 'matplotlib', 'pylab', '__main__'])"%name); r.wait()
            




    def MapEngines(self):
        log=MyLogger.getLogger("IPCluster.MapEngines")
        print>>log, " ... Mapping existing engines properties:"
        self.FoundMS=False
        for i in self.EngIds:
            r=self.rc[i].execute('import socket; host=socket.gethostname()'); r.wait()
            r=self.rc[i].execute('try: name,freq,ChanFreq=MSdata.MSName,MSdata.Freq_Mean,MSdata.ChanFreq\nexcept: name,freq,ChanFreq=None,-1,-1'); r.wait()
            #r=self.rc[i].execute('name,freq=MS.MSName,MS.Freq_Mean'); r.wait()
            #print>>log, r.get()
            name=self.rc[i].get("name")
            host=self.rc[i].get("host")
            freq=self.rc[i].get("freq")
            ChanFreq=self.rc[i].get("ChanFreq")
            self.CatEngines.node[i]=host
            if "lce" in self.CatEngines.node[i]:
                ThisNode=self.CatEngines.node[i]
                self.CatEngines.nodeNum[i]=int(ThisNode.split("lce")[-1])

            self.CatEngines.MountedMS[i]=""
            self.CatEngines.used[i]=False
            self.CatEngines.EngNum[i]=i
            if name !=None:
                #print>>log, "%10s: %30s %5.1f MHz"%(host, name, freq/1.e6)
                self.FoundMS=True
                self.CatEngines.used[i]=True
                self.CatEngines.Freq[i]=freq
                self.CatEngines.MountedMS[i]=name
                self.CatEngines.NChan[i]=ChanFreq.size
                self.CatEngines.ChanFreq[i][0:ChanFreq.size]=ChanFreq.flatten()
                print>>log, i,self.CatEngines.ChanFreq[i][0:ChanFreq.size]
            
        if self.FoundMS==False:
            print>>log, "  ... no MS found attached to engines"
        else:
            self.CatEngines=self.CatEngines[self.CatEngines.node!=""]
            self.freqs=np.array(sorted(self.CatEngines.Freq[self.CatEngines.Freq!=0.].tolist()))
            #self.updateReachable()

        

