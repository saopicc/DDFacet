
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
import socket
import PrintRecArray
log=MyLogger.getLogger("ClassEngine")
import ModProbeCPU
from CheckJob import LaunchAndCheck
import ModColor
import sys

class Engine():

    def __init__(self,GD):
        self.rc=0
        self.GD=GD
        self.NEng=0
        self.hostlist=""
        self.DoneInit=False
        self.hostlist={}
        # for i in range(1,50):
        #     node="lce%3.3i"%i
        #     self.hostlist[node]=self.nengine

        # DoPlotEngine=True
        # if DoPlotEngine:
        #     self.hostlist["igor"]+=1
        
        
        self.CatMSNodes=GD.CatMSNodes
        self.DicoConfig=GD.DicoConfig
        self.DicoData={}

        self.BuildNodesList()
        PrintRecArray.Print(self.GD.CatMSNodes)

        self.ipclusterDir=os.environ["IPYTHON_CONF_DIR"]+"/profile_ssh"
        ModDicoFiles.DictToFile(self.hostlist,"%s/ClusterDico.txt"%self.ipclusterDir)

        self.pBAR = ProgressBar('white', block='=', empty=' ',Title="IPCluster")

        EngineOK=False

        self.LogFile = open('%s/logip.txt'%self.ipclusterDir, 'w')
        self.LStart=["ipcluster", "start", "--profile=ssh"]
        self.LStop =["ipcluster", "stop", "--profile=ssh"]

        while EngineOK==False:
            EngineOK=self.init_engines()

        try:
            self.give_client()
            #self.MapEngines()
        except:
            print>>log, "No engines found"
            self.init_engines()

        CatEngines=np.zeros((len(self.rc.ids),),dtype=[
                ("node","|S200"),("nodeNum",int),
                ("EngNum",int),("TypeEng","|S200"),("dirMSName","|S200"),
                ("Freq",float),("MSMounted",bool),
                ("available",bool),("ChanFreq",np.float32,(256,)),
                ("NChan",int),("LoadCPU",float),("PointingID",int)])
        GD.CatEngines=CatEngines.view(np.recarray)
        GD.CatEngines.available=False
        GD.CatEngines.available[self.rc.ids]=True
        GD.CatEngines.EngNum[self.rc.ids]=self.rc.ids
        GD.CatEngines.isProberCPU=False
        self.SetEnginesTypes()


    def setData(self,key,Data):
        self.DicoData[key]=Data

    def getData(self,key):
        if not(key in self.DicoData.keys()):
            return False
        else:
            return self.DicoData[key]

    def BuildNodesList(self):

        CatMSNodes=self.CatMSNodes
        DicoConfig=self.DicoConfig

        NCalcEnginePerNode=int(DicoConfig["Cluster"]["NCalcEnginePerNode"])
        self.NCalcEnginePerNode=NCalcEnginePerNode
        


        LocalHost=socket.gethostname()
        if "." in LocalHost: LocalHost,_=LocalHost.split(".")
        LocalHostName=LocalHost

        if "igor" == socket.gethostname():
            AvailableNodes=["igor"]
            self.timeout=30
        elif ("lce" in socket.gethostname()):
            print "I am on CEP1 cluster"
            self.timeout=180
            AvailableNodes=["lce%3.3i"%i for i in range(1,50)]
        elif DicoConfig["Cluster"]["MapEngineType"]=="DDFacet":
            self.timeout=30
            AvailableNodes=[LocalHost]


        NMaxCalcEngines=int(DicoConfig["Cluster"]["NMaxCalcEngines"])
        NCalcEngines=0
        for node in AvailableNodes:
            if NCalcEngines>NMaxCalcEngines: NCalcEnginePerNode=0
            self.hostlist[node]=NCalcEnginePerNode
            NCalcEngines+=NCalcEnginePerNode

        listNodes=sorted(list(set(CatMSNodes.node.tolist())))
        for node in self.hostlist.keys():
            nMSThisNode=np.where(CatMSNodes.node==node)[0].size
            self.hostlist[node]+=nMSThisNode



        self.NImagEngine=np.min([int(DicoConfig["Cluster"]["NImagEngine"]),int(DicoConfig["Facet"]["MainFacetOptions"]["NFacets"])**2])
        self.hostlist[LocalHostName]+=self.NImagEngine

        if DicoConfig["Cluster"]["ListExceptNode"]!=None:#['lce043','lce039']
            ListExceptNode=DicoConfig["Cluster"]["ListExceptNode"]
            for node in ListExceptNode:
                if (node in self.hostlist.keys())|(self.hostlist[node]==0):
                    del(self.hostlist[node])
        for node in self.hostlist.keys():
            if (self.hostlist[node]==0):
                del(self.hostlist[node])
                    
        print self.hostlist.keys()
        ProbeCPU=DicoConfig["Cluster"]["CheckCPU"]
        if ProbeCPU:
            for key in self.hostlist.keys():
                self.hostlist[key]+=1



    def GiveLoads(self):
        log=MyLogger.getLogger("IPCluster.InitProberCPU")
        print>>log, "GiveLoads"

        VProbe=self.GiveSubCluster("ProbeCPU")["V"]
        #r=VProbe.execute("loadhost=(host,T.AvgLoad)"); r.wait(); print>>log, r.get()
        LaunchAndCheck(VProbe,"loadhost=(host,T.AvgLoad)")

        LLoad= VProbe["loadhost"]
        for node,load in LLoad:
            ind =np.where(self.GD.CatEngines.node == node)[0]
            self.GD.CatEngines.LoadCPU[ind]=load
        print>>log, "done GiveLoads"


    def InitProberCPU(self):
        V=self.GiveSubCluster("ProbeCPU")["V"]
        strExec="%s/Scripts/StartProbe.py"%self.GD.HYPERCAL_DIR
        #r=V.execute('execfile("%s")'%strExec); r.wait(); print r.get()
        LaunchAndCheck(V,'execfile("%s")'%strExec)
        

        print>>log, "done updateReachable"

    def SetEnginesTypes(self):
        log=MyLogger.getLogger("IPCluster.MapEngines")
        print>>log, " ... Mapping empty engines types"
        self.FoundMS=False

        V=self.rc[:]
        strExec="%s/Scripts/ScriptHostName.py"%self.GD.HYPERCAL_DIR
        LaunchAndCheck(V,'execfile("%s")'%strExec)
        #r=V.execute('import socket; host=socket.gethostname()'); r.wait()
        host=V["LocalHostName"]
        self.GD.CatEngines.node=host
        hostList=sorted(list(set(host)))

        for ThisHost in hostList:
            indEngThisNode=np.where(self.GD.CatEngines.node==ThisHost)[0]
            indMSThisNode=np.where(self.GD.CatMSNodes.node==ThisHost)[0]
            
            CatEngThisNode=self.GD.CatEngines[indEngThisNode]
            CatMSThisNode=self.GD.CatMSNodes[indMSThisNode]
            nMS=indMSThisNode.size
            nCalc=self.NCalcEnginePerNode
            CatEngThisNode.TypeEng[0:nMS]="Data"
            CatEngThisNode.dirMSName[0:nMS]=CatMSThisNode.dirMSname
            CatEngThisNode.PointingID[0:nMS]=CatMSThisNode.PointingID
            i0=nMS
            i1=nMS+nCalc
            CatEngThisNode.TypeEng[i0:i1]="Calc"
            i0=i1
            i1=i1+self.NImagEngine
            CatEngThisNode.TypeEng[i0:i1]="Imag"
            CatEngThisNode.TypeEng[i1:]="ProbeCPU"
            self.GD.CatEngines[indEngThisNode]=CatEngThisNode
            
        


        PrintRecArray.Print(self.GD.CatEngines)

    # def GiveSubCluster(self,Type,PointingID=0):
    #     CatEngines=self.GD.CatEngines
    #     CondEngineType=(CatEngines.TypeEng==Type)
    #     if Type=="Data":
    #         CondAvailable=(CatEngines.MSMounted==True)
    #         CondPointingID=(CatEngines.PointingID==PointingID)
    #         Cond = CondAvailable & CondEngineType & CondPointingID
    #     else:
    #         Cond = CondEngineType
    #     ind=np.where(Cond)[0]
    #     SubCat=CatEngines[ind]
    #     ids = SubCat.EngNum.tolist()
    #     SubCluster={"ids":ids,"V":self.rc[ids],"Cat":SubCat}
    #     return SubCluster

    def GiveSubCluster(self,Type,PointingID=0,MSMounted=True,available=True):
        CatEngines=self.GD.CatEngines
        CondEngineType=(CatEngines.TypeEng==Type)
        DicoSel={}
        DicoSel["TypeEng"]={"val":Type,"oper":"="}
        DicoSel["PointingID"]={"val":PointingID,"oper":"="}
        DicoSel["MSMounted"]={"val":MSMounted,"oper":"="}
        DicoSel["available"]={"val":available,"oper":"="}
        if (Type=="Data"):
            del(DicoSel["MSMounted"])
        if (Type=="Calc")|(Type=="ProbeCPU")|(Type=="Imag"):
            del(DicoSel["PointingID"])
            del(DicoSel["MSMounted"])
        if (Type=="ProbeCPU"):
            del(DicoSel["available"])

        CondList=[]
        
        #PrintRecArray.Print(self.GD.CatEngines)
        for FieldName in DicoSel.keys():
            val,oper=DicoSel[FieldName]["val"],DicoSel[FieldName]["oper"]
            if oper=="=":
                op=lambda x,y: x==y
            elif oper=="<":
                op=lambda x,y: x<y
            elif oper==">":
                op=lambda x,y: x>y
            cond=op(CatEngines[FieldName],val)
            CondList.append(cond)
            #print FieldName,val,cond

        ind=np.all(CondList,axis=0)
        #print ind
        SubCat=CatEngines[ind]
        ids = SubCat.EngNum.tolist()
        SubCluster={"ids":ids,"V":self.rc[ids],"Cat":SubCat}
        # print DicoSel
        # PrintRecArray.Print(SubCat)
        # print 

        return SubCluster


    def killAll(self):
        S=subprocess.Popen(self.LStop, stdout=self.LogFile,stderr=self.LogFile)
        S.wait()

    def launch_ipcluster(self):
        #self.killAll()
        
        print self.LStart
        S=subprocess.Popen(self.LStart, stdout=self.LogFile,stderr=self.LogFile)
        
    
    def give_client(self):
        self.rc=Client(profile="ssh")
        #print len(self.rc)
        self.NEng=len(self.rc)
        self.EngIds=self.rc.ids
        self.V=self.rc.direct_view()
        self.Vb=self.rc.load_balanced_view()
        
        if time.time()-self.t0>60:
            r=self.rc[:].execute('import socket; host=socket.gethostname()'); r.wait()
            self.loadedHost=sorted(list(set(self.V["host"])))
            for host in self.hostlist:
                if not(host in self.loadedHost):
                    print 
                    print "%s not loaded yet"%host
                    print 

    def init_engines(self):
        ntot=0
        for key in self.hostlist.keys(): ntot+=self.hostlist[key]

        self.NTotEng=ntot
        self.launch_ipcluster()
        t0=time.time()
        self.t0=t0

        while True:
            try:
                #print>>log, "try0"
                self.give_client()
                #print>>log, "\ntry"
                comment='src %i/%i' % (self.NEng,ntot)
                self.pBAR.render(int(100* float(self.NEng) / ntot), comment)
                #print>>log, 
                #print>>log, len(self.rc)
                if (len(self.rc)==ntot):
                    break
                elif (time.time()-t0)>self.timeout:
                    return False
                else:
                    #del(self.rc)
                    time.sleep(2)
            #except:
            except Exception as ex:
                template = "An exception of type {0} occured. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print>>log, (message,)
                if (time.time()-t0)>self.timeout:
                    return False
                time.sleep(2)
                # print>>log, "except"
                # return False
        self.pBAR.reset()
        HYPERCAL_DIR=os.environ["HYPERCAL_DIR"]
        
#        self.rc[:].run("%s/HyperCal/importAll.py"%HYPERCAL_DIR)
        #r=self.rc[:].execute("import Source; Source.Source()"); r.wait(); print r.get()
        self.DoneInit=True
        r=self.V.execute('import os; import sys; os.environ["HYPERCAL_DIR"]="%s"'%HYPERCAL_DIR); r.wait()#; print r.get()
        PATH=os.getcwd()
        #r=self.V.execute('sys.path=["%s"]+sys.path'%PATH)
        r=self.V.execute('sys.path.append("%s")'%"%s/pythonlibs"%HYPERCAL_DIR)
        r=self.V.execute('sys.path.append("%s")'%"%s/pythonlibs/Widget"%HYPERCAL_DIR)
        r=self.V.execute('sys.path.append("%s")'%"%s/HyperCal"%HYPERCAL_DIR)
        r=self.V.execute('sys.path.append("%s")'%"/usr/local/lib/python2.7/dist-packages")
        r=self.V.execute('sys.path.append("%s")'%"/home/tasse/build/LOFAR/lib/python2.6/dist-packages")
        r=self.V.execute('sys.path.append("%s")'%"/home/tasse/lib/lib/python2.7/site-packages")
        r=self.V.execute('sys.path.append("%s")'%"/home/tasse/lib/lib/python2.7/site-packages/numexpr-2.0.1-py2.7-linux-x86_64.egg")
        r=self.V.execute('sys.path.append("%s")'%os.environ["STATION_RESPONSE_DIR"])

        # PYTHONPATH=os.environ["PYTHONPATH"]
        # ll=os.environ["PYTHONPATH"].split(":")
        # ll=[l.replace("\n","") for l in ll]
        # self.V["PYTHONPATH"]=ll
        # self.V.execute('sys.path=PYTHONPATH+sys.path')
        ll=sys.path
        self.V["PYTHONPATH"]=ll
        LaunchAndCheck(self.V,'sys.path=PYTHONPATH')
        #LaunchAndCheck(self.V,'PP=sys.path')
        #print self.V["PP"][0]
        #LaunchAndCheck(self.V,'import os; cwd=os.getcwd()')
        #print self.V["cwd"]
        #LaunchAndCheck(self.V,'import pyrap')
        #stop
        r=self.V.execute('sys.path.append("%s"); donecd=1'%"/home/tasse/lib/lib/python2.7/site-packages")
        #print>>log, self.V.execute("import os; os.environ['DISPLAY'] = ':0.0'").get()
        r=self.V.execute('import socket')

    def mapReachable(self):
        import logging
        log=logging.getLogger("paramiko.transport")
        log.propagate=False

        import paramiko
        s=paramiko.SSHClient()
        s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        NonReachable=[]
        for host in sorted(self.hostlist.keys()):
            print host
            try:
                s.connect(host,timeout=0.05)
                s.close()
            except:
                print>>log, ("%s not reachable"%host)
                NonReachable.append(host)

        return NonReachable


    def kill_engines(self):
        #proc = subprocess.Popen(['screen','-list'],stdout=subprocess.PIPE)
        for host in self.hostlist.keys():
            #ss="ssh %10s screen -X -S engines quit"%(host)
            #os.system(ss)
            ss='ssh %10s "pkill screen" &'%host
            print>>log, ss
            os.system(ss)

        #os.system("screen -X -S controller quit")
            
        os.system("pkill screen")
        time.sleep(4)


    def clear(self):
        i=0
        #print ModColor.Str("============================== PURGE")
        self.V.client.results.clear()
        self.V.client.metadata.clear()
        #self.V.purge_results("all")
        #self.rc.purge_results("all")
        self.V.client.results.clear()
        self.V.client.metadata.clear()
        if not self.rc.outstanding:
            self.V.purge_results("all")
            self.rc.purge_results("all")
        self.rc.history = []
        self.V.history = []       
        self.rc.history = []
        self.V.history = []       
        import gc
        gc.enable()
        gc.collect()
        #print ModColor.Str("============================== PURGE DONE")


        # #print ModColor.Str("============================== PURGE")
        # rc=self.rc
        # V=self.V
        # rc.results.clear()
        # rc.metadata.clear()
        # V.results.clear()
        # rc.history = []
        # V.history = []

        # #import time
        # self.V.purge_results("all")
        # self.rc.purge_results("all")
        # if not rc.outstanding:
        #     rc.purge_results('all') #clears controller
        # rc.purge_results('all') #clears controller
        # # else:
        # #     while len(rc.outstanding)>0:
        # #         print len(rc.outstanding)
        # #         time.sleep(0.3)

        # #print ModColor.Str("============================== PURGE DONE")
