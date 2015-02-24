from ClassME import MeasurementEquation
from PredictDir import ClassHyperH
import ClassTimeIt
import MyLogger
import numpy as np
from IPClusterDir.CheckJob import LaunchAndCheck, SendAndCheck

class ClassDistributedH():
    def __init__(self,E,MDC):
        self.E=E
        self.MDC=MDC
        self.GD=self.E.GD
        self.BuildLocal()
        self.BuildRemote()

    def BuildLocal(self):
        MME=MeasurementEquation()
        MDC=self.MDC
        GD=self.GD
        HYPERCAL_DIR=GD.HYPERCAL_DIR
        TypeME="RIME"
        execfile("%s/HyperCal/Scripts/ScriptSetMultiRIME.py"%HYPERCAL_DIR)
        self.MME=MME
        self.LocalH=ClassHyperH.ClassHyperH(MME,GD)

    def setDistributedCurrentFlags(self,DicoData):
        MDC=self.MDC
        V=self.E.GiveSubCluster("Calc")["V"]

        DicoCurrentFlags={}
        for PointingID in self.MDC.ListID:
            DicoCurrentFlags[PointingID]=DicoData[PointingID]["flags"]

        self.MDC.setCurrentData("Flags",DicoCurrentFlags)
        #V["DicoCurrentFlags"]=DicoCurrentFlags
        SendAndCheck(V,"DicoCurrentFlags",DicoCurrentFlags,TitlePBAR="Send Flags",Progress=True)

        LaunchAndCheck(V,'MDC.setCurrentData("Flags",DicoCurrentFlags)')
        # MDC=self.MDC
        # self.MDC.setCurrentFlags(DicoData)
        # V=self.E.GiveSubCluster("Calc")["V"]

        # DicoCurrentFlags={}
        # for PointingID in self.MDC.ListID:
        #     DicoCurrentFlags[PointingID]={"flags":DicoData[PointingID]["flags"]}

        # V["DicoCurrentFlags"]=DicoCurrentFlags
        # r=V.execute('MDC.setCurrentFlags(DicoCurrentFlags)'); r.wait(); print r.get()

    def setDistributedRandomMapRepr(self,RandomMapRepr):
        MDC=self.MDC
        self.MDC.setCurrentData("RandomMapRepr",RandomMapRepr)
        V=self.E.GiveSubCluster("Calc")["V"]

        #V["RandomMapRepr"]=RandomMapRepr
        SendAndCheck(V,"RandomMapRepr",RandomMapRepr,TitlePBAR="Send RandomMap",Progress=True)

        LaunchAndCheck(V,'MDC.setCurrentData("RandomMapRepr",RandomMapRepr)')

    def setDistributedCurrentData(self,anyDataName,anyData):
        MDC=self.MDC
        self.MDC.setCurrentData(anyDataName,anyData)
        V=self.E.GiveSubCluster("Calc")["V"]

        #V[anyDataName]=anyData
        SendAndCheck(V,anyDataName,anyData,Progress=True)
        #r=V.execute('MDC.setCurrentData("%s",%s)'%(anyDataName,anyDataName)); r.wait(); print r.get()
        LaunchAndCheck(V,'MDC.setCurrentData("%s",%s)'%(anyDataName,anyDataName))

    def BuildRemote(self):
        log=MyLogger.getLogger("ClassDistributedH.CuildRemote")
        MDC=self.MDC
        NPointings=MDC.NPointing
        import ClassTimeIt
        T=ClassTimeIt.ClassTimeIt("IPCluster.BuildRemoteRIME")
        C=self
        V=self.E.GiveSubCluster("Calc")["V"]

        # V["MDC"]=MDC
        # V["GD"]=self.GD
        # V["TypeME"]="RIME"
        SendAndCheck(V,"MDC",MDC,Progress=True)
        SendAndCheck(V,"GD",self.GD,Progress=True)
        SendAndCheck(V,"TypeME","RIME",Progress=True)
        V["NPointings"]=NPointings



        V.execute("from ClassME import MeasurementEquation; MME=MeasurementEquation()")
        strExec="%s/HyperCal/Scripts/ScriptSetMultiRIME.py"%self.GD.HYPERCAL_DIR
        #r=V.execute('execfile("%s")'%strExec); r.wait(); print>>log, r.get()
        #r=V.execute("from PredictDir import ClassHyperH; h=ClassHyperH.ClassHyperH(MME,GD)"); print>>log, r.get()
        LaunchAndCheck(V,'execfile("%s")'%strExec,Progress=True,TitlePBAR="Declare MME")
        LaunchAndCheck(V,"from PredictDir import ClassHyperH; h=ClassHyperH.ClassHyperH(MME,GD)",Progress=True,TitlePBAR="Declare H")

    def DistributedHR(self,x,it0=0,it1=None,Noise=None,ToVec=True,Zero=False,ToRepr=True,CorrectJ=False):
        log=MyLogger.getLogger("IPCluster.h")
        V=self.E.GiveSubCluster("Calc")["V"]
        indrc=self.E.GiveSubCluster("Calc")["ids"]
        #import objgraph
        #print>>log, objgraph.show_growth(limit=7)
        V.push({"out":None})
        j=0
        NToDo=len(x)
        # DoThis=np.int64(np.linspace(0,NToDo,self.NEng+1))
        DoThis=np.int64(sorted(list(set(np.int64(np.linspace(0,NToDo,len(V)+1)).tolist()))))
        V["out"]=None
        
        # r=V.push({"MaskFreq":self.MaskFreq}); r.wait()
        # r=V.execute("h.MaskFreq=MaskFreq"); r.wait()
        T=ClassTimeIt.ClassTimeIt("IPCluster.h")
        for ii in range(len(DoThis)-1):
            i=indrc[ii]
            self.E.rc[i].push({"x":x[DoThis[ii]:DoThis[ii+1]],"out":[],"it0":it0,"it1":it1,"Noise":Noise,
                             "Zero":Zero,"ToVec":ToVec,"ToRepr":ToRepr,"CorrectJ":CorrectJ})

            strdo=[]
            ido=0
            for jj in range(DoThis[ii],DoThis[ii+1]):
                #strdo.append("out.append(h(x[%i],it0=it0,it1=it1,Noise=Noise,DoGain=False,ToFFT=False))"%ido)
                #strdo.append("out.append(h(x[%i],it0=it0,it1=it1,Noise=Noise,DoGain=False))"%ido)
                strdo.append("out.append(h.HR(x[%i],it0=it0,it1=it1,Noise=Noise,Zero=Zero,ToRepr=ToRepr,CorrectJ=CorrectJ))"%ido)
                ido+=1
            #print>>log, "engine%i: %s"%(i,strdo)
            r=self.E.rc[i].execute(";".join(strdo))

            # ###########
            # # debug
            # r.wait()
            # print>>log, r.get()
            # #######

        T.timeit("   IPC: Send order")
        self.E.rc.wait()
        T.timeit("   IPC: Done Calc")
        listout=V.get("out")
        T.timeit("   IPC: Get result")
        #N=listout[0][0].shape[0]*listout[0][0].shape[1]

        listout=[l for l in listout if ((l !=None)&(l!=[]))]
        
        sizeObs=listout[-1][0].size
        dtype=listout[-1][0].dtype
        shapeout=(len(x),sizeObs)
        out=np.zeros(shapeout,dtype)
        indi=0
        for i in range(len(listout)):
            for j in range(len(listout[i])):
                out[indi]=listout[i][j].reshape((sizeObs,))
                indi+=1

#        out=np.array([item.reshape((item.size,)) for sublist in listout for item in sublist if len(item)>0]).T
            
        out=out.T
        T.timeit("   IPC: Reshape")

        self.E.clear()
        T.timeit("   IPC: Clear")

        return out




