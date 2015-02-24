import numpy as np
import multiprocessing
from progressbar import ProgressBar
import ClassApplyJones
from ClassME import MeasurementEquation
import os
import MyLogger
MyLogger.setLoud("ModelBeamSVD")
MyLogger.setLoud("ClassParam")
MyLogger.setLoud("ModToolBox")


# from multiprocessing import sharedctypes
# size = S.size
# shape = S.shape
# S.shape = size
# S_ctypes = sharedctypes.RawArray('d', S)
# S = numpy.frombuffer(S_ctypes, dtype=numpy.float64, count=size)
# S.shape = shape
# from numpy import ctypeslib
# S = ctypeslib.as_array(S_ctypes)
# S.shape = shape


class ClassJonesContainer():
    def __init__(self,GD,MDC,NCPU=6):
        self.NCPU=NCPU
        self.GD=GD
        self.MDC=MDC
        

    ##############################
    #### Init AJM ################
    ##############################

    def InitAJM(self,raFacet,decFacet):
        self.DicoAJM={}
        args=[]
        for i in range(raFacet.size):
            args.append({"iFacet":i,"RaDec":(raFacet[i],decFacet[i])})
        self.LaunchParallel(JobArgs=args,ResultFunction=self.R2_DicoAJM,Mode="Init",TitleCounter="Init AJM")


    def R2_DicoAJM(self,DicoResult):
        iFacet=DicoResult["iFacet"]
        AJM=DicoResult["AJM"]
        self.DicoAJM[iFacet]={"AJM":AJM}

    ##############################
    #### Calc Jones ##############
    ##############################

    def CalcJones(self,times,A0A1,PointingID=0):
        #from multiprocessing import Process, Value, Array
        args=[]
        for i in range(len(self.DicoAJM)):
            args.append({"iFacet":i,"times":times,"A0A1":A0A1,"PointingID":PointingID})
        self.LaunchParallel(Mode="CalcJones",TitleCounter="CalcJones",
                            JobArgs=args,
                            ResultFunction=self.R2_DicoJones)


    def R2_DicoJones(self,DicoResult):
        iFacet=DicoResult["iFacet"]
        stop
        #AJM=DicoResult["AJM"]
        #self.DicoAJM[iFacet]={"AJM":AJM}

    




    def LaunchParallel(self,Mode="Init",TitleCounter="Init",
                       WorkerAttr=None,JobArgs=None,
                       ResultFunction=None):

        NCPU=self.NCPU
        work_queue = multiprocessing.Queue()
        #jobs = range(10)#sorted(self.DicoImager.keys())

        for job in JobArgs:
            work_queue.put(job)
        result_queue = multiprocessing.Queue()

        workerlist={}
        for ii in range(NCPU):
            ThisWorker=WorkerAJM(work_queue, result_queue,self.GD,self.MDC)
            ThisWorker.setMode(Mode)
            if WorkerAttr!=None:
                ThisWorker.setAttr(WorkerAttr)
            workerlist[ii]=ThisWorker
            workerlist[ii].start()
 
        results = []
        lold=0

        NJobs=len(JobArgs)

        pBAR= ProgressBar('white', block='=', empty=' ',Title=TitleCounter)
        pBAR.render(0, '%i/%i' % (0,NJobs))
        while len(results) < NJobs:
            result = result_queue.get()
            results.append([])#result)
            DicoResult=result
            ResultFunction(DicoResult)
            
            if len(results)>lold:
                lold=len(results)
                pBAR.render(int(100* float(lold) / (NJobs)), '%i/%i' % (lold,NJobs))

        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()
    


##########################################
####### Workers
##########################################
           
class WorkerAJM(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,GD,MDC):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.Mode="Init"
        self.GD=GD
        self.MDC=MDC

    def setMode(self,Mode):
        self.Mode=Mode

    def setAttr(self,Dico):
        for key in Dico.keys():
            setattr(self,key,Dico[key])

    def shutdown(self):
        self.exit.set()
    def run(self):
        while not self.kill_received:
            try:
                kwargs = self.work_queue.get()
            except:
                break

            if self.Mode=="Init":
                iFacet=kwargs["iFacet"]
                rac,decc=kwargs["RaDec"]

                GD=self.GD
                MDC=self.MDC
                MME=MeasurementEquation()
                HYPERCAL_DIR=os.environ["HYPERCAL_DIR"]
                execfile("%s/HyperCal/Scripts/ScriptSetMultiRIME.py"%HYPERCAL_DIR)
                AJM=ClassApplyJones.ClassApplyJones(MME)

                ra=np.array([rac])
                dec=np.array([decc])
                AJM.setRaDec(ra,dec)

                self.result_queue.put({"iFacet":iFacet,"AJM":AJM})

            elif self.Mode=="CalcJones":
                iFacet=kwargs["iFacet"]
                times=kwargs["times"]
                A0A1=kwargs["A0A1"]
                freqs=kwargs["freqs"]
                PointingID=kwargs["PointingID"]

                

                nch=freqs.size
                self.norm=np.zeros((nch,2,2),float)
                Xp=self.MME.Xp
        
                MS=self.MDC.giveMS(PointingID)
                na=MS.na
        
                if times==None:
                    times=self.Sols["times"]
                LTimes=sorted(list(set(times.tolist())))
                NTimes=len(LTimes)
        
                if A0A1==None:
                    A0,A1=np.mgrid[0:na,0:na]
                    A0List,A1List=[],[]
                    for i in range(na):
                        for j in range(i,na):
                            if i==j: continue
                            A0List.append(A0[i,j])
                            A1List.append(A1[i,j])
                    A0=np.array(A0List*NTimes)
                    A1=np.array(A1List*NTimes)
                else:
                    A0,A1=A0A1
        
        
                self.DicoATerm={}
        
                for ThisTime,itime0 in zip(LTimes,range(NTimes)):
                    TSols=self.Sols["times"]
                    XiSols=self.Sols["xi"]
                    itimeSol=np.argmin(np.abs(TSols-ThisTime))
                    xi=XiSols[itimeSol]
                    Xp.FromVec(xi)
        
                    indThisTime=np.where(times==ThisTime)[0]
                    ThisA0=A0[indThisTime]
                    ThisA1=A1[indThisTime]
                    ThisA0A1=ThisA0,ThisA1
                    itimes=(itime0,itime0+1)
                    self.AJM.BuildNormJones(Description="Right.noinv",itimes=itimes)
        
                    Jones,JonesH=self.AJM.DicoNormJones[PointingID]["Right.noinv"]["M,MH"]
        
                    JJH=ModLinAlg.BatchDot(Jones[ThisA0,:,:],JonesH[ThisA1,:,:])
                    JJH_sq=np.mean(JJH*JJH.conj(),axis=0).reshape(nch,2,2)
                    self.norm+=JJH_sq.real
                    self.DicoATerm[ThisTime]=copy.deepcopy(self.AJM.DicoNormJones[PointingID]["Right.noinv"]["M,MH"])
                
                self.norm/=NTimes
                self.norm=np.sqrt(self.norm)
                self.norm=self.norm.reshape(nch,2,2)
                self.norm=ModLinAlg.BatchInverse(self.norm)
                self.norm=self.norm.reshape(1,nch,4)
                self.norm[np.abs(self.norm)<1e-6]=1
                self.norm.fill(1)

                self.result_queue.put({"iFacet":iFacet})
         
