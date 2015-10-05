import numpy as np
from Other.progressbar import ProgressBar




class ClassFitIslands():
    def __init__(self,IslandClass):
        self.Islands=IslandClass

    def FitSerial(self):

        Islands=self.Islands

        ImOut=np.zeros_like(Islands.Noise)
        pBAR = ProgressBar('white', block='=', empty=' ',Title="Fit islands")
        
        sourceList=[]
        for i in range(len(Islands.ListX)):
            comment='Isl %i/%i' % (i+1,len(Islands.ListX))
            pBAR.render(int(100* float(i+1) / len(Islands.ListX)), comment)
            
            xin,yin,zin=np.array(Islands.ListX[i]),np.array(Islands.ListY[i]),np.array(Islands.ListS[i])
            xm=int(np.sum(xin*zin)/np.sum(zin))
            ym=int(np.sum(yin*zin)/np.sum(zin))
            # Fit=ClassFit(xin,yin,zin,psf=(PMaj/incr,PMin/incr,PPA),noise=Islands.Noise[xm,ym])
            Fit=ClassFit(xin,yin,zin,psf=(PMaj/incr,PMin/incr,PPA+np.pi/2),noise=StdResidual)#,FreePars=["l", "m","s"])
            sourceList.append(Fit.DoAllFit())
            Fit.PutFittedArray(ImOut)
            
        Islands.FitIm=ImOut




#======================================
import multiprocessing

class WorkerAntennaLM(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,SM,PolMode,SolverType,IdSharedMem,ConfigJacobianAntenna=None,GD=None):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.SM=SM
        self.PolMode=PolMode
        self.SolverType=SolverType
        self.IdSharedMem=IdSharedMem
        self.ConfigJacobianAntenna=ConfigJacobianAntenna
        self.GD=GD

        self.InitPM()

        #self.DoCalcEvP=DoCalcEvP
        #self.ThisTime=ThisTime
        #self.e,=kwargs["args"]
        

    def InitPM(self):

        x=np.linspace(0.,15,100000)
        Exp=np.float32(np.exp(-x))
        LExp=[Exp,x[1]-x[0]]
        
        self.PM=ClassPredict(Precision="S",DoSmearing=self.GD["SkyModel"]["Decorrelation"],IdMemShared=self.IdSharedMem,LExp=LExp)

        if self.GD["ImageSkyModel"]["BaseImageName"]!="":
            self.PM.InitGM(self.SM)

    def shutdown(self):
        self.exit.set()
    def run(self):

        while not self.kill_received:
            try:
                iAnt,DoCalcEvP,ThisTime,rms,DoEvP = self.work_queue.get()
            except:
                break
            #self.e.wait()
            

            T=ClassTimeIt.ClassTimeIt("Worker")
            T.disable()
            # if DoCalcEvP:
            #     T.disable()
            JM=ClassJacobianAntenna(self.SM,iAnt,PolMode=self.PolMode,PM=self.PM,IdSharedMem=self.IdSharedMem,GD=self.GD,
                                    **dict(self.ConfigJacobianAntenna))
            T.timeit("ClassJacobianAntenna")
            JM.setDATA_Shared()
            T.timeit("setDATA_Shared")

            G=NpShared.GiveArray("%sSharedGains"%self.IdSharedMem)
            G0Iter=NpShared.GiveArray("%sSharedGains0Iter"%self.IdSharedMem)
            P=NpShared.GiveArray("%sSharedCovariance"%self.IdSharedMem)
            #Q=NpShared.GiveArray("%sSharedCovariance_Q"%self.IdSharedMem)
            evP=NpShared.GiveArray("%sSharedEvolveCovariance"%self.IdSharedMem)
            T.timeit("GiveArray")

            if self.SolverType=="CohJones":
                x,_,_=JM.doLMStep(G)
                self.result_queue.put([iAnt,x,None,None,{"std":-1.,"max":-1.,"kapa":None}])
            elif self.SolverType=="KAFCA":
                #T.disable()
                if DoCalcEvP:
                    evP[iAnt]=JM.CalcMatrixEvolveCov(G,P,rms)
                    T.timeit("Estimate Evolve")

                # EM=ClassModelEvolution(iAnt,
                #                        StepStart=3,
                #                        WeigthScale=2,
                #                        DoEvolve=False,
                #                        order=1,
                #                        sigQ=0.01)

                EM=ClassModelEvolution(iAnt,
                                       StepStart=0,
                                       WeigthScale=0.5,
                                       DoEvolve=True,
                                       BufferNPoints=10,
                                       sigQ=0.01,IdSharedMem=self.IdSharedMem)
                T.timeit("Init EM")

                Pa=None

                # Ga,Pa=EM.Evolve0(G,P,self.ThisTime)
                # if Ga!=None:
                #     G[iAnt]=Ga
                #     P[iAnt]=Pa

                x,Pout,InfoNoise=JM.doEKFStep(G,P,evP,rms,Gains0Iter=G0Iter)
                T.timeit("EKFStep")
                rmsFromData=JM.rmsFromData

                if DoEvP:
                    Pa=EM.Evolve0(x,Pout)#,kapa=kapa)
                    T.timeit("Evolve")
                else:
                    Pa=P[iAnt].copy()
                #_,Pa=EM.Evolve(x,Pout,ThisTime)

                if type(Pa)!=type(None):
                    Pout=Pa

                self.result_queue.put([iAnt,x,Pout,rmsFromData,InfoNoise])

