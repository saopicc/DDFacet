import numpy as np
from SkyModel.Other.progressbar import ProgressBar
from SkyModel.PSourceExtract.ClassGaussFit import ClassGaussFit as ClassFit




class ClassFitIslands():
    def __init__(self,IslandClass,NCPU=6):
        self.Islands=IslandClass
        self.NCPU=NCPU

    def FitSerial(self,psf,incr,StdResidual):
        PMin,PMaj,PPA=psf
        
        Islands=self.Islands

        ImOut=np.zeros(Islands.MaskImage.shape,np.float32)
        pBAR = ProgressBar('white', block='=', empty=' ',Title="Fit islands")
        
        sourceList=[]
        for i in range(len(Islands.ListX)):
            comment='Isl %i/%i' % (i+1,len(Islands.ListX))
            pBAR.render(int(100* float(i+1) / len(Islands.ListX)), comment)
            
            xin,yin,zin=np.array(Islands.ListX[i]),np.array(Islands.ListY[i]),np.array(Islands.ListS[i])
            #xm=int(np.sum(xin*zin)/np.sum(zin))
            #ym=int(np.sum(yin*zin)/np.sum(zin))
            # Fit=ClassFit(xin,yin,zin,psf=(PMaj/incr,PMin/incr,PPA),noise=Islands.Noise[xm,ym])
            Fit=ClassFit(xin,yin,zin,psf=(PMaj/incr,PMin/incr,PPA+np.pi/2),noise=StdResidual)#,FreePars=["l", "m","s"])
            sourceList.append(Fit.DoAllFit())
            Fit.PutFittedArray(ImOut)
            
        Islands.FitIm=ImOut
        return sourceList


    def FitParallel(self,psf,incr,StdResidual):

        NCPU=self.NCPU
        PMin,PMaj,PPA=psf
        
        Islands=self.Islands

        ImOut=np.zeros(Islands.MaskImage.shape,np.float32)
        
        sourceList=[]

        work_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        NJobs=len(Islands.ListX)
        for iJob in range(NJobs):
            work_queue.put([iJob,np.array(Islands.ListX[iJob]),np.array(Islands.ListY[iJob]),np.array(Islands.ListS[iJob])])

        workerlist=[]
        for ii in range(NCPU):
            W=Worker(work_queue, result_queue,psf,incr,StdResidual)
            workerlist.append(W)
            workerlist[ii].start()

        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="      Init W ", HeaderSize=10,TitleSize=13)
        pBAR.render(0, '%4i/%i' % (0,NJobs))
        iResult=0


        SourceList=[]
        while iResult < NJobs:
            DicoResult=result_queue.get()
            if DicoResult["Success"]:
                iResult+=1
            NDone=iResult
            intPercent=int(100*  NDone / float(NJobs))
            pBAR.render(intPercent, '%4i/%i' % (NDone,NJobs))
            SourceList.append(DicoResult["FitPars"])

        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()


        return SourceList




#======================================
import multiprocessing

class Worker(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 psf,incr,StdResidual):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.psf=psf
        self.incr=incr
        self.StdResidual=StdResidual

    def shutdown(self):
        self.exit.set()

    def run(self):

        while not self.kill_received:
            
            try:
                iIsland,xin,yin,zin = self.work_queue.get()
            except:
                break
            
            PMin,PMaj,PPA=self.psf
            incr=self.incr
            StdResidual=self.StdResidual
            
            Fit=ClassFit(xin,yin,zin,psf=(PMaj/incr,PMin/incr,PPA+np.pi/2),noise=StdResidual)
            FitPars=Fit.DoAllFit()
            #Fit.PutFittedArray(ImOut)
            
            self.result_queue.put({"Success":True,"iIsland":iIsland,"FitPars":FitPars})
            
            
            
