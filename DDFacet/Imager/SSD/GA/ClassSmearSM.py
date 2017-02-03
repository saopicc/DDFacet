
import numpy as np
import scipy.signal
import ClassConvMachine
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassSmearSM")
import multiprocessing
from DDFacet.Array import NpShared
from DDFacet.Array import ModLinAlg
import time
from DDFacet.Other.progressbar import ProgressBar

class ClassSmearSM():
    def __init__(self,MeanResidual,MeanModelImage,PSFServer,DeltaChi2=4.,IdSharedMem="",NCPU=6):
        IdSharedMem+="SmearSM."
        NpShared.DelAll(IdSharedMem)
        self.IdSharedMem=IdSharedMem
        self.NCPU=NCPU
        self.MeanModelImage=NpShared.ToShared("%sMeanModelImage"%self.IdSharedMem,MeanModelImage)
        self.MeanResidual=NpShared.ToShared("%sMeanResidual"%self.IdSharedMem,MeanResidual)
        NPixStats=10000
        RandomInd=np.int64(np.random.rand(NPixStats)*(MeanResidual.size))
        self.RMS=np.std(np.real(self.MeanResidual.ravel()[RandomInd]))
        self.FWHMMin=3.

        self.PSFServer=PSFServer
        self.DeltaChi2=DeltaChi2
        self.Var=self.RMS**2
        self.NImGauss=31
        self.CubeMeanVariablePSF=NpShared.ToShared("%sCubeMeanVariablePSF"%self.IdSharedMem,self.PSFServer.DicoVariablePSF['CubeMeanVariablePSF'])
        self.DicoConvMachine={}

        N=self.NImGauss
        dx,dy=np.mgrid[-(N/2):N/2:1j*N,-(N/2):N/2:1j*N]

        ListPixParms=[(int(dx.ravel()[i]),int(dy.ravel()[i])) for i in range(dx.size)]
        ListPixData=ListPixParms
        ConvMode="Matrix"
        N=self.NImGauss


        #stop
        #for 
        #ClassConvMachine():
        #def __init__(self,PSF,ListPixParms,ListPixData,ConvMode):


        d=np.sqrt(dx**2+dy**2)
        self.dist=d
        self.NGauss=10

        GSig=np.linspace(0.,2,self.NGauss)
        self.GSig=GSig
        ListGauss=[]
        One=np.zeros_like(d)
        One[N/2,N/2]=1.
        ListGauss.append(One)
        for sig in GSig[1::]:
            v=np.exp(-d**2/(2.*sig**2))
            Sv=np.sum(v)
            v/=Sv
            ListGauss.append(v)

        self.ListGauss=ListGauss
        
        print>>log, "Declare convolution machines"
        NJobs=self.PSFServer.NFacets
        pBAR= ProgressBar(Title=" Declare      ")
        #pBAR.disable()
        pBAR.render(0, '%4i/%i' % (0,NJobs))
        for iFacet in range(self.PSFServer.NFacets):
            #print iFacet,"/",self.PSFServer.NFacets
            PSF=self.PSFServer.DicoVariablePSF['CubeMeanVariablePSF'][iFacet]#[0,0]
            _,_,NPixPSF,_=PSF.shape
            PSF=PSF[:,:,NPixPSF/2-N:NPixPSF/2+N+1,NPixPSF/2-N:NPixPSF/2+N+1]
            #print PSF.shape
            #sig=1
            #PSF=(np.exp(-self.dist**2/(2.*sig**2))).reshape(1,1,N,N)

            self.DicoConvMachine[iFacet]=ClassConvMachine.ClassConvMachine(PSF,ListPixParms,ListPixData,ConvMode)
            CM=self.DicoConvMachine[iFacet].CM
            NpShared.ToShared("%sCM_Facet%4.4i"%(self.IdSharedMem,iFacet),CM)
            #invCM=ModLinAlg.invSVD(np.float64(CM[0,0]))/self.Var
            #NpShared.ToShared("%sInvCov_Facet%4.4i"%(self.IdSharedMem,iFacet),invCM)

            NDone=iFacet+1
            intPercent=int(100*  NDone / float(NJobs))
            pBAR.render(intPercent, '%4i/%i' % (NDone,NJobs))


        PSFMean=np.mean(self.PSFServer.DicoVariablePSF['CubeMeanVariablePSF'],axis=0)
        self.ConvMachineMeanPSF=ClassConvMachine.ClassConvMachine(PSFMean,ListPixParms,ListPixData,ConvMode)
        CM=self.ConvMachineMeanPSF.CM
        invCM=ModLinAlg.invSVD(np.float64(CM[0,0]),Cut=1e-8)/self.Var
        NpShared.ToShared("%sInvCov_AllFacet"%(self.IdSharedMem),invCM)
        self.FindSupport()
    

    def CleanUpSHM(self):
        NpShared.DelAll(self.IdSharedMem)


    def FindSupport(self):
        ConvMachine=self.CurrentConvMachine=self.ConvMachineMeanPSF
        N=self.NImGauss
        Dirty=np.zeros((N,N),dtype=np.float32)
        Dirty[N/2,N/2]=1.
        Dirty=self.CurrentConvMachine.Convolve(Dirty.reshape(1,Dirty.size)).reshape((N,N))
        InvCov=ConvMachine.GiveInvertCov(1.)
        Sol=np.dot(InvCov,Dirty.reshape((Dirty.size,1))).reshape((N,N))
        Profile=(Sol[N/2,:]+Sol[:,N/2])[N/2:]
        
        xp=np.arange(N/2+1)
        Val=np.max(Profile)/2.
        xx=np.linspace(0,N/2,1000)
        a=np.interp(xx,xp,Profile-Val)
        ind=np.where(np.abs(a)==np.min(np.abs(a)))[0]
        FWHM=a[ind[0]]*2
        if FWHM<self.FWHMMin: FWHM=self.FWHMMin
        self.RestoreFWHM=FWHM
        self.SigMin=(FWHM/2.)/np.sqrt(2.*np.log(2.))

        RestoringBeam=(np.exp(-self.dist**2/(2.*self.SigMin**2))).reshape(N,N)
        
        ListRestoredGauss=[]

        for Func in self.ListGauss:
            v=scipy.signal.fftconvolve(Func, RestoringBeam, mode='same')
            ListRestoredGauss.append(v)
        self.ListRestoredGauss=ListRestoredGauss
        print>>log, "Support for restoring beam: %5.2f pixels (sigma = %5.2f pixels)"%(FWHM,self.SigMin)
        
        # import pylab
        # pylab.clf()
        # pylab.plot(xp,Profile)
        # pylab.scatter(xx,a)
        # # pylab.subplot(1,2,1)
        # # pylab.imshow(Dirty,interpolation="nearest")
        # # pylab.colorbar()
        # # vmax=Sol.max()
        # # pylab.subplot(1,2,2)
        # # pylab.imshow(Sol,interpolation="nearest",vmax=vmax,vmin=-0.1*vmax)
        # # pylab.colorbar()
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop

    def Smear(self,Parallel=True):
        if Parallel:
            NCPU=self.NCPU
        else:
            NCPU=1
        StopWhenQueueEmpty=True
        print>>log, "Building queue"
        self.ModelOut=np.zeros_like(self.MeanModelImage)
        indx,indy=np.where(self.MeanModelImage[0,0]!=0)
        #indx,indy=np.where(self.MeanModelImage==np.max(self.MeanModelImage))
        work_queue = multiprocessing.Queue()
        result_queue=multiprocessing.Queue()

        SizeMax=int(indx.size/float(NCPU)/100.)
        SizeMax=np.max([SizeMax,1])
        iPix=0
        iQueue=0
        Queue=[]
        while iPix<indx.size:
            xc,yc=indx[iPix],indy[iPix]
            FacetID=self.PSFServer.giveFacetID2(xc,yc)
            #DicoOrder={"xy":(xc,yc),
            #           "FacetID":FacetID}
            Queue.append([xc,yc,FacetID])

            iPix+=1
            if (len(Queue)==SizeMax)|(iPix==indx.size):
                NpShared.ToShared("%sQueue_%3.3i"%(self.IdSharedMem,iQueue),np.array(Queue))
                work_queue.put(iQueue)
                Queue=[]
                iQueue+=1


        NJobs=indx.size



        workerlist=[]

        pBAR= ProgressBar(Title=" Find gaussian")
        #pBAR.disable()
        pBAR.render(0, '%4i/%i' % (0,NJobs))
        for ii in range(NCPU):
            W=WorkerSmear(work_queue, 
                          result_queue,
                          IdSharedMem=self.IdSharedMem,
                          StopWhenQueueEmpty=StopWhenQueueEmpty,
                          NImGauss=self.NImGauss,
                          DeltaChi2=self.DeltaChi2,
                          ListGauss=self.ListGauss,
                          GSig=self.GSig,
                          Var=self.Var,
                          SigMin=self.SigMin)
            workerlist.append(W)
            if Parallel:
                workerlist[ii].start()
            else:
                workerlist[ii].run()

            
        N=self.NImGauss
        iResult=0
        #print "!!!!!!!!!!!!!!!!!!!!!!!!",iResult,NJobs
        while iResult < NJobs:
            DicoResult=None
            # for result_queue in List_Result_queue:
            #     if result_queue.qsize()!=0:
            #         try:
            #             DicoResult=result_queue.get_nowait()
                        
            #             break
            #         except:
                        
            #             pass
            #         #DicoResult=result_queue.get()
            #print "!!!!!!!!!!!!!!!!!!!!!!!!! Qsize",result_queue.qsize()
            #print work_queue.qsize(),result_queue.qsize()
            if result_queue.qsize()!=0:
                try:
                    DicoResult=result_queue.get_nowait()
                except:
                    pass
                    #DicoResult=result_queue.get()


            if DicoResult is None:
                time.sleep(0.001)
                continue



            if DicoResult["Success"]:
                iQueue=DicoResult["iQueue"]
                Queue=NpShared.GiveArray("%sQueue_%3.3i"%(self.IdSharedMem,iQueue))
                for iJob in range(Queue.shape[0]):
                    x0,y0,iGauss=Queue[iJob]
                    SMax=self.MeanModelImage[0,0,x0,y0]
                    SubModelOut=self.ModelOut[0,0][x0-N/2:x0+N/2+1,y0-N/2:y0+N/2+1]
                    SubModelOut+=self.ListRestoredGauss[iGauss]*SMax
                    #iGauss=0
                    #print
                    #print SMax
                    #print np.sum(self.ListGauss[iGauss])
                    #print
                    SubModelOut+=self.ListGauss[iGauss]*SMax

                    iResult+=1
                    NDone=iResult
                    intPercent=int(100*  NDone / float(NJobs))
                    pBAR.render(intPercent, '%4i/%i' % (NDone,NJobs))



        for ii in range(NCPU):
            try:
                workerlist[ii].shutdown()
                workerlist[ii].terminate()
                workerlist[ii].join()
            except:
                pass
        
        return self.ModelOut



# ##############################################################################################################
# ##############################################################################################################
# ##############################################################################################################
# ##############################################################################################################


class WorkerSmear(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 IdSharedMem=None,
                 StopWhenQueueEmpty=False,
                 NImGauss=31,
                 DeltaChi2=4.,
                 ListGauss=None,
                 GSig=None,
                 SigMin=None,
                 Var=None):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.IdSharedMem=IdSharedMem
        self.StopWhenQueueEmpty=StopWhenQueueEmpty
        self.CubeMeanVariablePSF=NpShared.GiveArray("%sCubeMeanVariablePSF"%self.IdSharedMem)
        self.MeanModelImage=NpShared.GiveArray("%sMeanModelImage"%self.IdSharedMem)
        self.MeanResidual=NpShared.GiveArray("%sMeanResidual"%self.IdSharedMem)
        self.NImGauss=NImGauss
        self.DeltaChi2=DeltaChi2
        self.ListGauss=ListGauss
        self.NGauss=len(ListGauss)
        self.GSig=GSig
        self.SigMin=SigMin
        self.Var=Var

    def shutdown(self):
        self.exit.set()

    def CondContinue(self):
        if self.StopWhenQueueEmpty:
            return not(self.work_queue.qsize()==0)
        else:
            return True

    def GiveChi2(self,Resid):
        Chi2=np.sum(Resid**2)/self.Var
        return Chi2
        InvCov=self.CurrentInvCov#ConvMachine.GiveInvertCov(self.Var)
        NPixResid=Resid.size
        return np.dot(np.dot(Resid.reshape((1,NPixResid)),InvCov),Resid.reshape((NPixResid,1))).ravel()[0]
            
 
    def GiveConv(self,SubModelOrig):
        N=self.NImGauss
        #ConvModel=self.CurrentConvMachine.Convolve(SubModelOrig.reshape(1,SubModelOrig.size)).reshape((N,N))
        ConvModel=np.dot(self.CurrentCM[0,0],SubModelOrig.reshape((SubModelOrig.size,1))).reshape((N,N))
        return ConvModel


    def SmearThisComp(self,x0,y0):
        FacetID=self.CurrentFacetID
        PSF=self.CubeMeanVariablePSF[FacetID][0,0]
        N=self.NImGauss
        SubResid=self.MeanResidual[0,0][x0-N/2:x0+N/2+1,y0-N/2:y0+N/2+1]
        SubModelOrig=self.MeanModelImage[0,0][x0-N/2:x0+N/2+1,y0-N/2:y0+N/2+1].copy()
        xc=yc=N/2

        NPSF,_=PSF.shape
        
        xcPSF=ycPSF=NPSF/2
        SubPSF=PSF[xcPSF-N/2:xcPSF+N/2+1,ycPSF-N/2:ycPSF+N/2+1]
        #SubModelOrig.fill(0)
        #SubModelOrig[N/2,N/2]=1
        #SubModelOrig[N/2+10,N/2+10]=10
        #ConvModel=self.CurrentConvMachine.Convolve(SubModelOrig.reshape(1,SubModelOrig.size)).reshape((N,N))
        #ConvModel1=scipy.signal.fftconvolve(SubModelOrig, SubPSF, mode='same')
        ConvModel=self.GiveConv(SubModelOrig)

        Dirty=SubResid+ConvModel
        
        # for i in range(10):
        #     Noise=self.GiveConv(np.random.randn(*(Dirty.shape)))
        #     #Noise*=1e-1/np.max(Noise)
        #     #Dirty+=Noise
        #     Dirty=ConvModel#+Noise
        #     InvCov=self.CurrentConvMachine.GiveInvertCov(1.)#self.Var)
        #     Sol=np.dot(InvCov,Dirty.reshape((Dirty.size,1))).reshape((N,N))
        #     #Sol=np.dot(InvCov,Sol.reshape((Dirty.size,1))).reshape((N,N))
        #     import pylab
        #     pylab.clf()
        #     pylab.subplot(1,2,1)
        #     pylab.imshow(Dirty,interpolation="nearest")
        #     pylab.colorbar()
        #     vmax=Sol.max()
        #     pylab.subplot(1,2,2)

        #     pylab.imshow(Sol,interpolation="nearest",vmax=vmax,vmin=-0.1*vmax)
        #     pylab.colorbar()
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        #     stop


        



        DeltaChi2=self.DeltaChi2
        #Chi2Min=np.sum(SubResid**2)/self.Var
        Chi2Min=self.GiveChi2(SubResid)
        
        SMax=SubModelOrig[xc,yc]#self.MeanModelImage[x0,y0]
        SubModel0=SubModelOrig.copy()
        SubModel0[xc,yc]=0

        iGauss=0
        Chi2=Chi2Min
        
        while True:
            if iGauss==self.NGauss-1:
                #print "max size"
                break
            # if self.GSig[iGauss]<self.SigMin:
            #     iGauss+=1
            #     continue

            v=self.ListGauss[iGauss]
            Add=v*SMax
            
            ModifiedSubModel=SubModel0+Add
            # import pylab

            # pylab.subplot(1,2,1)
            # pylab.imshow(SubModelOrig,interpolation="nearest")
            # pylab.subplot(1,2,2)
            # pylab.imshow(ModifiedSubModel,interpolation="nearest")
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            # if sig!=0:
            #     stop


            # ModifiedSubModel.fill(0)
            # ModifiedSubModel[xc,xc]=1.
            #ConvModel=scipy.signal.fftconvolve(ModifiedSubModel, SubPSF, mode='same')
            ConvModel=self.GiveConv(ModifiedSubModel)

            ThisDirty=ConvModel
            ThisResid=Dirty-ThisDirty
            #Chi2=np.sum((ThisResid)**2)/self.Var

            #Chi2=np.dot(np.dot(ThisResid.reshape((1,NPixResid)),InvCov),ThisResid.reshape((NPixResid,1)))
            Chi2=self.GiveChi2(ThisResid)#/Chi2Min


            #print Chi2,Chi2Min,Chi2Min+DeltaChi2

            # print sig,Chi2,Chi2Min

            # # try:
            # import pylab
            # vmin,vmax=SubResid.min(),SubResid.max()
            # pylab.subplot(1,3,1)
            # pylab.imshow(SubResid,interpolation="nearest",vmin=vmin,vmax=vmax)
            # pylab.subplot(1,3,2)
            # pylab.imshow(ThisDirty,interpolation="nearest")#,vmin=vmin,vmax=vmax)
            # pylab.subplot(1,3,3)
            # pylab.imshow(ThisResid,interpolation="nearest")#,vmin=vmin,vmax=vmax)
            # pylab.title("%f"%Chi2)
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            # # except:
            # #     stop

            if Chi2/Chi2Min> DeltaChi2:#Chi2Min+DeltaChi2:
            #if Chi2> DeltaChi2:
                #print "reached max chi2 %f"%self.GSig[iGauss]
                break

            iGauss+=1

            #import time
            #time.sleep(0.5)


        # import pylab
        # #vmin,vmax=SubResid.min(),SubResid.max()
        # pylab.imshow(ModifiedSubModel,interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        if self.GSig[iGauss]<self.SigMin:
            iGauss=0

        return iGauss


    def run(self):
        while not self.kill_received and self.CondContinue():
            #gc.enable()
            try:
                iQueue = self.work_queue.get_nowait()#(True,2)
            except Exception,e:
                #print "Exception worker: %s"%str(e)
                break

            #print "Start %i"%iQueue

            Queue=NpShared.GiveArray("%sQueue_%3.3i"%(self.IdSharedMem,iQueue))
            self.CurrentInvCov=NpShared.GiveArray("%sInvCov_AllFacet"%(self.IdSharedMem))
            
            for iJob in range(Queue.shape[0]):
                x0,y0,FacetID=Queue[iJob]
                
                iFacet=FacetID
                self.CurrentFacetID=FacetID
                #self.CurrentCF=NpShared.GiveArray("%sConvMatrix_Facet_%4.4i"%(self.IdSharedMem,iFacet))
                self.CurrentCM=NpShared.GiveArray("%sCM_Facet%4.4i"%(self.IdSharedMem,iFacet))
                #self.CurrentInvCov=NpShared.GiveArray("%sInvCov_Facet%4.4i"%(self.IdSharedMem,iFacet))
                # if self.CurrentInvCov is None:
                #     invCM=ModLinAlg.invSVD(np.float64(self.CurrentCM[0,0]))/self.Var
                #     self.CurrentInvCov=NpShared.ToShared("%sInvCov_Facet%4.4i"%(self.IdSharedMem,iFacet),invCM)

                iGauss=self.SmearThisComp(x0,y0)
                Queue[iJob,2]=iGauss
            
            

            self.result_queue.put({"Success":True,"iQueue":iQueue})

