
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

        
class InterfMS():

    def __init__(self,EngClass,CatMS,GD,SM=None,ChanSel=None,ColName="DATA"):
        ind=np.ones((CatMS.shape[0],),bool)
        for iMS,host in zip(range(CatMS.shape[0]),CatMS.node):
            if host not in EngClass.hostlist.keys():
                ind[iMS]=False
        CatMS=CatMS[ind].copy()
        self.CatMS=CatMS

        self.C=EngClass
        self.V=EngClass.V
        self.rc=EngClass.rc
        self.NEng=len(self.rc)
        CatEngines=np.zeros((500,),dtype=[("node","|S200"),("nodeNum",int),("EngNum",int),("MountedMS","|S200"),
                                          ("Freq",float),("used",bool),("ScreenID","|S200"),
                                          ("available",bool),("ifreq",int),("ChanFreq",np.float32,(256,)),
                                          ("NChan",int),("LoadCPU",float),("isProberCPU",bool)])
        self.CatEngines=CatEngines.view(np.recarray)
        self.CatEngines.available=False
        self.CatEngines.available[self.rc.ids]=True
        self.EngIds=self.C.EngIds
        self.CatEngines.isProberCPU=False
        self.MapEngines()
        self.ChanSel=ChanSel
        self.InitProberCPU()
        self.InitMS(CatMS,SM,ColName=ColName)
        self.updateReachable()
        self.GiveLoads()
        self.GD=GD

    def BuildRemoteRIME(self,MDC):
        log=MyLogger.getLogger("IPCluster.BuildDataContainer")
 
        NPointings=MDC.NPointing
        import ClassTimeIt
        T=ClassTimeIt.ClassTimeIt("IPCluster.BuildRemoteRIME")


        C=self
        C.VAll["MDC"]=MDC
        C.VAll["GD"]=self.GD
        # for (ID,MSnodata,freqs,SM,BLMapping) in zip(range(len(ListMS)),ListMS,ListFreqs,ListSM,ListBLMapping):

        # for ID in MDCnodata.ListID:

        #     MSnodata=MDCnodata.giveMS(ID)
        #     freqs=MDCnodata.giveFreqs(ID)
        #     C.VAll["P_%3.3i_freqs"%ID]=freqs
            
        #     C.VAll["P_%3.3i_uvw"%ID]=MSnodata.uvw
        #     T.timeit("copy uvw %s %s"%(str(MSnodata.uvw.shape),str(MSnodata.uvw.dtype)))
            
        #     C.VAll["P_%3.3i_A0"%ID]=MSnodata.A0
        #     T.timeit("copyA0 %s %s"%(str(MSnodata.A0.shape),str(MSnodata.A0.dtype)))
            
        #     C.VAll["P_%3.3i_A1"%ID]=MSnodata.A1
        #     T.timeit("copyA1 %s %s"%(str(MSnodata.A1.shape),str(MSnodata.A1.dtype)))
            
        #     C.VAll["P_%3.3i_times_all"%ID]=MSnodata.times_all
        #     T.timeit("copy time%s %s"%(str(MSnodata.times_all.shape),str(MSnodata.times_all.dtype)))
            
        #     #C.VAll["P_%3.3i_BLMapping"%ID]=BLMapping
        #     #T.timeit("copy BLMapping %s %s"%(str(BLMapping.shape),str(BLMapping.dtype)))
            
            
        #     #C.VAll["P_%3.3i_flag_all"%ID]=np.int32(MSdata.flag_all)
        #     #T.timeit("copy flags %s %s"%(str(MSdata.flag_all.shape),str(MSdata.flag_all.dtype)))
            
        #     #SM=MDCnodata.giveSM(ID)
        #     #C.VAll["P_%3.3i_SM"%ID]=SM
        #     #T.timeit("copy SM")
            
        C.VAll["NPointings"]=NPointings
        
        C.VAll.execute("from ClassME import MeasurementEquation; MME=MeasurementEquation()")

        HYPERCAL_DIR=os.environ["HYPERCAL_DIR"]
        # strExec="%s/HyperCal/Scripts/ScriptSet_MS_SM.py"%HYPERCAL_DIR
        # r=C.VAll.execute('execfile("%s")'%strExec); r.wait(); print>>log,(strExec, r.get())

        strExec="%s/HyperCal/Scripts/ScriptSetMultiRIME.py"%HYPERCAL_DIR
        r=C.VAll.execute('execfile("%s")'%strExec); r.wait(); print>>log, r.get()
        
        r=C.VAll.execute("from PredictDir import ClassHyperH; h=ClassHyperH.ClassHyperH(MME,GD)"); print>>log, r.get()



    def GiveLoads(self):
        log=MyLogger.getLogger("IPCluster.InitProberCPU")
        print>>log, "GiveLoads"
        r=self.VProbe.execute("loadhost=(host,T.AvgLoad)"); r.wait(); print>>log, r.get()
        LLoad= self.VProbe["loadhost"]
        for node,load in LLoad:
            ind =np.where(self.CatEngines.node == node)[0]
            self.CatEngines.LoadCPU[ind]=load
        print>>log, "done GiveLoads"


    def InitProberCPU(self):
        log=MyLogger.getLogger("IPCluster.InitProberCPU")
        print>>log, "InitProberCPU"

        self.rc[:].execute("import ModProbeCPU")
        hostlist=self.C.hostlist.keys()
        for host in hostlist:
            ind=np.where((self.CatEngines.node == host)&(self.CatEngines.MountedMS=="")&(self.CatEngines.isProberCPU==False))[0]
            engsel=ind[0]
            rr=self.rc[engsel].execute('T=ModProbeCPU.TrackCPU()'); rr.wait(); print>>log, rr.get()
            rr=self.rc[engsel].execute('T.start()'); rr.wait(); print>>log, rr.get()
            self.CatEngines.isProberCPU[engsel]=True
        CondisProberCPU=(self.CatEngines.isProberCPU==True)
        EngProbe=np.where(CondisProberCPU)[0]

        if EngProbe.size>0:
            self.VProbe=self.rc[EngProbe.tolist()]
        print>>log, "done InitProberCPU"



    def updateReachable(self):
        log=MyLogger.getLogger("IPCluster.updateReachable")
        
        print>>log, "updateReachable"
        #ListNonReachable=[]#self.C.mapReachable()
        self.CatEngines.available=True
        self.GiveLoads()
        CondReject=(self.CatEngines.LoadCPU>400)
        ListNonReachable=sorted(list(set((self.CatEngines.node[CondReject]).tolist())))
        # self.CatEngines.available[3]=False
        # self.CatEngines.available[6:9]=False

        for i in range(len(ListNonReachable)):
            host=ListNonReachable[i]
            self.CatEngines.available[self.CatEngines.node==host]=False
            print>>log, "host %s is non reachable"%host


        MaskFreq=np.ones(self.freqs.shape,bool)
        MaskFreqMS=np.ones((self.freqs.shape[0],),bool)
        #freqMask=self.CatEngines.Freq[self.CatEngines.available==1]
        #freqMask=self.CatEngines.Freq[(self.CatEngines.available==1)&(self.CatEngines.MountedMS!="")]
        freqMask=self.CatEngines.Freq[(self.CatEngines.MountedMS!="")]
        for i in range(freqMask.shape[0]):
            ind=np.where(np.abs(np.mean(self.freqs,axis=1)-freqMask[i])<.01e6)[0]
            MaskFreq[ind]=False
            MaskFreqMS[ind]=False
        self.MaskFreq=MaskFreq
        self.MaskFreqMS=MaskFreqMS
        
        print "MS available:",freqMask.size
        print "MaskFreq size:",MaskFreq.shape
        print "MaskFreq number falses:",np.where(MaskFreq==False)[0].size
        print "MaskFreqMS size:",MaskFreqMS.shape
        print "MaskFreqMS number falses:",np.where(MaskFreqMS==False)[0].size
        print ""



        CondMS=((self.CatEngines.used==True)&(self.CatEngines.isProberCPU==False))
        CondAv=(self.CatEngines.available==True)&(self.CatEngines.isProberCPU==False)
        #indMSrc=(np.where(CondMS & CondAv)[0])
        indMSrc=(np.where(CondMS)[0])
        FreqSel=self.CatEngines.Freq[indMSrc]
        
        indSortFreq=np.argsort(FreqSel)
        indMSrc=indMSrc[indSortFreq]

        #print ListNonReachable
        #print indMSrc,indMSrc

        self.indMSrc=indMSrc
        self.VMS=self.rc[indMSrc.tolist()]

        indrc=(np.where(CondAv)[0]).tolist()
        self.indrc=indrc
        self.NEng=len(self.indrc)
        self.EngIds=(np.array(self.C.EngIds)[indrc]).tolist()
        self.V=self.rc[indrc]

        CondAll=(self.CatEngines.isProberCPU==False)
        indrcAll=(np.where(CondAll)[0]).tolist()
        self.VAll=self.rc[indrcAll]
        

        print>>log, "done updateReachable"


    def InitMS(self,CatMS,SMname,ColName="DATA"):
        # if self.DoneInit!=True:
        #     self.init_engines(len(LMS))

        log=MyLogger.getLogger("IPCluster.InitMS")

        if CatMS.shape[0]>self.NEng:
            print>>log, "More MS than engines!"
            return
        
        

        # self.FoundMS=False
        # self.CatEngines.MountedMS=""
        if self.FoundMS==False:

            self.rc[:].execute("import ClassMS; import ClassSM; import numpy as np")
            self.rc[:].execute("from progressbar import ProgressBar; ProgressBar.silent=1")
            r=[]
            for i in range(CatMS.shape[0]):
                ind=np.where((self.CatEngines.node == CatMS.node[i])&(self.CatEngines.MountedMS=="")&(self.CatEngines.isProberCPU==False))[0]
                if ind.size==0:
                    print "node is probably MS-full"
                    stop
                engsel=ind[0]


                #rr=self.rc[engsel].execute('import os; os.system("export LD_LIBRARY_PATH=/opt/cep/casacore/builds/casacore-svn21266/trunk/build/lib")'); rr.wait()
                #rr=self.rc[engsel].execute('import sys; path=sys.path; import pyrap.tables'); rr.wait()
                #print>>log, self.rc[engsel].get("path"); print>>log, rr.get()

                rr=self.rc[engsel].execute('import ClassMS; import ClassSM'); rr.wait()
                #rr=self.rc[engsel].execute('path=sys.path'); rr.wait(); print>>log, self.rc[engsel].get("path")
                print>>log, rr.get()
                strexec='MSdata=ClassMS.ClassMS("%s",Col="%s",SelectSPW=[0])'%(CatMS.dirMSname[i],ColName)
                print>>log, strexec
                r.append((engsel,self.rc[engsel].execute(strexec)))
                
                
                self.CatEngines.MountedMS[engsel]=CatMS.dirMSname[i]
                self.CatEngines.used[engsel]=True
                self.CatEngines.node[engsel]=CatMS.node[i]

                    

            for i in range(len(r)):
                engsel,ThisRC=r[i]
                print>>log, (self.CatEngines.node[engsel],"ThisRC.get()")
                ThisRC.wait()
                print>>log, ThisRC.get()
                if self.ChanSel!=None:
                    rr=self.rc[engsel].execute("MSdata.SelChannel(%s)"%str(self.ChanSel)); rr.wait()
                rr=self.rc[engsel].execute('freq=MSdata.Freq_Mean'); rr.wait()
                rr=self.rc[engsel].execute('ChanFreq=MSdata.ChanFreq'); rr.wait()
                #print>>log, rr.get()
                freq=self.rc[engsel].get("freq")
                ChanFreq=self.rc[engsel].get("ChanFreq")
                #print>>log, freq
                self.CatEngines.Freq[engsel]=freq
                self.CatEngines.NChan[engsel]=ChanFreq.size
                self.CatEngines.ChanFreq[engsel][0:ChanFreq.size]=ChanFreq.flatten()

        self.CatEngines=self.CatEngines[self.CatEngines.node!=""]


        CondMS=(self.CatEngines.used==True)
        CondAv=(self.CatEngines.available==True)
        indMSrc=(np.where(CondMS & CondAv)[0]).tolist()
        #indMSrc=(np.where(self.CatEngines.used==True)[0]).tolist()
        self.indMSrc=indMSrc
        

        self.VMS=self.rc[self.indMSrc]
        rr=self.VMS.execute("f=MSdata.ChanFreq.tolist()"); rr.wait(); print>>log, "rr.get()",rr.get()
        freqsDispo=np.array(sorted(np.array(self.VMS.get("f")).flatten().tolist()))

        if freqsDispo.size>1:
            dfreq=np.median(freqsDispo[1::]-freqsDispo[0:-1])
            freqs=np.arange(freqsDispo[0],freqsDispo[-1]+0.01,dfreq)
            
            Nchan=np.max(self.CatEngines.NChan)
            Nbands=freqs.size/Nchan
            freqs=freqs.reshape((Nbands,Nchan))
            self.freqs=freqs
            self.dfreq=dfreq
        else:
            self.freqs=freqsDispo
        
        print>>log, self.freqs


        self.MaskFreq=np.zeros(self.freqs.shape,bool)
        self.updateReachable()

        for i in range(self.freqs.shape[0]):
            ind=np.where(self.CatEngines.Freq==self.freqs[i])[0]
            self.CatEngines.ifreq[ind]=i

        r=self.VMS.execute("nbl=MSdata.nbl; na=MSdata.na"); r.wait()
        self.nbl=self.VMS.get("nbl")[0]
        self.na=self.VMS.get("na")[0]
        r=self.VAll.push({"freqs":self.freqs}); r.wait()
        r=self.VAll.execute('SM=ClassSM.ClassSM("%s")'%SMname); r.wait(); print>>log, r.get()



    def GiveBeam(self,time,RefBeam=False):

        log=MyLogger.getLogger("IPCluster.GiveBeam")
        if not(RefBeam):
            rr=self.VMS.execute("BeamMS=MSdata.GiveBeam(%f,Xp.SM.ClusterCat.ra,Xp.SM.ClusterCat.dec)"%time); rr.wait()
            print>>log, ("loadbeam",rr.get())
        else:
            rr=self.VMS.execute("BeamMS=MSdata.GiveBeam(%f,np.array([Xp.MSdata.rac]),np.array([Xp.MSdata.decc]))"%time); rr.wait()
            print>>log, ("loadbeam",rr.get())
            
        lout=self.VMS["BeamMS"]
            
        null=np.zeros_like(lout[0])

        # pdb> null.shape
        # (1, 1, 53, 2, 2)
        null[:,:,:,0,0]=1
        null[:,:,:,1,1]=1
        for i in range(self.MaskFreqMS.shape[0]):
            if self.MaskFreqMS[i]:
                print>>log, "slot %i is empty"%i
                lout.insert(i,null)

        
        Beam=np.array(lout)#[:,:,:,0,:,:]
        # shape (2, 5, 50, 36, 2, 2)
        print>>log, Beam.shape
        nb,nd, nf, na, _,_=Beam.shape
        Beam=np.swapaxes(Beam,1,2)
        print>>log, Beam.shape
        Beam=Beam.reshape((nb*nf,nd,na,2,2))
        print>>log, Beam.shape
        Beam=np.swapaxes(Beam,0,1)
        print>>log, Beam.shape
        #Beam=np.swapaxes(Beam,1,2)
        #print Beam.shape

        return Beam
    

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

        

    def h(self,x,it0=0,it1=None,Noise=None,ToVec=True,Zero=False,ToRepr=True,CorrectJ=False):
        log=MyLogger.getLogger("IPCluster.h")
        #import objgraph
        #print>>log, objgraph.show_growth(limit=7)
        self.V.push({"out":None})
        j=0
        NToDo=len(x)
        # DoThis=np.int64(np.linspace(0,NToDo,self.NEng+1))
        DoThis=np.int64(sorted(list(set(np.int64(np.linspace(0,NToDo,self.NEng+1)).tolist()))))
        self.V["out"]=None
        
        r=self.V.push({"MaskFreq":self.MaskFreq}); r.wait()
        r=self.V.execute("h.MaskFreq=MaskFreq"); r.wait()
        T=ClassTimeIt.ClassTimeIt("IPCluster.h")
        for ii in range(len(DoThis)-1):
            i=self.indrc[ii]
            self.rc[i].push({"x":x[DoThis[ii]:DoThis[ii+1]],"out":[],"it0":it0,"it1":it1,"Noise":Noise,
                             "Zero":Zero,"ToVec":ToVec,"ToRepr":ToRepr,"CorrectJ":CorrectJ})
            strdo=[]
            ido=0
            for jj in range(DoThis[ii],DoThis[ii+1]):
                #strdo.append("out.append(h(x[%i],it0=it0,it1=it1,Noise=Noise,DoGain=False,ToFFT=False))"%ido)
                #strdo.append("out.append(h(x[%i],it0=it0,it1=it1,Noise=Noise,DoGain=False))"%ido)
                strdo.append("out.append(h.HR(x[%i],it0=it0,it1=it1,Noise=Noise,Zero=Zero,ToRepr=ToRepr,CorrectJ=CorrectJ))"%ido)
                ido+=1
            #print>>log, "engine%i: %s"%(i,strdo)
            r=self.rc[i].execute(";".join(strdo))

            # ###########
            # # debug
            # r.wait()
            # print>>log, r.get()
            # #######

        T.timeit("   IPC: Send order")
        self.rc.wait()
        T.timeit("   IPC: Done Calc")
        listout=self.V.get("out")
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

        self.clear()
        T.timeit("   IPC: Clear")

        return out

    def clear(self):
        i=0
        self.C.V.purge_results("all")
        self.C.rc.purge_results("all")
        self.C.V.client.results.clear()
        self.C.V.client.metadata.clear()
        self.V.purge_results("all")
        self.rc.purge_results("all")
        self.V.client.results.clear()
        self.V.client.metadata.clear()
        self.C.rc.history = []
        self.C.V.history = []       
        self.rc.history = []
        self.V.history = []       
        import gc
        gc.enable()
        gc.collect()



def test():        

    import glob
    import ClassParam

    # import ModPickle
    # S=ModPickle.Load("L74464.pickle")
    # Cat,e=S.GiveRecord()





    ll=glob.glob("/home/tasse/.DISK/MS/SimulWSRT/TestWSR*")
    SkyModel="/media/6B5E-87D0/TestIPCluster4/ModelTwo.txt"
 
    ll=glob.glob("/home/tasse/.DISK/TestFilterWRST/qmc2c_30s_spw0.MS.simul")
    ll=glob.glob("/home/tasse/.DISK/TestFilterWRST/qmc2c_30s_spw0.MS")
    SkyModel="/home/tasse/.DISK/TestFilterWRST/qmc2.bbs.npy"
 




    print ModColor.Str("  ... Readind MS(s):")
    print ll
    print ModColor.Str("  Skymodel: "),SkyModel

    nodes=["igor"]*len(ll)

    Cat=np.zeros((len(ll),),dtype=[("node","|S200"),("dirMSname","|S200")])
    Cat=Cat.view(np.recarray)
    Cat.node=np.array(nodes)
    Cat.dirMSname=np.array(ll)

    #Cat=Cat[50:200:3]


    C=Engine()
    #MS=InterfMS(C,Cat,"/home/tasse/IF/Send/3C295_TWO_MSSS.skymodel")


#    MS=InterfMS(C,Cat,SkyModel,ChanSel=(6,60,2),ColName="CORRECTED_DATA")
    MS=InterfMS(C,Cat,SkyModel,ChanSel=(6,60,2),ColName="CORRECTED_DATA")

    return MS

