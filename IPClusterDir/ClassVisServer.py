import numpy as np
import ClassMS
from pyrap.tables import table
import ClassWeighting
import MyLogger
log=MyLogger.getLogger("ClassVisServer")
import MyPickle

class ClassVisServer():
    def __init__(self,GD,MSName):
        #self.MDC=MDC
        self.GD=GD
        self.ReInitChunkCount()
        self.TChunkSize=self.GD.DicoConfig["Facet"]["TChunkSize"]
        self.MSName=MSName
        self.Init()
        self.VisWeights=None
        self.CountPickle=0

    def Init(self,PointingID=0):
        #MSName=self.MDC.giveMS(PointingID).MSName
        MS=ClassMS.ClassMS(self.MSName,Col=self.GD.DicoConfig["Files"]["ColName"],DoReadData=False)
        TimesInt=np.arange(0,MS.DTh,self.TChunkSize).tolist()
        if not(MS.DTh in TimesInt): TimesInt.append(MS.DTh)
        self.TimesInt=TimesInt
        self.NTChunk=len(self.TimesInt)-1
        self.MS=MS

    def ReInitChunkCount(self):
        self.CurrentTimeChunk=0

    def CalcWeigths(self,ImShape,CellSizeRad):
        if self.VisWeights!=None: return

        WeightMachine=ClassWeighting.ClassWeighting(ImShape,CellSizeRad)
        uvw=self.GiveAllUVW()
        VisWeights=np.ones((uvw.shape[0],),dtype=np.float32)
        Robust=self.GD.DicoConfig['Facet']["Robust"]
        self.VisWeights=WeightMachine.CalcWeights(uvw,VisWeights,Robust=Robust)
        
        #self.VisWeights.fill(1)



    def GiveNextVisChunk(self):
        if self.CurrentTimeChunk==self.NTChunk:
            print>>log, "Reached end of chunks"
            self.ReInitChunkCount()
            return None
        MS=self.MS
        iT0,iT1=self.CurrentTimeChunk,self.CurrentTimeChunk+1
        self.CurrentTimeChunk+=1

        print>>log, "Reading next data chunk in [%5.2f, %5.2f] hours"%(self.TimesInt[iT0],self.TimesInt[iT1])
        MS.ReadData(t0=self.TimesInt[iT0],t1=self.TimesInt[iT1])
        #print>>log, "    Rows= [%i, %i]"%(MS.ROW0,MS.ROW1)
        #print float(MS.ROW0)/MS.nbl,float(MS.ROW1)/MS.nbl
        DATA=self.GiveVisChunk()
        W=self.VisWeights[MS.ROW0:MS.ROW1]
        DATA["Weights"]=W
        return DATA

    def GiveAllUVW(self):
        t=table(self.MS.MSName,ack=False)
        uvw=t.getcol("UVW")
        t.close()
        return uvw

    def GiveVisChunk(self,it0=0,it1=-1):
        MS=self.MS

        row0=it0*MS.nbl
        row1=it1*MS.nbl
        if it1==-1:
            row1=None

        times=MS.times_all[row0:row1]
        data=MS.data[row0:row1]
        A0=MS.A0[row0:row1]
        A1=MS.A1[row0:row1]
        uvw=MS.uvw[row0:row1]
        flags=MS.flag_all[row0:row1]
        freqs=MS.ChanFreq.flatten()
        nbl=MS.nbl

        Field=self.GD.DicoConfig["Select"]["UVRangeKm"]
        if Field!=None:
            d0,d1=Field
            d0*=1e3
            d1*=1e3
            u,v,w=MS.uvw.T
            duv=np.sqrt(u**2+v**2)
            ind=np.where((duv<d0)|(duv>d1))[0]
            flags[ind,:,:]=1

        
        DicoDataOut={"itimes":(it0,it1),
                     "times":times,#[ind],
                     "freqs":freqs,
                     #"A0A1":(A0[ind],A1[ind]),
                     "A0A1":(A0,A1),
                     "uvw":uvw,#[ind],
                     "flags":flags,#[ind],
                     "nbl":nbl,
                     "data":data#[ind]
                     }
        
        #MyPickle.Save(DicoDataOut,"Pickle_All_%2.2i"%self.CountPickle)
        #self.CountPickle+=1

        return DicoDataOut
