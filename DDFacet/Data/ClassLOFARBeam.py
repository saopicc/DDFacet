
from DDFacet.Other import MyLogger
log= MyLogger.getLogger("ClassLOFARBeam")

import numpy as np

class ClassLOFARBeam():
    def __init__(self,MS,GD):
        self.GD=GD
        self.MS=MS
        self.InitLOFARBeam()
        self.CalcFreqDomains()


        

    def InitLOFARBeam(self):
        GD=self.GD
        LOFARBeamMode=GD["Beam"]["LOFARBeamMode"]
        print>>log, "  LOFAR beam model in %s mode"%(LOFARBeamMode)
        #self.BeamMode,self.DtBeamMin,self.BeamRAs,self.BeamDECs = LofarBeam
        useArrayFactor=("A" in LOFARBeamMode)
        useElementBeam=("E" in LOFARBeamMode)
        self.MS.LoadSR(useElementBeam=useElementBeam,useArrayFactor=useArrayFactor)

    def getBeamSampleTimes(self,times):
        DtBeamMin = self.GD["Beam"]["DtBeamMin"]
        DtBeamSec = DtBeamMin*60
        tmin=times[0]
        tmax=times[-1]+1
        TimesBeam=np.arange(tmin,tmax,DtBeamSec).tolist()
        if not(tmax in TimesBeam): TimesBeam.append(tmax)
        return TimesBeam

    def getFreqDomains(self):
        return self.FreqDomains

    def CalcFreqDomains(self):
        ChanWidth=self.MS.ChanWidth.ravel()[0]
        ChanFreqs=self.MS.ChanFreq.flatten()

        NChanJones=self.GD["Beam"]["NChanBeamPerMS"]
        if NChanJones==0:
            NChanJones=self.MS.NSPWChan
        ChanEdges=np.linspace(ChanFreqs.min()-ChanWidth/2.,ChanFreqs.max()+ChanWidth/2.,NChanJones+1)

        FreqDomains=[[ChanEdges[iF],ChanEdges[iF+1]] for iF in range(NChanJones)]
        FreqDomains=np.array(FreqDomains)
        self.FreqDomains=FreqDomains
        self.NChanJones=NChanJones

        MeanFreqJonesChan=(FreqDomains[:,0]+FreqDomains[:,1])/2.
        DFreq=np.abs(self.MS.ChanFreq.reshape((self.MS.NSPWChan,1))-MeanFreqJonesChan.reshape((1,NChanJones)))
        self.VisToJonesChanMapping=np.argmin(DFreq,axis=1)

    def GiveInstrumentBeam(self,*args,**kwargs):

        Beam=self.MS.GiveBeam(*args,**kwargs)
        nd,na,nch,_,_=Beam.shape
        
        MeanBeam=np.zeros((nd,na,self.NChanJones,2,2),dtype=Beam.dtype)
        for ich in range(self.NChanJones):
            indCh=np.where(self.VisToJonesChanMapping==ich)[0]
            MeanBeam[:,:,ich,:,:]=np.mean(Beam[:,:,indCh,:,:],axis=2)

        return MeanBeam

