'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

from DDFacet.Other import MyLogger
log= MyLogger.getLogger("ClassLOFARBeam")
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor

import numpy as np

class ClassLOFARBeam():
    def __init__(self,MS,GD):
        self.GD=GD
        self.MS=MS
        self.SR=None
        self.InitLOFARBeam()
        self.CalcFreqDomains()

    def InitLOFARBeam(self):
        GD=self.GD
        LOFARBeamMode=GD["Beam"]["LOFARBeamMode"]
        print>>log, "  LOFAR beam model in %s mode"%(LOFARBeamMode)
        useArrayFactor=("A" in LOFARBeamMode)
        useElementBeam=("E" in LOFARBeamMode)
        if self.SR is not None: return

        import lofar.stationresponse as lsr

        self.SR = lsr.stationresponse(self.MS.MSName,
                                      useElementResponse=useElementBeam,
                                      #useElementBeam=useElementBeam,
                                      useArrayFactor=useArrayFactor)#,useChanFreq=True)
        self.SR.setDirection(self.MS.rarad,self.MS.decrad)


    def getBeamSampleTimes(self,times, **kwargs):
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

        NChanJones=self.GD["Beam"]["NBand"]
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

    def GiveRawBeam(self,time,ra,dec):
        #self.LoadSR()
        Beam=np.zeros((ra.shape[0],self.MS.na,self.MS.NSPWChan,2,2),dtype=np.complex)
        for i in range(ra.shape[0]):
            self.SR.setDirection(ra[i],dec[i])
            Beam[i]=self.SR.evaluate(time)
        #Beam=np.swapaxes(Beam,1,2)
        return Beam

    def GiveInstrumentBeam(self,*args,**kwargs):
        
        T=ClassTimeIt.ClassTimeIt("GiveInstrumentBeam")
        T.disable()
        Beam=self.GiveRawBeam(*args,**kwargs)
        nd,na,nch,_,_=Beam.shape
        T.timeit("0")
        MeanBeam=np.zeros((nd,na,self.NChanJones,2,2),dtype=Beam.dtype)
        for ich in range(self.NChanJones):
            indCh=np.where(self.VisToJonesChanMapping==ich)[0]
            MeanBeam[:,:,ich,:,:]=np.mean(Beam[:,:,indCh,:,:],axis=2)
        T.timeit("1")

        return MeanBeam


