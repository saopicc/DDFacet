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

from DDFacet.Other import logger
log= logger.getLogger("ClassATCABeam")
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor

import numpy as np

def AngDist(ra0,dec0,ra1,dec1):
    AC=np.arccos
    C=np.cos
    S=np.sin
    D=S(dec0)*S(dec1)+C(dec0)*C(dec1)*C(ra0-ra1)
    if type(D).__name__=="ndarray":
        D[D>1.]=1.
        D[D<-1.]=-1.
    else:
        if D>1.: D=1.
        if D<-1.: D=-1.
    return AC(D)

class ClassATCABeam():
    def __init__(self,MS,GD):
        self.GD=GD
        self.MS=MS
        self.SR=None
        self.CalcFreqDomains()
        


    def getBeamSampleTimes(self,times, **kwargs):
        DtBeamMin = self.GD["DtBeamMin"]
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

        NChanJones=self.GD["NBand"]
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

        self.calcCoefs()

        
    def calcCoefs(self):
        ChanFreqs=self.MS.ChanFreq.flatten()
        DicoCoefs={}

        self.ATCABeamType="csiro"

        if self.ATCABeamType=="csiro":
            # coefficients for 16cm band of the ATCA, taken from the "+" values of  Tab. 3 in
            # Sec. 4.1.4 of https://www.narrabri.atnf.csiro.au/people/ste616/beamshapes/beamshape_16cm.html
            # coded by Etienne Bonnassieux. Bandwidth for each of these bands is 128 MHz except for
            # edges. Fit valid only up to 100 arcmin.
            DicoCoefs[0]={"Name":"ATCA-16-0", "f0":1268.e6,
                          "C":np.array([1,0,-1.154e-03,0,+5.351e-07,0,-1.265e-10,0,
                                        +1.634e-14,0,-1.099e-18,0,+3.009e-23])}
            DicoCoefs[1]={"Name":"ATCA-16-1", "f0":1396.e6,
                          "C":np.array([1,0,-1.092e-03,0,+4.660e-07,0,-9.875e-11,0,
                                        +1.119e-14,0,-6.504e-19,0,+1.522e-23])}
            DicoCoefs[2]={"Name":"ATCA-16-2", "f0":1524.e6,
                          "C":np.array([1,0,-1.048e-03,0,+4.305e-07,0,-8.824e-11,0,
                                        +9.684e-15,0,-5.446e-19,0,+1.232e-23])}
            DicoCoefs[3]={"Name":"ATCA-16-3", "f0":1652.e6,
                          "C":np.array([1,0,-1.033e-03,0,+4.221e-07,0,-8.692e-11,0,
                                        +9.679e-15,0,-5.571e-19,0,+1.298e-23])}
            DicoCoefs[4]={"Name":"ATCA-16-4", "f0":1780.e6,
                          "C":np.array([1,0,-1.003e-03,0,+4.016e-07,0,-8.170e-11,0,
                                        +9.046e-15,0,-5.199e-19,0,+1.213e-23])}
            DicoCoefs[5]={"Name":"ATCA-16-5", "f0":1908.e6,
                          "C":np.array([1,0,-9.794e-04,0,+3.831e-07,0,-7.562e-11,0,
                                        +8.067e-15,0,-4.440e-19,0,+9.882e-24])}
            DicoCoefs[6]={"Name":"ATCA-16-6", "f0":2036.e6,
                          "C":np.array([1,0,-9.692e-04,0,+3.848e-07,0,-7.809e-11,0,
                                        +8.621e-15,0,-4.927e-19,0,+1.140e-23])}
            DicoCoefs[7]={"Name":"ATCA-16-7", "f0":2164.e6,
                          "C":np.array([1,0,-9.919e-04,0,+4.028e-07,0,-8.251e-11,0,
                                        +9.077e-15,0,-5.122e-19,0,+1.163e-23])}
            DicoCoefs[8]={"Name":"ATCA-16-8", "f0":2292.e6,
                          "C":np.array([1,0,-1.007e-03,0,+4.131e-07,0,-8.480e-11,0,
                                        +9.305e-15,0,-5.220e-19,0,+1.176e-23])}
            DicoCoefs[9]={"Name":"ATCA-16-9", "f0":2420.e6,
                          "C":np.array([1,0,-1.016e-03,0,+4.235e-07,0,-8.817e-11,0,
                                        +9.772e-15,0,-5.521e-19,0,+1.251e-23])}
            DicoCoefs[10]={"Name":"ATCA-16-10", "f0":2548.e6,
                           "C":np.array([1,0,-1.033e-03,0,+4.375e-07,0,-9.223e-11,0,
                                         +1.033e-14,0,-5.885e-19,0,+1.343e-23])}
            DicoCoefs[11]={"Name":"ATCA-16-11", "f0":2676.e6,
                           "C":np.array([1,0,-1.035e-03,0,+4.395e-07,0,-9.269e-11,0,
                                         +1.038e-14,0,-5.919e-19,0,+1.353e-23])}
            DicoCoefs[12]={"Name":"ATCA-16-12", "f0":2804.e6,
                           "C":np.array([1,0,-1.064e-03,0,+4.622e-07,0,-9.913e-11,0,
                                         +1.125e-14,0,-6.502e-19,0,+1.505e-23])}
            DicoCoefs[13]={"Name":"ATCA-16-13", "f0":2932.e6,
                           "C":np.array([1,0,-1.081e-03,0,+4.769e-07,0,-1.045e-10,0,
                                         +1.222e-14,0,-7.325e-19,0,+1.768e-23])}
        if self.ATCABeamType=="gaussian":
            DicoCoefs[0] ={"f0":1.000e9,"f1":1.332e9,"C":np.array([46.53])}
            DicoCoefs[1] ={"f0":1.332e9,"f1":1.460e9,"C":np.array([47.58])}
            DicoCoefs[2] ={"f0":1.460e9,"f1":1.588e9,"C":np.array([48.91])}
            DicoCoefs[3] ={"f0":1.588e9,"f1":1.716e9,"C":np.array([49.35])}
            DicoCoefs[4] ={"f0":1.716e9,"f1":1.844e9,"C":np.array([50.05])}
            DicoCoefs[5] ={"f0":1.844e9,"f1":1.972e9,"C":np.array([50.70])}
            DicoCoefs[6] ={"f0":1.972e9,"f1":2.100e9,"C":np.array([51.22])}
            DicoCoefs[7] ={"f0":2.100e9,"f1":2.228e9,"C":np.array([50.61])}
            DicoCoefs[8] ={"f0":2.228e9,"f1":2.356e9,"C":np.array([50.01])}
            DicoCoefs[9] ={"f0":2.356e9,"f1":2.484e9,"C":np.array([49.66])}
            DicoCoefs[10]={"f0":2.484e9,"f1":2.612e9,"C":np.array([49.61])}
            DicoCoefs[11]={"f0":2.612e9,"f1":2.740e9,"C":np.array([49.48])}
            DicoCoefs[12]={"f0":2.740e9,"f1":2.868e9,"C":np.array([48.81])}
            DicoCoefs[13]={"f0":2.868e9,"f1":3.996e9,"C":np.array([48.31])}

        NBand=len(DicoCoefs)

                
        chanBand0=np.zeros((ChanFreqs.size),dtype=int)
        chanBand1=np.zeros((ChanFreqs.size),dtype=int)
        xNull=np.zeros((ChanFreqs.size,),np.float32)
        yNull=np.zeros((ChanFreqs.size,),np.float32)
        beamfreq0=np.zeros((ChanFreqs.size,),np.float32)
        for ich in range(ChanFreqs.size):
            f=ChanFreqs[ich]
            NoMatch=True
            # by default, set the band ranges to be the biggest available before interpolation.
            band0=[0]
            band1=[NBand-1]
            # narrow down the band range to find the nearest neighbours for interpolation.
            for iBand in DicoCoefs.keys():
                if DicoCoefs[iBand]["f0"]>=f:
                    band1.append(iBand)
                if DicoCoefs[iBand]["f0"]<=f:
                    band0.append(iBand)
            band0,band1=np.max(band0),np.min(band1)
            xNull[ich]=100.
            yNull[ich]=0.0036
            chanBand0[ich]=band0
            chanBand1[ich]=band1
            NoMatch=False
            if NoMatch:
                raise NotImplementedError("The ATCA %s beam is not modeled for that frequency (channel %i @ %.3f MHz)"%(self.ATCABeamType,ich,ChanFreqs[ich]/1e6))
        self.DicoCoefs=DicoCoefs
        self.chanBand0=chanBand0
        self.chanBand1=chanBand1
        self.xNull=xNull
        self.yNull=yNull
        self.beamfreq0=beamfreq0


    def evaluateBeam(self,*args,**kwargs):		
        return self.GiveInstrumentBeam(*args,**kwargs)
    
    def GiveRawBeam(self,time,ra,dec):
        nch=self.MS.ChanFreq.size
        Beam=np.zeros((ra.shape[0],self.MS.na,self.MS.NSPWChan,2,2),dtype=complex)
        rac,decc=self.MS.OriginalRadec
        d=AngDist(ra,dec,rac,decc)*180./np.pi*60
        DicoCoefs=self.DicoCoefs

        for ich in range(nch):
            bf0=self.beamfreq0
            Dnu=d*self.MS.ChanFreq.flat[ich]*1e-9
            band0=self.chanBand0[ich]
            band1=self.chanBand1[ich]
            Dnu0=d*DicoCoefs[band0]["f0"]*1e-9
            Dnu1=d*DicoCoefs[band1]["f0"]*1e-9

            if self.ATCABeamType=="csiro":
                deltafreq0=np.abs(self.MS.ChanFreq.flat[ich]-DicoCoefs[band0]["f0"])
                deltafreq1=np.abs(self.MS.ChanFreq.flat[ich]-DicoCoefs[band1]["f0"])
                if deltafreq0==0 and deltafreq1==0:
                    # This case does not require interpolation
                    B=np.sqrt(np.abs(np.polynomial.polynomial.polyval(Dnu,np.array(DicoCoefs[band0]["C"]))))
                    self.yNull[ich]=np.sqrt(np.abs(np.polynomial.polynomial.polyval(100.,np.array(DicoCoefs[band0]["C"]))))
                else:
                    # Perform interpolation
                    beam0=np.sqrt(np.abs(np.polynomial.polynomial.polyval(Dnu0,np.array(DicoCoefs[band0]["C"]))))
                    beam1=np.sqrt(np.abs(np.polynomial.polynomial.polyval(Dnu1,np.array(DicoCoefs[band1]["C"]))))
                    B = (deltafreq0*beam0 + deltafreq1*beam1) / (deltafreq0 + deltafreq1)
                    beam0=np.sqrt(np.abs(np.polynomial.polynomial.polyval(100.,np.array(DicoCoefs[band0]["C"]))))
                    beam1=np.sqrt(np.abs(np.polynomial.polynomial.polyval(100.,np.array(DicoCoefs[band1]["C"]))))
                    self.yNull[ich] = (deltafreq0*beam0 + deltafreq1*beam1) / (deltafreq0 + deltafreq1)
                B[Dnu>self.xNull[ich]]=self.yNull[ich]/( (Dnu[Dnu>self.xNull[ich]]-self.yNull[ich]) )
            B=B.reshape((-1,1))
            Beam[:,:,ich,0,0]=B[:,:]
            Beam[:,:,ich,1,1]=B[:,:]
        

        return Beam

    def GiveInstrumentBeam(self,*args,**kwargs):
        
        T=ClassTimeIt.ClassTimeIt("GiveInstrumentBeam")
        T.disable()
        Beam=self.GiveRawBeam(*args,**kwargs)
        nd,na,nch,_,_=Beam.shape
        T.timeit("0")
        MeanBeam=np.zeros((nd,na,self.NChanJones,2,2),dtype=Beam.dtype)
        #L_ichZero=[]
        for ich in range(self.NChanJones):
            indCh=np.where(self.VisToJonesChanMapping==ich)[0]
            # if indCh.size==0:
            #     L_ichZero.append(ich)
            #     continue
            
            MeanBeam[:,:,ich,:,:]=np.mean(Beam[:,:,indCh,:,:],axis=2)

        # # to deal with this weird MS
        # print(self.VisToJonesChanMapping)
        # print(self.VisToJonesChanMapping)
        # print(self.VisToJonesChanMapping)
        # if len(L_ichZero)>0:
        #     M=np.mean(MeanBeam[:,:,:,0,0],axis=-1)
        #     for ich in L_ichZero:
        #         for iAnt in range(na):
        #             for iDir in range(nd):
        #                 j=M[iDir,iAnt]
        #                 MeanBeam[iDir,iAnt,ich,0,0]=j
        #                 MeanBeam[iDir,iAnt,ich,1,1]=j
            
        T.timeit("1")

        return MeanBeam

