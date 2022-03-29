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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import numpy as np

class ClassSpectralFunctions():
    def __init__(self,DicoMappingDesc,BeamEnable=True,RefFreq=None):

        self.DicoMappingDesc=DicoMappingDesc
        self.NFreqBand=len(self.DicoMappingDesc["freqs"])
        self.BeamEnable=BeamEnable
        self.RefFreq=RefFreq
        #self.setFreqs()
        self.DicoBeamFactors={}

    # def setFreqs(self):
    #     AllFreqs=[]
    #     AllFreqsMean=np.zeros((self.NFreqBand,),np.float32)
    #     for iChannel in range(self.NFreqBand):
    #         AllFreqs+=self.DicoMappingDesc["freqs"][iChannel]
    #         AllFreqsMean[iChannel]=np.mean(self.DicoMappingDesc["freqs"][iChannel])
    #     RefFreq=np.sum(AllFreqsMean.ravel()*self.DicoMappingDesc["WeightChansImages"].ravel())
    #     self.AllFreqs=AllFreqs
    #     self.RefFreq=RefFreq

    def GiveBeamFactorsFacet(self,iFacet):
        if iFacet in self.DicoBeamFactors:
            return self.DicoBeamFactors[iFacet]

        SumJonesChan=self.DicoMappingDesc["SumJonesChan"]
        ChanMappingGrid=self.DicoMappingDesc["ChanMappingGrid"]
        
        # ListBeamFactor=[]
        # ListBeamFactorWeightSq=[]

        # for iChannel in range(self.NFreqBand):
        #     ThisSumJonesChan=[]
        #     ThisSumJonesChanWeightSq=[]
        #     for iMS in range(len(SumJonesChan)):
        #         ind=np.where(ChanMappingGrid[iMS]==iChannel)[0]
        #         ThisSumJonesChan+=SumJonesChan[iMS][ind].tolist()
        #         ThisSumJonesChanWeightSq+=SumJonesChanWeightSq[iMS][ind].tolist()
        #     ListBeamFactor.append(np.array(ThisSumJonesChan))
        #     ListBeamFactorWeightSq.append(np.array(ThisSumJonesChanWeightSq))

        
        ChanMappingGridChan=self.DicoMappingDesc["ChanMappingGridChan"]
        ListBeamFactor=[]
        ListBeamFactorWeightSq=[]
        for iChannel in range(self.NFreqBand):
            nfreq = len(self.DicoMappingDesc["freqs"][iChannel])
            ThisSumJonesChan = np.zeros(nfreq,np.float64)
            ThisSumJonesChanWeightSq = np.zeros(nfreq,np.float64)
            for iMS in SumJonesChan.keys():
                ind = np.where(ChanMappingGrid[iMS]==iChannel)[0]
                channels = ChanMappingGridChan[iMS][ind]
                ThisSumJonesChan[channels] += SumJonesChan[iMS][iFacet,0,ind]
                ThisSumJonesChanWeightSq[channels] += SumJonesChan[iMS][iFacet,1,ind]
            ListBeamFactor.append(ThisSumJonesChan)
            ListBeamFactorWeightSq.append(ThisSumJonesChanWeightSq)

        self.DicoBeamFactors[iFacet] = ListBeamFactor, ListBeamFactorWeightSq

        return ListBeamFactor, ListBeamFactorWeightSq


    def GiveFreqBandsFluxRatio(self,iFacet,Alpha):

        NFreqBand=self.NFreqBand
        NAlpha=Alpha.size
        FreqBandsFluxRatio=np.zeros((NAlpha,NFreqBand),np.float32)

        for iChannel in range(NFreqBand):
            for iAlpha in range(NAlpha):
                ThisAlpha=Alpha[iAlpha]
                
                FreqBandsFluxRatio[iAlpha,iChannel]=self.IntExpFunc(Alpha=ThisAlpha,iChannel=iChannel,iFacet=iFacet)


        return FreqBandsFluxRatio

    def CalcFluxBands(self,NAlpha=21):
        Alphas=np.linspace(-1,1,NAlpha)
        NFacets=len(self.DicoMappingDesc["MeanJonesBand"])
        self.FluxBands=np.zeros((NFacets,NAlpha,self.NFreqBand),np.float32)
        for iFacet in range(NFacets):
            self.FluxBands[iFacet]=self.GiveFreqBandsFluxRatio(iFacet,Alphas)
        
        
    def IntExpFunc(self,S0=1.,Alpha=0.,iChannel=0,iFacet=0):
        
        RefFreq=self.RefFreq

        ThisAlpha=Alpha
        ThisFreqs=np.array(self.DicoMappingDesc["freqs"][iChannel])
        
        S0=np.array(S0)
        Npix=S0.size



        if self.BeamEnable:
            ListBeamFactor,ListBeamFactorWeightSq = self.GiveBeamFactorsFacet(iFacet)
            BeamFactor=ListBeamFactor[iChannel].reshape((1,ThisFreqs.size))
            BeamFactorWeightSq=ListBeamFactorWeightSq[iChannel].reshape((1,ThisFreqs.size))
            MeanJonesBand=self.DicoMappingDesc["MeanJonesBand"][iFacet][iChannel]
        else:
            BeamFactor=1.
            BeamFactorWeightSq=1.
            MeanJonesBand=1.


        ThisFreqs=ThisFreqs.reshape((1,ThisFreqs.size))
        ThisAlpha=ThisAlpha.reshape((Npix,1))
        FreqBandsFlux=np.sqrt(np.sum(BeamFactor*((ThisFreqs/RefFreq)**ThisAlpha)**2,axis=1))/np.sqrt(np.sum(BeamFactorWeightSq))
        FreqBandsFlux/=np.sqrt(MeanJonesBand)

        S0=S0.reshape((Npix,))
        FreqBandsFlux*=S0



        return FreqBandsFlux.ravel()

    def IntExpFuncPoly(self,PolyArray,iChannel=0,iFacet=0,FluxScale="Exp"):#, S0=1.,Alpha=0.):
        
        RefFreq=self.RefFreq

        ThisFreqs=np.array(self.DicoMappingDesc["freqs"][iChannel])
        
        S0=np.array(PolyArray[:,0])
        Npix=S0.size
        
            

        if self.BeamEnable:
            ListBeamFactor,ListBeamFactorWeightSq = self.GiveBeamFactorsFacet(iFacet)
            BeamFactor=ListBeamFactor[iChannel].reshape((1,ThisFreqs.size))
            BeamFactorWeightSq=ListBeamFactorWeightSq[iChannel].reshape((1,ThisFreqs.size))
            MeanJonesBand=self.DicoMappingDesc["MeanJonesBand"][iFacet][iChannel]
        else:
            BeamFactor=1.
            BeamFactorWeightSq=1.
            MeanJonesBand=1.


        
        ThisFreqs=ThisFreqs.reshape((1,ThisFreqs.size))
        # ThisAlpha=ThisAlpha.reshape((Npix,1))
        # SUnityFreq=(ThisFreqs/RefFreq)**ThisAlpha

        # SUnityFreq=np.zeros((Npix,ThisFreqs.size),np.float32)
        # for iPix in range(Npix):
        #     p=PolyArray[iPix,:].copy()
        #     p[0]=0.
        #     logS=np.poly1d(p[::-1])(np.log(ThisFreqs.ravel()/RefFreq))
        #     SUnityFreq[iPix,:]=np.exp(logS)
        
        Npix,NOrder=PolyArray.shape
        n=np.arange(NOrder)
        n=n.reshape((1,1,NOrder))
        f=ThisFreqs.reshape((1,-1,1))
        a=(PolyArray.copy()).reshape((Npix,1,NOrder))
        
        if FluxScale=="Exp":
            a[:,:,0]=0.
            SUnityFreq0=a*(np.log(f/RefFreq))**n
            SUnityFreq0=np.exp(np.sum(SUnityFreq0,axis=-1))
        elif FluxScale=="Linear":
            SUnityFreq0=a*((f-RefFreq)/RefFreq)**n
            SUnityFreq0=np.sum(SUnityFreq0,axis=-1)
        SUnityFreq=SUnityFreq0
        
        FreqBandsFlux=np.sqrt(np.sum(BeamFactor*( SUnityFreq )**2,axis=1))/np.sqrt(np.sum(BeamFactorWeightSq))
        FreqBandsFlux/=np.sqrt(MeanJonesBand)
        
        S0=S0.reshape((Npix,))
        if FluxScale=="Exp":
            FreqBandsFlux*=S0
            
        return FreqBandsFlux.ravel()
    
