import numpy as np

class ClassPSFServer():
    def __init__(self,GD):
        self.GD=GD

    def setDicoVariablePSF(self,DicoVariablePSF):
        # NFacets=len(DicoVariablePSF.keys())
        # NPixMin=1e6
        # for iFacet in sorted(DicoVariablePSF.keys()):
        #     _,npol,n,n=DicoVariablePSF[iFacet]["PSF"][0].shape
        #     if n<NPixMin: NPixMin=n

        # nch=self.GD["MultiFreqs"]["NFreqBands"]
        # CubeVariablePSF=np.zeros((NFacets,nch,npol,NPixMin,NPixMin),np.float32)
        # CubeMeanVariablePSF=np.zeros((NFacets,1,npol,NPixMin,NPixMin),np.float32)
        # for iFacet in sorted(DicoVariablePSF.keys()):
        #     _,npol,n,n=DicoVariablePSF[iFacet]["PSF"][0].shape
        #     for ch in range(nch):
        #         i=n/2-NPixMin/2
        #         j=n/2+NPixMin/2+1
        #         CubeVariablePSF[iFacet,ch,:,:,:]=DicoVariablePSF[iFacet]["PSF"][ch][0,:,i:j,i:j]
        #     CubeMeanVariablePSF[iFacet,0,:,:,:]=DicoVariablePSF[iFacet]["MeanPSF"][0,:,i:j,i:j]

        self.DicoVariablePSF=DicoVariablePSF
        self.CubeVariablePSF=DicoVariablePSF["CubeVariablePSF"]
        self.CubeMeanVariablePSF=DicoVariablePSF["CubeMeanVariablePSF"]
        self.NFacets,nch,npol,NPixMin,_=self.CubeVariablePSF.shape
        self.ShapePSF=nch,npol,NPixMin,NPixMin
        self.NPSF=NPixMin

    def setLocation(self,xp,yp):
        #print "set loc"
        dmin=1e6
        for iFacet in range(self.NFacets):
            d=np.sqrt((xp-self.DicoVariablePSF[iFacet]["pixCentral"][0])**2+(yp-self.DicoVariablePSF[iFacet]["pixCentral"][1])**2)
            if d<dmin:
                dmin=d
                self.iFacet=iFacet

    def setFacet(self,iFacet):
        #print "set facetloc"
        self.iFacet=iFacet


    def GivePSF(self):
        return self.CubeVariablePSF[self.iFacet],self.CubeMeanVariablePSF[self.iFacet]



    def GiveFreqBandsFluxRatio(self,iFacet,Alpha):
        NAlpha=Alpha.size
        NFreqBand=self.DicoVariablePSF["CubeVariablePSF"].shape[1]
        SumJonesChan=self.DicoVariablePSF["SumJonesChan"][iFacet]
        SumJonesChanWeightSq=self.DicoVariablePSF["SumJonesChanWeightSq"][iFacet]
        ChanMappingGrid=self.DicoVariablePSF["ChanMappingGrid"]
        ChanMappingGridChan=self.DicoVariablePSF["ChanMappingGridChan"]
        RefFreq=self.RefFreq
        FreqBandsFluxRatio=np.zeros((NAlpha,NFreqBand),np.float32)

        
        ListBeamFactor=[]
        ListBeamFactorWeightSq=[]
        #print "============"
        for iChannel in range(NFreqBand):
            nfreq = len(self.DicoVariablePSF["freqs"][iChannel])
            ThisSumJonesChan = np.zeros(nfreq,np.float64)
            ThisSumJonesChanWeightSq = np.zeros(nfreq,np.float64)
            for iMS in range(len(SumJonesChan)):
                ind = np.where(ChanMappingGrid[iMS]==iChannel)[0]
                channels = ChanMappingGridChan[iMS][ind]
                ThisSumJonesChan[channels] += SumJonesChan[iMS][ind]
                ThisSumJonesChanWeightSq[channels] += SumJonesChanWeightSq[iMS][ind]
            
            #print "== ",iFacet,iChannel,np.sqrt(np.sum(np.array(ThisSumJonesChan))/np.sum(np.array(ThisSumJonesChanWeightSq)))

            ListBeamFactor.append(ThisSumJonesChan)
            ListBeamFactorWeightSq.append(ThisSumJonesChanWeightSq)

        # SumListBeamFactor=0
        # NChan=0
        # for iChannel in range(NFreqBand):
        #     SumListBeamFactor+=np.sum(ListBeamFactor[iChannel])
        #     NChan+=ListBeamFactor[iChannel].size
        # SumListBeamFactor/=NChan
        # for iChannel in range(NFreqBand):
        #     ListBeamFactor[iChannel]/=SumListBeamFactor

        # for iChannel in range(NFreqBand):
        # #     ListBeamFactor[iChannel]/=np.mean(ListBeamFactor[iChannel])
        # #     # ListBeamFactor[iChannel]=np.sqrt(ListBeamFactor[iChannel])
        # #     # ListBeamFactor[iChannel]/=np.mean(ListBeamFactor[iChannel])
        #     ListBeamFactor[iChannel]/=(self.DicoVariablePSF["SumJonesBand"][iFacet][iChannel])
        # #     # print self.DicoVariablePSF["MeanJonesBand"][iFacet]


        for iChannel in range(NFreqBand):
            BeamFactor=ListBeamFactor[iChannel]
            BeamFactorWeightSq=ListBeamFactorWeightSq[iChannel]
            
            ThisFreqs=self.DicoVariablePSF["freqs"][iChannel]
            print "FreqArrays",len(ThisFreqs),len(BeamFactor)
            #if iFacet==60:
            #    print iChannel,iMS,BeamFactor
            #BeamFactor.fill(1.)
            for iAlpha in range(NAlpha):
                ThisAlpha=Alpha[iAlpha]
                
                FreqBandsFluxRatio[iAlpha,iChannel]=np.sqrt(np.sum(BeamFactor*((ThisFreqs/RefFreq)**ThisAlpha)**2))/np.sqrt(np.sum(BeamFactorWeightSq))
                FreqBandsFluxRatio[iAlpha,iChannel]/=np.sqrt(self.DicoVariablePSF["MeanJonesBand"][iFacet][iChannel])
        #MeanFreqBandsFluxRatio=np.mean(FreqBandsFluxRatio,axis=1)
        #FreqBandsFluxRatio=FreqBandsFluxRatio/MeanFreqBandsFluxRatio.reshape((NAlpha,1))

        # print "=============="
        # print iFacet
        # print FreqBandsFluxRatio

        return FreqBandsFluxRatio
