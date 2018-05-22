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

import numpy as np
from DDFacet.Other import MyLogger
from DDFacet.ToolsDir import ClassSpectralFunctions

log= MyLogger.getLogger("ClassPSFServer")

class ClassPSFServer():
    def __init__(self,GD=None):
        self.GD=GD
        self.blc=None

    def setDicoVariablePSF(self,DicoVariablePSF,RefFreq=None,NormalisePSF=False,quiet=False):
        # NFacets=len(DicoVariablePSF.keys())
        # NPixMin=1e6
        # for iFacet in sorted(DicoVariablePSF.keys()):
        #     _,npol,n,n=DicoVariablePSF[iFacet]["PSF"][0].shape
        #     if n<NPixMin: NPixMin=n

        # nch=self.GD["Freq"]["NBand"]
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

        if NormalisePSF:
            if not quiet:
                print>>log,"Using peak-normalised PSFs"
            # for iFacet in range(self.NFacets):
            #     self.CubeMeanVariablePSF[iFacet]/=np.max(self.CubeMeanVariablePSF[iFacet])
            #     for iChan in range(nch):
            #         self.CubeVariablePSF[iFacet,iChan]/=np.max(self.CubeVariablePSF[iFacet,iChan])
            self.CubeMeanVariablePSF=DicoVariablePSF["PeakNormed_CubeMeanVariablePSF"]
            self.CubeVariablePSF=DicoVariablePSF["PeakNormed_CubeVariablePSF"]

        DicoMappingDesc={"freqs":DicoVariablePSF["freqs"],
                         "WeightChansImages":DicoVariablePSF["WeightChansImages"],
                         "SumJonesChan":DicoVariablePSF["SumJonesChan"],
                         "ChanMappingGrid":DicoVariablePSF["ChanMappingGrid"],
                         "ChanMappingGridChan":DicoVariablePSF["ChanMappingGridChan"],
                         "MeanJonesBand":DicoVariablePSF["MeanJonesBand"]}

        self.DicoMappingDesc=DicoMappingDesc
        
        self.SpectralFunctionsMachine=ClassSpectralFunctions.ClassSpectralFunctions(DicoMappingDesc)
        #self.RefFreq=self.SpectralFunctionsMachine.RefFreq=RefFreq
        #self.AllFreqs=self.SpectralFunctionsMachine.AllFreqs
        #print "PSFServer:",self.RefFreq, self.AllFreqs
        #self.CalcJacobian()

    def setRefFreq(self,RefFreq):
        if RefFreq is None:
            raise ValueError("Reference frequency should be set")
        self.RefFreq=RefFreq
        self.SpectralFunctionsMachine.RefFreq=RefFreq
        self.DicoMappingDesc["RefFreq"]=RefFreq

    def setBLC(self,blc):
        self.blc=blc


    def setLocation(self,xp,yp):
        self.iFacet=self.giveFacetID2(xp,yp)
        #print "ifacet:",self.iFacet

    def giveFacetID(self,xp,yp):
        dmin=1e6
        for iFacet in range(self.NFacets):
            d=np.sqrt((xp-self.DicoVariablePSF["Facets"][iFacet]["pixCentral"][0])**2+(yp-self.DicoVariablePSF["Facets"][iFacet]["pixCentral"][1])**2)
            if d<dmin:
                dmin=d
                ClosestFacet=iFacet
        return ClosestFacet
                
    def giveFacetID2(self,xp,yp):
        if self.blc is not None:
            x0,y0=self.blc
            xp,yp=x0+xp,y0+yp

        dmin=1e6
        CellSizeRad=self.DicoVariablePSF["CellSizeRad"]
        _,_,nx,_=self.DicoVariablePSF["OutImShape"]
        for iFacet in range(self.NFacets):
            l=CellSizeRad*(xp-nx/2)
            m=CellSizeRad*(yp-nx/2)
            lSol,mSol=self.DicoVariablePSF["Facets"][iFacet]["lmSol"]
            #print "lsol, msol = ",lSol,mSol #,self.DicoVariablePSF[iFacet]["pixCentral"][0],self.DicoVariablePSF[iFacet]["pixCentral"][1]

            d=np.sqrt((l-lSol)**2+(m-mSol)**2)

            if d<dmin:
                dmin=d
                ClosestFacet=iFacet
        # print "[%i, %i] ->  %i"%(xp,yp,ClosestFacet)
        return ClosestFacet

    def setFacet(self,iFacet):
        #print "set facetloc"
        self.iFacet=iFacet

    def CalcJacobian(self):
        self.CubeJacobianMeanVariablePSF={}
        dx=31
        # CubeMeanVariablePSF shape = NFacets,1,npol,NPixMin,NPixMin
        _,_,_,nx,_=self.CubeMeanVariablePSF.shape

        # import pylab
        for iFacet in range(self.NFacets):
            ThisCubePSF=self.CubeMeanVariablePSF[iFacet,0,0][nx/2-dx-1:nx/2+dx+1+1,nx/2-dx-1:nx/2+dx+1+1]
            Jx=(ThisCubePSF[:-2,:]-ThisCubePSF[2::,:])/2
            Jy=(ThisCubePSF[:,:-2]-ThisCubePSF[:,2::])/2
            Jx=Jx[:,1:-1]
            Jy=Jy[1:-1,:]
            J=np.zeros((Jx.size,2),np.float32)
            J[:,0]=Jx.ravel()
            J[:,1]=Jy.ravel()
            self.CubeJacobianMeanVariablePSF[iFacet]={}
            self.CubeJacobianMeanVariablePSF[iFacet]["J"]=J
            self.CubeJacobianMeanVariablePSF[iFacet]["JHJ"]=np.dot(J.T,J)
            
            

            # pylab.clf()
            # pylab.subplot(1,3,1)
            # pylab.imshow(ThisCubePSF,interpolation="nearest")
            # pylab.subplot(1,3,2)
            # pylab.imshow(Jx,interpolation="nearest")
            # pylab.subplot(1,3,3)
            # pylab.imshow(Jy,interpolation="nearest")
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            # stop

    def SolveOffsetLM(self,Dirty,xc0,yc0):
        iFacet=self.iFacet
        Lambda=1.
        nIter=5
        beta=np.zeros((2,1),np.float32)
        J=self.CubeJacobianMeanVariablePSF[iFacet]["J"]
        s,_=J.shape
        nx=int(np.sqrt(s))
        JHJ=self.CubeJacobianMeanVariablePSF[iFacet]["JHJ"]
        JHJ1inv=np.linalg.inv(JHJ+Lambda*np.diag(np.diag(JHJ)))

        dx=nx/2
        
        _,_,_,nx_psf,_=self.CubeMeanVariablePSF.shape
        xc_psf=nx_psf/2
        ThisCubePSF=self.CubeMeanVariablePSF[iFacet,0,0][xc_psf-dx:xc_psf+dx+1,xc_psf-dx:xc_psf+dx+1]
        xc,yc=xc0,yc0
        Val=Dirty[xc,yc]
        for Iter in range(nIter):
            D=Dirty[xc-dx:xc+dx+1,yc-dx:yc+dx+1]
            Val=Dirty[xc,yc]
            Model=(ThisCubePSF/np.max(ThisCubePSF))*Val
            R=(D-Val*Model).reshape((D.size,1))
            delta=np.dot(JHJ1inv,np.dot(J.T,R))
            delta_x=int(round(delta[0,0]))
            delta_y=int(round(delta[1,0]))
            if (delta_x==0)&(delta_y==0):
                break
            delta_x=np.min([1,np.abs(delta_x)])*np.sign(delta_x)
            delta_y=np.min([1,np.abs(delta_y)])*np.sign(delta_y)

            xc+=delta_x
            yc+=delta_y
            #print delta_x,delta_y
        return xc,yc

    def GivePSF(self):
        return self.CubeVariablePSF[self.iFacet],self.CubeMeanVariablePSF[self.iFacet]

    def CropPSF(self, PSF, npix):
        PSFshape = PSF.shape
        if len(PSFshape) == 4:
            nfreq, npol, nx_psf, ny_psf = PSF.shape
        else:
            nx_psf, ny_psf = PSF.shape

        xc_psf = nx_psf / 2

        if npix % 2 == 0:
            print >> log, "Cropping size should be odd (npix=%d) !!! Adding 1 pixel" % npix
            npix = npix + 1

        if npix > nx_psf or npix > ny_psf:
            print >> log, "Cropping size larger than PSF size !!!"
            raise NameError("Cropping size larger than PSF size !!!")

        npixside = (npix - 1) / 2  # pixel to include from PSF center.
        if len(PSFshape) == 4:
            PSFCrop = PSF[nfreq, npol, xc_psf - npixside:xc_psf + npixside + 1,
                      xc_psf - npixside:xc_psf + npixside + 1]  # making it square psf
        else:
            PSFCrop = PSF[xc_psf - npixside:xc_psf + npixside + 1,
                      xc_psf - npixside:xc_psf + npixside + 1]  # making it square psf

        return PSFCrop

    def GiveFreqBandsFluxRatio(self,iFacet,Alpha):
        NAlpha=Alpha.size
        NFreqBand=self.DicoVariablePSF["CubeVariablePSF"].shape[1]
        SumJonesChan=self.DicoVariablePSF["SumJonesChan"]
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
                ThisSumJonesChan[channels] += SumJonesChan[iMS][iFacet,0,ind]
                ThisSumJonesChanWeightSq[channels] += SumJonesChan[iMS][iFacet,1,ind]
            
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
#            print "FreqArrays",len(ThisFreqs),len(BeamFactor)
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
