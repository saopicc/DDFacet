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
log= MyLogger.getLogger("ClassBeamMean")
from DDFacet.ToolsDir import ModCoord
import ClassJones
from DDFacet.Imager import ModCF
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Other import ClassTimeIt
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Other import ModColor
from DDFacet.Array import NpShared


class ClassBeamMean():
    def __init__(self,VS):
        self.VS=VS
        self.GD=self.VS.GD
        self.CheckCache()
        if self.CacheValid: return


        self.ListMS=self.VS.ListMS
        self.MS=self.ListMS[0]
        rac,decc=self.MS.radec
        self.CoordMachine=ModCoord.ClassCoordConv(rac,decc)
        self.CalcGrid()
        #self.Padding=Padding

    def CalcGrid(self):
        _,_,nx,_=self.VS.FullImShape
        CellSizeRad=self.VS.CellSizeRad
        FOV=nx*CellSizeRad
        npix=11
        lm=np.linspace(-FOV/2.,FOV/2.,npix+1)
        ll=(lm[0:-1]+lm[1::])/2.
        lmin=ll.min()
        lmax=ll.max()
        lc,mc=np.mgrid[lmin:lmax:1j*npix,lmin:lmax:1j*npix]
        lc=lc.flatten()
        mc=mc.flatten()
        ra=np.zeros_like(lc)
        dec=np.zeros_like(lc)
        for i in range(ra.size):
            ra[i],dec[i]=self.CoordMachine.lm2radec(np.array([lc[i]]),
                                                    np.array([mc[i]]))
        self.radec=ra,dec
        self.npix=npix

    def LoadData(self):
        print>>log, "Loading some data for all MS..."
        # make lists of tables and row counts (one per MS)
        tabs = [ ms.GiveMainTable() for ms in self.ListMS ]
        nrows = [ tab.nrows() for tab in tabs ]
        nr = sum(nrows)
        # preallocate arrays
        # NB: this assumes nchan and ncorr is the same across all MSs in self.ListMS. Tough luck if it isn't!


        self.Data={}
        
        for iMS,MS in zip(range(self.VS.nMS),self.ListMS):

            tab = MS.GiveMainTable()
            times = tab.getcol("TIME")
            flags = tab.getcol("FLAG")
            A0 = tab.getcol("ANTENNA1")
            A1 = tab.getcol("ANTENNA2")
            tab.close()



            for iChunk in range(ms.numChunks()):
                weights = self.VS.GetVisWeights(iMS, iChunk)


                ThisMSData={}
                ThisMSData["A0"]=A0
                ThisMSData["A1"]=A1
                ThisMSData["times"]=times
                ThisMSData["flags"]=flags
                ThisMSData["W"]=weights
            
    
            self.Data[iMS]=ThisMSData

    def CalcMeanBeam(self):
        if self.CacheValid:
            return 

        print>>log, ModColor.Str("========================= Calculating smooth beams =======================")
        self.LoadData()
        Dt=self.GD["Beam"]["DtBeamMin"]*60.

        RAs,DECs = self.radec

        SumJJsq=np.zeros((self.npix,self.npix,self.MS.Nchan),np.float64)
        SumWsq=0.


        for iMS,MS in zip(range(self.VS.nMS),self.ListMS):
            print>>log,"Compute beam for %s"%MS.MSName
            print>>log,"  in %i directions"%RAs.size
            JonesMachine=ClassJones.ClassJones(self.GD,MS,self.VS.FacetMachine,IdSharedMem=self.VS.IdSharedMem)
            JonesMachine.InitBeamMachine()

            ThisMSData=self.Data[iMS]
            times=ThisMSData["times"]
            A0=ThisMSData["A0"]
            A1=ThisMSData["A1"]
            flags=ThisMSData["flags"]
            W=ThisMSData["W"]
            
            beam_times = np.array(JonesMachine.BeamMachine.getBeamSampleTimes(times))
            
            CurrentBeamITime=-1
            #print "  Estimate beam in %i directions"%(RAs.size)
            DicoBeam=JonesMachine.EstimateBeam(beam_times, RAs, DECs)
            T=ClassTimeIt.ClassTimeIt()
            T.disable()
            NTRange=DicoBeam["t0"].size
            pBAR= ProgressBar(Title="      Mean Beam")
            pBAR.render(0, '%4i/%i' % (0,NTRange))
            for iTRange in range(DicoBeam["t0"].size):

                t0=DicoBeam["t0"][iTRange]
                t1=DicoBeam["t1"][iTRange]
                J=np.abs(DicoBeam["Jones"][iTRange])
                ind=np.where((times>=t0)&(times<t1))[0]
                T.timeit("0")
                A0s=A0[ind]
                A1s=A1[ind]
                fs=flags[ind]
                Ws=W[ind]
                T.timeit("1")
                
                nt,na,nch,_,_=J.shape

                # # ######################
                # # This call is slow
                # J0=J[:,A0s,:,:,:]
                # J1=J[:,A1s,:,:,:]
                # T.timeit("2")
                # # ######################
                J0=np.zeros((nt,A0s.size,nch,2,2),dtype=J.dtype)
                J0List=[J[:,A0s[i],:,:,:] for i in range(A0s.size)]
                J1=np.zeros((nt,A0s.size,nch,2,2),dtype=J.dtype)
                J1List=[J[:,A1s[i],:,:,:] for i in range(A0s.size)]
                for i in range(A0s.size):
                    J0[:,i,:,:,:]=J0List[i]
                    J1[:,i,:,:,:]=J1List[i]
                T.timeit("2b")



                JJ=(J0[:,:,:,0,0]*J1[:,:,:,0,0]+J0[:,:,:,1,1]*J1[:,:,:,1,1])/2.
                T.timeit("3")

                WW=Ws**2
                T.timeit("4")
                WW=WW.reshape((1,ind.size,self.MS.Nchan))
                T.timeit("5")
                JJsq=WW*JJ**2
                T.timeit("6")

                SumWsqThisRange=np.sum(JJsq,axis=1)
                T.timeit("7")
                SumJJsq+=SumWsqThisRange.reshape((self.npix,self.npix,self.MS.Nchan))
                T.timeit("8")
                SumWsq+=np.sum(WW,axis=1)
                T.timeit("9")

                NDone = iTRange+1
                intPercent = int(100 * NDone / float(NTRange))
                pBAR.render(intPercent, '%4i/%i' % (NDone, NTRange))

        SumJJsq/=SumWsq.reshape(1,1,self.MS.Nchan)
        #self.SumJJsq=np.rollaxis(SumJJsq,2)#np.mean(SumJJsq,axis=2)
        self.SumJJsq=np.mean(SumJJsq,axis=2)

        self.Smooth()
        
    def Smooth(self):
        _,_,nx,_=self.VS.FullImShape
        
        SpheM=ModCF.SpheMachine(Support=self.npix,SupportSpheCalc=111)
        CF, fCF, ifzfCF=SpheM.MakeSphe(nx)
        
        
        ifzfCF.fill(1)
        SpheM.if_cut_fCF.fill(1)
        
        FT=ModFFTW.FFTW_2Donly_np()
        
        SumJJsq_Sphe=self.SumJJsq.copy()*SpheM.if_cut_fCF

        A=np.complex64(SumJJsq_Sphe.reshape((1,1,self.npix,self.npix)))
        f_SumJJsq=FT.fft(A).reshape((self.npix,self.npix))
        z_f_SumJJsq=np.complex64(ModCF.ZeroPad(f_SumJJsq,outshape=nx))
        
        if_z_f_SumJJsq=FT.ifft(z_f_SumJJsq.reshape((1,1,nx,nx))).real.reshape((nx,nx))
        if_z_f_SumJJsq/=np.real(ifzfCF)#.reshape((1,1,nx,nx)))
        #if_z_f_SumJJsq[ifzfCF.real<1e-2]=-1.

        # vmin=0#self.SumJJsq.min()
        # vmax=self.SumJJsq.max()
        # import pylab
        # pylab.clf()
        # ax=pylab.subplot(1,3,1)
        # pylab.imshow(self.SumJJsq,interpolation="nearest",vmin=vmin,vmax=vmax,extent=(0,1,0,1))
        # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
        # pylab.imshow(ifzfCF.real,interpolation="nearest",vmin=vmin,vmax=vmax,extent=(0,1,0,1))
        # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
        # pylab.imshow(if_z_f_SumJJsq,interpolation="nearest",vmin=vmin,vmax=vmax,extent=(0,1,0,1))
        # pylab.show(False)
        # stop

        self.ifzfCF=np.real(ifzfCF)
        self.SmoothBeam=np.real(if_z_f_SumJJsq)
        np.save(self.CachePath,self.SmoothBeam)
        self.VS.maincache.saveCache("SmoothBeam.npy")
        print>>log, ModColor.Str("======================= Done calculating smooth beams ====================")

       
    def CheckCache(self):
        self.CachePath, self.CacheValid = self.VS.maincache.checkCache("SmoothBeam.npy", 
                                                                  dict([("MSNames", [ms.MSName for ms in self.VS.ListMS])] +
                                                                       [(section, self.GD[section]) 
                                                                        for section in "Data", 
                                                                        "Beam", "Selection",
                                                                        "Freq", "Image", 
                                                                        "Comp", "Facets", 
                                                                        "Weight", "RIME"]), 
                                                                  reset=self.GD["Cache"]["ResetSmoothBeam"])


        if self.CacheValid:
            print>>log,"found valid cached dirty image in %s"%self.CachePath
            self.SmoothBeam=np.load(self.CachePath)

    def GiveMergedWithDiscrete(self,DiscreteMeanBeam):
        Mask=(self.ifzfCF<1e-2)
        self.SmoothBeam[Mask]=DiscreteMeanBeam[Mask]
        return self.SmoothBeam
