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
import numexpr
from DDFacet.Other import MyLogger
log= MyLogger.getLogger("ClassBeamMean")
from DDFacet.ToolsDir import ModCoord
import ClassJones
from DDFacet.Imager import ModCF
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Other import ClassTimeIt
from DDFacet.Array import shared_dict

from DDFacet.Other.AsyncProcessPool import APP
import copy


class ClassBeamMean():
    def __init__(self,VS):
        MyLogger.setSilent(["ClassJones","ClassLOFARBeam"])
        self.VS=VS
        
        self.GD=copy.deepcopy(self.VS.GD)
        self.DoCentralNorm=self.GD["Beam"]["CenterNorm"]
        self.SmoothBeam=None
        self.CheckCache()
        if self.CacheValid: return

        #self.GD["Beam"]["CenterNorm"]=0

        self.ListMS=self.VS.ListMS
        self.MS=self.ListMS[0]
        rac,decc=self.MS.radec
        self.CoordMachine=ModCoord.ClassCoordConv(rac,decc)
        self.CalcGrid()
        #self.Padding=Padding

        #self.SumJJsq=np.zeros((self.npix,self.npix,self.MS.Nchan),np.float64)
        #self.SumWsq=np.zeros((1,self.MS.Nchan),np.float64)

        self.StackedBeamDict = shared_dict.create("StackedBeamDict")
        for iDir in range(self.NDir):
            sd = self.StackedBeamDict.addSubdict(iDir)
            sd.addSharedArray("SumJJsq", (self.MS.Nchan,), np.float64)
            sd.addSharedArray("SumWsq", (self.MS.Nchan,), np.float64)

        self.DicoJonesMachine={}
        for iMS,MS in enumerate(self.ListMS):
            JonesMachine=ClassJones.ClassJones(self.GD,MS,self.VS.FacetMachine)
            JonesMachine.InitBeamMachine()
            self.DicoJonesMachine[iMS]=JonesMachine
        MyLogger.setLoud(["ClassJones","ClassLOFARBeam"])

    def CalcGrid(self):
        _,_,nx,_=self.VS.FullImShape
        CellSizeRad=self.VS.CellSizeRad
        FOV=nx*CellSizeRad
        npix=self.GD["Beam"]["SmoothNPix"]
        lm=np.linspace(-FOV/2.,FOV/2.,npix+1)
        ll=(lm[0:-1]+lm[1::])/2.
        lmin=ll.min()
        lmax=ll.max()
        lc,mc=np.mgrid[lmin:lmax:1j*npix,lmin:lmax:1j*npix]
        iPix,jPix=np.mgrid[0:npix,0:npix]
        self.iPix=iPix.ravel()
        self.jPix=jPix.ravel()
        lc=lc.flatten()
        mc=mc.flatten()
        ra=np.zeros_like(lc)
        dec=np.zeros_like(lc)
        for i in range(ra.size):
            ra[i],dec[i]=self.CoordMachine.lm2radec(np.array([lc[i]]),
                                                    np.array([mc[i]]))
        self.radec=ra,dec
        self.npix=npix
        self.NDir=ra.size
        


    def StackBeam(self,ThisMSData,iDir):
        self.StackedBeamDict.reload()
        MyLogger.setSilent("ClassJones")
        Dt=self.GD["Beam"]["DtBeamMin"]*60.
        JonesMachine=self.DicoJonesMachine[ThisMSData["iMS"]]
        RAs,DECs = self.radec

        times=ThisMSData["times"]
        A0=ThisMSData["A0"]
        A1=ThisMSData["A1"]
        flags=ThisMSData["flags"]
        W=ThisMSData["Weights"]
        
        beam_times = np.array(JonesMachine.BeamMachine.getBeamSampleTimes(times, quiet=True))

        T2=ClassTimeIt.ClassTimeIt()
        T2.disable()
        CurrentBeamITime=-1
        # #print "  Estimate beam in %i directions"%(RAs.size)
        # MS=self.ListMS[ThisMSData["iMS"]]
        # JonesMachine=ClassJones.ClassJones(self.GD,MS,self.VS.FacetMachine)
        # JonesMachine.InitBeamMachine()
        DicoBeam=JonesMachine.EstimateBeam(beam_times, RAs[iDir:iDir+1], DECs[iDir:iDir+1], progressBar=False, quiet=True)
        T2.timeit("GetBeam 1")
        #DicoBeam=JonesMachine.EstimateBeam(beam_times, RAs[0:10], DECs[0:10],progressBar=False)
        #T2.timeit("GetBeam 10")
        #print DicoBeam["Jones"].shape
        NTRange=DicoBeam["t0"].size
        #pBAR= ProgressBar(Title="      Mean Beam")
        #pBAR.render(0, '%4i/%i' % (0,NTRange))
        T=ClassTimeIt.ClassTimeIt("Stacking")
        T.disable()

        # DicoBeam["Jones"].shape = nt, nd, na, nch, _, _

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
            
            nd,na,nch,_,_=J.shape
            
            # ######################
            # This call is slow
            J0=J[:,A0s,:,:,:]
            J1=J[:,A1s,:,:,:]
            T.timeit("2")
            # ######################

            # J0=np.zeros((nd,A0s.size,nch,2,2),dtype=J.dtype)
            # #T.timeit("1a")
            # J0List=[J[:,A0s[i],:,:,:] for i in range(A0s.size)]
            # #T.timeit("1b")
            # J1=np.zeros((nd,A0s.size,nch,2,2),dtype=J.dtype)
            # #T.timeit("1c")
            # J1List=[J[:,A1s[i],:,:,:] for i in range(A0s.size)]
            # #T.timeit("1d")
            # for i in range(A0s.size):
            #     J0[:,i,:,:,:]=J0List[i]
            #     J1[:,i,:,:,:]=J1List[i]
            # T.timeit("2b")

        
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

            #self.SumJJsq+=SumWsqThisRange.reshape((self.npix,self.npix,self.MS.Nchan))
            #T.timeit("8")
            SumWsq=np.sum(WW,axis=1)
            #self.SumWsq+=SumWsq

            self.StackedBeamDict[iDir]["SumJJsq"]+=SumWsqThisRange.reshape((self.MS.Nchan,))
            self.StackedBeamDict[iDir]["SumWsq"]+=SumWsq.reshape((self.MS.Nchan,))

            #print SumWsq,self.SumWsq,self.SumJJsq.shape,J0.shape
            T.timeit("9")
            
            #NDone = iTRange+1
            #intPercent = int(100 * NDone / float(NTRange))
            #pBAR.render(intPercent, '%4i/%i' % (NDone, NTRange))

        T2.timeit("Stack")
        MyLogger.setLoud("ClassJones")
  
        
    def Smooth(self):
        #print self.SumWsq
        self.StackedBeamDict.reload()
        self.SumJJsq=np.zeros((self.npix,self.npix,self.MS.Nchan),np.float64)
        self.SumWsq=np.zeros((1,self.MS.Nchan),np.float64)
        self.SumWsq[0,:]=self.StackedBeamDict[0]["SumWsq"]
        if np.max(self.StackedBeamDict[0]["SumWsq"])==0:
            return "NoStackedData"
        for iDir in range(self.NDir):
            i,j=self.iPix[iDir],self.jPix[iDir]
            self.SumJJsq[i,j,:]=self.StackedBeamDict[iDir]["SumJJsq"]

        self.SumJJsq/=self.SumWsq.reshape(1,1,self.MS.Nchan)
        #self.SumJJsq=np.rollaxis(SumJJsq,2)#np.mean(SumJJsq,axis=2)
        self.SumJJsq=np.mean(self.SumJJsq,axis=2)


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
        # pylab.colorbar()
        # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
        # pylab.imshow(ifzfCF.real,interpolation="nearest",vmin=vmin,vmax=vmax,extent=(0,1,0,1))
        # pylab.colorbar()
        # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
        # pylab.imshow(if_z_f_SumJJsq,interpolation="nearest",vmin=vmin,vmax=vmax,extent=(0,1,0,1))
        # pylab.colorbar()
        # pylab.show(False)
        # stop

        self.ifzfCF=np.real(ifzfCF)
        self.SmoothBeam=np.real(if_z_f_SumJJsq)
        np.save(self.CachePath,self.SmoothBeam)
        self.VS.maincache.saveCache("SmoothBeam.npy")
        # print>>log, ModColor.Str("======================= Done calculating smooth beams ====================")
        return "HasBeenComputed"

       
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
            print>>log,"Found valid smooth beam in %s"%self.CachePath
            self.SmoothBeam=np.load(self.CachePath)

    def GiveMergedWithDiscrete(self,DiscreteMeanBeam):
        Mask=(self.ifzfCF<1e-2)
        self.SmoothBeam[Mask]=DiscreteMeanBeam[Mask]
        return self.SmoothBeam
