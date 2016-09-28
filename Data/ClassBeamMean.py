import numpy as np
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassBeamMean")
from DDFacet.ToolsDir import ModCoord
import ClassJones
from DDFacet.Imager import ModCF
from DDFacet.ToolsDir import ModFFTW

class ClassBeamMean():
    def __init__(self,VS):
        self.VS=VS
        self.ListMS=self.VS.ListMS
        self.MS=self.ListMS[0]
        rac,decc=self.MS.radec
        self.CoordMachine=ModCoord.ClassCoordConv(rac,decc)
        self.CalcGrid()
        self.GD=self.VS.GD
        self.Padding=Padding

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
        # make lists of tables and row counts (one per MS)
        tabs = [ ms.GiveMainTable() for ms in self.ListMS ]
        nrows = [ tab.nrows() for tab in tabs ]
        nr = sum(nrows)
        # preallocate arrays
        # NB: this assumes nchan and ncorr is the same across all MSs in self.ListMS. Tough luck if it isn't!

        print>>log, "Loading some data for all MS..."

        self.Data={}
        for iMS,MS in zip(range(self.VS.nMS),self.ListMS):
            tab = MS.GiveMainTable()
            times = tab.getcol("TIME")
            flags = tab.getcol("FLAG")
            A0 = tab.getcol("ANTENNA1")
            A1 = tab.getcol("ANTENNA2")
            tab.close()

            weights = self.VS.VisWeights[iMS][:]
            ThisMSData={}
            ThisMSData["A0"]=A0
            ThisMSData["A1"]=A1
            ThisMSData["times"]=times
            ThisMSData["flags"]=flags
            ThisMSData["W"]=weights
    
        self.Data[iMS]=ThisMSData

    def CalcMeanBeam(self):
        
        print>>log, "Calculating mean beam..."
        Dt=self.GD["Beam"]["DtBeamMin"]*60.

        RAs,DECs = self.radec

        SumJJsq=np.zeros((self.npix,self.npix,self.MS.Nchan),np.float64)
        SumWsq=0.

        for iMS,MS in zip(range(self.VS.nMS),self.ListMS):
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
            DicoBeam=JonesMachine.EstimateBeam(beam_times, RAs, DECs)

            NTRange=DicoBeam["t0"].size
            for iTRange in range(len(DicoBeam)):
                t0=DicoBeam["t0"][iTRange]
                t1=DicoBeam["t1"][iTRange]
                J=DicoBeam["Jones"][iTRange]
                ind=np.where((times>=t0)&(times<t1))[0]
                A0s=A0[ind]
                A1s=A1[ind]
                fs=flags[ind]
                Ws=W[ind]
                
                J0=J[:,A0s,:,:,:]
                J1=J[:,A1s,:,:,:]
                JJ=(np.abs(J0[:,:,:,0,0])*np.abs(J1[:,:,:,0,0])+np.abs(J0[:,:,:,1,1])*np.abs(J1[:,:,:,1,1]))/2.

                WW=Ws**2
                WW=WW.reshape((1,ind.size,self.MS.Nchan))
                JJsq=WW*JJ**2

                SumWsqThisRange=np.sum(JJsq,axis=1)
                SumJJsq+=SumWsqThisRange.reshape((self.npix,self.npix,self.MS.Nchan))
                SumWsq+=np.sum(WW,axis=1)
        SumJJsq/=SumWsq.reshape(1,1,self.MS.Nchan)
        self.SumJJsq=np.mean(SumJJsq,axis=2)
        self.Smooth()
        
    def Smooth(self):
        _,_,nx,_=self.VS.FullImShape
        
        SpheM=ModCF.SpheMachine(Support=self.npix,SupportSpheCalc=111)
        CF, fCF, ifzfCF=SpheM.MakeSphe(nx)

        FT=ModFFTW.FFTW_2Donly_np()
        
        SumJJsq_Sphe=self.SumJJsq.copy()*SpheM.if_cut_fCF
        A=np.complex64(SumJJsq_Sphe.reshape((1,1,self.npix,self.npix)))
        f_SumJJsq=FT.fft(A).reshape((self.npix,self.npix))
        z_f_SumJJsq=np.complex64(ModCF.ZeroPad(f_SumJJsq,outshape=nx))
        
        if_z_f_SumJJsq=FT.ifft(z_f_SumJJsq.reshape((1,1,nx,nx))).real.reshape((nx,nx))
        if_z_f_SumJJsq/=ifzfCF


        vmin=self.SumJJsq.min()
        vmax=self.SumJJsq.max()
        import pylab
        pylab.clf()
        pylab.subplot(1,2,1)
        pylab.imshow(self.SumJJsq,interpolation="nearest")
        pylab.subplot(1,2,2)
        pylab.imshow(if_z_f_SumJJsq,interpolation="nearest",vmin=vmin,vmax=vmax)
        pylab.show(False)
        
        stop
