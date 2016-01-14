import numpy as np
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassBeamMean")
from DDFacet.ToolsDir import ModCoord
import ClassJones

class ClassBeamMean():
    def __init__(self,VS):
        self.VS=VS
        self.ListMS=self.VS.ListMS
        self.MS=self.ListMS[0]
        rac,decc=self.MS.radec
        self.CoordMachine=ModCoord.ClassCoordConv(rac,decc)
        self.CalcGrid()
        self.GD=self.VS.GD

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
        self.MeanJJsq=np.zeros((npix,npix),np.float64)

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
            stop

            for irow in range(A0.size):
                ThisTime=times[irow]
                ClosestITime=np.argmin(ThisTime-beam_times)

                BeamThisTime=DicoBeam["Jones"][ClosestITime]
                stop
                    
