import numpy as np
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassSoothJones")

class ClassSoothJones():
    def __init__(self,GD,IdSharedMemData):
        self.GD=GD
        self.IdSharedMemData=IdSharedMemData

    def GiveDicoJonesMatrices(self):
        print>>log, "  Getting Jones matrices from Shared Memory"
        DicoJonesMatrices=None

        GD=self.GD

        SolsFile=GD["DDESolutions"]["DDSols"]

        if SolsFile!="":
            DicoJones_killMS=NpShared.SharedToDico("%sJonesFile_killMS"%self.IdSharedMemData)
            DicoJonesMatrices["DicoJones_killMS"]=DicoJones_killMS
            DicoJonesMatrices["DicoJones_killMS"]["MapJones"]=NpShared.GiveArray("%sMapJones_killMS"%self.IdSharedMemData)
            DicoClusterDirs_killMS=NpShared.SharedToDico("%sDicoClusterDirs_killMS"%self.IdSharedMemData)
            DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"]=DicoClusterDirs_killMS

        ApplyBeam=(GD["Beam"]["BeamModel"]!=None)
        if ApplyBeam:
            DicoJones_Beam=NpShared.SharedToDico("%sJonesFile_Beam"%self.IdSharedMemData)
            DicoJonesMatrices["DicoJones_Beam"]=DicoJones_Beam
            DicoJonesMatrices["DicoJones_Beam"]["MapJones"]=NpShared.GiveArray("%sMapJones_Beam"%self.IdSharedMemData)
            DicoClusterDirs_Beam=NpShared.SharedToDico("%sDicoClusterDirs_Beam"%self.IdSharedMemData)
            DicoJonesMatrices["DicoJones_Beam"]["DicoClusterDirs"]=DicoClusterDirs_Beam

        return DicoJonesMatrices

    def SoothJones(self,DicoJonesMatrices):
        print>>log, "Smoothing Jones matrices"
        DicoJonesMatrices=self.GiveDicoJonesMatrices()

        Apply_killMS=("DicoJones_killMS" in DicoJonesMatrices.keys())
        Apply_Beam=("DicoJones_Beam" in DicoJonesMatrices.keys())

        l0,m0=self.lmShift
        idir_kMS=0
        w_kMS=np.array([],np.float32)

        if Apply_killMS:
            DicoClusterDirs=DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"]
            lc=DicoClusterDirs["l"]
            mc=DicoClusterDirs["m"]
            sI=DicoClusterDirs["I"]
            d=np.sqrt((l0-lc)**2+(m0-mc)**2)
            idir_kMS=np.argmin(d)

        idir_Beam=0
        if Apply_Beam:
            DicoClusterDirs=DicoJonesMatrices["DicoJones_Beam"]["DicoClusterDirs"]
            lc=DicoClusterDirs["l"]
            mc=DicoClusterDirs["m"]
            d=np.sqrt((l0-lc)**2+(m0-mc)**2)
            idir_Beam=np.argmin(d)
            
        stop

