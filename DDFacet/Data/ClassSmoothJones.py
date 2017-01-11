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
log= MyLogger.getLogger("ClassSoothJones")
from DDFacet.Array import NpShared

class ClassSmoothJones():
    def __init__(self,GD,IdSharedMem):
        self.GD=GD
        self.IdSharedMem=IdSharedMem
        self.AlphaReg=None

    def GiveDicoJonesMatrices(self):
        print>>log, "  Getting Jones matrices from Shared Memory"
        DicoJonesMatrices={}

        GD=self.GD

        SolsFile=GD["DDESolutions"]["DDSols"]

        if SolsFile!="":
            DicoJones_killMS= NpShared.SharedToDico("%sJonesFile_killMS" % self.IdSharedMem)
            DicoJonesMatrices["DicoJones_killMS"]=DicoJones_killMS
            DicoJonesMatrices["DicoJones_killMS"]["MapJones"]=NpShared.GiveArray("%sMapJones_killMS"%self.IdSharedMem)
            DicoClusterDirs_killMS= NpShared.SharedToDico("%sDicoClusterDirs_killMS" % self.IdSharedMem)
            DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"]=DicoClusterDirs_killMS

        ApplyBeam=(GD["Beam"]["Model"] is not None)
        if ApplyBeam:
            DicoJones_Beam= NpShared.SharedToDico("%sJonesFile_Beam" % self.IdSharedMem)
            DicoJonesMatrices["DicoJones_Beam"]=DicoJones_Beam
            DicoJonesMatrices["DicoJones_Beam"]["MapJones"]= NpShared.GiveArray("%sMapJones_Beam" % self.IdSharedMem)
            DicoClusterDirs_Beam= NpShared.SharedToDico("%sDicoClusterDirs_Beam" % self.IdSharedMem)
            DicoJonesMatrices["DicoJones_Beam"]["DicoClusterDirs"]=DicoClusterDirs_Beam

        return DicoJonesMatrices


    def FindAlpha(self):
        self.DicoJonesMatrices=self.GiveDicoJonesMatrices()
        DicoJonesMatrices=self.DicoJonesMatrices
        print>>log, "  Find Alpha for smoothing"
        l_List=DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"]["l"].tolist()
        m_List=DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"]["m"].tolist()

        NDir=len(l_List)
        Jm=DicoJonesMatrices["DicoJones_killMS"]["Jones"]
        nt,nd,na,nf,_,_=Jm.shape
        self.AlphaReg=np.zeros((NDir,na),np.float32)

        for (iDir,l,m) in zip(range(NDir),l_List,m_List):
            self.AlphaReg[iDir,:]=self.FindAlphaSingleDir(DicoJonesMatrices,l,m)
        NpShared.ToShared("%sAlphaReg" % self.IdSharedMem, self.AlphaReg)


    def SmoothJones(self):
        if self.AlphaReg is None:
            self.FindAlpha()
        DicoJonesMatrices=self.DicoJonesMatrices

        Jm=DicoJonesMatrices["DicoJones_killMS"]["Jones"]
        nt,nd,na,nf,_,_=Jm.shape
        J0=np.zeros_like(Jm)
        J0[:,:,:,:,0,0]=1
        J0[:,:,:,:,1,1]=1
        NDir,na=self.AlphaReg.shape

        for idir_kMS in range(NDir):
            for iAnt in range(na):
                alpha=self.AlphaReg[idir_kMS,iAnt]
                Jm[:,idir_kMS,iAnt,:,:,:]=Jm[:,idir_kMS,iAnt,:,:,:]*alpha+(1.-alpha)*J0[:,idir_kMS,iAnt,:,:,:]


    def FindAlphaSingleDir(self,DicoJonesMatrices,l0,m0):



        Apply_killMS=("DicoJones_killMS" in DicoJonesMatrices.keys())
        Apply_Beam=("DicoJones_Beam" in DicoJonesMatrices.keys())


        idir_kMS=0
        w_kMS=np.array([],np.float32)

        if Apply_killMS:
            DicoClusterDirs=DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"]
            lc=DicoClusterDirs["l"]
            mc=DicoClusterDirs["m"]
            d=np.sqrt((l0-lc)**2+(m0-mc)**2)
            idir_kMS=np.argmin(d)
            sI=DicoClusterDirs["I"]

            # (10, 4, 36, 5, 2, 2)
            JonesMatrices_killMS=DicoJonesMatrices["DicoJones_killMS"]["Jones"]
            JonesMatrices_killMS=(np.abs(JonesMatrices_killMS[:,idir_kMS,:,:,0,0])+np.abs(JonesMatrices_killMS[:,idir_kMS,:,:,1,1]))/2.
            JonesMatrices_killMS=np.mean(JonesMatrices_killMS,axis=-1)
            JonesMatrices_killMS=np.mean(JonesMatrices_killMS,axis=0)

            MapJones_killMS=DicoJonesMatrices["DicoJones_killMS"]["MapJones"]
            VisToJonesChanMapping_killMS=np.int32(DicoJonesMatrices["DicoJones_killMS"]["VisToJonesChanMapping"])

        idir_Beam=0
        if Apply_Beam:
            DicoClusterDirs=DicoJonesMatrices["DicoJones_Beam"]["DicoClusterDirs"]
            lc=DicoClusterDirs["l"]
            mc=DicoClusterDirs["m"]
            d=np.sqrt((l0-lc)**2+(m0-mc)**2)
            idir_Beam=np.argmin(d)

            JonesMatrices_Beam=DicoJonesMatrices["DicoJones_Beam"]["Jones"]
            JonesMatrices_Beam=(np.abs(JonesMatrices_Beam[:,idir_Beam,:,:,0,0])+np.abs(JonesMatrices_Beam[:,idir_Beam,:,:,1,1]))/2.
            JonesMatrices_Beam=np.mean(JonesMatrices_Beam,axis=-1)
            JonesMatrices_Beam=np.mean(JonesMatrices_Beam,axis=0)

            MapJones_Beam=DicoJonesMatrices["DicoJones_Beam"]["MapJones"]
            VisToJonesChanMapping_Beam=np.int32(DicoJonesMatrices["DicoJones_Beam"]["VisToJonesChanMapping"])

        Jm=DicoJonesMatrices["DicoJones_killMS"]["Jones"]
        nt,nd,na,nf,_,_=Jm.shape
        #All_alpha=JonesMatrices_killMS*sI[idir_kMS]/np.max(sI)
        All_alpha=JonesMatrices_Beam*sI[idir_kMS]/np.max(sI)
        AlphaDir=np.zeros((na,),np.float32)

        for iAnt in range(na):
            alpha=np.min([1.,All_alpha[iAnt]])
            alpha=np.max([0.,alpha])
            AlphaDir[iAnt]=np.sqrt(alpha)

        return AlphaDir

        



