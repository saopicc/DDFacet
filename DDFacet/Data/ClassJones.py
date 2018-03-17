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
log = MyLogger.getLogger("ClassJones")
from DDFacet.Other import reformat
from DDFacet.Array import NpShared
import os
from DDFacet.Array import ModLinAlg
from DDFacet.Other.progressbar import ProgressBar
import ClassLOFARBeam
import ClassFITSBeam
# import ClassSmoothJones is not used anywhere, should be able to remove it
import tables


class ClassJones():

    def __init__(self, GD, MS, FacetMachine=None):
        self.GD = GD
        self.FacetMachine = FacetMachine
        self.MS = MS
        self.HasKillMSSols = False
        self.BeamTimes_kMS = np.array([], np.float32)

        # self.JonesNormSolsFile_killMS="%s/JonesNorm_killMS.npz"%ThisMSName
        # self.JonesNormSolsFile_Beam="%s/JonesNorm_Beam.npz"%ThisMSName

    def InitDDESols(self, DATA, quiet=False):
        GD = self.GD
        SolsFile = GD["DDESolutions"]["DDSols"]
        self.ApplyCal = False
        if SolsFile != "":
            self.ApplyCal = True
            self.JonesNormSolsFile_killMS, valid = self.MS.cache.checkCache(
                "JonesNorm_killMS",
                dict(VisData=GD["Data"], 
                     DDESolutions=GD["DDESolutions"], 
                     DataSelection=self.GD["Selection"],
                     ImagerMainFacet=self.GD["Image"],
                     Facets=self.GD["Facets"],
                     PhaseCenterRADEC=self.GD["Image"]["PhaseCenterRADEC"]))
            if valid:
                print>>log, "  using cached Jones matrices from %s" % self.JonesNormSolsFile_killMS
                DicoSols, TimeMapping, DicoClusterDirs = self.DiskToSols(self.JonesNormSolsFile_killMS)
            else:
                DicoSols, TimeMapping, DicoClusterDirs = self.MakeSols("killMS", DATA, quiet=quiet)
                self.MS.cache.saveCache("JonesNorm_killMS")

            DATA["killMS"] =  dict(Jones=DicoSols, TimeMapping=TimeMapping, Dirs=DicoClusterDirs)
            self.DicoClusterDirs_kMS=DicoClusterDirs

            self.HasKillMSSols = True


        ApplyBeam=(GD["Beam"]["Model"] is not None)
        if ApplyBeam:
            self.ApplyCal = True
            self.JonesNormSolsFile_Beam, valid = self.MS.cache.checkCache("JonesNorm_Beam.npz", 
                                                                          dict(VisData=GD["Data"], 
                                                                               Beam=GD["Beam"], 
                                                                               Facets=self.GD["Facets"],
                                                                               DataSelection=self.GD["Selection"],
                                                                               DDESolutions=GD["DDESolutions"],
                                                                               ImagerMainFacet=self.GD["Image"]))
            if valid:
                print>>log, "  using cached Jones matrices from %s" % self.JonesNormSolsFile_Beam
                DicoSols, TimeMapping, DicoClusterDirs = self.DiskToSols(self.JonesNormSolsFile_Beam)
            else:
                DicoSols, TimeMapping, DicoClusterDirs = self.MakeSols("Beam", DATA, quiet=quiet)
                self.MS.cache.saveCache("JonesNorm_Beam.npz")
            DATA["Beam"] =  dict(Jones=DicoSols, TimeMapping=TimeMapping, Dirs=DicoClusterDirs)

    # def ToShared(self, StrType, DicoSols, TimeMapping, DicoClusterDirs):
    #     print>>log, "  Putting %s Jones in shm" % StrType
    #     NpShared.DelAll("%sDicoClusterDirs_%s" % (self.IdSharedMem, StrType))
    #     NpShared.DelAll("%sJonesFile_%s" % (self.IdSharedMem, StrType))
    #     NpShared.DelAll("%sMapJones_%s" % (self.IdSharedMem, StrType))
    #     NpShared.DicoToShared(
    #         "%sDicoClusterDirs_%s" %
    #         (self.IdSharedMem, StrType), DicoClusterDirs)
    #     NpShared.DicoToShared(
    #         "%sJonesFile_%s" %
    #         (self.IdSharedMem, StrType), DicoSols)
    #     NpShared.ToShared(
    #         "%sMapJones_%s" %
    #         (self.IdSharedMem, StrType), TimeMapping)

    def SolsToDisk(self, OutName, DicoSols, DicoClusterDirs, TimeMapping):

        print>>log, "  Saving %s" % OutName
        l = DicoClusterDirs["l"]
        m = DicoClusterDirs["m"]
        I = DicoClusterDirs["I"]
        ra = DicoClusterDirs["ra"]
        dec = DicoClusterDirs["dec"]
        Cluster = DicoClusterDirs["Cluster"]
        t0 = DicoSols["t0"]
        t1 = DicoSols["t1"]
        tm = DicoSols["tm"]
        Jones = DicoSols["Jones"]
        TimeMapping = TimeMapping
        VisToJonesChanMapping = DicoSols["VisToJonesChanMapping"]

        # np.savez(self.JonesNorm_killMS,l=l,m=m,I=I,Cluster=Cluster,t0=t0,t1=t1,tm=tm,Jones=Jones,TimeMapping=TimeMapping)

        os.system("touch %s"%OutName)
        np.savez(file("%s.npz"%OutName, "w"),
                 l=l, m=m, I=I, Cluster=Cluster,
                 t0=t0, t1=t1, tm=tm,
                 ra=ra,dec=dec,
                 TimeMapping=TimeMapping,
                 VisToJonesChanMapping=VisToJonesChanMapping)
        np.save(file("%s.npy"%OutName, "w"),
                Jones)


    def DiskToSols(self, InName):
        # SolsFile_killMS=np.load(self.JonesNorm_killMS)
        # print>>log, "  Loading %s"%InName
        # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",InName
        SolsFile = np.load("%s.npz"%InName)
        print>>log, "  %s.npz loaded" % InName
        Jones=np.load("%s.npy"%InName)
        print>>log, "  %s.npy loaded" % InName


        DicoClusterDirs = {}
        DicoClusterDirs["l"] = SolsFile["l"]
        DicoClusterDirs["m"] = SolsFile["m"]
        DicoClusterDirs["ra"] = SolsFile["ra"]
        DicoClusterDirs["dec"] = SolsFile["dec"]
        DicoClusterDirs["I"] = SolsFile["I"]
        DicoClusterDirs["Cluster"] = SolsFile["Cluster"]
        DicoSols = {}
        DicoSols["t0"] = SolsFile["t0"]
        DicoSols["t1"] = SolsFile["t1"]
        DicoSols["tm"] = SolsFile["tm"]

        DicoSols["Jones"] = Jones

        DicoSols["VisToJonesChanMapping"] = SolsFile["VisToJonesChanMapping"]
        TimeMapping = SolsFile["TimeMapping"]
        return DicoSols, TimeMapping, DicoClusterDirs

    def MakeSols(self, StrType, DATA, quiet=False):
        print>>log, "Build solution Dico for %s" % StrType

        if StrType == "killMS":
            DicoClusterDirs_killMS, DicoSols = self.GiveKillMSSols()
            DicoClusterDirs = DicoClusterDirs_killMS
            print>>log, "  Build VisTime-to-Solution mapping"
            TimeMapping = self.GiveTimeMapping(DicoSols, DATA["times"])
            DicoClusterDirs["l"],DicoClusterDirs["m"]=self.MS.radec2lm_scalar(DicoClusterDirs["ra"],DicoClusterDirs["dec"])
            self.SolsToDisk(
                self.JonesNormSolsFile_killMS,
                DicoSols,
                DicoClusterDirs_killMS,
                TimeMapping)
        BeamJones = None
        if StrType == "Beam":

            if self.FacetMachine is not None:
                if not(self.HasKillMSSols) or self.GD["Beam"]["At"] == "facet":
                    print>>log, "  Getting beam Jones directions from facets"
                    DicoImager = self.FacetMachine.DicoImager
                    NFacets = len(DicoImager)
                    self.ClusterCatBeam = self.FacetMachine.JonesDirCat
                    DicoClusterDirs = {}
                    DicoClusterDirs["l"] = self.ClusterCatBeam.l
                    DicoClusterDirs["m"] = self.ClusterCatBeam.m
                    DicoClusterDirs["ra"] = self.ClusterCatBeam.ra
                    DicoClusterDirs["dec"] = self.ClusterCatBeam.dec
                    DicoClusterDirs["I"] = self.ClusterCatBeam.I
                    DicoClusterDirs["Cluster"] = self.ClusterCatBeam.Cluster
                else:
                    print>>log, "  Getting beam Jones directions from DDE solution tessels"
                    DicoClusterDirs = self.DicoClusterDirs_kMS
                    NDir = DicoClusterDirs["l"].size
                    self.ClusterCatBeam = np.zeros(
                        (NDir,),
                        dtype=[('Name', '|S200'),
                               ('ra', np.float),
                               ('dec', np.float),
                               ('SumI', np.float),
                               ("Cluster", int),
                               ("l", np.float),
                               ("m", np.float),
                               ("I", np.float)])
                    self.ClusterCatBeam = self.ClusterCatBeam.view(np.recarray)
                    self.ClusterCatBeam.I = self.DicoClusterDirs_kMS["I"]
                    self.ClusterCatBeam.SumI = self.DicoClusterDirs_kMS["I"]
                    self.ClusterCatBeam.ra[:] = self.DicoClusterDirs_kMS["ra"]
                    self.ClusterCatBeam.dec[:] = self.DicoClusterDirs_kMS["dec"]
            else:

                self.ClusterCatBeam = np.zeros(
                    (1,),
                    dtype=[('Name', '|S200'),
                           ('ra', np.float),
                           ('dec', np.float),
                           ('SumI', np.float),
                           ("Cluster", int),
                           ("l", np.float),
                           ("m", np.float),
                           ("I", np.float)])
                self.ClusterCatBeam = self.ClusterCatBeam.view(np.recarray)
                self.ClusterCatBeam.I = 1
                self.ClusterCatBeam.SumI = 1
                self.ClusterCatBeam.ra[0] = self.MS.rac
                self.ClusterCatBeam.dec[0] = self.MS.decc
                DicoClusterDirs = {}
                DicoClusterDirs["l"] = np.array([0.], np.float32)
                DicoClusterDirs["m"] = np.array([0.], np.float32)
                DicoClusterDirs["ra"] = self.MS.rac
                DicoClusterDirs["dec"] = self.MS.decc
                DicoClusterDirs["I"] = np.array([1.], np.float32)
                DicoClusterDirs["Cluster"] = np.array([0], np.int32)

            DicoClusterDirs_Beam = DicoClusterDirs
            DicoSols = self.GiveBeam(DATA["uniq_times"], quiet=quiet)
            print>>log, "  Build VisTime-to-Beam mapping"
            TimeMapping = self.GiveTimeMapping(DicoSols, DATA["times"])
            DicoClusterDirs["l"],DicoClusterDirs["m"]=self.MS.radec2lm_scalar(DicoClusterDirs["ra"],DicoClusterDirs["dec"])

            self.SolsToDisk(
                self.JonesNormSolsFile_Beam,
                DicoSols,
                DicoClusterDirs_Beam,
                TimeMapping)

        # if (BeamJones is not None)&(KillMSSols is not None):
        #     print>>log,"  Merging killMS and Beam Jones matrices"
        #     DicoSols=self.MergeJones(KillMSSols,BeamJones)
        # elif BeamJones is not None:
        #     DicoSols=BeamJones
        # elif KillMSSols is not None:
        #     DicoSols=KillMSSols

        DicoSols["Jones"] = np.require(
            DicoSols["Jones"],
            dtype=np.complex64,
            requirements="C")

        # ThisMSName=reformat.reformat(os.path.abspath(self.CurrentMS.MSName),LastSlash=False)
        # TimeMapName="%s/Mapping.DDESolsTime.npy"%ThisMSName
        return DicoSols, TimeMapping, DicoClusterDirs

    def GiveTimeMapping(self, DicoSols, times):
        """Builds mapping from MS rows to Jones solutions.
        Args:
            DicoSols: dictionary of Jones matrices, which includes t0 and t1 entries

        Returns:
            Vector of indices, one per each row in DATA, giving the time index of the Jones matrix
            corresponding to that row.
        """
        print>>log, "  Build Time Mapping"
        DicoJonesMatrices = DicoSols
        ind = np.zeros((times.size,), np.int32)
        nt, na, nd, _, _, _ = DicoJonesMatrices["Jones"].shape
        ii = 0
        for it in xrange(nt):
            t0 = DicoJonesMatrices["t0"][it]
            t1 = DicoJonesMatrices["t1"][it]
            ## new code: no assumption of sortedness
            ind[(times >= t0) & (times < t1)] = it
            ## old code: assumed times was sorted
            # indMStime = np.where((times >= t0) & (times < t1))[0]
            # indMStime = np.ones((indMStime.size,), np.int32)*it
            # ind[ii:ii+indMStime.size] = indMStime[:]
            # ii += indMStime.size
        return ind

    def GiveKillMSSols(self):
        GD = self.GD
        SolsFile = GD["DDESolutions"]["DDSols"]
        if isinstance(SolsFile, list):
            SolsFileList = SolsFile
        else:
            SolsFileList = [SolsFile]

        GlobalNorm=GD["DDESolutions"]["GlobalNorm"]
        if GlobalNorm is None:
            GlobalNorm=""

        GlobalNormList = GD["DDESolutions"]["GlobalNorm"]
        if not isinstance(GlobalNormList, list):
            GlobalNormList = [GD["DDESolutions"]["GlobalNorm"]
                              ]*len(GD["DDESolutions"]["DDSols"])

        JonesNormList=GD["DDESolutions"]["JonesNormList"]
        if JonesNormList is None:
            JonesNormList="AP"

        JonesNormList=GD["DDESolutions"]["JonesNormList"]
        if type(JonesNormList)!=list:
            JonesNormList=[GD["DDESolutions"]["JonesNormList"]]*len(GD["DDESolutions"]["DDSols"])

        JonesNormList = GD["DDESolutions"]["JonesNormList"]
        if not isinstance(JonesNormList, list):
            JonesNormList = [GD["DDESolutions"]["JonesNormList"]
                             ]*len(GD["DDESolutions"]["DDSols"])

        ListDicoSols = []

        for File, ThisGlobalMode, ThisJonesMode in zip(
                SolsFileList, GlobalNormList, JonesNormList):

            
            DicoClusterDirs, DicoSols, VisToJonesChanMapping = self.GiveKillMSSols_SingleFile(
                File, GlobalMode=ThisGlobalMode, JonesMode=ThisJonesMode)
            print>>log, "  VisToJonesChanMapping: %s" % str(VisToJonesChanMapping)
            ListDicoSols.append(DicoSols)

        DicoJones = ListDicoSols[0]
        for DicoJones1 in ListDicoSols[1::]:
            DicoJones = self.MergeJones(DicoJones1, DicoJones)

        DicoJones["VisToJonesChanMapping"] = VisToJonesChanMapping

        return DicoClusterDirs, DicoJones

    def ReadNPZ(self,SolsFile):
        print>>log, "  Loading solution file %s" % (SolsFile)

        self.ApplyCal = True
        DicoSolsFile = np.load(SolsFile)

        ClusterCat = DicoSolsFile["SkyModel"]
        ClusterCat = ClusterCat.view(np.recarray)
        self.ClusterCat = ClusterCat
        DicoClusterDirs = {}
        DicoClusterDirs["l"] = ClusterCat.l
        DicoClusterDirs["m"] = ClusterCat.m
        DicoClusterDirs["ra"] = ClusterCat.ra
        DicoClusterDirs["dec"] = ClusterCat.dec
        # DicoClusterDirs["l"]=ClusterCat.l
        # DicoClusterDirs["m"]=ClusterCat.m
        DicoClusterDirs["I"] = ClusterCat.SumI
        DicoClusterDirs["Cluster"] = ClusterCat.Cluster

        Sols = DicoSolsFile["Sols"]
        Sols = Sols.view(np.recarray)
        DicoSols = {}
        DicoSols["t0"] = Sols.t0
        DicoSols["t1"] = Sols.t1
        DicoSols["tm"] = (Sols.t1+Sols.t0)/2.
        nt, nf, na, nd, _, _ = Sols.G.shape
        G = np.swapaxes(Sols.G, 1, 3).reshape((nt, nd, na, nf, 2, 2))

        if "FreqDomains" in DicoSolsFile.keys():
            FreqDomains = DicoSolsFile["FreqDomains"]
            VisToJonesChanMapping = self.GiveVisToJonesChanMapping(FreqDomains)
        else:
            VisToJonesChanMapping = np.zeros((self.MS.NSPWChan,), np.int32)

        self.BeamTimes_kMS = DicoSolsFile["BeamTimes"]

        return VisToJonesChanMapping,DicoClusterDirs,DicoSols,G


    def ReadH5(self,SolsFile):
        print>>log, "  Loading H5 solution file %s" % (SolsFile)

        self.ApplyCal = True
        H=tables.open_file(SolsFile)
        raNode,decNode=H.root.sol000.source[:]["dir"].T
        times=H.root.sol000.tec000.time[:]
        lFacet, mFacet = self.FacetMachine.CoordMachine.radec2lm(raNode, decNode)
        # nt, na, nd, 1
        tec=H.root.sol000.tec000.val[:]
        scphase=H.root.sol000.scalarphase000.val[:]
        H.close()
        del(H)

        DicoClusterDirs = {}
        DicoClusterDirs["l"] = lFacet
        DicoClusterDirs["m"] = mFacet
        DicoClusterDirs["ra"] = raNode
        DicoClusterDirs["dec"] = decNode
        DicoClusterDirs["I"] = np.ones((lFacet.size,),np.float32)
        DicoClusterDirs["Cluster"] = np.arange(lFacet.size)

        ClusterCat=np.zeros((lFacet.size,),dtype=[('Name','|S200'),
                                                     ('ra',np.float),('dec',np.float),
                                                     ('l',np.float),('m',np.float),
                                                     ('SumI',np.float),("Cluster",int)])
        ClusterCat=ClusterCat.view(np.recarray)
        ClusterCat.l=lFacet
        ClusterCat.m=mFacet
        ClusterCat.ra=raNode
        ClusterCat.dec=decNode
        ClusterCat.I=DicoClusterDirs["I"]
        ClusterCat.Cluster=DicoClusterDirs["Cluster"]
        self.ClusterCat = ClusterCat



        dts=times[1::]-times[0:-1]
        if not np.max(np.abs(dts-dts[0]))<0.1:
            raise ValueError("The solutions dt should be the same")
        dt=dts[0]
        t0=times-dt/2.
        t1=times+dt/2.
        DicoSols = {}
        DicoSols["t0"] = t0
        DicoSols["t1"] = t1
        DicoSols["tm"] = (t1+t0)/2.


        nt, na, nd, _=tec.shape
        tecvals=tec.reshape((nt,na,nd,1))
        #freqs=self.FacetMachine.VS.GlobalFreqs.reshape((1,1,1,-1))
        freqs=self.MS.ChanFreq.ravel()

        scphase=scphase.reshape((nt,na,nd,1))
        freqs=freqs.reshape((1,1,1,-1))
        phase = (-8.4479745e9 * tecvals/freqs) + scphase
        # nt,na,nd,nf,1
        phase=np.swapaxes(phase,1,2)
        nf=freqs.size
        G=np.zeros((nt, nd, na, nf, 2, 2),np.complex64)
        z=np.exp(1j*phase)
        G[:,:,:,:,0,0]=z
        G[:,:,:,:,1,1]=z

        VisToJonesChanMapping = np.int32(np.arange(self.MS.NSPWChan,))

        #self.BeamTimes_kMS = DicoSolsFile["BeamTimes"]

        return VisToJonesChanMapping,DicoClusterDirs,DicoSols,G




    def GiveKillMSSols_SingleFile(
        self,
        SolsFile,
        JonesMode="AP",
        GlobalMode=""):


        if not ".h5" in SolsFile:
            if not(".npz" in SolsFile):
                Method = SolsFile
                ThisMSName = reformat.reformat(
                    os.path.abspath(self.MS.MSName),
                    LastSlash=False)
                SolsFile = "%s/killMS.%s.sols.npz" % (ThisMSName, Method)
            VisToJonesChanMapping,DicoClusterDirs,DicoSols,G=self.ReadNPZ(SolsFile)
        else:
            VisToJonesChanMapping,DicoClusterDirs,DicoSols,G=self.ReadH5(SolsFile)
            
        nt, nd, na, nf, _, _ = G.shape

        # G[:,:,:,:,0,0]=0.
        # G[:,:,:,:,1,1]=0.
        # G[:,0,:,:,0,0]=1.
        # G[:,0,:,:,1,1]=1.

        # print>>log, "!!!!!!!!!!!!!!"
        # #G[:,:,:,:,1,1]=G[:,:,:,:,0,0]
        # G.fill(0)
        # G[:,:,:,:,0,0]=1
        # G[:,:,:,:,1,1]=1
        # print>>log, "!!!!!!!!!!!!!!"

        if GlobalMode == "MeanAbsAnt":
            print>>log, "  Normalising by the mean of the amplitude (against time, freq)"
            gmean_abs = np.mean(
                np.mean(np.abs(G[:, :, :, :, 0, 0]), axis=0), axis=2)
            gmean_abs = gmean_abs.reshape((1, nd, na, 1))
            G[:, :, :, :, 0, 0] /= gmean_abs
            G[:, :, :, :, 1, 1] /= gmean_abs

        if GlobalMode == "MeanAbs":
            print>>log, "  Normalising by the mean of the amplitude (against time, freq, antenna)"
            gmean_abs = np.mean(np.mean(
                                    np.mean(
                                        np.abs(G[:, :, :, :, 0, 0]),
                                        axis=0),
                                    axis=1),
                                axis=1)
            gmean_abs = gmean_abs.reshape((1, nd, 1, 1))
            G[:, :, :, :, 0, 0] /= gmean_abs
            G[:, :, :, :, 1, 1] /= gmean_abs

        if GlobalMode == "BLBased":
            # print>>log, "  Normalising by the mean of the amplitude (against time, freq, antenna)"
            # gmean_abs = np.mean(np.mean(
            #                         np.mean(
            #                             np.abs(G[:, :, :, :, 0, 0]),
            #                             axis=0),
            #                         axis=1),
            #                     axis=1)
            # gmean_abs = gmean_abs.reshape((1, nd, 1, 1))
            # G[:, :, :, :, 0, 0] /= gmean_abs
            # G[:, :, :, :, 1, 1] /= gmean_abs

            print>>log, "  Extracting correction factor per-baseline"
            #(nt, nd, na, nf, 2, 2)
            for iDir in range(nd):
                g=G[:,iDir,:,:,0,0]
                M=np.zeros((na,na),np.float32)
                for iAnt in range(na):
                    for jAnt in range(na):
                        M[iAnt,jAnt]=np.mean(np.abs(g[:,iAnt]*g[:,jAnt].conj()))
                
                u,s,v=np.linalg.svd(M)
                gu=u[:,0].reshape((-1,1))*np.sqrt(s[0])
                #M2=gu*gu.conj().T*s[0]
                gu=np.abs(gu).reshape((1,na,1))
                gu/=np.median(gu)
                G[:,iDir,:,:,0,0]=G[:,iDir,:,:,0,0]/gu
                G[:,iDir,:,:,1,1]=G[:,iDir,:,:,1,1]/gu



        if GlobalMode == "SumBLBased":
            print>>log, "  Normalising by the mean of the amplitude (against time, freq, antenna)"
            gmean_abs = np.mean(np.mean(
                                    np.mean(
                                        np.abs(G[:, :, :, :, 0, 0]),
                                        axis=0),
                                    axis=1),
                                axis=1)
            gmean_abs = gmean_abs.reshape((1, nd, 1, 1))
            G[:, :, :, :, 0, 0] /= gmean_abs
            G[:, :, :, :, 1, 1] /= gmean_abs

            print>>log, "  Extracting normalisation factor (sum of all baselines, time, freq)"
            #(nt, nd, na, nf, 2, 2)
            for iDir in range(nd):
                g=G[:,iDir,:,:,0,0]
                M=np.zeros((na,na),np.float32)
                for iAnt in range(na):
                    for jAnt in range(na):
                        M[iAnt,jAnt]=np.mean(np.abs(g[:,iAnt]*g[:,jAnt].conj()))
                
                gu=np.sqrt(np.mean(M))
                G[:,iDir,:,:,0,0]=G[:,iDir,:,:,0,0]/gu
                G[:,iDir,:,:,1,1]=G[:,iDir,:,:,1,1]/gu





        if not("A" in JonesMode):
            print>>log, "  Normalising by the amplitude"
            G[G != 0.] /= np.abs(G[G != 0.])
        if not("P" in JonesMode):
            print>>log, "  Zero-ing the phases"
            dtype = G.dtype
            G = (np.abs(G).astype(dtype)).copy()

        # G=self.NormDirMatrices(G)

        # print "G!!!!!!!!!!!!!!!"#nt,nd,na,nf,2,2

        # G.fill(0)
        # G[:,:,:,:,0,0]=1
        # G[:,:,:,:,1,1]=1

        # Gc=G.copy()
        # Gc.fill(0)
        # N=5

        # Gc[:,N,:,:,:,:]=G[:,N,:,:,:,:]
        # G=Gc

        DicoSols["Jones"] = G

        return DicoClusterDirs, DicoSols, VisToJonesChanMapping

    def NormDirMatrices(self, G):
        return G
        RefAnt = 0
        print>>log, "  Normalising Jones Matrices with reference Antenna %i ..." % RefAnt
        nt, nd, na, nf, _, _ = G.shape

        for iDir in xrange(nd):
            for it in xrange(nt):
                for iF in xrange(nf):
                    Gt = G[it, iDir, :, iF, :, :]
                    u, s, v = np.linalg.svd(Gt[RefAnt])
                    U = np.dot(u, v)
                    for iAnt in xrange(0, na):
                        G[it, iDir, iAnt, iF, :, :] = np.dot(
                            U.T.conj(), Gt[iAnt, :, :])

        return G

    #######################################################
    ######################## BEAM #########################
    #######################################################

    def InitBeamMachine(self):
        GD = self.GD
        if GD["Beam"]["Model"] == "LOFAR":
            self.ApplyBeam = True
            self.BeamMachine = ClassLOFARBeam.ClassLOFARBeam(self.MS, self.GD)
            self.GiveInstrumentBeam = self.BeamMachine.GiveInstrumentBeam
            #print>>log, "  Estimating LOFAR beam model in %s mode every %5.1f min."%(LOFARBeamMode,DtBeamMin)
            # self.GiveInstrumentBeam=self.MS.GiveBeam
            # estimate beam sample times using DtBeamMin

        elif GD["Beam"]["Model"] == "FITS":
            self.BeamMachine = ClassFITSBeam.ClassFITSBeam(self.MS, GD["Beam"])
            self.GiveInstrumentBeam = self.BeamMachine.evaluateBeam

            # self.DtBeamDeg = GD["Beam"]["FITSParAngleIncrement"]
            # print>>log, "  Estimating FITS beam model every %5.1f min."%DtBeamMin

    def GiveBeam(self, times, quiet=False):
        GD = self.GD
        if (GD["Beam"]["Model"] is None) | (GD["Beam"]["Model"] == ""):
            print>>log, "  Not applying any beam"
            return

        self.InitBeamMachine()

        if self.BeamTimes_kMS.size != 0:
            print>>log, "  Taking beam-times from DDE-solutions"
            beam_times = self.BeamTimes_kMS
        else:
            beam_times = self.BeamMachine.getBeamSampleTimes(times, quiet=quiet)

        RAs = self.ClusterCatBeam.ra
        DECs = self.ClusterCatBeam.dec

        # from killMS2.Other.rad2hmsdms import rad2hmsdms
        # for i in range(RAs.size):
        #     ra,dec=RAs[i],DECs[i]
        #     print rad2hmsdms(ra,Type="ra").replace(" ",":"),rad2hmsdms(dec,Type="dec").replace(" ",".")

        DicoBeam = self.EstimateBeam(beam_times, RAs, DECs)

        return DicoBeam

    def GiveVisToJonesChanMapping(self, FreqDomains):
        NChanJones = FreqDomains.shape[0]
        MeanFreqJonesChan = (FreqDomains[:, 0]+FreqDomains[:, 1])/2.
        DFreq = np.abs(self.MS.ChanFreq.reshape(
            (self.MS.NSPWChan, 1))-MeanFreqJonesChan.reshape((1, NChanJones)))
        return np.argmin(DFreq, axis=1)

    def EstimateBeam(self, TimesBeam, RA, DEC,progressBar=True, quiet=False):
        TimesBeam = np.float64(np.array(TimesBeam))
        T0s = TimesBeam[:-1].copy()
        T1s = TimesBeam[1:].copy()
        Tm = (T0s+T1s)/2.
        # RA,DEC=self.BeamRAs,self.BeamDECs

        NDir=RA.size
        
        DicoBeam={}
        FreqDomains=self.BeamMachine.getFreqDomains()

        DicoBeam["VisToJonesChanMapping"]=self.GiveVisToJonesChanMapping(FreqDomains)
        if not quiet:
            print>>log,"VisToJonesChanMapping: %s"%DicoBeam["VisToJonesChanMapping"]


        DicoBeam["Jones"]=np.zeros((Tm.size,NDir,self.MS.na,FreqDomains.shape[0],2,2),dtype=np.complex64)
        DicoBeam["t0"]=np.zeros((Tm.size,),np.float64)
        DicoBeam["t1"]=np.zeros((Tm.size,),np.float64)
        DicoBeam["tm"]=np.zeros((Tm.size,),np.float64)
        
        
        rac,decc=self.MS.OriginalRadec
        pBAR= ProgressBar(Title="  Init E-Jones ")#, HeaderSize=10,TitleSize=13)
        if not progressBar: pBAR.disable()
        pBAR.render(0, Tm.size)
        for itime in range(Tm.size):
            DicoBeam["t0"][itime]=T0s[itime]
            DicoBeam["t1"][itime]=T1s[itime]
            DicoBeam["tm"][itime]=Tm[itime]
            ThisTime=Tm[itime]
            Beam=self.GiveInstrumentBeam(ThisTime,RA,DEC)
            #
            if self.GD["Beam"]["CenterNorm"]==1:
                Beam0=self.GiveInstrumentBeam(ThisTime,np.array([rac]),np.array([decc]))
                Beam0inv= ModLinAlg.BatchInverse(Beam0)
                nd,_,_,_,_=Beam.shape
                Ones=np.ones((nd, 1, 1, 1, 1),np.float32)
                Beam0inv=Beam0inv*Ones
                Beam= ModLinAlg.BatchDot(Beam0inv, Beam)
                
 
            DicoBeam["Jones"][itime]=Beam
            NDone=itime+1
            pBAR.render(NDone,Tm.size)

            DicoBeam["Jones"][itime] = Beam

        nt, nd, na, nch, _, _ = DicoBeam["Jones"].shape

        # DicoBeam["Jones"]=np.mean(DicoBeam["Jones"],axis=3).reshape((nt,nd,na,1,2,2))

        # print TimesBeam-TimesBeam[0]
        # print t0-t1
        # print DicoBeam["t1"][-1]-DicoBeam["t0"][0]

        return DicoBeam

    def MergeJones(self, DicoJ0, DicoJ1):
        T0 = DicoJ0["t0"][0]
        DicoOut = {}
        DicoOut["t0"] = []
        DicoOut["t1"] = []
        DicoOut["tm"] = []
        it = 0
        CurrentT0 = T0

        while True:
            DicoOut["t0"].append(CurrentT0)
            T0 = DicoOut["t0"][it]

            dT0 = DicoJ0["t1"]-T0
            dT0 = dT0[dT0 > 0]
            dT1 = DicoJ1["t1"]-T0
            dT1 = dT1[dT1 > 0]
            if(dT0.size == 0) & (dT1.size == 0):
                break
            elif dT0.size == 0:
                dT = dT1[0]
            elif dT1.size == 0:
                dT = dT0[0]
            else:
                dT = np.min([dT0[0], dT1[0]])

            T1 = T0+dT
            DicoOut["t1"].append(T1)
            Tm = (T0+T1)/2.
            DicoOut["tm"].append(Tm)
            CurrentT0 = T1
            it += 1

        DicoOut["t0"] = np.array(DicoOut["t0"])
        DicoOut["t1"] = np.array(DicoOut["t1"])
        DicoOut["tm"] = np.array(DicoOut["tm"])

        _, nd, na, nch, _, _ = DicoJ0["Jones"].shape
        nt = DicoOut["tm"].size
        DicoOut["Jones"] = np.zeros((nt, nd, na, 1, 2, 2), np.complex64)

        nt0 = DicoJ0["t0"].size
        nt1 = DicoJ1["t0"].size

        iG0 = np.argmin(np.abs(DicoOut["tm"].reshape(
            (nt, 1))-DicoJ0["tm"].reshape((1, nt0))), axis=1)
        iG1 = np.argmin(np.abs(DicoOut["tm"].reshape(
            (nt, 1))-DicoJ1["tm"].reshape((1, nt1))), axis=1)

        for itime in xrange(nt):
            G0 = DicoJ0["Jones"][iG0[itime]]
            G1 = DicoJ1["Jones"][iG1[itime]]
            DicoOut["Jones"][itime] = ModLinAlg.BatchDot(G0, G1)

        return DicoOut
