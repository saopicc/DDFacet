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
from DDFacet.Other import logger
log = logger.getLogger("ClassJones")
from DDFacet.Other import reformat
from DDFacet.Array import NpShared
import os
from DDFacet.Array import ModLinAlg
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Data import ClassLOFARBeam
from DDFacet.Data import ClassFITSBeam
from DDFacet.Data import ClassGMRTBeam
from DDFacet.Data import ClassATCABeam as ClassATCABeam

# import ClassSmoothJones is not used anywhere, should be able to remove it

import tables
import glob
from scipy.interpolate import interp1d
import casacore.tables as pt
import hashlib
import DDFacet.Other.PrintList

def _which_solsfile(sol_files, req_times, solset, apply_map):
    central_time = np.mean(np.unique(req_times))
    sol_times = []
    for sol_file in sol_files:
        with tables.open_file(sol_file) as H:
            _solset = getattr(H.root, solset)
            for soltab, v in apply_map.items():
                if not v:
                    continue
                _soltab = getattr(_solset, soltab)
                sol_times.append(np.mean(_soltab.time[:]))
                break
    closest = np.argmin(np.abs(np.subtract(central_time, np.array(sol_times))))
    return sol_files[closest]

def _parse_solsfile(SolsFile):
    """Parses the input SolsFile in order to use an input h5parm file to apply solutions on-the-fly.
        The solution directions dictate the on-the-fly facet layout.
        Args:
        :param SolsFile: str the h5parm solution file to apply, with encoded
            solution spec. `SolsFile` should follow the format:
                <pathto>/<solution_file>.h5:<solset_1>+<solset_2>+.../<soltab_a>+<soltab_b>+...

            Everything before the `:` is the h5parm path.
            The h5parm file must follow the format defined by losoto>=2.0
            The axis ordering for all soltabs is internally converted to:
                [pol, dir, ant, freq, time] for soltabs with frequency dependence
                [pol, dir, ant, time] for soltabs without frequency dependence

            Everything after the `:` is called `instructions` and specifies how to use the h5parm.
            `instructions` follow the format:
                ['+' separated list of solsets to use]/['+' separated list of soltabs to apply]

            Example:
                ../solutions/reference_solutions.h5:screen_posterior_sol+posterior_sol/tec000+amplitude000

            would create gains from dTEC and amplitudes in the solsets `posterior_sol` and
            `screen_posterior_sol` and concatenate them into one solution (by concatenating the directions).
            The different solsets must have the same layout except for the directions.

            Valid soltabs must come from the list.
            ['tec000','phase000','amplitude000']
            Only valid combinations are allowed, i.e. don't try to use `tec000` and `phase000`.

            If nothing comes after the `:` or there is no `:` then we assume the
            solution spec of `sol000/tec000`.
        Returns:
        h5file: str
        apply_solsets: list of solsets as str
        apply_map: dict of soltabs to apply
    """
    print("  Parsing solutions %s" % (SolsFile), file=log)
    # parse SolsFile
    _valid_soltabs = ['tec000', 'phase000', 'amplitude000']
    apply_map = {s: False for s in _valid_soltabs}
    apply_solsets = []
    split = SolsFile.split(":")
    if len(split) > 2:
        raise ValueError(
            "SolsFile {} should be of format `<pathto>/<solution_file>.h5:<solset_1>+<solset_2>+.../<soltab_a>+<soltab_b>+...`".format(
                SolsFile))
    elif len(split) == 2:
        h5file, instructions = split
    elif len(split) == 1:
        h5file = split[0]
        instructions = None
    if len(h5file) == 0:
        raise ValueError("Invalid H5parm name in SolsFile {}".format(SolsFile))
    if instructions is None:
        apply_map['tec000'] = True
        apply_solsets.append('sol000')
    else:
        instructions_split = instructions.split("/")
        if len(instructions_split) != 2:
            raise ValueError(
                "Invalid instructions {}, should be `<solset_1>+<solset_2>+.../<soltab_a>+<soltab_b>+...`".format(
                    instructions))
        solsets, soltabs = instructions_split
        for solset in solsets.split('+'):
            if len(solset) > 0:
                apply_solsets.append(solset)
        for soltab in soltabs.split('+'):
            if len(soltab) > 0:
                if soltab not in _valid_soltabs:
                    raise ValueError('Invalid soltab {} must be one of {}'.format(soltab, _valid_soltabs))
                apply_map[soltab] = True
    if len(apply_solsets) == 0:
        raise ValueError('No solsets provided')
    if apply_map['tec000'] and apply_map['phase000']:
        raise ValueError("Cannot apply both phase and tec")
    if ~np.any(np.array(list(apply_map.values()))):
        raise ValueError("No valid soltabs specified")
    return h5file, apply_solsets, apply_map


class ClassJones():

    def __init__(self, GD, MS, FacetMachine=None, CacheMode=True):
        self.GD = GD
        self.FacetMachine = FacetMachine
        self.MS = MS
        self.HasKillMSSols = False
        self.BeamTimes_kMS = np.array([], np.float32)
        self.CacheMode=CacheMode
        
        # self.JonesNormSolsFile_killMS="%s/JonesNorm_killMS.npz"%ThisMSName
        # self.JonesNormSolsFile_Beam="%s/JonesNorm_Beam.npz"%ThisMSName

    def InitDDESols(self, DATA, quiet=False):
        GD = self.GD
        SolsFile = GD["DDESolutions"]["DDSols"]
        self.ApplyCal = False
        if SolsFile != "" and SolsFile is not None:
            self.ApplyCal = True
            valid=False
            if self.CacheMode:
                self.JonesNormSolsFile_killMS, valid = self.MS.cache.checkCache("JonesNorm_killMS",
                                                                                dict(VisData=GD["Data"], 
                                                                                     DDESolutions=GD["DDESolutions"], 
                                                                                     DataSelection=self.GD["Selection"],
                                                                                     ImagerMainFacet=self.GD["Image"],
                                                                                     Facets=self.GD["Facets"],
                                                                                     PhaseCenterRADEC=self.GD["Image"]["PhaseCenterRADEC"]))
            if valid:
                print("  using cached Jones matrices from %s" % self.JonesNormSolsFile_killMS, file=log)
                DicoSols, TimeMapping, DicoClusterDirs = self.DiskToSols(self.JonesNormSolsFile_killMS)
            else:
                DicoSols, TimeMapping, DicoClusterDirs = self.MakeSols("killMS", DATA, quiet=quiet)
                if self.CacheMode: self.MS.cache.saveCache("JonesNorm_killMS")

            # DEBUG plot
            #if True:
            #    print('Debugging plots...')
            #    import pylab as plt
            #    import os
            #    output = os.path.abspath('./debug_figs2')
            #    if not os.path.exists(output):
            #        os.makedirs(output)
            #    Nt, Nd, Na, Nf, _, _ = DicoSols['Jones'].shape
            #    eff_phase = np.angle(DicoSols['Jones'])
            #    print('len time mapping:', len(TimeMapping))
            #    for d in range(Nd):
            #        print('d %i' % d)
            #        for a in range(Na):
            #            print('a %i' % a)
            #            fig, axs = plt.subplots(1,1,figsize=(20,20))
            #            #plt.colorbar(img)
            #            img=axs.imshow(eff_phase[TimeMapping,d,a,:,0,0].T, vmin = - np.pi, vmax = np.pi,aspect='auto', cmap = plt.cm.jet)
            #            print(eff_phase[TimeMapping,d,a,:,0,0].T.shape)
            #    
            #            #plt.colorbar(img)
            #            plt.savefig(os.path.join(output,'gains_{}_{}.png'.format(d,a)))
            #            plt.close('all')
            #    exit()

            DATA["killMS"] =  dict(Jones=DicoSols, TimeMapping=TimeMapping, Dirs=DicoClusterDirs)
            self.DicoClusterDirs_kMS=DicoClusterDirs

            self.HasKillMSSols = True

        ApplyBeam=(GD["Beam"]["Model"] is not None) and (GD["Beam"]["Model"]!="")
        if ApplyBeam:
            self.ApplyCal = True
            valid=False
            if self.CacheMode:
                def __hashjsonbeamsets(GD):
                    beamsets = GD["Beam"]["FITSFile"]
                    if not isinstance(beamsets, list):
                        beamsets = beamsets.split(',')
                    lbeamsethashes = {}
                    for bs in beamsets:
                        if os.path.splitext(bs)[1] == ".json":
                            if os.path.exists(bs):
                                with open(bs, "r") as fbs:
                                    lbeamsethashes["bs"] = hashlib.md5(fbs.read().encode()).hexdigest()
                    return lbeamsethashes

                self.JonesNormSolsFile_Beam, valid = self.MS.cache.checkCache("JonesNorm_Beam.npz", 
                                                                              dict(VisData=GD["Data"], 
                                                                                   Beam=GD["Beam"], 
                                                                                   Facets=self.GD["Facets"],
                                                                                   DataSelection=self.GD["Selection"],
                                                                                   DDESolutions=GD["DDESolutions"],
                                                                                   ImagerMainFacet=self.GD["Image"]))
            if valid:
                print("  using cached Jones matrices from %s" % self.JonesNormSolsFile_Beam, file=log)
                DicoSols, TimeMapping, DicoClusterDirs = self.DiskToSols(self.JonesNormSolsFile_Beam)
            else:
                DicoSols, TimeMapping, DicoClusterDirs = self.MakeSols("Beam", DATA, quiet=quiet)
                if self.CacheMode: self.MS.cache.saveCache("JonesNorm_Beam.npz")
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

        print("  Saving %s" % OutName, file=log)
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
        np.savez(open("%s.npz"%OutName, "wb"),
                 l=l, m=m, I=I, Cluster=Cluster,
                 t0=t0, t1=t1, tm=tm,
                 ra=ra,dec=dec,
                 TimeMapping=TimeMapping,
                 VisToJonesChanMapping=VisToJonesChanMapping)
        np.save(open("%s.npy"%OutName, "wb"),
                Jones)


    def DiskToSols(self, InName):
        # SolsFile_killMS=np.load(self.JonesNorm_killMS)
        # print>>log, "  Loading %s"%InName
        # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",InName
        SolsFile = np.load("%s.npz"%InName)
        print("  %s.npz loaded" % InName, file=log)
        Jones=np.load("%s.npy"%InName)
        print("  %s.npy loaded" % InName, file=log)

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
        print("Build solution Dico for %s" % StrType, file=log)

        if StrType == "killMS":
            DicoClusterDirs_killMS, DicoSols = self.GiveKillMSSols()
            DicoClusterDirs = DicoClusterDirs_killMS
            print("  Build VisTime-to-Solution mapping", file=log)
            TimeMapping = self.GiveTimeMapping(DicoSols, DATA["times"])
            DicoClusterDirs["l"],DicoClusterDirs["m"]=self.MS.radec2lm_scalar(DicoClusterDirs["ra"],DicoClusterDirs["dec"])
            if self.CacheMode:
                self.SolsToDisk(self.JonesNormSolsFile_killMS,
                                DicoSols,
                                DicoClusterDirs_killMS,
                                TimeMapping)
        BeamJones = None
        if StrType == "Beam":

            if self.FacetMachine is not None:
                if not(self.HasKillMSSols) or self.GD["Beam"]["At"] == "facet":
                    print("  Getting beam Jones directions from facets", file=log)
                    DicoImager = self.FacetMachine.DicoImager
                    NFacets = len(DicoImager)
                    self.ClusterCatBeam = self.FacetMachine.FacetDirCat
                    DicoClusterDirs = {}
                    DicoClusterDirs["l"] = self.ClusterCatBeam.l
                    DicoClusterDirs["m"] = self.ClusterCatBeam.m
                    DicoClusterDirs["ra"] = self.ClusterCatBeam.ra
                    DicoClusterDirs["dec"] = self.ClusterCatBeam.dec
                    DicoClusterDirs["I"] = self.ClusterCatBeam.I
                    DicoClusterDirs["Cluster"] = self.ClusterCatBeam.Cluster
                else:
                    print("  Getting beam Jones directions from DDE solution tessels", file=log)
                    DicoClusterDirs = self.DicoClusterDirs_kMS
                    NDir = DicoClusterDirs["l"].size
                    self.ClusterCatBeam = np.zeros(
                        (NDir,),
                        dtype=[('Name', '|S200'),
                               ('ra', float),
                               ('dec', float),
                               ('SumI', float),
                               ("Cluster", int),
                               ("l", float),
                               ("m", float),
                               ("I", float)])
                    self.ClusterCatBeam = self.ClusterCatBeam.view(np.recarray)
                    self.ClusterCatBeam.I = self.DicoClusterDirs_kMS["I"]
                    self.ClusterCatBeam.SumI = self.DicoClusterDirs_kMS["I"]
                    self.ClusterCatBeam.ra[:] = self.DicoClusterDirs_kMS["ra"]
                    self.ClusterCatBeam.dec[:] = self.DicoClusterDirs_kMS["dec"]
            else:

                self.ClusterCatBeam = np.zeros(
                    (1,),
                    dtype=[('Name', '|S200'),
                           ('ra', float),
                           ('dec', float),
                           ('SumI', float),
                           ("Cluster", int),
                           ("l", float),
                           ("m", float),
                           ("I", float)])
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
            print("  Build VisTime-to-Beam mapping", file=log)
            TimeMapping = self.GiveTimeMapping(DicoSols, DATA["times"])
            DicoClusterDirs["l"],DicoClusterDirs["m"]=self.MS.radec2lm_scalar(DicoClusterDirs["ra"],DicoClusterDirs["dec"])

            if self.CacheMode:
                self.SolsToDisk(self.JonesNormSolsFile_Beam,
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
        print("  Build Time Mapping", file=log)
        DicoJonesMatrices = DicoSols
        ind = np.zeros((times.size,), np.int32)
        nt, _, _, _, _, _ = DicoJonesMatrices["Jones"].shape
        #ii = 0
        for it in range(nt):
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
        isol=0
        for File, ThisGlobalMode, ThisJonesMode in zip(
                SolsFileList, GlobalNormList, JonesNormList):
            
            DicoClusterDirs, DicoSols, VisToJonesChanMapping = self.GiveKillMSSols_SingleFile(
                File, GlobalMode=ThisGlobalMode, JonesMode=ThisJonesMode)

            print("  VisToJonesChanMapping: %s" % DDFacet.Other.PrintList.ListToStr(VisToJonesChanMapping), file=log)
            ListDicoSols.append(DicoSols)
            #if isol==1: stop
            #isol+=1

        DicoJones = ListDicoSols[0]
        # DicoJones["Jones"][...,0,0]=1
        # DicoJones["Jones"][...,1,0]=0
        # DicoJones["Jones"][...,0,1]=0
        # DicoJones["Jones"][...,1,1]=1
        for DicoJones1 in ListDicoSols[1::]:
            DicoJones = self.MergeJones(DicoJones1, DicoJones)
            VisToJonesChanMapping = self.GiveVisToJonesChanMapping(DicoJones["FreqDomains"])
            print("  VisToJonesChanMapping: %s" % DDFacet.Other.PrintList.ListToStr(VisToJonesChanMapping), file=log)
        #stop
        DicoJones["VisToJonesChanMapping"] = VisToJonesChanMapping


        return DicoClusterDirs, DicoJones

    def ReadNPZ(self,SolsFile):
        print("  Loading solution file %s" % (SolsFile), file=log)

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

        if "MaskedSols" in DicoSolsFile.keys():
            m0=np.bool8(DicoSolsFile["MaskedSols"][0,:,0,0,0,0])
            m=np.bool8(1-DicoSolsFile["MaskedSols"][0,:,0,0,0,0])
            # GSel=Sols.G[:,m,:,:,:,:]
            GSel=Sols.G
            GSel[:,m0,:,:,0,0]=1
            GSel[:,m0,:,:,0,1]=0
            GSel[:,m0,:,:,1,0]=0
            GSel[:,m0,:,:,1,1]=1
        else:
            m=slice(None)
            GSel=Sols.G

        
        nt, nf, na, nd, _, _ = GSel.shape
        G = np.swapaxes(GSel, 1, 3).reshape((nt, nd, na, nf, 2, 2))

        if "FreqDomains" in DicoSolsFile.keys():
            print("  Getting Jones frequency domains from solutions file", file=log)
            FreqDomains = DicoSolsFile["FreqDomains"]
            # FreqDomains = FreqDomains[m,:]
            VisToJonesChanMapping = self.GiveVisToJonesChanMapping(FreqDomains)
            DicoSols["FreqDomains"]=FreqDomains
        else:
            print("  No frequency domains informations...", file=log)
            VisToJonesChanMapping = np.zeros((self.MS.NSPWChan,), np.int32)

        #print((G.shape,FreqDomains.shape,VisToJonesChanMapping ), file=log)

        self.BeamTimes_kMS = DicoSolsFile["BeamTimes"]

        return VisToJonesChanMapping,DicoClusterDirs,DicoSols,G

    def ReadH5(self, SolsFile):
        """Use an input h5parm file to apply solutions on-the-fly.
        The solution direcions dictate the on-the-fly facet layout.
        Args:
        :param SolsFile: str the h5parm solution file to apply, with encoded
            solution spec. `SolsFile` should follow the format:
                <pathto>/<solution_file>.h5:<solset_1>+<solset_2>+.../<soltab_a>+<soltab_b>+...

            Everything before the `:` is the h5parm path.
            The h5parm file must follow the format defined by losoto>=2.0
            The axis ordering for all soltabs is:
                [pol, dir, ant, freq, time] for soltabs with frequency dependence
                [pol, dir, ant, time] for soltabs without frequency dependence

            Everything after the `:` is called `instructions` and specifies how to use the h5parm.
            `instructions` follow the format:
                ['+' separated list of solsets to use]/['+' separated list of soltabs to apply]

            Example:
                ../solutions/reference_solutions.h5:screen_posterior_sol+posterior_sol/tec000+amplitude000

            would create gains from dTEC and amplitudes in the solsets `posterior_sol` and
            `screen_posterior_sol` and concatenate them into one solution (by concatenating the directions).
            The different solsets must have the same layout except for the directions.

            Valid soltabs must come from the list.
            ['tec000','phase000','amplitude000']
            Only valid combinations are allowed, i.e. don't try to use `tec000` and `phase000`.

            If nothing comes after the `:` or there is no `:` then we assume the
            solution spec of `sol000/tec000`.
        """
        # TODO: if more than one solset present for a MS file and dirs or times differ, this will fail.

        def reorderAxes( a, oldAxes, newAxes ):
            """
            Reorder axis of an array to match a new name pattern.
        
            Parameters
            ----------
            a : np array
                The array to transpose.
            oldAxes : list of str
                A list like ['time','freq','pol'].
                It can contain more axes than the new list, those are ignored.
                This is to pass to oldAxis the soltab.getAxesNames() directly even on an array from getValuesIter()
            newAxes : list of str
                A list like ['time','pol','freq'].
        
            Returns
            -------
            np array
                With axis transposed to match the newAxes list.
            """
            oldAxes = [ax for ax in oldAxes if ax in newAxes]
            idx = [ oldAxes.index(ax) for ax in newAxes ]
            return np.transpose(a, idx)

        self.ApplyCal = True

        h5file, apply_solsets, apply_map = _parse_solsfile(SolsFile)
        print("Parsing h5file pattern {}".format(h5file), file=log)
        h5files = glob.glob(h5file)
        with pt.table(self.MS.MSName, ack=False) as t:
            req_times = np.unique(t.getcol('TIME'))
        with pt.table(self.MS.MSName+'/ANTENNA', ack=False) as t:
            req_ants = t.getcol('NAME')
        h5file = _which_solsfile(h5files, req_times, apply_solsets[0], apply_map)
        print( "  Applying {} solset {} soltabs {}".format(h5file, apply_solsets, apply_map), file=log)

        # times = None
        with tables.open_file(h5file) as H:
            # Prepare unite Jones matrix array to be filled with the gains
            first_soltab = [key for key, value in apply_map.items() if value][0] # get name of the first soltab
            # Restrict Jones matrices to times present in the MS file.
            t_h5 = getattr(getattr(H.root,apply_solsets[0]), first_soltab).time # times in h5parm
            t_h5_startidx = np.argmin(np.abs(t_h5 - req_times[0])) # time idx in h5 closest to first time in MS
            t_h5_stopidx = np.argmin(np.abs(t_h5 - req_times[-1])) + 1 # time idx in h5 after closest to end time in MS
            times = t_h5[t_h5_startidx:t_h5_stopidx]
            Nt = len(times)
            Nd = len(getattr(H.root,apply_solsets[0]).source[:]['name'])
            Na = len(req_ants)
            freqs = self.MS.ChanFreq.ravel().astype(np.float32)
            Nf = freqs.size
            Np = 2
            gains = np.ones((Nt,Nd,Na,Nf,Np), dtype=np.complex64)

            lm, radec = [], []
            for solset in apply_solsets:
                _solset = getattr(H.root, solset)

                dirnames_solset = _solset.source[:]['name'] # keep track to re-arrange order of solutions later
                raNode, decNode = _solset.source[:]["dir"].T
                lFacet, mFacet = self.FacetMachine.CoordMachine.radec2lm(raNode, decNode)
                radec.append(np.stack([raNode, decNode], axis=1))
                lm.append(np.stack([lFacet, mFacet], axis=1))

                for soltab, v in apply_map.items():
                    if not v:
                        continue
                    _soltab = getattr(_solset, soltab)
                    # assert times
                    if not len(_soltab.time) >= t_h5_stopidx:
                        raise ValueError("Times not the same between solsets")
                    if ~np.all(np.isclose(_soltab.time[t_h5_startidx:t_h5_stopidx], times)):
                        raise ValueError("Times not the same between solsets")

                    val = np.array(_soltab.val, np.float32) # Npols, Nd, Na, (Nf), Nt - arbitrary order
                    val[_soltab.weight == 0] = np.nan # set flagged data to nan
                    # check axes order and reshape
                    axes_order = _soltab.val.attrs['AXES'].decode().split(',')
                    if 'freq' in axes_order: # phase, amplitude
                        val = reorderAxes(val, axes_order, ['dir', 'ant', 'freq', 'time', 'pol'])
                    else: # tec
                        val = reorderAxes(val, axes_order, ['dir', 'ant', 'time', 'pol'])
                    val = val[...,t_h5_startidx:t_h5_stopidx,:] # select only time range also in MS
                    # times = _soltab.time[:]

                    antnames_soltab = _soltab.ant[:].tolist()
                    ant_idx = [antnames_soltab.index(name.encode()) for name in req_ants] # find only antennas useful for this dataset
                    try:
                        dirnames_soltab = _soltab.dir[:].tolist()
                    except:
                        dirnames_soltab = _soltab.dir[:] # to be removed when no old soltab are around
                    dir_idx = [dirnames_soltab.index(name) for name in dirnames_solset] # find only directions useful for this dataset
                    if soltab == 'tec000':
                        tec_conv = (-8.4479745e6 / freqs).astype(np.float32)
                        val = val[dir_idx][:,ant_idx]
                        # Nd, Na, Nt, Np, Nf
                        phase = tec_conv * val[..., None]
                        # Nt, Nd, Na, Nf, Np
                        phase = phase.transpose((2, 0, 1, 4, 3))
                        gains *= np.exp(1j * phase)
                    if soltab == 'phase000':
                        val = val[dir_idx][:,ant_idx]
                        _freqs = np.array(_soltab.freq, np.float32)
                        # Nd, Na, Nf, Nt, Np
                        phase = interp1d(_freqs, val, axis=2, kind='nearest', bounds_error=False,
                                    fill_value='extrapolate')(freqs) # linear?
                        # Nt, Nd, Na, Nf, Np
                        phase = phase.transpose((3, 0, 1, 2, 4))
                        gains *= np.exp(1j * phase)
                    if soltab == 'amplitude000':
                        val = val[dir_idx][:,ant_idx]
                        _freqs = np.array(_soltab.freq, np.float32)
                        # Nd, Na, Nf, Nt, Np
                        amplitude = np.abs(interp1d(_freqs, val, axis=2, kind='nearest', bounds_error=False,
                                             fill_value='extrapolate')(freqs)) # linear?
                        # Nt, Nd, Na, Nf, Np
                        amplitude = amplitude.transpose((3, 0, 1, 2, 4))
                        gains *= amplitude
            # Nd,2
            lm = np.concatenate(lm, axis=0)
            radec = np.concatenate(radec, axis=0)

        DicoClusterDirs = {}
        DicoClusterDirs["l"] = lm[:, 0]
        DicoClusterDirs["m"] = lm[:, 1]
        DicoClusterDirs["ra"] = radec[:, 0]
        DicoClusterDirs["dec"] = radec[:, 1]
        DicoClusterDirs["I"] = np.ones((lm.shape[0],), np.float32)
        DicoClusterDirs["Cluster"] = np.arange(lm.shape[0])

        ClusterCat = np.zeros((lm.shape[0],), dtype=[('Name', '|S200'),
                                                     ('ra', float), ('dec', float),
                                                     ('l', float), ('m', float),
                                                     ('SumI', float), ("Cluster", int)])
        ClusterCat = ClusterCat.view(np.recarray)
        ClusterCat.l = lm[:, 0]
        ClusterCat.m = lm[:, 1]
        ClusterCat.ra = radec[:, 0]
        ClusterCat.dec = radec[:, 1]
        ClusterCat.I = DicoClusterDirs["I"]
        ClusterCat.Cluster = DicoClusterDirs["Cluster"]
        self.ClusterCat = ClusterCat

        # find time ranges of solutions
        diff = np.diff(times)/2.
        t_mid = times[1:] - diff # mid point per interval
        DicoSols = {}
        DicoSols["t0"] = np.insert(t_mid, 0, times[0]-diff[0])
        DicoSols["t1"] = np.append(t_mid, times[-1]+diff[-1])
        DicoSols["tm"] = times

        Nt, Nd, Na, Nf, Np = gains.shape

        G = np.zeros((Nt, Nd, Na, Nf, 2, 2), np.complex64)

        gains[np.isnan(gains)] = 1.
        G[:, :, :, :, 0, 0] = gains[...,0] # XX
        G[:, :, :, :, 1, 1] = gains[...,1] # YY

        # DEBUG plot
        #if True:
        #    import pylab as plt
        #    import os
        #    output = os.path.abspath('./debug_figs')
        #    if not os.path.exists(output):
        #        os.makedirs(output)
        #    eff_phase = np.angle(G)
        #    eff_amp = np.abs(G)
        #    #print(freqs, _freqs)
        #    for d in range(Nd):
        #        for a in range(23,Na):
        #            fig, axs = plt.subplots(4,1,figsize=(20,20))
        #            img = axs[0].imshow(eff_amp[:,d,a,:,0,0].T, cmap='hsv', vmin = 0.8, vmax = 1.2,aspect='auto')
        #            #plt.colorbar(img)
        #            img=axs[1].imshow(eff_phase[:,d,a,:,0,0].T, vmin = - np.pi, vmax = np.pi,aspect='auto', cmap = plt.cm.jet)
        #            img = axs[2].plot(np.std(eff_amp[:,d,a,:,0,0],axis=1))
        #            #plt.colorbar(img)
        #            img=axs[3].plot(np.std(eff_phase[:,d,a,:,0,0],axis=1))
        #    
        #            #plt.colorbar(img)
        #            plt.savefig(os.path.join(output,'gains_{}_{}.png'.format(d,a)))
        #            plt.close('all')

        VisToJonesChanMapping = np.int32(np.arange(self.MS.NSPWChan, ))

        # self.BeamTimes_kMS = DicoSolsFile["BeamTimes"]

        return VisToJonesChanMapping, DicoClusterDirs, DicoSols, G


    def GiveKillMSSols_SingleFile(
        self,
        SolsFile,
        JonesMode="AP",
        GlobalMode=""):

        if not ".h5" in SolsFile:
            # if not(".npz" in SolsFile):
            #     Method = SolsFile
            #     ThisMSName = reformat.reformat(
            #         os.path.abspath(self.MS.MSName),
            #         LastSlash=False)
            #     SolsFile = "%s/killMS.%s.sols.npz" % (ThisMSName, Method)

            if not(".npz" in SolsFile):
                SolsDir=self.GD["DDESolutions"]["SolsDir"]
                if SolsDir is None or SolsDir=="":
                    Method = SolsFile
                    ThisMSName = reformat.reformat(os.path.abspath(self.MS.MSName), LastSlash=False)
                    SolsFile = "%s/killMS.%s.sols.npz" % (ThisMSName, SolsFile)
                else:
                    _MSName=reformat.reformat(os.path.abspath(self.MS.MSName).split("/")[-1])
                    DirName=os.path.abspath("%s%s"%(reformat.reformat(SolsDir),_MSName))
                    if not os.path.isdir(DirName):
                        os.makedirs(DirName)
                    SolsFile="%s/killMS.%s.sols.npz"%(DirName,SolsFile)

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
        # print>>log, "SOLUTIONS RESET TO UNITY!!!!!!!!!!!!!!"

        if GlobalMode == "MeanAbsAnt":
            print("  Normalising by the mean of the amplitude (against time, freq)", file=log)
            gmean_abs = np.mean(
                np.mean(np.abs(G[:, :, :, :, 0, 0]), axis=0), axis=2)
            gmean_abs = gmean_abs.reshape((1, nd, na, 1))
            G[:, :, :, :, 0, 0] /= gmean_abs
            G[:, :, :, :, 1, 1] /= gmean_abs

        if GlobalMode == "MeanAbs":
            print("  Normalising by the mean of the amplitude (against time, freq, antenna)", file=log)
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

            print("  Extracting correction factor per-baseline", file=log)
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
            print("  Normalising by the mean of the amplitude (against time, freq, antenna)", file=log)
            gmean_abs = np.mean(np.mean(
                                    np.mean(
                                        np.abs(G[:, :, :, :, 0, 0]),
                                        axis=0),
                                    axis=1),
                                axis=1)
            gmean_abs = gmean_abs.reshape((1, nd, 1, 1))
            G[:, :, :, :, 0, 0] /= gmean_abs
            G[:, :, :, :, 1, 1] /= gmean_abs

            print("  Extracting normalisation factor (sum of all baselines, time, freq)", file=log)
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
            print("  Normalising by the amplitude", file=log)
            G[G != 0.] /= np.abs(G[G != 0.])
        if not("P" in JonesMode):
            print("  Zero-ing the phases", file=log)
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
        #print G[:,:,:,VisToJonesChanMapping,:,:]

        return DicoClusterDirs, DicoSols, VisToJonesChanMapping

    def NormDirMatrices(self, G):
        return G
        RefAnt = 0
        print("  Normalising Jones Matrices with reference Antenna %i ..." % RefAnt, file=log)
        nt, nd, na, nf, _, _ = G.shape

        for iDir in range(nd):
            for it in range(nt):
                for iF in range(nf):
                    Gt = G[it, iDir, :, iF, :, :]
                    u, s, v = np.linalg.svd(Gt[RefAnt])
                    U = np.dot(u, v)
                    for iAnt in range(0, na):
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
        elif GD["Beam"]["Model"] == "GMRT":
            self.BeamMachine = ClassGMRTBeam.ClassGMRTBeam(self.MS, GD["Beam"])
            self.GiveInstrumentBeam = self.BeamMachine.GiveInstrumentBeam
        elif GD["Beam"]["Model"] == "ATCA":
            self.BeamMachine = ClassATCABeam.ClassATCABeam(self.MS,GD["Beam"])
            self.GiveInstrumentBeam = self.BeamMachine.GiveInstrumentBeam            
        else:
            raise ValueError("Unknown keyword for Beam-Model. Only accepts 'FITS', 'LOFAR', 'GMRT' or 'ATCA'")

    def GiveBeam(self, times, quiet=False,RaDec=None):
        GD = self.GD
        if (GD["Beam"]["Model"] is None) | (GD["Beam"]["Model"] == ""):
            print("  Not applying any beam", file=log)
            return

        self.InitBeamMachine()

        if self.BeamTimes_kMS.size != 0:
            print("  Taking beam-times from DDE-solutions", file=log)
            beam_times = self.BeamTimes_kMS
        else:
            beam_times = self.BeamMachine.getBeamSampleTimes(times, quiet=quiet)

        if RaDec is None:
            RAs = self.ClusterCatBeam.ra
            DECs = self.ClusterCatBeam.dec
        else:
            RAs,DECs=RaDec
            
        # from killMS2.Other.rad2hmsdms import rad2hmsdms
        # for i in range(RAs.size):
        #     ra,dec=RAs[i],DECs[i]
        #     print rad2hmsdms(ra,Type="ra").replace(" ",":"),rad2hmsdms(dec,Type="dec").replace(" ",".")

        DicoBeam = self.EstimateBeam(beam_times, RAs, DECs)

        return DicoBeam

    def GiveVisToJonesChanMapping(self, FreqDomains):
        NChanJones = FreqDomains.shape[0]
        MeanFreqJonesChan = (FreqDomains[:, 0]+FreqDomains[:, 1])/2.
        #print NChanJones,MeanFreqJonesChan 
        DFreq = np.abs(self.MS.ChanFreq.reshape(
            (self.MS.NSPWChan, 1))-MeanFreqJonesChan.reshape((1, NChanJones)))
        #print np.argmin(DFreq, axis=1)

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
            print("VisToJonesChanMapping: %s"%DDFacet.Other.PrintList.ListToStr(DicoBeam["VisToJonesChanMapping"]), file=log)


        DicoBeam["Jones"]=np.zeros((Tm.size,NDir,self.MS.na,FreqDomains.shape[0],2,2),dtype=np.complex64)
        DicoBeam["t0"]=np.zeros((Tm.size,),np.float64)
        DicoBeam["t1"]=np.zeros((Tm.size,),np.float64)
        DicoBeam["tm"]=np.zeros((Tm.size,),np.float64)
        DicoBeam["FreqDomains"]=FreqDomains
        
        
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
                BeamN= ModLinAlg.BatchDot(Beam0inv, Beam)
                Beam=BeamN

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
        import DDFacet.Other.ClassJonesDomains
        DomainMachine=DDFacet.Other.ClassJonesDomains.ClassJonesDomains()
        JonesSols=DomainMachine.MergeJones(DicoJ0, DicoJ1)
        print("There are %i channels in the merged Jones array"%JonesSols["FreqDomains"].shape[0], file=log)
        return JonesSols
    
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
        _, nd1, na1, nch1, _, _ = DicoJ1["Jones"].shape
        nt = DicoOut["tm"].size
        nchout=np.max([nch,nch1])
        
        DicoOut["Jones"] = np.zeros((nt, nd, na, nchout, 2, 2), np.complex64)

        nt0 = DicoJ0["t0"].size
        nt1 = DicoJ1["t0"].size

        iG0 = np.argmin(np.abs(DicoOut["tm"].reshape(
            (nt, 1))-DicoJ0["tm"].reshape((1, nt0))), axis=1)
        iG1 = np.argmin(np.abs(DicoOut["tm"].reshape(
            (nt, 1))-DicoJ1["tm"].reshape((1, nt1))), axis=1)

        

        for itime in range(nt):
            G0 = DicoJ0["Jones"][iG0[itime]]
            G1 = DicoJ1["Jones"][iG1[itime]]
            DicoOut["Jones"][itime] = ModLinAlg.BatchDot(G0, G1)

        return DicoOut
