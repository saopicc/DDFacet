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

import DDFacet.cbuild.Gridder._pyGridderSmearPols as _pyGridderSmear
# import DDFacet.cbuild.Gridder._pyGridderSmearPolsFaster as _pyGridderSmearFaster
##########################################################"
# Please do not remove this import again - Needed for killMS
import DDFacet.cbuild.Gridder._pyGridder as _pyGridder
##########################################################"

import numpy as np
import os
import ModCF
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Parset import ReadCFG
from DDFacet.Other import ClassTimeIt
from DDFacet.Data import ClassVisServer
from DDFacet.Other import MyLogger
log = MyLogger.getLogger("ClassDDEGridMachine")


def testGrid():
    import pylab
    # Parset=ReadCFG.Parset("%s/Parset/DefaultParset.cfg"%os.environ["DDFACET_DIR"])
    Parset = ReadCFG.Parset(
        "%s/DDFacet/Parset/DefaultParset.cfg" %
     os.environ["DDFACET_DIR"])
    DC = Parset.DicoPars
    # 19 (-0.01442078294460315, 0.014406238534169863) 2025 3465 -10.0

    #(array([0]), array([0]), array([1015]), array([1201]))
    #(array([0]), array([0]), array([1050]), array([1398]))
    # 17 (-0.014391694123736577, 0.01437714971330329) 2025 3465 -10.0
    #(array([0]), array([0]), array([1030]), array([1303]))

    npix = 2025
    Cell = 1.5
    # Cell=.5
    offy, offx = 3465/2-1030, 3465/2-1303
    offx = offx
    offy = -offy
    CellRad = (Cell/3600.)*np.pi/180
    L = offy*(Cell/3600.)*np.pi/180
    M = -offx*(Cell/3600.)*np.pi/180

    l0, m0 = -0.014391694123736577, 0.01437714971330329
    #l0,m0=-0.009454, 0.
    L += l0
    M += m0

    DC["Image"]["Cell"] = Cell
    DC["Image"]["NPix"] = npix
    # DC["Image"]["Padding"]=1
    DC["Data"]["MS"] = "Simul.MS.W0.tsel"
    #/media/6B5E-87D0/DDFacet/Test/TestDegridOleg/TestOlegVLA.MS_p0

    DC["CF"]["OverS"] = 81
    DC["CF"]["Support"] = 9
    DC["CF"]["Nw"] = 2
    DC["CF"]["wmax"] = 100000.
    DC["Stores"]["DeleteDDFProducts"] = False  # True
    IdSharedMem = "123."
    # DC["Selection"]["UVRangeKm"]=[0.2,2000.e6]
    DC["Comp"]["CompDeGridDecorr"] = 0.0
    DC["Image"]["Robust"] = -1
    DC["Image"]["Weighting"] = "Briggs"
    #DC["Comp"]["CompDeGridMode"] = False
    #DC["Comp"]["CompGridMode"] = False
    #DC["Comp"]["DegridMode"] = True

    VS = ClassVisServer.ClassVisServer(DC["Data"]["MS"],
                                       ColName=DC["Data"]["ColName"],
                                       TVisSizeMin=DC["Data"][
                                           "ChunkHours"] * 60 * 1.1,
                                       # DicoSelectOptions=DicoSelectOptions,
                                       TChunkSize=DC["Data"]["ChunkHours"],
                                       Robust=DC["Image"]["Robust"],
                                       Weighting=DC["Image"][
                                           "Weighting"],
                                       Super=DC["Image"]["SuperUniform"],
                                       DicoSelectOptions=dict(
                                           DC["Selection"]),
                                       NCPU=DC["Parallel"]["NCPU"], GD=DC)

    Padding = DC["Image"]["Padding"]
    #_,npix=EstimateNpix(npix,Padding)
    sh = [1, 1, npix, npix]
    VS.setFOV(sh, sh, sh, CellRad)

    VS.CalcWeights()
    Load = VS.LoadNextVisChunk()
    DATA = VS.VisChunkToShared()

    # DicoConfigGM={"NPix":NpixFacet,
    #               "Cell":Cell,
    #               "ChanFreq":ChanFreq,
    #               "DoPSF":False,
    #               "Support":Support,
    #               "OverS":OverS,
    #               "wmax":wmax,
    #               "Nw":Nw,
    #               "WProj":True,
    #               "DoDDE":self.DoDDE,
    #               "Padding":Padding}
    # GM=ClassDDEGridMachine(Parset.DicoPars,DoDDE=False,WProj=True,lmShift=(0.,0.),JonesDir=3,SpheNorm=True,IdSharedMem="caca")
    # GM=ClassDDEGridMachine(Parset.DicoPars,
    #                        IdSharedMem="caca",
    #                        **DicoConfigGMself.DicoImager[iFacet]["DicoConfigGM"])

    ChanFreq = VS.CurrentMS.ChanFreq.flatten()
    GM = ClassDDEGridMachine(DC,
                             ChanFreq,
                             npix,
                             lmShift=(
                                 l0, m0),  # self.DicoImager[iFacet]["lmShift"],
                             IdSharedMem=IdSharedMem)

    row0 = 0
    row1 = DATA["uvw"].shape[0]  # -1
    uvw = np.float64(DATA["uvw"])  # [row0:row1]
    # uvw[:,2]=0
    times = np.float64(DATA["times"])  # [row0:row1]
    data = np.complex64(DATA["data"])  # [row0:row1]
    # data.fill(1.)
    # data[:,:,0]=1
    # data[:,:,3]=1
    A0 = np.int32(DATA["A0"])  # [row0:row1]
    A1 = np.int32(DATA["A1"])  # [row0:row1]

    DOrig = data.copy()

    # uvw.fill(0)
    
    flag = np.bool8(DATA["flags"])  # [row0:row1,:,:].copy()
    # ind=np.where(np.logical_not((A0==12)&(A1==14)))[0]
    # flag[ind,:,:]=1
    # flag.fill(0)

    # ind=np.where(np.logical_not((A0==0)&(A1==27)))[0]
    # uvw=uvw[ind].copy()
    # data=data[ind].copy()
    # flag[ind,:,:]=1
    # A0=A0[ind].copy()
    # A1=A1[ind].copy()
    # times=times[ind].copy()

    # MapSmear=NpShared.GiveArray("%sMappingSmearing"%("caca"))
    # stop
    # row=19550
    # print A0[row],A1[row],flag[row]
    # stop

    # DicoJonesMatrices={}
    # DicoClusterDirs=NpShared.SharedToDico("%sDicoClusterDirs"%IdSharedMem)
    # DicoJonesMatrices["DicoClusterDirs"]=DicoClusterDirs

    # DicoJones_Beam=NpShared.SharedToDico("%sJonesFile_Beam"%IdSharedMem)
    # DicoJonesMatrices["DicoJones_Beam"]=DicoJones_Beam
    # DicoJonesMatrices["DicoJones_Beam"]["MapJones"]=NpShared.GiveArray("%sMapJones_Beam"%IdSharedMem)

    DicoJonesMatrices = None

    T = ClassTimeIt.ClassTimeIt("main")

    # print "Start"
    # Grid=GM.put(times,uvw,data,flag,(A0,A1),W=DATA["Weights"],PointingID=0,DoNormWeights=True, DicoJonesMatrices=DicoJonesMatrices)
    # print "OK"
    # pylab.clf()
    # ax=pylab.subplot(1,3,1)
    # pylab.imshow(np.real(Grid[0,0]),cmap="gray",interpolation="nearest")#,vmin=-600,vmax=600)
    # G0=(Grid/np.max(Grid)).copy()

    # pylab.imshow(np.random.rand(50,50))

    # ####

    # GM=ClassDDEGridMachine(DC,
    #                        ChanFreq,
    #                        npix,
    #                        lmShift=(0.,0.),#self.DicoImager[iFacet]["lmShift"],
    #                        IdSharedMem=IdSharedMem)
    # data.fill(1.)
    # Grid=GM.put(times,uvw,data,flag,(A0,A1),W=DATA["Weights"],PointingID=0,DoNormWeights=True, DicoJonesMatrices=DicoJonesMatrices)
    # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
    # pylab.imshow(np.real(Grid[0,0]),cmap="gray",interpolation="nearest")#,vmin=-600,vmax=600)
    # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
    # pylab.imshow(np.real(Grid[0,0])-np.real(G0[0,0]),cmap="gray",interpolation="nearest")#,vmin=-600,vmax=600)
    # pylab.colorbar()
    # pylab.draw()
    # pylab.show(False)

    # return

    Grid = np.zeros(sh, np.complex64)
    T.timeit("grid")
    # Grid[np.isnan(Grid)]=-1

    # Grid[0,0,100,100]=10.

    # Grid.fill(0)
    _, _, n, n = Grid.shape
    Grid[:, :, n/2+offx, n/2+offy] = 10.

    data.fill(0)

    #GM.GD["Comp"]["CompDeGridMode"] = True
    data = GM.get(times, uvw, data, flag, (A0, A1), Grid, freqs=ChanFreq)# , DicoJonesMatrices=DicoJonesMatrices)
    data0 = -data.copy()

    # data.fill(0)
    # GM.GD["Comp"]["CompDeGridMode"] = False
    # data1=-GM.get(times,uvw,data,flag,(A0,A1),Grid,freqs=ChanFreq)#,
    # DicoJonesMatrices=DicoJonesMatrices)

    # ind=np.where(((A0==12)&(A1==14)))[0]
    # data0=data0[ind]
    # data1=data1[ind]
    # print data0-data1

    op0 = np.abs
    op1 = np.imag

    # op0=np.abs
    # op1=np.angle

    nbl = VS.CurrentMS.nbl

    U, V, W = uvw.T
    C = 299792458.
    N = np.sqrt(1.-L**2-M**2)
    U = U.reshape(U.size, 1)
    V = V.reshape(U.size, 1)
    W = W.reshape(U.size, 1)
    #L,M=-0.0194966364621, 0.0112573688
    ChanFreq = ChanFreq.reshape(1, ChanFreq.size)
    K = 10.*np.exp(2.*np.pi*1j*(ChanFreq[0]/C)*(U*L+V*M+W*(N-1)))
    # ind=np.where((d0-d1)[:]!=0)

    #print -0.0194966364621, 0.0112573688
    #-0.0194967821858 0.0112573736754
    # print L,M

    ind = range(U.size)  # np.where((A0==49)&(A1==55))[0]
    d0 = data0[ind, -1, 0].ravel()
    # d1=data1[ind,-1,0].ravel()
    k = K[ind, -1]
    # k=DOrig[ind,-1,0].ravel()

    # d0=data0[:,:,0].ravel()
    # d1=data1[:,:,0].ravel()
    # k=K[:,:]

    X0 = d0.ravel()
    # X1=d1.ravel()
    Y = k.ravel()

    pylab.clf()
    pylab.subplot(2, 1, 1)
    # pylab.plot(op0(d0))
    pylab.plot(op0(X0))
    # pylab.plot(op0(X1))
    pylab.plot(op0(Y))
    pylab.plot(op0(X0)-op0(Y))
    pylab.subplot(2, 1, 2)
    pylab.plot(op1(X0))
    # pylab.plot(op1(X1))
    pylab.plot(op1(Y))
    pylab.plot(op1(X0)-op1(Y))
    pylab.draw()
    pylab.show()


class ClassDDEGridMachine():

    def __init__(self,
                 GD,
                 ChanFreq,
                 Npix,
                 lmShift=(0., 0.),
                 IDFacet=0,
                 SpheNorm=True,
                 NFreqBands=1,
                 DataCorrelationFormat=[5, 6, 7, 8],
                 ExpectedOutputStokes=[1],
                 ListSemaphores=None,
                 cf_dict=None, compute_cf=False,
                 wmax=None,   # must be supplied if compute_cf=True
                 bda_grid=None, bda_degrid=None,
                 ):
        """

        Args:
            GD:
            ChanFreq:
            Npix:
            lmShift:
            IdSharedMem:
            IdSharedMemData:
            FacetDataCache:
            ChunkDataCache:
            IDFacet:
            SpheNorm:
            NFreqBands:
            DataCorrelationFormat:
            ExpectedOutputStokes:
            ListSemaphores:
            cf_dict: SharedDict from/to which WTerms and Sphes are saved
            compute_cf: if True, wterm/sphe is recomputed and saved to store_dict
        """
        T = ClassTimeIt.ClassTimeIt("Init_ClassDDEGridMachine")
        T.disable()
        self.GD = GD
        self.IDFacet = IDFacet
        self.SpheNorm = SpheNorm
        self.ListSemaphores = ListSemaphores
        self._bda_grid = bda_grid
        self._bda_degrid = bda_degrid

        # self.DoPSF=DoPSF
        self.DoPSF = False
        # if DoPSF:
        #     self.DoPSF=True
        #     Npix=Npix*2

        Precision = GD["RIME"]["Precision"]
        PolMode = ExpectedOutputStokes

        if Precision == "S":
            self.dtype = np.complex64
        elif Precision == "D":
            self.dtype = np.complex128

        self.dtype = np.complex64
        T.timeit("0")
        Padding = GD["Facets"]["Padding"]
        self.NonPaddedNpix, Npix = EstimateNpix(Npix, Padding)
        self.Padding = Npix/float(self.NonPaddedNpix)
        # self.Padding=Padding

        self.LSmear = []
        self.PolMode = PolMode
        # SkyType & JonesType
        # 0: scalar
        # 1: diag
        # 2: full
        #if PolMode == "I":
        #    self.npol = 1
        #    self.PolMap = np.array([0, 5, 5, 0], np.int32)
        #    self.SkyType = 1
        #    self.PolModeID = 0
        #elif PolMode == "IQUV":
        #    self.SkyType = 2
        #    self.npol = 4
        #    self.PolMap = np.array([0, 1, 2, 3], np.int32)
        #    self.PolModeID = 1
        #DEPRICATION:
        #These are only to be used in the degridder, they are depricated for the gridder
	self.npol = len(ExpectedOutputStokes)
	self.SkyType=1
	self.PolMap=np.array([0, 5, 5, 0], np.int32)
	self.PolModeID=0
        self.Npix = Npix

        self.NFreqBands = NFreqBands
        self.NonPaddedShape = (
            self.NFreqBands,
            self.npol,
            self.NonPaddedNpix,
            self.NonPaddedNpix)

        self.GridShape = (self.NFreqBands, self.npol, self.Npix, self.Npix)

        x0 = (self.Npix-self.NonPaddedNpix)/2  # +1
        self.PaddingInnerCoord = (x0, x0+self.NonPaddedNpix)

        T.timeit("1")

        OverS = GD["CF"]["OverS"]
        Support = GD["CF"]["Support"]
        Nw = GD["CF"]["Nw"]
        Cell = GD["Image"]["Cell"]

        # T=ClassTimeIt.ClassTimeIt("ClassImager")
        # T.disable()

        self.Cell = Cell
        self.incr = (
            np.array([-Cell, Cell], dtype=np.float64)/3600.)*(np.pi/180)
        # CF.fill(1.)
        # print self.ChanEquidistant
        # self.FullScalarMode=int(GD["DDESolutions"]["FullScalarMode"])
        # self.FullScalarMode=0

        JonesMode = GD["DDESolutions"]["JonesMode"]
        if JonesMode == "Scalar":
            self.JonesType = 0
        elif JonesMode == "Diag":
            self.JonesType = 1
        elif JonesMode == "Full":
            self.JonesType = 2

        T.timeit("3")

        self.ChanFreq = ChanFreq
        self.Sup = Support
        self.WProj = True
        self.Nw = Nw
        self.OverS = OverS
        self.lmShift = lmShift

        T.timeit("4")
        # if neither is set, then machine is being constructed for ffts only
        if cf_dict or compute_cf:
            self.InitCF(cf_dict, compute_cf, wmax)
        T.timeit("5")

        self.reinitGrid()
        self.CasaImage = None
        self.DicoATerm = None
        T.timeit("6")
        self.DataCorrelationFormat = DataCorrelationFormat
        self.ExpectedOutputStokes = ExpectedOutputStokes
        self._fftw_machine = None

    # make sure an FFTW machine is initialized only once per process, and only as needed
    _global_fftw_machines = {}

    @staticmethod
    def _getGlobalFFTWMachine (FFTMachineType, GridShape, dtype):
        """Returns an FFTWMachine matching the arguments.
        Makes sure an FFTW machine is initialized only once per process, and only as needed.
        """
        machine = ClassDDEGridMachine._global_fftw_machines.get((GridShape, dtype))
        if machine is None:
            if FFTMachineType=="FFTW":
                # use single-core FFT because we parallelize by facet instead
                ClassDDEGridMachine._global_fftw_machines[(GridShape, dtype)] = machine = ModFFTW.FFTW_2Donly(GridShape, dtype,ncores=1)
            elif FFTMachineType=="LAPACK":
                ClassDDEGridMachine._global_fftw_machines[(GridShape, dtype)] = machine = ModFFTW.FFTW_2Donly_np(GridShape, dtype)

        return machine

    def getFFTWMachine(self):
        """Returns an fftw_machine for the grid. Makes sure it is initialized once per process."""
        if self._fftw_machine is None:
            self._fftw_machine = self._getGlobalFFTWMachine(self.GD["RIME"]["FFTMachine"], self.GridShape, self.dtype)
        return self._fftw_machine

    @staticmethod
    def verifyCFDict(cf_dict, nw):
        """Checks that cf_dict has all the correct entries"""
        for key in "SW", "Sphe", "W", "CuCv", "wmax":
            if key not in cf_dict:
                raise KeyError(key)

    def InitCF(self, cf_dict, compute_cf, wmax):
        T = ClassTimeIt.ClassTimeIt("InitCF_ClassDDEGridMachine")
        T.disable()
        self.WTerm = ModCF.ClassWTermModified(Cell=self.Cell,
                                              Sup=self.Sup,
                                              Npix=self.Npix,
                                              Freqs=self.ChanFreq,
                                              wmax=wmax,
                                              Nw=self.Nw,
                                              OverS=self.OverS,
                                              lmShift=self.lmShift,
                                              cf_dict=cf_dict,
                                              compute_cf=compute_cf,
                                              IDFacet=self.IDFacet)
        T.timeit("2")
        self.ifzfCF = self.WTerm.ifzfCF

    def setSols(self, times, xi):
        self.Sols = {"times": times, "xi": xi}

    def ShiftVis(self, uvw, vis, reverse=False):
        l0, m0 = self.lmShift
        u, v, w = uvw.T
        U = u.reshape((u.size, 1))
        V = v.reshape((v.size, 1))
        W = w.reshape((w.size, 1))
        n0 = np.sqrt(1-l0**2-m0**2)-1
        if reverse:
            corr = np.exp(-self.UVNorm*(U*l0+V*m0+W*n0))
        else:
            corr = np.exp(self.UVNorm*(U*l0+V*m0+W*n0))
        
        U += W*self.WTerm.Cu
        V += W*self.WTerm.Cv

        corr = corr.reshape((U.size, self.UVNorm.size, 1))
        vis *= corr

        U = U.reshape((U.size,))
        V = V.reshape((V.size,))
        W = W.reshape((W.size,))
        uvw = np.array((U, V, W)).T.copy()

        return uvw, vis

    def reinitGrid(self):
        # self.Grid.fill(0)
        self.NChan, self.npol, _, _ = self.GridShape
        self.SumWeigths = np.zeros((self.NChan, self.npol), np.float64)
        self.SumJones = np.zeros((2, self.NChan), np.float64)

    def setDecorr(self,uvw_dt,DT,Dnu,SmearMode="FT",
                  lm_min=None,
                  lm_PhaseCenter=None):
        DoSmearFreq=0
        if "F" in SmearMode:
            DoSmearFreq = 1
        DoSmearTime = 0
        if "T" in SmearMode:
            DoSmearTime=1
            
        lmin,mmin=self.lmShift
        if lm_min is not None:
            lmin,mmin=lm_min
        if lm_PhaseCenter is not None:
            lmin-=lm_PhaseCenter[0]
            mmin-=lm_PhaseCenter[1]

        if not(uvw_dt.dtype == np.float64):
            raise NameError(
                'uvw_dt.dtype %s %s' %
                (str(
                    uvw_dt.dtype), str(
                    np.float64)))

        self.LSmear=[uvw_dt,DT,Dnu,DoSmearTime,DoSmearFreq,lmin,mmin]

    def GiveParamJonesList(self, DicoJonesMatrices, times, A0, A1, uvw):

        Apply_killMS = ("DicoJones_killMS" in DicoJonesMatrices)
        Apply_Beam = ("DicoJones_Beam" in DicoJonesMatrices)


        l0,m0=self.lmShift
        idir_kMS=0
        w_kMS=np.array([],np.float32)
        InterpMode=self.GD["DDESolutions"]["Type"]
        d0=self.GD["DDESolutions"]["Scale"]*np.pi/180
        gamma=self.GD["DDESolutions"]["gamma"]
        if Apply_killMS:
            DicoClusterDirs=DicoJonesMatrices["DicoJones_killMS"]["Dirs"]
            lc=DicoClusterDirs["l"]
            mc=DicoClusterDirs["m"]
            sI=DicoClusterDirs["I"]
            d=np.sqrt((l0-lc)**2+(m0-mc)**2)
            idir_kMS=np.argmin(d)
            # print rad2hmsdms(DicoClusterDirs["ra"][idir_kMS],Type="ra"), rad2hmsdms(DicoClusterDirs["dec"][idir_kMS])
            
            w=sI/(1.+d/d0)**gamma
            w/=np.sum(w)
            w[w<(0.2*w.max())]=0
            ind=np.argsort(w)[::-1]
            w[ind[3::]]=0
            w/=np.sum(w)
            w_kMS=w

        idir_Beam = 0
        if Apply_Beam:
            DicoClusterDirs = DicoJonesMatrices["DicoJones_Beam"]["Dirs"]
            lc = DicoClusterDirs["l"]
            mc = DicoClusterDirs["m"]
            d = np.sqrt((l0-lc)**2+(m0-mc)**2)
            idir_Beam = np.argmin(d)

        # pylab.clf()
        # pylab.scatter(lc,mc,c=w)
        # pylab.scatter([l0],[m0],marker="+")
        # pylab.draw()
        # pylab.show(False)
        
        if InterpMode=="Nearest":
            InterpMode=0
        elif InterpMode=="Krigging":
            InterpMode=1
                
        #ParamJonesList=[MapJones,A0.astype(np.int32),A1.astype(np.int32),JonesMatrices.astype(np.complex64),idir]
        if A0.size!=uvw.shape[0]:
            raise RuntimeError("Antenna array is expected to have the same number of rows as the uvw array")
        

        if InterpMode == "Nearest":
            InterpMode = 0
        elif InterpMode == "Krigging":
            InterpMode = 1

        # ParamJonesList=[MapJones,A0.astype(np.int32),A1.astype(np.int32),JonesMatrices.astype(np.complex64),idir]
        if A0.size != uvw.shape[0]:
            raise RuntimeError(
                "Antenna array is expected to have the same number of rows as the uvw array")

        JonesMatrices_Beam = np.array([], np.complex64).reshape((0, 0, 0, 0))
        MapJones_Beam = np.array([], np.int32).reshape((0,))
        VisToJonesChanMapping_Beam = np.array([], np.int32).reshape((0,))

        JonesMatrices_killMS = np.array([], np.complex64).reshape((0, 0, 0, 0))
        AlphaReg_killMS = np.array([], np.float32).reshape((0, 0))
        MapJones_killMS = np.array([], np.int32).reshape((0,))
        VisToJonesChanMapping_killMS = np.array([], np.int32).reshape((0,))

        if Apply_Beam:
            JonesMatrices_Beam = DicoJonesMatrices["DicoJones_Beam"]["Jones"]["Jones"]
            MapJones_Beam = DicoJonesMatrices["DicoJones_Beam"]["TimeMapping"]
            VisToJonesChanMapping_Beam = np.int32(DicoJonesMatrices["DicoJones_Beam"]["Jones"]["VisToJonesChanMapping"])
            self.CheckTypes(A0=A0, A1=A1, Jones=JonesMatrices_Beam)

        if Apply_killMS:
            JonesMatrices_killMS = DicoJonesMatrices["DicoJones_killMS"]["Jones"]["Jones"]
            MapJones_killMS = DicoJonesMatrices["DicoJones_killMS"]["TimeMapping"]
            AlphaReg = DicoJonesMatrices["DicoJones_killMS"].get("AlphaReg")
            if AlphaReg is not None:
                AlphaReg_killMS=AlphaReg
            VisToJonesChanMapping_killMS = np.int32(DicoJonesMatrices["DicoJones_killMS"]["Jones"]["VisToJonesChanMapping"])
            self.CheckTypes(A0=A0, A1=A1, Jones=JonesMatrices_killMS)

        # print JonesMatrices_Beam.shape,VisToJonesChanMapping_Beam

        ParamJonesList = [JonesMatrices_killMS,
                          MapJones_killMS,
                          JonesMatrices_Beam,
                          MapJones_Beam,
                          A0,
                          A1,
                          np.array([idir_kMS], np.int32),
                          np.float32(w_kMS),
                          np.array([idir_Beam], np.int32),
                          np.array([InterpMode], np.int32),
                          VisToJonesChanMapping_killMS,
                          VisToJonesChanMapping_Beam,
                          AlphaReg_killMS]

        return ParamJonesList

    def put(self, times, uvw, visIn, flag, A0A1, W=None,
            PointingID=0, DoNormWeights=True, DicoJonesMatrices=None,
            freqs=None, DoPSF=0, ChanMapping=None, ResidueGrid=None, sparsification=None):
        """
        Gridding routine, wraps external python extension C gridder
        Args:
            times:
            uvw:
            visIn:
            flag:
            A0A1:
            W:
            PointingID:
            DoNormWeights:
            DicoJonesMatrices:
            freqs:
            DoPSF:
            ChanMapping:
            ResidueGrid:
        Returns:

        """
        vis = visIn

        T = ClassTimeIt.ClassTimeIt("put")
        T.disable()
        self.DoNormWeights = DoNormWeights
        if not(self.DoNormWeights):
            self.reinitGrid()

        if freqs.size > 1:
            df = freqs[1::] - freqs[0:-1]
            ddf = np.abs(df - np.mean(df))
            ChanEquidistant = int(np.max(ddf) < 1.)
        else:
            ChanEquidistant = 0

        if ChanMapping is None:
            ChanMapping = np.zeros((visIn.shape[1],), np.int64)
        self.ChanMappingGrid = ChanMapping

        Grid = ResidueGrid

        if Grid.dtype != self.dtype:
            raise TypeError("Grid must be of type "+str(self.dtype))
        A0, A1 = A0A1

        npol = self.npol
        NChan = self.NChan

        NVisChan = vis.shape[1]
        self.SumJonesChan = np.zeros((2, NVisChan), np.float64)

        if isinstance(W, type(None)):
            W = np.ones((uvw.shape[0], NVisChan), dtype=np.float64)

        SumWeigths = self.SumWeigths
        if vis.shape != flag.shape:
            raise Exception(
                'vis[%s] and flag[%s] should have the same shape' %
                (str(
                    vis.shape), str(
                    flag.shape)))

        u, v, w = uvw.T

        l0, m0 = self.lmShift
        FacetInfos = np.float64(
            np.array([self.WTerm.Cu, self.WTerm.Cv, l0, m0]))

        self.CheckTypes(
            Grid=Grid,
            vis=vis,
            uvw=uvw,
            flag=flag,
            ListWTerm=self.WTerm.Wplanes,
            W=W)
        ParamJonesList = []
        if DicoJonesMatrices is not None:
            ApplyAmp = 0
            ApplyPhase = 0
            ScaleAmplitude = 0
            CalibError = 0.

            if "A" in self.GD["DDESolutions"]["DDModeGrid"]:
                ApplyAmp = 1
            if "P" in self.GD["DDESolutions"]["DDModeGrid"]:
                ApplyPhase = 1
            if self.GD["DDESolutions"]["ScaleAmpGrid"]:
                ScaleAmplitude=1
                CalibError=(self.GD["DDESolutions"]["CalibErr"]/3600.)*np.pi/180
            LApplySol=[ApplyAmp,ApplyPhase,ScaleAmplitude,CalibError]
            LSumJones=[self.SumJones]
            LSumJonesChan=[self.SumJonesChan]
            ParamJonesList=self.GiveParamJonesList(DicoJonesMatrices,times,A0,A1,uvw)
            ParamJonesList=ParamJonesList+LApplySol+LSumJones+LSumJonesChan+[np.float32(self.GD["DDESolutions"]["ReWeightSNR"])]

        #T2= ClassTimeIt.ClassTimeIt("Gridder")
        #T2.disable()
        T.timeit("stuff")
        if False: # # self.GD["Comp"]["GridMode"] == 0:  # really deprecated for now
            raise RuntimeError("Deprecated flag. Please use BDA gridder")
        else:
            OptimisationInfos = [
                self.JonesType,
                ChanEquidistant,
                self.SkyType,
                self.PolModeID]
            _pyGridderSmear.pyGridderWPol(Grid,
                                          vis,
                                          uvw,
                                          flag,
                                          W,
                                          SumWeigths,
                                          DoPSF,
                                          self.WTerm.Wplanes,
                                          self.WTerm.WplanesConj,
                                          np.array([self.WTerm.RefWave,
                                                    self.WTerm.wmax,
                                                    len(self.WTerm.Wplanes),
                                                    self.WTerm.OverS],
                                                   dtype=np.float64),
                                          self.incr.astype(np.float64),
                                          freqs,
                                          [self.PolMap,
                                              FacetInfos],
                                          ParamJonesList,
                                          self._bda_grid,
                                          sparsification if sparsification is not None else np.array([]),
                                          OptimisationInfos,
                                          self.LSmear,
                                          np.int32(ChanMapping),
                                          np.array(self.DataCorrelationFormat).astype(np.uint16),
                                          np.array(self.ExpectedOutputStokes).astype(np.uint16))

            T.timeit("gridder")
            T.timeit("grid %d" % self.IDFacet)

    def CheckTypes(
        self,
        Grid=None,
        vis=None,
        uvw=None,
        flag=None,
        ListWTerm=None,
        W=None,
        A0=None,
        A1=None,
        Jones=None):
        if not isinstance(Grid, type(None)):
            if not(Grid.dtype == np.complex64):
                raise NameError(
                    'Grid.dtype %s %s' %
                    (str(
                        Grid.dtype), str(
                        self.dtype)))
            if not(Grid.flags.c_contiguous):
                raise NameError("Grid has to be contiguous")
        if not isinstance(vis, type(None)):
            if not(vis.dtype == np.complex64):
                raise NameError('vis.dtype %s' % (str(vis.dtype)))
            if not(vis.flags.c_contiguous):
                raise NameError("vis has to be contiguous")
        if not isinstance(uvw, type(None)):
            if not(uvw.dtype == np.float64):
                raise NameError('uvw.dtype %s' % (str(uvw.dtype)))
            if not(uvw.flags.c_contiguous):
                raise NameError("uvw has to be contiguous")
        if not isinstance(flag, type(None)):
            if not(flag.dtype == np.bool8):
                raise NameError('flag.dtype %s' % (str(flag.dtype)))
            if not(flag.flags.c_contiguous):
                raise NameError("flag to be contiguous")
        if ListWTerm is not None:
            if not(ListWTerm[0].dtype == np.complex64):
                raise NameError('ListWTerm.dtype %s' % (str(ListWTerm.dtype)))
        if not isinstance(W, type(None)):
            if not(W.dtype == np.float32):
                raise NameError('W.dtype %s' % (str(W.dtype)))
            if not(W.flags.c_contiguous):
                raise NameError("W has to be contiguous")
        if not isinstance(A0, type(None)):
            if not(A0.dtype == np.int32):
                raise NameError('A0.dtype %s' % (str(A0.dtype)))
            if not(A0.flags.c_contiguous):
                raise NameError("A0 has to be contiguous")
        if not isinstance(A1, type(None)):
            if not(A1.dtype == np.int32):
                raise NameError('A1.dtype %s' % (str(A1.dtype)))
            if not(A1.flags.c_contiguous):
                raise NameError("A1 has to be contiguous")
        if not isinstance(Jones, type(None)):
            if not(Jones.dtype == np.complex64):
                raise NameError('Jones.dtype %s' % (str(Jones.dtype)))
            if not(Jones.flags.c_contiguous):
                raise NameError("Jones has to be contiguous")

    def get(self, 
            times, 
            uvw, 
            visIn, 
            flag, 
            A0A1, 
            ModelImage, 
            PointingID=0,
            Row0Row1=(0, -1),
            DicoJonesMatrices=None, freqs=None, ImToGrid=True,
            TranformModelInput="", ChanMapping=None, sparsification=None):
        T = ClassTimeIt.ClassTimeIt("get")
        T.disable()
        vis = visIn.view()
        A0, A1 = A0A1

        T.timeit("0")

        if ImToGrid:
            if np.max(np.abs(ModelImage)) == 0:
                return vis
            Grid = self.dtype(self.setModelIm(ModelImage))
        else:
            Grid = ModelImage

        if ChanMapping is None:
            ChanMapping = np.zeros((visIn.shape[1],), np.int32)


        if TranformModelInput == "FT":
            if np.max(np.abs(ModelImage)) == 0:
                return vis
            if self.GD["RIME"]["Precision"]=="S": 
                Cast=np.complex64
            elif self.GD["RIME"]["Precision"]=="D": 
                Cast=np.complex128
            Grid = np.complex64(self.getFFTWMachine().fft(Cast(ModelImage)))

        if freqs.size > 1:
            df = freqs[1::] - freqs[0:-1]
            ddf = np.abs(df - np.mean(df))
            ChanEquidistant = int(np.max(ddf) < 1.)
        else:
            ChanEquidistant = 0

        # np.save("Grid",Grid)
        NVisChan = visIn.shape[1]
        self.ChanMappingDegrid = np.int32(ChanMapping)
        self.SumJonesChan = np.zeros((2, NVisChan), np.float64)

        T.timeit("1")

        npol = self.npol
        NChan = self.NChan
        SumWeigths = self.SumWeigths
        if vis.shape != flag.shape:
            raise Exception(
                'vis[%s] and flag[%s] should have the same shape' %
                (str(
                    vis.shape), str(
                    flag.shape)))

        l0, m0 = self.lmShift
        FacetInfos = np.float64(
            np.array([self.WTerm.Cu, self.WTerm.Cv, l0, m0]))
        Row0, Row1 = Row0Row1
        if Row1 == -1:
            Row1 = uvw.shape[0]
        RowInfos = np.array([Row0, Row1]).astype(np.int32)

        T.timeit("2")

        self.CheckTypes(
            Grid=Grid,
            vis=vis,
            uvw=uvw,
            flag=flag,
            ListWTerm=self.WTerm.Wplanes)

        ParamJonesList = []

        if DicoJonesMatrices is not None:
            ApplyAmp = 0
            ApplyPhase = 0
            ScaleAmplitude = 0
            CalibError = 0.

            if "A" in self.GD["DDESolutions"]["DDModeDeGrid"]:
                ApplyAmp = 1
            if "P" in self.GD["DDESolutions"]["DDModeDeGrid"]:
                ApplyPhase = 1
            if self.GD["DDESolutions"]["ScaleAmpDeGrid"]:
                ScaleAmplitude = 1
                CalibError = (self.GD["DDESolutions"][
                              "CalibErr"]/3600.)*np.pi/180

            LApplySol = [ApplyAmp, ApplyPhase, ScaleAmplitude, CalibError]
            LSumJones = [self.SumJones]
            LSumJonesChan = [self.SumJonesChan]
            ParamJonesList = self.GiveParamJonesList(
                DicoJonesMatrices, times, A0, A1, uvw)
            ParamJonesList = ParamJonesList+LApplySol+LSumJones+LSumJonesChan + \
                [np.float32(self.GD["DDESolutions"]["ReWeightSNR"])]

        T.timeit("3")
        #print vis
        #print "DEGRID:",Grid.shape,ChanMapping
        if self.GD["RIME"]["ForwardMode"]=="Classic":
            _ = _pyGridder.pyDeGridderWPol(Grid,
                                           vis,
                                           uvw,
                                           flag,
                                           SumWeigths,
                                           0,
                                           self.WTerm.WplanesConj,
                                           self.WTerm.Wplanes,
                                           np.array([self.WTerm.RefWave,self.WTerm.wmax,len(self.WTerm.Wplanes),self.WTerm.OverS],dtype=np.float64),
                                           self.incr.astype(np.float64),
                                           freqs,
                                           [self.PolMap,FacetInfos,RowInfos,ChanMapping],
                                           ParamJonesList,
                                           self.LSmear)
        elif self.GD["RIME"]["ForwardMode"]=="BDA-degrid":
            # OptimisationInfos=[self.FullScalarMode,self.ChanEquidistant]
            OptimisationInfos = [
                self.JonesType,
                ChanEquidistant,
                self.SkyType,
                self.PolModeID]
#            MapSmear = NpShared.GiveArray(
#                "%sBDA.DeGrid" %
#               (self.ChunkDataCache))
            _pyGridderSmear.pySetSemaphores(self.ListSemaphores)
            vis = _pyGridderSmear.pyDeGridderWPol(
                Grid, 
                vis, 
                uvw, 
                flag, 
                SumWeigths, 
                0, 
                self.WTerm.WplanesConj,
                self.WTerm.Wplanes, 
                np.array(
                    [self.WTerm.RefWave, 
                     self.WTerm.wmax,
                     len(self.WTerm.Wplanes),
                     self.WTerm.OverS],
                     dtype=np.float64),
                self.incr.astype(np.float64),
                freqs, 
                [self.PolMap, FacetInfos, RowInfos],
                ParamJonesList, 
                self._bda_degrid,
                sparsification if sparsification is not None else np.array([]),
                OptimisationInfos,
                self.LSmear, np.int32(ChanMapping),
                np.array(self.DataCorrelationFormat).astype(np.uint16),
                np.array(self.ExpectedOutputStokes).astype(np.uint16))


        T.timeit("4 (degrid)")
        # print vis

        # uvw,vis=self.ShiftVis(uvwOrig,vis,reverse=False)

        # T.timeit("5")
        return vis

    #########################################################
    # ADDITIONALS
    #########################################################

    def setModelIm(self, ModelIm):
        _, _, n, n = ModelIm.shape
        x0, x1 = self.PaddingInnerCoord
        # self.ModelIm[:,:,x0:x1,x0:x1]=ModelIm
        ModelImPadded = np.zeros(self.GridShape, dtype=self.dtype)
        ModelImPadded[:, :, x0:x1, x0:x1] = ModelIm

        Grid = self.ImToGrid(ModelImPadded)*n**2
        return Grid

    def ImToGrid(self, ModelIm):

        npol = self.npol
        ModelImCorr = ModelIm*(self.WTerm.OverS*self.Padding)**2

        nchan, npol, _, _ = ModelImCorr.shape
        for ichan in xrange(nchan):
            for ipol in xrange(npol):
                ModelImCorr[
                    ichan, ipol][
                    :, :] = ModelImCorr[
                    ichan, ipol][
                    :, :].real / self.ifzfCF.real

        ModelUVCorr = self.FT(ModelImCorr)

        return ModelUVCorr

    def GridToIm(self, Grid):
        Grid *= (self.WTerm.OverS)**2
        Dirty = self.getFFTWMachine().ifft(Grid)

        return Dirty
