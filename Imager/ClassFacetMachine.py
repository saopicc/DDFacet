from DDFacet.Other.progressbar import ProgressBar
import multiprocessing
from multiprocessing import Process
import psutil
import ClassDDEGridMachine
import numpy as np
import pylab
import ClassCasaImage
from DDFacet.ToolsDir import ModCoord
#import time
from DDFacet.Array import NpShared
from DDFacet.ToolsDir import ModFFTW
import pyfftw
from DDFacet.Other import ClassTimeIt
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.Imager.ClassImToGrid import ClassImToGrid
from DDFacet.Other import MyLogger
log = MyLogger.getLogger("ClassFacetImager")
MyLogger.setSilent("MyLogger")
from DDFacet.cbuild.Gridder import _pyGridderSmearPols
#from DDFacet.Other import reformat
#import Polygon
#from SkyModel.Sky import ModVoronoiToReg
from matplotlib.path import Path
#from scipy.spatial import Voronoi
#from SkyModel.Sky import ModVoronoi


# A generic producer that places data items from a list into a queue
def work_producer(queue, data):
    for item in data:
        try:
            queue.put(item, 60)  # possible issue if no space for 60 seconds
        except Queue.Full:
            print>> log, "work_producer: queue full"
            pass


def init_w_worker_tessel(m_work_queue, m_result_queue, GD, Mode, FFTW_Wisdom,
                         DicoImager, IdSharedMem, FacetDataCache, ApplyCal,
                         NFreqBands, ExpectedOutputStokes, CoordMachine,
                         nch, npol, MainRaDec, CellSizeRad, NFacets,
                         IdSharedMemData, ChunkDataCache, SpheNorm,
                         DataCorrelationFormat, ListSemaphores, EngineInstance):

    pill = True
    # While no poisoned pill has been given grab items from the queue.
    while pill:
        iFacet = None
        try:
            # Get queue item, or timeout and check if pill perscribed.
            iFacet = m_work_queue.get(True, 5)
        except Queue.Empty:
            print>> log, "grid_worker: empty worker queue"
            pass
        else:
            if FFTW_Wisdom != None:
                pyfftw.import_wisdom(FFTW_Wisdom)

            if iFacet == "POISON-E":
                pill = False  # The poisoned pill to stop the worker
                break
            elif iFacet is not None:
                # Create smoothned facet tessel mask:
                Npix = DicoImager[iFacet]["NpixFacetPadded"]
                l0, l1, m0, m1 = DicoImager[iFacet]["lmExtentPadded"]
                X, Y = np.mgrid[l0:l1:Npix * 1j, m0:m1:Npix * 1j]
                XY = np.dstack((X, Y))
                XY_flat = XY.reshape((-1, 2))
                vertices = DicoImager[iFacet]["Polygon"]
                mpath = Path(vertices)   # the vertices of the polygon
                mask_flat = mpath.contains_points(XY_flat)
                mask = mask_flat.reshape(X.shape)
                mpath = Path(EngineInstance.CornersImageTot)
                mask_flat2 = mpath.contains_points(XY_flat)
                mask2 = mask_flat2.reshape(X.shape)
                mask[mask2 == 0] = 0

                GaussPars = (10, 10, 0)

                SpacialWeigth = np.float32(mask.reshape((1, 1, Npix, Npix)))
                SpacialWeigth = ModFFTW.ConvolveGaussian(SpacialWeigth,
                                                         CellSizeRad=1,
                                                         GaussPars=[GaussPars])
                SpacialWeigth = SpacialWeigth.reshape((Npix, Npix))
                SpacialWeigth /= np.max(SpacialWeigth)
                NameSpacialWeigth = "%sSpacialWeight.Facet_%3.3i" % (
                                    FacetDataCache, iFacet)
                NpShared.ToShared(NameSpacialWeigth, SpacialWeigth)

                # Initialize a grid machine per facet:
                GridMachine = ClassDDEGridMachine.ClassDDEGridMachine(GD,  # this is not used
                            DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
                            DicoImager[iFacet]["DicoConfigGM"]["Npix"],
                            DicoImager[iFacet]["lmShift"],
                            IdSharedMem,
                            IdSharedMemData,
                            FacetDataCache,
                            ChunkDataCache,
                            iFacet,
                            SpheNorm,
                            NFreqBands,
                            DataCorrelationFormat,
                            ExpectedOutputStokes,
                            ListSemaphores)

                # Send result back
                m_result_queue.put({"Success": True, "iFacet": iFacet})


# Gridding worker that is called by Multiprocessing.Process
def grid_worker(m_work_queue, m_result_queue, GD, Mode, FFTW_Wisdom, DicoImager,
                IdSharedMem, IdSharedMemData, FacetDataCache, ChunkDataCache,
                ApplyCal, SpheNorm, PSFMode, NFreqBands, Weights, PauseOnStart,
                DataCorrelationFormat, ExpectedOutputStokes):

    pill = True
    # While no poisoned pill has been given grab items from the queue.
    while pill:
        try:
            # Get queue item, or timeout and check if pill perscribed.
            iFacet = m_work_queue.get(True, 5)
        except Queue.Empty:
            print>> log, "grid_worker: empty worker queue"
            pass
        else:
            if FFTW_Wisdom != None:
                pyfftw.import_wisdom(FFTW_Wisdom)

            if iFacet == "POISON-E":
                pill = False  # The poisoned pill to stop the worker
                break
            else:
                ListSemaphores = None
                # Create a new GridMachine
                GridMachine = ClassDDEGridMachine.ClassDDEGridMachine(GD,
                            DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
                            DicoImager[iFacet]["DicoConfigGM"]["Npix"],
                            DicoImager[iFacet]["lmShift"],
                            IdSharedMem,
                            IdSharedMemData,
                            FacetDataCache,
                            ChunkDataCache,
                            iFacet,
                            SpheNorm,
                            NFreqBands,
                            DataCorrelationFormat,
                            ExpectedOutputStokes,
                            ListSemaphores)

                DATA = NpShared.SharedToDico("%sDicoData" % IdSharedMemData)
                uvwThis = DATA["uvw"]
                visThis = DATA["data"]
                flagsThis = DATA["flags"]
                times = DATA["times"]
                A0 = DATA["A0"]
                A1 = DATA["A1"]
                A0A1 = A0, A1
                W = NpShared.GiveArray("file://"+Weights)
                freqs = DATA["freqs"]
                ChanMapping = DATA["ChanMapping"]

                DecorrMode = GD["DDESolutions"]["DecorrMode"]
                if ('F' in DecorrMode) | ("T" in DecorrMode):
                    uvw_dt = DATA["uvw_dt"]
                    DT, Dnu = DATA["MSInfos"]
                    GridMachine.setDecorr(uvw_dt, DT, Dnu, SmearMode=DecorrMode)

                GridName = "%sGridFacet.%3.3i" % (IdSharedMem, iFacet)
                Grid = NpShared.GiveArray(GridName)

                # Create Jones Matrices Dictionary
                DicoJonesMatrices = None
                Apply_killMS = (GD["DDESolutions"]["DDSols"] != "") \
                                & (GD["DDESolutions"]["DDSols"] != None)
                Apply_Beam = (GD["Beam"]["BeamModel"] != None)

                if ApplyCal:
                    DicoJonesMatrices = {}

                if Apply_killMS:
                    DicoJones_killMS = NpShared.SharedToDico(
                                        "%sJonesFile_killMS" % IdSharedMemData)
                    DicoJonesMatrices["DicoJones_killMS"] = DicoJones_killMS
                    DicoJonesMatrices["DicoJones_killMS"]["MapJones"] = NpShared.GiveArray(
                                        "%sMapJones_killMS" % IdSharedMemData)
                    DicoClusterDirs_killMS = NpShared.SharedToDico(
                                            "%sDicoClusterDirs_killMS" % IdSharedMemData)
                    DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"] = DicoClusterDirs_killMS
                    DicoJonesMatrices["DicoJones_killMS"]["AlphaReg"] = None

                if Apply_Beam:
                    DicoJones_Beam = NpShared.SharedToDico(
                                    "%sJonesFile_Beam" % IdSharedMemData)
                    DicoJonesMatrices["DicoJones_Beam"] = DicoJones_Beam
                    DicoJonesMatrices["DicoJones_Beam"]["MapJones"] = NpShared.GiveArray(
                                                                        "%sMapJones_Beam" % IdSharedMemData)
                    DicoClusterDirs_Beam = NpShared.SharedToDico(
                                            "%sDicoClusterDirs_Beam" % IdSharedMemData)
                    DicoJonesMatrices["DicoJones_Beam"]["DicoClusterDirs"] = DicoClusterDirs_Beam

                GridMachine.put(times, uvwThis, visThis, flagsThis, A0A1, W,
                                DoNormWeights=False,
                                DicoJonesMatrices=DicoJonesMatrices,
                                freqs=freqs, DoPSF=PSFMode,
                                ChanMapping=ChanMapping,
                                ResidueGrid=Grid)

                Sw = GridMachine.SumWeigths.copy()
                SumJones = GridMachine.SumJones.copy()
                SumJonesChan = GridMachine.SumJonesChan.copy()
                del (GridMachine)

                # Place results into queue
                m_result_queue.put({"Success": True, "iFacet": iFacet,
                                    "Weights": Sw, "SumJones": SumJones,
                                    "SumJonesChan": SumJonesChan})


def FFT_worker(m_work_queue, m_result_queue, GD, Mode, FFTW_Wisdom, DicoImager,
               IdSharedMem, IdSharedMemData, FacetDataCache, ChunkDataCache,
               ApplyCal, SpheNorm, PSFMode, NFreqBands, PauseOnStart,
               DataCorrelationFormat, ExpectedOutputStokes):

    """
    Fourier transforms the grids currently housed in shared memory
    Precondition:
        Should be called after all data has been gridded
    Returns:
        Dictionary of success and facet identifier
    """

    pill = True
    # While no poisoned pill has been given grab items from the queue.
    while pill:
        try:
            # Get queue item, or timeout and check if pill perscribed.
            iFacet = m_work_queue.get(True, 5)
        except Queue.Empty:
            print>> log, "grid_worker: empty worker queue"
            pass
        else:
            if iFacet == "POISON-E":
                pill = False  # The poisoned pill to stop the worker
                break
            else:
                ListSemaphores = None
                GridMachine = ClassDDEGridMachine.ClassDDEGridMachine(GD,
                            DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
                            DicoImager[iFacet]["DicoConfigGM"]["Npix"],
                            DicoImager[iFacet]["lmShift"],
                            IdSharedMem,
                            IdSharedMemData,
                            FacetDataCache,
                            ChunkDataCache,
                            iFacet,
                            SpheNorm,
                            NFreqBands,
                            DataCorrelationFormat,
                            ExpectedOutputStokes,
                            ListSemaphores)

                GridName = "%sGridFacet.%3.3i" % (IdSharedMem, iFacet)
                Grid = NpShared.GiveArray(GridName)
                Dirty = GridMachine.GridToIm(Grid)
                Grid[...] = Dirty[...]

                # place result into queue
                m_result_queue.put({"Success": True, "iFacet": iFacet})


def degrid_worker(m_work_queue, m_result_queue, GD, Mode, FFTW_Wisdom,
                  DicoImager, IdSharedMem, IdSharedMemData, FacetDataCache,
                  ChunkDataCache, ApplyCal, SpheNorm, NFreqBands, PauseOnStart,
                  DataCorrelationFormat, ExpectedOutputStokes, ListSemaphores):

    """
    Degrids input model facets and subtracts model visibilities from residuals.
    Assumes degridding input data is placed in DATA shared memory dictionary.
    Returns:
        Dictionary of success and facet identifier
    """

    pill = True
    # While no poisoned pill has been given grab items from the queue.
    while pill:
        try:
            # Get queue item, or timeout and check if pill perscribed.
            iFacet = m_work_queue.get(True, 5)
        except Queue.Empty:
            print>> log, "degrid_worker: empty worker queue"
            pass
        else:
            if FFTW_Wisdom != None:
                pyfftw.import_wisdom(FFTW_Wisdom)

            if iFacet == "POISON-E":
                pill = False  # The poisoned pill to stop the worker
                break
            else:
                # Create a new GridMachine
                GridMachine = ClassDDEGridMachine.ClassDDEGridMachine(GD,
                            DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
                            DicoImager[iFacet]["DicoConfigGM"]["Npix"],
                            DicoImager[iFacet]["lmShift"],
                            IdSharedMem,
                            IdSharedMemData,
                            FacetDataCache,
                            ChunkDataCache,
                            iFacet,
                            SpheNorm,
                            NFreqBands,
                            DataCorrelationFormat,
                            ExpectedOutputStokes,
                            ListSemaphores)

                DATA = NpShared.SharedToDico("%sDicoData" % IdSharedMemData)
                uvwThis = DATA["uvw"]
                visThis = DATA["data"]
                flagsThis = DATA["flags"]
                times = DATA["times"]
                A0 = DATA["A0"]
                A1 = DATA["A1"]
                A0A1 = A0, A1
                freqs = DATA["freqs"]
                ChanMapping = DATA["ChanMappingDegrid"]

                # Create Jones Matrices Dictionary
                DicoJonesMatrices = None
                Apply_killMS = (GD["DDESolutions"]["DDSols"] != "") \
                                & (GD["DDESolutions"]["DDSols"] != None)
                Apply_Beam = (GD["Beam"]["BeamModel"] != None)

                if ApplyCal:
                    DicoJonesMatrices = {}

                if Apply_killMS:
                    DicoJones_killMS = NpShared.SharedToDico(
                                        "%sJonesFile_killMS" % IdSharedMemData)
                    DicoJonesMatrices["DicoJones_killMS"] = DicoJones_killMS
                    DicoJonesMatrices["DicoJones_killMS"]["MapJones"] = NpShared.GiveArray(
                                                                        "%sMapJones_killMS" % IdSharedMemData)
                    DicoClusterDirs_killMS = NpShared.SharedToDico(
                                            "%sDicoClusterDirs_killMS" % IdSharedMemData)
                    DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"] = DicoClusterDirs_killMS
                    DicoJonesMatrices["DicoJones_killMS"]["AlphaReg"] = None

                if Apply_Beam:
                    DicoJones_Beam = NpShared.SharedToDico(
                                    "%sJonesFile_Beam" % IdSharedMemData)
                    DicoJonesMatrices["DicoJones_Beam"] = DicoJones_Beam
                    DicoJonesMatrices["DicoJones_Beam"]["MapJones"] = NpShared.GiveArray(
                                                                    "%sMapJones_Beam" % IdSharedMemData)
                    DicoClusterDirs_Beam = NpShared.SharedToDico(
                                            "%sDicoClusterDirs_Beam" % IdSharedMemData)
                    DicoJonesMatrices["DicoJones_Beam"]["DicoClusterDirs"] = DicoClusterDirs_Beam

                ModelSharedMemName = "%sModelImage.Facet_%3.3i" % (
                    IdSharedMem, iFacet)
                ModelGrid = NpShared.GiveArray(ModelSharedMemName)

                DecorrMode = GD["DDESolutions"]["DecorrMode"]

                if ('F' in DecorrMode) | ("T" in DecorrMode):
                    uvw_dt = DATA["uvw_dt"]
                    DT, Dnu = DATA["MSInfos"]
                    GridMachine.setDecorr(uvw_dt, DT, Dnu, SmearMode=DecorrMode)

                vis = GridMachine.get(times, uvwThis, visThis, flagsThis, A0A1,  # this is ever used
                                      ModelGrid, ImToGrid=False,
                                      DicoJonesMatrices=DicoJonesMatrices,
                                      freqs=freqs, TranformModelInput="FT",
                                      ChanMapping=ChanMapping)

                m_result_queue.put({"Success": True, "iFacet": iFacet})


class ClassFacetMachine():

    """
    This class contains all information about facets and projections.
    The class is responsible for tesselation, gridding, projection to image,
    unprojection to facets and degridding

    This class provides a basic gridded tesselation pattern.
    """

    def __init__(self,
                 VS,
                 GD,
                 #ParsetFile="ParsetNew.txt",
                 Precision="S",
                 PolMode="I",
                 Sols=None,
                 PointingID=0,
                 Parallel=False,  # True,
                 DoPSF=False,
                 Oversize=1,   # factor my which image is oversized
                 NCPU=psutil.cpu_count(),
                 IdSharedMem="",
                 IdSharedMemData=None,  # == IdSharedMem if None
                 ApplyCal=False):

        # IdSharedMem is used to identify structures in shared memory
        # used by this FacetMachine
        self.IdSharedMem = IdSharedMem
        self.HasFourierTransformed = False
        # IdSharedMemData is used to identify "global" structures in
        # shared memory such as DicoData
        self.IdSharedMemData = IdSharedMemData or IdSharedMem

        self.NCPU = int(GD["Parallel"]["NCPU"])
        self.ApplyCal = ApplyCal

        if Precision == "S":
            self.dtype = np.complex64
            self.CType = np.complex64
            self.FType = np.float32
            self.stitchedType = np.float32  # cleaning requires float32
        elif Precision == "D":
            self.dtype = np.complex128
            self.CType = np.complex64
            self.FType = np.float64
            self.stitchedType = np.float32  # cleaning requires float32

        self.DoDDE = False
        if Sols != None:
            self.setSols(Sols)

        self.PointingID = PointingID
        self.VS, self.GD = VS, GD
        self.npol = self.VS.StokesConverter.NStokesInImage()
        self.Parallel = Parallel
        DicoConfigGM = {}
        self.DicoConfigGM = DicoConfigGM
        self.DoPSF = DoPSF
        #self.MDC.setFreqs(ChanFreq)
        self.CasaImage = None
        self.IsDirtyInit = False
        self.IsDDEGridMachineInit = False
        self.SharedNames = []
        self.ConstructMode = GD["ImagerMainFacet"]["ConstructMode"]
        self.SpheNorm = True

        if self.ConstructMode == "Fader":
            self.SpheNorm = False
        else:
            raise RuntimeError(
                "Deprecated Facet construct mode. Only supports 'Fader'")
        self.Oversize = Oversize

        self.NormData = None
        self.NormImage = None
        self.FacetParallelEngine = WorkerImager

    def __del__(self):
        # print>>log,"Deleting shared memory"
        NpShared.DelAll(self.IdSharedMem)

    def SetLogModeSubModules(self, Mode="Silent"):
        SubMods = ["ModelBeamSVD", "ClassParam", "ModToolBox",
                   "ModelIonSVD2", "ClassPierce"]

        if Mode == "Silent":
            MyLogger.setSilent(SubMods)
        if Mode == "Loud":
            MyLogger.setLoud(SubMods)

    def setSols(self, SolsClass):
        self.DoDDE = True
        self.Sols = SolsClass

    def appendMainField(self, Npix=512, Cell=10., NFacets=5,
                        Support=11, OverS=5, Padding=1.2,
                        wmax=10000, Nw=11, RaDecRad=(0., 0.),
                        ImageName="Facet.image", **kw):

        """
        Add the primary field to the facet machine. This field is tesselated
        into NFacets by setFacetsLocs method
        Args:
            Npix:
            Cell:
            NFacets:
            Support:
            OverS:
            Padding:
            wmax:
            Nw:
            RaDecRad:
            ImageName:
            **kw:
        """

        Cell = self.GD["ImagerMainFacet"]["Cell"]

        self.ImageName = ImageName
        if self.DoPSF:
            Npix *= 1

        self.LraFacet = []
        self.LdecFacet = []

        self.ChanFreq = self.VS.GlobalFreqs

        self.NFacets = NFacets
        self.Cell = Cell
        self.CellSizeRad = (Cell / 3600.) * np.pi / 180.
        rac, decc = self.VS.CurrentMS.radec
        self.MainRaDec = (rac, decc)
        self.nch = self.VS.NFreqBands
        self.NChanGrid = self.nch
        self.SumWeights = np.zeros((self.NChanGrid, self.npol), float)

        self.CoordMachine = ModCoord.ClassCoordConv(rac, decc)
        # get the closest fast fft size:
        Npix = self.GD["ImagerMainFacet"]["Npix"]
        Padding = self.GD["ImagerMainFacet"]["Padding"]
        self.Padding = Padding
        Npix, _ = EstimateNpix(float(Npix), Padding=1)
        self.Npix = Npix
        self.OutImShape = (self.nch, self.npol, self.Npix, self.Npix)
        # image bounding box in radians:
        RadiusTot = self.CellSizeRad * self.Npix / 2
        self.RadiusTot = RadiusTot
        self.CornersImageTot = np.array([[-RadiusTot, -RadiusTot],
                                         [RadiusTot, -RadiusTot],
                                         [RadiusTot, RadiusTot],
                                         [-RadiusTot, RadiusTot]])
        self.setFacetsLocs()

    def AppendFacet(self, iFacet, l0, m0, diam):

        """
        Adds facet dimentions to info dict of facets (self.DicoImager[iFacet])
        Args:
            iFacet:
            l0:
            m0:
            diam:
        """

        diam *= self.Oversize

        DicoConfigGM = None
        lmShift = (l0, m0)
        self.DicoImager[iFacet]["lmShift"] = lmShift
        # CellRad=(Cell/3600.)*np.pi/180.

        raFacet, decFacet = self.CoordMachine.lm2radec(
                            np.array([lmShift[0]]), np.array([lmShift[1]]))

        NpixFacet, _ = EstimateNpix(diam / self.CellSizeRad, Padding=1)
        _, NpixPaddedGrid = EstimateNpix(NpixFacet, Padding=self.Padding)

        diam = NpixFacet * self.CellSizeRad
        diamPadded = NpixPaddedGrid * self.CellSizeRad
        RadiusFacet = diam * 0.5
        RadiusFacetPadded = diamPadded * 0.5
        self.DicoImager[iFacet]["lmDiam"] = RadiusFacet
        self.DicoImager[iFacet]["lmDiamPadded"] = RadiusFacetPadded
        self.DicoImager[iFacet]["RadiusFacet"] = RadiusFacet
        self.DicoImager[iFacet]["RadiusFacetPadded"] = RadiusFacetPadded
        self.DicoImager[iFacet]["lmExtent"] = l0 - RadiusFacet, \
            l0 + RadiusFacet, m0 - RadiusFacet, m0 + RadiusFacet
        self.DicoImager[iFacet]["lmExtentPadded"] = l0 - RadiusFacetPadded, \
                                                    l0 + RadiusFacetPadded, \
                                                    m0 - RadiusFacetPadded, \
                                                    m0 + RadiusFacetPadded

        lSol, mSol = self.lmSols
        raSol, decSol = self.radecSols
        dSol = np.sqrt((l0 - lSol) ** 2 + (m0 - mSol) ** 2)
        iSol = np.where(dSol == np.min(dSol))[0]
        self.DicoImager[iFacet]["lmSol"] = lSol[iSol], mSol[iSol]
        self.DicoImager[iFacet]["radecSol"] = raSol[iSol], decSol[iSol]
        self.DicoImager[iFacet]["iSol"] = iSol

        # print>>log,"#[%3.3i] %f, %f"%(iFacet,l0,m0)
        DicoConfigGM = {"Npix": NpixFacet,
                        "Cell": self.GD["ImagerMainFacet"]["Cell"],
                        "ChanFreq": self.ChanFreq,
                        "DoPSF": False,
                        "Support": self.GD["ImagerCF"]["Support"],
                        "OverS": self.GD["ImagerCF"]["OverS"],
                        "wmax": self.GD["ImagerCF"]["wmax"],
                        "Nw": self.GD["ImagerCF"]["Nw"],
                        "WProj": True,
                        "DoDDE": self.DoDDE,
                        "Padding": self.GD["ImagerMainFacet"]["Padding"]}

        _, _, NpixOutIm, NpixOutIm = self.OutImShape

        self.DicoImager[iFacet]["l0m0"] = lmShift
        self.DicoImager[iFacet]["RaDec"] = raFacet[0], decFacet[0]
        self.LraFacet.append(raFacet[0])
        self.LdecFacet.append(decFacet[0])
        xc, yc = int(round(l0 / self.CellSizeRad + NpixOutIm / 2)), \
                int(round(m0 / self.CellSizeRad + NpixOutIm / 2))

        self.DicoImager[iFacet]["pixCentral"] = xc, yc
        self.DicoImager[iFacet]["pixExtent"] = round(xc - NpixFacet / 2), \
                                               round(xc + NpixFacet / 2 + 1), \
                                               round(yc - NpixFacet / 2), \
                                               round(yc + NpixFacet / 2 + 1)

        self.DicoImager[iFacet]["NpixFacet"] = NpixFacet
        self.DicoImager[iFacet]["NpixFacetPadded"] = NpixPaddedGrid
        self.DicoImager[iFacet]["DicoConfigGM"] = DicoConfigGM
        self.DicoImager[iFacet]["IDFacet"] = iFacet
        # print self.DicoImager[iFacet]

        self.FacetCat.ra[iFacet] = raFacet[0]
        self.FacetCat.dec[iFacet] = decFacet[0]
        l, m = self.DicoImager[iFacet]["l0m0"]
        self.FacetCat.l[iFacet] = l
        self.FacetCat.m[iFacet] = m
        self.FacetCat.Cluster[iFacet] = iFacet

    def setFacetsLocs(self):

        """
        Routine to split the image into a grid of squares.
        This can be overridden to perform more complex tesselations
        """

        Npix = self.GD["ImagerMainFacet"]["Npix"]
        NFacets = self.GD["ImagerMainFacet"]["NFacets"]
        Padding = self.GD["ImagerMainFacet"]["Padding"]
        self.Padding = Padding
        NpixFacet, _ = EstimateNpix(float(Npix) / NFacets, Padding=1)
        Npix = NpixFacet * NFacets
        self.Npix = Npix
        self.OutImShape = (self.nch, self.npol, self.Npix, self.Npix)
        _, NpixPaddedGrid = EstimateNpix(NpixFacet, Padding=Padding)
        self.NpixPaddedFacet = NpixPaddedGrid
        self.NpixFacet = NpixFacet
        self.FacetShape = (self.nch, self.npol, NpixFacet, NpixFacet)
        self.PaddedGridShape = (self.NChanGrid, self.npol,
                                NpixPaddedGrid, NpixPaddedGrid)

        RadiusTot = self.CellSizeRad * self.Npix / 2
        self.RadiusTot = RadiusTot

        lMainCenter, mMainCenter = 0., 0.
        self.lmMainCenter = lMainCenter, mMainCenter
        self.CornersImageTot = np.array(
                                [[lMainCenter - RadiusTot, mMainCenter - RadiusTot],
                                [lMainCenter + RadiusTot, mMainCenter - RadiusTot],
                                [lMainCenter + RadiusTot, mMainCenter + RadiusTot],
                                [lMainCenter - RadiusTot, mMainCenter + RadiusTot]])

        print>> log, "Sizes (%i x %i facets):" % (NFacets, NFacets)
        print>> log, "   - Main field :   [%i x %i] pix" % \
            (self.Npix, self.Npix)
        print>> log, "   - Each facet :   [%i x %i] pix" % \
            (NpixFacet, NpixFacet)
        print>> log, "   - Padded-facet : [%i x %i] pix" % \
            (NpixPaddedGrid, NpixPaddedGrid)

        ############################

        self.NFacets = NFacets
        lrad = Npix * self.CellSizeRad * 0.5
        self.ImageExtent = [-lrad, lrad, -lrad, lrad]

        lfacet = NpixFacet * self.CellSizeRad * 0.5
        lcenter_max = lrad - lfacet
        lFacet, mFacet, = np.mgrid[-lcenter_max:lcenter_max:(NFacets) * 1j,
            -lcenter_max:lcenter_max:(NFacets) * 1j]
        lFacet = lFacet.flatten()
        mFacet = mFacet.flatten()
        x0facet, y0facet = np.mgrid[0:Npix:NpixFacet, 0:Npix:NpixFacet]
        x0facet = x0facet.flatten()
        y0facet = y0facet.flatten()

        # print "Append1"; self.IM.CI.E.clear()

        self.DicoImager = {}
        for iFacet in range(lFacet.size):
            self.DicoImager[iFacet] = {}

        # print "Append2"; self.IM.CI.E.clear()

        self.FacetCat = np.zeros((lFacet.size,),
                        dtype=[('Name', '|S200'), ('ra', np.float),
                        ('dec', np.float), ('SumI', np.float), ("Cluster", int),
                        ("l", np.float), ("m", np.float), ("I", np.float)])

        self.FacetCat = self.FacetCat.view(np.recarray)
        self.FacetCat.I = 1
        self.FacetCat.SumI = 1

        for iFacet in range(lFacet.size):
            l0 = x0facet[iFacet] * self.CellSizeRad
            m0 = y0facet[iFacet] * self.CellSizeRad
            l0 = lFacet[iFacet]
            m0 = mFacet[iFacet]

            # print x0facet[iFacet],y0facet[iFacet],l0,m0
            self.AppendFacet(iFacet, l0, m0, NpixFacet * self.CellSizeRad)

        self.DicoImagerCentralFacet = self.DicoImager[lFacet.size / 2]

        self.SetLogModeSubModules("Silent")
        self.MakeREG()

    def MakeREG(self):

        """
        Writes out ds9 tesselation region file
        """

        regFile = "%s.Facets.reg" % self.ImageName

        print>>log, "Writing facets locations in %s" % regFile

        f = open(regFile, "w")
        f.write("# Region file format: DS9 version 4.1\n")
        ss0 = 'global color=green dashlist=8 3 width=1 font="helvetica 10 \
            normal roman" select=1 highlite=1 dash=0'
        ss1 = ' fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'

        f.write(ss0+ss1)
        f.write("fk5\n")

        for iFacet in self.DicoImager.keys():
            #rac,decc=self.DicoImager[iFacet]["RaDec"]
            l0, m0 = self.DicoImager[iFacet]["l0m0"]
            diam = self.DicoImager[iFacet]["lmDiam"]
            dl = np.array([-1, 1, 1, -1, -1])*diam
            dm = np.array([-1, -1, 1, 1, -1])*diam
            l = ((dl.flatten()+l0)).tolist()
            m = ((dm.flatten()+m0)).tolist()

            x = []
            y = []
            for iPoint in range(len(l)):
                xp, yp = self.CoordMachine.lm2radec(np.array(
                    [l[iPoint]]), np.array([m[iPoint]]))
                x.append(xp)
                y.append(yp)

            x = np.array(x)#+[x[2]])
            y = np.array(y)#+[y[2]])

            x *= 180/np.pi
            y *= 180/np.pi

            for iline in range(x.shape[0]-1):
                x0 = x[iline]
                y0 = y[iline]
                x1 = x[iline+1]
                y1 = y[iline+1]
                f.write("line(%f,%f,%f,%f) # line=0 0\n" % (x0, y0, x1, y1))

        f.close()


    #####################################################
    ################ Initialisation #####################
    #####################################################

    def PlotFacetSols(self):

        DicoClusterDirs = NpShared.SharedToDico(
            "%sDicoClusterDirs" % self.IdSharedMemData)
        lc = DicoClusterDirs["l"]
        mc = DicoClusterDirs["m"]
        sI = DicoClusterDirs["I"]
        x0, x1 = lc.min()-np.pi/180, lc.max()+np.pi/180
        y0, y1 = mc.min()-np.pi/180, mc.max()+np.pi/180
        InterpMode = self.GD["DDESolutions"]["Type"]
        if InterpMode == "Krigging":
            for iFacet in sorted(self.DicoImager.keys()):
                l0, m0 = self.DicoImager[iFacet]["lmShift"]
                d0 = self.GD["DDESolutions"]["Scale"]*np.pi/180
                gamma = self.GD["DDESolutions"]["gamma"]

                d = np.sqrt((l0-lc)**2+(m0-mc)**2)
                idir = np.argmin(d)  # this is not used
                w = sI/(1.+d/d0) ** gamma
                w /= np.sum(w)
                w[w < (0.2 * w.max())] = 0
                ind = np.argsort(w)[::-1]
                w[ind[4::]] = 0

                ind = np.where(w != 0)[0]
                pylab.clf()
                pylab.scatter(lc[ind], mc[ind], c=w[ind], vmin=0, vmax=w.max())
                pylab.scatter([l0], [m0], marker="+")
                pylab.xlim(x0, x1)
                pylab.ylim(y0, y1)
                pylab.draw()
                pylab.show(False)
                pylab.pause(0.1)

    def Init(self):
        """
        Initialize either in parallel or serial
        """
        if self.IsDDEGridMachineInit: return
        self.DicoGridMachine = {}

        for iFacet in self.DicoImager.keys():
            self.DicoGridMachine[iFacet] = {}

        self.setWisdom()

        if self.Parallel:
            self.InitParallel(Parallel=True)
        else:
            self.InitParallel(Parallel=False)

        self.IsDDEGridMachineInit = True
        self.SetLogModeSubModules("Loud")

    def setWisdom(self):

        """
        Set fft wisdom
        """

        self.FFTW_Wisdom = None

        print>>log, "Set fftw widsdom for shape = %s" % str(self.PaddedGridShape)

        a = np.random.randn(*(self.PaddedGridShape)) \
            + 1j*np.random.randn(*(self.PaddedGridShape))
        FM = ModFFTW.FFTW_2Donly(self.PaddedGridShape, np.complex64)
        b = FM.fft(a)  # this is never used
        self.FFTW_Wisdom = pyfftw.export_wisdom()

        for iFacet in sorted(self.DicoImager.keys()):
            A = ModFFTW.GiveFFTW_aligned(self.PaddedGridShape, np.complex64)
            NpShared.ToShared("%sFFTW.%i" % (self.IdSharedMem, iFacet), A)

    def InitParallel(self, Parallel=True):

        """
        Does initialization routines (e.g. gridding machine initialization)
        in parallel.
        Args:
            Parallel: Can force the initialization to serial if this
            is set to false.
        Post-conditions:
            self.SpacialWeigth, the tesselation area weights are set to
            those computed by the workers.

        """

        import Queue

        NFacets = len(self.DicoImager.keys())
        self.SpacialWeigth = {}

        procs = list()  # list of processes that are running
        n_cpus = psutil.cpu_count()  # number of CPU's. Only tested without HT
        procinfo = psutil.Process()  # this will be used to control CPU affinity

        # check if spacial weights are cached
        cachepath, cachevalid = self.VS.maincache.checkCache("FacetData",
                                dict(ImagerCF=self.GD["ImagerCF"],
                                ImagerMainFacet=self.GD["ImagerMainFacet"]),
                                directory=True)

        self.FacetDataCache = "file://" + cachepath + "/"

        if Parallel:
            qlimit = n_cpus*8
        else:
            qlimit = 0

        if not cachevalid:
            # if data is larger use (maxsize=qlimit), only ints in this case
            m_work_queue = multiprocessing.Queue()
            m_result_queue = multiprocessing.JoinableQueue()

            Mode = "Init"
            FFTW_Wisdom = self.FFTW_Wisdom
            DicoImager = self.DicoImager
            IdSharedMem = self.IdSharedMem
            IdSharedMemData = self.IdSharedMemData
            FacetDataCache = self.FacetDataCache
            ChunkDataCache = None
            ApplyCal = self.ApplyCal
            CornersImageTot = self.CornersImageTot
            NFreqBands = self.VS.NFreqBands
            DataCorrelationFormat = self.VS.StokesConverter.AvailableCorrelationProductsIds()
            ExpectedOutputStokes = self.VS.StokesConverter.RequiredStokesProductsIds()
            CoordMachine = self.CoordMachine
            npol = self.VS.StokesConverter.NStokesInImage()
            nch = self.VS.NFreqBands
            MainRaDec = self.MainRaDec
            CellSizeRad = self.CellSizeRad
            NFacets = self.NFacets
            IdSharedMemData = self.IdSharedMemData
            SpheNorm = self.SpheNorm
            ListSemaphores = None

            # Create a list of work items for the producer process
            NFacetsList = []
            for iFacet in xrange(NFacets):
                NFacetsList.append(iFacet)

            work_p = Process(target=work_producer,
                            args=(m_work_queue, NFacetsList,))

            EngineInstance = self.FacetParallelEngine(m_work_queue,
                            m_result_queue,
                            self.GD,
                            Mode="Init",
                            FFTW_Wisdom=self.FFTW_Wisdom,
                            DicoImager=self.DicoImager,
                            IdSharedMem=self.IdSharedMem,
                            IdSharedMemData=self.IdSharedMemData,
                            FacetDataCache=self.FacetDataCache,
                            ChunkDataCache=None,
                            ApplyCal=self.ApplyCal,
                            CornersImageTot=self.CornersImageTot,
                            NFreqBands=self.VS.NFreqBands,
                            DataCorrelationFormat=self.VS.StokesConverter.AvailableCorrelationProductsIds(),
                            ExpectedOutputStokes=self.VS.StokesConverter.RequiredStokesProductsIds())

            for cpu in xrange(n_cpus):
                p = Process(target=init_w_worker_tessel, args=(
                            m_work_queue,
                            m_result_queue,
                            self.GD,
                            Mode,
                            FFTW_Wisdom,
                            DicoImager,
                            IdSharedMem,
                            FacetDataCache,
                            ApplyCal,
                            NFreqBands,
                            ExpectedOutputStokes,
                            CoordMachine,
                            nch,
                            npol,
                            MainRaDec,
                            CellSizeRad,
                            NFacets,
                            IdSharedMemData,
                            ChunkDataCache,
                            SpheNorm,
                            DataCorrelationFormat,
                            ListSemaphores,
                            EngineInstance,))

                procs.append(p)

            if Parallel:
                main_core = 0  # CPU affinity placement
                # start a pinned work producer process
                work_p.start()
                procinfo.cpu_affinity([n_cpus-1])

                # start all processes and pin them each to a core
                for p in procs:
                    p.start()
                    procinfo.cpu_affinity([main_core])
                    main_core += 1

            timer = ClassTimeIt.ClassTimeIt()
            print>> log, "initializing W kernels"

            pBAR = ProgressBar('white', width=50, block='=', empty=' ',
                                Title="      Init W ", HeaderSize=10, TitleSize=13)
            pBAR.render(0, '%4i/%i' % (0, NFacets))
            iResult = 0

            if not Parallel:
                for p in procs:
                    p.run()  # just run until all work is completed

            NJobs = NFacets

            while iResult < NJobs:
                DicoResult = {}
                try:
                    DicoResult = m_result_queue.get(True, 10)
                except Queue.Empty:
                    pass
                    print>> log, "checking for dead workers"
                    # shoot the zombie process
                    multiprocessing.active_children()
                    # check for dead workers
                    pids_to_restart = []
                    for w in procs:
                        if not w.is_alive():
                            pids_to_restart[w]
                            raise RuntimeError, "a worker process has died on \
                                us with exit code %d. This is probably a bug." \
                                % w.exitcode

                            for id in pids_to_restart:
                                print>> log, "need to restart worker %d." % id
                                pass

                if len(DicoResult) != 0 and DicoResult["Success"]:
                    iResult += 1
                    m_result_queue.task_done()  # call task_done on the queue

                    NDone = iResult
                    intPercent = int(100 * NDone / float(NFacets))
                    pBAR.render(intPercent, '%4i/%i' % (NDone, NFacets))

            if Parallel:
                # if all work is done send poinsed pill to workers
                if iResult == NJobs:
                    for p in procs:
                        m_work_queue.put("POISON-E")

                    # join and close queues
                    m_result_queue.join()
                    m_work_queue.close()
                    m_result_queue.close()

                    # join producer process
                    work_p.join()

                    # join producer process
                    for p in procs:
                        p.join()

            print>> log, "init W finished in %s" % timer.timehms()
            self.VS.maincache.saveCache("FacetData")
        else:
            print>>log, "using W kernels from cache %s" % cachepath

        # now load cached weights
        for iFacet in sorted(self.DicoImager.keys()):
            NameSpacialWeigth = "%sSpacialWeight.Facet_%3.3i" % \
                (self.FacetDataCache, iFacet)
            SpacialWeigth = NpShared.GiveArray(NameSpacialWeigth)
            self.SpacialWeigth[iFacet] = SpacialWeigth
        return True


    ############################################################################################
    ############################################################################################
    ############################################################################################

    def setCasaImage(self,ImageName=None,Shape=None,Freqs=None,Stokes=["I"]):
        if ImageName is None:
            ImageName=self.ImageName

        if Shape is None:
            Shape=self.OutImShape
        self.CasaImage=ClassCasaImage.ClassCasaimage(ImageName,Shape,self.Cell,self.MainRaDec,Freqs=Freqs,Stokes=Stokes)

    def ToCasaImage(self,ImageIn,Fits=True,ImageName=None,beam=None,beamcube=None,Freqs=None,Stokes=["I"]):
        self.setCasaImage(ImageName=ImageName,Shape=ImageIn.shape,Freqs=Freqs,Stokes=Stokes)

        self.CasaImage.setdata(ImageIn,CorrT=True)

        if Fits:
            self.CasaImage.ToFits()
            if beam is not None:
                self.CasaImage.setBeam(beam,beamcube=beamcube)
        self.CasaImage.close()
        self.CasaImage=None

    def GiveEmptyMainField(self):
        """
        Gives empty image of the correct shape to act as buffer for e.g. the stitching process
        Returns:
            ndarray of type complex
        """
        return np.zeros(self.OutImShape,dtype=self.stitchedType)

    def putChunk(self,*args,**kwargs):
        """
        Args:
            *args: should consist of the following:
                time nparray
                uvw nparray
                vis nparray
                flags nparray
                A0A1 tuple of antenna1 and antenna2 nparrays
            **kwargs:
                keyword args must include the following:
                doStack
        """
        self.SetLogModeSubModules("Silent")
        if not(self.IsDDEGridMachineInit):
            self.Init()

        if not(self.IsDirtyInit):
            self.ReinitDirty()

        if self.Parallel:
            kwargs["Parallel"] = True
            self.CalcDirtyImagesParallel(*args,**kwargs)
        else:
            kwargs["Parallel"] = False
            self.CalcDirtyImagesParallel(*args,**kwargs)
        self.SetLogModeSubModules("Loud")

    def getChunk(self,*args,**kwargs):
        self.SetLogModeSubModules("Silent")
        if self.Parallel:
            kwargs["Parallel"] = True
            self.GiveVisParallel(*args,**kwargs)
        else:
            kwargs["Parallel"] = False
            self.GiveVisParallel(*args,**kwargs)
        self.SetLogModeSubModules("Loud")

    def FacetsToIm(self,NormJones=False):
        """
        Fourier transforms the individual facet grids and then
        Stitches the gridded facets and builds the following maps:
            self.stitchedResidual (initial residual is the dirty map)
            self.NormImage (grid-correcting map, see also: BuildFacetNormImage)
            self.MeanResidual ("average" residual map taken over all continuum bands of the residual cube,
                               this will be the same as stitchedResidual if there is only one continuum band in the residual
                               cube)
            self.DicoPSF if the facet machine is set to produce a PSF. This contains, amongst others a PSF and mean psf per facet
            Note that only the stitched residuals are currently normalized and converted to stokes images for cleaning.
            This is because the coplanar facets should be jointly cleaned on a single map.
        Args:
            NormJones: if True (and there is Jones Norm data available) also computes self.NormData (ndarray) of jones
            averages.


        Returns:
            Dictionary containing:
            "ImagData" = self.stitchedResidual
            "NormImage" = self.NormImage (grid-correcting map)
            "NormData" = self.NormData (if computed, see above)
            "MeanImage" = self.MeanResidual
            "freqs" = channel information on the bands being averaged into each of the continuum slices of the residual
            "SumWeights" = sum of visibility weights used in normalizing the gridded correlations
            "WeightChansImages" = normalized weights
        """
        if not self.HasFourierTransformed:
            self.FourierTransform(Parallel=self.GD["Parallel"]["Enable"])
            self.HasFourierTransformed = True
        _,npol,Npix,Npix=self.OutImShape
        DicoImages={}
        DicoImages["freqs"]={}

        DoCalcNormData=False
        if (NormJones)&(self.NormData is None):
            DoCalcNormData=True

        #assume all facets have the same weight sums. Store the normalization weights for reference
        # -------------------------------------------------
        DicoImages["SumWeights"]=np.zeros((self.VS.NFreqBands,self.npol),np.float64)
        for band,channels in enumerate(self.VS.FreqBandChannels):
            DicoImages["freqs"][band] = channels
            DicoImages["SumWeights"][band] = self.DicoImager[0]["SumWeights"][band]
        DicoImages["WeightChansImages"] = DicoImages["SumWeights"] / np.sum(DicoImages["SumWeights"])

        #Build a residual image consisting of multiple continuum bands
        # -------------------------------------------------
        if self.NormImage is None:
            self.NormImage = self.BuildFacetNormImage()
            self.NormImageReShape = self.NormImage.reshape([1,1,
                                                            self.NormImage.shape[0],
                                                            self.NormImage.shape[1]])
        self.stitchedResidual = self.FacetsToIm_Channel()
        if DoCalcNormData:
            self.NormData = self.FacetsToIm_Channel(BeamWeightImage=True)

        #Normalize each of the continuum bands of the combined residual by the weights that contributed to that band:
        # -------------------------------------------------
        if self.VS.MultiFreqMode:
            ImMean=np.zeros_like(self.stitchedResidual)
            W=np.array([DicoImages["SumWeights"][Channel] for Channel in range(self.VS.NFreqBands)])
            W/=np.sum(W,axis=0) #sum frequency contribution to weights per correlation
            W=np.float32(W.reshape((self.VS.NFreqBands,npol,1,1)))
            self.MeanResidual=np.sum(self.stitchedResidual*W,axis=0).reshape((1,npol,Npix,Npix)) #weight each of the cube slices and average
        else:
            self.MeanResidual=self.stitchedResidual.copy() #if there are no bands in the continuum image then the mean image is just the same

        if self.DoPSF:
            print>>log, "  Build PSF facet-slices "
            self.DicoPSF={}
            for iFacet in self.DicoGridMachine.keys():
                #first normalize by spheriodals - these facet psfs will be used in deconvolution per facet
                SharedMemName="%sSpheroidal.Facet_%3.3i"%(self.FacetDataCache,iFacet)
                SPhe=NpShared.GiveArray(SharedMemName)
                nx=SPhe.shape[0]
                SPhe=SPhe.reshape((1,1,nx,nx)).real
                self.DicoPSF[iFacet]={}
                self.DicoPSF[iFacet]["PSF"]=(self.DicoGridMachine[iFacet]["Dirty"]).copy().real
                self.DicoPSF[iFacet]["PSF"]/=SPhe
                #self.DicoPSF[iFacet]["PSF"][SPhe < 1e-2] = 0
                self.DicoPSF[iFacet]["l0m0"]=self.DicoImager[iFacet]["l0m0"]
                self.DicoPSF[iFacet]["pixCentral"]=self.DicoImager[iFacet]["pixCentral"]
                self.DicoPSF[iFacet]["lmSol"] = self.DicoImager[iFacet]["lmSol"]

                nch,npol,n,n=self.DicoPSF[iFacet]["PSF"].shape
                PSFChannel=np.zeros((nch,npol,n,n),self.stitchedType)
                for ch in range(nch):
                    self.DicoPSF[iFacet]["PSF"][ch][SPhe[0] < 1e-2] = 0
                    self.DicoPSF[iFacet]["PSF"][ch][0]=self.DicoPSF[iFacet]["PSF"][ch][0].T[::-1,:]
                    SumJonesNorm=self.DicoImager[iFacet]["SumJonesNorm"][ch]
                    self.DicoPSF[iFacet]["PSF"][ch]/=np.sqrt(SumJonesNorm) #normalize to bring back transfer functions to approximate convolution
                    for pol in range(npol):
                        ThisSumWeights=self.DicoImager[iFacet]["SumWeights"][ch][pol]
                        self.DicoPSF[iFacet]["PSF"][ch][pol]/=ThisSumWeights #normalize the response per facet channel if jones corrections are enabled
                    PSFChannel[ch,:,:,:]=self.DicoPSF[iFacet]["PSF"][ch][:,:,:]

                W=DicoImages["WeightChansImages"]
                W=np.float32(W.reshape((self.VS.NFreqBands,npol,1,1)))
                MeanPSF=np.sum(PSFChannel*W,axis=0).reshape((1,npol,n,n)) #weight each of the cube slices and average
                self.DicoPSF[iFacet]["MeanPSF"]=MeanPSF

            DicoVariablePSF=self.DicoPSF
            NFacets=len(DicoVariablePSF.keys())

            if self.GD["ImagerMainFacet"]["Circumcision"]:
                NPixMin = self.GD["ImagerMainFacet"]["Circumcision"]
#                print>>log,"using explicit Circumcision=%d"%NPixMin
            else:
                NPixMin=1e6
                for iFacet in sorted(DicoVariablePSF.keys()):
                    _,npol,n,n=DicoVariablePSF[iFacet]["PSF"].shape
                    if n<NPixMin: NPixMin=n

                NPixMin = int(NPixMin/self.GD["ImagerMainFacet"]["Padding"])
                if not NPixMin%2:
                    NPixMin += 1
#                print>>log,"using computed Circumcision=%d"%NPixMin

            nch = self.VS.NFreqBands
            CubeVariablePSF=np.zeros((NFacets,nch,npol,NPixMin,NPixMin),np.float32)
            CubeMeanVariablePSF=np.zeros((NFacets,1,npol,NPixMin,NPixMin),np.float32)

            print>>log, "  Cutting PSFs facet-slices "
            for iFacet in sorted(DicoVariablePSF.keys()):
                _,npol,n,n=DicoVariablePSF[iFacet]["PSF"].shape
                for ch in range(nch):
                    i=n/2-NPixMin/2
                    j=n/2+NPixMin/2+1
                    CubeVariablePSF[iFacet,ch,:,:,:]=DicoVariablePSF[iFacet]["PSF"][ch][:,i:j,i:j]
                CubeMeanVariablePSF[iFacet,0,:,:,:]=DicoVariablePSF[iFacet]["MeanPSF"][0,:,i:j,i:j]

            self.DicoPSF["CubeVariablePSF"]=CubeVariablePSF
            self.DicoPSF["CubeMeanVariablePSF"]=CubeMeanVariablePSF
            self.DicoPSF["MeanFacetPSF"]=np.mean(CubeMeanVariablePSF,axis=0).reshape((1,npol,NPixMin,NPixMin))
            self.DicoPSF["MeanJonesBand"]=[]
            self.DicoPSF["OutImShape"] = self.OutImShape
            self.DicoPSF["CellSizeRad"] = self.CellSizeRad
            for iFacet in sorted(self.DicoImager.keys()):
                MeanJonesBand=np.zeros((self.VS.NFreqBands,),np.float64)
                for Channel in range(self.VS.NFreqBands):
                    ThisSumSqWeights=self.DicoImager[iFacet]["SumJones"][1][Channel]
                    if ThisSumSqWeights==0: ThisSumSqWeights=1.
                    ThisSumJones=(self.DicoImager[iFacet]["SumJones"][0][Channel]/ThisSumSqWeights)
                    if ThisSumJones==0:
                        ThisSumJones=1.
                    MeanJonesBand[Channel]=ThisSumJones
                self.DicoPSF["MeanJonesBand"].append(MeanJonesBand)


            self.DicoPSF["SumJonesChan"]=[]
            self.DicoPSF["SumJonesChanWeightSq"]=[]
            for iFacet in sorted(self.DicoImager.keys()):
                ThisFacetSumJonesChan=[]
                ThisFacetSumJonesChanWeightSq=[]
                for iMS in range(self.VS.nMS):
                    A=self.DicoImager[iFacet]["SumJonesChan"][iMS][1,:]
                    A[A==0]=1.
                    A=self.DicoImager[iFacet]["SumJonesChan"][iMS][0,:]
                    A[A==0]=1.
                    SumJonesChan=self.DicoImager[iFacet]["SumJonesChan"][iMS][0,:]
                    SumJonesChanWeightSq=self.DicoImager[iFacet]["SumJonesChan"][iMS][1,:]
                    ThisFacetSumJonesChan.append(SumJonesChan)
                    ThisFacetSumJonesChanWeightSq.append(SumJonesChanWeightSq)

                self.DicoPSF["SumJonesChan"].append(ThisFacetSumJonesChan)
                self.DicoPSF["SumJonesChanWeightSq"].append(ThisFacetSumJonesChanWeightSq)
            self.DicoPSF["ChanMappingGrid"]=self.VS.DicoMSChanMapping
            self.DicoPSF["ChanMappingGridChan"]=self.VS.DicoMSChanMappingChan
            self.DicoPSF["freqs"]=DicoImages["freqs"]
            self.DicoPSF["WeightChansImages"]=DicoImages["WeightChansImages"]

        DicoImages["ImagData"] = self.stitchedResidual
        DicoImages["NormImage"] = self.NormImage #grid-correcting map
        DicoImages["NormData"] = self.NormData
        DicoImages["MeanImage"] = self.MeanResidual

        return DicoImages

    def BuildFacetNormImage(self):
        """
        Creates a stitched tesselation weighting map. This can be useful to downweight areas where facets overlap
        (e.g. padded areas) before stitching the facets into one map.
        Returns
            ndarray with norm image
        """
        print>>log,"  Building Facet-normalisation image"
        nch,npol=self.nch,self.npol
        _,_,NPixOut,NPixOut=self.OutImShape
        NormImage=np.zeros((NPixOut,NPixOut),dtype=self.stitchedType)
        for iFacet in self.DicoImager.keys():
            xc,yc=self.DicoImager[iFacet]["pixCentral"]
            NpixFacet=self.DicoImager[iFacet]["NpixFacetPadded"]

            Aedge,Bedge=GiveEdges((xc,yc),NPixOut,(NpixFacet/2,NpixFacet/2),NpixFacet)
            x0d,x1d,y0d,y1d=Aedge
            x0p,x1p,y0p,y1p=Bedge

            SpacialWeigth=self.SpacialWeigth[iFacet].T[::-1,:]
            SW=SpacialWeigth[::-1,:].T[x0p:x1p,y0p:y1p]
            NormImage[x0d:x1d,y0d:y1d]+=np.real(SW)

        return NormImage

    def FacetsToIm_Channel(self,BeamWeightImage=False):
        """
        Preconditions: assumes the stitched tesselation weighting map has been created previously
        Args:
            BeamWeightImage: if true creates a stitched jones amplitude image instead of a stitched
            risidual / psf map
        Returns:
            Image cube, which may contain multiple correlations and continuum channel bands
        """
        T=ClassTimeIt.ClassTimeIt("FacetsToIm_Channel")
        T.disable()
        Image=self.GiveEmptyMainField()

        nch,npol,NPixOut,NPixOut=self.OutImShape

        if BeamWeightImage:
            print>>log, "Combining facets to average Jones-amplitude image"
        else:
            print>>log, "Combining facets to residual image"


        NormImage=self.NormImage

        for iFacet in self.DicoImager.keys():

            SharedMemName="%s/Spheroidal.Facet_%3.3i"%(self.FacetDataCache,iFacet)
            SPhe=NpShared.GiveArray(SharedMemName)


            xc,yc=self.DicoImager[iFacet]["pixCentral"]
            NpixFacet=self.DicoGridMachine[iFacet]["Dirty"][0].shape[2]

            Aedge,Bedge=GiveEdges((xc,yc),NPixOut,(NpixFacet/2,NpixFacet/2),NpixFacet)
            x0main,x1main,y0main,y1main=Aedge
            x0facet,x1facet,y0facet,y1facet=Bedge

            self.DicoImager[iFacet]["SumJonesNorm"]=np.zeros((self.VS.NFreqBands,),np.float64)

            for Channel in range(self.VS.NFreqBands):


                ThisSumWeights=self.DicoImager[iFacet]["SumWeights"][Channel]
                ThisSumJones=1.

                ThisSumSqWeights=self.DicoImager[iFacet]["SumJones"][1][Channel]
                if ThisSumSqWeights==0:
                    ThisSumSqWeights=1.
                ThisSumJones=self.DicoImager[iFacet]["SumJones"][0][Channel]/ThisSumSqWeights
                if ThisSumJones==0:
                    ThisSumJones=1.
                self.DicoImager[iFacet]["SumJonesNorm"][Channel]=ThisSumJones


                SpacialWeigth=self.SpacialWeigth[iFacet].T[::-1,:]

                T.timeit("3")
                for pol in range(npol):
                    sumweight=ThisSumWeights[pol]#ThisSumWeights.reshape((nch,npol,1,1))[Channel, pol, 0, 0]

                    if BeamWeightImage:
                        Im=SpacialWeigth[::-1,:].T[x0facet:x1facet,y0facet:y1facet]*ThisSumJones
                    else:
                        Im=self.DicoGridMachine[iFacet]["Dirty"][Channel][pol].copy()
                        Im/=SPhe.real           #grid-correct the image with the gridding convolution function
                        Im[SPhe<1e-3]=0
                        Im=(Im[::-1,:].T/sumweight)
                        SW=SpacialWeigth[::-1,:].T
                        Im*=SW

                        Im/=np.sqrt(ThisSumJones)
                        #Im/=(ThisSumJones)

                        Im=Im[x0facet:x1facet,y0facet:y1facet]


                    Image[Channel,pol,x0main:x1main,y0main:y1main]+=Im.real


        for Channel in range(self.VS.NFreqBands):
            for pol in range(npol):
                Image[Channel,pol]/=self.NormImage

        return Image

    def GiveNormImage(self):
        """
        Creates a stitched normalization image of the grid-correction function. This image should be point-wise
        divided from the stitched gridded map to create a grid-corrected map.
        Returns:
            stitched grid-correction norm image
        """
        Image=self.GiveEmptyMainField()
        nch,npol=self.nch,self.npol
        _,_,NPixOut,NPixOut=self.OutImShape
        SharedMemName="%sSpheroidal"%(self.IdSharedMemData)
        NormImage=np.zeros((NPixOut,NPixOut),dtype=self.stitchedType)
        SPhe=NpShared.GiveArray(SharedMemName)
        N1=self.NpixPaddedFacet

        for iFacet in self.DicoImager.keys():

            xc,yc=self.DicoImager[iFacet]["pixCentral"]
            Aedge,Bedge=GiveEdges((xc,yc),NPixOut,(N1/2,N1/2),N1)
            x0d,x1d,y0d,y1d=Aedge
            x0p,x1p,y0p,y1p=Bedge

            for ch in range(nch):
                for pol in range(npol):
                    NormImage[x0d:x1d,y0d:y1d]+=SPhe[::-1,:].T.real[x0p:x1p,y0p:y1p]


        return NormImage

    def ImToGrids(self,Image):
        """
        Unprojects image to facets (necessary for degridding). This also applies the tesselation
        mask weights to each of the facets. The group of facets are stored in shared memory with
        identifier: sModelImage.Facet_%3.3i
        Args:
            Image: The stitched image to be unprojected / "unstitched"
        """
        Im2Grid=ClassImToGrid(OverS=self.GD["ImagerCF"]["OverS"],GD=self.GD)
        nch,npol=self.nch,self.npol
        ChanSel=sorted(list(set(self.VS.DicoMSChanMappingDegridding[self.VS.iCurrentMS].tolist())))
        for iFacet in sorted(self.DicoImager.keys()):

            SharedMemName="%s/Spheroidal.Facet_%3.3i"%(self.FacetDataCache,iFacet)
            SPhe=NpShared.GiveArray(SharedMemName)
            SpacialWeight=self.SpacialWeigth[iFacet]
            # Grid,_=Im2Grid.GiveGridTessel(Image,self.DicoImager,iFacet,self.NormImage,SPhe,SpacialWeight)
            # GridSharedMemName="%sModelGrid.Facet_%3.3i"%(self.IdSharedMem,iFacet)
            # NpShared.ToShared(GridSharedMemName,Grid)

            ModelFacet,_=Im2Grid.GiveModelTessel(Image,self.DicoImager,iFacet,self.NormImage,SPhe,SpacialWeight,ChanSel=ChanSel)
            ModelSharedMemName="%sModelImage.Facet_%3.3i"%(self.IdSharedMem,iFacet)

            NpShared.ToShared(ModelSharedMemName,ModelFacet)

    def ReinitDirty(self):
        """
        Reinitializes dirty map and weight buffers for the next round of residual calculation
        Postconditions:
        Resets the following:
            self.DicoGridMachine[iFacet]["Dirty"],
            self.DicoImager[iFacet]["SumWeights"],
            self.DicoImager[iFacet]["SumJones"]
            self.DicoImager[iFacet]["SumJonesChan"]
        """
        self.SumWeights.fill(0)
        self.IsDirtyInit=True
        self.HasFourierTransformed = False
        for iFacet in self.DicoGridMachine.keys():
            NX=self.DicoImager[iFacet]["NpixFacetPadded"]
            if "Dirty" in self.DicoGridMachine[iFacet]:
                self.DicoGridMachine[iFacet]["Dirty"][...] = 0
            else:
                GridName = "%sGridFacet.%3.3i" % (self.IdSharedMem, iFacet)
                ResidueGrid = NpShared.ToShared(GridName, np.zeros((self.VS.NFreqBands,self.npol,NX,NX),self.CType))
                self.DicoGridMachine[iFacet]["Dirty"]=ResidueGrid
            self.DicoImager[iFacet]["SumWeights"] = np.zeros((self.VS.NFreqBands,self.npol),np.float64)
            self.DicoImager[iFacet]["SumJones"]   = np.zeros((2,self.VS.NFreqBands),np.float64)
            self.DicoImager[iFacet]["SumJonesChan"]=[]
            for iMS in range(self.VS.nMS):
                MS=self.VS.ListMS[iMS]
                nVisChan=MS.ChanFreq.size
                self.DicoImager[iFacet]["SumJonesChan"].append(np.zeros((2,nVisChan),np.float64))

    def CalcDirtyImagesParallel(self,Weights=None,Parallel=True):
        """
        Grids a chunk of input visibilities onto many facets
        Visibility data is already in shared memory (packed there by VisServer.VisChunkToShared(),
        only the weights are passed as a string, since they refer to an mmap()d file.

        Post conditions:
        Sets the following normalization weights, as produced by the gridding process:
            self.DicoImager[iFacet]["SumWeights"]
            self.DicoImager[iFacet]["SumJones"]
            self.DicoImager[iFacet]["SumJonesChan"][self.VS.iCurrentMS]
        """
        # the input parameters are not actually used, see
        ## https://github.com/cyriltasse/DDFacet/issues/32#issuecomment-176072113
        import Queue

        NFacets=len(self.DicoImager.keys())

        procs = list() #list of processes that are running
        n_cpus = psutil.cpu_count() #ask the OS for the number of CPU's. Only tested without HT
        procinfo = psutil.Process() #this will be used to control CPU affinity

        if Parallel:
            qlimit = n_cpus*8
        else:
            qlimit=0

        m_work_queue = multiprocessing.Queue() #(maxsize=qlimit)
        m_result_queue = multiprocessing.JoinableQueue()

        PSFMode=False
        if self.DoPSF:
            #visIn.fill(1)
            PSFMode=True

        SpheNorm=self.SpheNorm

        #Paramaters for grid_worker execution (from Worker_Imager)
        Mode="Grid"
        FFTW_Wisdom=self.FFTW_Wisdom
        DicoImager=self.DicoImager
        IdSharedMem=self.IdSharedMem
        IdSharedMemData=self.IdSharedMemData
        FacetDataCache=self.FacetDataCache
        ChunkDataCache="file://" + self.VS.cache.dirname + "/"
        ApplyCal=self.ApplyCal
        SpheNorm=SpheNorm
        PSFMode=PSFMode
        NFreqBands=self.VS.NFreqBands
        Weights=Weights
        PauseOnStart=self.GD["Debugging"]["PauseGridWorkers"]
        DataCorrelationFormat=self.VS.StokesConverter.AvailableCorrelationProductsIds()
        ExpectedOutputStokes=self.VS.StokesConverter.RequiredStokesProductsIds()

        #Create a list of work items for the producer process
        NFacetsList = []
        for iFacet in xrange(NFacets):
           NFacetsList.append(iFacet)
        work_p = Process(target=work_producer, args=(m_work_queue, NFacetsList,))

        #loop for the number of deteccted CPU's
        for cpu in xrange(n_cpus):
                 p = Process(target=grid_worker, args=(m_work_queue,m_result_queue,
                                             self.GD,
                                             Mode,
                                             FFTW_Wisdom,
                                             DicoImager,
                                             IdSharedMem,
                                             IdSharedMemData,
                                             FacetDataCache,
                                             ChunkDataCache,
                                             ApplyCal,
                                             SpheNorm,
                                             PSFMode,
                                             NFreqBands,
                                             Weights,
                                             PauseOnStart,
                                            DataCorrelationFormat,
                                             ExpectedOutputStokes,))

                 procs.append(p)

        if Parallel:
            main_core=0 #CPU affinity placement

            #start a pinned work producer process
            work_p.start()
            procinfo.cpu_affinity([n_cpus-1])

            #start all processes and pin them each to a core
            for p in procs:
                p.start()
                procinfo.cpu_affinity([main_core])
                main_core+=1

        timer = ClassTimeIt.ClassTimeIt()
        print>> log, "starting gridding"

        pBAR = ProgressBar('white', width=50, block='=', empty=' ', Title="  Gridding ", HeaderSize=10, TitleSize=13)
        #pBAR.disable()
        pBAR.render(0, '%4i/%i' % (0, NFacets))
        iResult = 0

        NJobs=NFacets
        print>> log, "jobs %d" % NJobs

        if not Parallel:
            for p in procs:
                p.run()  # just run until all work is completed

        while iResult < NJobs:
            DicoResult = {}
            try:
                DicoResult = m_result_queue.get(True, 10)
            except Queue.Empty:
                pass
                print>> log, "checking for dead workers"
                #shoot the zombie process
                multiprocessing.active_children()
                # check for dead workers
                pids_to_restart = []
                for w in procs:
                    if not w.is_alive():
                       pids_to_restart[w]
                       raise RuntimeError, "a worker process has died on us with exit code %d. This is probably a bug." % w.exitcode

                       for id in pids_to_restart:
                           print>> log, "need to restart worker %d." % id
                           pass

            if len(DicoResult) !=0 and DicoResult["Success"]:
                iResult += 1
                m_result_queue.task_done() #call task_done on the queue
                iFacet=DicoResult["iFacet"]
                self.DicoImager[iFacet]["SumWeights"] += DicoResult["Weights"]
                self.DicoImager[iFacet]["SumJones"] += DicoResult["SumJones"]
                self.DicoImager[iFacet]["SumJonesChan"][self.VS.iCurrentMS] += DicoResult["SumJonesChan"]
            else:
                print>> log, "grid_worker: returned no result"

            NDone=iResult
            intPercent=int(100*  NDone / float(NFacets))
            pBAR.render(intPercent, '%4i/%i' % (NDone,NFacets))


        if Parallel:
            #if all work is done send poinsed pill to workers
            if iResult == NJobs:
                for p in procs:
                    m_work_queue.put("POISON-E")

                #join and close queues
                m_result_queue.join()
                m_work_queue.close()
                m_result_queue.close()

                #join producer process
                work_p.join()

                #join producer process
                for p in procs:
                    p.join()

                print>> log, "gridding finished in %s" % timer.timehms()

                return True

    def FourierTransform(self,doStack=False,Parallel=True):
        '''
        Fourier transforms the individual facet grids
        Postconditions:
            self.DicoGridMachine[iFacet]["Dirty"] is created (or added to if doStack is true)
        Args:
            doStack: Add fourier transformed facets to existing facet grids.
            Parallel: Calls FFTW in parallel
        '''
        import Queue

        NFacets = len(self.DicoImager.keys())

        procs = list() #list of processes that are running
        n_cpus = psutil.cpu_count() #ask the OS for the number of CPU's. Only tested without HT
        procinfo = psutil.Process() #this will be used to control CPU affinity

        if Parallel:
            qlimit = n_cpus*8
        else:
            qlimit=0

        m_work_queue = multiprocessing.Queue()#(maxsize=qlimit)
        m_result_queue = multiprocessing.JoinableQueue()

        Mode="FourierTransform"
        FFTW_Wisdom=self.FFTW_Wisdom
        DicoImager=self.DicoImager
        IdSharedMem=self.IdSharedMem
        IdSharedMemData=self.IdSharedMemData
        FacetDataCache=self.FacetDataCache
        ChunkDataCache="file://" + self.VS.cache.dirname + "/"
        ApplyCal=self.ApplyCal
        SpheNorm=self.SpheNorm
        PSFMode=self.DoPSF
        NFreqBands=self.VS.NFreqBands
        PauseOnStart=self.GD["Debugging"]["PauseGridWorkers"]
        DataCorrelationFormat=self.VS.StokesConverter.AvailableCorrelationProductsIds()
        ExpectedOutputStokes=self.VS.StokesConverter.RequiredStokesProductsIds()

        #Create a list of work items for the producer process
        NFacetsList = []
        for iFacet in xrange(NFacets):
           NFacetsList.append(iFacet)
        work_p = Process(target=work_producer, args=(m_work_queue, NFacetsList,))

        #loop for the number of deteccted CPU's
        for cpu in xrange(n_cpus):
                 p = Process(target=FFT_worker, args=(m_work_queue,m_result_queue,
                                             self.GD,
                                             Mode,
                                             FFTW_Wisdom,
                                             DicoImager,
                                             IdSharedMem,
                                             IdSharedMemData,
                                             FacetDataCache,
                                             ChunkDataCache,
                                             ApplyCal,
                                             SpheNorm,
                                             PSFMode,
                                             NFreqBands,
                                             PauseOnStart,
                                             DataCorrelationFormat,
                                             ExpectedOutputStokes,))

                 procs.append(p)

        if Parallel:
            main_core=0 #CPU affinity placement

            #start a pinned work producer process
            work_p.start()
            procinfo.cpu_affinity([n_cpus-1])
            #start all processes and pin them each to a core
            for p in procs:
                p.start()
                procinfo.cpu_affinity([main_core])
                main_core+=1

        timer = ClassTimeIt.ClassTimeIt()
        print>> log, "Fourier transforming facet grids"

        pBAR = ProgressBar('white', width=50, block='=', empty=' ', Title="  Fourier Transforming ", HeaderSize=10, TitleSize=13)
        #pBAR.disable()
        pBAR.render(0, '%4i/%i' % (0, NFacets))
        iResult = 0

        if not Parallel:
            for p in procs:
                p.run()  # just run until all work is completed

        NJobs = NFacets

        while iResult < NJobs:
            DicoResult = {}
            try:
                DicoResult = m_result_queue.get(True, 10)
            except Queue.Empty:
                pass
                print>> log, "checking for dead workers"
                #shoot the zombie process
                multiprocessing.active_children()
                # check for dead workers
                pids_to_restart = []
                for w in procs:
                    if not w.is_alive():
                       pids_to_restart[w]
                       raise RuntimeError, "a worker process has died on us with exit code %d. This is probably a bug." % w.exitcode

                       for id in pids_to_restart:
                           print>> log, "need to restart worker %d." % id
                           pass

            if len(DicoResult) !=0 and DicoResult["Success"]:
                iResult += 1
                m_result_queue.task_done()
                iFacet = DicoResult["iFacet"]
                GridName = "%sGridFacet.%3.3i" % (self.IdSharedMem, iFacet)
                Grid = NpShared.GiveArray(GridName)

                if (doStack == True) & ("Dirty" in self.DicoGridMachine[iFacet].keys()):
                    self.DicoGridMachine[iFacet]["Dirty"] += Grid
                else:
                    self.DicoGridMachine[iFacet]["Dirty"] = Grid
            else:
                print>> log, "FFT_worker: returned no result"

            NDone = iResult
            intPercent = int(100 * NDone / float(NFacets))
            pBAR.render(intPercent, '%4i/%i' % (NDone, NFacets))


        if Parallel:
            #if all work is done send poinsed pill to workers
            if iResult == NJobs:
                for p in procs:
                    m_work_queue.put("POISON-E")

                #join and close queues
                m_result_queue.join()
                m_work_queue.close()
                m_result_queue.close()

                #join producer process
                work_p.join()

                #join producer process
                for p in procs:
                    p.join()

        print>> log, "Fourier transforms finished in %s" % timer.timehms()

        return True

    def GiveVisParallel(self,times,uvwIn,visIn,flag,A0A1,ModelImage,Parallel=True):
        """
        Degrids visibilities from model image. The model image is unprojected into many facets
        before degridding and subtracting each of the model facets contributions from the residual image.
        Preconditions: the dirty image buffers should be cleared before calling the predict and regridding methods
        to construct a new residual map
        Args:
            times:
            uvwIn:
            visIn:
            flag:
            A0A1:
            ModelImage:
        """
        import Queue

        procs = list() #list of processes that are running
        n_cpus = psutil.cpu_count() #ask the OS for the number of CPU's. Only tested without HT
        procinfo = psutil.Process() #this will be used to control CPU affinity

        if Parallel:
            qlimit = n_cpus*8
        else:
            qlimit=0

        m_work_queue = multiprocessing.Queue() # (maxsize=qlimit)
        m_result_queue = multiprocessing.JoinableQueue()

        print>> log, "Model image to facets ..."
        self.ImToGrids(ModelImage)

        NFacets = len(self.DicoImager.keys())
        # ListModelImage=[]
        # for iFacet in self.DicoImager.keys():
        #     ListModelImage.append(self.DicoImager[iFacet]["ModelFacet"])
        # NpShared.PackListArray("%sModelImage"%self.IdSharedMem,ListModelImage)
        # del(ListModelImage)
        # for iFacet in self.DicoImager.keys():
        #     del(self.DicoImager[iFacet]["ModelFacet"])

        print>> log, "    ... done"

        NSemaphores = 3373
        ListSemaphores = ["%sSemaphore%4.4i" % (self.IdSharedMem, i) for i in range(NSemaphores)]
        _pyGridderSmearPols.pySetSemaphores(ListSemaphores)

        SpheNorm=self.SpheNorm

        #Paramaters for grid_worker execution (from Worker_Imager)
        Mode="DeGrid"
        FFTW_Wisdom=self.FFTW_Wisdom
        DicoImager=self.DicoImager
        IdSharedMem=self.IdSharedMem
        IdSharedMemData=self.IdSharedMemData
        FacetDataCache=self.FacetDataCache
        ChunkDataCache="file://" + self.VS.cache.dirname + "/"
        ApplyCal=self.ApplyCal
        NFreqBands=self.VS.NFreqBands
        PauseOnStart=self.GD["Debugging"]["PauseGridWorkers"]
        DataCorrelationFormat=self.VS.StokesConverter.AvailableCorrelationProductsIds()
        ExpectedOutputStokes=self.VS.StokesConverter.RequiredStokesProductsIds()

        #Create a list of work items for the producer process
        NFacetsList = []
        for iFacet in xrange(NFacets):
           NFacetsList.append(iFacet)
        work_p = Process(target=work_producer, args=(m_work_queue, NFacetsList,))

        for cpu in xrange(n_cpus):
            p = Process(target=degrid_worker, args=(m_work_queue,m_result_queue,
                                             self.GD,
                                             Mode,
                                             FFTW_Wisdom,
                                             DicoImager,
                                             IdSharedMem,
                                             IdSharedMemData,
                                             FacetDataCache,
                                             ChunkDataCache,
                                             ApplyCal,
                                             SpheNorm,
                                             NFreqBands,
                                             PauseOnStart,
                                             DataCorrelationFormat,
                                             ExpectedOutputStokes,
                                             ListSemaphores,))

            procs.append(p)

        if Parallel:
            main_core=0 #CPU affinity placement

            #start a pinned work producer process
            work_p.start()
            procinfo.cpu_affinity([n_cpus-1])

            #start all processes and pin them each to a core
            for p in procs:
                p.start()
                procinfo.cpu_affinity([main_core])
                main_core+=1

        timer = ClassTimeIt.ClassTimeIt()
        print>> log, "starting degridding"

        pBAR = ProgressBar('white', width=50, block='=', empty=' ', Title="DeGridding ", HeaderSize=10, TitleSize=13)
        # pBAR.disable()
        pBAR.render(0, '%4i/%i' % (0, NFacets))
        iResult = 0

        if not Parallel:
            for p in procs:
                p.run()  # just run until all work is completed

        NJobs=NFacets

        while iResult < NJobs:
            DicoResult = {}
            try:
                DicoResult = m_result_queue.get(True, 10)
            except Queue.Empty:
                pass
                print>> log, "checking for dead workers"
                #shoot the zombie process
                multiprocessing.active_children()
                # check for dead workers
                pids_to_restart = []
                for w in procs:
                    if not w.is_alive():
                       pids_to_restart[w]
                       raise RuntimeError, "a worker process has died on us with exit code %d. This is probably a bug." % w.exitcode

                       for id in pids_to_restart:
                           print>> log, "need to restart worker %d." % id
                           pass

            if len(DicoResult) !=0 and DicoResult["Success"]:
                iResult += 1
                m_result_queue.task_done() #call task_done on the queue
                NDone = iResult
                intPercent = int(100 * NDone / float(NFacets))
                pBAR.render(intPercent, '%4i/%i' % (NDone, NFacets))


        if Parallel:
            #if all work is done send poinsed pill to workers
            if iResult == NJobs:
                for p in procs:
                    m_work_queue.put("POISON-E")

                #join and close queues
                m_result_queue.join()
                m_work_queue.close()
                m_result_queue.close()

                #join producer process
                work_p.join()

                #join producer process
                for p in procs:
                    p.join()

        _pyGridderSmearPols.pyDeleteSemaphore(ListSemaphores)

        NpShared.DelAll("%sc" % (self.IdSharedMemData))
        print>> log, "degridding finished in %s" % timer.timehms()

        return True




##########################################
####### Workers
##########################################
import os
import signal
import Queue


class WorkerImager(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 GD,
                 Mode="Init",
                 FFTW_Wisdom=None,
                 DicoImager=None,
                 IdSharedMem=None,
                 IdSharedMemData=None,
                 FacetDataCache=None,
                 ChunkDataCache=None,
                 ApplyCal=False,
                 SpheNorm=True,
                 PSFMode=False,
                 CornersImageTot=None,
                 NFreqBands=1,
                 Weights=None,
                 PauseOnStart=False,
                 DataCorrelationFormat=[5,6,7,8],
                 ExpectedOutputStokes=[1],
                 ListSemaphores=None):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.Mode=Mode
        self.FFTW_Wisdom=FFTW_Wisdom
        self.GD=GD
        self.DicoImager=DicoImager
        self.IdSharedMem=IdSharedMem
        self.IdSharedMemData=IdSharedMemData
        self.FacetDataCache = FacetDataCache
        self.ChunkDataCache = ChunkDataCache
        self.Apply_killMS=(GD["DDESolutions"]["DDSols"]!="")&(GD["DDESolutions"]["DDSols"]!=None)
        self.Apply_Beam=(GD["Beam"]["BeamModel"]!=None)
        self.ApplyCal=(self.Apply_killMS)|(self.Apply_Beam)
        self.SpheNorm=SpheNorm
        self.PSFMode=PSFMode
        self.CornersImageTot = CornersImageTot
        self.NFreqBands = NFreqBands
        self._pause_on_start = PauseOnStart
        self.DataCorrelationFormat = DataCorrelationFormat
        self.ExpectedOutputStokes = ExpectedOutputStokes
        self.ListSemaphores = ListSemaphores
        self.Weights = Weights

    def shutdown(self):
        self.exit.set()

    def GiveGM(self,iFacet):
        """
        Factory: Initializes a gridding machine for this facet
        Args:
            iFacet: index of facet

        Returns:
            grid machine instance
        """
        GridMachine=ClassDDEGridMachine.ClassDDEGridMachine(self.GD,#RaDec=self.DicoImager[iFacet]["RaDec"],
                                                            self.DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
                                                            self.DicoImager[iFacet]["DicoConfigGM"]["Npix"],
                                                            lmShift=self.DicoImager[iFacet]["lmShift"],
                                                            IdSharedMem=self.IdSharedMem,
                                                            IdSharedMemData=self.IdSharedMemData,
                                                            FacetDataCache=self.FacetDataCache,
                                                            ChunkDataCache=self.ChunkDataCache,
                                                            IDFacet=iFacet,
                                                            SpheNorm=self.SpheNorm,
                                                            NFreqBands=self.NFreqBands,
                                                            DataCorrelationFormat=self.DataCorrelationFormat,
                                                            ExpectedOutputStokes=self.ExpectedOutputStokes,
                                                            ListSemaphores=self.ListSemaphores)
        return GridMachine

    def GiveDicoJonesMatrices(self):
        DicoJonesMatrices=None

        if self.ApplyCal:
            DicoJonesMatrices={}

        if self.Apply_killMS:
            DicoJones_killMS=NpShared.SharedToDico("%sJonesFile_killMS"%self.IdSharedMemData)
            DicoJonesMatrices["DicoJones_killMS"]=DicoJones_killMS
            DicoJonesMatrices["DicoJones_killMS"]["MapJones"]=NpShared.GiveArray("%sMapJones_killMS"%self.IdSharedMemData)
            DicoClusterDirs_killMS=NpShared.SharedToDico("%sDicoClusterDirs_killMS"%self.IdSharedMemData)
            DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"]=DicoClusterDirs_killMS
            DicoJonesMatrices["DicoJones_killMS"]["AlphaReg"]=None

        if self.Apply_Beam:
            DicoJones_Beam=NpShared.SharedToDico("%sJonesFile_Beam"%self.IdSharedMemData)
            DicoJonesMatrices["DicoJones_Beam"]=DicoJones_Beam
            DicoJonesMatrices["DicoJones_Beam"]["MapJones"]=NpShared.GiveArray("%sMapJones_Beam"%self.IdSharedMemData)
            DicoClusterDirs_Beam=NpShared.SharedToDico("%sDicoClusterDirs_Beam"%self.IdSharedMemData)
            DicoJonesMatrices["DicoJones_Beam"]["DicoClusterDirs"]=DicoClusterDirs_Beam

        return DicoJonesMatrices

    def init(self, iFacet):
        """
        Initializes the gridding machines (primarily the convolution kernels) and
        the weighting grid for a particular facet.
        Post conditions:
            Griders initialized
            Tesselation mask stored in sSpacialWeight.Facet_%3.3i. This should be used to downweight the areas of
            padding (overlap) in the reprojected image. This makes the transition between edges less noticible.

        Returns:
            Dictionary of {Success and iFacet, the facet identifier}
        """
        #in the basic tesselation scheme the facets are all in a grid layout, so we want to use all of the area
        # except the padding. Weight down the padding and smoothen the edges:
        NpixPadded = self.DicoImager[iFacet]["NpixFacetPadded"]
        Npix = self.DicoImager[iFacet]["NpixFacet"]
        maskOffset = (NpixPadded - Npix)/2
        mask = np.zeros((NpixPadded,NpixPadded))
        mask[maskOffset:maskOffset+Npix,maskOffset:maskOffset+Npix] = 1
        GaussPars = (10, 10, 0)
        SpacialWeigth = ModFFTW.ConvolveGaussian(mask.reshape(1,1,NpixPadded,NpixPadded),
                                                 CellSizeRad=1,
                                                 GaussPars=[GaussPars]).reshape(NpixPadded,NpixPadded)
        NameSpacialWeigth = "%sSpacialWeight.Facet_%3.3i" % (self.FacetDataCache, iFacet)
        NpShared.ToShared(NameSpacialWeigth, SpacialWeigth)
        # Initialize a grid machine per facet:
        self.GiveGM(iFacet)
        self.result_queue.put({"Success": True, "iFacet": iFacet})

    def grid(self, iFacet):
        """
        Grids the data currently housed in shared memory

        Returns:
            Dictionary of gridder output products and weights
        """
        GridMachine = self.GiveGM(iFacet)
        DATA = NpShared.SharedToDico("%sDicoData" % self.IdSharedMemData)
        uvwThis = DATA["uvw"]
        visThis = DATA["data"]
        flagsThis = DATA["flags"]
        times = DATA["times"]
        A0 = DATA["A0"]
        A1 = DATA["A1"]
        A0A1 = A0, A1
        W = NpShared.GiveArray("file://"+self.Weights)
        freqs = DATA["freqs"]
        ChanMapping = DATA["ChanMapping"]

        DecorrMode = self.GD["DDESolutions"]["DecorrMode"]
        if ('F' in DecorrMode) | ("T" in DecorrMode):
            uvw_dt = DATA["uvw_dt"]
            DT, Dnu = DATA["MSInfos"]
            GridMachine.setDecorr(uvw_dt, DT, Dnu, SmearMode=DecorrMode)

        GridName = "%sGridFacet.%3.3i" % (self.IdSharedMem, iFacet)
        Grid = NpShared.GiveArray(GridName)
        DicoJonesMatrices = self.GiveDicoJonesMatrices()
        # if iFacet == 12:
        #     print "before put ##grid", Grid.sum(dtype=np.float64)

        GridMachine.put(times, uvwThis, visThis, flagsThis, A0A1, W,
                        DoNormWeights=False,
                        DicoJonesMatrices=DicoJonesMatrices,
                        freqs=freqs, DoPSF=self.PSFMode,
                        ChanMapping=ChanMapping,
                        ResidueGrid=Grid)
        # if iFacet == 12:
        #     print "after put ##grid",Grid.sum(dtype=np.float64)

        Sw = GridMachine.SumWeigths.copy()
        SumJones = GridMachine.SumJones.copy()
        SumJonesChan = GridMachine.SumJonesChan.copy()
        del (GridMachine)

        self.result_queue.put(
            {"Success": True, "iFacet": iFacet, "Weights": Sw, "SumJones": SumJones,
             "SumJonesChan": SumJonesChan})

    def degrid(self, iFacet):
        """
        Degrids input model facets and subtracts model visibilities from residuals. Assumes degridding input data is
        placed in DATA shared memory dictionary.
        Returns:
            Dictionary of success and facet identifier
        """
        GridMachine = self.GiveGM(iFacet)
        DATA = NpShared.SharedToDico("%sDicoData" % self.IdSharedMemData)
        uvwThis = DATA["uvw"]
        visThis = DATA["data"]
        flagsThis = DATA["flags"]
        times = DATA["times"]
        A0 = DATA["A0"]
        A1 = DATA["A1"]
        A0A1 = A0, A1
        freqs = DATA["freqs"]
        ChanMapping = DATA["ChanMappingDegrid"]

        DicoJonesMatrices = self.GiveDicoJonesMatrices()
        ModelSharedMemName = "%sModelImage.Facet_%3.3i" % (self.IdSharedMem, iFacet)
        ModelGrid = NpShared.GiveArray(ModelSharedMemName)

        DecorrMode = self.GD["DDESolutions"]["DecorrMode"]
        if ('F' in DecorrMode) | ("T" in DecorrMode):
            uvw_dt = DATA["uvw_dt"]
            DT, Dnu = DATA["MSInfos"]
            GridMachine.setDecorr(uvw_dt, DT, Dnu, SmearMode=DecorrMode)

        vis = GridMachine.get(times, uvwThis, visThis, flagsThis, A0A1, ModelGrid, ImToGrid=False,
                              DicoJonesMatrices=DicoJonesMatrices, freqs=freqs, TranformModelInput="FT",
                              ChanMapping=ChanMapping)

        self.result_queue.put({"Success": True, "iFacet": iFacet})

    def fourierTransform(self, iFacet):
        """
        Fourier transforms the grids currently housed in shared memory
        Precondition:
            Should be called after all data has been gridded
        Returns:
            Dictionary of success and facet identifier
        """
        GridMachine = self.GiveGM(iFacet)
        GridName = "%sGridFacet.%3.3i" % (self.IdSharedMem, iFacet)
        Grid = NpShared.GiveArray(GridName)
        Dirty = GridMachine.GridToIm(Grid)
        Grid[...] = Dirty[...]
        self.result_queue.put({"Success": True, "iFacet": iFacet})

    def run(self):
        """
        Runs a task in parallel
        Post conditions:
            Results of the task is placed on self.result_queue
        Accepts self.Mode to be one of "Init", "Grid" or "DeGrid"
        """
        # pause self in debugging mode
        #if self._pause_on_start:
        #    os.kill(os.getpid(),signal.SIGSTOP)

        #print>> log, "Worker_Imager: worker started"
        while not self.kill_received: #and not self.work_queue.empty():
            try:
                iFacet = self.work_queue.get(True, 0.5)
            except Queue.Empty:
                break
                print>> log, "Worker_Imager: empty worker queue"
            else:
                if self.FFTW_Wisdom!=None:
                    pyfftw.import_wisdom(self.FFTW_Wisdom)

                if self.Mode == "Init":
                   self.init(iFacet)
                elif self.Mode == "Grid":
                   self.grid(iFacet)
                elif self.Mode == "DeGrid":
                   self.degrid(iFacet)
                elif self.Mode == "FourierTransform":
                   self.fourierTransform(iFacet)
                else:
                   print>> log, "Strange message"








