import multiprocessing
import time
from multiprocessing import Process
import Queue
import psutil
import ClassDDEGridMachine
import numpy as np
import ClassCasaImage
import pyfftw
import cPickle
from matplotlib.path import Path
import pylab
import numpy.random
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyLogger
from DDFacet.ToolsDir import ModCoord
from DDFacet.Array import NpShared
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import Multiprocessing
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.Imager.ClassImToGrid import ClassImToGrid
from DDFacet.Other import MyLogger
from DDFacet.cbuild.Gridder import _pyGridderSmearPols

log = MyLogger.getLogger("ClassFacetImager")
MyLogger.setSilent("MyLogger")


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
                 # ParsetFile="ParsetNew.txt",
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
                 ApplyCal=False,
                 cfdict=None):

        # IdSharedMem is used to identify structures in shared memory
        # used by this FacetMachine
        self.IdSharedMem = IdSharedMem
        self.HasFourierTransformed = False
        # IdSharedMemData is used to identify "global" structures in
        # shared memory such as DicoData
        self.IdSharedMemData = IdSharedMemData or IdSharedMem
        # dictionary of precomputed convolution functions
        self._cfdict = cfdict

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
        if Sols is not None:
            self.setSols(Sols)

        self.PointingID = PointingID
        self.VS, self.GD = VS, GD
        self.npol = self.VS.StokesConverter.NStokesInImage()
        self.Parallel = Parallel
        DicoConfigGM = {}
        self.DicoConfigGM = DicoConfigGM
        self.DoPSF = DoPSF
        # self.MDC.setFreqs(ChanFreq)
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
        for iFacet in xrange(lFacet.size):
            self.DicoImager[iFacet] = {}

        # print "Append2"; self.IM.CI.E.clear()

        self.FacetCat = np.zeros(
            (lFacet.size,),
            dtype=[('Name', '|S200'),
                   ('ra', np.float),
                   ('dec', np.float),
                   ('SumI', np.float),
                   ("Cluster", int),
                   ("l", np.float),
                   ("m", np.float),
                   ("I", np.float)])

        self.FacetCat = self.FacetCat.view(np.recarray)
        self.FacetCat.I = 1
        self.FacetCat.SumI = 1

        for iFacet in xrange(lFacet.size):
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
            # rac,decc=self.DicoImager[iFacet]["RaDec"]
            l0, m0 = self.DicoImager[iFacet]["l0m0"]
            diam = self.DicoImager[iFacet]["lmDiam"]
            dl = np.array([-1, 1, 1, -1, -1])*diam
            dm = np.array([-1, -1, 1, 1, -1])*diam
            l = ((dl.flatten()+l0)).tolist()
            m = ((dm.flatten()+m0)).tolist()

            x = []
            y = []
            for iPoint in xrange(len(l)):
                xp, yp = self.CoordMachine.lm2radec(np.array(
                    [l[iPoint]]), np.array([m[iPoint]]))
                x.append(xp)
                y.append(yp)

            x = np.array(x)  # +[x[2]])
            y = np.array(y)  # +[y[2]])

            x *= 180/np.pi
            y *= 180/np.pi

            for iline in xrange(x.shape[0]-1):
                x0 = x[iline]
                y0 = y[iline]
                x1 = x[iline+1]
                y1 = y[iline+1]
                f.write("line(%f,%f,%f,%f) # line=0 0\n" % (x0, y0, x1, y1))

        f.close()

    # ############### Initialisation #####################

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
        if self.IsDDEGridMachineInit:
            return
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
        path, valid = self.VS.maincache.checkCache("FFTW_Wisdom", dict(shape=self.PaddedGridShape))
        if not valid:
            print>>log, "Set fftw widsdom for shape = %s" % str(self.PaddedGridShape)
            a = np.random.randn(*(self.PaddedGridShape)) \
                + 1j*np.random.randn(*(self.PaddedGridShape))
            FM = ModFFTW.FFTW_2Donly(self.PaddedGridShape, np.complex64)
            FM.fft(a)  # this is never used
            self.FFTW_Wisdom = pyfftw.export_wisdom()
            cPickle.dump(self.FFTW_Wisdom, file(path, "w"))
            self.VS.maincache.saveCache("FFTW_Wisdom")
        else:
            print>>log, "Loading cached fftw wisdom from %s" % path
            self.FFTW_Wisdom = cPickle.load(file(path))

            # for iFacet in sorted(self.DicoImager.keys()):
        #     A = ModFFTW.GiveFFTW_aligned(self.PaddedGridShape, np.complex64)
        #     NpShared.ToShared("%sFFTW.%i" % (self.IdSharedMem, iFacet), A)

    @staticmethod
    def _init_w_worker_tessel(jobitem, GD, DicoImager,
                             SpheNorm, NFreqBands, DataCorrelationFormat, ExpectedOutputStokes, CornersImageTot):
        """Worker method of InitParallel"""
        iFacet, NameSpacialWeight, NameWTerm, NameSphe = jobitem

        # Create smoothned facet tessel mask:
        Npix = DicoImager[iFacet]["NpixFacetPadded"]
        l0, l1, m0, m1 = DicoImager[iFacet]["lmExtentPadded"]
        X, Y = np.mgrid[l0:l1:Npix * 1j, m0:m1:Npix * 1j]
        XY = np.dstack((X, Y))
        XY_flat = XY.reshape((-1, 2))
        vertices = DicoImager[iFacet]["Polygon"]
        mpath = Path(vertices)  # the vertices of the polygon
        mask_flat = mpath.contains_points(XY_flat)
        mask = mask_flat.reshape(X.shape)
        mpath = Path(CornersImageTot)
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
        NpShared.ToShared(NameSpacialWeight, SpacialWeigth)

        # Initialize a grid machine per facet:
        ClassDDEGridMachine.ClassDDEGridMachine(
            GD, DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
            DicoImager[iFacet]["DicoConfigGM"]["Npix"],
            DicoImager[iFacet]["lmShift"],
            iFacet, SpheNorm, NFreqBands,
            DataCorrelationFormat, ExpectedOutputStokes, ListSemaphores=None,
            wterm=NameWTerm, sphe=NameSphe,
            compute_cf=True)

        return {"iFacet": iFacet}

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
        NFacets = len(self.DicoImager.keys())

        # check if spacial weights are cached
        cachepath, cachevalid = self.VS.maincache.checkCache(
            "FacetData",
            dict(
                ImagerCF=self.GD["ImagerCF"],
                ImagerMainFacet=self.GD["ImagerMainFacet"]),
            directory=True)

        if not cachevalid:
            joblist = [ (iFacet,
                            self.VS.maincache.getCacheURL("SW", facet=iFacet),
                            self.VS.maincache.getCacheURL("WTerm", facet=iFacet),
                            self.VS.maincache.getCacheURL("Sphe", facet=iFacet)) for iFacet in xrange(NFacets) ]

            procpool = Multiprocessing.ProcessPool(self.GD)
            procpool.runjobs(joblist, title="Init W", target=self._init_w_worker_tessel,
                                kwargs=dict(GD=self.GD,
                                            DicoImager=self.DicoImager,
                                            SpheNorm=self.SpheNorm,
                                            NFreqBands=self.VS.NFreqBands,
                                            DataCorrelationFormat=self.VS.StokesConverter.AvailableCorrelationProductsIds(),
                                            ExpectedOutputStokes=self.VS.StokesConverter.RequiredStokesProductsIds(),
                                            CornersImageTot=self.CornersImageTot))
            self.VS.maincache.saveCache("FacetData")
        else:
            print>>log,"using W kernels from cache %s"%cachepath
        # now load cached spatial weights, wterms and spheroidals from cache and lock them into memory
        self._wterms = {}
        self._sphes = {}
        for iFacet in sorted(self.DicoImager.keys()):
            NameSpacialWeigth = self.VS.maincache.getCacheURL("SW", facet=iFacet)
            SpacialWeigth = NpShared.GiveArray(NameSpacialWeigth)
            NpShared.Lock(SpacialWeigth)
            self.SpacialWeigth[iFacet] = SpacialWeigth
            wterm = NpShared.GiveArray( self.VS.maincache.getCacheURL("WTerm", facet=iFacet) )
            sphe = NpShared.GiveArray( self.VS.maincache.getCacheURL("Sphe", facet=iFacet) )
            # temporary: to see if RAM is faster
            wterm = wterm.copy()
            sphe = sphe.copy()
            # store in dict
            self._wterms[iFacet] = wterm
            self._sphes[iFacet] = sphe
        print>> log, "W kernels loaded and locked into memory"

        return True

    def setCasaImage(self, ImageName=None, Shape=None, Freqs=None, Stokes=["I"]):
        if ImageName is None:
            ImageName = self.ImageName

        if Shape is None:
            Shape = self.OutImShape
        self.CasaImage = ClassCasaImage.ClassCasaimage(
            ImageName, Shape, self.Cell, self.MainRaDec, Freqs=Freqs, Stokes=Stokes)

    def ToCasaImage(self, ImageIn, Fits=True, ImageName=None,
                    beam=None, beamcube=None, Freqs=None, Stokes=["I"]):
        self.setCasaImage(ImageName=ImageName, Shape=ImageIn.shape,
                          Freqs=Freqs, Stokes=Stokes)

        self.CasaImage.setdata(ImageIn, CorrT=True)

        if Fits:
            self.CasaImage.ToFits()
            if beam is not None:
                self.CasaImage.setBeam(beam, beamcube=beamcube)
        self.CasaImage.close()
        self.CasaImage = None

    def GiveEmptyMainField(self):
        """
        Gives empty image of the correct shape to act as buffer for e.g. the stitching process
        Returns:
            ndarray of type complex
        """
        return np.zeros(self.OutImShape, dtype=self.stitchedType)

    def putChunk(self, *args, **kwargs):
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

        self.CalcDirtyImagesParallel(*args, **kwargs)
        self.SetLogModeSubModules("Loud")

    def getChunk(self, *args, **kwargs):
        self.SetLogModeSubModules("Silent")
        if self.Parallel:
            kwargs["Parallel"] = True
            self.GiveVisParallel(*args, **kwargs)
        else:
            kwargs["Parallel"] = False
            self.GiveVisParallel(*args, **kwargs)
        self.SetLogModeSubModules("Loud")

    def FacetsToIm(self, NormJones=False, psf=False):
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
            psf: if True (and PSF grids are available), also computes PSF terms


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
            self.FourierTransform()
            self.HasFourierTransformed = True
        _, npol, Npix, Npix = self.OutImShape
        DicoImages = {}
        DicoImages["freqs"] = {}

        DoCalcNormData = False
        if (NormJones) & (self.NormData is None):
            DoCalcNormData = True

        # Assume all facets have the same weight sums.
        # Store the normalization weights for reference
        DicoImages["SumWeights"] = np.zeros((self.VS.NFreqBands, self.npol), np.float64)
        for band, channels in enumerate(self.VS.FreqBandChannels):
            DicoImages["freqs"][band] = channels
            DicoImages["SumWeights"][band] = self.DicoImager[0]["SumWeights"][band]
        DicoImages["WeightChansImages"] = DicoImages["SumWeights"] / np.sum(DicoImages["SumWeights"])

        # Build a residual image consisting of multiple continuum bands
        if self.NormImage is None:
            self.NormImage = self.BuildFacetNormImage()
            self.NormImageReShape = self.NormImage.reshape([1, 1, self.NormImage.shape[0], self.NormImage.shape[1]])
        self.stitchedResidual = self.FacetsToIm_Channel("Dirty")
        if DoCalcNormData:
            self.NormData = self.FacetsToIm_Channel("Jones-amplitude")

        # Normalize each of the continuum bands of the combined
        # residual by the weights that contributed to that band:
        if self.VS.MultiFreqMode:
            WBAND = np.array([DicoImages["SumWeights"][Channel]
                          for Channel in xrange(self.VS.NFreqBands)])
            # sum frequency contribution to weights per correlation
            WBAND /= np.sum(WBAND, axis=0)
            WBAND = np.float32(WBAND.reshape((self.VS.NFreqBands, npol, 1, 1)))
            # weight each of the cube slices and average
            self.MeanResidual = np.sum(self.stitchedResidual * WBAND, axis=0).reshape((1, npol, Npix, Npix))
        else:
            # if there are no bands in the continuum image
            # then the mean image is just the same
            self.MeanResidual = self.stitchedResidual.copy()
            WBAND = 1
        if psf and self._psf_grids:
            print>>log, "  Build PSF facet-slices "
            self.DicoPSF = {}
            for iFacet in self.DicoGridMachine.keys():
                # first normalize by spheriodals - these
                # facet psfs will be used in deconvolution per facet
                SPhe = self._sphes[iFacet]
                nx = SPhe.shape[0]
                SPhe = SPhe.reshape((1, 1, nx, nx)).real
                self.DicoPSF[iFacet] = {}
                self.DicoPSF[iFacet]["PSF"] = self._psf_grids[iFacet].real.copy()
                self.DicoPSF[iFacet]["PSF"] /= SPhe
                #self.DicoPSF[iFacet]["PSF"][SPhe < 1e-2] = 0
                self.DicoPSF[iFacet]["l0m0"] = self.DicoImager[iFacet]["l0m0"]
                self.DicoPSF[iFacet]["pixCentral"] = self.DicoImager[iFacet]["pixCentral"]
                self.DicoPSF[iFacet]["lmSol"] = self.DicoImager[iFacet]["lmSol"]

                nch, npol, n, n = self.DicoPSF[iFacet]["PSF"].shape
                PSFChannel = np.zeros((nch, npol, n, n), self.stitchedType)
                for ch in xrange(nch):
                    self.DicoPSF[iFacet]["PSF"][ch][SPhe[0] < 1e-2] = 0
                    self.DicoPSF[iFacet]["PSF"][ch][0] = self.DicoPSF[iFacet]["PSF"][ch][0].T[::-1, :]
                    SumJonesNorm = self.DicoImager[iFacet]["SumJonesNorm"][ch]
                    # normalize to bring back transfer
                    # functions to approximate convolution
                    self.DicoPSF[iFacet]["PSF"][ch] /= np.sqrt(SumJonesNorm)
                    for pol in xrange(npol):
                        ThisSumWeights = self.DicoImager[iFacet]["SumWeights"][ch][pol]
                        # normalize the response per facet
                        # channel if jones corrections are enabled
                        self.DicoPSF[iFacet]["PSF"][ch][pol] /= ThisSumWeights
                    PSFChannel[ch, :, :, :] = self.DicoPSF[iFacet]["PSF"][ch][:, :, :]

                W = DicoImages["WeightChansImages"]
                W = np.float32(W.reshape((self.VS.NFreqBands, npol, 1, 1)))
                # weight each of the cube slices and average
                MeanPSF = np.sum(PSFChannel * W, axis=0).reshape((1, npol, n, n))
                self.DicoPSF[iFacet]["MeanPSF"] = MeanPSF

            DicoVariablePSF = self.DicoPSF
            NFacets = len(DicoVariablePSF.keys())

            if self.GD["ImagerMainFacet"]["Circumcision"]:
                NPixMin = self.GD["ImagerMainFacet"]["Circumcision"]
                # print>>log,"using explicit Circumcision=%d"%NPixMin
            else:
                NPixMin = 1e6
                for iFacet in sorted(DicoVariablePSF.keys()):
                    _, npol, n, n = DicoVariablePSF[iFacet]["PSF"].shape
                    if n < NPixMin:
                        NPixMin = n

                NPixMin = int(NPixMin/self.GD["ImagerMainFacet"]["Padding"])
                if not NPixMin % 2:
                    NPixMin += 1
                    # print>>log,"using computed Circumcision=%d"%NPixMin

            nch = self.VS.NFreqBands
            CubeVariablePSF = np.zeros((NFacets, nch, npol, NPixMin, NPixMin), np.float32)
            CubeMeanVariablePSF = np.zeros((NFacets, 1, npol, NPixMin, NPixMin), np.float32)

            print>>log, "  Cutting PSFs facet-slices "
            for iFacet in sorted(DicoVariablePSF.keys()):
                _, npol, n, n = DicoVariablePSF[iFacet]["PSF"].shape
                for ch in xrange(nch):
                    i = n/2 - NPixMin/2
                    j = n/2 + NPixMin/2 + 1
                    CubeVariablePSF[iFacet, ch, :, :, :] = \
                        DicoVariablePSF[iFacet]["PSF"][ch][:, i:j, i:j]
                CubeMeanVariablePSF[iFacet, 0, :, :, :] = \
                    DicoVariablePSF[iFacet]["MeanPSF"][0, :, i:j, i:j]

            self.DicoPSF["CubeVariablePSF"] = CubeVariablePSF
            self.DicoPSF["CubeMeanVariablePSF"] = CubeMeanVariablePSF
            self.DicoPSF["MeanFacetPSF"] = np.mean(CubeMeanVariablePSF, axis=0).reshape((1, npol, NPixMin, NPixMin))
            self.DicoPSF["MeanJonesBand"] = []
            self.DicoPSF["OutImShape"] = self.OutImShape
            self.DicoPSF["CellSizeRad"] = self.CellSizeRad
            for iFacet in sorted(self.DicoImager.keys()):
                MeanJonesBand = np.zeros((self.VS.NFreqBands,), np.float64)
                for Channel in xrange(self.VS.NFreqBands):
                    ThisSumSqWeights = self.DicoImager[
                        iFacet]["SumJones"][1][Channel]
                    if ThisSumSqWeights == 0:
                        ThisSumSqWeights = 1.
                    ThisSumJones = (
                        self.DicoImager[iFacet]["SumJones"][0][Channel] /
                        ThisSumSqWeights)
                    if ThisSumJones == 0:
                        ThisSumJones = 1.
                    MeanJonesBand[Channel] = ThisSumJones
                self.DicoPSF["MeanJonesBand"].append(MeanJonesBand)

            self.DicoPSF["SumJonesChan"] = []
            self.DicoPSF["SumJonesChanWeightSq"] = []
            for iFacet in sorted(self.DicoImager.keys()):
                ThisFacetSumJonesChan = []
                ThisFacetSumJonesChanWeightSq = []
                for iMS in xrange(self.VS.nMS):
                    A = self.DicoImager[iFacet]["SumJonesChan"][iMS][1, :]
                    A[A == 0] = 1.
                    A = self.DicoImager[iFacet]["SumJonesChan"][iMS][0, :]
                    A[A == 0] = 1.
                    SumJonesChan = self.DicoImager[
                        iFacet]["SumJonesChan"][iMS][0, :]
                    SumJonesChanWeightSq = self.DicoImager[
                        iFacet]["SumJonesChan"][iMS][1, :]
                    ThisFacetSumJonesChan.append(SumJonesChan)
                    ThisFacetSumJonesChanWeightSq.append(SumJonesChanWeightSq)

                self.DicoPSF["SumJonesChan"].append(ThisFacetSumJonesChan)
                self.DicoPSF["SumJonesChanWeightSq"].append(
                    ThisFacetSumJonesChanWeightSq)
            self.DicoPSF["ChanMappingGrid"] = self.VS.DicoMSChanMapping
            self.DicoPSF["ChanMappingGridChan"] = self.VS.DicoMSChanMappingChan
            self.DicoPSF["freqs"] = DicoImages["freqs"]
            self.DicoPSF["WeightChansImages"] = DicoImages["WeightChansImages"]

            self.DicoPSF["ImagData"] = self.FacetsToIm_Channel("PSF")
            if self.VS.MultiFreqMode:
                self.DicoPSF["MeanImage"] = np.sum(self.DicoPSF["ImagData"] * WBAND, axis=0).reshape((1, npol, Npix, Npix))
            else:
                self.DicoPSF["MeanImage"] = self.DicoPSF["ImagData"]

            self._psf_grids = None
            for array in self._psf_grid_names.iterkeys():
                NpShared.DelArray(array)
            self._psf_grid_names = None

        DicoImages["ImagData"] = self.stitchedResidual
        DicoImages["NormImage"] = self.NormImage  # grid-correcting map
        DicoImages["NormData"] = self.NormData
        DicoImages["MeanImage"] = self.MeanResidual

        return DicoImages

    def BuildFacetNormImage(self):
        """
        Creates a stitched tesselation weighting map. This can be useful
        to downweight areas where facets overlap (e.g. padded areas)
        before stitching the facets into one map.
        Returns
            ndarray with norm image
        """
        print>>log, "  Building Facet-normalisation image"
        nch, npol = self.nch, self.npol
        _, _, NPixOut, NPixOut = self.OutImShape
        NormImage = np.zeros((NPixOut, NPixOut), dtype=self.stitchedType)
        for iFacet in self.DicoImager.keys():
            xc, yc = self.DicoImager[iFacet]["pixCentral"]
            NpixFacet = self.DicoImager[iFacet]["NpixFacetPadded"]

            Aedge, Bedge = GiveEdges((xc, yc), NPixOut,
                                     (NpixFacet/2, NpixFacet/2), NpixFacet)
            x0d, x1d, y0d, y1d = Aedge
            x0p, x1p, y0p, y1p = Bedge

            SpacialWeigth = self.SpacialWeigth[iFacet].T[::-1, :]
            SW = SpacialWeigth[::-1, :].T[x0p:x1p, y0p:y1p]
            NormImage[x0d:x1d, y0d:y1d] += np.real(SW)

        return NormImage

    def FacetsToIm_Channel(self, kind="Dirty"):
        """
        Preconditions: assumes the stitched tesselation weighting map has been
        created previously
        Args:
            kind: one of "Jones-amplitude", "Dirty", or "PSF", to create a stitched Jones amplitude, dirty or psf image
        Returns:
            Image cube, which may contain multiple correlations
            and continuum channel bands
        """
        T = ClassTimeIt.ClassTimeIt("FacetsToIm_Channel")
        T.disable()
        Image = self.GiveEmptyMainField()

        nch, npol, NPixOut, NPixOut = self.OutImShape

        print>>log, "Combining facets to stitched %s image" % kind

        for iFacet in self.DicoImager.keys():

            SPhe = self._sphes[iFacet]

            xc, yc = self.DicoImager[iFacet]["pixCentral"]
            NpixFacet = self.DicoGridMachine[iFacet]["Dirty"][0].shape[2]

            Aedge, Bedge = GiveEdges((xc, yc), NPixOut,
                                     (NpixFacet/2, NpixFacet/2), NpixFacet)
            x0main, x1main, y0main, y1main = Aedge
            x0facet, x1facet, y0facet, y1facet = Bedge

            self.DicoImager[iFacet]["SumJonesNorm"] = np.zeros(
                (self.VS.NFreqBands,), np.float64)

            for Channel in xrange(self.VS.NFreqBands):

                ThisSumWeights = self.DicoImager[iFacet]["SumWeights"][Channel]
                ThisSumJones = 1.

                ThisSumSqWeights = self.DicoImager[iFacet]["SumJones"][1][Channel]
                if ThisSumSqWeights == 0:
                    ThisSumSqWeights = 1.
                ThisSumJones = self.DicoImager[iFacet]["SumJones"][0][Channel] / ThisSumSqWeights
                if ThisSumJones == 0:
                    ThisSumJones = 1.
                self.DicoImager[iFacet]["SumJonesNorm"][Channel] = ThisSumJones

                SpacialWeigth = self.SpacialWeigth[iFacet].T[::-1, :]

                T.timeit("3")
                for pol in xrange(npol):
                    # ThisSumWeights.reshape((nch,npol,1,1))[Channel, pol, 0, 0]
                    sumweight = ThisSumWeights[pol]

                    if kind == "Jones-amplitude":
                        Im = SpacialWeigth[::-1, :].T[x0facet:x1facet, y0facet:y1facet] * ThisSumJones
                    else:
                        if kind == "Dirty":
                            Im = self.DicoGridMachine[iFacet]["Dirty"][Channel][pol].copy()
                        elif kind == "PSF":
                            Im = self.DicoPSF[iFacet]["PSF"][Channel][pol].copy()
                        else:
                            raise RuntimeError,"unknown kind=%s argument -- this is a silly bug"%kind
                        # grid-correct the image with the
                        # gridding convolution function
                        Im /= SPhe.real
                        Im[SPhe < 1e-3] = 0
                        Im = (Im[::-1, :].T / sumweight)
                        SW = SpacialWeigth[::-1, :].T
                        Im *= SW

                        Im /= np.sqrt(ThisSumJones)
                        # Im/=(ThisSumJones)

                        Im = Im[x0facet:x1facet, y0facet:y1facet]

                    Image[Channel, pol, x0main:x1main, y0main:y1main] += Im.real

        for Channel in xrange(self.VS.NFreqBands):
            for pol in xrange(npol):
                Image[Channel, pol] /= self.NormImage

        return Image

    def GiveNormImage(self):
        """
        Creates a stitched normalization image of the grid-correction function.
        This image should be point-wise divided from the stitched gridded map
        to create a grid-corrected map.
        Returns:
            stitched grid-correction norm image
        """
        Image = self.GiveEmptyMainField()
        nch, npol = self.nch, self.npol
        _, _, NPixOut, NPixOut = self.OutImShape
        SharedMemName = "%sSpheroidal" % (self.IdSharedMemData)
        NormImage = np.zeros((NPixOut, NPixOut), dtype=self.stitchedType)
        SPhe = NpShared.GiveArray(SharedMemName)
        N1 = self.NpixPaddedFacet

        for iFacet in self.DicoImager.keys():

            xc, yc = self.DicoImager[iFacet]["pixCentral"]
            Aedge, Bedge = GiveEdges((xc, yc), NPixOut, (N1/2, N1/2), N1)
            x0d, x1d, y0d, y1d = Aedge
            x0p, x1p, y0p, y1p = Bedge

            for ch in xrange(nch):
                for pol in xrange(npol):
                    NormImage[x0d:x1d, y0d:y1d] += SPhe[::-1,
                                                        :].T.real[x0p:x1p, y0p:y1p]

        return NormImage

    def ReinitDirty(self, psf=False):
        """
        Reinitializes dirty map and weight buffers for the next round
        of residual calculation
        Postconditions:
        Resets the following:
            self.DicoGridMachine[iFacet]["Dirty"],
            self.DicoImager[iFacet]["SumWeights"],
            self.DicoImager[iFacet]["SumJones"]
            self.DicoImager[iFacet]["SumJonesChan"]
        Also sets up self._facet_grids as a dict of facet numbers to shared grid arrays.
        self.DoPSF is set to psf; if True, shared psf grid arrays alaso initialized
        """
        self.SumWeights.fill(0)
        self.IsDirtyInit = True
        self.HasFourierTransformed = False
        self._facet_grids = {}
        self.DoPSF = psf
        self._psf_grids = {} if psf else None
        self._psf_grid_names = {}

        for iFacet in self.DicoGridMachine.keys():
            NX = self.DicoImager[iFacet]["NpixFacetPadded"]
            if "Dirty" in self.DicoGridMachine[iFacet]:
                self._facet_grids[iFacet] = self.DicoGridMachine[iFacet]["Dirty"]
                self.DicoGridMachine[iFacet]["Dirty"][...] = 0
            else:
                GridName = Multiprocessing.getShmURL("Grid",facet=iFacet)
                ResidueGrid = NpShared.CreateShared(GridName,(self.VS.NFreqBands, self.npol, NX, NX), self.CType)
                self._facet_grids[iFacet] = self.DicoGridMachine[iFacet]["Dirty"] = ResidueGrid
            if self.DoPSF:
                self._psf_grid_names[iFacet] = PSFGridName = Multiprocessing.getShmURL("PSFGrid", facet=iFacet)
                self._psf_grids[iFacet] = NpShared.CreateShared(PSFGridName,(self.VS.NFreqBands, self.npol, NX, NX), self.CType)
            self.DicoImager[iFacet]["SumWeights"] = np.zeros((self.VS.NFreqBands, self.npol), np.float64)
            self.DicoImager[iFacet]["SumJones"] = np.zeros((2, self.VS.NFreqBands), np.float64)
            self.DicoImager[iFacet]["SumJonesChan"] = []
            for iMS in xrange(self.VS.nMS):
                MS = self.VS.ListMS[iMS]
                nVisChan = MS.ChanFreq.size
                self.DicoImager[iFacet]["SumJonesChan"].append(np.zeros((2, nVisChan), np.float64))

    def applySparsification(self, DATA, factor):
        """Computes a sparsification vector for use in the BDA gridder. This is a vector of bools,
        same size as the number of BDA blocks, with a True for every block that will be gridded.
        Blocks ae chosen at random with a probability of 1/factor"""
        if not factor or "BDAGrid" not in DATA:
            DATA["Sparsification"] = np.array([])
        else:
            # randomly select blocks with 1/sparsification probability
            num_blocks = DATA["BDAGrid"][0]
            DATA["Sparsification.Grid"] = numpy.random.sample(num_blocks) < 1.0 / factor
            print>> log, "applying sparsification factor of %f to %d BDA grid blocks, left with %d" % (factor, num_blocks, DATA["Sparsification.Grid"].sum())
            #num_blocks = DATA["BDADegrid"][0]
            #DATA["Sparsification.Degrid"] = numpy.random.sample(num_blocks) < 1.0 / factor
            #print>> log, "applying sparsification factor of %f to %d BDA degrid blocks, left with %d" % (factor, num_blocks, DATA["Sparsification.Degrid"].sum())

    # Gridding worker that is called by Multiprocessing.Process
    @staticmethod
    def _grid_worker(iFacet,
                    GD, DATA, Grids, PSFGrids, WTerms, Sphes, FFTW_Wisdom, DicoImager,
                    ApplyCal, SpheNorm, NFreqBands,
                    DataCorrelationFormat, ExpectedOutputStokes):
        T = ClassTimeIt.ClassTimeIt()
        T.disable()
        if FFTW_Wisdom is not None:
            pyfftw.import_wisdom(FFTW_Wisdom)
        # T.timeit("init %d" % iFacet)
        ListSemaphores = None
        # Create a new GridMachine
        GridMachine = ClassDDEGridMachine.ClassDDEGridMachine(
            GD, DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
            DicoImager[iFacet]["DicoConfigGM"]["Npix"],
            DicoImager[iFacet]["lmShift"],
            iFacet, SpheNorm, NFreqBands,
            DataCorrelationFormat, ExpectedOutputStokes, ListSemaphores,
            wterm=WTerms[iFacet], sphe=Sphes[iFacet],
            bda_grid=DATA["BDAGrid"], bda_degrid=DATA["BDADegrid"])
        T.timeit("create %d" % iFacet)
        uvwThis = DATA["uvw"]
        visThis = DATA["data"]
        flagsThis = DATA["flags"]
        times = DATA["times"]
        A0 = DATA["A0"]
        A1 = DATA["A1"]
        A0A1 = A0, A1
        W = DATA["Weights"]  ## proof of concept for now
        freqs = DATA["freqs"]
        ChanMapping = DATA["ChanMapping"]

        DecorrMode = GD["DDESolutions"]["DecorrMode"]
        if ('F' in DecorrMode) or ("T" in DecorrMode):
            uvw_dt = DATA["uvw_dt"]
            DT, Dnu = DATA["MSInfos"]
            GridMachine.setDecorr(uvw_dt, DT, Dnu, SmearMode=DecorrMode)

        # Create Jones Matrices Dictionary
        DicoJonesMatrices = None
        Apply_killMS = GD["DDESolutions"]["DDSols"]
        Apply_Beam = GD["Beam"]["BeamModel"] is not None

        if ApplyCal:
            DicoJonesMatrices = {}

        if Apply_killMS:
            DicoSols, TimeMapping, DicoClusterDirs = DATA["killMS"]
            DicoJonesMatrices["DicoJones_killMS"] = DicoSols
            DicoJonesMatrices["DicoJones_killMS"]["MapJones"] = TimeMapping
            DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"] = DicoClusterDirs
            DicoJonesMatrices["DicoJones_killMS"]["AlphaReg"] = None

        if Apply_Beam:
            DicoSols, TimeMapping, DicoClusterDirs = DATA["Beam"]
            DicoJonesMatrices["DicoJones_Beam"] = DicoSols
            DicoJonesMatrices["DicoJones_Beam"]["MapJones"] = TimeMapping
            DicoJonesMatrices["DicoJones_Beam"]["DicoClusterDirs"] = DicoClusterDirs
            DicoJonesMatrices["DicoJones_Beam"]["AlphaReg"] = None

        # T.timeit("prepare %d"%iFacet)
        # NpShared.Lock(W)
        T.timeit("lock %d" % iFacet)
        GridMachine.put(times, uvwThis, visThis, flagsThis, A0A1, W,
                        DoNormWeights=False,
                        DicoJonesMatrices=DicoJonesMatrices,
                        freqs=freqs, DoPSF=False,
                        ChanMapping=ChanMapping,
                        ResidueGrid=Grids[iFacet],
                        sparsification=DATA.get("Sparsification.Grid")
                        )
        T.timeit("put %d" % iFacet)

        Sw = GridMachine.SumWeigths.copy()
        SumJones = GridMachine.SumJones.copy()
        SumJonesChan = GridMachine.SumJonesChan.copy()

        # an extra call to grid the PSFs
        if PSFGrids:
            GridMachine.put(times, uvwThis, visThis, flagsThis, A0A1, W,
                            DoNormWeights=False,
                            DicoJonesMatrices=DicoJonesMatrices,
                            freqs=freqs, DoPSF=True,
                            ChanMapping=ChanMapping,
                            ResidueGrid=PSFGrids[iFacet],
                            sparsification = DATA.get("Sparsification.Grid")
                            )

        return {"iFacet": iFacet, "Weights": Sw, "SumJones": SumJones, "SumJonesChan": SumJonesChan}

    def CalcDirtyImagesParallel(self):
        """
        Grids a chunk of input visibilities onto many facets
        Visibility data is already in shared memory (packed there by
        VisServer.VisChunkToShared(), only the weights are passed as a string,
        since they refer to an mmap()d file.

        Post conditions:
        Sets the following normalization weights, as produced by the gridding process:
            self.DicoImager[iFacet]["SumWeights"]
            self.DicoImager[iFacet]["SumJones"]
            self.DicoImager[iFacet]["SumJonesChan"][self.VS.iCurrentMS]
        """
        NFacets = len(self.DicoImager.keys())
        # our job list is just a list of facet numbers
        joblist = range(NFacets)

        procpool = Multiprocessing.ProcessPool(self.GD)

        results = procpool.runjobs(joblist,
            title="Gridding (+PSF)" if self._psf_grids else "Gridding", target=self._grid_worker,
            kwargs=dict(GD=self.GD,
                    DATA=self.VS.DATA,
                    Grids=self._facet_grids,
                    PSFGrids=self._psf_grids,
                    WTerms=self._wterms,
                    Sphes=self._sphes,
                    FFTW_Wisdom=self.FFTW_Wisdom,
                    DicoImager=self.DicoImager,
                    ApplyCal=self.ApplyCal,
                    SpheNorm=self.SpheNorm,
                    NFreqBands=self.VS.NFreqBands,
                    DataCorrelationFormat=self.VS.StokesConverter.AvailableCorrelationProductsIds(),
                    ExpectedOutputStokes=self.VS.StokesConverter.RequiredStokesProductsIds()),
            pause_on_start=self.GD["Debugging"]["PauseGridWorkers"])


        for DicoResult in results:
            iFacet = DicoResult["iFacet"]
            self.DicoImager[iFacet]["SumWeights"] += DicoResult["Weights"]
            self.DicoImager[iFacet]["SumJones"] += DicoResult["SumJones"]
            self.DicoImager[iFacet]["SumJonesChan"][self.VS.iCurrentMS] += DicoResult["SumJonesChan"]

        return True

    @staticmethod
    def _fft_worker(iFacet,
                    GD, Grids, PSFGrids, WTerms, Sphes, FFTW_Wisdom, DicoImager,
                    SpheNorm, NFreqBands,
                    DataCorrelationFormat, ExpectedOutputStokes):
        """
        Fourier transforms the grids currently housed in shared memory
        Precondition:
            Should be called after all data has been gridded
        Returns:
            Dictionary of success and facet identifier
        """
        if FFTW_Wisdom is not None:
            pyfftw.import_wisdom(FFTW_Wisdom)
        GridMachine = ClassDDEGridMachine.ClassDDEGridMachine(
            GD, DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
            DicoImager[iFacet]["DicoConfigGM"]["Npix"],
            DicoImager[iFacet]["lmShift"],
            iFacet, SpheNorm, NFreqBands,
            DataCorrelationFormat, ExpectedOutputStokes, ListSemaphores=None,
            wterm=WTerms[iFacet], sphe=Sphes[iFacet],
        )
        Grid = Grids[iFacet]
        Grid[...] = GridMachine.GridToIm(Grid)
        if PSFGrids:
            PSFGrid = PSFGrids[iFacet]
            PSFGrid[...] = GridMachine.GridToIm(PSFGrid)

        return {"iFacet": iFacet}


    def FourierTransform(self):
        '''
        Fourier transforms the individual facet grids
            self.DicoGridMachine[iFacet]["Dirty"] is FTd in-place
        '''
        ## NB: I removed the doStack=True option because it seemed to be tautological (added grid to itself??)
        NFacets = len(self.DicoImager.keys())
        # our job list is just a list of facet numbers
        joblist = range(NFacets)

        procpool = Multiprocessing.ProcessPool(self.GD)

        procpool.runjobs(joblist, title="Fourier transforms",
                            target=self._fft_worker,
                            kwargs=dict(GD=self.GD,
                                Grids=self._facet_grids,
                                PSFGrids=self._psf_grids,
                                WTerms=self._wterms,
                                Sphes=self._sphes,
                                FFTW_Wisdom = self.FFTW_Wisdom,
                                DicoImager = self.DicoImager,
                                SpheNorm = self.SpheNorm,
                                NFreqBands = self.VS.NFreqBands,
                                DataCorrelationFormat = self.VS.StokesConverter.AvailableCorrelationProductsIds(),
                                ExpectedOutputStokes = self.VS.StokesConverter.RequiredStokesProductsIds()))

    # DeGrid worker that is called by Multiprocessing.Process
    @staticmethod
    def _degrid_worker(iFacet, GD, DATA, WTerms, Sphes, SpacialWeights,
                        ModelImage, Im2Grid, ChanSel, NormImage,
                        FFTW_Wisdom, DicoImager,
                        ApplyCal, SpheNorm, NFreqBands,
                        DataCorrelationFormat, ExpectedOutputStokes, ListSemaphores):
        """
        Degrids input model facets and subtracts model visibilities from residuals.
        Assumes degridding input data is placed in DATA shared memory dictionary.
        Returns:
            Dictionary of success and facet identifier
        """
        if FFTW_Wisdom is not None:
            pyfftw.import_wisdom(FFTW_Wisdom)

        # extract facet model from model image
        ModelGrid, _ = Im2Grid.GiveModelTessel(ModelImage, DicoImager, iFacet, NormImage,
                                                Sphes[iFacet], SpacialWeights[iFacet],
                                                ChanSel=ChanSel)

        # Create a new GridMachine
        GridMachine = ClassDDEGridMachine.ClassDDEGridMachine(
            GD, DicoImager[iFacet]["DicoConfigGM"]["ChanFreq"],
            DicoImager[iFacet]["DicoConfigGM"]["Npix"],
            DicoImager[iFacet]["lmShift"],
            iFacet, SpheNorm, NFreqBands,
            DataCorrelationFormat, ExpectedOutputStokes, ListSemaphores,
            wterm=WTerms[iFacet], sphe=Sphes[iFacet],
            bda_grid=DATA["BDAGrid"], bda_degrid=DATA["BDADegrid"],
        )

        # DATA = NpShared.SharedToDico("%sDicoData" % IdSharedMemData)
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
        Apply_killMS = GD["DDESolutions"]["DDSols"]
        Apply_Beam = GD["Beam"]["BeamModel"] is not None

        if ApplyCal:
            DicoJonesMatrices = {}

        if Apply_killMS:
            DicoSols, TimeMapping, DicoClusterDirs = DATA["killMS"]
            DicoJonesMatrices["DicoJones_killMS"] = DicoSols
            DicoJonesMatrices["DicoJones_killMS"]["MapJones"] = TimeMapping
            DicoJonesMatrices["DicoJones_killMS"]["DicoClusterDirs"] = DicoClusterDirs
            DicoJonesMatrices["DicoJones_killMS"]["AlphaReg"] = None

        if Apply_Beam:
            DicoSols, TimeMapping, DicoClusterDirs = DATA["Beam"]
            DicoJonesMatrices["DicoJones_Beam"] = DicoSols
            DicoJonesMatrices["DicoJones_Beam"]["MapJones"] = TimeMapping
            DicoJonesMatrices["DicoJones_Beam"]["DicoClusterDirs"] = DicoClusterDirs
            DicoJonesMatrices["DicoJones_Beam"]["AlphaReg"] = None

        DecorrMode = GD["DDESolutions"]["DecorrMode"]

        if ('F' in DecorrMode) or ("T" in DecorrMode):
            uvw_dt = DATA["uvw_dt"]
            DT, Dnu = DATA["MSInfos"]
            GridMachine.setDecorr(uvw_dt, DT, Dnu, SmearMode=DecorrMode)

        GridMachine.get(times, uvwThis, visThis, flagsThis, A0A1,
                          ModelGrid, ImToGrid=False,
                          DicoJonesMatrices=DicoJonesMatrices,
                          freqs=freqs, TranformModelInput="FT",
                          ChanMapping=ChanMapping,
                          sparsification=DATA.get("Sparsification.Degrid")
                        )

        return {"iFacet": iFacet}

    def GiveVisParallel(self, ModelImage, Parallel=True):
        """
        Degrids visibilities from model image. The model image is unprojected
        into many facets before degridding and subtracting each of the model
        facets contributions from the residual image.
        Preconditions: the dirty image buffers should be cleared before calling
        the predict and regridding methods
        to construct a new residual map
        Args:
            times:
            uvwIn:
            visIn:
            flag:
            A0A1:
            ModelImage:
        """
        # our job list is just a list of facet numbers
        joblist = sorted(self.DicoImager.keys())

        Im2Grid = ClassImToGrid(OverS=self.GD["ImagerCF"]["OverS"], GD=self.GD)
        ChanSel = sorted(list(set(self.VS.DicoMSChanMappingDegridding[self.VS.iCurrentMS].tolist())))

        NSemaphores = 3373
        ListSemaphores = [ Multiprocessing.getShmName("Semaphore", sem=i) for i in xrange(NSemaphores) ]
        _pyGridderSmearPols.pySetSemaphores(ListSemaphores)


        try:
            procpool = Multiprocessing.ProcessPool(self.GD)

            procpool.runjobs(joblist, title="Degridding",
                             target=self._degrid_worker,
                             pause_on_start=self.GD["Debugging"]["PauseGridWorkers"],
                             kwargs=dict(
                                    GD=self.GD,
                                    DATA=self.VS.DATA,
                                    WTerms=self._wterms,
                                    Sphes=self._sphes,
                                    SpacialWeights=self.SpacialWeigth,
                                    ModelImage=ModelImage,
                                    Im2Grid=Im2Grid,
                                    ChanSel=ChanSel,
                                    NormImage=self.NormImage,
                                    FFTW_Wisdom = self.FFTW_Wisdom,
                                    DicoImager = self.DicoImager,
                                    ApplyCal = self.ApplyCal,
                                    SpheNorm = self.SpheNorm,
                                    NFreqBands = self.VS.NFreqBands,
                                    ListSemaphores = ListSemaphores,
                                    DataCorrelationFormat = self.VS.StokesConverter.AvailableCorrelationProductsIds(),
                                    ExpectedOutputStokes = self.VS.StokesConverter.RequiredStokesProductsIds()))
        finally:
            _pyGridderSmearPols.pyDeleteSemaphore(ListSemaphores)
            for sem in ListSemaphores:
                NpShared.DelArray(sem)
