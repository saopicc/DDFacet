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

import ClassDDEGridMachine
import numpy as np
import ClassCasaImage
import pyfftw
from DDFacet.Array import NpShared
from DDFacet.Imager.ClassImToGrid import ClassImToGrid
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyLogger
from DDFacet.Other.progressbar import ProgressBar
import cPickle
import atexit
import traceback
from matplotlib.path import Path
import numpy.random
from DDFacet.ToolsDir import ModCoord
from DDFacet.Array import NpShared
from DDFacet.Array import shared_dict
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import Multiprocessing
from DDFacet.Other import ModColor
from DDFacet.ToolsDir.ModToolBox import EstimateNpix
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.Imager.ClassImToGrid import ClassImToGrid
from DDFacet.cbuild.Gridder import _pyGridderSmearPols
from DDFacet.Data.ClassStokes import ClassStokes
#from DDFacet.Array import NpParallel
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassFacetMachine")
from DDFacet.Other.AsyncProcessPool import APP
import numexpr
MyLogger.setSilent("MyLogger")
from DDFacet.cbuild.Gridder import _pyGridderSmearPols
from DDFacet.Other import ModColor
MyLogger.setSilent("MyLogger")
import cpuinfo
from DDFacet.ToolsDir import ModFFTW
import scipy.ndimage

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
                 PolMode=["I"],
                 Sols=None,
                 PointingID=0,
                 DoPSF=False,
                 Oversize=1,   # factor by which image is oversized
                 custom_id=None
                 ):

        self.HasFourierTransformed = False

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
        self.StokesConverter = ClassStokes(self.VS.StokesConverter.AvailableCorrelationProductsIds(),
                                           PolMode)
        self.npol = self.StokesConverter.NStokesInImage()
        self.Parallel = True
        if APP is not None:
            APP.registerJobHandlers(self)
            self._fft_job_counter = APP.createJobCounter("fft")
            self._app_id = ("FMPSF" if DoPSF else "FM") if custom_id is None else custom_id

        DicoConfigGM = {}
        self.DicoConfigGM = DicoConfigGM
        self.DoPSF = DoPSF
        # self.MDC.setFreqs(ChanFreq)
        self.CasaImage = None
        self.IsDirtyInit = False
        self.IsDDEGridMachineInit = False
        self.SharedNames = []
        self.ConstructMode = "Fader"
        self.SpheNorm = False
        self.Oversize = Oversize

        DecorrMode=self.GD["RIME"]["DecorrMode"]
        if DecorrMode is not None and DecorrMode is not "":
            print>>log,ModColor.Str("Using decorrelation mode %s"%DecorrMode)
        self.AverageBeamMachine=None
        self.SmoothJonesNorm=None
        self.MeanSmoothJonesNorm=None
        self.JonesNorm = None
        self.FacetNorm = None

        self._facet_grids = self._CF = self.DATA = None
        self._grid_job_id = self._fft_job_id = self._degrid_job_id = None
        self._smooth_job_label=None

        # create semaphores if not already created
        if not ClassFacetMachine._degridding_semaphores:
            NSemaphores = 3373
            ClassFacetMachine._degridding_semaphores = [Multiprocessing.getShmName("Semaphore", sem=i) for i in xrange(NSemaphores)]
            _pyGridderSmearPols.pySetSemaphores(ClassFacetMachine._degridding_semaphores)
            atexit.register(ClassFacetMachine._delete_degridding_semaphores)

        # this is used to store model images in shared memory, for the degridder
        self._model_dict = None
        # this is used to store NormImage in shared memory, for the degridder
        self._norm_dict = None

        # build the 'history' list used for writing FITS files
        self.make_history()

    # static attribute initialized below, once
    _degridding_semaphores = None

    @staticmethod
    def _delete_degridding_semaphores():
        if ClassFacetMachine._degridding_semaphores:
            _pyGridderSmearPols.pyDeleteSemaphore(ClassFacetMachine._degridding_semaphores)
            for sem in ClassFacetMachine._degridding_semaphores:
                NpShared.DelArray(sem)

    def __del__(self):
        self.releaseGrids()
        if self._delete_cf_in_destructor:
            self.releaseCFs()

    def releaseGrids(self):
        if self._facet_grids is not None:
            self._facet_grids.delete()
            self._facet_grids = None
        for GM in self.DicoGridMachine.itervalues():
            if "Dirty" in GM:
                del GM["Dirty"]

    def releaseCFs(self):
        if self._CF is not None:
            self._CF.delete()
            self._CF = None

    def setAverageBeamMachine(self,AverageBeamMachine):
        self.AverageBeamMachine=AverageBeamMachine
        if self.AverageBeamMachine.SmoothBeam is not None:
            print>>log,"  Smooth beam machine already has a smooth beam"
            Npix=self.OutImShape[-1]
            self.SmoothJonesNorm = self.AverageBeamMachine.SmoothBeam.reshape((self.VS.NFreqBands,1,Npix,Npix))
            self.MeanSmoothJonesNorm = self.AverageBeamMachine.MeanSmoothBeam.reshape((1,1,Npix,Npix))


    def SetLogModeSubModules(self,Mode="Silent"):
        SubMods=["ModelBeamSVD","ClassParam","ModToolBox","ModelIonSVD2","ClassPierce"]

        if Mode == "Silent":
            MyLogger.setSilent(SubMods)
        if Mode == "Loud":
            MyLogger.setLoud(SubMods)

    def make_history(self):
        history=[]
        for k in self.GD:
            if isinstance(self.GD[k],dict):
                history.append('=== '+k+' ===')
                for dk in self.GD[k]:
                    if dk[0]!='_':
                        history.append(k+'-'+dk+' = '+str(self.GD[k][dk]))
            else:
                # catchall
                history.append(k+' = '+str(self.GD[k]))
        self.history=history

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
        Cell = self.GD["Image"]["Cell"]

        self.ImageName = ImageName

        self.LraFacet = []
        self.LdecFacet = []

        self.ChanFreq = self.VS.GlobalFreqs

        self.NFacets = NFacets
        self.Cell = Cell
        self.CellSizeRad = (Cell / 3600.) * np.pi / 180.
        rac, decc = self.VS.ListMS[0].radec
        self.MainRaDec = (rac, decc)
        self.nch = self.VS.NFreqBands
        self.NChanGrid = self.nch
        self.SumWeights = np.zeros((self.NChanGrid, self.npol), float)

        self.CoordMachine = ModCoord.ClassCoordConv(rac, decc)
        # get the closest fast fft size:
        Npix = self.GD["Image"]["NPix"]
        Padding = self.GD["Facets"]["Padding"]
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
        # print>>log,"Facet %d l %f m %f RA %f Dec %f"%(iFacet, l0, m0, raFacet, decFacet)

        NpixFacet, _ = EstimateNpix(diam / self.CellSizeRad, Padding=1)
        _, NpixPaddedGrid = EstimateNpix(NpixFacet, Padding=self.Padding)

        if NpixPaddedGrid / NpixFacet > self.Padding:
            print>> log, ModColor.Str("W.A.R.N.I.N.G: Your FFTs are too small. We will pad it %.2f x "\
                                      "instead of %.2f x" % (float(NpixPaddedGrid)/NpixFacet, self.Padding),
                                      col="yellow")

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
        DicoConfigGM = {"NPix": NpixFacet,
                        "Cell": self.GD["Image"]["Cell"],
                        "ChanFreq": self.ChanFreq,
                        "DoPSF": False,
                        "Support": self.GD["CF"]["Support"],
                        "OverS": self.GD["CF"]["OverS"],
                        "Nw": self.GD["CF"]["Nw"],
                        "WProj": True,
                        "DoDDE": self.DoDDE,
                        "Padding": self.GD["Facets"]["Padding"]}

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

        # self.JonesDirCat.ra[iFacet] = raFacet[0]
        # self.JonesDirCat.dec[iFacet] = decFacet[0]
        # l, m = self.DicoImager[iFacet]["l0m0"]
        # self.JonesDirCat.l[iFacet] = l
        # self.JonesDirCat.m[iFacet] = m
        # self.JonesDirCat.Cluster[iFacet] = iFacet

    def setFacetsLocs(self):
        """
        Routine to split the image into a grid of squares.
        This can be overridden to perform more complex tesselations
        """
        Npix = self.GD["Image"]["NPix"]
        NFacets = self.GD["Facets"]["NFacets"]
        Padding = self.GD["Facets"]["Padding"]
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

        self.JonesDirCat = np.zeros(
            (lFacet.size,),
            dtype=[('Name', '|S200'),
                   ('ra', np.float),
                   ('dec', np.float),
                   ('SumI', np.float),
                   ("Cluster", int),
                   ("l", np.float),
                   ("m", np.float),
                   ("I", np.float)])

        self.JonesDirCat = self.JonesDirCat.view(np.recarray)
        self.JonesDirCat.I = 1
        self.JonesDirCat.SumI = 1

        for iFacet in xrange(lFacet.size):
            l0 = x0facet[iFacet] * self.CellSizeRad
            m0 = y0facet[iFacet] * self.CellSizeRad
            l0 = lFacet[iFacet]
            m0 = mFacet[iFacet]

            # print x0facet[iFacet],y0facet[iFacet],l0,m0
            self.AppendFacet(iFacet, l0, m0, NpixFacet * self.CellSizeRad)


        
        #self.iCentralFacet = self.DicoImager[lFacet.size / 2]

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

        DicoClusterDirs= NpShared.SharedToDico("%sDicoClusterDirs" % self.IdSharedMemData)
        lc=DicoClusterDirs["l"]
        mc=DicoClusterDirs["m"]
        sI=DicoClusterDirs["I"]
        x0,x1=lc.min()-np.pi/180,lc.max()+np.pi/180
        y0,y1=mc.min()-np.pi/180,mc.max()+np.pi/180
        InterpMode=self.GD["DDESolutions"]["Type"]
        if InterpMode=="Krigging":
            import pylab
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
        self.DicoGridMachine = {}
        for iFacet in self.DicoImager.keys():
            self.DicoGridMachine[iFacet] = {}
        self.setWisdom()
        self._CF = None
        self.IsDDEGridMachineInit = False
        self.SetLogModeSubModules("Loud")
        self._Im2Grid = ClassImToGrid(OverS=self.GD["CF"]["OverS"], GD=self.GD)


    def setWisdom(self):
        """
        Set fft wisdom
        """
        import socket, os
        from os.path import expanduser
        if self.GD["RIME"]["FFTMachine"]!="FFTW": return
        self.wisdom_cache_path = self.GD["Cache"]["DirWisdomFFTW"]
        #hostname=socket.gethostname()
        cpuname=cpuinfo.get_cpu_info()["brand"].replace(" ","")
        if "~" in self.wisdom_cache_path:
            home = expanduser("~")        
            self.wisdom_cache_path=self.wisdom_cache_path.replace("~",home)
        self.wisdom_cache_path_host = "/".join([self.wisdom_cache_path,cpuname])
        self.wisdom_cache_file =  "/".join([self.wisdom_cache_path_host,"Wisdom.pickle"])
        #self.wisdom_cache_path_host="'%s'"%self.wisdom_cache_path_host
        #self.wisdom_cache_file="'%s'"%self.wisdom_cache_file
        


        if not os.path.isdir(self.wisdom_cache_path_host):
            print>>log, "Wisdom file %s does not exist, create it" % (self.wisdom_cache_path_host)
            os.makedirs(self.wisdom_cache_path_host)

        if os.path.isfile(self.wisdom_cache_file):
            print>>log, "Loading wisdom file %s" % (self.wisdom_cache_file)
            DictWisdom = cPickle.load(file(self.wisdom_cache_file))
            pyfftw.import_wisdom(DictWisdom["Wisdom"])
            WisdomTypes=DictWisdom["WisdomTypes"]
        else:
            WisdomTypes=[]

        HasTouchedWisdomFile = False
        T=ClassTimeIt.ClassTimeIt("setWisdom")
        T.disable()
        for iFacet in sorted(self.DicoImager.keys()):
            NPixPadded=self.DicoImager[iFacet]["NpixFacetPadded"]
            if self.GD["RIME"]["Precision"]=="S":
                TypeKey=(NPixPadded,np.complex64)
            elif self.GD["RIME"]["Precision"]=="D":
                TypeKey=(NPixPadded,np.complex128)

            if TypeKey not in WisdomTypes:
                HasTouchedWisdomFile = True
                ModFFTW.learnFFTWWisdom(*TypeKey)
                WisdomTypes.append(TypeKey)

        NOut = self.OutImShape[-1]
        TypeKey=(NOut,np.float32)
        if TypeKey not in WisdomTypes:
            HasTouchedWisdomFile = True
            ModFFTW.learnFFTWWisdom(*TypeKey)
            WisdomTypes.append(TypeKey)
        self.FFTW_Wisdom = pyfftw.export_wisdom()
        DictWisdom={"Wisdom":self.FFTW_Wisdom,
                    "WisdomTypes":WisdomTypes}

        if HasTouchedWisdomFile:
            print>>log, "Saving wisdom file to %s"%self.wisdom_cache_file
            cPickle.dump(DictWisdom, file(self.wisdom_cache_file, "w"))


    def initCFInBackground (self, other_fm=None):
        # if we have another FacetMachine supplied, check if the same CFs apply
        if other_fm and self.Oversize == other_fm.Oversize:
            self._CF = other_fm._CF
            self._delete_cf_in_destructor = False
            self.IsDDEGridMachineInit = True
            return
        self._delete_cf_in_destructor = True
        # get wmax from MS (if needed)
        wmax = self.GD["CF"]["wmax"]
        if wmax:
            print>>log,"max w=%.6g as per --CF-wmax setting"%wmax
        else:
            wmax = self.VS.getMaxW()
            print>>log,"max w=%.6g from MS (--CF-wmax=0)"%wmax
        # subprocesses will place W-terms etc. here. Reset this first.
        self._CF = shared_dict.create("CFPSF" if self.DoPSF else "CF")
        # check if w-kernels, spacial weights, etc. are cached

        if self.GD["Cache"]["CacheCF"]:
            cachekey = dict(ImagerCF=self.GD["CF"], 
                            ImagerMainFacet=self.GD["Image"], 
                            Facets=self.GD["Facets"], 
                            RIME=self.GD["RIME"],
                            DDESolutions={"DDSols":self.GD["DDESolutions"]["DDSols"]})
            cachename = self._cf_cachename = "CF"
            # in oversize-PSF mode, make separate cache for PSFs
            if self.DoPSF and self.Oversize != 1:
                cachename = self._cf_cachename = "CFPSF"
                cachekey["Oversize"] = self.Oversize
            # check cache
            cachepath, cachevalid = self.VS.maincache.checkCache(cachename, cachekey, directory=True)
        else:
            print>>log,ModColor.Str("Explicitly not caching nor using cache for the Convolution Function")
            cachepath, cachevalid="",False
            
        # up to workers to load/save cache
        for iFacet in self.DicoImager.iterkeys():
            facet_dict = self._CF.addSubdict(iFacet)
            APP.runJob("%s.InitCF.f%s"%(self._app_id, iFacet), self._initcf_worker,
                            args=(iFacet, facet_dict.readwrite(), cachepath, cachevalid, wmax))
        #workers_res=APP.awaitJobResults("%s.InitCF.*"%self._app_id, progress="Init CFs")


    def _initcf_worker (self, iFacet, facet_dict, cachepath, cachevalid, wmax):
        """Worker method of InitParal"""
        path = "%s/%s.npz" % (cachepath, iFacet)
        T=ClassTimeIt.ClassTimeIt("_initcf_worker")
        # try to load the cache, and copy it to the shared facet dict
        if cachevalid:
            try:
                npzfile = np.load(file(path))
                for key, value in npzfile.iteritems():
                    facet_dict[key] = value
                # validate dict
                ClassDDEGridMachine.ClassDDEGridMachine.verifyCFDict(facet_dict, self.GD["CF"]["Nw"])
                return "cached",path,iFacet
            except Exception as e:
                #print>>log,traceback.format_exc() #confusing...
                print>>log, 'Exception on Cache loading and checking was',str(e)
                print>>log, "Error loading %s, will re-generate"%path
                facet_dict.delete()
        # ok, regenerate the terms at this point
        FacetInfo = self.DicoImager[iFacet]
        # Create smoothned facet tessel mask:
        Npix = FacetInfo["NpixFacetPadded"]
        l0, l1, m0, m1 = FacetInfo["lmExtentPadded"]
        X, Y = np.mgrid[l0:l1:Npix * 1j, m0:m1:Npix * 1j]
        XY = np.dstack((X, Y))
        XY_flat = XY.reshape((-1, 2))
        vertices = FacetInfo["Polygon"]
        mpath = Path(vertices)  # the vertices of the polygon
        mask_flat = mpath.contains_points(XY_flat)
        mask = mask_flat.reshape(X.shape)
        mpath = Path(self.CornersImageTot)
        mask_flat2 = mpath.contains_points(XY_flat)
        mask2 = mask_flat2.reshape(X.shape)
        mask[mask2 == 0] = 0

        #NB: this spatial weighting is a bit arbitrary.... 
        #it may be better to do something like Montage's background
        #normalization (http://montage.ipac.caltech.edu/docs/algorithms.html#background)
        GaussPars = (10, 10, 0)

        # compute spatial weight term
        sw = np.float32(mask.reshape((1, 1, Npix, Npix)))

        # already happening in parallel so make sure the FFT library doesn't spawn its own threads
        sw = ModFFTW.ConvolveGaussian(shareddict = {"in": sw, "out": sw},
                                      field_in = "in",
                                      field_out = "out",
                                      ch = 0,
                                      CellSizeRad=1,
                                      GaussPars_ch=GaussPars)
        sw = sw.reshape((Npix, Npix))
        sw /= np.max(sw)
        ## Will speedup degridding NB: will it?
        sw[sw<1e-3] = 0.
        facet_dict["SW"] = sw

        # Initialize a grid machine per iFacet, this will implicitly compute wterm and Sphe
        self._createGridMachine(iFacet, cf_dict=facet_dict, compute_cf=True, wmax=wmax)

        # # save cache
        # DoPrintErr=False
        # while True:
        #     try:
        #         np.savez(file(path, "w"), **facet_dict)
        #         if DoPrintErr:
        #             print>>log,ModColor.Str("  ok could save %s"%path,col="green")
        #         break
        #     except:
        #         import time
        #         DoPrintErr=True
        #         print>>log,ModColor.Str("Failed to save %s"%path)
        #         time.sleep(1.)
        return "compute",path, iFacet

    def awaitInitCompletion (self):
        if not self.IsDDEGridMachineInit:
            workers_res=APP.awaitJobResults("%s.InitCF.*"%self._app_id, progress="Init CFs")
            self._CF.reload()
            # mark cache as safe
            for res in workers_res:
                Type,path,iFacet=res
                if Type=="compute" and self.GD["Cache"]["CacheCF"]:
                    #print iFacet
                    facet_dict=self._CF[iFacet]
                    d={}
                    for key in facet_dict.keys():
                        d[key]=facet_dict[key]
                    np.savez(file(path, "w"), **d)
            if self.GD["Cache"]["CacheCF"]: self.VS.maincache.saveCache(self._cf_cachename)
            self.IsDDEGridMachineInit = True

    def setCasaImage(self, ImageName=None, Shape=None, Freqs=None, Stokes=["I"]):
        if ImageName is None:
            ImageName = self.ImageName

        if Shape is None:
            Shape = self.OutImShape
        self.CasaImage = ClassCasaImage.ClassCasaimage(
            ImageName, Shape, self.Cell, self.MainRaDec, Freqs=Freqs,
            Stokes=Stokes, history=self.history, header_dict=self.VS.obs_detail)

    def _createGridMachine(self, iFacet, **kw):
        """Helper method for workers: creates a GridMachine with the given extra keyword arguments"""
        FacetInfo = self.DicoImager[iFacet]
        return ClassDDEGridMachine.ClassDDEGridMachine(
            self.GD,
            FacetInfo["DicoConfigGM"]["ChanFreq"],
            FacetInfo["DicoConfigGM"]["NPix"],
            FacetInfo["lmShift"],
            iFacet, self.SpheNorm, self.VS.NFreqBands,
            self.StokesConverter.AvailableCorrelationProductsIds(),
            self.StokesConverter.RequiredStokesProductsIds(),
            **kw)

    def ToCasaImage(self, ImageIn, Fits=True, ImageName=None,
                    beam=None, beamcube=None, Freqs=None, Stokes=["I"]):

        if Freqs is None:
            # if we have a reference frequency, use it
            try:
                Freqs=np.array([self.VS.RefFreq])
            except:
                pass
        self.setCasaImage(ImageName=ImageName, Shape=ImageIn.shape,
                          Freqs=Freqs, Stokes=Stokes)

        self.CasaImage.setdata(ImageIn, CorrT=True)

        if Fits:
            if beam is not None:
                self.CasaImage.setBeam(beam, beamcube=beamcube)
            self.CasaImage.ToFits()
        else:
            raise RunTimeError('Fits = False not supported')
        self.CasaImage.close()
        self.CasaImage = None

    def GiveEmptyMainField(self):
        """
        Gives empty image of the correct shape to act as buffer for e.g. the stitching process
        Returns:
            ndarray of type complex
        """
        return np.zeros(self.OutImShape, dtype=self.stitchedType)

    def putChunkInBackground(self, DATA):
        """
        """
        self.SetLogModeSubModules("Silent")
        if not self.IsDirtyInit:
            self.ReinitDirty()
        self.gridChunkInBackground(DATA)
        self.SetLogModeSubModules("Loud")

    def getChunkInBackground(self, DATA):
        """Gets visibilities corresponding to current model image."""
        if self.DoPSF:
            raise RuntimeError("Can't call getChunk on a PSF mode FacetMachine. This is a bug!")
        self.SetLogModeSubModules("Silent")
        self.degridChunkInBackground(DATA)
        self.SetLogModeSubModules("Loud")


    def setModelImage(self, ModelImage):
        """Sets current model image. Copies it to a shared dict and returns shared array version of image."""
        if self.DoPSF:
            raise RuntimeError("Can't call getChunk on a PSF mode FacetMachine. This is a bug!")
        self._model_dict = shared_dict.create("Model")
        self._model_dict["Image"] = ModelImage
        for iFacet in range(self.NFacets):
            self._model_dict.addSubdict(iFacet)
        return self._model_dict["Image"]

    def releaseModelImage(self):
        """Deletes current model image from SHM. USe to save RAM."""
        if self._model_dict is not None:
            self._model_dict.delete()
            self._model_dict = None

    def _buildFacetSlice_worker(self, iFacet, facet_grids, facetdict, cfdict, sumjonesnorm, sumweights, W):
        # first normalize by spheroidals - these
        # facet psfs will be used in deconvolution per facet
        SPhe = cfdict["Sphe"]
        nx = SPhe.shape[0]
        SPhe = SPhe.reshape((1, 1, nx, nx)).real
        fd = facetdict.addSubdict(iFacet)
        ## @cyriltasse reported a problem with this:
        # fd["PSF"] = facet_grids[iFacet].real
        ## so do this instead
        psf = fd.addSharedArray("PSF", facet_grids[iFacet].shape, np.float32)
        psf[...] = facet_grids[iFacet].real
        psf /= SPhe
        # DicoImages[iFacet]["PSF"][SPhe < 1e-2] = 0
        fd["l0m0"] = self.DicoImager[iFacet]["l0m0"]
        fd["pixCentral"] = self.DicoImager[iFacet]["pixCentral"]
        fd["lmSol"] = self.DicoImager[iFacet]["lmSol"]

        nch, npol, n, n = psf.shape
        PSFChannel = np.zeros((nch, npol, n, n), self.stitchedType)
        for ch in xrange(nch):
            psf[ch][SPhe[0] < 1e-2] = 0
            psf[ch][0] = psf[ch][0].T[::-1, :]
            SumJonesNorm = sumjonesnorm[ch]
            # normalize to bring back transfer
            # functions to approximate convolution
            psf[ch] /= np.sqrt(SumJonesNorm)
            for pol in xrange(npol):
                ThisSumWeights = sumweights[ch][pol]
                # normalize the response per facet
                # channel if jones corrections are enabled
                psf[ch][pol] /= ThisSumWeights
            PSFChannel[ch, :, :, :] = psf[ch][:, :, :]

        # weight each of the cube slices and average
        fd["MeanPSF"]  = np.sum(PSFChannel * W, axis=0).reshape((1, npol, n, n))

    def _cutFacetSlice_worker(self, iFacet, DicoImages, nch, NPixMin):
        psf = DicoImages["Facets"][iFacet]["PSF"]
        _, npol, n, n = psf.shape
        for ch in xrange(nch):
            i = n / 2 - NPixMin / 2
            j = n / 2 + NPixMin / 2 + 1
            DicoImages["CubeVariablePSF"][iFacet, ch, :, :, :] = psf[ch][:, i:j, i:j]
        DicoImages["CubeMeanVariablePSF"][iFacet, 0, :, :, :] = DicoImages["Facets"][iFacet]["MeanPSF"][0, :, i:j, i:j]

    def _computeFacetMeanResidual_worker(self, iFacet, fmr_dict, grid_dict, cf_dict, SumWeights, SumJonesNorm, WBAND):
        dirty = grid_dict[iFacet]
        nch, npol, npix_x, npix_y = dirty.shape
        ThisW = SumWeights.reshape((self.VS.NFreqBands, npol, 1, 1))
        SumJonesNorm = np.sqrt(SumJonesNorm)
        if np.max(SumJonesNorm) > 0.:
            ThisW = ThisW * SumJonesNorm.reshape((self.VS.NFreqBands, 1, 1, 1))
        ThisDirty = dirty.real / ThisW
        fmr = fmr_dict.addSharedArray(iFacet, (1, npol, npix_x, npix_y), ThisDirty.dtype)
        fmr[:] = np.sum(ThisDirty * WBAND, axis=0).reshape((1, npol, npix_x, npix_y))
        fmr /= cf_dict[iFacet]["Sphe"]

    def FacetsToIm(self, NormJones=False):
        """
        Fourier transforms the individual facet grids and then
        Stitches the gridded facets and builds the following maps:
            self.stitchedResidual (initial residual is the dirty map)
            self.FacetNorm (grid-correcting map, see also: BuildFacetNormImage() method)
            self.MeanResidual ("average" residual map taken over all continuum bands of the residual cube,
                               this will be the same as stitchedResidual if there is only one continuum band in the residual
                               cube)
            self.DicoPSF if the facet machine is set to produce a PSF. This contains, amongst others a PSF and mean psf per facet
            Note that only the stitched residuals are currently normalized and converted to stokes images for cleaning.
            This is because the coplanar facets should be jointly cleaned on a single map.
        Args:
            NormJones: if True (and there is Jones Norm data available) also computes self.JonesNorm (ndarray) of jones
            averages.
            psf: if True (and PSF grids are available), also computes PSF terms


        Returns:
            Dictionary containing:
            "ImageCube" = self.stitchedResidual
            "FacetNorm" = self.FacetImage (grid-correcting map)
            "JonesNorm" = self.JonesNorm (if computed, see above)
            "MeanImage" = self.MeanResidual
            "freqs" = channel information on the bands being averaged into each of the continuum slices of the residual
            "SumWeights" = sum of visibility weights used in normalizing the gridded correlations
            "WeightChansImages" = normalized weights
        """
        # wait for any outstanding grid jobs to finish
        self.collectGriddingResults()

        if not self.HasFourierTransformed:
            self.fourierTransformInBackground()
            self.collectFourierTransformResults()
            self.HasFourierTransformed = True
        _, npol, Npix, Npix = self.OutImShape
        DicoImages = shared_dict.create("%s_AllImages"%self._app_id)
        DicoImages["freqs"] = {}
        DicoImages.addSubdict("freqs")
        DicoImages.addSubdict("ImageInfo")
        DicoImages["ImageInfo"]["CellSizeRad"]=self.CellSizeRad
        DicoImages["ImageInfo"]["OutImShape"]=self.OutImShape
        

        # Assume all facets have the same weight sums.
        # Store the normalization weights for reference
        DicoImages["SumWeights"] = np.zeros((self.VS.NFreqBands, self.npol), np.float64)
        for band, channels in enumerate(self.VS.FreqBandChannels):
            DicoImages["freqs"][band] = channels
            DicoImages["SumWeights"][band] = self.DicoImager[0]["SumWeights"][band]
        DicoImages["WeightChansImages"] = DicoImages["SumWeights"] / np.sum(DicoImages["SumWeights"])

        # compute sum of Jones terms per facet and channel
        for iFacet in self.DicoImager.keys():
            self.DicoImager[iFacet]["SumJonesNorm"] = np.zeros(self.VS.NFreqBands, np.float64)
            for Channel in xrange(self.VS.NFreqBands):
                ThisSumSqWeights = self.DicoImager[iFacet]["SumJones"][1][Channel]
                if ThisSumSqWeights == 0:
                    ThisSumSqWeights = 1.
                ThisSumJones = self.DicoImager[iFacet]["SumJones"][0][Channel] / ThisSumSqWeights
                if ThisSumJones == 0:
                    ThisSumJones = 1.
                self.DicoImager[iFacet]["SumJonesNorm"][Channel] = ThisSumJones

        # build facet-normalization image
        self.BuildFacetNormImage()
        FacetNorm=self._norm_dict["FacetNorm"]
        # self.stitchedResidual = self.FacetsToIm_Channel()

        # build Jones amplitude image
        DoCalcJonesNorm = NormJones and not "JonesNorm" in self._norm_dict
        if DoCalcJonesNorm:
            self._norm_dict["JonesNorm"] = self.FacetsToIm_Channel("Jones-amplitude")

        JonesNorm = self._norm_dict["JonesNorm"]

        # compute normalized per-band weights (WBAND)
        if self.VS.MultiFreqMode:
            WBAND = np.array([DicoImages["SumWeights"][Channel] for Channel in xrange(self.VS.NFreqBands)])
            # sum frequency contribution to weights per correlation
            WBAND /= np.sum(WBAND, axis=0)
            WBAND = np.float32(WBAND.reshape((self.VS.NFreqBands, npol, 1, 1)))
        else:
            WBAND = 1
        #  ok, make sure the FTs have been computed
        # self.collectFourierTransformResults()
        # PSF mode: construct PSFs
        if self.DoPSF:
            DicoVariablePSF = DicoImages.addSubdict("Facets")
            facets = sorted(self.DicoGridMachine.keys())
            W = DicoImages["WeightChansImages"]
            W = np.float32(W.reshape((self.VS.NFreqBands, npol, 1, 1)))

            for iFacet in facets:
                APP.runJob("buildpsf:%s"%iFacet, self._buildFacetSlice_worker,
                           args=(iFacet, self._facet_grids.readonly(), DicoVariablePSF.writeonly(), self._CF[iFacet].readonly(),
                                 self.DicoImager[iFacet]["SumJonesNorm"], self.DicoImager[iFacet]["SumWeights"], W))
            APP.awaitJobResults("buildpsf:*", progress="Build PSF facet slices")
            DicoVariablePSF.reload()

            NFacets = len(DicoVariablePSF)

            if self.GD["Facets"]["Circumcision"]:
                NPixMin = self.GD["Facets"]["Circumcision"]
                # print>>log,"using explicit Circumcision=%d"%NPixMin
            else:
                NPixMin = 1e6
                for iFacet in facets:
                    _, npol, n, n = DicoVariablePSF[iFacet]["PSF"].shape
                    if n < NPixMin:
                        NPixMin = n

                NPixMin = int(NPixMin/self.GD["Facets"]["Padding"])
                if not NPixMin % 2:
                    NPixMin += 1
                    # print>>log,"using computed Circumcision=%d"%NPixMin

            nch = self.VS.NFreqBands
            DicoImages.addSharedArray("CubeVariablePSF",(NFacets, nch, npol, NPixMin, NPixMin), np.float32)
            DicoImages.addSharedArray("CubeMeanVariablePSF",(NFacets, 1, npol, NPixMin, NPixMin), np.float32)

            #CubeVariablePSF = np.zeros((NFacets, nch, npol, NPixMin, NPixMin), np.float32)
            #CubeMeanVariablePSF = np.zeros((NFacets, 1, npol, NPixMin, NPixMin), np.float32)

            print>>log, "cutting PSF facet-slices of shape %dx%d" % (NPixMin, NPixMin)
            for iFacet in facets:
                APP.runJob("cutpsf:%s" % iFacet, self._cutFacetSlice_worker, args=(iFacet, DicoImages.readonly(), nch, NPixMin))
            APP.awaitJobResults("cutpsf:*", progress="Cut PSF facet slices")

            DicoImages["CentralFacet"] = self.iCentralFacet
            DicoImages["MeanJonesBand"] = []
            CubeVariablePSF=DicoImages["CubeVariablePSF"]
            CubeMeanVariablePSF=DicoImages["CubeMeanVariablePSF"]
            print>>log,"  Building Facets-PSF normalised by their maximum"
            DicoImages.addSharedArray("PeakNormed_CubeVariablePSF",(NFacets, nch, npol, NPixMin, NPixMin), np.float32)
            DicoImages.addSharedArray("PeakNormed_CubeMeanVariablePSF",(NFacets, 1, npol, NPixMin, NPixMin), np.float32)

            for iFacet in facets:
                DicoImages["PeakNormed_CubeMeanVariablePSF"][iFacet]=CubeMeanVariablePSF[iFacet]/np.max(CubeMeanVariablePSF[iFacet])
                for iChan in range(nch):
                    DicoImages["PeakNormed_CubeVariablePSF"][iFacet,iChan]=CubeVariablePSF[iFacet,iChan]/np.max(CubeVariablePSF[iFacet,iChan])
                    #DicoImages["PeakNormed_CubeVariablePSF"][iFacet,iChan]=CubeVariablePSF[iFacet,iChan]/np.max(CubeMeanVariablePSF[iFacet])

            PeakNormed_CubeMeanVariablePSF=DicoImages["PeakNormed_CubeMeanVariablePSF"]

            DicoImages["MeanFacetPSF"]=np.mean(PeakNormed_CubeMeanVariablePSF,axis=0).reshape((1,npol,NPixMin,NPixMin))
            ListMeanJonesBand=[]
            DicoImages["OutImShape"] = self.OutImShape
            DicoImages["CellSizeRad"] = self.CellSizeRad
            for iFacet in sorted(self.DicoImager.keys()):
                MeanJonesBand = np.zeros((self.VS.NFreqBands,), np.float64)
                for Channel in xrange(self.VS.NFreqBands):
                    ThisSumSqWeights = self.DicoImager[iFacet]["SumJones"][1][Channel] or 1
                    ThisSumJones = (self.DicoImager[iFacet]["SumJones"][0][Channel] / ThisSumSqWeights) or 1
                    MeanJonesBand[Channel] = ThisSumJones
                ListMeanJonesBand.append(MeanJonesBand)
            DicoImages["MeanJonesBand"]=ListMeanJonesBand

            ## OMS: see issue #484. Restructuring this to use shm, and less of it. Note that the only user
            ## of this structure is ClassSpectralFunctions and ClassPSFServer
            ## [iMS][iFacet,0,:] is the sum of the per-channel weights
            ## [iMS][iFacet,1,:] is the sum of the per-channel weights squared
            ListSumJonesChan = DicoImages.addSubdict("SumJonesChan")
            for iMS in xrange(self.VS.nMS):
                nVisChan = self.VS.ListMS[iMS].ChanFreq.size
                ThisMSSumJonesChan = ListSumJonesChan.addSharedArray(iMS, (len(facets), 2, nVisChan), np.float64)
                for iFacet in facets:
                    sumjones = self.DicoImager[iFacet]["SumJonesChan"][iMS]
                    sumjones[sumjones == 0] = 1.
                    ThisMSSumJonesChan[iFacet,:] = sumjones[:]

            DicoImages["ChanMappingGrid"] = self.VS.DicoMSChanMapping
            DicoImages["ChanMappingGridChan"] = self.VS.DicoMSChanMappingChan

            DicoImages["ImageCube"] = self.FacetsToIm_Channel("PSF")
            if self.VS.MultiFreqMode:
                DicoImages["MeanImage"] = np.sum(DicoImages["ImageCube"] * WBAND, axis=0).reshape((1, npol, Npix, Npix))
            else:
                DicoImages["MeanImage"] = DicoImages["ImageCube"]

            DicoImages["FacetNorm"] = FacetNorm
            DicoImages["JonesNorm"] = JonesNorm
            
            #for iFacet in sorted(self.DicoImager.keys()):
                #DicoImages["Facets"][iFacet].delete_item("PSF")
                #DicoImages["Facets"][iFacet].delete_item("MeanPSF")

            for iFacet in facets:
                DicoImages["Facets"][iFacet].delete_item("PSF")
                DicoImages["Facets"][iFacet].delete_item("MeanPSF")
            # print>>log,"copying dictPSF"
            # DicoImages.reload()
            self._psf_dict = DicoImages
            return DicoImages

        # else build Dirty (residual) image
        else:
            fmr_dict = DicoImages.addSubdict("FacetMeanResidual")

            for iFacet in sorted(self.DicoImager.keys()):
                APP.runJob("facetmeanresidual:%s" % iFacet, self._computeFacetMeanResidual_worker,
                           args=(iFacet, fmr_dict.writeonly(), self._facet_grids.readonly(), self._CF.readonly(),
                                 self.DicoImager[iFacet]["SumWeights"], self.DicoImager[iFacet]["SumJonesNorm"], WBAND))
            APP.awaitJobResults("facetmeanresidual:*", progress="Mean per-facet dirties")
            fmr_dict.reload()

            # Build a residual image consisting of multiple continuum bands
            stitchedResidual = self.FacetsToIm_Channel("Dirty")

            if self.VS.MultiFreqMode:
                MeanResidual = np.sum(stitchedResidual * WBAND, axis=0).reshape((1, npol, Npix, Npix))
            else:
                ### (Oleg 24/12/2016: removed the .copy(), why was this needed? Note that in e.g.
                ### ClassImageDeconvMachineMSMF.SubStep(), there is an if-clause such as
                ###    "if self._MeanDirty is not self._CubeDirty: do_expensive_operation"
                ### which the .copy() operation here defeats, so I remove it
                MeanResidual = stitchedResidual  #.copy()
            DicoImages["ImageCube"] = stitchedResidual
            DicoImages["MeanImage"] = MeanResidual
            DicoImages["FacetNorm"] = FacetNorm  # grid-correcting map
            DicoImages["JonesNorm"] = JonesNorm
            return DicoImages

    def getNormDict(self): return self._norm_dict

    def setNormImages(self,DicoImages):
        # There's only one normDict, and its shared between FacetMachine and FacetMachinePSF.
        # it is initialized only once, either here, or in BuildFacetNormImage.
        # So we do nothing if it is already initialized.
        if self._norm_dict is None:
            self._norm_dict = shared_dict.attach("normDict")
        if "FacetNorm" not in self._norm_dict:
            JonesNorm = DicoImages["JonesNorm"]
            nch, npol, nx, ny = DicoImages["ImageCube"].shape
            MeanJonesNorm = np.mean(JonesNorm, axis=0).reshape((1, npol, nx, ny))
            self._norm_dict["JonesNorm"] = JonesNorm
            self._norm_dict["MeanJonesNorm"] = MeanJonesNorm
            
            if "SmoothJonesNorm" in DicoImages.keys():
                self.SmoothJonesNorm=DicoImages["SmoothJonesNorm"]
                if self.AverageBeamMachine is not None:
                    Npix=self.OutImShape[-1]
                    self.AverageBeamMachine.SmoothBeam=self.SmoothJonesNorm.reshape((self.VS.NFreqBands,1,Npix,Npix))
                    self.AverageBeamMachine.MeanSmoothBeam=np.mean(self.AverageBeamMachine.SmoothBeam,axis=0).reshape((1,1,Npix,Npix))

            FacetNorm = DicoImages["FacetNorm"]
            FacetNormReShape = DicoImages["FacetNorm"].reshape([1,1,
                                                                FacetNorm.shape[0],
                                                                FacetNorm.shape[1]])
            # put arrays into shared
            self._norm_dict["FacetNorm"]=FacetNorm
            self._norm_dict["FacetNormReShape"]=FacetNormReShape
            
            self.DoCalcJonesNorm = False

        self.JonesNorm=self._norm_dict["JonesNorm"]
        self.MeanJonesNorm=self._norm_dict["MeanJonesNorm"]
        self.FacetNorm=self._norm_dict["FacetNorm"]
        self.FacetNormReShape=self._norm_dict["FacetNormReShape"]


    def BuildFacetNormImage(self):
        """
        Creates a stitched tesselation weighting map. This can be useful
        to downweight areas where facets overlap (e.g. padded areas)
        before stitching the facets into one map.
        Returns
            ndarray with norm image
        """
        # There's only one normDict, and its shared between FacetMachine and FacetMachinePSF.
        # it is initialized only once, either here, or in BuildFacetNormImage.
        # So we do nothing if it is already initialized.
        if self._norm_dict is None:
            self._norm_dict = shared_dict.attach("normDict")
        if "FacetNorm" not in self._norm_dict:
            print>>log, "  Building Facet-normalisation image"
            nch, npol = self.nch, self.npol
            _, _, NPixOut, NPixOut = self.OutImShape
            # in PSF mode, make the norm image in memory. In normal mode, make it in the shared dict,
            # since the degridding workers require it
            FacetNorm = np.zeros((NPixOut, NPixOut), dtype=self.stitchedType)
            for iFacet in self.DicoImager.keys():
                xc, yc = self.DicoImager[iFacet]["pixCentral"]
                NpixFacet = self.DicoImager[iFacet]["NpixFacetPadded"]
                
                Aedge, Bedge = GiveEdges((xc, yc), NPixOut,
                                         (NpixFacet/2, NpixFacet/2), NpixFacet)
                x0d, x1d, y0d, y1d = Aedge
                x0p, x1p, y0p, y1p = Bedge
                
                SpacialWeigth = self._CF[iFacet]["SW"].T[::-1, :]
                SW = SpacialWeigth[::-1, :].T[x0p:x1p, y0p:y1p]
                FacetNorm[x0d:x1d, y0d:y1d] += np.real(SW)

            self._norm_dict["FacetNorm"]=FacetNorm
            self._norm_dict["FacetNormReShape"]=FacetNorm.reshape([1,1,
                                                                   FacetNorm.shape[0],
                                                                   FacetNorm.shape[1]])





    def FacetsToIm_Channel(self, kind="Dirty",ChanSel=None):
        """
        Preconditions: assumes the stitched tesselation weighting map has been
        created previous
            kind: one of "Jones-amplitude", "Dirty", or "PSF", to create a stitched Jones amplitude, dirty or psf image
        Returns:
            Image cube, which may contain multiple correlations
            and continuum channel bands
        """
        T = ClassTimeIt.ClassTimeIt("FacetsToIm_Channel")
        T.disable()
        Image = self.GiveEmptyMainField()

        nch, npol, NPixOut, NPixOut = self.OutImShape

        if ChanSel is None:
            ChanSel=range(self.VS.NFreqBands)
        
        print>>log, "Combining facets to stitched %s image" % kind

        for Channel in ChanSel:
            ThisSumWeights=self.DicoImager[0]["SumWeights"][Channel][0]
            if ThisSumWeights==0:
                print>>log,ModColor.Str("The sum of the weights are zero for FreqBand #%i, data is all flagged?"%Channel)
                print>>log,ModColor.Str("  (... will skip normalisation for this FreqBand)")
                
        pBAR = ProgressBar(Title="Glue facets")
        NFacets=len(self.DicoImager.keys())
        pBAR.render(0, NFacets)

        numexpr.set_num_threads(self.GD["Parallel"]["NCPU"])  # done in DDF.py

        for iFacet in self.DicoImager.keys():

            SPhe = self._CF[iFacet]["Sphe"]
            InvSPhe = self._CF[iFacet]["InvSphe"]
            SpacialWeigth = self._CF[iFacet]["SW"].T[::-1, :]

            xc, yc = self.DicoImager[iFacet]["pixCentral"]
            NpixFacet = self.DicoGridMachine[iFacet]["Dirty"][0].shape[2]


            Aedge, Bedge = GiveEdges((xc, yc), NPixOut,
                                     (NpixFacet/2, NpixFacet/2), NpixFacet)
            x0main, x1main, y0main, y1main = Aedge
            x0facet, x1facet, y0facet, y1facet = Bedge

            for Channel in ChanSel:
                ThisSumWeights = self.DicoImager[iFacet]["SumWeights"][Channel]
                ThisSumJones = self.DicoImager[iFacet]["SumJonesNorm"][Channel]
                T.timeit("3")
                for pol in xrange(npol):
                    # ThisSumWeights.reshape((nch,npol,1,1))[Channel, pol, 0, 0]
                    if kind == "Jones-amplitude":
                        Im = SpacialWeigth[::-1, :].T[x0facet:x1facet, y0facet:y1facet] * ThisSumJones
                    else:
                        if kind == "Dirty" or kind == "PSF":
                            # make copy since subsequent operations are in-place
                            Im = self.DicoGridMachine[iFacet]["Dirty"][Channel][pol].real.copy()
                        else:
                            raise RuntimeError,"unknown kind=%s argument -- this is a silly bug"%kind
                        # normalize by sum of weights, and Jones weight
                        weights = ThisSumWeights[pol]*np.sqrt(ThisSumJones)
                        # ...and the spatial weights
                        numexpr.evaluate('Im*InvSPhe*SpacialWeigth/weights',out=Im,casting="unsafe")
                        Im[SPhe < 1e-3] = 0
                        # flip axis and extract facet
                        Im = Im[::-1, :].T[x0facet:x1facet, y0facet:y1facet]
                    # add into main image
                    a = Image[Channel, pol, x0main:x1main, y0main:y1main]
                    numexpr.evaluate('a+Im',out=a,casting="unsafe")
                    #Image[Channel, pol, x0main:x1main, y0main:y1main] += Im.real
                    
                    



            pBAR.render(iFacet+1, NFacets)

        for Channel in ChanSel:
            for pol in xrange(npol):
                Image[Channel, pol] /= self._norm_dict["FacetNorm"]

        return Image

    # def GiveNormImage(self):
    #     """
    #     Creates a stitched normalization image of the grid-correction function.
    #     This image should be point-wise divided from the stitched gridded map
    #     to create a grid-corrected map.
    #     Returns:
    #         stitched grid-correction norm image
    #     """
    #     Image = self.GiveEmptyMainField()
    #     nch, npol = self.nch, self.npol
    #     _, _, NPixOut, NPixOut = self.OutImShape
    #     SharedMemName = "%sSpheroidal" % (self.IdSharedMemData)
    #     NormImage = np.zeros((NPixOut, NPixOut), dtype=self.stitchedType)
    #     SPhe = NpShared.GiveArray(SharedMemName)
    #     N1 = self.NpixPaddedFacet
    #
    #     for iFacet in self.DicoImager.keys():
    #
    #         xc, yc = self.DicoImager[iFacet]["pixCentral"]
    #         Aedge, Bedge = GiveEdges((xc, yc), NPixOut, (N1/2, N1/2), N1)
    #         x0d, x1d, y0d, y1d = Aedge
    #         x0p, x1p, y0p, y1p = Bedge
    #
    #         for ch in xrange(nch):
    #             for pol in xrange(npol):
    #                 NormImage[x0d:x1d, y0d:y1d] += SPhe[::-1,
    #                                                     :].T.real[x0p:x1p, y0p:y1p]
    #
    #     return NormImage


    def ReinitDirty(self):
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
        """
        self.SumWeights.fill(0)
        self.IsDirtyInit = True
        self.HasFourierTransformed = False
        # are we creating a new grids dict?
        if self._facet_grids is None:
            self._facet_grids = shared_dict.create("PSFGrid" if self.DoPSF else "Grid")

        for iFacet in self.DicoGridMachine.keys():
            NX = self.DicoImager[iFacet]["NpixFacetPadded"]
            # init or zero grid array
            grid = self._facet_grids.get(iFacet)
            if grid is None:
                grid = self._facet_grids.addSharedArray(iFacet, (self.VS.NFreqBands, self.npol, NX, NX), self.CType)
            else:
                grid.fill(0)
            self.DicoGridMachine[iFacet]["Dirty"] = grid
            self.DicoImager[iFacet]["SumWeights"] = np.zeros((self.VS.NFreqBands, self.npol), np.float64)
            self.DicoImager[iFacet]["SumJones"] = np.zeros((2, self.VS.NFreqBands), np.float64)
            self.DicoImager[iFacet]["SumJonesChan"] = []
            for iMS in xrange(self.VS.nMS):
                nVisChan = self.VS.ListMS[iMS].ChanFreq.size
                self.DicoImager[iFacet]["SumJonesChan"].append(np.zeros((2, nVisChan), np.float64))

    def applySparsification(self, DATA, factor):
        """Computes a sparsification vector for use in the BDA gridder. This is a vector of bools,
        same size as the number of BDA blocks, with a True for every block that will be gridded.
        Blocks ae chosen at random with a probability of 1/factor"""
        if not factor or "BDA.Grid" not in DATA:
            DATA["Sparsification"] = np.array([])
        else:
            # randomly select blocks with 1/sparsification probability
            num_blocks = DATA["BDA.Grid"][0]
            DATA["Sparsification.Grid"] = numpy.random.sample(num_blocks) < 1.0 / factor
            print>> log, "applying sparsification factor of %f to %d BDA grid blocks, left with %d" % (factor, num_blocks, DATA["Sparsification.Grid"].sum())
            #num_blocks = DATA["BDADegrid"][0]
            #DATA["Sparsification.Degrid"] = numpy.random.sample(num_blocks) < 1.0 / factor
            #print>> log, "applying sparsification factor of %f to %d BDA degrid blocks, left with %d" % (factor, num_blocks, DATA["Sparsification.Degrid"].sum())

    def _grid_worker(self, iFacet, DATA, cf_dict, griddict):
        T = ClassTimeIt.ClassTimeIt()
        T.disable()

        ## FFTW wisdom already loaded by main process
        # if FFTW_Wisdom is not None:
        #     pyfftw.import_wisdom(FFTW_Wisdom)
        # T.timeit("%s: import wisdom" % iFacet)

        # Create a new GridMachine
        GridMachine = self._createGridMachine(iFacet, cf_dict=cf_dict,
            bda_grid=DATA["BDA.Grid"], bda_degrid=DATA["BDA.Degrid"])
        T.timeit("%s: create GM" % iFacet)

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

        DecorrMode = self.GD["RIME"]["DecorrMode"]
        if 'F' in DecorrMode or "T" in DecorrMode:
            uvw_dt = DATA["uvw_dt"]
            DT, Dnu = DATA["dt"], DATA["dnu"][0]
            lm_min=None
            if self.GD["RIME"]["DecorrLocation"]=="Edge":
                lm_min=self.DicoImager[iFacet]["lm_min"]
            GridMachine.setDecorr(uvw_dt, DT, Dnu, 
                                  SmearMode=DecorrMode, 
                                  lm_min=lm_min,
                                  lm_PhaseCenter=DATA["lm_PhaseCenter"])

        # DecorrMode = GD["DDESolutions"]["DecorrMode"]
        # if ('F' in DecorrMode) or ("T" in DecorrMode):
        #     uvw_dt = DATA["uvw_dt"]
        #     DT, Dnu = DATA["dt_dnu"]
        #     GridMachine.setDecorr(uvw_dt, DT, Dnu, SmearMode=DecorrMode)

        # Create Jones Matrices Dictionary
        DicoJonesMatrices = None
        Apply_killMS = self.GD["DDESolutions"]["DDSols"]
        Apply_Beam = self.GD["Beam"]["Model"] is not None

        if Apply_killMS or Apply_Beam:
            DicoJonesMatrices = {}
        if Apply_killMS:
            DicoJonesMatrices["DicoJones_killMS"] = DATA["killMS"]
        if Apply_Beam:
            DicoJonesMatrices["DicoJones_Beam"] = DATA["Beam"]

        GridMachine.put(times, uvwThis, visThis, flagsThis, A0A1, W,
                        DoNormWeights=False,
                        DicoJonesMatrices=DicoJonesMatrices,
                        freqs=freqs, DoPSF=self.DoPSF,
                        ChanMapping=ChanMapping,
                        ResidueGrid=griddict[iFacet],
                        sparsification=DATA.get("Sparsification.Grid")
                        )
        T.timeit("put %s" % iFacet)

        T.timeit("Grid")
        Sw = GridMachine.SumWeigths.copy()
        SumJones = GridMachine.SumJones.copy()
        SumJonesChan = GridMachine.SumJonesChan.copy()
        

        return {"iFacet": iFacet, "Weights": Sw, "SumJones": SumJones, "SumJonesChan": SumJonesChan}

    def gridChunkInBackground(self, DATA):
        """
        Grids a chunk of input visibilities onto many facets. Issues jobs to the compute threads.
        Visibility data is already in the data shared dict.

        """
        # wait for any init to finish
        self.awaitInitCompletion()
        # wait for any previous gridding/degridding jobs to finish, if still active
        self.collectGriddingResults()
        self.collectDegriddingResults()
        # run new set of jobs
        self._grid_iMS, self._grid_iChunk = DATA["iMS"], DATA["iChunk"]
        self._grid_job_label = DATA["label"]
        self._grid_job_id = "%s.Grid.%s:" % (self._app_id, self._grid_job_label)
        for iFacet in self.DicoImager.keys():
            APP.runJob("%sF%d" % (self._grid_job_id, iFacet), self._grid_worker,
                            args=(iFacet, DATA.readonly(), self._CF[iFacet].readonly(),
                                  self._facet_grids.readonly()))

    # ##############################################
    # ##### Smooth beam ############################
    def _SmoothAverageBeam_worker(self, DATA, iDir):
        self.AverageBeamMachine.StackBeam(DATA, iDir)

    def StackAverageBeam(self, DATA):
        # the FacetMachinePSF does not have an AverageBeamMachine
        if not self.AverageBeamMachine: 
            return
        # if AverageBeamMachine has loaded a cached SmoothBeam
        if self.AverageBeamMachine.SmoothBeam is not None: 
            return
        # wait for any init to finish
        self.awaitInitCompletion()
        # wait for any previous gridding/degridding jobs to finish, if still active
        self.collectGriddingResults()
        self.collectDegriddingResults()
        # run new set of jobs
        self._smooth_job_label=DATA["label"]
        JobName="StackBeam%sF"%self._smooth_job_label
        for iDir in range(self.AverageBeamMachine.NDir):
            APP.runJob("%s%d" % (JobName,iDir), 
                       self._SmoothAverageBeam_worker,
                       args=(DATA.readonly(), iDir))


    def finaliseSmoothBeam(self):
        # the FacetMachinePSF does not have an AverageBeamMachine
        if not self.AverageBeamMachine: return
        # if AverageBeamMachine has loaded a cached SmoothBeam
        if self.AverageBeamMachine.SmoothBeam is None: 
            if self.AverageBeamMachine.Smooth()=="NoStackedData":
                print>>log,"Has tried to compute the smoothed beam, but there was no stacked beam"
                return
            else:
                print>>log,"Successfully computed the smooth beam"
                
        Npix=self.OutImShape[-1]
        self.SmoothJonesNorm = self.AverageBeamMachine.SmoothBeam.reshape((self.VS.NFreqBands,1,Npix,Npix))
        self.MeanSmoothJonesNorm = self.AverageBeamMachine.MeanSmoothBeam.reshape((1,1,Npix,Npix))

    # ##############################################
    # ##############################################

    # ##############################################
    # ##### Smooth beam ############################
    # def _SmoothAverageBeam_worker(self, datadict_path,iDir):
    #     DATA=shared_dict.attach(datadict_path)
    #     self.AverageBeamMachine.StackBeam(DATA,iDir)



    # def finaliseSmoothBeam(self):
    #     # the FacetMachinePSF does not have an AverageBeamMachine
    #     if not self.AverageBeamMachine: return
    #     # if AverageBeamMachine has loaded a cached SmoothBeam
    #     if self.AverageBeamMachine.SmoothBeam is None: 
    #         if self.AverageBeamMachine.Smooth()=="NoStackedData":
    #             print>>log,"Has tried to compute the smoothed beam, but there was no stacked beam"
    #             return
    #         else:
    #             print>>log,"Successfully computed the smooth beam"
                
    #     Npix=self.OutImShape[-1]
    #     self.SmoothJonesNorm = self.AverageBeamMachine.SmoothBeam.reshape((self.VS.NFreqBands,1,Npix,Npix))

    # ##############################################
    # ##############################################

    def collectGriddingResults(self):
        """
        If any grid workers are still at work, waits for them to finish and collects the results.
        Otherwise does nothing.

        Post conditions:
            Updates the following normalization weights, as produced by the gridding process:
                self.DicoImager[iFacet]["SumWeights"]
                self.DicoImager[iFacet]["SumJones"]
                self.DicoImager[iFacet]["SumJonesChan"][DATA["iMS"]]
        """
        # if this is set to None, then results already collected
        if self._grid_job_id is None:
            return
        # collect results of grid workers
        results = APP.awaitJobResults(self._grid_job_id+"*",progress=
                            ("Grid PSF %s" if self.DoPSF else "Grid %s") % self._grid_job_label)

        for DicoResult in results:
            # if we hit a returned exception, raise it again
            if isinstance(DicoResult, Exception):
                raise DicoResult
            iFacet = DicoResult["iFacet"]
            self.DicoImager[iFacet]["SumWeights"] += DicoResult["Weights"]
            self.DicoImager[iFacet]["SumJones"] += DicoResult["SumJones"]
            self.DicoImager[iFacet]["SumJonesChan"][self._grid_iMS] += DicoResult["SumJonesChan"]
        self._grid_job_id = None

        if self.AverageBeamMachine is not None and \
           self.AverageBeamMachine.SmoothBeam is None and\
           self._smooth_job_label is not None:
            JobName="StackBeam%sF"%self._smooth_job_label
            APP.awaitJobResults(JobName+"*",
                                progress=("Stack Beam %s" % self._smooth_job_label))

        return True

    def _fft_worker(self, iFacet, cf_dict, griddict):
        """
        Fourier transforms the grids currently housed in shared memory
        Precondition:
            Should be called after all data has been gridded
        Returns:
            Dictionary of success and facet identifier
        """
        # reload shared dicts
        GridMachine = self._createGridMachine(iFacet, cf_dict=cf_dict)
        Grid = griddict[iFacet]
        # note that this FFTs in-place
        GridMachine.GridToIm(Grid)
        return {"iFacet": iFacet}

    def fourierTransformInBackground(self):
        '''
        Fourier transforms the individual facet grids in-place.
        Runs background jobs for this.
        '''
        # wait for any previous gridding jobs to finish, if still active
        self.collectGriddingResults()
        # run FFT jobs
        self._fft_job_id = "%s.FFT:" % self._app_id
        for iFacet in self.DicoImager.keys():
            APP.runJob("%sF%d" % (self._fft_job_id, iFacet), self._fft_worker,
                            args=(iFacet, self._CF[iFacet].readonly(), self._facet_grids.readonly()),
                            )
        # APP.awaitJobResults(self._fft_job_id+"*", progress=("FFT PSF" if self.DoPSF else "FFT"))

    def collectFourierTransformResults (self):
        if self._fft_job_id is None:
            return
        # collect results of FFT workers
        # (use label of previous gridding job for the progress bar)
        APP.awaitJobResults(self._fft_job_id+"*", progress=("FFT PSF" if self.DoPSF else "FFT"))
        self._fft_job_id = None

    def _set_model_grid_worker(self, iFacet, model_dict, cf_dict, ChanSel, ToSHMDict=False,ToGrid=False,ApplyNorm=True):
        # We get the psf dict directly from the shared dict name (not from the .path of a SharedDict)
        # because this facet machine is not necessarilly the one where we have computed the PSF
        norm_dict = shared_dict.attach("normDict")
        # extract facet model from model image
        ModelGrid, SumFlux = self._Im2Grid.GiveModelTessel(model_dict["Image"],
                                                           self.DicoImager, iFacet, norm_dict["FacetNorm"],
                                                           cf_dict["Sphe"], cf_dict["SW"], ChanSel=ChanSel,ToGrid=ToGrid,ApplyNorm=ApplyNorm)

        model_dict[iFacet]["SumFlux"] = SumFlux
        if ToSHMDict:
            model_dict[iFacet]["FacetGrid"] = ModelGrid
        return ModelGrid

    def set_model_grid (self,ToGrid=True,ApplyNorm=True):
        self.awaitInitCompletion()

        #modeldict_path=self._model_dict.path
        #cfdict_path=self._CF.path
        #self._model_dict = SharedDict.attach(modeldict_path)

        # create FacetNorm in shared dict if not exist
        self.BuildFacetNormImage()
        nch,_,_,_=self._model_dict["Image"].shape
        ChanSel=range(nch)
        ToSHMDict=True
        
        self._set_model_grid_job_id = "%s.MakeGridModel:" % (self._app_id)
        
        for iFacet in self.DicoImager.keys():
            APP.runJob("%sF%d" % (self._set_model_grid_job_id, iFacet), 
                       self._set_model_grid_worker,
                       args=(iFacet, self._model_dict.readwrite(), self._CF[iFacet].readonly(),
                             ChanSel,ToSHMDict,ToGrid,ApplyNorm))
        APP.awaitJobResults(self._set_model_grid_job_id + "*", progress="Make model grids")


    # #####################################################"
    def _convolveShift_worker(self, iFacet, d_mat, dl,dm,
                              model_dict,DicoImages,cf_dict,
                              RestoredFacetDict,PSFGaussParsAvg):
        Model=model_dict[iFacet]["FacetGrid"]
        _,npol,nx,ny=Model.shape
        Model=np.mean(Model,axis=0).reshape((1,npol,nx,ny))
        Residual=DicoImages["FacetMeanResidual"][iFacet]
        
        majax,minax,PA=PSFGaussParsAvg
        PA+=np.pi/2
        
        ModelConv=ModFFTW.ConvolveGaussianSimpleWrapper(Model, CellSizeRad=self.CellSizeRad,
                                           GaussPars=(majax,minax,PA))
        
        #indx,indy=np.where(self._CF[iFacet]["SW"]!=0)
        #ModelConv[0,0,indx,indy]=ModelConv[0,0,indx,indy]/self._CF[iFacet]["SW"][indx,indy]
        Restored=Residual+ModelConv#/self._CF[iFacet]["SW"]
        
        
        indx,indy=np.where(cf_dict[iFacet]["Sphe"]<1e-3)
        Restored[0,0,indx,indy]=0
        
        # import pylab
        # pylab.clf()
        # pylab.imshow(Restored[0,0])
        # pylab.draw()
        # pylab.show()
        
        if d_mat is not None:
            d=d_mat[iFacet]
            iDir=np.argmin(d)
            if not(dl[iDir]==0. and dm[iDir]==0.):
                Restored=scipy.ndimage.interpolation.shift(Restored, (0,0,dm[iDir],dl[iDir]))
        RestoredFacetDict[iFacet]=Restored
        #Restored.fill(1.)

        

    def giveRestoredFacets(self,DicoImages,PSFGaussParsAvg,ShiftFile=None):
        self.set_model_grid (ToGrid=False,ApplyNorm=False)
        d_mat=None
        if ShiftFile is not None:
            ra_rad,dec_rad,dl,dm=np.genfromtxt(ShiftFile).T
            a1,d1=ra_rad.reshape(-1,1),dec_rad.reshape(-1,1)
            a0=np.array([self.DicoImager[iFacet]["RaDec"][0] for iFacet in sorted(self.DicoImager.keys())]).reshape(-1,1)
            d0=np.array([self.DicoImager[iFacet]["RaDec"][1] for iFacet in sorted(self.DicoImager.keys())]).reshape(-1,1)

            c=np.cos
            s=np.sin
            d_mat=np.arccos(s(d0)*s(d1.T)+c(d0)*c(d1.T)*c(a0-a1.T))
            #d_mat[d_mat==0]=1e10

        RestoredFacetDict = shared_dict.create("RestoredFacetDict")


        for iFacet in self.DicoImager.keys():
            APP.runJob("convolveShiftF%d" % (iFacet), 
                       self._convolveShift_worker,
                       args=(iFacet, d_mat, dl,dm,
                             self._model_dict.readonly(),DicoImages.readonly(),self._CF.readonly(),
                             RestoredFacetDict.readwrite(),PSFGaussParsAvg))#,serial=True)
        APP.awaitJobResults("convolveShiftF*", progress="Build restored facets")

        RestoredFacetDict.reload()
        for iFacet in sorted(self.DicoImager.keys()):
            self._model_dict.reload()
            Restored=RestoredFacetDict[iFacet]
            self.DicoGridMachine[iFacet]["Dirty"]=Restored*self._CF[iFacet]["Sphe"]#/self._CF[iFacet]["SW"]#*self._CF[iFacet]["Sphe"]
            #self.DicoGridMachine[iFacet]["Dirty"]=self.DicoGridMachine[iFacet]["Dirty"]*self._CF[iFacet]["SW"]
            self.DicoImager[iFacet]["SumWeights"]=self.SumWeights.copy()
            self.DicoImager[iFacet]["SumWeights"].fill(1.)
            self.DicoImager[iFacet]["SumJonesNorm"]=np.ones(self.VS.NFreqBands, np.float64)


        Restored=self.FacetsToIm_Channel(kind="Dirty",ChanSel=[0])
        _,npol,nx,ny=Restored.shape
        
        return Restored[0].reshape((1,npol,nx,ny))

    # #####################################################"

    # DeGrid worker that is called by Multiprocessing.Process
    def _degrid_worker(self, iFacet, DATA, cf_dict, ChanSel, modeldict):
        ModelGrid = self._set_model_grid_worker(iFacet, modeldict, cf_dict, ChanSel)

        # Create a new GridMachine
        GridMachine = self._createGridMachine(iFacet, cf_dict=cf_dict,
            ListSemaphores=ClassFacetMachine._degridding_semaphores,
            bda_grid=DATA["BDA.Grid"], bda_degrid=DATA["BDA.Degrid"])

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
        Apply_killMS = self.GD["DDESolutions"]["DDSols"]
        Apply_Beam = self.GD["Beam"]["Model"] is not None

        if Apply_killMS or Apply_Beam:
            DicoJonesMatrices = {}
        if Apply_killMS:
            DicoJonesMatrices["DicoJones_killMS"] = DATA["killMS"]
        if Apply_Beam:
            DicoJonesMatrices["DicoJones_Beam"] = DATA["Beam"]

        DecorrMode = self.GD["RIME"]["DecorrMode"]
        if 'F' in DecorrMode or "T" in DecorrMode:
            uvw_dt = DATA["uvw_dt"]
            DT, Dnu = DATA["dt"], DATA["dnu"][0]
            lm_min=None
            if self.GD["RIME"]["DecorrLocation"]=="Edge":
                lm_min=self.DicoImager[iFacet]["lm_min"]
            GridMachine.setDecorr(uvw_dt, DT, Dnu, 
                                  SmearMode=DecorrMode, 
                                  lm_min=lm_min,
                                  lm_PhaseCenter=DATA["lm_PhaseCenter"])

        GridMachine.get(times, uvwThis, visThis, flagsThis, A0A1,
                          ModelGrid, ImToGrid=False,
                          DicoJonesMatrices=DicoJonesMatrices,
                          freqs=freqs, TranformModelInput="FT",
                          ChanMapping=ChanMapping,
                          sparsification=DATA.get("Sparsification.Degrid")
                        )

        return {"iFacet": iFacet}

    def degridChunkInBackground (self, DATA):
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
        # wait for any init to finish
        self.awaitInitCompletion()

        # run new set of jobs
        ChanSel = sorted(set(DATA["ChanMappingDegrid"]))  # unique channel numbers for degrid

        # create FacetNorm in shared dict if not exist
        self.BuildFacetNormImage()

        self._degrid_job_label = DATA["label"]
        self._degrid_job_id = "%s.Degrid.%s:" % (self._app_id, self._degrid_job_label)

        for iFacet in self.DicoImager.keys():
            APP.runJob("%sF%d" % (self._degrid_job_id, iFacet), self._degrid_worker,
                            args=(iFacet, DATA.readonly(), self._CF[iFacet].readonly(),
                                  ChanSel, self._model_dict.readonly()))#,serial=True)
        #APP.awaitJobResults(self._degrid_job_id + "*", progress="Degrid %s" % self._degrid_job_label)


    def collectDegriddingResults(self):
        """
        If any degrid workers are still at work, waits for them to finish and collects the results.
        Otherwise does nothing.
        """
        # if this is set to None, then results already collected
        if self._degrid_job_id is None:
            return
        # collect results of degrid workers
        APP.awaitJobResults(self._degrid_job_id + "*", progress="Degrid %s" % self._degrid_job_label)
        self._degrid_job_id = None
        return True

