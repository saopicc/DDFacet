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

import itertools
import numpy as np
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("ClassModelMachineHogbom")
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Other import MyPickle
from DDFacet.Other import reformat
from DDFacet.Imager import ClassModelMachine as ClassModelMachinebase
from DDFacet.Imager import ClassFrequencyMachine
import scipy.ndimage
import os
from pyrap.images import image # don't remove - part of KMS requirements
from SkyModel.Sky import ClassSM # don't remove - part of KMS requirements

class ClassModelMachine(ClassModelMachinebase.ClassModelMachine):
    def __init__(self,*args,**kwargs):
        ClassModelMachinebase.ClassModelMachine.__init__(self, *args, **kwargs)
        self.DicoSMStacked={}
        self.DicoSMStacked["Type"]="Hogbom"

    def setRefFreq(self, RefFreq, Force=False):
        if self.RefFreq is not None and not Force:
            print(ModColor.Str("Reference frequency already set to %f MHz" % (self.RefFreq/1e6)), file=log)
            return

        self.RefFreq = RefFreq
        self.DicoSMStacked["RefFreq"] = RefFreq

    def setPSFServer(self, PSFServer):
        self.PSFServer = PSFServer

        _, _, self.Npix, _ = self.PSFServer.ImageShape
        self.NpixPadded = int(np.ceil(self.GD["Facets"]["Padding"] * self.Npix))
        # make sure it is odd numbered
        if self.NpixPadded % 2 == 0:
            self.NpixPadded += 1
        self.Npad = (self.NpixPadded - self.Npix) // 2

    def setFreqMachine(self, GridFreqs, DegridFreqs, weights=None, PSFServer=None):
        self.PSFServer = PSFServer
        # Initiaise the Frequency Machine
        self.DegridFreqs = DegridFreqs
        self.GridFreqs = GridFreqs
        self.FreqMachine = ClassFrequencyMachine.ClassFrequencyMachine(GridFreqs, DegridFreqs,
                                                                       self.DicoSMStacked["RefFreq"], self.GD,
                                                                       weights=weights, PSFServer=self.PSFServer)
        self.FreqMachine.set_Method()

        if np.size(self.GridFreqs) > 1:
            self.Coeffs = np.zeros(self.GD["Hogbom"]["PolyFitOrder"])
        else:
            self.Coeffs = np.zeros([1])

        self.Nchan = self.FreqMachine.nchan
        self.Npol = 1

    def ToFile(self, FileName, DicoIn=None):
        print("Saving dico model to %s" % FileName, file=log)
        if DicoIn is None:
            D = self.DicoSMStacked
        else:
            D = DicoIn

        if self.GD is None:
            print("Warning - You have not initialised self.GD in ModelMachine so " \
                        "we can't write it to the DicoModel", file=log)

        D["GD"] = self.GD
        D["Type"] = "Hogbom"
        D["ListScales"] = "Delta"
        D["ModelShape"] = self.ModelShape
        MyPickle.Save(D, FileName)

    def FromFile(self, FileName):
        print("Reading dico model from %s" % FileName, file=log)
        self.DicoSMStacked = MyPickle.Load(FileName)
        self.FromDico(self.DicoSMStacked)

    def FromDico(self, DicoSMStacked):
        self.DicoSMStacked = DicoSMStacked
        self.RefFreq = self.DicoSMStacked["RefFreq"]
        self.ListScales = self.DicoSMStacked["ListScales"]
        self.ModelShape = self.DicoSMStacked["ModelShape"]

    def setModelShape(self, ModelShape):
        self.ModelShape = ModelShape

    def AppendComponentToDictStacked(self, key, Sols, pol_array_index=0):
        """
        Adds component to model dictionary (with key l,m location tupple). Each
        component may contain #basis_functions worth of solutions. Note that
        each basis solution will have multiple Stokes components associated to it.
        Args:
            key: the (l,m) centre of the component
            Fpol: Weight of the solution
            Sols: Nd array of solutions with length equal to the number of basis functions representing the component.
            pol_array_index: Index of the polarization (assumed 0 <= pol_array_index < number of Stokes terms in the model)
        Post conditions:
        Added component list to dictionary (with keys (l,m) coordinates). This dictionary is stored in
        self.DicoSMStacked["Comp"] and has keys:
            "SolsArray": solutions ndArray with shape [#basis_functions,#stokes_terms]
            "SumWeights": weights ndArray with shape [#stokes_terms]
        """
        nchan, npol, nx, ny = self.ModelShape
        if not (pol_array_index >= 0 and pol_array_index < npol):
            raise ValueError("Pol_array_index must specify the index of the slice in the "
                             "model cube the solution should be stored at. Please report this bug.")

        DicoComp = self.DicoSMStacked.setdefault("Comp", {})

        if not (key in DicoComp.keys()):
            DicoComp[key] = {}
            for p in range(npol):
                DicoComp[key]["SolsArray"] = np.zeros((Sols.size, npol), np.float32)
                DicoComp[key]["SumWeights"] = np.zeros((npol), np.float32)

        SolNorm = Sols.ravel() * self.GD["Deconv"]["Gain"]

        DicoComp[key]["SumWeights"][pol_array_index] += 1.0
        DicoComp[key]["SolsArray"][:, pol_array_index] += SolNorm

    def GiveModelList(self, FreqIn=None, DoAbs=False, threshold=0.1):
        """
        Iterates through components in the "Comp" dictionary of DicoSMStacked,
        returning a list of model sources in tuples looking like
        (model_type, coord, flux, ref_freq, alpha, model_params).

        model_type is obtained from self.ListScales
        coord is obtained from the keys of "Comp"
        flux is obtained from the entries in Comp["SolsArray"]
        ref_freq is obtained from DicoSMStacked["RefFreq"]
        alpha is obtained from self.ListScales
        model_params is obtained from self.ListScales

        If multiple scales exist, multiple sources will be created
        at the same position, but different fluxes, alphas etc.

        """
            
        DicoComp = self.DicoSMStacked["Comp"]
        ref_freq = self.DicoSMStacked["RefFreq"]
        
        if FreqIn is None:
           FreqIn=np.array([ref_freq], dtype=np.float32)

        # Assumptions:
        # DicoSMStacked is a dictionary of "Solution" dictionaries
        # keyed on (l, m), corresponding to some point  source. 
        # Components associated with the source for each scale are
        # located in self.ListScales.

        def _model_map(coord, component):
            """
            Given a coordinate and component obtained from DicoMap
            returns a tuple with the following information
            (ModelType, coordinate, vector of STOKES solutions per basis function, alpha, shape data)
            """
            sa = component["SolsArray"]
            return [("Delta",                         # type
                     coord,                           # coordinate
                     self.FreqMachine.Eval_Degrid(sa, FreqIn).flatten(), # only a solution for I
                     ref_freq,                        # reference frequency
                     np.nan,                          # supposed to be alpha estimate but not used anymore
                     None)]                           # shape

        # Lazily iterate through DicoComp entries and associated ListScales and SolsArrays,
        # assigning values to arrays
        source_iter = itertools.chain.from_iterable(_model_map(coord, comp)
            for coord, comp in getattr(DicoComp, "iteritems", DicoComp.items)())

        # Create list with iterator results
        return [s for s in source_iter]


    def GiveModelImage(self, FreqIn=None, out=None):

        RefFreq=self.DicoSMStacked["RefFreq"]
        # Default to reference frequency if no input given
        if FreqIn is None:
            FreqIn=np.array([RefFreq], dtype=np.float32)

        FreqIn = np.array([np.array(FreqIn).ravel()], dtype=np.float32).flatten()

        DicoComp = self.DicoSMStacked.setdefault("Comp", {})
        _, npol, nx, ny = self.ModelShape

        # The model shape has nchan=len(GridFreqs)
        nchan = FreqIn.size
        if out is not None:
            if out.shape != (nchan,npol,nx,ny) or out.dtype != np.float32:
                raise RuntimeError("supplied image has incorrect type (%s) or shape (%s)" % (out.dtype, out.shape))
            ModelImage = out
        else:
            ModelImage = np.zeros((nchan,npol,nx,ny),dtype=np.float32)
        DicoSM = {}
        for key in DicoComp.keys():
            for pol in range(npol):
                Sol = DicoComp[key]["SolsArray"][:, pol]  # /self.DicoSMStacked[key]["SumWeights"]
                x, y = key

                try:
                    interp = self.FreqMachine.Eval_Degrid(Sol, FreqIn)
                except:
                    interp = np.polyval(Sol[::-1], FreqIn / RefFreq)

                if interp is None:
                    raise RuntimeError("Could not interpolate model onto degridding bands. Inspect your data, check "
                                       "'Hogbom-NumFreqBasisFuncs' or if you think this is a bug report it.")
                else:
                    ModelImage[:, pol, x, y] += interp

        return ModelImage

    def GiveSpectralIndexMap(self, GaussPars=[(1, 1, 0)], ResidCube=None,
                             GiveComponents=False, ChannelWeights=None):

        # convert to radians
        ex, ey, pa = GaussPars
        ex *= np.pi/180/np.sqrt(2)/2
        ey *= np.pi/180/np.sqrt(2)/2
        epar = (ex + ey)/2.0
        pa = 0.0

        # get in terms of number of cells
        CellSizeRad = self.GD['Image']['Cell'] * np.pi / 648000

        # get Gaussian kernel
        GaussKern = ModFFTW.GiveGauss(self.Npix, CellSizeRad=CellSizeRad, GaussPars=(epar, epar, pa), parallel=False)

        # take FT
        Fs = np.fft.fftshift
        iFs = np.fft.ifftshift

        # evaluate model
        ModelImage = self.GiveModelImage(self.GridFreqs)

        # pad GausKern and take FT
        GaussKern = np.pad(GaussKern, self.Npad, mode='constant')
        FTshape, _ = GaussKern.shape
        from scipy import fftpack as FT
        GaussKernhat = FT.fft2(iFs(GaussKern))

        # pad and FT of ModelImage
        ModelImagehat = np.zeros((self.Nchan, FTshape, FTshape), dtype=np.complex128)
        ConvModelImage = np.zeros((self.Nchan, self.Npix, self.Npix), dtype=np.float64)
        I = slice(self.Npad, -self.Npad)
        for i in range(self.Nchan):
            tmp_array = np.pad(ModelImage[i, 0], self.Npad, mode='constant')
            ModelImagehat[i] = FT.fft2(iFs(tmp_array)) * GaussKernhat
            ConvModelImage[i] = Fs(FT.ifft2(ModelImagehat[i]))[I, I].real

        if ResidCube is not None:
            ConvModelImage += ResidCube.squeeze()
            RMS = np.std(ResidCube.flatten())
            Threshold = self.GD["SPIMaps"]["AlphaThreshold"] * RMS
        else:
            AbsModel = np.abs(ModelImage).squeeze()
            MinAbsImage = np.amin(AbsModel, axis=0)
            RMS = np.min(np.abs(MinAbsImage.flatten())) # base cutoff on smallest value in model
            Threshold = self.GD["SPIMaps"]["AlphaThreshold"] * RMS

        # get minimum along any freq axis
        MinImage = np.amin(ConvModelImage, axis=0)
        MaskIndices = np.argwhere(MinImage > Threshold)
        FitCube = ConvModelImage[:, MaskIndices[:, 0], MaskIndices[:, 1]]


        if ChannelWeights is None:
            weights = np.ones(self.Nchan, dtype=np.float32)
        else:
            weights = ChannelWeights.astype(np.float32)
            if ChannelWeights.size != self.Nchan:
                import warnings
                warnings.warn("The provided channel weights are of incorrect length. Ignoring weights.", RuntimeWarning)
                weights = np.ones(self.Nchan, dtype=np.float32)

        try:
            import traceback
            from africanus.model.spi.dask import fit_spi_components
            NCPU = self.GD["Parallel"]["NCPU"]
            if NCPU:
                from multiprocessing.pool import ThreadPool
                import dask

                dask.config.set(pool=ThreadPool(NCPU))
            else:
                import multiprocessing
                NCPU = multiprocessing.cpu_count()

            import dask.array as da
            _, ncomps = FitCube.shape
            FitCubeDask = da.from_array(FitCube.T.astype(np.float64), chunks=(ncomps//NCPU, self.Nchan))
            weightsDask = da.from_array(weights.astype(np.float64), chunks=(self.Nchan))
            freqsDask = da.from_array(np.array(self.GridFreqs).astype(np.float64), chunks=(self.Nchan))

            alpha, varalpha, Iref, varIref = fit_spi_components(FitCubeDask, weightsDask,
                                                                freqsDask, self.RefFreq,
                                                                dtype=np.float64).compute()
        except Exception as e:
            traceback_str = traceback.format_exc(e)
            print("Warning - Failed at importing africanus spi fitter. This could be an issue with the dask " \
                        "version. Falling back to (slow) scipy version", file=log)
            print("Original traceback - ", traceback_str, file=log)
            alpha, varalpha, Iref, varIref = self.FreqMachine.FitSPIComponents(FitCube, self.GridFreqs, self.RefFreq)

        _, _, nx, ny = ModelImage.shape
        alphamap = np.zeros([nx, ny])
        Irefmap = np.zeros([nx, ny])
        alphastdmap = np.zeros([nx, ny])
        Irefstdmap = np.zeros([nx, ny])

        alphamap[MaskIndices[:, 0], MaskIndices[:, 1]] = alpha
        Irefmap[MaskIndices[:, 0], MaskIndices[:, 1]] = Iref
        alphastdmap[MaskIndices[:, 0], MaskIndices[:, 1]] = np.sqrt(varalpha)
        Irefstdmap[MaskIndices[:, 0], MaskIndices[:, 1]] = np.sqrt(varIref)

        if GiveComponents:
            return alphamap[None, None], alphastdmap[None, None], alpha
        else:
            return alphamap[None, None], alphastdmap[None, None]

    def PutBackSubsComps(self):
        # if self.GD["Data"]["RestoreDico"] is None: return

        SolsFile = self.GD["DDESolutions"]["DDSols"]
        if not (".npz" in SolsFile):
            Method = SolsFile
            ThisMSName = reformat.reformat(os.path.abspath(self.GD["Data"]["MS"]), LastSlash=False)
            SolsFile = "%s/killMS.%s.sols.npz" % (ThisMSName, Method)
        DicoSolsFile = np.load(SolsFile)
        SourceCat = DicoSolsFile["SourceCatSub"]
        SourceCat = SourceCat.view(np.recarray)
        # RestoreDico=self.GD["Data"]["RestoreDico"]
        RestoreDico = DicoSolsFile["ModelName"][()][0:-4] + ".DicoModel"

        print("Adding previously subtracted components", file=log)
        ModelMachine0 = ClassModelMachine(self.GD)

        ModelMachine0.FromFile(RestoreDico)

        _, _, nx0, ny0 = ModelMachine0.DicoSMStacked["ModelShape"]

        _, _, nx1, ny1 = self.ModelShape
        dx = nx1 - nx0

        for iSource in range(SourceCat.shape[0]):
            x0 = SourceCat.X[iSource]
            y0 = SourceCat.Y[iSource]

            x1 = x0 + dx
            y1 = y0 + dx

            if not ((x1, y1) in self.DicoSMStacked["Comp"].keys()):
                self.DicoSMStacked["Comp"][(x1, y1)] = ModelMachine0.DicoSMStacked["Comp"][(x0, y0)]
            else:
                self.DicoSMStacked["Comp"][(x1, y1)] += ModelMachine0.DicoSMStacked["Comp"][(x0, y0)]

    def ToNPYModel(self,FitsFile,SkyModel,BeamImage=None):
        """ Makes a numpy model for use for killms calibration using SkyModel/MakeModel.py """
        nchan, npol, nx, ny = self.ModelShape
        assert nx == ny, "Only works with square images for now"
        self.Npix = nx 
        self.Nchan = len(self.GridFreqs)
        self.NpixPadded = int(np.ceil(self.GD["Facets"]["Padding"] * self.Npix))
        # make sure it is odd numbered
        if self.NpixPadded % 2 == 0:
            self.NpixPadded += 1
        self.Npad = (self.NpixPadded - self.Npix) // 2

        AlphaMap=self.GiveSpectralIndexMap(GaussPars=(self.GD["Image"]["Cell"]/3600*5,
                                                      self.GD["Image"]["Cell"]/3600*5,
                                                      0))[0] #kludgy 5 beams worth of pixels
        ModelMap=self.GiveModelImage()
        nch,npol,_,_=ModelMap.shape

        for ch in range(nch):
            for pol in range(npol):
                ModelMap[ch,pol]=ModelMap[ch,pol][::-1]#.T
                AlphaMap[ch,pol]=AlphaMap[ch,pol][::-1]#.T

        if BeamImage is not None:
            ModelMap*=(BeamImage)

        im=image(FitsFile)
        pol,freq,decc,rac=im.toworld((0,0,0,0))

        Lx,Ly=np.where(ModelMap[0,0]!=0)

        X=np.array(Lx)
        Y=np.array(Ly)

        #pol,freq,decc1,rac1=im.toworld((0,0,1,0))
        dx=abs(im.coordinates().dict()["direction0"]["cdelt"][0])

        SourceCat=np.zeros((X.shape[0],),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('Sref',np.float),('I',np.float),('Q',np.float),\
                                           ('U',np.float),('V',np.float),('RefFreq',np.float),('alpha',np.float),('ESref',np.float),\
                                           ('Ealpha',np.float),('kill',np.int),('Cluster',np.int),('Type',np.int),('Gmin',np.float),\
                                           ('Gmaj',np.float),('Gangle',np.float),("Select",np.int),('l',np.float),('m',np.float),("Exclude",bool),
                                           ("X",np.int32),("Y",np.int32)])
        SourceCat=SourceCat.view(np.recarray)

        IndSource=0

        SourceCat.RefFreq[:]=self.DicoSMStacked["RefFreq"]
        _,_,nx,ny=ModelMap.shape

        for iSource in range(X.shape[0]):
            x_iSource,y_iSource=X[iSource],Y[iSource]
            _,_,dec_iSource,ra_iSource=im.toworld((0,0,y_iSource,x_iSource))
            SourceCat.ra[iSource]=ra_iSource
            SourceCat.dec[iSource]=dec_iSource
            SourceCat.X[iSource]=(nx-1)-X[iSource]
            SourceCat.Y[iSource]=Y[iSource]

            Flux=ModelMap[0,0,x_iSource,y_iSource]
            Alpha=AlphaMap[0,0,x_iSource,y_iSource]
            SourceCat.I[iSource]=Flux
            SourceCat.alpha[iSource]=Alpha


        SourceCat=(SourceCat[SourceCat.ra!=0]).copy()
        np.save(SkyModel,SourceCat)
        self.AnalyticSourceCat=ClassSM.ClassSM(SkyModel)
