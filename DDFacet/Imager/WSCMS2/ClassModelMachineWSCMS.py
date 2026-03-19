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
import numba
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("ClassModelMachine")
from DDFacet.ToolsDir import ModFFTW
from DDFacet.Other import MyPickle
from DDFacet.Other import reformat
from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric,GiveEdges
from DDFacet.Imager import ClassModelMachine as ClassModelMachinebase
from DDFacet.Imager import ClassFrequencyMachine
import os


class ClassModelMachine(ClassModelMachinebase.ClassModelMachine):
    def __init__(self,*args,**kwargs):
        ClassModelMachinebase.ClassModelMachine.__init__(self, *args, **kwargs)
        self.DicoSMStacked={}
        self.DicoSMStacked["Type"]="WSCMS2"

    def setRefFreq(self, RefFreq, Force=False):
        if self.RefFreq is not None and not Force:
            print(ModColor.Str("Reference frequency already set to %f MHz" % (self.RefFreq/1e6)), file=log)
            return

        self.RefFreq = RefFreq
        self.DicoSMStacked["RefFreq"] = RefFreq

    def setPSFServer(self, PSFServer):
        self.PSFServer = PSFServer

        _, _, self.Npix_x, self.Npix_y = self.PSFServer.ImageShape
        self.NpixPadded_x = int(np.ceil(self.GD["Facets"]["Padding"] * self.Npix_x))
        self.NpixPadded_y = int(np.ceil(self.GD["Facets"]["Padding"] * self.Npix_y))
        
        # make sure it is odd numbered
        if self.NpixPadded_x % 2 == 0:
            self.NpixPadded_x += 1
        self.Npad_x = (self.NpixPadded_x - self.Npix_x) // 2
        
        if self.NpixPadded_y % 2 == 0:
            self.NpixPadded_y += 1
        self.Npad_y = (self.NpixPadded_y - self.Npix_y) // 2

    def setFreqMachine(self, GridFreqs, DegridFreqs, weights=None, PSFServer=None):
        self.PSFServer = PSFServer
        # Initiaise the Frequency Machine
        self.DegridFreqs = DegridFreqs
        self.GridFreqs = GridFreqs
        self.FreqMachine = ClassFrequencyMachine.ClassFrequencyMachine(GridFreqs, DegridFreqs,
                                                                       self.DicoSMStacked["RefFreq"], self.GD,
                                                                       weights=weights, PSFServer=self.PSFServer)
        self.FreqMachine.set_Method()

        if (self.GD["Freq"]["NBand"] > 1):
            self.Coeffs = np.zeros(self.GD["WSCMS"]["NumFreqBasisFuncs"])
        else:
            self.Coeffs = np.zeros([1])

        self.Nchan = self.FreqMachine.nchan
        self.Npol = 1

        # self.DicoSMStacked["Eval_Degrid"] = self.FreqMachine.Eval_Degrid

    def setFacetMachine(self, FacetMachine, BaseName):
        self.FacetMachine = FacetMachine
        self.BaseName = BaseName


    def ToFile(self, FileName, DicoIn=None):
        print("Saving dico model to %s" % FileName, file=log)
        if DicoIn is None:
            D = self.DicoSMStacked
        else:
            D = DicoIn

        if self.GD is None:
            print("Warning - you are haven't initialised GD before writing to the DicoModel")
        D["GD"] = self.GD
        D["Type"] = "WSCMS2"
        D["ModelShape"] = self.ModelShape
        MyPickle.Save(D, FileName)

    def FromFile(self, FileName):
        print("Reading dico model from %s" % FileName, file=log)
        self.DicoSMStacked = MyPickle.Load(FileName)
        self.FromDico(self.DicoSMStacked)

    def FromDico(self, DicoSMStacked):
        self.DicoSMStacked = DicoSMStacked
        self.RefFreq = self.DicoSMStacked["RefFreq"]
        # self.ListScales = self.DicoSMStacked["ListScales"]
        self.ModelShape = self.DicoSMStacked["ModelShape"]

    def setModelShape(self, ModelShape):
        self.ModelShape = ModelShape
        self.Npix_x,self.Npix_y = self.ModelShape[-2:]
        
        self.NpixPadded_x = int(np.ceil(self.GD["WSCMS"]["Padding"]*self.Npix_x))
        self.NpixPadded_y = int(np.ceil(self.GD["WSCMS"]["Padding"]*self.Npix_y))
        # make sure it is odd numbered
        if not self.NpixPadded_x % 2:
            self.NpixPadded_x += 1
        if not self.NpixPadded_y % 2:
            self.NpixPadded_y += 1
        self.Npad_x = (self.NpixPadded_x - self.Npix_x)//2
        self.Npad_y = (self.NpixPadded_y - self.Npix_y)//2
        
        # self.NpixPSF=self.NpixPSF_x,self.NpixPSF_y = self.PSFServer.NPSF  # need this to initialise the FTMachine
        # self.NpixPaddedPSF_x = int(np.ceil(self.GD["WSCMS"]["Padding"]*self.NpixPSF_x))
        # if not self.NpixPaddedPSF_x % 2:
        #     self.NpixPaddedPSF_x += 1
        # self.NpadPSF_x = (self.NpixPaddedPSF_x - self.NpixPSF_x) // 2

        # self.NpixPaddedPSF_y = int(np.ceil(self.GD["WSCMS"]["Padding"]*self.NpixPSF_y))
        # if not self.NpixPaddedPSF_y % 2:
        #     self.NpixPaddedPSF_y += 1
        # self.NpadPSF_y = (self.NpixPaddedPSF_y - self.NpixPSF_y) // 2

    def AppendComponentToDictStacked(self, key, Sols, iScale, Gain):
        """
        Adds component to model dictionary at a scale specified by Scale.
        The dictionary corresponding to each scale is keyed on pixel values (l,m location tupple).
        Each model component is therefore represented parametrically by a pixel value a scale and a set of coefficients
        describing the spectral axis.
        Currently only Stokes I is supported.
        Args:
            key: the (l,m) centre of the component in pixels
            Sols: Nd array of coeffs with length equal to the number of basis functions representing the component.
            iScale: the scale index
            Gain: clean loop gain

        Added component list to dictionary for particular scale. This dictionary is stored in
        self.DicoSMStacked["Comp"][iScale] and has keys:
            "SolsArray": solutions ndArray with shape [#basis_functions,#stokes_terms]
            "NumComps": scalar keeps tracl of the number of components found at a particular scale
        """
        DicoComp = self.DicoSMStacked.setdefault("Comp", {})

        if iScale not in DicoComp.keys():
            DicoComp[iScale] = {}
            DicoComp[iScale]["NumComps"] = np.zeros(1, np.int16)  # keeps track of number of components at this scale

        if key not in DicoComp[iScale].keys():
            DicoComp[iScale][key] = {}
            DicoComp[iScale][key]["SolsArray"] = np.zeros(Sols.size, np.float32)

        DicoComp[iScale]["NumComps"] += 1
        DicoComp[iScale][key]["SolsArray"] += Sols.ravel() * Gain

    def GiveModelImage(self, FreqIn=None, out=None):
        RefFreq=self.DicoSMStacked["RefFreq"]
        # Default to reference frequency if no input given
        if FreqIn is None:
            FreqIn=np.array([RefFreq], dtype=np.float32)

        FreqIn = np.array([FreqIn.ravel()], dtype=np.float32).flatten()

        DicoComp = self.DicoSMStacked.setdefault("Comp", {})
        _, npol, nx, ny = self.ModelShape

        # The model shape has nchan = len(GridFreqs)
        nchan = FreqIn.size
        if out is not None:  # LB - is this for appending components to an existing model?
            if out.shape != (nchan,npol,nx,ny) or out.dtype != np.float32:
                raise RuntimeError("supplied image has incorrect type (%s) or shape (%s)" % (out.dtype, out.shape))
            ModelImage = out
        else:
            ModelImage = np.zeros((nchan,npol,nx,ny),dtype=np.float32)

        for iScale in DicoComp.keys():
            ScaleModel = np.zeros((nchan, npol, nx, ny), dtype=np.float32)
            # get scale kernel
            if self.GD["WSCMS"]["MultiScale"]:
                sigma = self.DicoSMStacked["Scale_Info"][iScale]["sigma"]
                kernel = self.DicoSMStacked["Scale_Info"][iScale]["kernel"]
                extent = self.DicoSMStacked["Scale_Info"][iScale]["extent"]

            for key in DicoComp[iScale].keys():
                if key != "NumComps":  # LB - dirty dirty hack needs to die!!!
                    Sol = DicoComp[iScale][key]["SolsArray"]
                    # TODO - try soft thresholding components
                    x, y = key
                    try:  # LB - Should we drop support for anything other than polynomials maybe?
                        interp = self.FreqMachine.Eval_Degrid(Sol, FreqIn)
                    except:
                        interp = np.polyval(Sol[::-1], FreqIn/RefFreq)

                    if interp is None:
                        raise RuntimeError("Could not interpolate model onto degridding bands. Inspect your data, check "
                                           "'WSCMS-NumFreqBasisFuncs' or if you think this is a bug report it.")

                    if self.GD["WSCMS"]["MultiScale"] and iScale != 0:
                        Aedge, Bedge = GiveEdgesDissymetric(x, y, nx, ny, extent // 2, extent // 2, extent, extent)

                        x0d, x1d, y0d, y1d = Aedge
                        x0p, x1p, y0p, y1p = Bedge

                        out = np.atleast_1d(interp)[:, None, None, None] * kernel
                        ScaleModel[:, :, x0d:x1d, y0d:y1d] += out[:, :, x0p:x1p, y0p:y1p]
                    else:
                        ScaleModel[:, 0, x, y] += interp

            ModelImage += ScaleModel
        Mask=self.DicoSMStacked.get("Mask",None)
        if Mask is not None:
            Mask=Mask[0]
            for iCh in range(nchan):
                ModelImage[iCh].flat[Mask.flat[:]==1]=0
        return ModelImage
    
    def updateMask(self,Mask):
        Mask0=self.DicoSMStacked.get("Mask",None)
        if Mask0 is not None:
            Mask=~(~Mask0 | ~Mask)
        self.DicoSMStacked["Mask"]=Mask
    
    def GiveSpectralIndexMap(self, GaussPars=[(1, 1, 0)], ResidCube=None,
                             GiveComponents=False, ChannelWeights=None):

        # convert to radians
        ex, ey, pa = GaussPars[0] if isinstance(GaussPars, list) and len(GaussPars) == 1 else GaussPars
        ex *= np.pi/180/np.sqrt(2)/2
        ey *= np.pi/180/np.sqrt(2)/2
        epar = (ex + ey)/2.0
        pa = 0.0

        # get in terms of number of cells
        try:
            CellSizeRad_x,CellSizeRad_y = np.array(self.GD['Image']['Cell']) * np.pi / 648000
        except:
            CellSizeRad_x=CellSizeRad_y = self.GD['Image']['Cell'] * np.pi / 648000
            
        # get Gaussian kernel
        GaussKern = ModFFTW.GiveGauss([self.Npix_x,self.Npix_y], CellSizeRad=(CellSizeRad_x,CellSizeRad_y), GaussPars=(epar, epar, pa), parallel=False)

        # take FT
        Fs = np.fft.fftshift
        iFs = np.fft.ifftshift

        import pyfftw
        nthreads = int(self.GD['Parallel']['NCPU'])
        if not nthreads:
            import multiprocessing
            nthreads = multiprocessing.cpu_count()
        else:
            from multiprocessing.pool import ThreadPool
            import dask

            dask.config.set(pool=ThreadPool(nthreads))

        FFT = lambda x: pyfftw.interfaces.numpy_fft.fft2(x, axes=(-2, -1), planner_effort='FFTW_ESTIMATE', threads=nthreads) #, norm='ortho')
        iFFT = lambda x: pyfftw.interfaces.numpy_fft.ifft2(x, axes=(-2, -1), planner_effort='FFTW_ESTIMATE', threads=nthreads) #, norm='ortho')

        # evaluate model
        ModelImage = self.GiveModelImage(self.GridFreqs)

        # pad GausKern and take FT
        #GaussKern = np.pad(GaussKern, self.Npad, mode='constant')
        GaussKern = np.pad(GaussKern, ((self.Npad_x, self.Npad_x),(self.Npad_y, self.Npad_y)), mode='constant')        
        FTshape_x, FTshape_y = GaussKern.shape
        from scipy import fftpack as FT
        #GaussKernhat = FT.fft2(iFs(GaussKern))
        GaussKernhat = FFT(iFs(GaussKern))

        # pad and FT of ModelImage
        ModelImagehat = np.zeros((self.Nchan, FTshape_x, FTshape_y), dtype=np.complex128)
        ConvModelImage = np.zeros((self.Nchan, self.Npix_x, self.Npix_y), dtype=np.float64)
        Ix = slice(self.Npad_x, -self.Npad_x)
        Iy = slice(self.Npad_y, -self.Npad_y)
        for i in range(self.Nchan):
            #tmp_array = np.pad(ModelImage[i, 0], self.Npad, mode='constant')
            tmp_array = np.pad(ModelImage[i, 0], ((self.Npad_x, self.Npad_x),(self.Npad_y, self.Npad_y)), mode='constant')
            # ModelImagehat[i] = FT.fft2(iFs(tmp_array)) * GaussKernhat
            ModelImagehat[i] = FFT(iFs(tmp_array)) * GaussKernhat
            # ConvModelImage[i] = Fs(FT.ifft2(ModelImagehat[i]))[I, I].real
            ConvModelImage[i] = Fs(iFFT(ModelImagehat[i]))[Ix, Iy].real

        if ResidCube is not None:
            #ConvModelImage += ResidCube.squeeze()
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
            import dask.array as da
            _, ncomps = FitCube.shape
            FitCubeDask = da.from_array(FitCube.T.astype(np.float64),
                                        chunks=(np.maximum(100, ncomps//nthreads), self.Nchan))
            weightsDask = da.from_array(weights.astype(np.float64), chunks=(self.Nchan))
            freqsDask = da.from_array(np.array(self.GridFreqs).astype(np.float64), chunks=(self.Nchan))

            alpha, varalpha, Iref, varIref = fit_spi_components(FitCubeDask, weightsDask,
                                                                freqsDask, self.RefFreq).compute()
        except Exception as e:
            raise(e)

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


###################### Dark magic below this line ###################################
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
