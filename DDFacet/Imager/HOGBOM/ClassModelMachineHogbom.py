import itertools

import numpy as np
from DDFacet.Other import MyLogger
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassModelMachineHogbom")
from DDFacet.Array import NpParallel
from DDFacet.Array import ModLinAlg
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyPickle
from DDFacet.Other import reformat

from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.Imager import ClassModelMachine as ClassModelMachinebase
from DDFacet.Imager import ClassFrequencyMachine
import scipy.ndimage
from SkyModel.Sky import ModRegFile
from pyrap.images import image
from SkyModel.Sky import ClassSM
import os

class ClassModelMachine(ClassModelMachinebase.ClassModelMachine):
    def __init__(self,*args,**kwargs):
        ClassModelMachinebase.ClassModelMachine.__init__(self, *args, **kwargs)
        self.DicoSMStacked={}
        self.DicoSMStacked["Type"]="Hogbom"

    def setRefFreq(self, RefFreq, Force=False):
        if self.RefFreq is not None and not Force:
            print>>log, ModColor.Str("Reference frequency already set to %f MHz" % (self.RefFreq/1e6))
            return

        self.RefFreq = RefFreq
        self.DicoSMStacked["RefFreq"] = RefFreq

    def setFreqMachine(self,GridFreqs, DegridFreqs):
        # Initiaise the Frequency Machine
        self.DegridFreqs = DegridFreqs
        self.GridFreqs = GridFreqs
        self.FreqMachine = ClassFrequencyMachine.ClassFrequencyMachine(GridFreqs, DegridFreqs, self.DicoSMStacked["RefFreq"], self.GD)
        self.FreqMachine.set_Method(mode=self.GD["Hogbom"]["FreqMode"])

    def ToFile(self, FileName, DicoIn=None):
        print>> log, "Saving dico model to %s" % FileName
        if DicoIn is None:
            D = self.DicoSMStacked
        else:
            D = DicoIn

        D["GD"] = self.GD
        D["Type"] = "Hogbom"
        D["ListScales"] = "Delta"
        D["ModelShape"] = self.ModelShape
        MyPickle.Save(D, FileName)

    def FromFile(self, FileName):
        print>> log, "Reading dico model from %s" % FileName
        self.DicoSMStacked = MyPickle.Load(FileName)
        self.FromDico(self.DicoSMStacked)

    def FromDico(self, DicoSMStacked):
        self.DicoSMStacked = DicoSMStacked
        self.RefFreq = self.DicoSMStacked["RefFreq"]
        self.ListScales = self.DicoSMStacked["ListScales"]
        self.ModelShape = self.DicoSMStacked["ModelShape"]

    def setModelShape(self, ModelShape):
        self.ModelShape = ModelShape

    def AppendComponentToDictStacked(self, key, Fpol, Sols, pol_array_index=0):
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

        try:
            DicoComp=self.DicoSMStacked["Comp"]
        except:
            self.DicoSMStacked["Comp"]={}
            DicoComp=self.DicoSMStacked["Comp"]

        if not (key in DicoComp.keys()):
            DicoComp[key] = {}
            for p in range(npol):
                DicoComp[key]["SolsArray"] = np.zeros((Sols.size, npol), np.float32)
                DicoComp[key]["SumWeights"] = np.zeros((npol), np.float32)

        Weight = 1.
        Gain = self.GainMachine.GiveGain()

        #tmp = Sols.ravel()
        SolNorm = Sols.ravel() * Gain * np.mean(Fpol)

        DicoComp[key]["SumWeights"][pol_array_index] += Weight
        DicoComp[key]["SolsArray"][:, pol_array_index] += Weight * SolNorm


    def GiveModelImage(self, FreqIn=None, DoAbs=False, out=None):
        if DoAbs:
            f_apply = np.abs
        else:
            f_apply = lambda x: x

        RefFreq=self.DicoSMStacked["RefFreq"]
        # Default to reference frequency if no input given
        if FreqIn is None:
            FreqIn=np.array([RefFreq], dtype=np.float32)

        FreqIn = np.array([FreqIn.ravel()], dtype=np.float32).flatten()

        DicoComp = self.DicoSMStacked["Comp"]
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
                #tmp = self.FreqMachine.Eval_Degrid(Sol, FreqIn)
                interp = self.FreqMachine.Eval_Degrid(Sol, FreqIn)
                if interp is None:
                    raise RuntimeError("Could not interpolate model onto degridding bands. Inspect your data, check 'Hogbom-PolyFitOrder' or "
                                       "if you think this is a bug report it.")
                ModelImage[:, pol, x, y] += f_apply(interp)

        return ModelImage

    def GiveSpectralIndexMap(self, threshold=0.1, save_dict=True):
        # Get the model image
        IM = self.GiveModelImage(self.FreqMachine.Freqsp)
        nchan, npol, Nx, Ny = IM.shape

        # Fit the alpha map
        self.FreqMachine.FitAlphaMap(IM[:, 0, :, :],
                                     threshold=threshold)  # should set threshold based on SNR of final residual

        if save_dict:
            FileName = self.GD['Output']['Name'] + ".Dicoalpha"
            print>> log, "Saving componentwise SPI map to %s" % FileName

            MyPickle.Save(self.FreqMachine.alpha_dict, FileName)

        return self.FreqMachine.weighted_alpha_map.reshape((1, 1, Nx, Ny))

        # f0 = self.DicoSMStacked["AllFreqs"].min()
        # f1 = self.DicoSMStacked["AllFreqs"].max()
        # M0 = self.GiveModelImage(f0)
        # M1 = self.GiveModelImage(f1)
        # if DoConv:
        #     M0 = ModFFTW.ConvolveGaussian(M0, CellSizeRad=CellSizeRad, GaussPars=GaussPars)
        #     M1 = ModFFTW.ConvolveGaussian(M1, CellSizeRad=CellSizeRad, GaussPars=GaussPars)
        #
        # # compute threshold for alpha computation by rounding DR threshold to .1 digits (i.e. 1.65e-6 rounds to 1.7e-6)
        # minmod = float("%.1e" % (abs(M0.max()) / MaxDR))
        # # mask out pixels above threshold
        # mask = (M1 < minmod) | (M0 < minmod)
        # print>> log, "computing alpha map for model pixels above %.1e Jy (based on max DR setting of %g)" % (
        # minmod, MaxDR)
        # with np.errstate(invalid='ignore'):
        #     alpha = (np.log(M0) - np.log(M1)) / (np.log(f0 / f1))
        # alpha[mask] = 0
        # # mask out |alpha|>MaxSpi. These are not physically meaningful anyway
        # mask = alpha > MaxSpi
        # alpha[mask] = MaxSpi
        # masked = mask.any()
        # mask = alpha < -MaxSpi
        # alpha[mask] = -MaxSpi
        # if masked or mask.any():
        #     print>> log, ModColor.Str("WARNING: some alpha pixels outside +/-%g. Masking them." % MaxSpi, col="red")
        # return alpha

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

        print>> log, "Adding previously subtracted components"
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
