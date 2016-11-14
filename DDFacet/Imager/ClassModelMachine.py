import collections
import itertools

import numpy as np
import pylab
from DDFacet.Other import MyLogger
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
log = MyLogger.getLogger("ClassModelMachine")
from DDFacet.Array import NpParallel
from DDFacet.Array import ModLinAlg
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyPickle
from DDFacet.Other import reformat

from DDFacet.ToolsDir.GiveEdges import GiveEdges

from DDFacet.ToolsDir import ModFFTW
import scipy.ndimage
from SkyModel.Sky import ModRegFile
from pyrap.images import image
from SkyModel.Sky import ClassSM
import os


class ClassModelMachine():
    """
    Interface to ClassModelMachine (in progress)
    GiveModelImage(FreqIn)
        Input:
            FreqIn      = The frequencies at which to return the model image

    ToFile(FileName,DicoIn)
        Input:
            FileName    = The name of the file to write to
            DicoIn      = The dictionary to write to file. If None it writes the current dict in DicoSMStacked to file

    FromFile(FileName)
        Input:
            FileName    = The name of the file to read dict from

    FromDico(DicoIn)
        Input:
            DicoIn      = The dictionary to read in

    """
    def __init__(self, GD=None, Gain=None, GainMachine=None):
        self.GD = GD
        if Gain is None:
            self.Gain = self.GD["ImagerDeconv"]["Gain"]
        else:
            self.Gain = Gain
        self.GainMachine = GainMachine
        self.DicoSMStacked = {}
        self.DicoSMStacked["Comp"] = {}

    def setRefFreq(self, RefFreq, AllFreqs):
        self.RefFreq = RefFreq
        self.DicoSMStacked["RefFreq"] = RefFreq
        self.DicoSMStacked["AllFreqs"] = np.array(AllFreqs)

    def ToFile(self, FileName, DicoIn=None):
        print>>log, "Saving dico model to %s" % FileName
        if DicoIn is None:
            D = self.DicoSMStacked
        else:
            D = DicoIn

        D["ListScales"] = self.ListScales
        D["ModelShape"] = self.ModelShape
        MyPickle.Save(D, FileName)

    def FromFile(self, FileName):
        print>>log, "Reading dico model from %s" % FileName
        self.DicoSMStacked = MyPickle.Load(FileName)
        self.FromDico(self.DicoSMStacked)

    def FromDico(self,DicoSMStacked):
        self.DicoSMStacked = DicoSMStacked
        self.RefFreq = self.DicoSMStacked["RefFreq"]
        self.ListScales = self.DicoSMStacked["ListScales"]
        self.ModelShape = self.DicoSMStacked["ModelShape"]