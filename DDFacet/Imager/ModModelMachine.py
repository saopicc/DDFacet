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

from DDFacet.Imager import ClassModelMachine
from DDFacet.Imager import ClassGainMachine
from DDFacet.Other import MyPickle
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("GiveModelMachine")

class ClassModModelMachine():
    """
        This is the factory class for ModelMachine. Basically give it a dictionary containing the components of a model image
        and it instantiates and returns a copy of the correct ModelMachine. Each pickled dictionary should contain a field
        labelling which deconvolution algorithm it corresponds to.
    """
    def __init__(self,GD=None):
        """
        Input:
            GD          = Global dictionary
        """
        self.GD = GD
        self.SSDMM = None
        self.SSD2MM = None
        self.MSMFMM = None
        self.MultiSliceMM = None
        self.MORSANEMM = None
        self.HOGBOMMM = None
        self.WSCMSMM = None

    def GiveInitialisedMMFromFile(self,FileName):
        """
        Initialise a model machine from a file
        Input:
            FileName    = The file to read
        """


        DicoSMStacked = MyPickle.Load(FileName)
        if self.GD is None:
            self.GD=DicoSMStacked["GD"]
        MM=self.GiveMMFromDico(DicoSMStacked)
        MM.FromDico(DicoSMStacked)
        return MM

    def GiveMMFromFile(self,FileName=None):
        """
        Initialise a model machine from a file
        Input:
            FileName    = The file to read
        """
        if FileName is not None:
            DicoSMStacked = MyPickle.Load(FileName)
            return self.GiveMMFromDico(DicoSMStacked)
        else:
            return self.GiveMMFromDico()


    def GiveMMFromDico(self,DicoSMStacked=None):
        """
        Initialise a model machine from a dictionary
        Input:
            DicoSMStacked   = Dictionary to instantiate ModelMachine with
        """
        def safe_encode(s): 
                import six
                return s.decode() if isinstance(s, bytes) and six.PY3 else s
        Type = safe_encode(DicoSMStacked.get("Type", DicoSMStacked.get(b"Type", None)))
        if Type=="GA": 
            print(ModColor.Str("Model is of deprecated type GA, overwriting with type SSD"), file=log)
            DicoSMStacked["Type"]="SSD"

        if DicoSMStacked is not None: # If the Dict is provided use it to initialise a model machine
            Type = safe_encode(DicoSMStacked.get("Type", DicoSMStacked.get(b"Type", None)))
            # backwards compatibility
            if Type == "GA":
                Type = "SSD"
            elif Type == "MSMF":
                Type = "HMP"
            return self.GiveMM(Type)
        else: # If the dict is not provided use the MinorCycleMode to figure out which model machine to initialise
            return self.GiveMM(self.GD["Deconv"]["Mode"])

    def GiveMM(self,Mode=None):
        if Mode == "SSD":
            if self.SSDMM is None:
                print("Initialising SSD model machine", file=log)
                from DDFacet.Imager.SSD import ClassModelMachineSSD
                self.SSDMM = ClassModelMachineSSD.ClassModelMachine(self.GD,GainMachine=ClassGainMachine.get_instance())
            else:
                print("SSD model machine already initialised", file=log)
            return self.SSDMM
        elif Mode == "SSD2":
            if self.SSD2MM is None:
                print("Initialising SSD2 model machine", file=log)
                from DDFacet.Imager.SSD2 import ClassModelMachineSSD
                self.SSD2MM = ClassModelMachineSSD.ClassModelMachine(self.GD,GainMachine=ClassGainMachine.get_instance())
            else:
                print("SSD2 model machine already initialised", file=log)
            return self.SSD2MM
        elif Mode == "HMP":
            if self.MSMFMM is None:
                print("Initialising HMP model machine", file=log)
                from DDFacet.Imager.MSMF import ClassModelMachineMSMF
                self.MSMFMM = ClassModelMachineMSMF.ClassModelMachine(
                    self.GD,
                    GainMachine= ClassGainMachine.get_instance())
            else:
                print("HMP model machine already initialised", file=log)
            return self.MSMFMM
        elif Mode == "MultiSlice":
            if self.MultiSliceMM is None:
                print("Initialising MultiSlice model machine", file=log)
                from DDFacet.Imager.MultiSliceDeconv import ClassModelMachineMultiSlice
                self.MultiSliceMM = ClassModelMachineMultiSlice.ClassModelMachine(
                    self.GD,
                    GainMachine= ClassGainMachine.ClassGainMachine.get_instance())
            else:
                print("MultiSlice model machine already initialised", file=log)
            return self.MultiSliceMM
        elif Mode == "Hogbom":
            if self.HOGBOMMM is None:
                print("Initialising HOGBOM model machine", file=log)
                from DDFacet.Imager.HOGBOM import ClassModelMachineHogbom
                self.HOGBOMMM = ClassModelMachineHogbom.ClassModelMachine(self.GD,GainMachine=ClassGainMachine.get_instance())
            else:
                print("HOGBOM model machine already initialised", file=log)
            return self.HOGBOMMM
        elif Mode == "WSCMS":
            if self.WSCMSMM is None:
                print("Initialising WSCMS model machine", file=log)
                from DDFacet.Imager.WSCMS import ClassModelMachineWSCMS
                self.WSCMSMM = ClassModelMachineWSCMS.ClassModelMachine(self.GD,GainMachine=ClassGainMachine.get_instance())
            else:
                print("WSCMS model machine already initialised", file=log)
            return self.WSCMSMM
        else:
            raise NotImplementedError("Unknown model type '%s'"%Mode)
