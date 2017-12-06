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

import ClassModelMachine
import ClassGainMachine
from DDFacet.Other import MyPickle
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("GiveModelMachine")

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
        self.MSMFMM = None
        self.MORSANEMM = None
        self.MUFFINMM = None
        self.HOGBOMMM = None

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
        if DicoSMStacked["Type"]=="GA": 
            print>>log,ModColor.Str("Model is of deprecated type GA, overwriting with type SSD")
            DicoSMStacked["Type"]="SSD"

        if DicoSMStacked is not None: # If the Dict is provided use it to initialise a model machine
            Type = DicoSMStacked["Type"]
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
                print>> log, "Initialising SSD model machine"
                from DDFacet.Imager.SSD import ClassModelMachineSSD
                self.SSDMM = ClassModelMachineSSD.ClassModelMachine(self.GD,GainMachine=ClassGainMachine.ClassGainMachine())
            else:
                print>> log, "SSD model machine already initialised"
            return self.SSDMM
        elif Mode == "HMP":
            if self.MSMFMM is None:
                print>> log, "Initialising HMP model machine"
                from DDFacet.Imager.MSMF import ClassModelMachineMSMF
                self.MSMFMM = ClassModelMachineMSMF.ClassModelMachine(
                    self.GD,
                    GainMachine= ClassGainMachine.ClassGainMachine(GainMin=self.GD["Deconv"]["Gain"]))
            else:
                print>> log, "HMP model machine already initialised"
            return self.MSMFMM
        elif Mode == "MORESANE":
            if self.MORSANEMM is None:
                print>> log, "Initialising MORESANE model machine"
                from DDFacet.Imager.MORESANE import ClassModelMachineMORESANE
                self.MORESANEMM = ClassModelMachineMORESANE.ClassModelMachine(self.GD,GainMachine=ClassGainMachine.ClassGainMachine())
            else:
                print>> log, "MORSANE model machine already initialised"
            return self.MORESANEMM
        elif Mode == "MUFFIN":
            if self.MUFFINMM is None:
                print>> log, "Initialising MUFFIN model machine"
                from DDFacet.Imager.MUFFIN import ClassModelMachineMUFFIN
                self.MUFFINMM = ClassModelMachineMUFFIN.ClassModelMachine(self.GD,GainMachine=ClassGainMachine.ClassGainMachine())
            else:
                print>> log, "MUFFIN model machine already initialised"
            return self.MUFFINMM
        elif Mode == "Hogbom":
            if self.HOGBOMMM is None:
                print>> log, "Initialising HOGBOM model machine"
                from DDFacet.Imager.HOGBOM import ClassModelMachineHogbom
                self.HOGBOMMM = ClassModelMachineHogbom.ClassModelMachine(self.GD,GainMachine=ClassGainMachine.ClassGainMachine())
            else:
                print>> log, "HOGBOM model machine already initialised"
            return self.HOGBOMMM
        else:
            raise NotImplementedError("Unknown model type '%s'"%Mode)
