import ClassModelMachine
import ClassModelMachineGA
import ClassModelMachineMSMF
import ClassModelMachineMORESANE
import ClassGainMachine
from DDFacet.Other import MyPickle
from DDFacet.Other import MyLogger
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
        self.GAMM = None
        self.MSMFMM = None
        self.MORSANEMM = None

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
        if DicoSMStacked is not None: # If the Dict is provided use it to initialise a model machine
            return self.GiveMM(Mode=DicoSMStacked["Type"])
        else: # If the dict is not provided use the MinorCycleMode to figure out which model machine to initialise
            return self.GiveMM()

    def GiveMM(self,Mode=None):
        if Mode == "GA":
            if self.GAMM is None:
                print>> log, "Initialising GA model machine"
                self.GAMM = ClassModelMachineGA.ClassModelMachine(self.GD,GainMachine=ClassGainMachine.ClassGainMachine())
            else:
                print>> log, "GA model machine already initialised"
            return self.GAMM
        elif Mode == "MSMF":
            if self.MSMFMM is None:
                print>> log, "Initialising MSMF model machine"
                self.MSMFMM = ClassModelMachineMSMF.ClassModelMachine(self.GD,GainMachine=ClassGainMachine.ClassGainMachine())
            else:
                print>> log, "MSMF model machine already initialised"
            return self.MSMFMM
        elif Mode == "MORESANE":
            if self.MORSANEMM is None:
                print>> log, "Initialising MSMF model machine"
                self.MORESANEMM = ClassModelMachineMORESANE.ClassModelMachine(self.GD,GainMachine=ClassGainMachine.ClassGainMachine())
            else:
                print>> log, "MORSANE model machine already initialised"
            return self.MORESANEMM
        else:
            raise NotImplementedError("The %s minor cycle is not currently supported",self.GD["ImagerDeconv"]["MinorCycleMode"])