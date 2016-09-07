import ClassModelMachine
import ClassModelMachineGA
import ClassModelMachineMSMF
import ClassModelMachineMORESANE
from DDFacet.Other import MyPickle
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("GiveModelMachine")

class ClassModModelMachine():
    """
        This is the factory class for ModelMachine. Basically give it a dictionary containing the components of a model image
        and it instantiates and returns a copy of the correct ModelMachine. Each pickled dictionary should contain a field
        labelling which deconvolution algorithm it corresponds to.
    """
    def __init__(self,GD=None,FileName=None,DicoIn=None):
        """
        Input:
            GD          = Global dictionary
            FileName    = The file to read
            DicoIn      = Dictionary to instantiate ModelMachine with
        """
        self.GD = GD

    def GiveModelFromFile(self,FileName):
        """
        Input:
            FileName    = The file to read
        """
        DicoSMStacked = MyPickle.Load(FileName)
        return self.GiveModelFromDico(DicoSMStacked)


    def GiveModelFromDico(self,DicoSMStacked):
        """
        Input:
            DicoSMStacked   = Dictionary to instantiate ModelMachine with
        """
        if DicoSMStacked["Type"]=="GA":
            print>>log,"DicoModel is of type GA"
            return ClassModelMachineGA.ClassModelMachine,DicoSMStacked
        if DicoSMStacked["Type"]=="MSMF":
            print>>log,"DicoModel is of type MSMF"
            return ClassModelMachineMSMF.ClassModelMachine,DicoSMStacked
        if DicoSMStacked["Type"] == "MORESANE":
            print>> log, "DicoModel is of type MORESANE"
            return ClassModelMachineMORESANE.ClassModelMachine, DicoSMStacked
