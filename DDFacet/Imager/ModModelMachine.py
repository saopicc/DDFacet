import ClassModelMachineGA
import ClassModelMachineMSMF
from DDFacet.Other import MyLogger
from DDFacet.Other import MyPickle

log= MyLogger.getLogger("GiveModelMachine")

def GiveModelMachine(FileName):
    DicoSMStacked= MyPickle.Load(FileName)
    if DicoSMStacked["Type"]=="GA":
        print>>log,"DicoModel is of type GA"
        return ClassModelMachineGA.ClassModelMachine,DicoSMStacked
    if DicoSMStacked["Type"]=="MSMF":
        print>>log,"DicoModel is of type MSMF"
        return ClassModelMachineMSMF.ClassModelMachine,DicoSMStacked
