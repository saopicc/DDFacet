import numpy as np
from DDFacet.Other import logger
import pickle as cPickle
log=logger.getLogger("ImageNoiseMachineMultiField")
import six
from DDFacet.Imager import ClassImageNoiseMachine
import copy

class ClassImageNoiseMachineMultiField():
    def __init__(self, GD, VS=None,LModelMachine=None, DegridFreqs=None,
                 GridFreqs=None, MainCache=None,
                 DicoFields=None):

        self.GD=GD
        self.DicoFields=DicoFields
        self.NFields=len(self.DicoFields)
        self.VS=VS
        self.LModelMachine=LModelMachine
        self.LImageNoiseMachine=[]
        for iField in range(self.NFields):
            GD=copy.deepcopy(self.GD)
            GD["Image"]["iField"]=iField
            INM=ClassImageNoiseMachine.ClassImageNoiseMachine(GD,
                                                              self.LModelMachine[iField],
                                                              DegridFreqs=DegridFreqs,
                                                              GridFreqs=GridFreqs,
                                                              MainCache=MainCache,
                                                              ParallelMode=False)
            self.LImageNoiseMachine.append(INM)
            
    def setPSF(self,DicoImagesPSF):
        for iField in range(self.NFields):
            self.LImageNoiseMachine[iField].setPSF(DicoImagesPSF[iField])
