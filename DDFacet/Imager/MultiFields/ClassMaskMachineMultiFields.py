import numpy as np
from DDFacet.Other import logger
import pickle as cPickle
log=logger.getLogger("ImageMaskMachineMultiField")
import six
from DDFacet.Imager import ClassMaskMachine

class ClassMaskMachineMultiFields():
    def __init__(self,GD,LImageNoiseMachine,DicoFields):
        self.GD=GD
        self.LImageNoiseMachine=LImageNoiseMachine
        self.LMaskMachine=[]
        self.DicoFields=DicoFields
        self.NFields=len(self.DicoFields["ra"])
        self.LCurrentMask=None
        for iField in range(self.NFields):
            MaskMachine=ClassMaskMachine.ClassMaskMachine(self.GD)
            MaskMachine.setImageNoiseMachine(self.LImageNoiseMachine[iField])
            self.LMaskMachine.append(MaskMachine)
            
    def updateMask(self,DicoDirty):
        for iField in range(self.NFields):
            self.LMaskMachine[iField].updateMask(DicoDirty[iField])
        self.LCurrentMask=[self.LMaskMachine[iField].CurrentMask for iField in range(self.NFields)]
    
