import numpy as np
from DDFacet.Other import logger
import pickle as cPickle
log=logger.getLogger("ImageMaskMachineMultiField")
import six
from DDFacet.Imager import ClassMaskMachine
import copy
from DDFacet.Array import shared_dict

class ClassMaskMachineMultiFields():
    def __init__(self,GD,LImageNoiseMachine,DicoFields):
        self.GD=GD
        from DDFacet.Imager.MultiFields.AppendSubFieldInfo import AppendSubFieldInfo
        AppendSubFieldInfo(self)

        self.LImageNoiseMachine=LImageNoiseMachine
        self.LMaskMachine=[]
        self.DicoFields=DicoFields
        self.NFields=len(self.DicoFields["ra"])
        self.DicoCurrentMask=shared_dict.SharedDict("CurrentMask")
        #self.shared_
        self.LCurrentMask=None
        
        for iField in range(self.NFields):
            GD=copy.deepcopy(self.GD)
            GD["Image"]["iField"]=iField
            MaskMachine=ClassMaskMachine.ClassMaskMachine(self.GD)
            MaskMachine.setImageNoiseMachine(self.LImageNoiseMachine[iField])
            self.LMaskMachine.append(MaskMachine)
            
    def updateMask(self,DicoDirty):
        
        for iField in range(self.NFields):
            self.LMaskMachine[iField].updateMask(DicoDirty[iField])
            self.DicoCurrentMask[iField]=self.LMaskMachine[iField].CurrentMask
            
        self.LCurrentMask=[self.DicoCurrentMask[iField] for iField in range(self.NFields)]
    
