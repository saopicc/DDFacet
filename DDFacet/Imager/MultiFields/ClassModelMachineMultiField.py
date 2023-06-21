

import numpy as np
from DDFacet.Other import logger
import pickle as cPickle
log=logger.getLogger("ModelMachineMultiField")
import six
from DDFacet.Imager.ModModelMachine import ClassModModelMachine

class ClassModelMachineMultiField():
    def __init__(self,GD,VS,DicoFields):
        self.GD=GD
        self.VS=VS
        self.DicoFields=DicoFields
        self.NFields=len(self.DicoFields)
        self.ModConstructor = ClassModModelMachine(self.GD,MultiField=True)
        
        SubstractModel=self.GD["Predict"]["InitDicoModel"]
        self.DoSub=(SubstractModel!="")&(SubstractModel is not None)
        LModelMachine=[]
        for iField in range(self.NFields):
            ModelMachine=self.giveMMSingleField(iField)
            LModelMachine.append(ModelMachine)
        self.RefFreq=self.VS.RefFreq
        self.LModelMachine=LModelMachine
        
    def giveMMSingleField(self,iField):
        if self.DoSub:
            print(ModColor.Str("Initialise sky model using %s"%SubstractModel,col="blue"), file=log)
            ModelMachine = self.ModConstructor.GiveInitialisedMMFromFile(SubstractModel)
            def safe_encode(s):
                return s.decode() if isinstance(s, bytes) and six.PY3 else s
            modeltype = safe_encode(ModelMachine.DicoSMStacked.get("Type", ModelMachine.DicoSMStacked.get(b"Type", None)))
            if modeltype == "GA":
                modeltype = "SSD"
            elif modeltype == "MSMF":
                modeltype = "HMP"
            if self.GD["Deconv"]["Mode"] != modeltype:
                raise NotImplementedError("You want to use different minor cycle and IniDicoModel types [%s vs %s]"\
                                          %(self.GD["Deconv"]["Mode"], modeltype))

            if ModelMachine.RefFreq!=self.VS.RefFreq:
                print(ModColor.Str("Taking reference frequency from the model machine %f MHz (instead of %f MHz from the data)"%
                        (ModelMachine.RefFreq/1e6,self.VS.RefFreq/1e6)), file=log)
            self.RefFreq=self.VS.RefFreq=ModelMachine.RefFreq

            self.DoDirtySub=1
            # enable that to be able to restore even if we don't deconvolve
            self.HasDeconvolved=True
        else:
            ModelMachine = self.ModConstructor.GiveMM(Mode=self.GD["Deconv"]["Mode"])
            ModelMachine.setRefFreq(self.VS.RefFreq)
            self.DoDirtySub=0
        
        
        return ModelMachine

    def ToFile(self,*args,**kwargs):
        pass

    def GiveModelImage(self,*args,**kwargs):
        return [MM.GiveModelImage(*args,**kwargs) for MM in self.LModelMachine]


    
