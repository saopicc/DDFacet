

import numpy as np
from DDFacet.Other import logger
import pickle as cPickle
log=logger.getLogger("ModelMachineMultiField")
import six
from DDFacet.Imager.ModModelMachine import ClassModModelMachine
import os
from DDFacet.Other import ModColor
from DDFacet.Other import MyPickle
import glob

class ClassModelMachineMultiField():
    def __init__(self,GD=None,Gain=None,GainMachine=None,DicoFields=None):
        self.GD=GD
        # self.VS=VS
        # self.DicoFields=DicoFields
        # self.NFields=len(self.DicoFields)
        self.ModConstructor = ClassModModelMachine(self.GD,MultiField=True)
        self.RefFreq=None
        self.LModelMachine=[]

    def InitialiseMMFromFile(self,MMFileDir):
        
        self.DicoFields=MyPickle.FileToDicoNP("%s/DicoFields.DicoPickle"%MMFileDir)
        self.NFields=len(self.DicoFields["ra"])

        if self.GD is None:
            self.GD=MyPickle.FileToDicoNP("%s/GD.DicoPickle"%MMFileDir)
            
        #ll=sorted(glob.glob("%s/Field*"%MMFileDir))
        for iField in range(self.NFields):
            l="%s/Field%i.DicoModel"%(MMFileDir,iField)
            MM=self.ModConstructor.GiveInitialisedMMFromFile(l)
            self.LModelMachine.append(MM)
        self.RefFreq=self.LModelMachine[0].RefFreq
        
            
        # for iField in range(self.NFields):
        #     ModelMachine=self.giveMMSingleField(iField)
        #     LModelMachine.append(ModelMachine)
        # self.RefFreq=self.VS.RefFreq
        # self.LModelMachine=LModelMachine

    def InitMM(self,Mode=None,DicoFields=None):
        self.DicoFields=DicoFields
        self.NFields=len(self.DicoFields["ra"])
        for iField in range(self.NFields):
            ModelMachine=self.ModConstructor.GiveMM(Mode=self.GD["Deconv"]["Mode"])
            self.LModelMachine.append(ModelMachine)

        

    def setRefFreq(self,RefFreq):
        if self.RefFreq is not None:
            print(ModColor.Str("Reference frequency already set to %f MHz"%(self.RefFreq/1e6)), file=log)
        else:
            self.RefFreq=RefFreq
        for MM in self.LModelMachine:
            MM.setRefFreq(RefFreq)

    def ToFile(self,FileName,DicoIn=None):
        FileName="%s_MF"%FileName
        os.system("mkdir -p %s"%FileName)
        for iField,MM in enumerate(self.LModelMachine):
            MM.ToFile("%s/Field%i.DicoModel"%(FileName,iField))
            
        MyPickle.Save(self.DicoFields,"%s/DicoFields.DicoPickle"%FileName)
        MyPickle.Save(self.GD,"%s/GD.DicoPickle"%FileName)


    def GiveModelImage(self,*args,**kwargs):
        return [MM.GiveModelImage(*args,**kwargs) for MM in self.LModelMachine]


    
