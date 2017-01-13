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

import numpy as np
from DDFacet.ToolsDir import ModParset
import os

class ClassJonesData():
    def __init__(self):
        pass

    


class ClassInstrData():
    def __init__(self):
        pass

def SetValDico(DicoToSet,DicoVals):
    for key in DicoVals:
        val=DicoVals[key]
        if type(val)==dict:
            SetValDico(DicoToSet[key],DicoVals[key])
        else:
            DicoToSet[key]=DicoVals[key]

def testReplace(ParsetName="/media/6B5E-87D0/HyperCal2/test/TestDDFacet/ParsetDDFacet.txt"):
    ReplaceDico={ 'Files': {'FileMSCat': {'Name': ['caca'],
                                          'StartEndNf': [0.0, -1.0, 0.0]}}}


    GD=ClassGlobalData(ParsetFile=ParsetName,ReplaceDico=ReplaceDico)
    return GD


import socket
class ClassGlobalData():
    def __init__(self,ParsetFile="/media/tasse/data/HyperCal2/test/ParsetNew.txt",ReplaceDico=None):
        self.ParsetFile=ParsetFile
        self.setDicoConfig()
        # self.SelSubFreqCat()
        
        if ReplaceDico is not None:
            SetValDico(self.DicoConfig,ReplaceDico)

        #self.HYPERCAL_DIR=os.environ["HYPERCAL_DIR"]

        LocalHost=socket.gethostname()
        if "." in LocalHost: LocalHost,_=LocalHost.split(".")
        LocalHostName=LocalHost
        self.LocalHostName=LocalHostName



    def setDicoConfig(self):
        ParsetFile=self.ParsetFile
        DicoConfig= ModParset.FileToDict(ParsetFile)
        # DicoConfig["RIME"]=DicoConfig["RIME"].replace(".V","")
        # DicoConfig["RIME"]=DicoConfig["RIME"].replace(".Sky","")
        # DicoConfig["RIME_FullCorr"]=DicoConfig["RIME_FullCorr"].replace(".V","")
        # DicoConfig["RIME_FullCorr"]=DicoConfig["RIME_FullCorr"].replace(".Sky","")
        self.DicoConfig=DicoConfig


        if "Representation" in self.DicoConfig.keys():
            self.FilterOrder=[
                {"Type":[],
                 "Adapt": False,
                 "WeightUV": None,
                 "Repr":[
                     {"Type": self.DicoConfig["Representation"]["Type"],
                      "AverageType":self.DicoConfig["Representation"]["AverageType"],
                      "ImagerParam":{"FOV":3.,"OverS":1,
                                     "SupportSel":0.,
                                     "resolution":0.},
                      "nbands": self.DicoConfig["Representation"]["NBand"]}]}
            ]

        self.DefaultImagParam={"Support":5,
                             "wmax":50000,
                             "NPix":10,
                             "Cell":5.,
                             "incr":1.,
                             "padding":3}
        self.ReprData=None
   
class ClassSinglePointingData():
    def __init__(self,MDC,PointingID=0):
        self.MDC=MDC
        self.PointingID=PointingID
        self.DicoConfig=MDC.DicoConfig
        self.updateInternal()
        
    def updateInternal(self):
        self.MS=self.MDC.giveMS(self.PointingID)
        self.SM=self.MDC.giveSM(self.PointingID)
        self.MapSelBLs=self.MDC.giveMappingBL(self.PointingID)
        self.freqs=self.MDC.giveFreqs(self.PointingID)




class ClassMultiPointingData():

    def __init__(self,GlobalData):
        self.MapSelBLs=None
        self.DicoConfig=GlobalData.DicoConfig
        self.SimulMode=False
        self.DicoPointing={}
        self.DicoData={}
        self.NPointing=0
        self.ListID=[]

    def giveSinglePointingData(self,PointingID=0):
        return ClassSinglePointingData(self,PointingID=PointingID)
        
    def MountSolsSimul(self):
        if self.DicoConfig["Files"]["Simul"]["File"] is not None:
            DicoSimul=MyPickle.Load(self.DicoConfig["Files"]["Simul"]["File"])
            self.std=DicoSimul["std"]
            self.TrueSols=DicoSimul["Sols"]
            self.SimulMode=True

    def initDicoKey(self,key):
        if not(key in self.DicoPointing.keys()):
            self.DicoPointing[key]={}
            self.ListID.append(key)

    def setMS(self,MS,PointingID=0):
        self.initDicoKey(PointingID)
        self.DicoPointing[PointingID]["MS"]=MS
        self.NPointing=len(self.DicoPointing.keys())
        self.DicoPointing[PointingID]["MapSelBLs"]=None
        
    def giveMS(self,PointingID=0):
        return self.DicoPointing[PointingID]["MS"]


    def setSM(self,SM,PointingID=0):
        self.initDicoKey(PointingID)
        self.DicoPointing[PointingID]["SM"]=SM
        #self.DicoPointing[PointingID]["CurrentData"]={}

    def giveSM(self,PointingID=0):
        if "SM" in self.DicoPointing[PointingID].keys():
            return self.DicoPointing[PointingID]["SM"]
        else:
            return None

    def setFreqs(self,freqs,PointingID=0):
        self.DicoPointing[PointingID]["freqs"]=freqs

    def giveFreqs(self,PointingID=0):
        return self.DicoPointing[PointingID]["freqs"]

    def setMappingBL(self,PointingID=0):
        self.initDicoKey(PointingID)
        if self.DicoConfig["Select"]["FlagAntBL"] is not None:
            BLSel=self.DicoConfig["Select"]["FlagAntBL"]#.replace(" ","").split(',')
        else:
            BLSel=[]
        MS=self.giveMS(PointingID=PointingID)
        self.DicoPointing[PointingID]["MapSelBLs"]=MS.GiveMappingAnt(BLSel)

    def giveMappingBL(self,PointingID=0):
        return self.DicoPointing[PointingID]["MapSelBLs"]
        
    ###################
        

    def setCurrentData(self,key,Data):
        self.DicoData[key]=Data

    def getCurrentData(self,key):
        if not(key in self.DicoData.keys()):
            return False
        else:
            return self.DicoData[key]
        

    # def setCurrentData(self,key,Data,PointingID=None):

    #     if PointingID is None:
    #         for PointingID in self.ListID:
    #             self.setCurrentDataPointing(key,Data[PointingID],PointingID=PointingID)
    #     else:
    #         if not(key in self.DicoPointing.keys()):
    #             self.DicoPointing[key]={}
    #             self.DicoPointing[key]["CurrentData"]={}
                
    #         self.setCurrentDataPointing(key,Data,PointingID=PointingID)
    
    # def setCurrentDataPointing(self,key,Data,PointingID=0):
    #     self.DicoPointing[PointingID]["CurrentData"][key]=Data

    # # def setCurrentFlags(self,DicoData):
    # #     for PointingID in self.ListID:
    # #         self.setCurrentDataPointing("Flags",DicoData[PointingID]["flags"],PointingID=PointingID)
 
    

    # def getCurrentData(self,key,PointingID=0):
    #     if not(key in self.DicoPointing[PointingID]["CurrentData"]):
    #         return False
    #     else:
    #         return self.DicoPointing[PointingID]["CurrentData"][key]

    ###################

    def setCurrentTimeBin(self,(it0,it1)):
        self.itimes=it0,it1
        MS=self.giveMS()
        Row0,Row1=it0*MS.nbl,it1*MS.nbl
        self.CurrentTime=np.mean(MS.times_all[Row0:Row1])
        #self.updateBeam(self.CurrentTime)
        #self.updatePierce(self.CurrentTime)
        #if XpToCorrectJ:
        #self.BuildNormJones()
