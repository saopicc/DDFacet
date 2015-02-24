
import numpy as np
import ClassMS
from pyrap.tables import table
import ClassWeighting
import MyLogger
log=MyLogger.getLogger("ClassVisServer")
import MyPickle
from CheckJob import LaunchAndCheck
import NpShared
import ModColor

class ClassDistributedVisServer():

    def __init__(self,IM,PointingID=0):
        self.GD=IM.GD
        self.IM=IM
        self.dMS=self.IM.CI.GiveDMS(PointingID)
        self.ReInitChunkCount()
        self.TChunkSize=self.GD.DicoConfig["Facet"]["TChunkSize"]
        self.Init()
        self.VisWeights=None
        self.CountPickle=0
        self.SharedNames=[]

    def Init(self,PointingID=0):
        self.MS=self.dMS.MS
        TimesInt=np.arange(0,self.MS.DTh,self.TChunkSize).tolist()
        if not(self.MS.DTh in TimesInt): TimesInt.append(self.MS.DTh)
        self.TimesInt=TimesInt
        self.NTChunk=len(self.TimesInt)-1

    def ReInitChunkCount(self):
        self.CurrentTimeChunk=0

    def CalcWeigths(self,ImShape,CellSizeRad):
        if self.VisWeights!=None: return
        irc0=self.IM.E.GiveSubCluster("Data")["ids"][0]
        DicoSend={"ImShape":ImShape,"CellSizeRad":CellSizeRad}
        self.IM.E.rc[irc0].push(DicoSend)
        LaunchAndCheck(self.IM.E.rc[irc0],"VS.CalcWeigths(ImShape,CellSizeRad)")
        LaunchAndCheck(self.IM.E.rc[irc0],"W=VS.VisWeights")
        self.VisWeights=self.IM.E.rc[irc0].get("W")

    def GiveNextVisChunk(self):
        if self.CurrentTimeChunk==self.NTChunk:
            print>>log, "Reached end of chunks"
            self.ReInitChunkCount()
            return None
        iT0,iT1=self.CurrentTimeChunk,self.CurrentTimeChunk+1
        print>>log, "Reading next data chunk in [%5.2f, %5.2f] hours"%(self.TimesInt[iT0],self.TimesInt[iT1])
        self.CurrentTimeChunk+=1
        V=self.IM.E.GiveSubCluster("Data")["V"]
        LaunchAndCheck(V,"VS.GiveNextVisChunk()")

        DATA=self.dMS.GiveViSTchunk(GiveMapBLSel=True)

        MapBLSel=DATA["MapBLSel"]
        ROW0,ROW1=DATA["ROW_01"]
        A0,A1=DATA["A0A1"]
        DATA["A0"]=np.int32(A0)
        DATA["A1"]=np.int32(A1)
        DATA["A0A1"]=(DATA["A0"],DATA["A1"])
        DATA["uvw"]=np.float64(DATA["uvw"])
        DATA["times"]=np.float64(DATA["times"])
        W=self.VisWeights[ROW0:ROW1][MapBLSel]
        DATA["Weights"]=np.float64(W)
        DATA["data"]=np.complex64(DATA["data"])
        DATA["flags"]=np.bool8(DATA["flags"])

        if self.GD.DicoConfig["Files"]["VisInSharedMem"]:
            self.ClearSharedMemory()
            DATA=self.PutInShared(DATA)
            DATA["A0A1"]=(DATA["A0"],DATA["A1"])

        #DATA["data"].fill(1)

        #DATA.keys()
        #['uvw', 'MapBLSel', 'Weights', 'nbl', 'data', 'ROW_01', 'itimes', 'freqs', 'nf', 'times', 'A1', 'A0', 'flags', 'nt', 'A0A1']

        return DATA

    def ClearSharedMemory(self):
        for Name in self.SharedNames:
            NpShared.DelArray(Name)
        self.SharedNames=[]

    def PutInShared(self,Dico):
        print>>log, ModColor.Str("Sharing data: start")
        self.PrefixShared="SharedVis"
        V=self.IM.CI.E.GiveSubCluster("Imag")["V"]
        V["UseShared"]=True
        V["PrefixShared"]=self.PrefixShared        
        DicoOut={}
        for key in Dico.keys():
            if type(Dico[key])!=np.ndarray: continue
            Shared=NpShared.ToShared("%s.%s"%(self.PrefixShared,key),Dico[key])
            DicoOut[key]=Shared

            self.SharedNames.append("%s.%s"%(self.PrefixShared,key))
        print>>log, ModColor.Str("Sharing data: done")
        return DicoOut


