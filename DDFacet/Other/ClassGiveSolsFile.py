from DDFacet.Other import reformat
import os

class ClassGive_kMSFileName():
    def __init__(self,
                 MSName=None,
                 SolsDir=None,
                 GD=None):
        self.GD=GD
        self.GDType=None
        if self.GD is not None:
            if "VisData" in self.GD.keys():
                self.GDType="kMS"
            else:
                self.GDType="DDF"
                
        if MSName is None and self.GD is not None:
            if self.GDType=="kMS":
                MSName=self.GD["VisData"]["MSName"]
            else:
                MSName=self.GD["Data"]["MS"]
                
        self.MSName=MSName
        
    def GiveFileName(self,SolsName=None,Type="Sols",ROW0=0):
        MSName=self.MSName
        SolsDir=None
        if SolsName is None:
            SolsName=self.GD["Solvers"]["SolverType"]
            if self.GD["Solutions"]["OutSolsName"]!="":
                SolsName=self.GD["Solutions"]["OutSolsName"]
                
        if SolsName[-4:]==".npz": return SolsName
                
        if self.GD is not None:
            if self.GDType=="kMS":
                SolsDir=self.GD["Solutions"]["SolsDir"]
            else:
                SolsDir=self.GD["DDESolutions"]["SolsDir"]
                
        if Type=="Sols":
            ext="sols.npz"
        elif Type=="Weights":
            ext="Weights.%i.npy"%ROW0
        elif Type=="parset":
            ext="parset"
            


            
        if SolsDir is None or SolsDir=="":
            FileName="%skillMS.%s."%(reformat.reformat(MSName),SolsName)
        else:
            _MSName=reformat.reformat(MSName).split("/")[-2]
            DirName=os.path.abspath("%s%s"%(reformat.reformat(SolsDir),_MSName))
            if not os.path.isdir(DirName):
                os.makedirs(DirName)
            FileName="%s/killMS.%s."%(DirName,SolsName)
        FileName="%s%s"%(FileName,ext)
        return FileName
        
