#!/usr/bin/env python
import collections
import optparse as OptParse
import PrintOptParse
import ReadCFG
import MyPickle
import ClassPrint
import ModColor
#global Parset
#Parset=ReadCFG.Parset("/media/tasse/data/DDFacet/Parset/DefaultParset.cfg")
#D=Parset.DicoPars 

class MyOptParse():
    def __init__(self,usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',
                 description="""Questions and suggestions: cyril.tasse@obspm.fr""",
                 DefaultDict=None):
        self.opt = OptParse.OptionParser(usage=usage,version=version,description=description)
        self.DefaultDict=DefaultDict
        self.CurrentGroup=None
        self.DicoGroupDesc=collections.OrderedDict()

    def OptionGroup(self,Name,key=None):
        if self.CurrentGroup!=None:
            self.Finalise()
        self.CurrentGroup = OptParse.OptionGroup(self.opt, Name)
        self.CurrentGroupKey=key
        self.DicoGroupDesc[key]=Name


    def add_option(self,Name='Mode',help='Default %default',type="str",default=None):
        if default==None:
            default=self.DefaultDict[self.CurrentGroupKey][Name]
        
        self.CurrentGroup.add_option('--%s'%Name,help=help,type=type,default=default,dest=self.GiveKeyDest(self.CurrentGroupKey,Name))

    def GiveKeyDest(self,GroupKey,Name):
        return "_".join([GroupKey,Name])

    def GiveKeysOut(self,KeyDest):
        return KeyDest.split("_")

    def Finalise(self):
        self.opt.add_option_group(self.CurrentGroup)

    def ReadInput(self):
        self.options, self.arguments = self.opt.parse_args()
        self.GiveDicoConfig()
        self.DicoConfig=self.DefaultDict
        
    def GiveDicoConfig(self):
        DicoDest=vars(self.options)
        for key in DicoDest.keys():
            GroupName,Name=self.GiveKeysOut(key)
            self.DefaultDict[GroupName][Name]=DicoDest[key]

        return self.DefaultDict


    def Print(self,RejectGroup=[]):
        P=ClassPrint.ClassPrint(HW=30)
        print ModColor.Str(" Selected Options:")
    
        for Group,V in self.DefaultDict.items():
            Skip=False
            for Name in RejectGroup:
                if Name in Group:
                    Skip=True
    
            if Skip: continue
            try:
                GroupTitle=self.DicoGroupDesc[Group]
            except:
                GroupTitle=Group
            print ModColor.Str(GroupTitle,col="green")
    
            option_list=self.DefaultDict[Group]
            for oname in option_list:
                V=self.DefaultDict[Group][oname]
                
                if True:#V!="":
                    if V=="": V="''"
                    P.Print(oname,V)
            print



def test():
    OP=MyOptParse()
    
    OP.OptionGroup("* Data","VisData")
    OP.add_option('MSName',help='Input MS')
    OP.add_option('ColName')
    OP.Finalise()
    OP.ReadInput()
    Dico=OP.GiveDicoConfig()
    OP.Print()
    
    return OP.DefaultDict
    

if __name__=="__main__":
    test()

