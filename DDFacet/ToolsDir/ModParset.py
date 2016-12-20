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

def LoadModule(ThisDicoTerm,args=[]):
    Module=ThisDicoTerm["Module"]
    kwargs={}
    if "ModuleOptions" in ThisDicoTerm.keys():
        kwargs=ThisDicoTerm["ModuleOptions"]
        
    #ModuleName="Module_%s"%TermName
    ClassName=Module.split(".")[-1]
    ModuleName="This"+ClassName
    exec("from %s import %s as %s"%(Module,ClassName,ModuleName))
    exec('ThisModel=%s(*args,**kwargs)'%ModuleName)
    return ThisModel


def DictToFile(Dict,fout):
    f=open(fout,"w")
    Lkeys=Dict.keys()
    
    for key in Lkeys:
        keyw=key
        f.write("%s = %s\n"%(keyw,Dict[key]))
    f.close()

def FormatValue(ValueIn):

    if "#" in ValueIn:
        ValueIn=ValueIn.split("#")[0]
    MayBeInt=False
    if not("." in ValueIn): MayBeInt=True
    if "True" in ValueIn:
        Value=True
    elif "False" in ValueIn:
        Value=False
    elif "None" in ValueIn:
        Value=None
    elif '"' in ValueIn:
        Value=ValueIn.replace(" ","").replace('"',"")
    elif ("[" in ValueIn):

        Value0=ValueIn[1:-1].split(",")
        try:
            Value=[float(v) for v in Value0]
        except:
            Value=Value0
    elif ("," in ValueIn):
        Value0=ValueIn.split(",")
        try:
            Value=[float(v) for v in Value0]
        except:
            Value=Value0
        
    else:
        try:
            Value=float(ValueIn)
            if MayBeInt: Value=int(Value)
        except:
            Value=ValueIn
            Value=Value.replace(" ","")
    return Value


def setValue(Dico,key,Value):
    keys=key.split(".")
    Nlev=len(keys)
    ZeroKey=keys[0]
    if Nlev==1:
        Dico[ZeroKey]=Value
    else:
        NewKey=".".join(keys[1::])
        if not(ZeroKey in Dico.keys()):
            Dico[ZeroKey]={}
        setValue(Dico[ZeroKey],NewKey,Value)


def FileToDict(fname=None,AppendToDico=None,ListOptions=None):
    if AppendToDico is None:
        Dict={}
    else:
        Dict=AppendToDico
        
    
    if fname is not None:
        f=file(fname,"r")
        ListOut=f.readlines()
    if ListOptions is not None:
        ListOut=ListOptions

    ListToDict(Dict,ListOut)

    return Dict

def ListToDict(Dict,ListOut):


    for line in ListOut:
        StrToDict(Dict,line)

def StrToDict(Dict,line):
    if line=='\n': return
    if line[0]=="#": return
    if line[0:5]=="input":
        _,a=line[5::].split('{')
        subfile,_=a.split('}')
        FileToDict(subfile,AppendToDico=Dict)
        return
    key,val=line[0:-1].split("=")
    key=key.replace(" ","")
    key=key.replace("\t","")
    val=val.replace(" ","")
    if val[0]=="$":
        VarName=val[1::]
        val=Dict[VarName]
    valFormat=FormatValue(val)
    #print val,valFormat,type(val),type(valFormat)
    setValue(Dict,key,valFormat)

