'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA-SA, Rhodes University

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

import ConfigParser
from collections import OrderedDict

def test():
    P=Parset()

def FormatDico(DicoIn):
    Dico=OrderedDict()
    for key in DicoIn.keys():
        Dico[key]=FormatValue(DicoIn[key])
    return Dico

def FormatValue(ValueIn,StrMode=False):

    if ValueIn is None: return None


    if "#" in ValueIn:
        ValueIn=ValueIn.split("#")[0]

    

    if StrMode:
        return ValueIn

    MayBeInt = not( "." in ValueIn or "e" in ValueIn.lower() )
    
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

class Parset():
    def __init__(self,File="../Parset/DefaultParset.cfg"):
        self.File=File
        self.Read()

    def update (self, other):
        """Updates this Parset with all keys found in other parset"""
        for secname, secmap in other.DicoPars.iteritems():
            if secname in self.DicoPars:
                self.DicoPars[secname].update(secmap)
            else:
                self.DicoPars[secname] = secmap
    

    def Read(self):
        config = ConfigParser.ConfigParser(dict_type=OrderedDict)
        config.optionxform = str
        L=config.read(self.File)
        self.Success=True
        if len(L)==0:
            self.Success=False
            return
        self.Config=config
        DicoPars=OrderedDict()
        LSections=config.sections()

        for Section in LSections:
            DicoPars[Section]=self.ConfigSectionMap(Section)
        self.DicoPars=DicoPars


    def ConfigSectionMap(self,section):
        dict1 = OrderedDict()
        Config=self.Config
        options = Config.options(section)
        for option in options:

            Val=Config.get(section, option)
            Val=Val.replace(" ","")
            Val=Val.replace('"',"")
            Val=Val.replace("'","")
            FVal=FormatValue(Val)#,StrMode=True)

            dict1[option] = FVal
            # if dict1[option] == -1:
            #     DebugPrint("skip: %s" % option)
        return dict1


        

