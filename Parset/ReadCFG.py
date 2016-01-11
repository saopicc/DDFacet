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

    if ValueIn==None: return None


    if "#" in ValueIn:
        ValueIn=ValueIn.split("#")[0]

    

    if StrMode:
        return ValueIn

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
            FVal=FormatValue(Val)#,StrMode=True)

            dict1[option] = FVal
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        return dict1


        

