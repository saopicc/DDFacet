

def AppendSubFieldInfo(self):
    iField=self.GD["Image"].get("iField",None)
    setattr(self,"iField",iField)
    StrField=""
    if iField is not None:
        StrField="_Field%i"%iField
    setattr(self,"StrField",StrField)
