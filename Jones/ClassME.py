
class SinglePointingMeasurementEquation():
    def __init__(self,MME,PointingID=0):
        self.MME=MME
        self.PointingID=PointingID
        self.mount()

    def mount(self):
        self.JM=self.MME.giveJM(self.PointingID)
        self.SM=self.MME.giveSM(self.PointingID)
        self.Xp=self.MME.Xp


class MeasurementEquation():
    def __init__(self,NameParsetME="RIME"):
        self.DicoPointing={}
        self.PointingIDs=[]
        self.NameParsetME=NameParsetME

    def giveSinglePointingME(self,PointingID=0):
        return SinglePointingMeasurementEquation(self,PointingID=PointingID)


    def initDicoKey(self,key):
        if not(key in self.DicoPointing.keys()):
            self.DicoPointing[key]={}

    def setJM(self,JonesMachine,PointingID=0):
        self.initDicoKey(PointingID)
        self.DicoPointing[PointingID]["JonesMachine"]=JonesMachine
        self.PointingIDs=self.DicoPointing.keys()
        self.Xp=JonesMachine.Xp

    def setSM(self,SkyMachine,PointingID=0):
        self.initDicoKey(PointingID)
        self.DicoPointing[PointingID]["SkyMachine"]=SkyMachine
        self.PointingIDs=self.DicoPointing.keys()

    def giveJM(self,PointingID=0):
        return self.DicoPointing[PointingID]["JonesMachine"]
    
    def giveSM(self,PointingID=0):
        return self.DicoPointing[PointingID]["SkyMachine"]
    
