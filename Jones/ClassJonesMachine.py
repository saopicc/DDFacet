import numpy as np
import MyPickle
import ModColor

from Array import ModLinAlg

import MyLogger
log=MyLogger.getLogger("ClassJonesMachine")
from ToolsDir import ModParset

def test():
    import ClassMS
    ParsetFile="ParsetNew.txt"

    MS=ClassMS.ClassMS("/media/tasse/data/MS/SimulTec/many/0000.MS/")
    
    J=ClassJonesMachine(DC)
    J.InitFromParset(ParsetFile)
    import ClassSM
    SM=ClassSM.ClassSM("/media/tasse/data/HYPERCAL/test/ModelIon.txt")
    return J

class ClassJonesMachine():
    def __init__(self,Xp,PointingID=0):
        self.DC=Xp.MDC.giveSinglePointingData(PointingID=PointingID)
        self.MS=self.DC.MS
        self.PointingID=PointingID
        self.DicoJones=Xp.DicoJones
        self.Xp=Xp
        

    def GetJones(self,time,freqs,GiveIdentity=False,Descriptive="Right",SkipJones="",radec=None):
        self.KeyOrder=self.Xp.KeyOrder
        
        NDir=self.DC.SM.ClusterCat.ra.size
        if radec!=None:
            NDir=1
        ListSkipJones=SkipJones.split(",")
        Jout=np.zeros((NDir,freqs.size,self.MS.na,4),dtype=np.complex128)

        Jout[:,:,:,0]=1
        Jout[:,:,:,3]=1
        if GiveIdentity:
            return Jout
        import ClassTimeIt
        T=ClassTimeIt.ClassTimeIt("JonesMachine")
        T.disable()


        for key in self.KeyOrder:#self.DicoJones.keys():
            if self.DicoJones[key]["SkipJones"]: 
                continue
            if (self.DicoJones[key]["Descriptive"]==Descriptive)|(key==Descriptive):
                if key in ListSkipJones: 
                    continue
                # print>>log, ("GetJones do key = %s [Description=%s]"%(key,Descriptive))
                coef=self.DicoJones[key]["coef"]
                ThisJones=self.DicoJones[key]["Jones"](time,coef,freqs,self.PointingID,radec=radec)
                T.timeit("ThisJones %s"%key)
                
                Jout=ModLinAlg.BatchDot(ThisJones.reshape(Jout.shape).astype(np.complex128),Jout.astype(np.complex128))
                #Jout=ModLinAlg.BatchDot(Jout.astype(np.complex128),ThisJones.reshape(Jout.shape).astype(np.complex128))
                #print "Jout.shape",Jout.shape 
                #Jout2=ModLinAlg.BatchDot2(ThisJones.reshape(Jout.shape).astype(np.complex128),Jout.astype(np.complex128))
                #print "Jout2.shape",Jout2.shape
                #stop

                T.timeit("Prod %s"%key)

        return Jout


