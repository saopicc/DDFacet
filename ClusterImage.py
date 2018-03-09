import numpy as np
import pylab
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClusterImage")
import pyfits

def test():
    
    CIM=ClusterImage()
    CIM.setCatName("image_dirin_SSD_m.app.restored.pybdsm.srl.fits")
    CIM.GroupSources()

def angDist(a0,a1,d0,d1):
    s=np.sin
    c=np.cos
    return np.arccos(s(d0)*s(d1)+c(d0)*c(d1)*c(a0-a1))
    
class ClusterImage():
    def __init__(self):
        pass

    def setCatName(self,CatName):
        self.CatName=CatName
        self.c=pyfits.open(CatName)[1]
        self.c.data.RA*=np.pi/180.
        self.c.data.DEC*=np.pi/180.
        
    def GroupSources(self,RadiusArcmin=2.):
        Rad=RadiusArcmin/60.*np.pi/180.
        c=self.c
        self.DicoPos={}
        
        Ns=c.data.RA.size
        print>>log,"Merging sources..."
        for iS in range(Ns):
            self.DicoPos[iS]={}
            self.DicoPos[iS]["RA"]=c.data.RA[iS]
            self.DicoPos[iS]["DEC"]=c.data.DEC[iS]
            self.DicoPos[iS]["Maj"]=c.data.Maj[iS]
            self.DicoPos[iS]["Total_flux"]=c.data.Total_flux[iS]
            self.DicoPos[iS]["Checked"]=False

        self.DicoAssoc={}
        for iS in range(Ns):
            RA=self.DicoPos[iS]["RA"]
            DEC=self.DicoPos[iS]["DEC"]
            Maj=self.DicoPos[iS]["Maj"]
            S=self.DicoPos[iS]["Total_flux"]
            
            #print>>log,"======================"
            #print>>log,"Inspecting %i"%iS
            if len(self.DicoAssoc)==0:
                self.DicoAssoc[0]={"RA":[RA],
                                   "DEC":[DEC],
                                   "S":[S],
                                   "Maj":[Maj]}
                continue

            HasAssociatedThis=False
            for iSc in self.DicoAssoc.keys():
                ra,dec=np.array(self.DicoAssoc[iSc]["RA"]),np.array(self.DicoAssoc[iSc]["DEC"])
                dra,ddec=RA-ra,DEC-dec
                d=angDist(ra,RA,dec,DEC)#np.sqrt((dra)**2+(ddec)**2)
                if d.min()<Rad:
                    #print>>log,"  Associating %i <- %i"%(iSc,iS)
                    self.DicoAssoc[iSc]["RA"].append(RA)
                    self.DicoAssoc[iSc]["DEC"].append(DEC)
                    self.DicoAssoc[iSc]["S"].append(S)
                    self.DicoAssoc[iSc]["Maj"].append(Maj)
                    HasAssociatedThis=True
                    break
            if not HasAssociatedThis:
                ThisInd=max(self.DicoAssoc.keys())+1
                self.DicoAssoc[ThisInd]={"RA":[RA],
                                         "DEC":[DEC],
                                         "S":[S],
                                         "Maj":[Maj]}

        
        raOut=[]
        decOut=[]
        SOut=[]
        MajOut=[]
        for iS in self.DicoAssoc.keys():
            ra=self.DicoAssoc[iS]["RA"]
            dec=self.DicoAssoc[iS]["DEC"]
            S=self.DicoAssoc[iS]["S"]
            Maj=self.DicoAssoc[iS]["Maj"]
            raOut.append(np.mean(ra))
            decOut.append(np.mean(dec))
            SOut.append(np.sum(S))
            MajOut.append(np.min(Maj))

        Cat=np.zeros((len(raOut),),dtype=[("ra",np.float32),
                                          ("dec",np.float32),
                                          ("S",np.float32),
                                          ("Maj",np.float32)])
        Cat=Cat.view(np.recarray)
        Cat.ra=raOut
        Cat.dec=decOut
        Cat.S=SOut
        Cat.Maj=MajOut
        self.Cat=Cat
        print>>log,"Have merged %i -> %s"%(Ns,Cat.ra.size)

    
    def Cluster(self):
        
