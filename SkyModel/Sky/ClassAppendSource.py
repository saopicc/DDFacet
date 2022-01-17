import numpy as np
import ephem
from astropy.time import Time
from DDFacet.Other import logger
log=logger.getLogger("ClassAppendSource")
from SkyModel.Sky.ModBBS2np import ReadBBSModel
import os
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms
def angDist(a0,a1,d0,d1):
    s=np.sin
    c=np.cos
    return np.arccos(s(d0)*s(d1)+c(d0)*c(d1)*c(a0-a1))


class ClassAppendSource():
    def __init__(self,SM,ListBody):
        self.ListBody=ListBody
        self.SM=SM

        
    def appendAll(self):
        ListBody=self.ListBody
        for Body in ListBody:
            tm = Time(Body["Time"] / 86400.0, scale="utc", format='mjd')
            if Body["Name"]=="Sun": 
                B=ephem.Sun()
                B.compute(tm.iso)
                A = np.zeros((1,),dtype=self.SM.SourceCat.dtype)
                A=A.view(np.recarray)
                A.ra=float(B.ra)
                A.dec=float(B.dec)
                d=angDist(self.SM.rarad,float(B.ra),self.SM.decrad,float(B.dec))*180/np.pi
                log.print("  [%s] On %s, position is ra/dec = %s %s [%.2f deg from target]"%(Body["Name"],tm.datetime,B.ra,B.dec,d))
                A.I=1000.
                A.Sref=A.I
                if not self.SM.InputCatIsEmpty:
                    A.RefFreq=self.SM.SourceCat.RefFreq[0]
                    C=np.max(self.SM.SourceCat.Cluster)+1
                else:
                    A.RefFreq=100e6
                    C=0
                    self.SM.InputCatIsEmpty=False
                
                        
                A.Cluster=C
                A.Type=2
                A.Gmaj=30./60*np.pi/180
                A.Gmin=A.Gmaj
                A.Name="c%is%i."%(C,0)



                
            elif Body["Name"]=="CygA" or Body["Name"]=="CasA" or Body["Name"]=="VirA":
                path = os.path.dirname(os.path.abspath(__file__))
                FName="%s/Models/LOFAR/%s.txt"%(path,Body["Name"])
                A=ReadBBSModel(FName)
                log.print("  [%s] Reading file %s"%(Body["Name"],FName))
                ra,dec=np.mean(A.ra),np.mean(A.dec)
                d=angDist(self.SM.rarad,ra,self.SM.decrad,dec)*180/np.pi
                ras  = rad2hmsdms(ra,Type="ra").replace(" ",":")
                decs = rad2hmsdms(dec,Type="dec").replace(" ",".")
                log.print("  [%s] Position is ra/dec = %s %s [%.2f deg from target]"%(Body["Name"],ras,decs,d))


                if not self.SM.InputCatIsEmpty:
                    A.RefFreq=self.SM.SourceCat.RefFreq[0]
                    C=np.max(self.SM.SourceCat.Cluster)+1
                else:
                    A.RefFreq=100e6
                    C=0
                    self.SM.InputCatIsEmpty=False

                A.Type[:]=0
                A.Gmin[:]=0
                A.Gmaj[:]=0
                A.Gangle[:]=0
                A.Cluster[:]=C
                Ns=A.shape[0]
                for iSource in range(Ns):
                    A.Name[iSource]="c%is%i."%(C,iSource)

            self.SM.SourceCat=np.hstack([self.SM.SourceCat,A])
            self.SM.SourceCat=self.SM.SourceCat.view(np.recarray)
                
                
