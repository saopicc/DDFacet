import numpy as np
import ephem
from astropy.time import Time

class ClassAppendSource():
    def __init__(self,SM,ListBody):
        self.ListBody=ListBody
        self.SM=SM
        self.SourceCat=self.SM.SourceCat
        
    def appendAll(self):
        ListBody=self.ListBody
        for Body in ListBody:
            tm = Time(Body["Time"] / 86400.0, scale="utc", format='mjd')
            if Body["Name"]=="Sun": 
                B=ephem.Sun()
                B.compute(tm.iso)
                log.print("[%s] On %s, position is ra/dec = %s %s"%(Body["Name"],tm.datetime,B.ra,B.dec))
                A = np.zeros((1,),dtype=self.SourceCat.dtype)
                A=A.view(np.recarray)
                A.ra=float(B.ra)
                A.dec=float(B.dec)
                A.I=1000.
                A.Sref=A.I
                if not self.InputCatIsEmpty:
                    A.RefFreq=self.SourceCat.RefFreq[0]
                    C=np.max(self.SourceCat.Cluster)+1
                else:
                    A.RefFreq=100e6
                    C=0
                    self.InputCatIsEmpty=False
                
                        
                A.Cluster=C
                A.Type=2
                A.Gmaj=30./60*np.pi/180
                A.Gmin=A.Gmaj
                A.Name="c%is%i."%(C,0)
                
                self.SourceCat=np.hstack([self.SourceCat,A])
                self.SourceCat=self.SourceCat.view(np.recarray)
            elif Body["Name"]=="CygA":
                print(__file__)
                stop
                
