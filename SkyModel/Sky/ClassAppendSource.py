import numpy as np
import ephem
from astropy.time import Time
from DDFacet.Other import logger
log=logger.getLogger("ClassAppendSource")
from SkyModel.Sky.ModBBS2np import ReadBBSModel
import os
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms
from DDFacet.Other import ModColor

from pyproj import Proj, transform
from astropy.coordinates import AltAz, EarthLocation, ITRS, SkyCoord
from astropy.time import Time
import astropy.units as u
from pyproj import Transformer, CRS

def angDist(a0,a1,d0,d1):
    s=np.sin
    c=np.cos
    return np.arccos(s(d0)*s(d1)+c(d0)*c(d1)*c(a0-a1))

def ecef_to_lla(x, y, z):
    ecef = Proj(proj="geocent", datum="WGS84")
    lla = Proj(proj="latlong", datum="WGS84")
    lon, lat, alt = transform(ecef, lla, x, y, z, radians=True)
    return lat, lon, alt

def ecef_to_lla2(x, y, z):
    ecef_crs = CRS.from_epsg(4978)
    wgs84_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(ecef_crs, wgs84_crs)
    lat, lon, alt = transformer.transform(x, y, z, radians=True)
    return lat, lon, alt


def TestHorizon(Body,ra,dec):
    X,Y,Z=Body["MS"].StationPos.mean(axis=0)
    #lat,lon,alt=ecef_to_lla(X, Y, Z)
    lat,lon,alt=ecef_to_lla2(X, Y, Z)

    loc_Instr=EarthLocation(lon=lon, lat=lat, height=0*u.m)
    
    F_times=Body["MS"].F_times
    TestTimes=np.linspace(F_times.min(),F_times.max(),10)
    dir_pointing = SkyCoord(np.mean(ra) * u.rad, np.mean(dec) * u.rad)

    LAlt=[]
    for ThisTestTimes in TestTimes:
        tm = Time(ThisTestTimes / 86400.0, scale="utc", format='mjd')
        dir_pointing_altaz = dir_pointing.transform_to(AltAz(obstime=tm, location=loc_Instr))
        LAlt.append(dir_pointing_altaz.alt.deg)

    Max=np.max(LAlt)
    log.print("%s max. elevation of %.1f deg"%(Body["Name"],Max))
    if Max<0:
        log.print(ModColor.Str("  Rejecting [%s] based on max. elevation"%Body["Name"]))
    return (Max>0)

class ClassAppendSource():
    def __init__(self,SM,ListBody):
        self.ListBody=ListBody
        self.SM=SM
        
        
    def appendAll(self):
        ListBody=self.ListBody
        LA=[]
        for Body in ListBody:
            tm = Time(Body["Time"] / 86400.0, scale="utc", format='mjd')

                        
            if Body["Name"]=="Sun": 
                B=ephem.Sun()
                B.compute(tm.iso)

                if not TestHorizon(Body,B.ra,B.dec):
                    continue
                
                A = np.zeros((1,),dtype=self.SM.SourceCat.dtype)
                A=A.view(np.recarray)
                A.ra=float(B.ra)
                A.dec=float(B.dec)
                d=angDist(self.SM.rarad,float(B.ra),self.SM.decrad,float(B.dec))*180/np.pi
                log.print("  [%s] On %s, position is ra/dec = %s %s [%.2f deg from target]"%(Body["Name"],tm.datetime,B.ra,B.dec,d))
                A.I=1000.
                A.Sref=A.I
                if not self.SM.InputCatIsEmpty:
                    #A.RefFreq=self.SM.SourceCat.RefFreq[0]
                    C=np.max(self.SM.SourceCat.Cluster)+1
                else:
                    #A.RefFreq=100e6
                    C=0
                    self.SM.InputCatIsEmpty=False
                
                A.RefFreq=100e6
                A.Cluster=C
                A.Type=2
                A.Gmaj=30./60*np.pi/180
                A.Gmin=A.Gmaj
                A.Name="c%is%i.ATeam_%s"%(C,0,Body["Name"])
            else:
                if Body["FileName"]=="Default":
                    path = os.path.dirname(os.path.abspath(__file__))
                    FName="%s/Models/LOFAR/%s.txt"%(path,Body["Name"])
                    log.print("  [%s] Reading file %s"%(Body["Name"],FName))
                    A,IsClustered=ReadBBSModel(FName)
                else:
                    FName=Body["FileName"]
                    log.print("  [%s] Reading file %s"%(Body["Name"],FName))
                    A,IsClustered=ReadBBSModel(FName,PatchName=Body["Name"])
                    
                ra,dec=np.mean(A.ra),np.mean(A.dec)
                if not TestHorizon(Body,A.ra,A.dec):
                    continue
                
                d=angDist(self.SM.rarad,ra,self.SM.decrad,dec)*180/np.pi
                ras  = rad2hmsdms(ra,Type="ra").replace(" ",":")
                decs = rad2hmsdms(dec,Type="dec").replace(" ",".")
                log.print("         Position is ra/dec = %s %s [%.2f deg from target]"%(ras,decs,d))


                if not self.SM.InputCatIsEmpty:
                    #A.RefFreq=self.SM.SourceCat.RefFreq[0]
                    C=np.max(self.SM.SourceCat.Cluster)+1
                else:
                    #A.RefFreq=100e6
                    C=0
                    self.SM.InputCatIsEmpty=False

                A.Type[:]=0
                A.Gmin[:]=0
                A.Gmaj[:]=0
                A.Gangle[:]=0
                A.Cluster[:]=C
                Ns=A.shape[0]
                for iSource in range(Ns):
                    A.Name[iSource]="c%is%i.ATeam_%s"%(C,iSource,Body["Name"])

            # print("!!!!!!!!!!! testnenufar")
            # A.I/=100
            # A.Sref/=100
            self.SM.SourceCat=np.hstack([self.SM.SourceCat,A])
            self.SM.SourceCat=self.SM.SourceCat.view(np.recarray)
        
                
