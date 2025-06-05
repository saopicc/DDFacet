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

from DDFacet.Other import logger
log= logger.getLogger("ClassEveryBeam")
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
from astropy.time  import Time, TimeDelta

# SKA BEAM DEPENDENCIES
from astropy.coordinates import SkyCoord
from astropy.coordinates import AltAz, EarthLocation, ITRS, SkyCoord
import astropy.units as u
import everybeam
import numpy as np
    
# be bad
import warnings
warnings.filterwarnings("ignore")#, category=DeprecationWarning) 

def AngDist(ra0,dec0,ra1,dec1):
    AC=np.arccos
    C=np.cos
    S=np.sin
    D=S(dec0)*S(dec1)+C(dec0)*C(dec1)*C(ra0-ra1)
    if type(D).__name__=="ndarray":
        D[D>1.]=1.
        D[D<-1.]=-1.
    else:
        if D>1.: D=1.
        if D<-1.: D=-1.
    return AC(D)

### update below...right now it is LOFAR TODO TODO
def radec_to_xyz(ra, dec, time, loc):
    obstime = Time(time/3600/24, scale='utc', format='mjd')
    dir_pointing = SkyCoord(ra, dec)
    dir_pointing_altaz = dir_pointing.transform_to(AltAz(obstime=obstime, location=loc))
    dir_pointing_xyz = dir_pointing_altaz.transform_to(ITRS)
    pointing_xyz = np.asarray([dir_pointing_xyz.x, dir_pointing_xyz.y, dir_pointing_xyz.z])
    return pointing_xyz


class ClassEveryBeam():
    def __init__(self,MS,GD):
        self.GD=GD
        self.MS=MS
        self.SR=None
        self.CalcFreqDomains()
        # cache the locations for everybeam
        self.AntLocs=[]
        for iant in range(self.MS.na):
            x,y,z=self.MS.StationPos[iant]
            locITRS = EarthLocation(x*u.m,y*u.m,z*u.m)
            self.AntLocs.append(locITRS)
            # below is equivalent to above.
            #self.antlocs.append(EarthLocation(lon=locITRS.lon.deg*u.deg,lat=locITRS.lat.deg*u.deg,height=locITRS.height))
        
    def getBeamSampleTimes(self,times, **kwargs):
        DtBeamMin = self.GD["DtBeamMin"]
        DtBeamSec = DtBeamMin*60
        tmin=times[0]
        tmax=times[-1]+1
        TimesBeam=np.arange(tmin,tmax,DtBeamSec).tolist()
        if not(tmax in TimesBeam): TimesBeam.append(tmax)
        return TimesBeam

    def getFreqDomains(self):
        return self.FreqDomains

    def CalcFreqDomains(self):
        ChanWidth=self.MS.ChanWidth.ravel()[0]
        ChanFreqs=self.MS.ChanFreq.flatten()
        NChanJones=self.GD["NBand"]
        if NChanJones==0:
            NChanJones=self.MS.NSPWChan
        ChanEdges=np.linspace(ChanFreqs.min()-ChanWidth/2.,ChanFreqs.max()+ChanWidth/2.,NChanJones+1)
        FreqDomains=[[ChanEdges[iF],ChanEdges[iF+1]] for iF in range(NChanJones)]
        FreqDomains=np.array(FreqDomains)
        self.FreqDomains=FreqDomains
        self.NChanJones=NChanJones
        MeanFreqJonesChan=(FreqDomains[:,0]+FreqDomains[:,1])/2.
        DFreq=np.abs(self.MS.ChanFreq.reshape((self.MS.NSPWChan,1))-MeanFreqJonesChan.reshape((1,NChanJones)))
        self.VisToJonesChanMapping=np.argmin(DFreq,axis=1)

    def evaluateBeam(self,time,ras,decs):
        return self.GiveInstrumentBeam(time,ras,decs)

    def GiveInstrumentBeam(self,time,ras,decs):
        # DDF internal: I assume this times the call to the beam for logging purposes
        T=ClassTimeIt.ClassTimeIt("GiveInstrumentBeam")
        T.disable()
        # get number of directions for later iterations
        nd=len(ras)
        # get frequencies
        freqs=self.MS.ChanFreq.flatten()*u.Hz
        # initialise internal beam matrix shape
        Beam=np.zeros((nd,self.MS.na,self.MS.NSPWChan,2,2),dtype=float)
        # load the telescope; everybeam automatically finds the appropriate setup for the provided dataset
        obs = everybeam.load_telescope(self.MS.MSName)
        # calculate the array response
        for iant in range(self.MS.na):
            obs_coords_xyz  = radec_to_xyz(self.MS.PointingRadec[0] * u.rad, self.MS.PointingRadec[1] * u.rad, time, self.AntLocs[iant])
            for idir in range(ras.size):
                # calculate the observation coords and phase coord for this station's XYZ position
                phase_xyz = radec_to_xyz(ras[idir] * u.rad, decs[idir] * u.rad, time, self.AntLocs[iant])
                for ifreq,freq in enumerate(freqs):
                    Beam[idir,iant,ifreq,:,:] = obs.array_factor(time, iant, freq, phase_xyz, obs_coords_xyz)
        # initialise average beam over frequency chunk
        MeanBeam=np.zeros((nd,self.MS.na,self.NChanJones,2,2),dtype=Beam.dtype)
        # calculate the average beam
        for ich in range(self.NChanJones):
            indCh=np.where(self.VisToJonesChanMapping==ich)[0]
            MeanBeam[:,:,ich,:,:]=np.mean(Beam[:,:,indCh,:,:],axis=2)
        T.timeit("NChan")
        return MeanBeam

