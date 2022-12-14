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
log= logger.getLogger("ClassNenuBeam")
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
from astropy.time  import Time, TimeDelta

# NENUFAR DEPENDENCIES
try:
    from nenupy.instru         import MiniArray, NenuFAR, Polarization, NenuFAR_Configuration, miniarrays_rotated_like
    from nenupy.astro.pointing import Pointing
    from nenupy.astro.sky      import Sky
    from nenupy.astro.target   import FixedTarget
    from astropy.coordinates   import SkyCoord
except ImportError:
    print("The DDFacet implementation of the NenuFAR beam response")
    print("relies on the nenupy library, which is not installed by")
    print("default. You can install it via pypy as follows:")
    print("")
    print("pip3 install --user --upgrade https://github.com/AlanLoh/nenupy/tarball/master")
    print("")
    print("For more information, see: https://nenupy.readthedocs.io/en/latest/install.html")


import numpy as np
import astropy.units as u


    
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

class ClassNenuBeam():
    def __init__(self,MS,GD):
        self.GD=GD
        self.MS=MS
        self.SR=None
        self.CalcFreqDomains()

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
        # initialise internal beam matrix shape
        Beam=np.zeros((nd,self.MS.na,self.MS.NSPWChan,2,2),dtype=float)
        ### convert time to astropy object with units of mjd for nenupy use
        time=Time(time/24./3600,format="mjd",scale="utc")
        ### create skycoord object of pointing direction: phase centre of MS
        obs_coordinates=SkyCoord(self.MS.OriginalRadec[0],self.MS.OriginalRadec[1],unit="rad")
        obs_coords=FixedTarget(obs_coordinates)
        ### intrinsic dt of nenufar pointing: 6min. try to encode that somehow. Currently use 1s TODO
        pointing=Pointing.target_tracking(target=obs_coords,
                                          time=time,
                                          duration=TimeDelta(1, format="sec"))
        ### calculate frequencies
        freqs=self.MS.ChanFreq.flatten()*u.Hz
        ### initialise nenufar configuration to account for beam squint
        conf = NenuFAR_Configuration(beamsquint_correction=True)
        T.timeit("Init")
        ### get sky coordinates at which to estimate nenufar beam
        # XX polarisation
        beam_coords_XX=Sky(SkyCoord(ras*u.rad,decs*u.rad,frame="icrs").ravel(),time=time,frequency=freqs,polarization=Polarization.NW)
        # YY polarisation
        beam_coords_YY=Sky(SkyCoord(ras*u.rad,decs*u.rad,frame="icrs").ravel(),time=time,frequency=freqs,polarization=Polarization.NE)
        T.timeit("beam_coords")
        ### there are only 6 mini-array rotations in the NenuFAR array. Initialise them
        ma_list=self.MS.StationNames
        rotations = np.arange(0, 60, 10)
        ma_rotated_like = list(map(miniarrays_rotated_like, rotations.reshape(6, 1)))
        available_rotations = dict(zip(rotations.astype(str), ma_rotated_like))
        # beams
        T.timeit("available_rotations")

        ### below is the rotation optimisation initialisation
        beam_rot = {}
        for rotation, mas in available_rotations.items():
            # redefine pointing per rotation - at time of writing this is necessary to avoid
            # errors due to the initial pointing object being modified at each call (21/04/2022)
            pointing=Pointing.target_tracking(target=obs_coords,
                                              time=time,
                                              duration=TimeDelta(1, format="sec"))
            # loop 6 iterations
            T.timeit("   Pointing.target_tracking")
            ma = MiniArray(index=mas[0])
            T.timeit("   ma")
            ### calculate beam response
            # calculate beam. daskarray.value.compute() returns a np.array from a np.darray
            # configuration=conf is the old parameter call for beamsquint; TODO check if we still need it!
            beamvals_XX=ma.array_factor(sky=beam_coords_XX,pointing=pointing,return_complex=True).compute()
            beamvals_YY=ma.array_factor(sky=beam_coords_YY,pointing=pointing,return_complex=True).compute()
            #beamvals_XX=ma.beam(sky=beam_coords_XX,pointing=pointing,configuration=conf).value.compute()
            #beamvals_YY=ma.beam(sky=beam_coords_YY,pointing=pointing,configuration=conf).value.compute()

            beam_rot[rotation] = np.array([beamvals_XX,beamvals_YY])
            T.timeit("   beam_rot")
        T.timeit("for rot")
        ### assign the appropriate rotations to the mini-arrays in the current observation
        for i, ma in enumerate(ma_list):
            ma_index = int(ma.strip("MR").strip("NEN"))
            rotation_key = [key for key, val in available_rotations.items() if ma_index in val][0]
            #gros_tableau[i, ....] = beam_rot["rotation_key"]
            ### reshape to DDF internal beam matrix
            for idir in range(nd):
                # for polarisations: a prioi NW is XX, NE is YY
                Beam[idir,i,:,0,0]=beam_rot[rotation_key][0][0,:,0,idir] # polarisation 1
                Beam[idir,i,:,1,1]=beam_rot[rotation_key][1][0,:,0,idir] # polarisation 2

        T.timeit("ma_list")

        MeanBeam=np.zeros((nd,self.MS.na,self.NChanJones,2,2),dtype=Beam.dtype)
        for ich in range(self.NChanJones):
            indCh=np.where(self.VisToJonesChanMapping==ich)[0]
            MeanBeam[:,:,ich,:,:]=np.mean(Beam[:,:,indCh,:,:],axis=2)
        T.timeit("NChan")

        return MeanBeam

