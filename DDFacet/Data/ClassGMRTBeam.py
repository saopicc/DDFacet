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
log= logger.getLogger("ClassGMRTBeam")
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor

import numpy as np

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

class ClassGMRTBeam():
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
        self.calcCoefs()


        
    def calcCoefs(self):
        ChanFreqs=self.MS.ChanFreq.flatten()
        DicoCoefs={}
        DicoCoefs[3]={"Name":"Band-3", "f0":250e6, "f1":500e6,
                      "C":np.array([1, 0, -2.939/1e3, 0, 33.312/1e7, 0, -16.659/1e10, 0, 3.066/1e13])}
        DicoCoefs[4]={"Name":"Band-4", "f0":550e6,"f1":850e6,
                      "C":np.array([1, 0, -3.190/1e3, 0, 38.642/1e7, 0, -20.471/1e10, 0, 3.964/1e13])}
        DicoCoefs[5]={"Name":"Band-5", "f0":1050e6,"f1":1450e6,
                      "C":np.array([1, 0, -2.608/1e3, 0, 27.357/1e7, 0, -13.091/1e10, 0, 2.368/1e13])}
        NBand=len(DicoCoefs)

        nu0=ChanFreqs[0]
        for iBand in  DicoCoefs.keys():
            C=DicoCoefs[iBand]["C"][::-1]
            x0=np.roots(C)
            x0deg=x0/(nu0/1e9)
            x=np.linspace(0,3.,1000)*60.
            P=np.poly1d(C)
            y=P(x*(nu0/1e9))
            Pd=np.polyder(P, m=1)
            yd=Pd(x*(nu0/1e9))
            xd0=np.roots(Pd)

            ind=np.where((np.abs(xd0.imag)<1e-6)&(xd0.real>0.))[0][0]
            xn=np.abs(xd0[ind])
            DicoCoefs[iBand]["x0"]=xn
            DicoCoefs[iBand]["y0"]=P(xn)
            # import pylab
            # pylab.clf()
            # pylab.plot(x,y)
            # pylab.plot(x,yd)
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            # stop
            # #


        
        C=np.zeros((ChanFreqs.size,9),np.float32)
        xNull=np.zeros((ChanFreqs.size,),np.float32)
        yNull=np.zeros((ChanFreqs.size,),np.float32)
        for ich in range(ChanFreqs.size):
            f=ChanFreqs[ich]
            NoMatch=True
            for iBand in DicoCoefs.keys():
                f0,f1=DicoCoefs[iBand]["f0"],DicoCoefs[iBand]["f1"]
                if (f>=f0) and (f<=f1):
                    C[ich,:]=DicoCoefs[iBand]["C"]
                    xNull[ich]=DicoCoefs[iBand]["x0"]/(f/1e9)
                    yNull[ich]=DicoCoefs[iBand]["y0"]
                    NoMatch=False
                    break
            if NoMatch:
                raise NotImplementedError("The GMRT beam is not modeled for that frequency (channel %i @ %.3f MHz)"%(ich,ChanFreqs[ich]/1e6))
        self.DicoCoefs=DicoCoefs
        self.ChansToCoefs=C
        self.xNull=xNull
        self.yNull=yNull
        
    def GiveRawBeam(self,time,ra,dec):
        #self.LoadSR()
        nch=self.MS.ChanFreq.size
        Beam=np.zeros((ra.shape[0],self.MS.na,self.MS.NSPWChan,2,2),dtype=complex)
        rac,decc=self.MS.OriginalRadec
        d=AngDist(ra,dec,rac,decc)*180./np.pi*60
        
        for ich in range(nch):
            C=self.ChansToCoefs[ich]
            Dnu=d*self.MS.ChanFreq.flat[ich]/1e9
            B=np.polynomial.polynomial.polyval(Dnu,C)
            B[d>self.xNull[ich]]=self.yNull[ich]
            B=B.reshape((-1,1))
            Beam[:,:,ich,0,0]=B[:,:]
            Beam[:,:,ich,1,1]=B[:,:]
            
        return Beam

    def GiveInstrumentBeam(self,*args,**kwargs):
        
        T=ClassTimeIt.ClassTimeIt("GiveInstrumentBeam")
        T.disable()
        Beam=self.GiveRawBeam(*args,**kwargs)
        nd,na,nch,_,_=Beam.shape
        T.timeit("0")
        MeanBeam=np.zeros((nd,na,self.NChanJones,2,2),dtype=Beam.dtype)
        for ich in range(self.NChanJones):
            indCh=np.where(self.VisToJonesChanMapping==ich)[0]
            MeanBeam[:,:,ich,:,:]=np.mean(Beam[:,:,indCh,:,:],axis=2)
        T.timeit("1")

        return MeanBeam


