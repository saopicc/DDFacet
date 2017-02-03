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

from pyrap.tables import table
import numpy as np
import os
import gc
import ephem
import re
import pyrap.quanta as qa
import pyrap.measures as pm
import rad2hmsdms


def Rotate2((ra,dec),(ra1,dec1),uvw,data,wavelength,RotateType=["uvw","vis"]):
    

    # preparing rotation matrices (ORIGINAL PHASE DIRECTION)
    x = np.sin(ra)*np.cos(dec)
    y = np.cos(ra)*np.cos(dec)
    z = np.sin(dec)
    w = np.array([[x,y,z]]).T
    x = -np.sin(ra)*np.sin(dec)
    y = -np.cos(ra)*np.sin(dec)
    z = np.cos(dec)
    v = np.array([[x,y,z]]).T
    x = np.cos(ra)
    y = -np.sin(ra)
    z = 0
    u = np.array([[x,y,z]]).T
    T = np.concatenate([u,v,w], axis = -1 )


    TT=np.identity(3)
    x1 = np.sin(ra1)*np.cos(dec1)
    y1 = np.cos(ra1)*np.cos(dec1)
    z1 = np.sin(dec1)
    w1 = np.array([[x1,y1,z1]]).T
    x1 = -np.sin(ra1)*np.sin(dec1)
    y1 = -np.cos(ra1)*np.sin(dec1)
    z1 = np.cos(dec1)
    v1 = np.array([[x1,y1,z1]]).T
    x1 = np.cos(ra1)
    y1 = -np.sin(ra1)
    z1 = 0
    u1 = np.array([[x1,y1,z1]]).T

    Tshift = np.concatenate([u1,v1,w1], axis=-1)

    TT = np.dot(Tshift.T,T)

    # uvw_all = MS.uvw
    # uvw=uvw_all

    Phase=np.dot(np.dot((w-w1).T, T) , uvw.T)

    if "uvw" in RotateType:
        uvw[:]=np.dot(uvw, TT.T)

    #MS.uvw=uvwNew

    if "vis" in RotateType:
        nrows,nchan,npol=data.shape
    
        C=299792458.
        wavelength=np.float64(wavelength)
        ChanFreq=C/wavelength
        ChanEquidistant=0
        if ChanFreq.size>2:
            df=ChanFreq[1::]-ChanFreq[0:-1]
            ddf=np.abs(df-np.mean(df))
            ChanEquidistant=int(np.max(ddf)<1.)


        for chan in range(nchan):
            #wavelength = MS.wavelength_chan.flatten()[chan]
            if ChanEquidistant==0:
                f = np.exp(Phase * 2 * np.pi * 1j/wavelength[chan])
            else:
                if chan==0:
                    f0 = np.exp(Phase * 2 * np.pi * 1j/wavelength[0])
                    df = np.exp(Phase * 2 * np.pi * 1j* (ChanFreq[1]-ChanFreq[0])/C)
                    f=f0
                else:
                    f*=df
    
            for pol in range(npol):
                data[:,chan,pol]=data[:,chan,pol] * f.reshape((nrows,))



def Rotate(MS,ToRaDec):
    
    ra, dec = MS.radec
    ra1,dec1=ToRaDec

    # preparing rotation matrices (ORIGINAL PHASE DIRECTION)
    x = np.sin(ra)*np.cos(dec)
    y = np.cos(ra)*np.cos(dec)
    z = np.sin(dec)
    w = np.array([[x,y,z]]).T
    x = -np.sin(ra)*np.sin(dec)
    y = -np.cos(ra)*np.sin(dec)
    z = np.cos(dec)
    v = np.array([[x,y,z]]).T
    x = np.cos(ra)
    y = -np.sin(ra)
    z = 0
    u = np.array([[x,y,z]]).T
    T = np.concatenate([u,v,w], axis = -1 )


    TT=np.identity(3)
    x1 = np.sin(ra1)*np.cos(dec1)
    y1 = np.cos(ra1)*np.cos(dec1)
    z1 = np.sin(dec1)
    w1 = np.array([[x1,y1,z1]]).T
    x1 = -np.sin(ra1)*np.sin(dec1)
    y1 = -np.cos(ra1)*np.sin(dec1)
    z1 = np.cos(dec1)
    v1 = np.array([[x1,y1,z1]]).T
    x1 = np.cos(ra1)
    y1 = -np.sin(ra1)
    z1 = 0
    u1 = np.array([[x1,y1,z1]]).T

    Tshift = np.concatenate([u1,v1,w1], axis=-1)

    TT = np.dot(Tshift.T,T)

    uvw_all = MS.uvw
    uvw=uvw_all

    Phase=np.dot(np.dot((w-w1).T, T) , uvw.T)
    uvwNew=np.dot(uvw, TT.T)
    MS.uvw=uvwNew



    for chan in range(MS.NSPWChan):
        wavelength = MS.wavelength_chan.flatten()[chan]
        f = np.exp(Phase * 2 * np.pi * 1j/wavelength)
        for pol in range(4):
            MS.data[:,chan,pol]=MS.data[:,chan,pol] * f.reshape((MS.nrows,))

# def GiveRot(a,d,remap=np.array([2,0,1])):
#     cos=np.cos
#     sin=np.sin

#     R0=np.array(
#         [[cos(a), -sin(a), 0 ],
#          [sin(a),  cos(a), 0 ],
#          [     0,       0, 1 ]]
#         )
#     R1=np.array(
#         [[cos(d), 0, -sin(d) ],
#          [     0, 1,       0 ],
#          [sin(d), 0,  cos(d) ]]
#         )
#     R=np.dot(R1,R0)
#     R=R[remap,:][:,remap]
#     return R




# def Rotate(MS,ToRaDec):
    
#     ra, dec = MS.radec
#     ra1,dec1=ToRaDec

#     # preparing rotation matrices (ORIGINAL PHASE DIRECTION)
#     R0=GiveRot(ra, dec).T
#     R1=GiveRot(ra, dec)
#     R=np.dot(R1,R0)

#     uvw_all = MS.uvw
#     uvw=uvw_all

#     Phase=np.dot(np.dot(R[-1,:], R0) , uvw.T)
#     uvwNew=np.dot(R, uvw.T)

#     import pylab
#     pylab.clf()
#     pylab.plot(Phase)
#     pylab.draw()
#     pylab.show()
#     stop
    
#     # for chan in range(MS.NSPWChan):
#     #     wavelength = MS.wavelength_chan.flatten()[chan]
#     #     f = np.exp(Phase * 2 * np.pi * 1j/wavelength)
#     #     for pol in range(4):
#     #         MS.data[:,chan,pol]=MS.data[:,chan,pol] * f.reshape((MS.nrows,))



