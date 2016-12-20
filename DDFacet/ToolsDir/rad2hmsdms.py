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

import numpy as np

def rad2hmsdms(rad,Type="dec",deg=False):
    
    if deg==False:
        deg=rad*180/np.pi
    else:
        deg=rad

    strsgn="+"
    if Type=="ra":
        deg/=15.
        strsgn=""

    if np.sign(deg)==-1.: strsgn="-"
    deg=np.abs(deg)
    degd=np.int(deg)
    degms=(deg-degd)*60.
    degm=np.int(degms)
    degs=((degms-degm)*60)
    return "%s%2.2i %2.2i %06.3f"%(strsgn,degd,degm,degs)
