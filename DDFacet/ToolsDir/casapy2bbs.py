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
import rad2hmsdms
from pyrap.images import image

def Print(ImageName="testImage.fits"):

    im=image(ImageName)
    d=im.getdata()
    _,_,lx,ly=np.where(d!=0)

    for x,y in zip(lx,ly):
        _,_,dec,ra=im.toworld([0,0,x,y])
        print rad2hmsdms.rad2hmsdms(ra,Type="ra").replace(" ",":"),", ",rad2hmsdms.rad2hmsdms(dec,Type="dec").replace(" ",".")
        
        
