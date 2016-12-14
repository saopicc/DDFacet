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

import _pyGridder
import numpy as np

a=np.zeros((2,2))
x=np.array([0,0,0])
y=np.array([0,0,0])
w=np.ones_like(x)
print _pyGridder.pyGridderPoints(a.astype(np.float64),x,y,w.astype(np.float64),0)


S=12
Np=10
a=np.zeros((S,S))
x=np.int64(np.random.rand(Np)*S)
y=np.int64(np.random.rand(Np)*S)
w=np.ones_like(x)
print x,y
print _pyGridder.pyGridderPoints(a.astype(np.float64),x.astype(np.int32),y.astype(np.int32),w.astype(np.float64),2)

print b.min(),b.max()
print b
# import pylab
# pylab.imshow(b)
# pylab.draw()
# pylab.show()

