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

class ClassCoordConv():
    def __init__(self,rac,decc):

        rarad=rac
        decrad=decc
        self.rarad=rarad
        self.decrad=decrad
        cos=np.cos
        sin=np.sin
        mrot=np.matrix([[cos(rarad)*cos(decrad), sin(rarad)*cos(decrad),sin(decrad)],[-sin(rarad),cos(rarad),0.],[-cos(rarad)*sin(decrad),-sin(rarad)*sin(decrad),cos(decrad)]]).T
        vm=np.matrix([[0.,0.,1.]]).T
        vl=np.matrix([[0.,1., 0.]]).T
        vn=np.matrix([[1., 0, 0.]]).T
        self.vl2=mrot*vl
        self.vm2=mrot*vm
        self.vn2=mrot*vn
        self.R=np.matrix([[cos(decrad)*cos(rarad),cos(decrad)*sin(rarad),sin(decrad)]]).T
        

    def lm2radec(self,l_list,m_list):

        ra_list=np.zeros(l_list.shape,dtype=np.float)
        dec_list=np.zeros(l_list.shape,dtype=np.float)
        
        for i in range(l_list.shape[0]):
            l=l_list[i]
            m=m_list[i]
            if (l_list[i]==0.)&(m_list[i]==0.):
                ra_list[i]=self.rarad
                dec_list[i]=self.decrad
                continue
            Rp=self.R+self.vl2*l+self.vm2*m-(1.-np.sqrt(1.-l**2-m**2))*self.vn2
            dec_list[i]=np.arcsin(Rp[2])
            ra_list[i]=np.arctan(Rp[1]/Rp[0])
            if Rp[0]<0.: ra_list[i]+=np.pi
    
        return ra_list,dec_list

    def radec2lm(self,ra,dec):
        l = np.cos(dec) * np.sin(ra - self.rarad)
        m = np.sin(dec) * np.cos(self.decrad) - np.cos(dec) * np.sin(self.decrad) * np.cos(ra - self.rarad)
        return l,m
