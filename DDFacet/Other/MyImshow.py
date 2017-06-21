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


def GiveVal(A,xin,yin):
    x,y=round(xin),round(yin)
    s=A.shape[0]-1
    cond=(x<0)|(x>s)|(y<0)|(y>s)
    if cond:
        value="out"
    else:
        value="%8.2f mJy"%(A.T[x,y]*1000.)
    return "x=%4i, y=%4i, value=%10s"%(x,y,value)

def imshow(ax,A,*args,**kwargs):
    ax.format_coord = lambda x,y : GiveVal(A,x,y)
    ax.imshow(A,*args,**kwargs)
