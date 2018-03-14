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

def GiveEdges((xc0,yc0),N0,(xc1,yc1),N1,Parity=None):
    M_xc=xc0
    M_yc=yc0
    NpixMain=N0
    F_xc=xc1
    F_yc=yc1
    NpixFacet=N1
    
    ## X
    M_x0=M_xc-NpixFacet/2
    x0main=np.max([0,M_x0])
    dx0=x0main-M_x0
    x0facet=dx0
    
    M_x1=M_xc+NpixFacet/2
    x1main=np.min([NpixMain-1,M_x1])
    dx1=M_x1-x1main
    x1facet=NpixFacet-dx1
    x1main+=1
    ## Y
    M_y0=M_yc-NpixFacet/2
    y0main=np.max([0,M_y0])
    dy0=y0main-M_y0
    y0facet=dy0
    
    M_y1=M_yc+NpixFacet/2
    y1main=np.min([NpixMain-1,M_y1])
    dy1=M_y1-y1main
    y1facet=NpixFacet-dy1
    y1main+=1
    
    IsEven=(lambda x: x%2==0)

    if Parity is not None:
        if Parity=="Even":
            F=lambda x: IsEven(x)
        elif Parity=="Odd":
            F=lambda x: not IsEven(x)

        dx=x1main-x0main
        dy=y1main-y0main
        if F(dx):
            x1main-=1
            x1facet-=1

        if F(dy):
            y1main-=1
            y1facet-=1

    Aedge=[x0main,x1main,y0main,y1main]
    Bedge=[x0facet,x1facet,y0facet,y1facet]
     

    return Aedge,Bedge


def GiveEdgesDissymetric((xc0,yc0),(N0x,N0y),(xc1,yc1),(N1x,N1y),WidthMax=None):
    M_xc=xc0
    M_yc=yc0
    NpixMain_x=N0x
    NpixMain_y=N0y
    F_xc=xc1
    F_yc=yc1
    NpixFacet_x=N1x
    NpixFacet_y=N1y
    
    ## X
    M_x0=M_xc-NpixFacet_x/2
    x0main=np.max([0,M_x0])
    dx0=x0main-M_x0
    x0facet=dx0
    
    M_x1=M_xc+NpixFacet_x/2
    x1main=np.min([NpixMain_x-1,M_x1])
    dx1=M_x1-x1main
    x1facet=NpixFacet_x-dx1
    x1main+=1
    ## Y
    M_y0=M_yc-NpixFacet_y/2
    y0main=np.max([0,M_y0])
    dy0=y0main-M_y0
    y0facet=dy0
    
    M_y1=M_yc+NpixFacet_y/2
    y1main=np.min([NpixMain_y-1,M_y1])
    dy1=M_y1-y1main
    y1facet=NpixFacet_y-dy1
    y1main+=1
    
    Aedge=[x0main,x1main,y0main,y1main]
    Bedge=[x0facet,x1facet,y0facet,y1facet]
    if WidthMax:
        dx=x1main-x0main
        dy=y1main-y0main
        Wx,Wy=WidthMax
        if dx>Wx:
            ddx=int((dx-Wx)/2.)
            #if ddx%2==1: ddx-=1
            x0main+=ddx
            x1main-=ddx
            x0facet+=ddx
            x1facet-=ddx
        if dy>Wy:
            ddy=int((dy-Wy)/2.)
            #if ddy%2==1: ddy-=1
            y0main+=ddy
            y1main-=ddy
            y0facet+=ddy
            y1facet-=ddy
    Aedge=[x0main,x1main,y0main,y1main]
    Bedge=[x0facet,x1facet,y0facet,y1facet]

    return Aedge,Bedge


