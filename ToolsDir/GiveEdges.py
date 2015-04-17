import numpy as np

def GiveEdges((xc0,yc0),N0,(xc1,yc1),N1):
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
    
    Aedge=[x0main,x1main,y0main,y1main]
    Bedge=[x0facet,x1facet,y0facet,y1facet]
    return Aedge,Bedge

