

from SkyModel.Other import ModCoord
import numpy as np
from scipy.spatial import Voronoi
import ModVoronoi

class VoronoiToReg():
    def __init__(self,rac,decc,xc,yc):
        self.rac=rac
        self.decc=decc
        self.xc=xc
        self.yc=yc
        self.CoordMachine=ModCoord.ClassCoordConv(rac,decc)

    def ToReg(self,regFile,radius=0.1):

        f=open(regFile,"w")
        f.write("# Region file format: DS9 version 4.1\n")
        ss0='global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0'
        ss1=' fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
        
        f.write(ss0+ss1)
        f.write("fk5\n")
 
        xc=self.xc
        yc=self.yc
        CoordMachine=self.CoordMachine

        xy=np.zeros((xc.size,2),np.float32)
        xy[:,0]=xc
        xy[:,1]=yc
        vor = Voronoi(xy)
        regions, vertices = ModVoronoi.voronoi_finite_polygons_2d(vor,radius=radius)


        for region in regions:
            polygon0 = vertices[region]
            P=polygon0.tolist()
            polygon=np.array(P+[P[0]])
            for iline in range(polygon.shape[0]-1):
                
                x0,y0=CoordMachine.lm2radec(np.array([polygon[iline][0]]),np.array([polygon[iline][1]]))
                x1,y1=CoordMachine.lm2radec(np.array([polygon[iline+1][0]]),np.array([polygon[iline+1][1]]))

                x0*=180./np.pi
                y0*=180./np.pi
                x1*=180./np.pi
                y1*=180./np.pi

                f.write("line(%f,%f,%f,%f) # line=0 0 color=red dash=1\n"%(x0,y0,x1,y1))
                #f.write("line(%f,%f,%f,%f) # line=0 0 color=red dash=1\n"%(x1,y0,x0,y1))
            
        f.close()
