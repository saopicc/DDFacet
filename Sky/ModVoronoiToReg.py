

from SkyModel.Other import ModCoord
import numpy as np
from scipy.spatial import Voronoi
import ModVoronoi
from SkyModel.Other import MyLogger
from SkyModel.Other import ModColor
log=MyLogger.getLogger("VoronoiToReg")

class VoronoiToReg():
    def __init__(self,rac,decc):
        self.rac=rac
        self.decc=decc
        self.CoordMachine=ModCoord.ClassCoordConv(rac,decc)

    def ToReg(self,regFile,xc,yc,radius=0.1,Col="red"):
        print>>log, "Writing voronoi in: %s"%ModColor.Str(regFile,col="blue")
        f=open(regFile,"w")
        f.write("# Region file format: DS9 version 4.1\n")
        ss0='global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0'
        ss1=' fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
        
        f.write(ss0+ss1)
        f.write("fk5\n")
 
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


                f.write("line(%f,%f,%f,%f) # line=0 0 color=%s dash=1\n"%(x0,y0,x1,y1,Col))
                #f.write("line(%f,%f,%f,%f) # line=0 0 color=red dash=1\n"%(x1,y0,x0,y1))
            
        f.close()


    def VorToReg(self,regFile,vor,radius=0.1,Col="red"):
        print>>log,"Writing voronoi in: %s"%ModColor.Str(regFile,col="blue")

        f=open(regFile,"w")
        f.write("# Region file format: DS9 version 4.1\n")
        ss0='global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0'
        ss1=' fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
        
        f.write(ss0+ss1)
        f.write("fk5\n")
 
        CoordMachine=self.CoordMachine

        regions, vertices = vor.regions,vor.vertices
        

        for region in regions:
            if len(region)==0: continue
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

                f.write("line(%f,%f,%f,%f) # line=0 0 color=%s dash=1\n"%(x0,y0,x1,y1,Col))
                #f.write("line(%f,%f,%f,%f) # line=0 0 color=red dash=1\n"%(x1,y0,x0,y1))
            
        f.close()

    def PolygonToReg(self,regFile,LPolygon,radius=0.1,Col="red",labels=None):
        print>>log, "Writing voronoi in: %s"%ModColor.Str(regFile,col="blue")

        f=open(regFile,"w")
        f.write("# Region file format: DS9 version 4.1\n")
        ss0='global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0'
        ss1=' fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
        
        f.write(ss0+ss1)
        f.write("fk5\n")
 
        CoordMachine=self.CoordMachine
        
        
        for iFacet,polygon0 in zip(range(len(LPolygon)),LPolygon):
            #polygon0 = vertices[region]
            P=polygon0.tolist()
            if len(polygon0)==0: continue
            polygon=np.array(P+[P[0]])
            ThisText=""
            if labels!=None:
                lmean0=np.mean(polygon[:,0])
                mmean0=np.mean(polygon[:,1])

                lmean,mmean,ThisText=labels[iFacet]
                # print "!!!!======"
                # print lmean0,mmean0
                # print lmean,mmean

                xm,ym=CoordMachine.lm2radec(np.array([lmean]),np.array([mmean]))
                xm*=180./np.pi
                ym*=180./np.pi
                f.write("point(%f,%f) # text={%s} point=circle 5 color=red width=2\n"%(xm,ym,ThisText))

            for iline in range(polygon.shape[0]-1):
                

                L0,M0=np.array([polygon[iline][0]]),np.array([polygon[iline][1]])
                x0,y0=CoordMachine.lm2radec(L0,M0)
                L1,M1=np.array([polygon[iline+1][0]]),np.array([polygon[iline+1][1]])
                x1,y1=CoordMachine.lm2radec(L1,M1)

                x0*=180./np.pi
                y0*=180./np.pi
                x1*=180./np.pi
                y1*=180./np.pi

                # print "===================="
                # print "[%3.3i] %f %f %f %f"%(iline,x0,y0,x1,y1)
                # print "       %s"%str(L0)
                # print "       %s"%str(L1)
                # print "       %s"%str(M0)
                # print "       %s"%str(M1)
                f.write("line(%f,%f,%f,%f) # line=0 0 color=%s dash=1 \n"%(x0,y0,x1,y1,Col))

                #f.write("line(%f,%f,%f,%f) # line=0 0 color=red dash=1\n"%(x1,y0,x0,y1))
            
        f.close()
