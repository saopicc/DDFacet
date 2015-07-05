import numpy as np
import pylab
from scipy.spatial import Voronoi
import ModVoronoi
from SkyModel.Other import ModCoord

def test():
    
    N0=1000
    x=np.random.randn(N0)
    y=np.random.randn(N0)
    s=np.random.rand(N0)**2
    #s.fill(1)
    s=(s-np.min(s))/(np.max(s)-np.min(s))
    s+=1
    s*=10
    
    s[0]=50
    s[1]=30


    Nk=10

    CM=ClassClusterKMean(x,y,s,Nk)
    CM.Cluster()

class ClassClusterKMean():
    def __init__(self,x,y,s,NCluster=10,DoPlot=True):
        self.X=x
        self.Y=y
        self.S=s
        self.NCluster=NCluster
        self.DoPlot=DoPlot

    def Cluster(self):
        x=self.X
        y=self.Y
        s=self.S

        nk=Nk=self.NCluster
        
        indC=np.int32(np.random.rand(Nk)*x.size)
        xc,yc=x[indC],y[indC]
        DicoSources={}
    
        ns=x.size
        if s.max()!=s.min():
            sz=(s-s.min())/(s.max()-s.min())*10+2
        else:
            sz=np.ones_like(s)*10
        while True:
            #d=s.reshape((ns,1))*np.sqrt((x.reshape((ns,1))-xc.reshape((1,Nk)))**2+(y.reshape((ns,1))-yc.reshape((1,Nk)))**2)
            d=np.sqrt((x.reshape((ns,1))-xc.reshape((1,Nk)))**2+(y.reshape((ns,1))-yc.reshape((1,Nk)))**2)
            indk=np.argmin(d,axis=1)
            xc0=xc.copy()
            yc0=yc.copy()
            if self.DoPlot:
                pylab.clf()
            for iK in range(Nk):
                ind=np.where(indk==iK)[0]
                if ind.size==0: continue
                xx=x[ind]
                yy=y[ind]
                ss=s[ind]
                
                xc[iK]=np.sum(ss*xx)/np.sum(ss)
                yc[iK]=np.sum(ss*yy)/np.sum(ss)
                c=np.ones(xx.size)*iK
                ssz=sz[ind]
                
                if self.DoPlot:
                    pylab.scatter(xx,yy,c=c,s=ssz,vmin=0,vmax=Nk,lw=0)
                    pylab.scatter(xc[iK],yc[iK],c="black",marker="s")


            if self.DoPlot:

                xy=np.zeros((xc.size,2),np.float32)
                xy[:,0]=xc
                xy[:,1]=yc
                vor = Voronoi(xy)#incremental=True)

                regions, vertices = ModVoronoi.voronoi_finite_polygons_2d(vor)
                for region in regions:
                    polygon = vertices[region]
                    pylab.fill(*zip(*polygon), alpha=0.4)
                    #pylab.plot(xy[:,0], xy[:,1], 'ko')
                    #pylab.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
                    #pylab.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
                    dx=0.01
                    pylab.xlim(xc.min() - dx, xc.max()+dx)
                    pylab.ylim(yc.min() - dx, yc.max()+dx)




                pylab.draw()
                pylab.show(False)
                pylab.pause(0.1)
            condx=np.allclose(xc,xc0)
            condy=np.allclose(yc,yc0)
            if condx&condy: break

        d=np.sqrt((x.reshape((ns,1))-xc.reshape((1,Nk)))**2+(y.reshape((ns,1))-yc.reshape((1,Nk)))**2)
        indk=np.argmin(d,axis=1)
        
        # if self.DoPlot:
        #     pylab.clf()
        #     pylab.scatter(x,y,c=indk,s=ss,vmin=0,vmax=Nk,lw=0)
        #     pylab.scatter(xc,yc,c="black",marker="s")
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
    

        KK={}
        keys=[]
        for i in range(nk):
            key="%3.3i"%i
            keys.append(key)
            KK[key]={"ListCluster":[]}
    
        xnode=xc
        ynode=yc

        self.xnode=xnode
        self.ynode=ynode

        for i in range(x.shape[0]):
            d=np.sqrt((x[i]-xnode)**2+(y[i]-ynode)**2)
            ind=np.where(d==np.min(d))[0][0]
            (KK[keys[ind]])["ListCluster"].append(i)

        
        return KK
    

    def ToReg(self,regFile,rac,decc):

        f=open(regFile,"w")
        self.REGName=True
        f.write("# Region file format: DS9 version 4.1\n")
        ss0='global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0'
        ss1=' fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
        
        f.write(ss0+ss1)
        f.write("fk5\n")
 
        xc=self.xnode
        yc=self.ynode

        xy=np.zeros((xc.size,2),np.float32)
        xy[:,0]=xc
        xy[:,1]=yc
        vor = Voronoi(xy)
        regions, vertices = ModVoronoi.voronoi_finite_polygons_2d(vor)

        self.CoordMachine=ModCoord.ClassCoordConv(rac,decc)

        for region in regions:
            polygon0 = vertices[region]
            P=polygon0.tolist()
            polygon=np.array(P+[P[0]])
            for iline in range(polygon.shape[0]-1):
                
                x0,y0=self.CoordMachine.lm2radec(np.array([polygon[iline][0]]),np.array([polygon[iline][1]]))
                x1,y1=self.CoordMachine.lm2radec(np.array([polygon[iline+1][0]]),np.array([polygon[iline+1][1]]))

                x0*=180./np.pi
                y0*=180./np.pi
                x1*=180./np.pi
                y1*=180./np.pi

                f.write("line(%f,%f,%f,%f) # line=0 0 color=red dash=1\n"%(x0,y0,x1,y1))
                #f.write("line(%f,%f,%f,%f) # line=0 0 color=red dash=1\n"%(x1,y0,x0,y1))
            
        f.close()
