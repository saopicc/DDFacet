import numpy as np
from scipy.spatial import Voronoi
import ModVoronoi
from SkyModel.Other import ModCoord

def test():
    
    N0=1000
    x=np.random.rand(N0)*0.1
    y=np.random.rand(N0)*0.1
    s=np.random.rand(N0)**2
    #s.fill(1)
    s=(s-np.min(s))/(np.max(s)-np.min(s))
    s+=1
    s*=10
    
    s[0]=500
    s[1]=30


    Nk=10

    CM=ClassClusterKMean(x,y,s,Nk)
    CM.Cluster()

class ClassClusterKMean():
    def __init__(self,x,y,s,NCluster=10,DoPlot=True,
                 PreCluster=None,
                 InitLM=None):
        self.X=x.copy()
        self.Y=y.copy()
        self.S=s.copy()
        self.NCluster=NCluster
        self.DoPlot=DoPlot
        self.PreCluster=PreCluster
        self.InitLM=InitLM
        




    def Cluster(self):
        x=self.X
        y=self.Y
        s=self.S

        nk=Nk=self.NCluster
        if self.InitLM is None:
            indC=np.int32(np.random.rand(Nk)*x.size)
            xc,yc=x[indC].copy(),y[indC].copy()
        else:
            xc,yc=self.InitLM


        if self.PreCluster!=None:
            xc1,yc1=self.PreCluster
            Npk=xc1.size
            xc[0:Npk]=xc1[:]
            yc[0:Npk]=yc1[:]
        

        DicoSources={}
    
        ns=x.size
        if s.max()!=s.min():
            sz=(s-s.min())/(s.max()-s.min())*10+2
        else:
            sz=np.ones_like(s)*10

        # sz=s

        # Sk=np.ones((Nk,),np.float32)
        # d=np.sqrt((x.reshape((ns,1))-xc.reshape((1,Nk)))**2+(y.reshape((ns,1))-yc.reshape((1,Nk)))**2)
        # indk=np.argmin(d,axis=1)
        # for iK in range(Nk):
        #     ind=np.where(indk==iK)[0]
        #     if ind.size==0: continue
        #     xx=x[ind]
        #     yy=y[ind]
        #     ss=s[ind]
        #     xc[iK]=np.sum(ss*xx)/np.sum(ss)
        #     yc[iK]=np.sum(ss*yy)/np.sum(ss)
        #     Sk=np.sum(ss)


        NITerMax=20
        NIter=0
        while NIter<NITerMax:#True:
            #d=np.abs(s.reshape((ns,1))**2)*np.sqrt((x.reshape((ns,1))-xc.reshape((1,Nk)))**2+(y.reshape((ns,1))-yc.reshape((1,Nk)))**2)

            NIter+=1
            if NIter==NITerMax:
                print "Has reached max iter of %i"%NITerMax
            d=np.sqrt((x.reshape((ns,1))-xc.reshape((1,Nk)))**2+(y.reshape((ns,1))-yc.reshape((1,Nk)))**2)
            indk=np.argmin(d,axis=1)
            
            # indk=np.zeros((ns,),np.int64)
            # for iSource in range(ns):
            #     xs=x[iSource]
            #     ys=y[iSource]
            #     dk=np.abs((xs-xc.reshape((Nk,1)))*Sk.reshape((Nk,1))-(xs-xc.reshape((1,Nk)))*Sk.reshape((1,Nk))))
            #     indk[iSource]=np.arg
                


            xc0=xc.copy()
            yc0=yc.copy()
            if self.DoPlot:
                import pylab
                pylab.clf()
            for iK in range(Nk):
                ind=np.where(indk==iK)[0]
                if ind.size==0: continue
                xx=x[ind]
                yy=y[ind]
                ss=s[ind]
                
                xc[iK]=np.sum(ss*xx)/np.sum(ss)
                yc[iK]=np.sum(ss*yy)/np.sum(ss)
                #xc[iK]=np.mean(xx)
                #yc[iK]=np.mean(yy)
                c=np.ones(xx.size)*iK
                ssz=sz[ind]
                



            if self.DoPlot:
                #pylab.scatter(xx,yy,c=c,s=ssz,vmin=0,vmax=Nk,lw=0)
                pylab.scatter(xc,yc,c="black",marker="s")

            if self.PreCluster!=None:
                xc1,yc1=self.PreCluster
                Npk=xc1.size
                xc[0:Npk]=xc1[:]
                yc[0:Npk]=yc1[:]
                if self.DoPlot:
                    pylab.scatter(xc1,yc1,c="red",marker="s")




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
        
        if self.DoPlot:
            pylab.clf()
            pylab.scatter(x,y,c=indk,s=ss,vmin=0,vmax=Nk,lw=0)
            pylab.scatter(xc,yc,c="black",marker="s")
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)
    

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
            ThisText=""

            # lmean=np.mean(polygon[:,0])
            # mmean=np.mean(polygon[:,1])
            # xm,ym=self.CoordMachine.lm2radec(np.array([lmean]),np.array([mmean]))
            # xm*=180./np.pi
            # ym*=180./np.pi
            # ThisText=str(labels[iFacet])
            # f.write("point(%f,%f) # text={%s} point=circle 5 color=red width=2\n"%(xm,ym,ThisText))

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
