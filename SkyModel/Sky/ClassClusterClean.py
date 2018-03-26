import numpy as np
from SkyModel.Other import ModColor
from scipy.spatial import Voronoi
import ModVoronoi
from SkyModel.Other import ModCoord

class ClassClusterClean():
    def __init__(self,x,y,s,NCluster=10,DoPlot=True):
        self.X=x
        self.Y=y
        self.S=s
        self.NCluster=NCluster
        self.DoPlot=DoPlot
        #self.PlotTessel()




    def Cluster(self):
        NCluster=self.NCluster
        nk=self.NCluster
        DoPlot=self.DoPlot
        x,y,s=self.X,self.Y,self.S
        x0,x1=np.min(x),np.max(x)#0.9*np.min(x),1.1*np.max(x)
        y0,y1=np.min(y),np.max(y)#0.9*np.min(y),1.1*np.max(y)
    
        Dx=1e-2#5*np.abs(x1-x0)
        Dy=1e-2#5*np.abs(y1-y0)
        x0,x1=x0-Dx,x1+Dx
        y0,y1=y0-Dy,y1+Dy
        # for i in range(x.shape[0]):
        #     print "(%6.2f, %6.2f)"%(x[i]*1000,y[i]*1000)

        #xx,yy=np.mgrid[x0:x1:40j,y0:y1:40j]
        Np=100
        xx,yy=np.mgrid[x0:x1:Np*1j,y0:y1:Np*1j]
        
        sigx=1.5*(x0-x1)/nk
        sigy=1.5*(y0-y1)/nk
        #sigx=1.*(x0-x1)/nk
        #sigy=1.*(y0-y1)/nk
    
        xnode=[]
        ynode=[]
        ss=np.zeros(xx.shape,dtype=xx.dtype)
        ss0=np.zeros(xx.shape,dtype=xx.dtype)
        dx,dy=(x1-x0)/Np,(y1-y0)/Np
    
    
        C=1./(2.*np.pi*sigx*sigy)
        for i in range(x.shape[0]):
            ss+=s[i]*C*np.exp(-((xx-x[i])**2/sigx**2+(yy-y[i])**2/sigy**2))*dx*dy
            ss0+=C*np.exp(-((xx-x[i])**2/sigx**2+(yy-y[i])**2/sigy**2))*dx*dy
            
        # xr=np.random.rand(1000)
        # yr=np.random.rand(1000)
        # C=1./(2.*np.pi*sigx*sigy)
        # for i in range(xr.shape[0]):
        #     ss0+=C*np.exp(-((xx-xr[i])**2/sigx**2+(yy-yr[i])**2/sigy**2))
        # ss0/=xr.shape[0]
        # ssn=ss/ss0#(ss-ss0)/ss0
    
        #ss/=np.sqrt(ss0)
        #ss+=np.random.randn(*ss.shape)*1e-6

        #ssn=ss
        # pylab.scatter(xx,yy,marker="s",c=ss,s=50)
        # pylab.scatter(x,y,c=s,s=np.sqrt(s)*50)
        # pylab.colorbar()
        # pylab.draw()
        # #time.sleep(1)
        # #return
        vmin,vmax=np.min(ss),np.max(ss)
        #print
        #print vmin,vmax
        if DoPlot:
            import pylab
            pylab.ion()
            #pylab.scatter(xx,yy,marker="s",c=ss,s=50,vmin=vmin,vmax=vmax)
            pylab.imshow(ss.reshape((Np,Np)).T[::-1,:],vmin=vmin,vmax=vmax,extent=(x0,x1,y0,y1))
            pylab.xlim(x0,x1)
            pylab.ylim(y0,y1)
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)
        for i in range(nk):
            ind=np.where(ss==np.max(ss))
            xnode.append(xx[ind])
            ynode.append(yy[ind])
            ss-=ss[ind]*np.exp(-((xx-xnode[-1])**2/(sigx)**2+(yy-ynode[-1])**2/(sigy)**2))


            if DoPlot:
                pylab.clf()
                #pylab.scatter(xx,yy,marker="s",c=ss,s=50)#,vmin=vmin,vmax=vmax)
                #pylab.scatter(xnode[-1],ynode[-1],marker="s",vmin=vmin,vmax=vmax)
                pylab.imshow(ss.reshape((Np,Np)).T[::-1,:],vmin=vmin,vmax=vmax,extent=(x0,x1,y0,y1))
                pylab.scatter(xnode,ynode,marker="s",color="red",vmin=vmin,vmax=vmax)
                #pylab.scatter(x,y,marker="o")#,vmin=vmin,vmax=vmax)
                #pylab.colorbar()
                pylab.xlim(x0,x1)
                pylab.ylim(y0,y1)
                pylab.draw()
                pylab.show()
                pylab.pause(0.1)

        if DoPlot:
            pylab.ioff()

        xnode=np.array(xnode)
        ynode=np.array(ynode)
        KK={}
        keys=[]
        for i in range(nk):
            key="%3.3i"%i
            keys.append(key)
            KK[key]={"ListCluster":[]}
    
    
        for i in range(x.shape[0]):
            d=np.sqrt((x[i]-xnode)**2+(y[i]-ynode)**2)
            ind=np.where(d==np.min(d))[0][0]
            (KK[keys[ind]])["ListCluster"].append(i)

        self.xnode=xnode
        self.ynode=ynode

        if DoPlot:
            pylab.clf()
            Dx=Dy=0.01
            extent=(np.min(x)-Dx,np.max(x)+Dx,np.min(y)-Dy,np.max(y)+Dy)
            self.PlotTessel(extent)
        #self.infile_cluster="%s.cluster"%self.TargetList
        #f=file(self.infile_cluster,"w")
    

        return KK

        # for key in keys:
        #     ll=KK[key]
                
        #     self.SourceCat.Cluster[ll]=int(key)
        #     ss="%s "*len(ll)
        #     #print self.SourceCat.Name[ll]
        #     lout=[key]+self.SourceCat.Name[ll].tolist()
        #     #sout=("Cluster%s "+ss)%(tuple(lout))
        #     sout=("%s "+ss)%(tuple(lout))
        #     #print sout
        #     ind=ll
        #     #f.write(sout+"\n")
        #     cc=np.ones((len(ll),))*int(key[-2::])

        #     if DoPlot: pylab.scatter(x[ind],y[ind],c=cc,s=np.sqrt(s[ind])*50,vmin=0,vmax=nk)
        # if DoPlot:
        #     pylab.tight_layout()
        #     pylab.draw()
        #     pylab.show()
        # self.NDir=np.max(self.SourceCat.Cluster)+1
        # #f.close()
        # #import os
        # #os.system("cat %s"%self.infile_cluster)

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
        xy[:,0]=xc.ravel()
        xy[:,1]=yc.ravel()
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


    def PlotTessel(self,extent=None):
        import pylab
        x=self.xnode.flatten()
        y=self.ynode.flatten()
        Ncells=400
        NcellsSq=Ncells**2
        Dx=Dy=0.
        #x0,x1,y0,y1=-1,1,-1,1
        if extent==None:
            x0,x1,y0,y1=np.min(x)-Dx,np.max(x)+Dx,np.min(y)-Dy,np.max(y)+Dy
        else:
            x0,x1,y0,y1=extent
        gx,gy=np.mgrid[x0:x1:Ncells*1j,y0:y1:Ncells*1j]
        
        CatCell=np.zeros((gx.flatten().shape[0],),dtype=[("ToNumNode",np.int),("xcell",float),("ycell",float)])
        CatCell=CatCell.view(np.recarray)
        CatCell.xcell=gx.reshape((NcellsSq,))
        CatCell.ycell=gy.reshape((NcellsSq,))
        
        d=np.sqrt((CatCell.xcell.reshape(NcellsSq,1)-x)**2+(CatCell.ycell.reshape(NcellsSq,1)-y)**2)
        CatCell.ToNumNode=np.argmin(d,axis=1)
        
        xt=x.reshape(x.shape[0],1)
        yt=y.reshape(y.shape[0],1)
        dSources=np.sqrt((xt-x)**2+(yt-y)**2)
        
        im=CatCell.ToNumNode.reshape(Ncells,Ncells)
        pylab.imshow(im.T[::-1,:],interpolation="nearest",extent=(x0,x1,y0,y1),aspect="auto")#,exten)
        pylab.xlabel("l [radian]")
        pylab.ylabel("m [radian]")
        pylab.xlim(x0,x1)
        pylab.ylim(y0,y1)
        pylab.draw()
        pylab.show()
