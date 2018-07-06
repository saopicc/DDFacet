import numpy as np
from SkyModel.Other import ModColor

def init(n=20):
    global x,y,s,ns
    ns=n
    x=np.random.randn(ns)#-0.5
    y=np.random.randn(ns)#-0.5
    s=np.random.rand(ns)
    s[0]=10
    s[1]=100

def test():
    tessel(x,y,s,3)

class ClassClusterTessel():
    def __init__(self,x,y,s,NCluster=10,DoPlot=True):
        self.X=x
        self.Y=y
        self.S=s
        self.DoPlot=DoPlot
        self.NCluster=NCluster


    def Cluster(self,DoPlot=True):
        NCluster=self.NCluster
        x,y,s=self.X,self.Y,self.S
    
        Dx=Dy=0.#(np.max(x)-np.min(x))*0.1#5
        Ncells=400
        NcellsSq=Ncells**2
        #x0,x1,y0,y1=-1,1,-1,1
        x0,x1,y0,y1=np.min(x)-Dx,np.max(x)+Dx,np.min(y)-Dy,np.max(y)+Dy
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
        #print "xrange=(%5.2f,%5.2f) yrange=(%5.2f,%5.2f)"%(x0,x1,y0,y1)
        if DoPlot:
            import pylab
            #pylab.ion()
            pylab.clf()
            pylab.imshow(im.T[::-1,:],interpolation="nearest",extent=(x0,x1,y0,y1))#,exten)
            pylab.scatter(x,y,s=np.sqrt(s)*10)#,vmin=0,vmax=nk)
            pylab.draw()
            pylab.show()
    
    
        N=x.shape[0]
        DictNode={}
        for i in range(N):
            npix=np.where(im==i)[0].shape[0]
            DictNode[i]={"SourceNum":[],"TotalFlux":s[i],"npix":npix,
                         "ListCluster":[i],"pos":(x[i],y[i]),"spread":0.}
        
        self.buildBorder(im,DictNode,ax=0)
        self.buildBorder(im,DictNode,ax=1)
        for i in range(N): DictNode[i]["FluxPerPixel"]=DictNode[i]["TotalFlux"]/DictNode[i]["npix"]
        for i in range(N):
            if DictNode[i]["SourceNum"]==[]:
                #print "%i has no neigboors!"%i
                xint=int(DictNode[i]["pos"][0]/(x1-x0)*(Ncells-1))
                yint=int(DictNode[i]["pos"][1]/(y1-y0)*(Ncells-1))
                ind=im[xint,yint]
                #DictNode[i]["SourceNum"]=DictNode[ind]["SourceNum"]
                DictNode[ind]["ListCluster"]+=DictNode[i]["ListCluster"]
                del(DictNode[i])
    
        ListDictNode=sorted(DictNode.iteritems(), key=lambda x: x[1]["TotalFlux"])#,reverse=True)
        
    
        while True:
            #printStatus(DictNode)
            ListDictNode=sorted(DictNode.iteritems(), key=lambda x: x[1]["TotalFlux"])#,reverse=True)
            #ListDictNode=sorted(DictNode.iteritems(), key=lambda x: x[1]["npix"])#*x[1]["TotalFlux"])#,reverse=True)
            iii=0#int(np.random.rand(1)[0]*len(ListDictNode))
            DictNode=self.mergeOne(DictNode,ListDictNode[iii][0])
            #print len(DictNode.keys())
            
            if len(DictNode.keys())<=NCluster: break
            #print
    
        #printStatus(DictNode)
        icol=1
        imout=np.zeros_like(im)
        for i in DictNode.keys():
            #print i
            for j in DictNode[i]["ListCluster"]:
                ind=np.where(im==j)
                imout[ind]=icol
            icol+=1
        im=imout
    
        if DoPlot:
            pylab.clf()
            pylab.imshow(im.T[::-1,:],interpolation="nearest",extent=(x0,x1,y0,y1))#,exten)
            pylab.scatter(x,y,s=np.sqrt(s)*10)#,vmin=0,vmax=nk)
            pylab.draw()
            pylab.show()
    
    
    
        return DictNode
    
    
    
    
    
    
    def mergeOne(self,DictNode,i):
        #if not(i in DictNode.keys()): continue
        #print "==================="
        #print "Cell: %i with flux %5.2f has neigbos %s"%(i,DictNode[i]["TotalFlux"],str(DictNode[i]["SourceNum"]))
        
        NeighBoorList=DictNode[i]["SourceNum"]
        NeighBoorFlux=np.zeros((len(NeighBoorList),),dtype=float)
        thisPos=(DictNode[i]["pos"][0],DictNode[i]["pos"][1])
        for j in range(len(NeighBoorList)):
            jj=NeighBoorList[j]
            dist=np.sqrt((thisPos[0]-DictNode[jj]["pos"][0])**2+(thisPos[1]-DictNode[jj]["pos"][1])**2)
            lc=np.array(DictNode[jj]["ListCluster"])
            #dist=np.max(np.sqrt((thisPos[0]-x[lc])**2+(thisPos[1]-y[lc])**2))
            ThisFlux=DictNode[jj]["TotalFlux"]
            #NeighBoorFlux[j]=dist**6
            NeighBoorFlux[j]=ThisFlux*dist**2#*dist#DictNode[jj]["spread"]**4#DictNode[jj]["TotalFlux"]*dist*DictNode[jj]["TotalFlux"]*dist**3
            #print "Neighboor %i: %5.2f"%(NeighBoorList[j],NeighBoorFlux[j])
        ind=NeighBoorList[np.argmin(NeighBoorFlux)]
        
    
        #print ModColor.Str(" Append cell %i to cell %i"%(i,ind))
        #append cell if still exists
    
        #print " The cell %i is already associated with: "%ind,DictNode[ind]["ListCluster"]
        DictNode[ind]["TotalFlux"]+=DictNode[i]["TotalFlux"]
        DictNode[ind]["npix"]+=DictNode[i]["npix"]
        DictNode[ind]["ListCluster"]+=DictNode[i]["ListCluster"]#.append(i)
        del(DictNode[i])
        for j in DictNode.keys():
            if i in DictNode[j]["SourceNum"]: 
                #print "   Remove cell %i from neigboors of %i"%(i,j)
                #print "before:",DictNode[j]["SourceNum"]
                DictNode[j]["SourceNum"].remove(i)
                #print "after:",DictNode[j]["SourceNum"]
    
    
        for j in NeighBoorList:
            if ind==j: continue
            #print "   Add cell %i to neigboors of %i"%(ind,j)
            if not(ind in DictNode[j]["SourceNum"]): DictNode[j]["SourceNum"].append(ind)
            if not(j in DictNode[ind]["SourceNum"]): DictNode[ind]["SourceNum"].append(j)
    
        stot=0.
        xc=0.
        yc=0.
        s=self.S
        x=self.X
        y=self.Y
        for i in DictNode[ind]["ListCluster"]:
            xc+=s[i]*x[i]
            yc+=s[i]*y[i]
            stot+=s[i]
        DictNode[ind]["pos"]=(xc/stot,yc/stot)
    
        lc=np.array(DictNode[ind]["ListCluster"])
        xc=np.sum(s[lc]*x[lc])/np.sum(s[lc])
        yc=np.sum(s[lc]*y[lc])/np.sum(s[lc])
        
            
        DictNode[ind]["spread"]=np.sqrt(np.sum((x[lc]-xc)**2+(y[lc]-yc)**2))
        return DictNode
    
    def printStatus(self,DictNode):
        
        print
        print "State:"
        for ind in DictNode.keys():
            print ModColor.Str("Cluster %i: "%ind,col="green")
            print "  - ListCluster : %s"%str(DictNode[ind]["ListCluster"])
            print "  - SourceNum   : %s"%str(DictNode[ind]["SourceNum"])
            print "  - Flux        : %s"%str(DictNode[ind]["TotalFlux"])
    
    
    
    def buildBorder(self,im,DictNode,ax=0):
        if ax==0:
            ind=np.where(im[0:-1,:]!=im[1::,:])
        else:
            ind=np.where(im[:,0:-1]!=im[:,1::])
        ind0=list(ind)
        a=(ind0[ax]).copy()
        a-=1
        a[a<0]=0
        ind0[ax]=a
    
        ind1=list(ind)
        a=(ind1[ax]).copy()
        a+=1
        a[a>im.shape[0]-1]=im.shape[0]-1
        ind1[ax]=a
    
        #print im
        #print
        Border=np.array([im[ind0],im[ind],im[ind1]])
        #print Border
    
    
        for i in range(Border.shape[1]):
            i,j,k=Border[:,i]
            
            if i!=j:
                if not(i in DictNode[j]["SourceNum"]): DictNode[j]["SourceNum"].append(i)
                if not(j in DictNode[i]["SourceNum"]): DictNode[i]["SourceNum"].append(j)
            if j!=k:
                if not(k in DictNode[j]["SourceNum"]): DictNode[j]["SourceNum"].append(k)
                if not(j in DictNode[k]["SourceNum"]): DictNode[k]["SourceNum"].append(j)
    
        #print DictNode
    
    def PlotTessel(self,extent=None):
        import pylab
        x=self.X.flatten()
        y=self.Y.flatten()
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
    
