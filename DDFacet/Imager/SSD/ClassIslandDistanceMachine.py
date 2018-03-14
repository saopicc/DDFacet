import time
import numpy as np
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassIslandDistanceMachine")
from DDFacet.Other.progressbar import ProgressBar
from SkyModel.PSourceExtract import ClassIslands
from SkyModel.PSourceExtract import ClassIncreaseIsland
from DDFacet.Array import NpShared
from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import psutil

class Island(object):
    '''
    helper class for MergeIsland
    '''
    def __init__(self,island):
        if isinstance(island,set):
            self.ilist=[[j[0],j[1]] for j in island]
            self.iset=island
        else:
            self.iset=None
            self.ilist=island
        self.npa=np.array(self.ilist)
        self.minx=np.min(self.npa[:,0])
        self.maxx=np.max(self.npa[:,0])
        self.miny=np.min(self.npa[:,1])
        self.maxy=np.max(self.npa[:,1])
        self.merged=False
        self.new=False
    def overlap(self,other):
        # check if the bounding box overlaps
        if (other.maxx<self.minx or other.maxy<self.miny or
            other.minx>self.maxx or other.miny>self.maxy):
            return False
        return True
    def make_set(self):
        if self.iset is None:
            self.iset=set(((j[0],j[1]) for j in self.ilist))
    def intersect(self,other):
        self.make_set()
        other.make_set()
        inter=self.iset.intersection(other.iset)
        return len(inter)>0
    def merge(self,other):
        self.make_set()
        other.make_set()
        merged=self.iset.union(other.iset)
        other.merged=True
        return Island(merged)
    def plot(self,**kwargs):
        plt.scatter(self.npa[:,0],self.npa[:,1],**kwargs)


class ClassIslandDistanceMachine():
    def __init__(self,GD,MaskArray,PSFServer,DicoDirty,IdSharedMem=""):
        self.GD=GD
        self._MaskArray=MaskArray
        self.PSFServer=PSFServer
        self.PSFCross=None
        self.DicoDirty=DicoDirty
        if self.GD is not None:
            self.NCPU=(self.GD["Parallel"]["NCPU"] or psutil.cpu_count())
        else:
            self.NCPU=psutil.cpu_count()
        self.IdSharedMem=IdSharedMem

    def SearchIslands(self,Threshold,Image=None):
        print>>log,"Searching Islands"
        if Image is not None:
            Dirty=Image
        else:
            Dirty=self.DicoDirty["MeanImage"]
        
        # self.IslandArray[0,0]=(Dirty[0,0]>Threshold)|(self.IslandArray[0,0])
        # MaskImage=(self.IslandArray[0,0])&(np.logical_not(self._MaskArray[0,0]))
        # MaskImage=(np.logical_not(self._MaskArray[0,0]))

        MaskImage=None
        if self._MaskArray is not None:
            MaskImage=(np.logical_not(self._MaskArray[0,0]))
        Islands=ClassIslands.ClassIslands(Dirty[0,0],MaskImage=MaskImage,
                                          MinPerIsland=0,DeltaXYMin=0)
        Islands.FindAllIslands()

        ListIslands=Islands.LIslands

        print>>log,"  found %i islands"%len(ListIslands)
        if self.GD is not None:
            dx=self.GD["SSDClean"]["NEnlargePars"]
            if dx>0:
                print>>log,"  increase their sizes by %i pixels"%dx
                IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
                for iIsland in range(len(ListIslands)):#self.NIslands):
                    ListIslands[iIsland]=IncreaseIslandMachine.IncreaseIsland(ListIslands[iIsland],dx=dx)

        
        return ListIslands

    def CalcLabelImage(self,ListIslands):
        print>>log,"  calculating label image"
        _,_,nx,_=self._MaskArray.shape
        Labels=np.zeros((nx,nx),dtype=np.float32)

        for iIsland,ThisIsland in enumerate(ListIslands):
            x,y=np.array(ThisIsland).T
            Labels[np.int32(x),np.int32(y)]=iIsland+1
        return Labels.reshape((1,1,nx,nx))

    def CalcCrossIslandPSF(self,ListIslands):
        print>>log,"  calculating global islands cross-contamination"
        PSF=np.mean(np.abs(self.PSFServer.DicoVariablePSF["MeanFacetPSF"][:,0]),axis=0)#self.PSFServer.DicoVariablePSF["MeanFacetPSF"][0,0]
        
        
        nPSF,_=PSF.shape
        xcPSF,ycPSF=nPSF/2,nPSF/2

        IN=lambda x: ((x>=0)&(x<nPSF))


        NIslands=len(ListIslands)
        # NDone=0
        # NJobs=NIslands
        # pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title=" Calc Cross Contam.", HeaderSize=10,TitleSize=13)
        # #pBAR.disable()
        # pBAR.render(0, '%4i/%i' % (0,NJobs))


        # PSFCross=np.zeros((NIslands,NIslands),np.float32)
        # for iIsland in range(NIslands):
        #     NDone+=1
        #     intPercent=int(100*  NDone / float(NJobs))
        #     pBAR.render(intPercent, '%4i/%i' % (NDone,NJobs))
        #     x0,y0=np.array(ListIslands[iIsland]).T
        #     xc0,yc0=int(np.mean(x0)),int(np.mean(y0))
        #     for jIsland in range(iIsland,NIslands):
        #         x1,y1=np.array(ListIslands[jIsland]).T
        #         xc1,yc1=int(np.mean(x1)),int(np.mean(y1))
        #         dx,dy=xc1-xc0+xcPSF,yc1-yc0+xcPSF
        #         if (IN(dx))&(IN(dy)):
        #             PSFCross[iIsland,jIsland]=np.abs(PSF[dx,dy])
        # Diag=np.diag(np.diag(PSFCross))
        # PSFCross+=PSFCross.T
        # PSFCross.flat[0::NIslands+1]=Diag.flat[0::NIslands+1]

        # xMean=np.zeros((NIslands,),np.int32)
        # yMean=xMean.copy()
        # for iIsland in range(NIslands):
        #     x0,y0=np.array(ListIslands[iIsland]).T
        #     xc0,yc0=int(np.mean(x0)),int(np.mean(y0))
        #     xMean[iIsland]=xc0
        #     yMean[iIsland]=yc0
        # dx=xMean.reshape((NIslands,1))-xMean.reshape((1,NIslands))
        # dy=yMean.reshape((NIslands,1))-yMean.reshape((1,NIslands))

        #self.calcDistanceMatrixMean(ListIslands)
        self.calcDistanceMatrixMinParallel(ListIslands)
        dx,dy=self.dx,self.dy
        self.DistCross=np.sqrt(dx**2+dy**2)

        dx+=xcPSF
        dy+=xcPSF
        PSFCross=np.zeros((NIslands,NIslands),np.float32)
        indPSF=np.arange(NIslands**2)
        Cx=((dx>=0)&(dx<nPSF))
        Cy=((dy>=0)&(dy<nPSF))
        C=(Cx&Cy)
        indPSF_sel=indPSF[C.ravel()]
        indPixPSF=dx.ravel()[C.ravel()]*nPSF+dy.ravel()[C.ravel()]
        PSFCross.flat[indPSF_sel]=np.abs(PSF.flat[indPixPSF.ravel()])



        
        self.PSFCross=PSFCross

    def GiveNearbyIsland(self,iIsland,SetThisIsland):
        Th=0.05
        # print "======================="
        # print SetThisIsland
        if iIsland in self.setCheckedIslands:
            #print "  Island #%3.3i already associated"%iIsland
            return SetThisIsland
        self.setCheckedIslands.add(iIsland)
        #indNearbyIsland=np.where((self.PSFCross[iIsland])>Th)[0]

        D0,D1=self.GD["SSDClean"]["MinMaxGroupDistance"]

        CTh=(self.PSFCross[iIsland]>Th)
        C0=(self.DistCross[iIsland]<D0)
        C1=(self.DistCross[iIsland]<D1)
        setNearbyIsland=set(np.where( (CTh | C0) & (C1) )[0].tolist())
        setNearbyIsland=setNearbyIsland.difference(SetThisIsland)
        
        #print "  #%3.3i <- %i islands"%(iIsland,len(setNearbyIsland))
        if len(setNearbyIsland)>0:
            SetThisIsland=SetThisIsland.union(setNearbyIsland)
            for iIsland in SetThisIsland:
                SetThisIsland=self.GiveNearbyIsland(iIsland,SetThisIsland)
        return SetThisIsland
            




    def CalcCrossIslandFlux(self,ListIslands):
        if self.PSFCross is None:
            self.CalcCrossIslandPSF(ListIslands)
        NIslands=len(ListIslands)
        print>>log,"  grouping cross contaminating islands..."

        MaxIslandFlux=np.zeros((NIslands,),np.float32)
        DicoIsland={}

        Dirty=self.DicoDirty["MeanImage"]


        for iIsland in range(NIslands):

            x0,y0=np.array(ListIslands[iIsland]).T
            PixVals0=Dirty[0,0,x0,y0]
            MaxIslandFlux[iIsland]=np.max(PixVals0)
            DicoIsland[iIsland]=ListIslands[iIsland]

        self.CrossFluxContrib=self.PSFCross*MaxIslandFlux.reshape((1,NIslands))
        self.DicoIsland=DicoIsland

        NDone=0
        NJobs=NIslands
        pBAR= ProgressBar(Title=" Group islands")
        pBAR.disable()
        pBAR.render(0, NJobs)

        Th=0.05
        
        ListIslandMerged=[]
        self.setCheckedIslands=set([])
        for iIsland in range(NIslands):
            x0,y0=np.array(ListIslands[iIsland]).T
            #print "Main %i (%f, %f)"%(iIsland,np.mean(x0),np.mean(y0))
            
            NDone+=1
            intPercent=int(100*  NDone / float(NJobs))
            pBAR.render(NDone,NJobs)
            
            ListIslandMerged.append(list(self.GiveNearbyIsland(iIsland,set([]))))

        
        ListIslands=[]
        for indIsland in ListIslandMerged:
            if len(indIsland)==0: continue
            ThisIsland=DicoIsland[indIsland[0]]
            for iIsland in indIsland[1::]:
                ThisIsland+=DicoIsland[iIsland]
            ListIslands.append(ThisIsland)

        print>>log,"    have grouped %i --> %i islands"%(NIslands, len(ListIslands))

        return ListIslands

    def calcDistanceMatrixMean(self,ListIslands):
        NIslands=len(ListIslands)
        xMean=np.zeros((NIslands,),np.int32)
        yMean=xMean.copy()
        for iIsland in range(NIslands):
            x0,y0=np.array(ListIslands[iIsland]).T
            xc0,yc0=int(np.mean(x0)),int(np.mean(y0))
            xMean[iIsland]=xc0
            yMean[iIsland]=yc0

        self.dx=xMean.reshape((NIslands,1))-xMean.reshape((1,NIslands))
        self.dy=yMean.reshape((NIslands,1))-yMean.reshape((1,NIslands))
        self.D=np.sqrt(self.dx**2+self.dy**2)
        

    
    def calcDistanceMatrixMin(self,ListIslands):
        NIslands=len(ListIslands)
        self.D=np.zeros((NIslands,NIslands),np.float32)
        self.dx=np.zeros((NIslands,NIslands),np.int32)
        self.dy=np.zeros((NIslands,NIslands),np.int32)

        pBAR= ProgressBar(Title=" Calc Dist")
        #pBAR.disable()
        NDone=0; NJobs=NIslands
        pBAR.render(0,NJobs)
        for iIsland in range(NIslands):
            x0,y0=np.array(ListIslands[iIsland]).T
            for jIsland in range(iIsland+1,NIslands):
                x1,y1=np.array(ListIslands[jIsland]).T
                
                dx=x0.reshape((-1,1))-x1.reshape((1,-1))
                dy=y0.reshape((-1,1))-y1.reshape((1,-1))
                d=np.sqrt(dx**2+dy**2)
                dmin=np.min(d)
                self.D[jIsland,iIsland]=self.D[iIsland,jIsland]=dmin
                indx,indy=np.where(d==dmin)
                #print dx[indx[0],indy[0]],dy[indx[0],indy[0]],dmin
                self.dx[jIsland,iIsland]=self.dx[iIsland,jIsland]=dx[indx[0],indy[0]]
                self.dy[jIsland,iIsland]=self.dy[iIsland,jIsland]=dy[indx[0],indy[0]]
            NDone+=1

            pBAR.render(NDone,NJobs)

    def giveEdgesIslands(self,ListIslands):
        print>>log,"  extracting Island edges"
        ListEdgesIslands=[]
        _,_,nx,_=self._MaskArray.shape
        #Ed=np.zeros_like(self._MaskArray)
        for Island in ListIslands:
            x,y=np.array(Island).T
            EdgesIsland=[]
            for iPix in range(x.size):
                xc,yc=x[iPix],y[iPix]
                Aedge,Bedge=GiveEdgesDissymetric((xc,yc),(nx,nx),(1,1),(3,3))
                x0d,x1d,y0d,y1d=Aedge
                m=self._MaskArray[0,0][x0d:x1d,y0d:y1d]
                if 1 in m:
                    EdgesIsland.append((xc,yc))
                    #Ed[0,0,xc,yc]=1
            ListEdgesIslands.append(EdgesIsland)
            
        # import pylab
        # ax=pylab.subplot(1,2,1)
        # pylab.imshow(self._MaskArray[0,0],interpolation="nearest")
        # pylab.subplot(1,2,2,sharex=ax,sharey=ax)
        # pylab.imshow(Ed[0,0],interpolation="nearest")
        # pylab.draw()
        # pylab.show(False)

        return ListEdgesIslands

    def ConvexifyIsland(self,ListIslands):
        print>>log,"  Convexify islands"
        ListConvexIslands=[]
        for Island in ListIslands:
            points=np.array(Island)
            x,y=points.T
            Cx=(np.abs(x.min()-x.max())==0)
            Cy=(np.abs(y.min()-y.max())==0)
            if (x.size<=3) or Cx or Cy:
                ListConvexIslands.append(Island)
                continue
            try:
                hull = ConvexHull(points)
                Contour = np.array(
                    [hull.points[hull.vertices, 0],
                     hull.points[hull.vertices, 1]])
                poly2 = Contour.T
                
                x0,x1=x.min(),x.max()
                y0,y1=y.min(),y.max()
                
                xx,yy=np.mgrid[x0:x1:(x1-x0+1)*1j,y0:y1:(y1-y0+1)*1j]
                xx=np.int16(xx)
                yy=np.int16(yy)
                
                pp=np.zeros((poly2.shape[0]+1,2),dtype=poly2.dtype)
                pp[0:-1,:]=poly2[:,:]
                pp[-1,:]=poly2[0,:]
                mpath = Path(pp)
                
                p_grid=np.zeros((xx.size,2),np.int16)
                p_grid[:,0]=xx.ravel()
                p_grid[:,1]=yy.ravel()
                mask_flat = mpath.contains_points(p_grid)
            
                IslandOut=np.array([xx.ravel()[mask_flat],yy.ravel()[mask_flat]]).T.tolist()
                ListConvexIslands.append(IslandOut)
                #x,y=np.array(Island).T
                #xedge,yedge=Contour
                # import pylab
                # pylab.clf()
                # pylab.scatter(xx.ravel(),yy.ravel(),marker="+",c="red")
                # pylab.scatter(xx.ravel()[mask_flat],yy.ravel()[mask_flat],marker="o",c="green")
                # pylab.scatter(x,y)
                # pylab.plot(xedge,yedge)
                # pylab.draw()
                # pylab.show(False)
            except:
                ListConvexIslands.append(Island)

        return ListConvexIslands

    def MergeIslands(self,ListIslands):
        print>>log,"  Merge intersecting islands"
        Islands=[Island(i) for i in ListIslands]

        i=0
        while i<len(Islands)-1:
            merged=False
            for j in range(i+1,len(Islands)):
                if Islands[j].merged:
                    continue
                if Islands[i].overlap(Islands[j]):
                    if Islands[i].intersect(Islands[j]):
                        merged=True
                        Islands[i]=Islands[i].merge(Islands[j])
                        Islands[i].new=True
            if not(merged):
                i+=1
        result=[i.ilist for i in Islands if not i.merged]
        print>>log,"  %i islands remaining after merge" % len(result)
        return result

    def calcDistanceMatrixMinParallel(self,ListIslands,Parallel=True):
        NIslands=len(ListIslands)
        self.D=np.zeros((NIslands,NIslands),np.float32)
        self.dx=np.zeros((NIslands,NIslands),np.int32)
        self.dy=np.zeros((NIslands,NIslands),np.int32)

        work_queue = multiprocessing.JoinableQueue()
        for iIsland in range(NIslands):
            work_queue.put({"iIsland":(iIsland)})

        result_queue=multiprocessing.JoinableQueue()
        NJobs=work_queue.qsize()
        workerlist=[]
        NCPU=self.NCPU

        ListEdgeIslands=self.giveEdgesIslands(ListIslands)

        for ii in range(NCPU):
            W = WorkerDistance(work_queue,
                               result_queue,
                               ListEdgeIslands,
                               self.IdSharedMem)
            workerlist.append(W)
            if Parallel:
                workerlist[ii].start()

        pBAR = ProgressBar(Title="  Calc. Dist. ")
        pBAR.render(0, NJobs)
        iResult = 0
        if not Parallel:
            for ii in range(NCPU):
                workerlist[ii].run()  # just run until all work is completed

        while iResult < NJobs:
            DicoResult = None
            if result_queue.qsize() != 0:
                try:
                    DicoResult = result_queue.get()
                except:
                    pass

            if DicoResult == None:
                time.sleep(0.5)
                continue

            if DicoResult["Success"]:
                iResult+=1
                NDone=iResult
                pBAR.render(NDone,NJobs)

                iIsland=DicoResult["iIsland"]
                Result=NpShared.GiveArray("%sDistances_%6.6i"%(self.IdSharedMem,iIsland))

                self.dx[iIsland]=Result[0]
                self.dy[iIsland]=Result[1]
                self.D[iIsland]=Result[2]
                NpShared.DelAll("%sDistances_%6.6i"%(self.IdSharedMem,iIsland))




        if Parallel:
            for ii in range(NCPU):
                workerlist[ii].shutdown()
                workerlist[ii].terminate()
                workerlist[ii].join()





##########################################
####### Workers
##########################################
import os
import signal
import multiprocessing

class WorkerDistance(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 ListIsland,
                 IdSharedMem):
        multiprocessing.Process.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.ListIslands=ListIsland
        self.IdSharedMem=IdSharedMem

    def shutdown(self):
        self.exit.set()


    def giveMinDist(self, DicoJob):
        iIsland=DicoJob["iIsland"]
        NIslands=len(self.ListIslands)
        Result=np.zeros((3,NIslands),np.int32)

        x0,y0=np.array(self.ListIslands[iIsland]).T
        for jIsland in range(NIslands):
            x1,y1=np.array(self.ListIslands[jIsland]).T
            dx=x0.reshape((-1,1))-x1.reshape((1,-1))
            dy=y0.reshape((-1,1))-y1.reshape((1,-1))
            d=np.sqrt(dx**2+dy**2)
            dmin=np.min(d)
            indx,indy=np.where(d==dmin)
            Res=dmin
            Result[0,jIsland]=dx[indx[0],indy[0]]
            Result[1,jIsland]=dy[indx[0],indy[0]]
            Result[2,jIsland]=dmin

        NpShared.ToShared("%sDistances_%6.6i"%(self.IdSharedMem,iIsland),Result)

        self.result_queue.put({"iIsland": iIsland, "Success":True})

    def run(self):
        while not self.kill_received and not self.work_queue.empty():
            
            DicoJob = self.work_queue.get()
            self.giveMinDist(DicoJob)
            # try:
            #     self.initIsland(DicoJob)
            # except:
            #     iIsland=DicoJob["iIsland"]
            #     print ModColor.Str("On island %i"%iIsland)
            #     print traceback.format_exc()
            #     print
            #     print self.ListIsland[iIsland]
            #     print







