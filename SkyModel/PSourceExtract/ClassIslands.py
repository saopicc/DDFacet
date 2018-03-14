import numpy as np
import time
from SkyModel.Other import ModColor
import scipy.ndimage
from SkyModel.Other.progressbar import ProgressBar
import findrms
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClassIsland")


class ClassIslands():
    def __init__(self,A,T=None,box=(100,100),MinPerIsland=4,DeltaXYMin=2,Boost=3,DoPlot=False,FillNoise=True,
                 MaskImage=None):
        self.A=A
        self.T=T
        self.MaskImage=MaskImage
        self.Noise=None
        self.box=box
        self.MinPerIsland=MinPerIsland
        self.DeltaXYMin=DeltaXYMin
        self.FitIm=None
        self.FittedComps=None
        self.Boost=Boost
        self.ExtendIsland=0#2
        if FillNoise:
            self.fillNoise()
        self.DoPlot=DoPlot
    


    def ExtIsland(self,xin,yin,sin):
        x=np.array(xin)
        y=np.array(yin)
        LAdd=[]
        for xi in range(np.min(x),np.max(x)+1):
            ind=np.where(x==xi)[0]
            xs=x[ind];ys=y[ind]
            y0,y1=np.min(ys),np.max(ys)
            if y0-1>=0: LAdd+=[(xi,y0-1,self.A[xi,y0-1])]
            if y1+1<=self.A.shape[1]-1: LAdd+=[(xi,y1+1,self.A[xi,y1+1])]
        for yi in range(np.min(y),np.max(y)+1):
            ind=np.where(y==yi)[0]
            xs=x[ind];ys=y[ind]
            x0,x1=np.min(xs),np.max(xs)
            if x0-1>=0: LAdd+=[(x0-1,yi,self.A[x0-1,yi])]
            if x1+1<=self.A.shape[0]-1: LAdd+=[(x1+1,yi,self.A[x1+1,yi])]
    
        for i in range(len(LAdd)):
            xin.append(LAdd[i][0])
            yin.append(LAdd[i][1])
            sin.append(LAdd[i][2])


    def fillNoise(self):
        return
        A=self.A
        print ModColor.Str("Fill blanks with noise...")
        Avec=A.reshape((A.shape[1]*A.shape[0],))
        ind=np.where(np.abs(Avec)>1e-8)[0]
        Avec=Avec[ind]
        rms=findrms.findrms(Avec)
        imnoise=np.random.randn(A.shape[0],A.shape[1])*rms
        ind=np.where(np.abs(A)<1e-8)
        A[ind]=imnoise[ind]
        print ModColor.Str("  rms in the image: %6.2f mJy"%(rms*1000.),col="green",Bold=False)
        print ModColor.Str("  done ...")
   
    def GiveVal(self,A,xin,yin):
        x,y=round(xin),round(yin)
        s=A.shape[0]-1
        cond=(x<0)|(x>s)|(y<0)|(y>s)
        if cond:
            value="out"
        else:
            value="%8.2f mJy"%(A.T[x,y]*1000.)
        return "x=%4i, y=%4i, value=%10s"%(x,y,value)


    def plot(self):
        if not(self.DoPlot): return 
        import pylab
        pylab.clf()
        ax1=pylab.subplot(2,3,1)
        vmin,vmax=-np.max(self.Noise),5*np.max(self.Noise)
        vmin,vmax=-np.max(self.Noise),10#5*np.max(self.Noise)
        MaxRms=np.max(self.Noise)
        ax1.imshow(self.A,vmin=vmin,vmax=vmax,interpolation="nearest",cmap="gray",origin="lower")
        ax1.format_coord = lambda x,y : self.GiveVal(self.A,x,y)
        pylab.title("Image")

        try:
            ax2=pylab.subplot(2,3,3,sharex=ax1,sharey=ax1)
            pylab.imshow(self.Noise,vmin=0.,vmax=np.max(self.Noise),interpolation="nearest",cmap="gray",origin="lower")
            ax2.format_coord = lambda x,y : self.GiveVal(self.Noise,x,y)
            pylab.title("Noise Image")
            pylab.xlim(0,self.A.shape[0]-1)
            pylab.ylim(0,self.A.shape[0]-1)
        except:
            pass

        if self.FitIm!=None:
            ax6=pylab.subplot(2,3,2,sharex=ax1,sharey=ax1)
            ax6.imshow(self.FitIm,vmin=vmin,vmax=vmax,interpolation="nearest",cmap="gray",origin="lower")
            ax6.format_coord = lambda x,y : self.GiveVal(self.FitIm,x,y)
            pylab.xlim(0,self.A.shape[0]-1)
            pylab.ylim(0,self.A.shape[0]-1)
            pylab.title("Fit Image")
            ax4=pylab.subplot(2,3,5,sharex=ax1,sharey=ax1)
            ax4.imshow(self.FitIm,vmin=vmin,vmax=vmax,interpolation="nearest",cmap="gray",origin="lower")
            ax4.format_coord = lambda x,y : self.GiveVal(self.FitIm,x,y)
            pylab.title("Fit Image + pos")
            pylab.xlim(0,self.A.shape[0]-1)
            pylab.ylim(0,self.A.shape[0]-1)
            if self.FittedComps!=None:
                x,y,s=self.FittedComps
                ax4.scatter(y,x,marker="o",color="red",s=3)

            ax5=pylab.subplot(2,3,4,sharex=ax1,sharey=ax1)
            Resid=self.A-self.FitIm
            ax5.imshow(Resid,vmin=vmin,vmax=vmax,interpolation="nearest",cmap="gray",origin="lower")
            ax5.format_coord = lambda x,y : self.GiveVal(Resid,x,y)
            pylab.title("Redidual Image")
            pylab.xlim(0,self.A.shape[0]-1)
            pylab.ylim(0,self.A.shape[0]-1)

        ax3=pylab.subplot(2,3,6,sharex=ax1,sharey=ax1)
        ax3.imshow(self.ImIsland,vmin=vmin,vmax=vmax,interpolation="nearest",cmap="gray",origin="lower")
        ax3.format_coord = lambda x,y : self.GiveVal(self.ImIsland,x,y)
        pylab.title("Island Image")
        pylab.xlim(0,self.A.shape[0]-1)
        pylab.ylim(0,self.A.shape[0]-1)

        pylab.draw()
        pylab.show()
        
    def ComputeNoiseMap(self):
        print ModColor.Str("Compute noise map...")
        Boost=self.Boost
        Acopy=self.A[0::Boost,0::Boost].copy()
        SBox=(self.box[0]/Boost,self.box[1]/Boost)
        Noise=np.sqrt(scipy.ndimage.filters.median_filter(np.abs(Acopy)**2,SBox))
        self.Noise=np.zeros_like(self.A)
        for i in range(Boost):
            for j in range(Boost):
                s00,s01=Noise.shape
                s10,s11=self.Noise[i::Boost,j::Boost].shape
                s0,s1=min(s00,s10),min(s10,s11)
                self.Noise[i::Boost,j::Boost][0:s0,0:s1]=Noise[:,:][0:s0,0:s1]
        print ModColor.Str(" ... done")
        ind=np.where(self.Noise==0.)
        self.Noise[ind]=1e-10


    def FindAllIslands(self):
        A=self.A
        if (self.Noise is None) and (self.MaskImage is None):
            self.ComputeNoiseMap()

        
        


        # N=self.Noise
        # T=self.T
        # if T!=None:
        #     #self.plot()
        #     indx,indy=np.where((A/N)>T)
        #     Abool=((A/N)>T)
        # else:
        #     indx,indy=np.where(self.MaskImage!=0)
        #     Abool=self.MaskImage

        # Lpos=[(indx[i],indy[i]) for i in range(indx.shape[0])]
        # LIslands=[]
        
        # # import pylab
        # # pylab.imshow(Abool)
        # # pylab.draw()
        # # pylab.show()

        # pBAR = ProgressBar('white', block='=', empty=' ',Title="Find islands")
        # Lini=len(Lpos)
        # while True:
        #     l=[]
        #     #print Lpos
        #     self.FindIsland(Abool,l,Lpos[0][0],Lpos[0][1])
        #     #print l
        #     LIslands.append(l)
        #     Lpos=list(set(Lpos)-set(l))
        #     comment=''
        #     pBAR.render(int(100* float(Lini-len(Lpos)) / Lini), comment)
        #     if len(Lpos)==0: break


        LIslandsOut=[]
        #ImIsland=np.zeros_like(self.A)
        inum=1

        self.ListX=[]
        self.ListY=[]
        self.ListS=[]


        import scipy.ndimage

        print>>log,"  Labeling islands"
        self.ImIsland,NIslands=scipy.ndimage.label(self.MaskImage)
        ImIsland=self.ImIsland
        #NIslands+=1
        nx,_=ImIsland.shape

        print>>log,"  Found %i islands"%NIslands
        

        NMaxPix=500**2
        
        #Island=np.zeros((NIslands,NMaxPix,2),np.int32)
        NIslandNonZero=np.zeros((NIslands,),np.int32)

        print>>log,"  Extractinng pixels in islands"
        pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="      Extracting ", HeaderSize=10,TitleSize=13)
        comment=''



        # for ipix in range(nx):
        #     pBAR.render(int(100*ipix / (nx-1)), comment)
        #     for jpix in range(nx):
        #         iIsland=self.ImIsland[ipix,jpix]
        #         if iIsland:
        #             NThis=NIslandNonZero[iIsland-1]
        #             Island[iIsland-1,NThis,0]=ipix
        #             Island[iIsland-1,NThis,1]=jpix
        #             NIslandNonZero[iIsland-1]+=1

        indx,indy=np.where(self.ImIsland)
        Label=self.ImIsland[indx,indy]
        nx=indx.size

        print>>log,"  Listing pixels in islands"
        LIslands=[]
        for iIsland in range(1,NIslands+1):
            ind,=np.where(Label==iIsland)
            LIslands.append(np.array([indx[ind],indy[ind]]).T.tolist())

        # ##############################
        # for iPix in range(nx):
        #     #pBAR.render(int(100*iPix / (nx-1)), comment)
        #     x,y=indx[iPix],indy[iPix]
        #     iIsland=self.ImIsland[x,y]
        #     if iIsland:
        #         NThis=NIslandNonZero[iIsland-1]
        #         Island[iIsland-1,NThis,0]=x
        #         Island[iIsland-1,NThis,1]=y
        #         NIslandNonZero[iIsland-1]+=1
        # print>>log,"  Listing pixels in islands"
        # LIslands=[]
        # for iIsland in range(NIslands):
        #     ind=np.where(Island[iIsland,:,0]!=0)[0]
        #     ThisIsland=[]
        #     Npix=ind.size
        #     for ipix in range(Npix):
        #         ThisIsland.append([Island[iIsland,ipix,0].tolist(),Island[iIsland,ipix,1].tolist()])
        #     LIslands.append(ThisIsland)
        # ##############################

        
        print>>log,"  Selecting pixels in islands"
        for i in LIslands:
            condMinPix=(len(i)>self.MinPerIsland)
            xpos=[i[ii][0] for ii in range(len(i))]
            ypos=[i[ii][1] for ii in range(len(i))]
            if len(xpos)==0: continue
            dx=np.max(xpos)-np.min(xpos)
            dy=np.max(ypos)-np.min(ypos)
            condDelta=(dx>=self.DeltaXYMin)&(dy>=self.DeltaXYMin)
            if condMinPix&condDelta:
                LIslandsOut.append(i)
                X=[];Y=[];S=[]
                for ii in i:

                    X.append(ii[0])
                    Y.append(ii[1])
                    S.append(A[ii[0],ii[1]])

#                print
#                print X,Y,S
                for iii in range(self.ExtendIsland):
                    self.ExtIsland(X,Y,S)
#                print
#                print X,Y,S

                for ii in range(len(X)):
                    ImIsland[X[ii],Y[ii]]=1#inum

#                stop
                self.ListX.append(X)
                self.ListY.append(Y)
                self.ListS.append(S)
                inum+=1

        print>>log,"  Final number of islands: %i"%len(self.ListX)
        self.ImIsland=ImIsland
        self.LIslands=LIslandsOut
    
    def FindIsland(self,A,Lpix,x,y,dirfrom=-1,threshold=1):
        T=threshold
        digo=set(range(4))-set([dirfrom])
        #print Lpix
        #time.sleep(1.)
        #if (x,y) in Lpix: return Lpix
        pos=(x,y)
        S=ModColor.Str("@(%i,%i) "%(x,y),col="blue")
        try:
            aa=A[x,y]
        except:
            return
    
    
        if A[x,y]==False:
            #print ModColor.Str("(%i,%i)"%(x,y))
            return
        if A[x,y]==True:
            #print S,ModColor.Str("(%i,%i)"%(x,y),col="green")
            Lpix.append((x,y))
        if 0 in digo:
            this=(x+1,y)
            if not(this in Lpix):
                #print S,"-> from %i and going to %i"%(dirfrom,0)
                self.FindIsland(A,Lpix,x+1,y,dirfrom=2)
        if 2 in digo:
            this=(x-1,y)
            if not(this in Lpix):
                #print S,"-> from %i and going to %i"%(dirfrom,2)
                self.FindIsland(A,Lpix,x-1,y,dirfrom=0)
        if 1 in digo:
            this=(x,y+1)
            if not(this in Lpix):
                #print S,"-> from %i and going to %i"%(dirfrom,1)
                self.FindIsland(A,Lpix,x,y+1,dirfrom=3)
        if 3 in digo:
            this=(x,y-1)
            if not(this in Lpix):
                #print "-> from %i and going to %i"%(dirfrom,3)
                self.FindIsland(A,Lpix,x,y-1,dirfrom=1)
    
    
def init():
    a=np.zeros((7,7),dtype=bool)
    a[1:3,3:5]=1
    a[1,5]=1
    a[1,1]=1
        #a[1,2]=1
    print a
    return a
    l=[]
    FindIsland(a,l,1,1)
    print l
    
def init2():
    a=np.zeros((7,7),dtype=bool)
    a[1:3,3:5]=1
    a[1,5]=1
    a[1,1]=1
    a[6,6]=1
    a[5,5]=1
    a[5,6]=1
        #a[1,2]=1
    l=[]
    A=np.zeros((a.shape),dtype=np.float)
    A[a]=10
    print A
    FindAllIslands(A,1.)
    
    
