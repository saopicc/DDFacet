
import numpy as np
import pylab
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
import ClassMultiScaleMachine



class ClassImageDeconvMachine():
    def __init__(self,Gain=0.3,
                 MaxMinorIter=100,NCPU=6,CycleFactor=2.5,
                 GD=None):
        #self.im=CasaImage
        
        self.Gain=Gain
        self.ModelImage=None
        self.MaxMinorIter=MaxMinorIter
        self.NCPU=NCPU
        self.CycleFactor=CycleFactor
        self.Chi2Thr=10000
        self.MaskArray=None
        self.GD=GD
        self.SubPSF=None
        self.MultiFreqMode=(self.GD["MultiFreqs"]["NFreqBands"]>1)
        self.MSMachine=ClassMultiScaleMachine.ClassMultiScaleMachine(self.GD,self.Gain)

    def GiveModelImage(self,*args): return self.MSMachine.ModelMachine.GiveModelImage(*args)

    def InitMSMF(self):
        self.MSMachine.FindPSFExtent(Method="FromSideLobe")
        self.MSMachine.MakeMultiScaleCube()
        self.MSMachine.MakeBasisMatrix()


    def SetDirtyPSF(self,DicoDirty,DicoPSF):
        # if len(PSF.shape)==4:
        #     self.PSF=PSF[0,0]
        # else:
        #     self.PSF=PSF

        self.DicoDirty=DicoDirty
        self.DicoPSF=DicoPSF
        #self.NChannels=self.DicoDirty["NChannels"]
        self.MSMachine.SetDirtyPSF(DicoDirty,DicoPSF)

        self._PSF=self.MSMachine._PSF
        self._Dirty=self.MSMachine._Dirty
        self._MeanPSF=self.MSMachine._MeanPSF
        self._MeanDirty=self.MSMachine._MeanDirty

        _,_,NPSF,_=self._PSF.shape
        _,_,NDirty,_=self._Dirty.shape

        off=(NPSF-NDirty)/2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)
        


        if self.ModelImage==None:
            self._ModelImage=np.zeros_like(self._Dirty)
        if self.MaskArray==None:
            self._MaskArray=np.zeros(self._Dirty.shape,dtype=np.bool8)




    def GiveEdges(self,(xc0,yc0),N0,(xc1,yc1),N1):
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


    def SubStep(self,(dx,dy),LocalSM):
        npol,_,_=self.Dirty.shape
        x0,x1,y0,y1=self.DirtyExtent

        xc,yc=dx,dy
        #NpixFacet=self.SubPSF.shape[-1]
        #PSF=self.CubePSFScales[iScale]
        N0=self.Dirty.shape[-1]
        N1=LocalSM.shape[-1]

        # PSF=PSF[N1/2-1:N1/2+2,N1/2-1:N1/2+2]
        # N1=PSF.shape[-1]

        Aedge,Bedge=self.GiveEdges((xc,yc),N0,(N1/2,N1/2),N1)

        


        #_,n,n=self.PSF.shape
        #PSF=self.PSF.reshape((n,n))
        #print "Fpol00",Fpol
        factor=-1.#Fpol[0,0,0]*self.Gain
        #print "Fpol01",Fpol

        nch,npol,nx,ny=LocalSM.shape
        # print Fpol[0,0,0]
        # print Aedge
        # print Bedge

        #print>>log, "    Removing %f Jy at (%i %i) (peak of %f Jy)"%(Fpol[0,0,0]*self.Gain,dx,dy,Fpol[0,0,0])
        #PSF=self.PSF[0]


        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge

        # nxPSF=self.CubePSFScales.shape[-1]
        # x0,x1=nxPSF/2-self.SupWeightWidth,nxPSF/2+self.SupWeightWidth+1
        # y0,y1=nxPSF/2-self.SupWeightWidth,nxPSF/2+self.SupWeightWidth+1
        # x0p=x0+x0p
        # x1p=x0+x1p
        # y0p=y0+y0p
        # y1p=y0+y1p
        # Bedge=x0p,x1p,y0p,y1p


        # pylab.clf()
        # ax=pylab.subplot(1,3,1)
        # vmin,vmax=self.Dirty.min(),self.Dirty.max()
        # pylab.imshow(self.Dirty[0,x0d:x1d,y0d:y1d],interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.subplot(1,3,2)
        # pylab.imshow(LocalSM[0,0,x0p:x1p,y0p:y1p],interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.draw()
        # #print "Fpol02",Fpol
        # # NpParallel.A_add_B_prod_factor((self.Dirty),LocalSM,Aedge,Bedge,factor=float(factor),NCPU=self.NCPU)

        self._Dirty[:,:,x0d:x1d,y0d:y1d]-=LocalSM[:,:,x0p:x1p,y0p:y1p]
        if self.MultiFreqMode:
            self._MeanDirty[0,:,x0d:x1d,y0d:y1d]-=np.mean(LocalSM[:,:,x0p:x1p,y0p:y1p],axis=0)

        # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
        # pylab.imshow(self.Dirty[0,x0d:x1d,y0d:y1d],interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.draw()
        # pylab.show(False)
        # print Aedge
        # print Bedge
        # print self.Dirty[0,x0d:x1d,y0d:y1d]

        
        


    def setChannel(self,ch=0):
        self.PSF=self._MeanPSF[ch]
        self.Dirty=self._MeanDirty[ch]
        self.ModelImage=self._ModelImage[ch]
        self.MaskArray=self._MaskArray[ch]

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe
        self.MSMachine.setSideLobeLevel(SideLobeLevel,OffsetSideLobe)

    def Clean(self,Nminor=None,ch=0):
        if Nminor==None:
            Nminor=self.MaxMinorIter

        self.setChannel(ch)

        _,npix,_=self.Dirty.shape
        xc=(npix)/2

        npol,_,_=self.Dirty.shape

        m0,m1=self.Dirty[0].min(),self.Dirty[0].max()
        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.imshow(self.Dirty[0],interpolation="nearest",vmin=m0,vmax=m1)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        print>>log, "  Running minor cycle [MaxMinorIter = %i, CycleFactor=%3.1f]"%(Nminor,self.CycleFactor)

        NPixStats=1000
        RandomInd=np.int64(np.random.rand(NPixStats)*npix**2)
        RMS=np.std(np.real(self.Dirty.ravel()[RandomInd]))
        
        self.RMS=RMS
        Threshold_RMS=5./(1.-self.SideLobeLevel)
        MaxDirty=np.max(np.abs(self.Dirty))
        FluxLimit=Threshold_RMS*RMS
        #FluxLimit_SideLobe=MaxDirty*(1.-self.SideLobeLevel)
        #Threshold_SideLobe=self.CycleFactor*MaxDirty*(self.SideLobeLevel)
        Threshold_SideLobe=((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*MaxDirty

        
        


        mm0,mm1=self.Dirty.min(),self.Dirty.max()
        print>>log, "    Dirty image peak flux   = %7.3f Jy [(min, max) = (%7.3f, %7.3f) Jy]"%(MaxDirty,mm0,mm1)
        print>>log, "    RMS threshold flux      = %7.3f Jy [rms      = %7.3f Jy]"%(FluxLimit, RMS)
        print>>log, "    Sidelobs threshold flux = %7.3f Jy [sidelobe = %7.3f of peak]"%(Threshold_SideLobe,self.SideLobeLevel)

        MaxModelInit=np.max(np.abs(self.ModelImage))

        
        # Fact=4
        # self.BookKeepShape=(npix/Fact,npix/Fact)
        # BookKeep=np.zeros(self.BookKeepShape,np.float32)
        # NPixBook,_=self.BookKeepShape
        # FactorBook=float(NPixBook)/npix
        
        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=1)
        #print x,y

        if ThisFlux < FluxLimit:
            print>>log, ModColor.Str("    Initial maximum peak %f Jy lower that rms-based limit of %f Jy (%i-sigma)" % (ThisFlux,FluxLimit,Threshold_RMS))
            return "DoneMinFlux"

        self._MaskArray.fill(1)
        self._MaskArray[np.abs(self._Dirty) > Threshold_SideLobe]=0

#        DoneScale=np.zeros((self.MSMachine.NScales,),np.float32)
        for i in range(Nminor):

            #x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=1)
            x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=1,Mask=self.MaskArray)

            # #x,y=1224, 1994
            # print x,y,ThisFlux
            # x,y=np.where(np.abs(self.Dirty[0])==np.max(np.abs(self.Dirty[0])))
            # ThisFlux=self.Dirty[0,x,y]
            # print x,y,ThisFlux
            # stop

            T.timeit("max0")

            if ThisFlux < FluxLimit:
                print>>log, "    [iter=%i] Maximum peak lower that rms-based limit of %f Jy (%i-sigma)" % (i,FluxLimit,Threshold_RMS)
                # DoneScale*=100./np.sum(DoneScale)
                # for iScale in range(DoneScale.size):
                #     print>>log,"       [Scale %i] %.1f%%"%(iScale,DoneScale[iScale])
                return "MinFlux"

            if ThisFlux < Threshold_SideLobe:
                print>>log, "    [iter=%i] Peak residual flux %f Jy higher than sidelobe-based limit of %f Jy" % (i,ThisFlux, Threshold_SideLobe)
                # DoneScale*=100./np.sum(DoneScale)
                # for iScale in range(DoneScale.size):
                #     print>>log,"       [Scale %i] %.1f%%"%(iScale,DoneScale[iScale])

                return "MinFlux"

            if (i>0)&((i%1000)==0):
                print>>log, "    [iter=%i] Peak residual flux %f Jy" % (i,ThisFlux)
                

            nch,npol,_,_=self._Dirty.shape
            Fpol=np.float32((self._Dirty[:,:,x,y].reshape((nch,npol,1,1))).copy())

            #print "Fpol",Fpol
            dx=x-xc
            dy=y-xc

            T.timeit("stuff")

            #iScale=self.MSMachine.FindBestScale((x,y),Fpol)
            LocalSM=self.MSMachine.GiveLocalSM((x,y),Fpol)

            T.timeit("FindScale")
            #print iScale

            #if iScale=="BadFit": continue

                

            # box=50
            # x0,x1=x-box,x+box
            # y0,y1=y-box,y+box
            # x0,x1=0,-1
            # y0,y1=0,-1
            # pylab.clf()
            # pylab.subplot(1,2,1)
            # pylab.imshow(self.Dirty[0][x0:x1,y0:y1],interpolation="nearest",vmin=mm0,vmax=mm1)
            # #pylab.subplot(1,3,2)
            # #pylab.imshow(self.MaskArray[0],interpolation="nearest",vmin=0,vmax=1,cmap="gray")
            # # pylab.subplot(1,2,2)
            # # pylab.imshow(self.ModelImage[0][x0:x1,y0:y1],interpolation="nearest",cmap="gray")
            # #pylab.imshow(PSF[0],interpolation="nearest",vmin=0,vmax=1)
            # #pylab.colorbar()
            

            
            self.SubStep((x,y),LocalSM*self.Gain)
            T.timeit("SubStep")



            # pylab.subplot(1,2,2)
            # pylab.imshow(self.Dirty[0][x0:x1,y0:y1],interpolation="nearest",vmin=mm0,vmax=mm1)#,vmin=m0,vmax=m1)

            # #pylab.imshow(PSF[0],interpolation="nearest",vmin=0,vmax=1)
            # #pylab.colorbar()
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)







            # ######################################

            # ThisComp=self.ListScales[iScale]




            # Scale=ThisComp["Scale"]
            # DoneScale[Scale]+=1

            # if ThisComp["ModelType"]=="Delta":
            #     for pol in range(npol):
            #        self.ModelImage[pol,x,y]+=Fpol[pol,0,0]*self.Gain
                
            # elif ThisComp["ModelType"]=="Gaussian":
            #     Gauss=ThisComp["Model"]
            #     Sup,_=Gauss.shape
            #     x0,x1=x-Sup/2,x+Sup/2+1
            #     y0,y1=y-Sup/2,y+Sup/2+1

            #     _,N0,_=self.ModelImage.shape
                
            #     Aedge,Bedge=self.GiveEdges((x,y),N0,(Sup/2,Sup/2),Sup)
            #     x0d,x1d,y0d,y1d=Aedge
            #     x0p,x1p,y0p,y1p=Bedge
                

            #     for pol in range(npol):
            #         self.ModelImage[pol,x0d:x1d,y0d:y1d]+=Gauss[x0p:x1p,y0p:y1p]*pol[pol,0,0]*self.Gain

            # else:
            #     stop




            T.timeit("End")



        print>>log, ModColor.Str("    [iter=%i] Reached maximum number of iterations" % (Nminor))
        # DoneScale*=100./np.sum(DoneScale)
        # for iScale in range(DoneScale.size):
        #     print>>log,"       [Scale %i] %.1f%%"%(iScale,DoneScale[iScale])
        return "MaxIter"

