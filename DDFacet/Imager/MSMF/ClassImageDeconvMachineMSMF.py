
import numpy as np
import math
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
from DDFacet.Imager.MSMF import ClassMultiScaleMachine
from pyrap.images import image
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
import ClassModelMachineMSMF
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Imager import ClassGainMachine
import cPickle

class ClassImageDeconvMachine():
    def __init__(self,Gain=0.3,
                 MaxMinorIter=100,NCPU=6,
                 CycleFactor=2.5,FluxThreshold=None,RMSFactor=3,PeakFactor=0,
                 GD=None,SearchMaxAbs=1,CleanMaskImage=None,ModelMachine=None,
                 NFreqBands=1,
                 MainCache=None,
                 **kw    # absorb any unknown keywords arguments into this
                 ):
        #self.im=CasaImage

        self.SearchMaxAbs=SearchMaxAbs
        self._ModelImage = None
        self.MaxMinorIter=MaxMinorIter
        self.NCPU=NCPU
        self.Chi2Thr=10000
        self._MaskArray=None
        self.GD=GD
        self.SubPSF=None
        self.MultiFreqMode = NFreqBands>1
        self.NFreqBands = NFreqBands
        self.FluxThreshold = FluxThreshold 
        self.CycleFactor = CycleFactor
        self.RMSFactor = RMSFactor
        self.PeakFactor = PeakFactor
        self.GainMachine=ClassGainMachine.ClassGainMachine(GainMin=Gain)
        self.ModelMachine = ModelMachine
        self.RefFreq=self.ModelMachine.RefFreq
        if self.ModelMachine.DicoSMStacked["Type"]!="MSMF":
            raise ValueError("ModelMachine Type should be MSMF")
        self.PSFHasChanged=False
        # reset overall iteration counter
        self.maincache = MainCache
        # reset overall iteration counter
        self._niter = 0
        
        if CleanMaskImage is not None:
            print>>log, "Reading mask image: %s"%CleanMaskImage
            MaskArray=image(CleanMaskImage).getdata()
            nch,npol,nx,ny=MaskArray.shape
            print>>log, "Using mask image %s of shape %dx%d"%(CleanMaskImage,nx,ny)
            # mask array has only one channel, one pol, so take the first plane from the image
            # (and transpose the axes to X,Y from the FITS Y,X)
            self._MaskArray = np.zeros((1,1,ny,nx),np.bool8)
            self._MaskArray[0,0,:,:] = np.bool8(1-MaskArray[0,0].T[::-1])

    def updateMask(self,Mask):
        nx,ny=Mask.shape
        self._MaskArray = np.zeros((1,1,nx,ny),np.bool8)
        self._MaskArray[0,0,:,:]=Mask[:,:]

    def resetCounter(self):
        self._niter = 0

    def updateModelMachine(self,ModelMachine):
        self.ModelMachine=ModelMachine
        if self.ModelMachine.RefFreq!=self.RefFreq:
            raise ValueError("freqs should be equal")

        for iFacet in range(self.PSFServer.NFacets):
            self.DicoMSMachine[iFacet].setModelMachine(self.ModelMachine)

    def GiveModelImage(self,*args): return self.ModelMachine.GiveModelImage(*args)

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe
        

    def SetPSF(self,DicoVariablePSF):
        self.PSFServer=ClassPSFServer(self.GD)
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF,NormalisePSF=True)
        self.PSFServer.setRefFreq(self.ModelMachine.RefFreq)


        #self.DicoPSF=DicoPSF
        self.DicoVariablePSF=DicoVariablePSF
        #self.NChannels=self.DicoDirty["NChannels"]

    def Init(self,**kwargs):
        self.SetPSF(kwargs["PSFVar"])
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
        self.InitMSMF()


    def InitMSMF(self):

        self.DicoMSMachine={}
        if self.maincache is not None:
            cachepath, valid = self.maincache.checkCache("MSMFMachine",dict(
                [(section, self.GD[section]) for section in "VisData", "Beam", "DataSelection",
                 "MultiFreqs", "ImagerGlobal", "Compression",
                 "ImagerCF", "ImagerMainFacet", "MultiScale" ],
                reset=(self.GD["Caching"]["ResetPSF"] or self.PSFHasChanged)
            ))
        else:
            valid=False

        if valid:
            print>>log,"Initialising MSMF Machine from cache %s"%cachepath
            facetcache = cPickle.load(file(cachepath))
        else:
            print>>log,"Initialising MSMF Machine"
            facetcache = {}

        T = ClassTimeIt.ClassTimeIt()
        for iFacet in range(self.PSFServer.NFacets):
            self.PSFServer.setFacet(iFacet)
            T.timeit("PSFServer")
            MSMachine=ClassMultiScaleMachine.ClassMultiScaleMachine(self.GD,self.GainMachine,NFreqBands=self.NFreqBands)
            T.timeit("__init__")
            MSMachine.setModelMachine(self.ModelMachine)
            MSMachine.setSideLobeLevel(self.SideLobeLevel,self.OffsetSideLobe)
            MSMachine.SetFacet(iFacet)
            MSMachine.SetPSF(self.PSFServer)#ThisPSF,ThisMeanPSF)
            T.timeit("set")
            MSMachine.FindPSFExtent(Method="FromSideLobe")
            T.timeit("find")
            cachedscales, cachedmatrix = facetcache.get(iFacet,(None, None))
            T.timeit("get")
            cachedscales = MSMachine.MakeMultiScaleCube(cachedscales)
            T.timeit("make")
            cachedmatrix = MSMachine.MakeBasisMatrix(cachedmatrix)
            T.timeit("make2")
            facetcache[iFacet] = cachedscales, cachedmatrix
            self.DicoMSMachine[iFacet]=MSMachine

        if self.maincache is not None:
            if not valid:
                try:
                    cPickle.dump(facetcache, file(cachepath, 'w'), 2)
                    self.maincache.saveCache("MSMFMachine")
                except:
                    print>>log, traceback.format_exc()
                    print>>log, ModColor.Str("WARNING: MSMF cache could not be written, see error report above. Proceeding anyway.")

    def SetDirty(self,DicoDirty):
        # if len(PSF.shape)==4:
        #     self.PSF=PSF[0,0]
        # else:
        #     self.PSF=PSF

        self.DicoDirty=DicoDirty
        #self.DicoPSF=DicoPSF
        #self.DicoVariablePSF=DicoVariablePSF

        for iFacet in range(self.PSFServer.NFacets):
            MSMachine=self.DicoMSMachine[iFacet]
            MSMachine.SetDirty(DicoDirty)

        #self._PSF=self.MSMachine._PSF
        self._CubeDirty=MSMachine._Dirty
        #self._MeanPSF=self.MSMachine._MeanPSF
        self._MeanDirty = MSMachine._MeanDirty
        NPSF=self.PSFServer.NPSF
        #_,_,NPSF,_=self._PSF.shape
        _,_,NDirty,_=self._CubeDirty.shape

        off=(NPSF-NDirty)/2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)

        if self._ModelImage is None:
            self._ModelImage=np.zeros_like(self._CubeDirty)
        if self._MaskArray is None:
            self._MaskArray=np.zeros(self._MeanDirty.shape,dtype=np.bool8)
        else:
            maskshape = (1,1,NDirty,NDirty)
            # check for mask shape
            if maskshape != self._MaskArray.shape:
                ma0 = self._MaskArray
                _,_,nx,ny = ma0.shape
                def match_shapes (n1,n2):
                    if n1<n2:
                        return slice(None), slice((n2-n1)/2,(n2-n1)/2+n1)
                    elif n1>n2:
                        return slice((n1-n2)/2,(n1-n2)/2+n2), slice(None)
                    else:
                        return slice(None), slice(None)
                sx1, sx2 = match_shapes(NDirty, nx) 
                sy1, sy2 = match_shapes(NDirty, ny) 
                self._MaskArray = np.zeros(maskshape, dtype=np.bool8)
                self._MaskArray[0,0,sx1,sy1] = ma0[0,0,sx2,sy2]
                print>>log,ModColor.Str("WARNING: reshaping mask image from %dx%d to %dx%d"%(nx, ny, NDirty, NDirty))
                print>>log,ModColor.Str("Are you sure you supplied the correct cleaning mask?")


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
        _,npol,_,_ = self._MeanDirty.shape
        x0,x1,y0,y1=self.DirtyExtent

        xc,yc=dx,dy
        #NpixFacet=self.SubPSF.shape[-1]
        #PSF=self.CubePSFScales[iScale]
        N0=self._MeanDirty.shape[-1]
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
        # pylab.imshow(self._CubeDirty[0,0,x0d:x1d,y0d:y1d],interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.subplot(1,3,2)
        # pylab.imshow(LocalSM[0,0,x0p:x1p,y0p:y1p],interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.draw()
        # #print "Fpol02",Fpol
        # # NpParallel.A_add_B_prod_factor((self.Dirty),LocalSM,Aedge,Bedge,factor=float(factor),NCPU=self.NCPU)

        self._CubeDirty[:,:,x0d:x1d,y0d:y1d] -= LocalSM[:,:,x0p:x1p,y0p:y1p]
        if self._MeanDirty is not self._CubeDirty:
            # W=np.float32(self.DicoDirty["WeightChansImages"])
            # self._MeanDirty[0,:,x0d:x1d,y0d:y1d]-=np.sum(LocalSM[:,:,x0p:x1p,y0p:y1p]*W.reshape((W.size,1,1,1)),axis=0)
            self._MeanDirty[0,:,x0d:x1d,y0d:y1d] = self._CubeDirty[:,:,x0d:x1d,y0d:y1d].mean(axis=0)

        # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
        # pylab.imshow(self._CubeDirty[0,0,x0d:x1d,y0d:y1d],interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.draw()
        # pylab.show(False)
        # print Aedge
        # print Bedge
        # print self.Dirty[0,x0d:x1d,y0d:y1d]

    def updateRMS(self):
        _,npol,npix,_ = self._MeanDirty.shape
        NPixStats = self.GD["ImagerDeconv"]["NumRMSSamples"]
        if NPixStats:
            RandomInd=np.int64(np.random.rand(NPixStats)*npix**2)
            RMS=np.std(np.real(self._MeanDirty.ravel()[RandomInd]))
        else:
            RMS=np.std(self._MeanDirty)
        self.RMS=RMS

    def Deconvolve(self, ch=0,UpdateRMS=True):
        """
        Runs minor cycle over image channel 'ch'.
        initMinor is number of minor iteration (keeps continuous count through major iterations)
        Nminor is max number of minor iteration

        Returns tuple of: return_code,continue,updated
        where return_code is a status string;
        continue is True if another cycle should be executed;
        update is True if model has been updated (note that update=False implies continue=False)
        """
        if self._niter >= self.MaxMinorIter:
            return "MaxIter", False, False

        _,npol,npix,_ = self._MeanDirty.shape
        xc=(npix)/2

        # m0,m1=self._CubeDirty.min(),self._CubeDirty.max()
        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.imshow(self.Dirty[0],interpolation="nearest",vmin=m0,vmax=m1)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        DoAbs=int(self.GD["ImagerDeconv"]["SearchMaxAbs"])
        print>>log, "  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter,self.MaxMinorIter,DoAbs)

        if UpdateRMS: self.updateRMS()
        RMS=self.RMS
        self.GainMachine.SetRMS(RMS)
        
        Fluxlimit_RMS = self.RMSFactor*RMS

        x,y,MaxDirty=NpParallel.A_whereMax(self._MeanDirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self._MaskArray)
        # print>>log,"npp: %d %d %g"%(x,y,MaxDirty)
        # xy = ma.argmax(ma.masked_array(abs(self._MeanDirty), self._MaskArray))
        # x1, y1 = xy/npix, xy%npix
        # MaxDirty1 = abs(self._MeanDirty[0,0,x1,y1])
        # print>>log,"argmax: %d %d %g"%(x1,y1,MaxDirty1)

        Fluxlimit_Peak = MaxDirty*self.PeakFactor
        Fluxlimit_Sidelobe = ((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*MaxDirty if self.CycleFactor else 0

        mm0,mm1 = self._MeanDirty.min(),self._MeanDirty.max()

        # work out uper threshold
        StopFlux = max(Fluxlimit_Peak, Fluxlimit_RMS, Fluxlimit_Sidelobe, Fluxlimit_Peak, self.FluxThreshold)

        print>>log, "    Dirty image peak flux      = %10.6g Jy [(min, max) = (%.3g, %.3g) Jy]"%(MaxDirty,mm0,mm1)
        print>>log, "      RMS-based threshold      = %10.6g Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor)
        print>>log, "      Sidelobe-based threshold = %10.6g Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor)
        print>>log, "      Peak-based threshold     = %10.6g Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor)
        print>>log, "      Absolute threshold       = %10.6g Jy"%(self.FluxThreshold)
        print>>log, "    Stopping flux              = %10.6g Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty)

        # MaxModelInit=np.max(np.abs(self.ModelImage))
        # Fact=4
        # self.BookKeepShape=(npix/Fact,npix/Fact)
        # BookKeep=np.zeros(self.BookKeepShape,np.float32)
        # NPixBook,_=self.BookKeepShape
        # FactorBook=float(NPixBook)/npix
        
        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        x,y,ThisFlux=NpParallel.A_whereMax(self._MeanDirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self._MaskArray)
        # #print x,y
        # print>>log, "npp: %d %d %g"%(x,y,ThisFlux)
        # xy = ma.argmax(ma.masked_array(abs(self._MeanDirty), self._MaskArray))
        # x, y = xy/npix, xy%npix
        # ThisFlux = abs(self._MeanDirty[0,0,x,y])
        # print>> log, "argmax: %d %d %g"%(x, y, ThisFlux)

        if ThisFlux < StopFlux:
            print>>log, ModColor.Str("    Initial maximum peak %10.6g Jy below threshold, we're done here" % (ThisFlux),col="green" )
            return "FluxThreshold", False, False

        #self._MaskArray.fill(1)
        #self._MaskArray.fill(0)
        #self._MaskArray[np.abs(self._MeanDirty) > Fluxlimit_Sidelobe]=0

        #        DoneScale=np.zeros((self.MSMachine.NScales,),np.float32)

        PreviousMaxFlux=1e30

        # pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="Cleaning   ", HeaderSize=20,TitleSize=30)
        # pBAR.disable()

        self.GainMachine.SetFluxMax(ThisFlux)
        # pBAR.render(0,"g=%3.3f"%self.GainMachine.GiveGain())

        def GivePercentDone(ThisMaxFlux):
            fracDone=1.-(ThisMaxFlux-StopFlux)/(MaxDirty-StopFlux)
            return max(int(round(100*fracDone)),100)

        try:
            for i in range(self._niter+1,self.MaxMinorIter+1):
                self._niter = i

                #x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=1)
                x,y,ThisFlux=NpParallel.A_whereMax(self._MeanDirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self._MaskArray)

                #x,y=self.PSFServer.SolveOffsetLM(self._MeanDirty[0,0],x,y); ThisFlux=self._MeanDirty[0,0,x,y]
                self.GainMachine.SetFluxMax(ThisFlux)

                # #x,y=1224, 1994
                # print x,y,ThisFlux
                # x,y=np.where(np.abs(self.Dirty[0])==np.max(np.abs(self.Dirty[0])))
                # ThisFlux=self.Dirty[0,x,y]
                # print x,y,ThisFlux
                # stop

                T.timeit("max0")

                if ThisFlux <= StopFlux:
                    # pBAR.render(100,"peak %.3g"%(ThisFlux,))
                    print>>log, ModColor.Str("    [iter=%i] peak of %.3g Jy lower than stopping flux" % (i,ThisFlux),col="green")
                    cont = ThisFlux > self.FluxThreshold
                    if not cont:
                          print>>log, ModColor.Str("    [iter=%i] absolute flux threshold of %.3g Jy has been reached" % (i,self.FluxThreshold),col="green",Bold=True)
                    # DoneScale*=100./np.sum(DoneScale)
                    # for iScale in range(DoneScale.size):
                    #     print>>log,"       [Scale %i] %.1f%%"%(iScale,DoneScale[iScale])
                    
                    return "MinFluxRms", cont, True    # stop deconvolution if hit absolute treshold; update model

    #            if (i>0)&((i%1000)==0):
    #                print>>log, "    [iter=%i] Peak residual flux %f Jy" % (i,ThisFlux)
    #             if (i>0)&((i%100)==0):
    #                 PercentDone=GivePercentDone(ThisFlux)
    #                 pBAR.render(PercentDone,"peak %.3g i%d"%(ThisFlux,self._niter))
                rounded_iter_step = 1 if i<10 else ( 10 if i<200 else ( 100 if i < 2000 else 1000 ))
                # min(int(10**math.floor(math.log10(i))), 1000)
                if i>=10 and i%rounded_iter_step == 0:
                    if self.GD["Debugging"]["PrintMinorCycleRMS"]:
                        print>>log, "    [iter=%i] peak residual %.3g, rms %g" % (i,ThisFlux, self._CubeDirty.std())
                    else:
                        print>> log, "    [iter=%i] peak residual %.3g" % (i, ThisFlux)
                    if ClassMultiScaleMachine.debug_dump_file:
                        ClassMultiScaleMachine.debug_dump_file.flush()

                nch,npol,_,_=self._CubeDirty.shape
                Fpol=np.float32((self._CubeDirty[:,:,x,y].reshape((nch,npol,1,1))).copy())
                #print "Fpol",Fpol
                dx=x-xc
                dy=y-xc

                T.timeit("stuff")

                #iScale=self.MSMachine.FindBestScale((x,y),Fpol)
                
                self.PSFServer.setLocation(x,y)
                PSF = self.PSFServer.GivePSF()
                MSMachine=self.DicoMSMachine[self.PSFServer.iFacet]

                LocalSM=MSMachine.GiveLocalSM((x,y),Fpol)

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
                

                # CurrentGain=self.GainMachine.GiveGain()
                CurrentGain=self.GD["ImagerDeconv"]["Gain"]

                self.SubStep((x,y),LocalSM*CurrentGain)
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
        except KeyboardInterrupt:
            print>>log, ModColor.Str("    [iter=%i] minor cycle interrupted with Ctrl+C, peak flux %.3g" % (self._niter, ThisFlux))
            # DoneScale*=100./np.sum(DoneScale)
            # for iScale in range(DoneScale.size):
            #     print>>log,"       [Scale %i] %.1f%%"%(iScale,DoneScale[iScale])
            return "MaxIter", False, True   # stop deconvolution but do update model


        print>>log, ModColor.Str("    [iter=%i] Reached maximum number of iterations, peak flux %.3g" % (self._niter, ThisFlux))
        # DoneScale*=100./np.sum(DoneScale)
        # for iScale in range(DoneScale.size):
        #     print>>log,"       [Scale %i] %.1f%%"%(iScale,DoneScale[iScale])
        return "MaxIter", False, True   # stop deconvolution but do update model


    def Update(self,DicoDirty,**kwargs):
        """
        Method to update attributes from ClassDeconvMachine
        """
        #Update image dict
        self.SetDirty(DicoDirty)

    def ToFile(self, fname):
        """
        Write model dict to file
        """
        self.ModelMachine.ToFile(fname)

    def FromFile(self, fname):
        """
        Read model dict from file SubtractModel
        """
        self.ModelMachine.FromFile(fname)

    def FromDico(self,DicoName):
        """
        Read in model dict
        """
        self.ModelMachine.FromDico(DicoName)
