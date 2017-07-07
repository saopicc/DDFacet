'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

import numpy as np
import MyLogger
import ModColor
log=MyLogger.getLogger(" ClassImageDeconvMachine")
import NpParallel
import ModFFTW
import ModToolBox




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
        self.CubePSFScales=None
        self.SubPSF=None

    def setMaskMachine(self,MaskMachine):
        self.MaskMachine=MaskMachine

    def SetDirtyPSF(self,Dirty,PSF):
        # if len(PSF.shape)==4:
        #     self.PSF=PSF[0,0]
        # else:
        #     self.PSF=PSF
        self._PSF=PSF
        self._Dirty=Dirty
        _,_,NPSF,_=PSF.shape
        _,_,NDirty,_=Dirty.shape
        off=(NPSF-NDirty)/2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)
        


        if self.ModelImage is None:
            self._ModelImage=np.zeros_like(self._Dirty)
        if self.MaskArray is None:
            self._MaskArray=np.zeros(self._Dirty.shape,dtype=np.bool8)



    def FindPSFExtent(self,Method="FromBox"):
        if self.SubPSF is not None: return
        PSF=self._PSF
        _,_,NPSF,_=PSF.shape
        xtest=np.int64(np.linspace(NPSF/2,NPSF,100))
        box=100
        itest=0

        if Method=="FromBox":
            while True:
                X=xtest[itest]
                psf=PSF[0,0,X-box:X+box,NPSF/2-box:NPSF/2+box]
                std0=np.abs(psf.min()-psf.max())#np.std(psf)
                psf=PSF[0,0,NPSF/2-box:NPSF/2+box,X-box:X+box]
                std1=np.abs(psf.min()-psf.max())#np.std(psf)
                std=np.max([std0,std1])
                if std<1e-2:
                    break
                else:
                    itest+=1
            x0=xtest[itest]
            dx0=(x0-NPSF/2)
            print>>log, "PSF extends to [%i] from center, with rms=%.5f"%(dx0,std)
        elif Method=="FromSideLobe":
            dx0=2*self.OffsetSideLobe
            print>>log, "PSF extends to [%i] from center"%(dx0)
        
        npix=2*dx0+1
        npix=ModToolBox.GiveClosestFastSize(npix,Odd=True)

        self.PSFMargin=(NPSF-npix)/2

        dx=npix/2
        self.PSFExtent=(NPSF/2-dx,NPSF/2+dx+1,NPSF/2-dx,NPSF/2+dx+1)
        x0,x1,y0,y1=self.PSFExtent
        self.SubPSF=self._PSF[:,:,x0:x1,y0:y1]



    def MakeMultiScaleCube(self):
        if self.CubePSFScales is not None: return
        print>>log, "Making MultiScale PSFs..."
        LScales=self.GD["HMP"]["Scales"]
        if 0 in LScales: LScales.remove(0)
        LRatios=self.GD["HMP"]["Ratios"]
        NTheta=self.GD["HMP"]["NTheta"]

        
        _,_,nx,ny=self.SubPSF.shape
        NScales=len(LScales)
        NRatios=len(LRatios)
        CubePSFScales=np.zeros((NScales+1+NRatios*NTheta*(NScales),nx,ny))

        Scales=np.array(LScales)
        Ratios=np.array(LRatios)


        self.ListScales=[]
        CubePSFScales[0,:,:]=self.SubPSF[0,0,:,:]
        self.ListScales.append({"ModelType":"Delta"})
        iSlice=1
        
        Support=61

        for i in range(NScales):
            Minor=Scales[i]/(2.*np.sqrt(2.*np.log(2.)))
            Major=Minor
            PSFGaussPars=(Major,Minor,0.)
            CubePSFScales[iSlice,:,:]=ModFFTW.ConvolveGaussian(self.SubPSF,CellSizeRad=1.,GaussPars=[PSFGaussPars])[0,0]
            Gauss=ModFFTW.GiveGauss(Support,CellSizeRad=1.,GaussPars=PSFGaussPars)
            self.ListScales.append({"ModelType":"Gaussian",
                                    "Model":Gauss, "ModelParams":PSFGaussPars,
                                    "Scale":i})

            iSlice+=1

        Theta=np.arange(0.,np.pi-1e-3,np.pi/NTheta)


        
        for iScale in range(NScales):
            for ratio in Ratios:
                for th in Theta:
                    Minor=Scales[iScale]/(2.*np.sqrt(2.*np.log(2.)))
                    Major=Minor*ratio
                    PSFGaussPars=(Major,Minor,th)
                    CubePSFScales[iSlice,:,:]=ModFFTW.ConvolveGaussian(self.SubPSF,CellSizeRad=1.,GaussPars=[PSFGaussPars])[0,0]
                    Max=np.max(CubePSFScales[iSlice,:,:])
                    CubePSFScales[iSlice,:,:]/=Max
                    # pylab.clf()
                    # pylab.subplot(1,2,1)
                    # pylab.imshow(CubePSFScales[0,:,:],interpolation="nearest")
                    # pylab.subplot(1,2,2)
                    # pylab.imshow(CubePSFScales[iSlice,:,:],interpolation="nearest")
                    # pylab.title("Scale = %s"%str(PSFGaussPars))
                    # pylab.draw()
                    # pylab.show(False)
                    # pylab.pause(0.1)
                    iSlice+=1
                    Gauss=ModFFTW.GiveGauss(Support,CellSizeRad=1.,GaussPars=PSFGaussPars)/Max
                    self.ListScales.append({"ModelType":"Gaussian",
                                            "Model":Gauss, "ModelParams":PSFGaussPars,
                                            "Scale":iScale})

        # Max=np.max(np.max(CubePSFScales,axis=1),axis=1)
        # Max=Max.reshape((Max.size,1,1))
        # CubePSFScales=CubePSFScales/Max

        self.CubePSFScales=np.float32(CubePSFScales)
        self.WeightWidth=6
        CellSizeRad=1.
        PSFGaussPars=(self.WeightWidth,self.WeightWidth,0.)
        self.WeightFunction=ModFFTW.GiveGauss(self.SubPSF.shape[-1],CellSizeRad=1.,GaussPars=PSFGaussPars)
        #self.WeightFunction.fill(1)
        self.SupWeightWidth=3.*self.WeightWidth
        print>>log, "   ... Done"


    def FindBestScale(self,(x,y),Fpol):
        x0,y0=x,y
        x,y=x0,y0
        

        N0=self.Dirty.shape[-1]
        N1=self.SubPSF.shape[-1]
        xc,yc=x,y

        nxPSF=self.CubePSFScales.shape[-1]
        x0,x1=nxPSF/2-self.SupWeightWidth,nxPSF/2+self.SupWeightWidth+1
        y0,y1=nxPSF/2-self.SupWeightWidth,nxPSF/2+self.SupWeightWidth+1
        CubePSF=self.CubePSFScales[:,x0:x1,y0:y1]
        N1=CubePSF.shape[-1]
        
        
        
        Aedge,Bedge=self.GiveEdges((xc,yc),N0,(N1/2,N1/2),N1)
        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge
        #print Aedge
        #print Bedge

        #CubePSF=self.CubePSFScales[:,x0p:x1p,y0p:y1p]*Fpol[0,0,0]
        CubePSF=CubePSF[:,x0p:x1p,y0p:y1p]*Fpol[0,0,0]

        dirty=self.Dirty[0,x0d:x1d,y0d:y1d]
        nx,ny=dirty.shape
        dirty=dirty.reshape((1,nx,ny))


        NSlice,nxPSF,_=self.CubePSFScales.shape
        
        WCubePSF=self.WeightFunction[x0:x1,y0:y1][x0p:x1p,y0p:y1p]
        resid=dirty-CubePSF
        WResid=WCubePSF*(dirty-CubePSF)
        resid2=(1./self.RMS**2)*WCubePSF*(resid)**2
        #resid2=(resid)**2

        chi2=np.sum(np.sum(resid2,axis=1),axis=1)/(np.sum(self.WeightFunction))

        iScale=np.argmin(chi2)

        # pylab.clf()
        # vmin=dirty.min()
        # vmax=dirty.max()
        # ax=pylab.subplot(1,3,1)
        # pylab.imshow(dirty[0],vmin=vmin,vmax=vmax,interpolation="nearest")
        # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
        # pylab.imshow(CubePSF[iScale],vmin=vmin,vmax=vmax,interpolation="nearest")
        # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
        # pylab.imshow(WResid[iScale],vmin=vmin,vmax=vmax,interpolation="nearest")
        # pylab.colorbar()
        # pylab.draw()
        # pylab.show(False)

        if np.min(chi2)>self.Chi2Thr:
            self._MaskArray[:,:,x,y]=True
            return "BadFit"






        return iScale

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


    def SubStep(self,(dx,dy),Fpol,iScale=0):
        npol,_,_=self.Dirty.shape
        x0,x1,y0,y1=self.DirtyExtent

        xc,yc=dx,dy
        #NpixFacet=self.SubPSF.shape[-1]
        PSF=self.CubePSFScales[iScale]
        N0=self.Dirty.shape[-1]
        N1=PSF.shape[-1]

        # PSF=PSF[N1/2-1:N1/2+2,N1/2-1:N1/2+2]
        # N1=PSF.shape[-1]

        Aedge,Bedge=self.GiveEdges((xc,yc),N0,(N1/2,N1/2),N1)

        


        #_,n,n=self.PSF.shape
        #PSF=self.PSF.reshape((n,n))
        #print "Fpol00",Fpol
        factor=-Fpol[0,0,0]*self.Gain
        #print "Fpol01",Fpol

        nx,ny=PSF.shape
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
        # pylab.imshow(PSF[x0p:x1p,y0p:y1p]*factor,interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.draw()
        #print "Fpol02",Fpol
        NpParallel.A_add_B_prod_factor((self.Dirty),PSF,Aedge,Bedge,factor=float(factor),NCPU=self.NCPU)
        #print "Fpol03",Fpol
        # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
        # pylab.imshow(self.Dirty[0,x0d:x1d,y0d:y1d],interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.draw()
        # pylab.show(False)
        # print Aedge
        # print Bedge
        # print self.Dirty[0,x0d:x1d,y0d:y1d]
        # stop
        
        


    def setChannel(self,ch=0):
        self.PSF=self._PSF[ch]
        self.Dirty=self._Dirty[ch]
        self.ModelImage=self._ModelImage[ch]
        self.MaskArray=self._MaskArray[ch]

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe

    def Deconvolve(self, Nminor=None, ch=0):
        if Nminor is None:
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
        Threshold_SideLobe=self.CycleFactor*MaxDirty*(self.SideLobeLevel)

        mm0,mm1=self.Dirty.min(),self.Dirty.max()
        print>>log, "    Dirty image peak flux   = %7.3f Jy [(min, max) = (%7.3f, %7.3f) Jy]"%(MaxDirty,mm0,mm1)
        print>>log, "    RMS threshold flux      = %7.3f Jy [rms      = %7.3f Jy]"%(FluxLimit, RMS)
        print>>log, "    Sidelobe threshold flux = %7.3f Jy [sidelobe = %7.3f of peak]"%(Threshold_SideLobe,self.SideLobeLevel)

        MaxModelInit=np.max(np.abs(self.ModelImage))

        
        # Fact=4
        # self.BookKeepShape=(npix/Fact,npix/Fact)
        # BookKeep=np.zeros(self.BookKeepShape,np.float32)
        # NPixBook,_=self.BookKeepShape
        # FactorBook=float(NPixBook)/npix
        
        import ClassTimeIt
        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=1)
        #print x,y

        if ThisFlux < FluxLimit:
            print>>log, ModColor.Str("    Initial maximum peak %f Jy lower that rms-based limit of %f Jy (%i-sigma)" % (ThisFlux,Threshold_RMS,Threshold_RMS))
            return "DoneMinFlux"


        for i in range(Nminor):

            #x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=1)
            x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=1,Mask=self.MaskArray)

            #x,y=1224, 1994
            # print x,y,ThisFlux
            # x,y=np.where(self.Dirty[0]==np.max(np.abs(self.Dirty[0])))
            # ThisFlux=self.Dirty[0,x,y]
            # print x,y,ThisFlux
            # stop

            T.timeit("max0")

            if ThisFlux < FluxLimit:
                print>>log, "    [iter=%i] Maximum peak lower that rms-based limit of %f Jy (%i-sigma)" % (i,FluxLimit,Threshold_RMS)
                return "MinFlux"

            if ThisFlux < Threshold_SideLobe:
                print>>log, "    [iter=%i] Peak residual flux %f Jy higher than sidelobe-based limit of %f Jy" % (i,ThisFlux, Threshold_SideLobe)

                return "MinFlux"

            Fpol=(self.Dirty[:,x,y].reshape(npol,1,1)).copy()

            #print "Fpol",Fpol
            dx=x-xc
            dy=y-xc

            T.timeit("stuff")

            iScale=self.FindBestScale((x,y),np.float32(Fpol))
            #print iScale
            if iScale=="BadFit": continue

            # box=30
            # x0,x1=x-box,x+box
            # y0,y1=y-box,y+box
            # pylab.clf()
            # pylab.subplot(1,3,1)
            # pylab.imshow(self.Dirty[0][x0:x1,y0:y1],interpolation="nearest")#,vmin=m0,vmax=m1)
            # #pylab.subplot(1,3,2)
            # #pylab.imshow(self.MaskArray[0],interpolation="nearest",vmin=0,vmax=1,cmap="gray")
            # pylab.subplot(1,3,2)
            # pylab.imshow(self.ModelImage[0][x0:x1,y0:y1],interpolation="nearest",cmap="gray")
            # #pylab.imshow(PSF[0],interpolation="nearest",vmin=0,vmax=1)
            # #pylab.colorbar()
            

            
            self.SubStep((x,y),Fpol,iScale)
            T.timeit("add0")



            # pylab.subplot(1,3,3)
            # pylab.imshow(self.Dirty[0][x0:x1,y0:y1],interpolation="nearest")#,vmin=m0,vmax=m1)

            # #pylab.imshow(PSF[0],interpolation="nearest",vmin=0,vmax=1)
            # #pylab.colorbar()
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)






            ThisComp=self.ListScales[iScale]




            if ThisComp["ModelType"]=="Delta":
                for pol in range(npol):
                   self.ModelImage[pol,x,y]+=Fpol[pol,0,0]*self.Gain
                
            elif ThisComp["ModelType"]=="Gaussian":
                Gauss=ThisComp["Model"]
                Sup,_=Gauss.shape
                x0,x1=x-Sup/2,x+Sup/2+1
                y0,y1=y-Sup/2,y+Sup/2+1

                _,N0,_=self.ModelImage.shape
                
                Aedge,Bedge=self.GiveEdges((x,y),N0,(Sup/2,Sup/2),Sup)
                x0d,x1d,y0d,y1d=Aedge
                x0p,x1p,y0p,y1p=Bedge
                

                for pol in range(npol):
                    self.ModelImage[pol,x0d:x1d,y0d:y1d]+=Gauss[x0p:x1p,y0p:y1p]*Fpol[pol,0,0]*self.Gain

            else:
                stop




            T.timeit("add1")



        print>>log, ModColor.Str("    [iter=%i] Reached maximum number of iterations" % (Nminor))
        return "MaxIter"

