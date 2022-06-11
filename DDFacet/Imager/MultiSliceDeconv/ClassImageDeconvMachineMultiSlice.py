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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import os
import numpy as np
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("ClassImageDeconvMachineMultiSlice")
from DDFacet.Array import NpParallel
from DDFacet.Array import NpShared
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
from pyrap.images import image
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Imager import ClassGainMachine
from DDFacet.Other import MyPickle
import multiprocessing
import time
from DDFacet.Array import shared_dict
from DDFacet.ToolsDir import ClassSpectralFunctions
from scipy.optimize import least_squares
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from scipy.signal import fftconvolve
import scipy.stats
MAD=scipy.stats.median_absolute_deviation

class ClassImageDeconvMachine():
    def __init__(self,GD=None,ModelMachine=None,RefFreq=None,*args,**kw):
        self.GD=GD
        self.ModelMachine = ModelMachine
        self.RefFreq=RefFreq
        if self.ModelMachine.DicoModel["Type"]!="MultiSlice":
            raise ValueError("ModelMachine Type should be MultiSlice")
        self.MultiFreqMode=(self.GD["Freq"]["NBand"]>1)
        self.CurrentNegMask=None
        self.FitFluxScale="Linear"
        self.FitFluxScale="Exp"
        self.MaskMachine=None

    def Reset(self):
        pass
        
    def SetPSF(self,DicoVariablePSF):
        self.PSFServer=ClassPSFServer(self.GD)
        DicoVariablePSF=shared_dict.attach(DicoVariablePSF.path)#["CubeVariablePSF"]
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF)
        self.PSFServer.setRefFreq(self.ModelMachine.RefFreq)
        self.DicoVariablePSF=DicoVariablePSF
        self.setFreqs(self.PSFServer.DicoMappingDesc)

    def setMaskMachine(self,MaskMachine):
        self.MaskMachine=MaskMachine

    def setFreqs(self,DicoMappingDesc):
        self.DicoMappingDesc=DicoMappingDesc
        if self.DicoMappingDesc is None: return
        self.SpectralFunctionsMachine=ClassSpectralFunctions.ClassSpectralFunctions(self.DicoMappingDesc,RefFreq=self.DicoMappingDesc["RefFreq"])#,BeamEnable=False)
        self.SpectralFunctionsMachine.CalcFluxBands()

    def GiveModelImage(self,*args): return self.ModelMachine.GiveModelImage(*args)

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

    def FromDico(self, DicoName):
        """
        Read in model dict
        """
        self.ModelMachine.FromDico(DicoName)

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe

    def Init(self,**kwargs):
        self.SetPSF(kwargs["PSFVar"])
        if "PSFSideLobes" not in self.DicoVariablePSF.keys():
            self.DicoVariablePSF["PSFSideLobes"]=kwargs["PSFAve"]
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
        
        self.ModelMachine.setRefFreq(self.RefFreq)
        # store grid and degrid freqs for ease of passing to MSMF
        #print kwargs["GridFreqs"],kwargs["DegridFreqs"]
        self.GridFreqs=kwargs["GridFreqs"]
        self.DegridFreqs=kwargs["DegridFreqs"]
        #self.ModelMachine.setFreqMachine(kwargs["GridFreqs"], kwargs["DegridFreqs"])

    def SetDirty(self,DicoDirty):
        self.DicoDirty=DicoDirty
        self._Dirty=self.DicoDirty["ImageCube"]
        self._MeanDirty=self.DicoDirty["MeanImage"]
        NPSF=self.PSFServer.NPSF
        _,_,NDirty,_=self._Dirty.shape
        off=(NPSF-NDirty)//2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)
        self.ModelMachine.setModelShape(self._Dirty.shape)

    def AdaptArrayShape(self,A,Nout):
        nch,npol,Nin,_=A.shape
        if Nin==Nout: 
            return A
        elif Nin>Nout:
            # dx=Nout//2
            # B=np.zeros((nch,npol,Nout,Nout),A.dtype)
            # print>>log,"  Adapt shapes: %s -> %s"%(str(A.shape),str(B.shape))
            # B[:]=A[...,Nin//2-dx:Nin//2+dx+1,Nin//2-dx:Nin//2+dx+1]

            N0=A.shape[-1]
            xc0=yc0=N0//2
            N1=Nout
            xc1=yc1=N1//2
            Aedge,Bedge=GiveEdges(xc0,yc0,N0,xc1,yc1,N1)
            x0d,x1d,y0d,y1d=Aedge
            x0p,x1p,y0p,y1p=Bedge
            B=A[...,x0d:x1d,y0d:y1d]

            return B
        else:
            return A

    def giveSliceCut(self,A,Nout):
        nch,npol,Nin,_=A.shape
        if Nin==Nout: 
            slice(None)
        elif Nin>Nout:
            N0=A.shape[-1]
            xc0=yc0=N0//2
            if Nout%2==0:
                x0d,x1d=xc0-Nout//2,xc0+Nout//2
            else:
                x0d,x1d=xc0-Nout//2,xc0+Nout//2+1
            return slice(x0d,x1d)
        else:
            return None



    def updateModelMachine(self,ModelMachine):
        self.ModelMachine=ModelMachine
        if self.ModelMachine.RefFreq!=self.RefFreq:
            raise ValueError("freqs should be equal")

    def updateMask(self,Mask):
        nx,ny=Mask.shape
        self._MaskArray = np.zeros((1,1,nx,ny),np.bool8)
        self._MaskArray[0,0,:,:]=Mask[:,:]


    def Deconvolve(self):

        
        if self._Dirty.shape[-1]!=self._Dirty.shape[-2]:
            # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            # print self._Dirty.shape
            # print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            return "Edge", True, True

        log.print("Deconvolve MultiSlice model cube...")

        dirty=self._Dirty
        nch,npol,_,_=dirty.shape
        Model=np.zeros_like(dirty)

        _,_,xp,yp=np.where(self._MeanDirty==np.max(self._MeanDirty))
        xp=xp[0]
        yp=yp[0]
        self.PSFServer.setLocation(xp,yp)
        self.iFacet=self.PSFServer.iFacet

        if self.CurrentNegMask is not None:
            print("  using externally defined Mask (self.CurrentNegMask)", file=log)
            CurrentNegMask=self.CurrentNegMask
        elif self.MaskMachine:
            print("  using MaskMachine Mask", file=log)
            CurrentNegMask=self.MaskMachine.CurrentNegMask
        elif self._MaskArray is not None:
            print("  using externally defined Mask (self._MaskArray)", file=log)
            CurrentNegMask=self._MaskArray
        else:
            print("  not using a mask", file=log)
            CurrentNegMask=None
            
        psf,_=self.PSFServer.GivePSF()
        Nout=np.min([dirty.shape[-1],psf.shape[-1]])
        if Nout%2!=0: Nout-=1

            
        # from skimage import restoration

        if self.GD["MultiSliceDeconv"]["Type"]=="MORESANE":
            s_dirty_cut=self.giveSliceCut(dirty,Nout)
            s_psf_cut=self.giveSliceCut(psf,2*Nout)
            if s_psf_cut is None:
                print(ModColor.Str("Could not adapt psf shape to 2*dirty shape!"), file=log)
                print(ModColor.Str("   shapes are (dirty, psf) = [%s, %s]"%(str(dirty.shape),str(psf.shape))), file=log)
                s_psf_cut=self.giveSliceCut(psf,Nout)
            from .MORESANE.ClassMoresaneSingleSlice import ClassMoresaneSingleSlice
            mask=None
            if CurrentNegMask is not None:
                mask=(1-CurrentNegMask)[0,0,s_dirty_cut,s_dirty_cut]

            # ch
            for ch in range(nch):
                log.print("Deconvolve slice #%i"%ch)
                A,B=dirty[ch,0,s_dirty_cut,s_dirty_cut].copy(), psf[ch,0,s_psf_cut,s_psf_cut].copy()
                CM=ClassMoresaneSingleSlice(A,B,
                                            #dirty[ch,0,s_dirty_cut,s_dirty_cut],
                                            #psf[ch,0,s_psf_cut,s_psf_cut],
                                            mask=mask,
                                            GD=None)
                model,resid=CM.giveModelResid(major_loop_miter=self.GD["MORESANE"]["NMajorIter"],
                                              minor_loop_miter=self.GD["MORESANE"]["NMinorIter"],
                                              loop_gain=self.GD["MORESANE"]["Gain"],
                                              sigma_level=self.GD["MORESANE"]["SigmaCutLevel"],# tolerance=1.,
                                              enforce_positivity=self.GD["MORESANE"]["ForcePositive"])
                Model[ch,0,s_dirty_cut,s_dirty_cut]=model[:,:]
                if CurrentNegMask is not None:
                    indx,indy=np.where(CurrentNegMask[0,0]==1)
                    nx,ny=Model[ch,0].shape
                    Model[ch,0].flat[indx*ny+indy]=0
                    
        elif self.GD["MultiSliceDeconv"]["Type"]=="Orieux":
            s_dirty_cut=self.giveSliceCut(dirty,Nout)
            s_psf_cut=self.giveSliceCut(psf,Nout)
            from .Orieux import ClassOrieux
            for ch in range(nch):
                A,B=dirty[ch,0,s_dirty_cut,s_dirty_cut].copy(), psf[ch,0,s_psf_cut,s_psf_cut].copy()
                #A,B=psf[ch,0,s_psf_cut,s_psf_cut].copy(), psf[ch,0,s_psf_cut,s_psf_cut].copy()
                CO=ClassOrieux.ClassOrieux(A,B)
                #CO=ClassOrieux.ClassOrieux(dirty[ch,0,:,:], psf[ch,0,:,:])
                model=CO.Deconv()
                # model = restoration.richardson_lucy(dirty[ch,0,s_dirty_cut,s_dirty_cut], psf[ch,0,s_psf_cut,s_psf_cut], iterations=30)

                Model[ch,0,s_dirty_cut,s_dirty_cut]=model[:,:]
                if CurrentNegMask is not None:
                    indx,indy=np.where(CurrentNegMask[0,0]==1)
                    nx,ny=Model[ch,0].shape
                    Model[ch,0].flat[indx*ny+indy]=0
            
        elif self.GD["MultiSliceDeconv"]["Type"]=="Sara":
            import pfb.deconv.sara as sara
            s_dirty_cut=self.giveSliceCut(dirty,Nout)
            s_psf_cut=self.giveSliceCut(psf,2*Nout)
            s_residual=dirty[:,0,s_dirty_cut,s_dirty_cut].copy()
            s_psf=psf[ch,0,s_psf_cut,s_psf_cut].copy()
            s_model=np.zeros_like(s_residual)

            NBands=dirty.shape[0]
            sig_21=MAD(s_residual)*NBands
            
            model, dual, residual_mfs, weights21=sara(s_psf, s_model, s_residual, sig_21, 0.5)

            for ch in range(nch):
                Model[ch,0,s_dirty_cut,s_dirty_cut]=model[ich,:,:]
                if CurrentNegMask is not None:
                    indx,indy=np.where(CurrentNegMask[0,0]==1)
                    nx,ny=Model[ch,0].shape
                    Model[ch,0].flat[indx*ny+indy]=0
            

                    
            #print(Model[ch,0].max(),Model[ch,0],np.count_nonzero(Model[ch,0]!=0))



            #N0=model.shape[0]//2
            
            # Dty=fftconvolve(Model[ch,0,s_dirty_cut,s_dirty_cut],B, mode='same')#[s_dirty_cut,s_dirty_cut]

            #import scipy.signal
            #Dty=scipy.signal.convolve2d(model,B, mode='same')
            
            # fact=np.max(dirty[ch,0,s_dirty_cut,s_dirty_cut])/np.max(Dty)
            # log.print("Peak-based correction factor: %f"%fact)
            # Model[ch,0,s_dirty_cut,s_dirty_cut]*=fact

            # # print(dirty[ch,0,s_dirty_cut,s_dirty_cut].max(), psf[ch,0,s_psf_cut,s_psf_cut].max(),np.sum(model))
            
            # import pylab
            # pylab.clf(),s_dirty_cut,s_dirty_cut
            # ax=pylab.subplot(2,2,1)
            # pylab.imshow(A,interpolation="nearest")
            # #pylab.imshow(CurrentNegMask[0,0,s_dirty_cut,s_dirty_cut],interpolation="nearest")
            # pylab.colorbar()
            # pylab.title("Dirty")
            # pylab.subplot(2,2,2,sharex=ax,sharey=ax)
            # pylab.imshow(B,interpolation="nearest")
            # pylab.colorbar()
            # pylab.title("PSF")
            # pylab.subplot(2,2,3,sharex=ax,sharey=ax)
            # pylab.imshow((Model[ch,0,s_dirty_cut,s_dirty_cut]),interpolation="nearest")#,vmin=-0.1,vmax=0.1)
            # pylab.colorbar()
            # pylab.title("Model")
            # pylab.subplot(2,2,4,sharex=ax,sharey=ax)
            # pylab.imshow(A-Dty,interpolation="nearest")
            # pylab.colorbar()
            # pylab.title("Dirty-Model*PSF")
            # # pylab.subplot(2,2,4)
            # # pylab.imshow(resid,interpolation="nearest")
            # # pylab.colorbar()
            # pylab.draw()
            # pylab.show()#block=False)
            # pylab.pause(0.1)

            
        # print 
        # print np.max(np.max(Model,axis=-1),axis=-1)
        # print 
        # print 




        #_,_,nx,ny=Model.shape
        #Model=np.mean(Model,axis=0).reshape((1,1,nx,ny))

        #Model.fill(0)
        #Model[:,:,xp,yp]=self._Dirty[:,:,xp,yp]

        # if CurrentNegMask is not None:
        #     indx,indy=np.where(CurrentNegMask[0,0]==1)
        #     for ch in range(nch):
        #         Model[ch,0,indx,indy]=0

        for ich in range(nch):
            absModel=np.abs(Model[ich])
            #_,indx,indy=np.where(np.any(absModel<absModel.max()/1e4,axis=0))
            _,indx,indy=np.where(absModel<absModel.max()/1e4)
            Model[ich,0,indx,indy]=0
        
        CoefImage=self.DoSpectralFit(Model)
        
        self.ModelMachine.setModel(CoefImage,FluxScale=self.FitFluxScale)

        
        return "MaxIter", True, True   # stop deconvolution but do update model

    def DoSpectralFit(self,Model):
        log.print("Fitting MultiSlice model cube...")
        NOrder=self.GD["MultiSliceDeconv"]["PolyFitOrder"]
        if NOrder==0:
            NOrder=self.GD["Freq"]["NBand"]
        
        def GiveResid(X,F,iFacet):
            R=np.zeros_like(F)
            Fit=np.zeros_like(F)
            for iBand in range(R.size):
                Fit[iBand]=self.SpectralFunctionsMachine.IntExpFuncPoly(X.reshape((1,NOrder)),
                                                                        iChannel=iBand,
                                                                        iFacet=iFacet,
                                                                        FluxScale=self.FitFluxScale)
            #Fit[Fit<0]*=100.
            
            R=F-Fit
            Lambda=np.max([np.abs(X[1]-(-0.7))/1,1.])
            R*= Lambda#np.abs(X[1])

            # print("aa",R)
            return R
        
        nx,ny=Model.shape[-2],Model.shape[-1]
        CoefImage=np.zeros((NOrder,1,nx,ny),np.float32)

        indx,indy=np.where((Model[:,0,:,:]).any(axis=0))
        iDone=0
        for iPix,jPix in zip(indx.tolist(),indy.tolist()):
            #log.print("%i/%i:[%i, %i]"%(iDone,indx.size,iPix,jPix))
            iDone+=1
            self.PSFServer.setLocation(iPix,jPix)
            iFacet=self.PSFServer.iFacet
            
            F=np.float64(Model[:,0,iPix,jPix])
            # print(iPix,jPix,iFacet,F)
            
            #JonesNorm=(self.DicoDirty["JonesNorm"][:,:,iPix,jPix]).reshape((-1,1,1,1))
            #W=self.DicoDirty["WeightChansImages"]
            #JonesNorm=np.sum(JonesNorm*W.reshape((-1,1,1,1)),axis=0).reshape((1,1,1,1))
            
            #F=F/np.sqrt(JonesNorm).ravel()
            F0=np.mean(F)
            if F0==0:
                continue
            x0=np.zeros((NOrder,),np.float64)
            x0[0]=F0
            if NOrder>1:
                x0[1]=0.
            x0=np.float64(np.random.randn(NOrder))
            R0=GiveResid(x0,F,iFacet)
            X=least_squares(GiveResid, x0, args=(F,iFacet))#,ftol=1e-2)#,gtol=1e-1,xtol=1e-1)
            R1=GiveResid(X['x'],F,iFacet)
            x=X['x']


            
            Fit=np.zeros_like(F)
            for iBand in range(F.size):
                Fit[iBand]=self.SpectralFunctionsMachine.IntExpFuncPoly(x.reshape((1,NOrder)),
                                                                        iChannel=iBand,
                                                                        iFacet=iFacet,
                                                                        FluxScale=self.FitFluxScale)

            Chi2Ratio=np.sum((F-Fit)**2)/np.sum(F**2)
            Chi2Cut=0.3

            if Chi2Ratio>Chi2Cut:
                x.fill(0)
                
            # if Chi2Ratio<0.2:
            #     #Fit[Fit<0]*=100.
            #     import pylab
            #     pylab.clf()
            #     pylab.plot(F)
            #     pylab.plot(Fit)
            #     pylab.title(Chi2Ratio)
            #     pylab.draw()
            #     pylab.show(block=False)
            #     pylab.pause(0.1)
            
            #print(F,x)
            #print("%0.5f"%(x[1]))
            #stop
            CoefImage[:,0,iPix,jPix]=x[:]
            
        log.print("   done...")

        # S=CoefImage[0,0].flat[:]
        # A=CoefImage[1,0].flat[:]
        # ind=np.where(S!=0)[0]
        # import pylab
        # pylab.clf()
        # pylab.scatter(np.log10(np.abs(S[ind])),A[ind])
        # pylab.draw()
        # pylab.show()
        
        CoefImage[:,:,:,:]*=self.GD["Deconv"]["Gain"]
        return CoefImage
    

    # def DoSpectralFit(self,Model):

    #     def GiveResid(X,F,iFacet):
    #         R=np.zeros_like(F)
    #         S0,Alpha=X
    #         for iBand in range(R.size):
    #             #print iBand,self.SpectralFunctionsMachine.IntExpFunc(Alpha=np.array([0.]),iChannel=iBand,iFacet=iFacet)
    #             R[iBand]=F[iBand]-S0*self.SpectralFunctionsMachine.IntExpFunc(Alpha=np.array([Alpha]).ravel(),iChannel=iBand,iFacet=iFacet)

    #         #stop
    #         return R

        
    #     nx,ny=Model.shape[-2],Model.shape[-1]
    #     S=np.zeros((1,1,nx,ny),np.float32)
    #     Al=np.zeros((1,1,nx,ny),np.float32)

    #     for iPix in range(Model.shape[-2]):
    #         for jPix in range(Model.shape[-1]):
    #             F=Model[:,0,iPix,jPix]

    #             JonesNorm=(self.DicoDirty["JonesNorm"][:,:,iPix,jPix]).reshape((-1,1,1,1))
    #             #W=self.DicoDirty["WeightChansImages"]
    #             #JonesNorm=np.sum(JonesNorm*W.reshape((-1,1,1,1)),axis=0).reshape((1,1,1,1))
                
    #             #F=F/np.sqrt(JonesNorm).ravel()
    #             F0=np.mean(F)
    #             if F0==0:
    #                 continue

    #             x0=(F0,-0.8)
                
    #             #print self.iFacet,iPix,jPix,F,F0
    #             X=least_squares(GiveResid, x0, args=(F,self.iFacet),ftol=1e-3,gtol=1e-3,xtol=1e-3)
    #             x=X['x']
    #             S[0,0,iPix,jPix]=x[0]
    #             Al[0,0,iPix,jPix]=x[1]

    #     return S,Al
    
