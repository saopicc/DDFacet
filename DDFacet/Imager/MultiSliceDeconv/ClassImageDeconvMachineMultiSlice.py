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
MAD=scipy.stats.median_abs_deviation
from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric


def pad_to_square(A):
    """Pad a 2D array to make it square."""
    max_dim = max(A.shape)
    pad_width = [(0, max_dim - A.shape[0]), (0, max_dim - A.shape[1])]
    Ap=np.pad(A, pad_width, mode='constant', constant_values=0)
    nx0,ny0=A.shape
    nx1,ny1=Ap.shape
    i0=0#(nx1-nx0)//2
    i1=nx0#i0+nx0
    j0=0#(ny1-ny0)//2
    j1=ny0#j0+ny0

    # i0=(nx1-nx0)//2
    # i1=i0+nx0
    # j0=(ny1-ny0)//2
    # j1=j0+ny0

    
    blc_trc=i0,i1,j0,j1
    return Ap, blc_trc


def unpad_to_original(square_array, blc_trc):
    """Crop a square array back to its original rectangular shape."""
    i0,i1,j0,j1=blc_trc
    return square_array[i0:i1,j0:j1]




class ClassImageDeconvMachine():
    def __init__(self,GD=None,ModelMachine=None,RefFreq=None,CacheFileName=None,*args,**kw):
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
        self.CacheFileName=CacheFileName
        

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
        self.DicoVariablePSF.reload()
        self.DicoFreqBandToTaylor=shared_dict.attach("DicoFreqBandToTaylor")#["CubeVariablePSF"]
        if "FreqBandToTaylor_FluxVec" not in list(self.DicoFreqBandToTaylor.keys()):
            #print("FKJKJDFKJKDJKGDJFKDG")
            self.cache_FreqBandToTaylor()
        #else:
            #print("ISSSS FKJKJDFKJKDJKGDJFKDG")
            

    def cache_FreqBandToTaylor(self):
        NOrder=self.GD["MultiSliceDeconv"]["PolyFitOrder"]
        if NOrder==0:
            NOrder=self.GD["Freq"]["NBand"]
        
        NModel=1
        Lx0x1=[]
        for iOrder in range(0,NOrder):
            if iOrder==0:
                # Lal=[0.5,1,2]
                # Ldal=[0.1,0.1]
                # LL=[]
                # for ii,dal in enumerate(Ldal):
                #     al0,al1=Lal[ii],Lal[ii+1]
                #     LL.append(np.arange(al0,al1,dal))
                # ThisParm=np.concatenate(LL).flatten()

                LL=(10**np.linspace(np.log10(0.1),np.log10(100),30)).tolist()
                #LL=(10**np.linspace(np.log10(0.1),np.log10(300),10)).tolist()
                LL.append(1.)
                LL=sorted(list(set(LL)))
                ThisParm=np.array(LL).flatten()
                log.print("   [S0]     : %s"%str(ThisParm.tolist()))
                Lx0x1.append(ThisParm)
                NModel*=ThisParm.size
            elif iOrder==1:
                #Lal=[-20,-10,-3,3,10,20]
                #Ldal=[2,1,0.1,1,2]
                #Lal=[-10,-2,2,10]
                #Ldal=[1,0.1,1]
                Lal=[-20, -10 , -7,  -5,   -1   , 1,    2,   5]
                Ldal=  [  2 , 1,  0.2,  0.1, 0.05,  0.1,  1]
                LL=[]
                for ii,dal in enumerate(Ldal):
                    al0,al1=Lal[ii],Lal[ii+1]
                    LL.append(np.arange(al0,al1,dal))
                
                #ThisParm=np.linspace(-2,2,11)
                ThisParm=np.concatenate(LL).flatten()
                log.print("   [Alpha]  : %s"%str(ThisParm.tolist()))
                Lx0x1.append(ThisParm)
                NModel*=ThisParm.size
            else:
                ThisParm=np.linspace(-1,1,5)
                #ThisParm=np.array([0.])
                log.print("   [Term%i] : %s"%(iOrder,str(ThisParm.tolist())))
                Lx0x1.append(ThisParm)
                NModel*=ThisParm.size
            # elif iOrder==2:
            #     ThisParm=np.linspace(-1,1,11)
            #     Lx0x1.append(ThisParm)
            #     NModel*=ThisParm.size
        ParmGrid=np.meshgrid(*Lx0x1)
        NComb=ParmGrid[0].size

        #ParmVec=np.array(ParmGrid).reshape((NComb,NOrder))
        NBand=self.GD["Freq"]["NBand"]
        
        ParmVec=np.zeros((NComb,NOrder),np.float32)
        ParmVec[:,:]=np.array(ParmGrid).reshape((NOrder,NComb)).T
        
        # ParmVec=np.zeros((NComb,NOrder),np.float32)
        # ParmVec[:,0]=1
        # ParmVec[:,1:]=np.array(ParmGrid).reshape((NComb,NOrder-1))


        NFacets=self.PSFServer.NFacets
        FluxVec=np.zeros((NFacets,NComb,NBand),np.float32)
        
        for iFacet in range(NFacets):
            #print(iFacet,NFacets)
            for iBand in range(NBand):
                for iComb in range(NComb):
                    X=ParmVec[iComb]
                    F=self.SpectralFunctionsMachine.IntExpFuncPoly(X.reshape((1,NOrder)),
                                                                   iChannel=iBand,
                                                                   iFacet=iFacet,
                                                                   FluxScale=self.FitFluxScale)
                    FluxVec[iFacet,iComb,iBand]=F[0]
        #MeanFluxVec=np.median(FluxVec,axis=-1).reshape((NFacets,NComb,1))
        #FluxVec/=MeanFluxVec
        self.DicoFreqBandToTaylor["FreqBandToTaylor_FluxVec"]=FluxVec
        self.DicoFreqBandToTaylor["FreqBandToTaylor_ParmVec"]=ParmVec

        # AA=np.zeros()
        # def GiveResid(X,F,iFacet):
        #     R=np.zeros_like(F)
        #     Fit=np.zeros_like(F)
        #     for iBand in range(R.size):
        #         Fit[iBand]=self.SpectralFunctionsMachine.IntExpFuncPoly(X.reshape((1,NOrder)),
        #                                                                 iChannel=iBand,
        #                                                                 iFacet=iFacet,
        #                                                                 FluxScale=self.FitFluxScale)
        
        
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
        NPSF_x,NPSF_y=self.PSFServer.NPSF
        _,_,NDirty_x,NDirty_y=self._Dirty.shape
        off_x=(NPSF_x-NDirty_x)//2
        off_y=(NPSF_y-NDirty_y)//2
        self.DirtyExtent=(off_x,off_x+NDirty_x,off_y,off_y+NDirty_y)
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

            N0x,N0y=A.shape[-2:]
            xc0=N0x//2
            yc0=N0y//2
            N1x,N1y=Nout,Nout
            xc1=N1x//2
            yc1=N1y//2
            
            Aedge,Bedge=GiveEdgesDissymetric(xc0,yc0,N0x,N0y,xc1,yc1,N1x,N1y)
            x0d,x1d,y0d,y1d=Aedge
            x0p,x1p,y0p,y1p=Bedge
            B=A[...,x0d:x1d,y0d:y1d]

            return B
        else:
            return A

    def giveSliceCut(self,A,Nout):
        nch,npol,Nin,_=A.shape
        if Nin==Nout: 
            return slice(None)
        elif Nin>Nout:
            N0=A.shape[-1]
            xc0=yc0=N0//2
            if Nout%2==0:
                x0d,x1d=xc0-Nout//2,xc0+Nout//2
            else:
                x0d,x1d=xc0-Nout//2,xc0+Nout//2+1
            return slice(x0d,x1d)
        else:
            return slice(None)



    def updateModelMachine(self,ModelMachine):
        self.ModelMachine=ModelMachine
        if self.ModelMachine.RefFreq!=self.RefFreq:
            raise ValueError("freqs should be equal")

    def updateMask(self,Mask):
        nx,ny=Mask.shape
        self._MaskArray = np.zeros((1,1,nx,ny),np.bool_)
        self._MaskArray[0,0,:,:]=Mask[:,:]

    def setXY(self,xc,yc):
        self.xcyc=xc,yc
        
    def Deconvolve(self,ThSpectralFit=True):
        T=ClassTimeIt.ClassTimeIt("ClassImageDeconvMachineMultiSlice")
        T.disable()
        xc,yc=self.xcyc
        # if xc!=6010 or yc!=2241:
        #     return "Skip", True, True

            
        dirty=self._Dirty
        self.IsPadded=False
        if self._Dirty.shape[-1]!=self._Dirty.shape[-2]:
            nch,npol,_,_=dirty.shape
            original_shape = self._Dirty[0,0].shape
            Ldirty=[]
            for ich in range(nch):
                for ipol in range(npol):
                    dirty,blc_trc=pad_to_square(self._Dirty[ich,ipol])
                    Ldirty.append(dirty)
            self.blc_trc=blc_trc
            nx,ny=dirty.shape
            dirty=np.array(Ldirty).reshape((nch,npol,nx,ny))
            self.IsPadded=True
            
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(self._Dirty.shape)
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #return "Edge", True, True


        
        log.print("Deconvolve MultiSlice model cube...")
        
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
        T.timeit("Init")
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
            
            Asave=np.zeros_like(Model)
            Bsave=np.zeros_like(Model)

            LResid=[]
            LAB=[]
            for ch in range(nch):
                A,B=dirty[ch,0,s_dirty_cut,s_dirty_cut].copy(), psf[ch,0,s_psf_cut,s_psf_cut].copy()
                #A,B=psf[ch,0,s_psf_cut,s_psf_cut].copy(), psf[ch,0,s_psf_cut,s_psf_cut].copy()
                #print("FDFKLDFGKLDFK")
                xcc,ycc=self.xcyc
                
                CO=ClassOrieux.ClassOrieux(A,B)
                
                #CO=ClassOrieux.ClassOrieux(dirty[ch,0,:,:], psf[ch,0,:,:])
                model=CO.Deconv(niter=20,
                                #hyper=1e3,
                                )
                # model = restoration.richardson_lucy(dirty[ch,0,s_dirty_cut,s_dirty_cut], psf[ch,0,s_psf_cut,s_psf_cut], iterations=30)

                # def give_shift_slice(s):
                #     new_slice = slice(
                #         (s.start - 1) if s.start is not None else None,
                #         (s.stop - 1) if s.stop is not None else None,
                #         (s.step - 1) if s.step is not None else None
                #     )
                #     return new_slice
                # def give_shift_slice(s, offset=-1):
                #     start = 0 if s.start is None else s.start + offset
                #     stop = None if s.stop is None else s.stop + offset
                #     return slice(start, stop, s.step)
                
                Model[ch,0,s_dirty_cut,s_dirty_cut]=np.roll(model[:,:],(1,-1))
                Asave[ch,0,s_dirty_cut,s_dirty_cut]=A[:,:]
                Bsave[ch,0,s_dirty_cut,s_dirty_cut]=B[:,:]
                
                #LResid.append(Resid)
                indx,indy=None,None
                if CurrentNegMask is not None:
                    indx,indy=np.where(CurrentNegMask[0,0]==1)
                    nx,ny=Model[ch,0].shape
                    Model[ch,0].flat[indx*ny+indy]=0
                LAB.append((ch,A,B,model,indx,indy))
                
            T.timeit("Deconv Orieux")
            
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
                Model[ch,0,s_dirty_cut,s_dirty_cut]=model[ch,:,:]
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


        if self.IsPadded:
            Lmodel=[]
            for ich in range(nch):
                for ipol in range(npol):
                    restored_array = unpad_to_original(Model[ich,ipol], self.blc_trc)
                    Lmodel.append(restored_array)
            nx,ny=original_shape
            Model=np.array(Lmodel).reshape((nch,npol,nx,ny))

        MaxDR=1e4
        for ich in range(nch):
            absModel=np.abs(Model[ich])
            #_,indx,indy=np.where(np.any(absModel<absModel.max()/1e4,axis=0))
            _,indx,indy=np.where(absModel<absModel.max()/MaxDR)
            Model[ich,0,indx,indy]=0
            if self.GD["SSD3"]["ForcePositiveModel"]:
                _,indx,indy=np.where(Model[ich]<0)
                Model[ich,0,indx,indy]=0
                
                
        
        #print()
        
        T.timeit("Cut")
        

        LRMS=self.DicoDirty["LRMS"]
        ARMS=np.array(LRMS,dtype=np.float32)
        # Initialise x0
        nx,ny=Model.shape[-2:]
        MM=Model.copy().reshape((nch,nx*ny)).T
        meanMM=np.median(MM,axis=-1).reshape((-1,1))
        indZero=np.where(meanMM==0)[0]
        meanMM[indZero]=1
        MM=MM/meanMM

        
        MM=MM.reshape((1,nx*ny,nch))
        FF=self.DicoFreqBandToTaylor["FreqBandToTaylor_FluxVec"][self.iFacet]
        NComb,NParm=self.DicoFreqBandToTaylor["FreqBandToTaylor_ParmVec"].shape
        FF=FF.reshape((NComb,1,nch))
        
        #R=(MM-FF)
        
        RsizeBytes=NComb*nx*ny*nch*4
        RsizeMBytes=RsizeBytes/1e6
        ChunkMaxMBytes=30.
        NChunk=int(RsizeMBytes//ChunkMaxMBytes)+1
        rows=np.uint32(np.linspace(0,NComb,NChunk+1))
        AChi2Min=np.zeros((nx,ny),np.float32)
        AindComb=np.zeros((nx,ny),np.uint32)
        for iChunk in range(NChunk):
            row0,row1=rows[iChunk],rows[iChunk+1]
            R=(MM-FF[row0:row1])/ARMS.reshape((1,1,-1))
            Chi2=np.sum(R**2,axis=-1)
            Chi2Min=np.min(Chi2,axis=0).reshape((nx,ny))
            indComb=np.argmin(Chi2,axis=0).reshape((nx,ny))+row0
            if iChunk==0:
                AChi2Min[:,:]=Chi2Min
                AindComb[:,:]=indComb
            else:
                indx,indy=np.where(Chi2Min<AChi2Min)
                if indx.size==0: continue
                AChi2Min[indx,indy]=Chi2Min[indx,indy]
                AindComb[indx,indy]=indComb[indx,indy]
        
        indComb=AindComb.ravel()
        
        CoefImage2=self.DicoFreqBandToTaylor["FreqBandToTaylor_ParmVec"][indComb].copy()
        CoefImage2[:,0]*=meanMM.ravel()
        CoefImage2[indZero,:]=0
        CoefImage2=CoefImage2.T.reshape((NParm,1,nx,ny))

        CoefImage=CoefImage2
        

        
        # #######################
        self.ModelMachine.setModel(CoefImage.copy(),FluxScale=self.FitFluxScale)
        # #######################
        self.NSpectralFit=(0,0)        
        return "MaxIter", True, True
        
        # # Compute stats of the fit
        # from scipy.stats import chi2
        ModelFit=self.ModelMachine.GiveModelImage(self.GridFreqs)
        Chi2=0.
        nChi2=0
        LResid=[]
        iPlot=1
        LMc=[]
        for ich,A,B,m,indx,indy in LAB:
            Mc=fftconvolve(ModelFit[ch,0,s_dirty_cut,s_dirty_cut],B, mode='same')#[s_dirty_cut,s_dirty_cut]
            
            if self.IsPadded:            
                A = unpad_to_original(A, self.blc_trc)
            LMc.append(Mc)
            Resid=A-Mc
            Resid[Mc==0]=0
            Resid/=LRMS[ich]
            LResid.append(Resid)
            nChi2+=indx.size
            Chi2+=np.sum((Resid)**2)
            
        # #     import pylab
        # #     pylab.subplot(3,3,iPlot); iPlot+=1
        # #     pylab.imshow(A)
        # #     pylab.subplot(3,3,iPlot); iPlot+=1
        # #     pylab.imshow(Mc)
        # #     pylab.subplot(3,3,iPlot); iPlot+=1
        # #     pylab.imshow(Resid)
        # # pylab.show()
            
        # #Chi2*=(1./nChi2)
        # #k=nChi2
        # k=nChi2/25
        # Chi2red=Chi2/k
        # p = 1 - chi2.cdf(Chi2, k)
        # #######################
        
        Resid=np.max(np.abs(LResid),axis=0)
        #print("PPPPPPPP",xc,yc,Chi2,Chi2red,p,(Chi2-k)/np.sqrt(2*k),(Chi2-k)/np.sqrt(2),(Chi2red-k)/np.sqrt(2*k))
        #print("SDFLJSDFLJFD1",self.ModelMachine.DicoModel["CoefImage"].max())

        self.NSpectralFit=(0,0)        

        if (ThSpectralFit is not False) and (ThSpectralFit is not None) and (ThSpectralFit!=0.):#p<0.5:
            # Do spectral fit
            CoefImage=self.DoSpectralFit(Model,X0Model=CoefImage,Resid=Resid,ThSpectralFit=ThSpectralFit)
            self.ModelMachine.resetModel()
            self.ModelMachine.setModel(CoefImage,FluxScale=self.FitFluxScale)
            # T.timeit("DoSpectralFit")
            # print()
            T.timeit("DoSpectralFit")
        #print("SDFLJSDFLJFD2",self.ModelMachine.DicoModel["CoefImage"].max())

        # ###
        # xcc,ycc=self.xcyc
        # A,B=dirty[:,0,s_dirty_cut,s_dirty_cut].copy(), psf[:,0,s_psf_cut,s_psf_cut].copy()
        # import os
        # os.system("mkdir -p FIG")
        # np.savez("FIG/AB_%i_%i.npz"%(xcc,ycc),
        #          A=A,B=B,
        #          Asave=Asave,Bsave=Bsave,
        #          s_dirty_cut=s_dirty_cut,
        #          s_psf_cut=s_psf_cut,
        #          Model=Model,
        #          ModelFit=ModelFit,
        #          nch=nch,
        #          FreqBandToTaylor_ParmVec=self.DicoFreqBandToTaylor["FreqBandToTaylor_ParmVec"],
        #          FreqBandToTaylor_FluxVec=self.DicoFreqBandToTaylor["FreqBandToTaylor_FluxVec"][self.iFacet],
        #          CoefImage2=CoefImage2,
        #          CoefImage=CoefImage,
        #          ModelMachine=self.ModelMachine,
        #          #LResid=LResid,
        #          GridFreqs=self.GridFreqs,
        #          DegridFreqs=self.DegridFreqs,
        #          ARMS=ARMS,
        #          LMc=LMc
        #          )
        
        # import pylab
        # fig=pylab.figure()
        # pylab.clf()
        # pylab.subplot(1,3,1)
        # pylab.imshow(A[0],interpolation="nearest")
        # pylab.subplot(1,3,2)
        # pylab.imshow(B[0],interpolation="nearest")
        # # pylab.subplot(1,3,3)
        # # pylab.imshow(model,interpolation="nearest")
        # pylab.draw()
        # fig.savefig("FIG/FIG_%i_%i.png"%(xcc,ycc))
        # pylab.close(fig)
        # # #pylab.show()
        # # "FreqMappingDegrid"
        # # stop
        
        T.timeit("DoSpectralFit2")


        # import pylab
        # pylab.clf()
        # ax=pylab.subplot(2,2,1)
        # v0,v1=CoefImage[0,0].min(),CoefImage[0,0].max()
        # pylab.imshow(CoefImage[0,0],interpolation="nearest",vmin=v0,vmax=v1)
        # pylab.subplot(2,2,2,sharex=ax,sharey=ax)
        # pylab.imshow(CoefImage2[0,0],interpolation="nearest",vmin=v0,vmax=v1)
        
        # v0,v1=CoefImage[1,0].min(),CoefImage[1,0].max()
        # pylab.subplot(2,2,3,sharex=ax,sharey=ax)
        # pylab.imshow(CoefImage[1,0],interpolation="nearest",vmin=v0,vmax=v1)
        # pylab.subplot(2,2,4,sharex=ax,sharey=ax)
        # pylab.imshow(CoefImage2[1,0],interpolation="nearest",vmin=v0,vmax=v1)
        # pylab.draw()
        # pylab.show()
        
        #print()
        T.timeit("setModel")

        
        return "MaxIter", True, True   # stop deconvolution but do update model

    def DoSpectralFit(self,Model,X0Model=None,Resid=None,ThSpectralFit=None):
        log.print("Fitting MultiSlice model cube...")
        NOrder=self.GD["MultiSliceDeconv"]["PolyFitOrder"]
        LRMS=self.DicoDirty["LRMS"]
        ARMS=np.array(LRMS,dtype=np.float32)
        
        if NOrder==0:
            NOrder=self.GD["Freq"]["NBand"]
        
        def GiveResid(X,F,iFacet):
            R=np.zeros_like(F)
            Fit=np.zeros_like(F)
            for iBand in range(R.size):
                # print(self.GD["MultiSliceDeconv"]["PolyFitOrder"],NOrder,X)
                # print(self.GD["MultiSliceDeconv"]["PolyFitOrder"],NOrder,X)
                # print(self.GD["MultiSliceDeconv"]["PolyFitOrder"],NOrder,X)
                # print(self.GD["MultiSliceDeconv"]["PolyFitOrder"],NOrder,X)
                # print(self.GD["MultiSliceDeconv"]["PolyFitOrder"],NOrder,X)
                # if isinstance(X,float): X=np.array([X])
                Fit[iBand]=self.SpectralFunctionsMachine.IntExpFuncPoly(X.reshape((1,NOrder)),
                                                                        iChannel=iBand,
                                                                        iFacet=iFacet,
                                                                        FluxScale=self.FitFluxScale)
                
            #Fit[Fit<0]*=100.

            R=(F-Fit)/ARMS.reshape(F.shape)

            if X.size>1:
                Lambda=np.max([np.abs(X[1]-(-0.7))/1,1.])
                R*= Lambda#np.abs(X[1])
                
            # print("aa",R)
            return R
        
        nx,ny=Model.shape[-2],Model.shape[-1]
        CoefImage=np.zeros((NOrder,1,nx,ny),np.float32)

        indx,indy=np.where((Model[:,0,:,:]!=0).any(axis=0))
        iDone=0
        for iPix,jPix in zip(indx.tolist(),indy.tolist()):
            #log.print("%i/%i:[%i, %i]"%(iDone,indx.size,iPix,jPix))
            if Resid[iPix,jPix]==0 or np.abs(Resid[iPix,jPix])<ThSpectralFit:
                CoefImage[:,0,iPix,jPix]=X0Model[:,0,iPix,jPix]
                continue
            self.PSFServer.setLocation(iPix,jPix)
            iDone+=1
            iFacet=self.PSFServer.iFacet
            
            F=(Model[:,0,iPix,jPix]).astype(np.float64).copy()
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
            x0=(np.random.randn(NOrder)).astype(np.float64).copy()

            if X0Model is not None:
                x0=X0Model[:,0,iPix,jPix]
            
            R0=GiveResid(x0,F,iFacet)
            X=least_squares(GiveResid, x0, args=(F,iFacet),ftol=1e-2)#,xtol=1e-1,gtol=1e-1)
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

            #if Chi2Ratio>Chi2Cut:
            #    x.fill(0)
                
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

        #print("SDLSDLFJ NDone",self.xcyc,iDone/indx.size, self.GD["Deconv"]["Gain"])
        #print("SDLSDLFJ NDone",self.xcyc,iDone/indx.size, self.GD["Deconv"]["Gain"])

        self.NSpectralFit=(iDone,indx.size)

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
    
