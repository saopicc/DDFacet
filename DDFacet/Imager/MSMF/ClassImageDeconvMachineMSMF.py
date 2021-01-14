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

import numpy as np
import math
from DDFacet.Other import logger
from DDFacet.Other import ModColor
log=logger.getLogger("ClassImageDeconvMachineMSMF")
#import pylab
import traceback
import psutil
import numexpr
from pyrap.images import image
from DDFacet.Array import NpParallel
from DDFacet.Other import ClassTimeIt
from DDFacet.Imager.MSMF import ClassMultiScaleMachine
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
from DDFacet.Imager import ClassGainMachine
from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric
from DDFacet.Other import MyPickle
from DDFacet.Other.AsyncProcessPool import APP
from DDFacet.Array import shared_dict

# # if not running under a profiler, declare a do-nothing @profile decorator
# if "profile" not in globals():
#     profile = lambda x:x
#


class ClassImageDeconvMachine():

    def __init__(self, Gain=0.3,
                 MaxMinorIter=100, 
                 NCPU=1, #psutil.cpu_count()
                 CycleFactor=2.5, 
                 FluxThreshold=None, 
                 RMSFactor=3, 
                 PeakFactor=0,
                 PrevPeakFactor=0,
                 GD=None, 
                 SearchMaxAbs=1, 
                 ModelMachine=None,
                 NFreqBands=1,
                 RefFreq=None,
                 MainCache=None,
                 IdSharedMem="",
                 ParallelMode=True,
                 CacheFileName="HMPBasis",
                 **kw    # absorb any unknown keywords arguments into this
                 ):
        """
        ImageDeconvMachine constructor. Note that this should be called pretty much when setting up the imager,
        before APP workers are started, because the object registers APP handlers.
        """
        self.IdSharedMem=IdSharedMem
        self.SearchMaxAbs=SearchMaxAbs
        self._ModelImage = None
        self.MaxMinorIter = MaxMinorIter
        self.NCPU = NCPU
        self.Chi2Thr = 10000
        self._MaskArray = None
        self.GD = GD
        self.SubPSF = None
        self.MultiFreqMode = NFreqBands > 1
        self.NFreqBands = NFreqBands
        self.RefFreq = RefFreq
        self.FluxThreshold = FluxThreshold
        self.CycleFactor = CycleFactor
        self.RMSFactor = RMSFactor
        self.PeakFactor = PeakFactor
        self.PrevPeakFactor = PrevPeakFactor
        self.CacheFileName=CacheFileName
        self.GainMachine=ClassGainMachine.get_instance()
        self.ModelMachine = None
        self.PSFServer = None
        if ModelMachine is not None:
            self.updateModelMachine(ModelMachine)
        self.PSFHasChanged=False
        self._previous_initial_peak = None
        self.maincache = MainCache
        # reset overall iteration counter
        self._niter = 0
        self.facetcache=None
        self._MaskArray=None
        self.MaskMachine=None
        self.ParallelMode=ParallelMode
        if self.ParallelMode:
            APP.registerJobHandlers(self)

        # we are in a worker
        if not self.ParallelMode:
            numexpr.set_num_threads(NCPU)

        # peak finding mode.
        # "normal" searches for peak in mean dirty image
        # "sigma" searches for peak in mean_dirty/noise_map (setNoiseMap will have been called)
        # "weighted" searched for peak in mean_dirty*weight
        self._peakMode = "normal"

        self.CurrentNegMask=None
        self._NoiseMap=None
        self._PNRStop=None      # in _peakMode "sigma", provides addiitonal stopping criterion

        if self.GD["HMP"]["PeakWeightImage"]:
            print("  Reading peak weighting image %s" % self.GD["HMP"]["PeakWeightImage"], file=log)
            img = image(self.GD["HMP"]["PeakWeightImage"]).getdata()
            _, _, nx, ny = img.shape
            # collapse freq and pol axes
            img = img.sum(axis=1).sum(axis=0).T[::-1].copy()
            self._peakWeightImage = img.reshape((1,1,ny,nx))
            self._peakMode = "weighted"

        self._prevPeak = None

    def setNCPU(self,NCPU):
        self.NCPU=NCPU
        numexpr.set_num_threads(NCPU)

        
    def __del__ (self):
        if shared_dict is not None and type(self.facetcache) is shared_dict.SharedDict:
            self.facetcache.delete()

    def updateMask(self,Mask):
        nx,ny=Mask.shape
        self._MaskArray = np.zeros((1,1,nx,ny),np.bool8)
        self._MaskArray[0,0,:,:]=Mask[:,:]

    def setMaskMachine(self,MaskMachine):
        self.MaskMachine=MaskMachine

    def resetCounter(self):
        self._niter = 0

    def updateModelMachine(self, ModelMachine):
        if ModelMachine.DicoSMStacked["Type"] not in ("MSMF", "HMP"):
            raise ValueError("ModelMachine Type should be HMP")
        if ModelMachine.RefFreq != self.RefFreq:
            raise ValueError("RefFreqs should be equal")

        self.ModelMachine = ModelMachine

        if self.PSFServer is not None:
            for iFacet in range(self.PSFServer.NFacets):
                self.DicoMSMachine[iFacet].setModelMachine(self.ModelMachine)

    def GiveModelImage(self,*args): return self.ModelMachine.GiveModelImage(*args)

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe
        

    def SetPSF(self, DicoVariablePSF, quiet=False):
        self.PSFServer=ClassPSFServer(self.GD)
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF,NormalisePSF=True, quiet=quiet)
        self.PSFServer.setRefFreq(self.RefFreq)
        self.DicoVariablePSF=DicoVariablePSF
        #self.NChannels=self.DicoDirty["NChannels"]

    def Init(self, PSFVar, PSFAve, approx=False, cache=None, facetcache=None, **kwargs):
        """
        Init method. This is called after the first round of gridding: PSFs and such are available.
        ModelMachine must be set by now.
        
        facetcache: dict of basis functions. If supplied, then InitMSMF is not called.
        
        cache: cache the basis functions. If None, GD["Cache"]["HMP"] setting is used
        """
        # close the solutions dump, in case it was opened by a previous HMP instance
        ClassMultiScaleMachine.CleanSolutionsDump.close()
        self.SetPSF(PSFVar)
        self.setSideLobeLevel(PSFAve[0], PSFAve[1])
        if cache is None:
            cache = self.GD["Cache"]["HMP"]
        self.InitMSMF(approx=approx, cache=cache, facetcache=facetcache)
        ## OMS: why is this needed? self.RefFreq is set from self.ModelMachine in the first place
        # self.ModelMachine.setRefFreq(self.RefFreq)

    def Reset(self):
        print("resetting HMP machine", file=log)
        self.DicoMSMachine = {}
        if type(self.facetcache) is shared_dict.SharedDict and self.facetcache.is_writeable():
            print("deleting HMP facet cache", file=log)
            self.facetcache.delete()
        self.facetcache = None

    def setNoiseMap(self, NoiseMap, PNRStop=10):
        """Sets the noise map. The mean dirty will be divided by the noise map before peak finding.
        If PNRStop is set, an additional stopping criterion (peak-to-noisemap) will be applied.
            Peaks are reported in units of sigmas.
        If PNRStop is not set, NoiseMap is treated as simply an (inverse) weighting that will bias
            peak selection in the minor cycle. In this mode, peaks are reported in units of flux.
        """
        self._NoiseMap = NoiseMap
        self._PNRStop = PNRStop
        self._peakMode = "sigma"
        
        
    def _initMSM_handler(self, fcdict, sfdict, psfdict, iFacet, SideLobeLevel, OffsetSideLobe, verbose):
        # init PSF server from PSF shared dict
        self.SetPSF(psfdict, quiet=True)
        MSMachine = self._initMSM_facet(iFacet,fcdict,sfdict,SideLobeLevel,OffsetSideLobe,verbose=verbose)
        del MSMachine

    def _initMSM_facet(self, iFacet, fcdict, sfdict, SideLobeLevel, OffsetSideLobe, MSM0=None, verbose=False):
        """initializes MSM for one facet"""
        self.PSFServer.setFacet(iFacet)
        MSMachine = ClassMultiScaleMachine.ClassMultiScaleMachine(self.GD, fcdict, self.GainMachine, NFreqBands=self.NFreqBands)
        MSMachine.setModelMachine(self.ModelMachine)
        MSMachine.setSideLobeLevel(SideLobeLevel, OffsetSideLobe)
        MSMachine.SetFacet(iFacet)
        MSMachine.SetPSF(self.PSFServer)  # ThisPSF,ThisMeanPSF)
        MSMachine.FindPSFExtent(verbose=verbose)  # only print to log for central facet
        if MSM0 is not None:
            MSMachine.CopyListScales(MSM0)
        else:
            MSMachine.MakeListScales(verbose=verbose, scalefuncs=sfdict)
        MSMachine.MakeMultiScaleCube()
        MSMachine.MakeBasisMatrix()
        return MSMachine


    def InitMSMF(self, approx=False, cache=True, facetcache=None):
        """Initializes MSMF basis functions. If approx is True, then uses the central facet's PSF for
        all facets.
        Populates the self.facetcache dict, unless facetcache is supplied
        """
        self.DicoMSMachine = {}
        valid = True
        if facetcache is not None:
            print("HMP basis functions pre-initialized", file=log)
            self.facetcache = facetcache
        else:
            cachehash = dict(
                [(section, self.GD[section]) for section in (
                    "Data", "Beam", "Selection", "Freq",
                    "Image", "Facets", "Weight", "RIME","DDESolutions",
                    "Comp", "CF",
                    "HMP")])
            cachepath, valid = self.maincache.checkCache(self.CacheFileName, cachehash, reset=not cache or self.PSFHasChanged)
            # do not use cache in approx mode
            if approx or not cache:
                valid = False
            if valid:
                print("Initialising HMP basis functions from cache %s"%cachepath, file=log)
                self.facetcache = shared_dict.create(self.CacheFileName)
                self.facetcache.restore(cachepath)
            else:
                self.facetcache = None


        init_cache = self.facetcache is None
        if init_cache:
            self.facetcache = shared_dict.create(self.CacheFileName)

        # in any mode, start by initializing a MS machine for the central facet. This will precompute the scale
        # functions
        centralFacet = self.PSFServer.DicoVariablePSF["CentralFacet"]

        self.DicoMSMachine[centralFacet] = MSM0 = \
            self._initMSM_facet(centralFacet,
                                self.facetcache.addSubdict(centralFacet) if init_cache else self.facetcache[centralFacet],
                                None, self.SideLobeLevel, self.OffsetSideLobe, verbose=True)
        if approx:
            print("HMP approximation mode: using PSF of central facet (%d)" % centralFacet, file=log)
            for iFacet in range(self.PSFServer.NFacets):
                self.DicoMSMachine[iFacet] = MSM0
        elif (self.GD["Facets"]["NFacets"]==1)&(not self.GD["DDESolutions"]["DDSols"]):
            self.DicoMSMachine[0] = MSM0
            
        else:
            # if no facet cache, init in parallel
            if init_cache:
                for iFacet in range(self.PSFServer.NFacets):
                    if iFacet != centralFacet:
                        fcdict = self.facetcache.addSubdict(iFacet)
                        if self.ParallelMode:
                            args=(fcdict.writeonly(), MSM0.ScaleFuncs.readonly(), self.DicoVariablePSF.readonly(),
                                  iFacet, self.SideLobeLevel, self.OffsetSideLobe, False)
                            APP.runJob("InitHMP:%d"%iFacet, self._initMSM_handler,
                                       args=args)
                        else:
                            self.DicoMSMachine[iFacet] = \
                                self._initMSM_facet(iFacet, fcdict, None,
                                                    self.SideLobeLevel, self.OffsetSideLobe, MSM0=MSM0, verbose=False)

                if self.ParallelMode and self.PSFServer.NFacets>1:
                    APP.awaitJobResults("InitHMP:*", progress="Init HMP")
                    self.facetcache.reload()

            #        t = ClassTimeIt.ClassTimeIt()
            # now reinit from cache (since cache was computed by subprocesses)
            for iFacet in range(self.PSFServer.NFacets):
                if iFacet not in self.DicoMSMachine:
                    self.DicoMSMachine[iFacet] = \
                        self._initMSM_facet(iFacet, self.facetcache[iFacet], None,
                                            self.SideLobeLevel, self.OffsetSideLobe, MSM0=MSM0, verbose=False)

            # write cache to disk, unless in a mode where we explicitly don't want it
            if facetcache is None and not valid and cache and not approx:
                try:
                    #MyPickle.DicoNPToFile(facetcache,cachepath)
                    #cPickle.dump(facetcache, open(cachepath, 'w'), 2)
                    print("  saving HMP cache to %s"%cachepath, file=log)
                    self.facetcache.save(cachepath)
                    #self.maincache.saveCache("HMPMachine")
                    self.maincache.saveCache(self.CacheFileName)
                    self.PSFHasChanged=False
                    print("  HMP init done", file=log)
                except:
                    print(traceback.format_exc(), file=log)
                    print(ModColor.Str(
                         "WARNING: HMP cache could not be written, see error report above. Proceeding anyway."), file=log)

    def SetDirty(self, DicoDirty):#,DoSetMask=True):
        # if len(PSF.shape)==4:
        #     self.PSF=PSF[0,0]
        # else:
        #     self.PSF=PSF

        self.DicoDirty = DicoDirty
        # self.DicoPSF=DicoPSF
        # self.DicoVariablePSF=DicoVariablePSF

        for iFacet in range(self.PSFServer.NFacets):
            MSMachine = self.DicoMSMachine[iFacet]
            MSMachine.SetDirty(DicoDirty)

        # self._PSF=self.MSMachine._PSF
        self._CubeDirty = MSMachine._Dirty
        self._MeanDirty = MSMachine._MeanDirty
        
        # vector of per-band overall weights -- starts out as N,1 in the dico, so reshape
        W = np.float32(self.DicoDirty["WeightChansImages"])
        self._band_weights = W.reshape(W.size)[:, np.newaxis, np.newaxis, np.newaxis]

        if self._peakMode == "sigma":
            print("Will search for the peak in the SNR-weighted dirty map", file=log)
            a, b = self._MeanDirty, self._NoiseMap.reshape(self._MeanDirty.shape)
            self._PeakSearchImage = numexpr.evaluate("a/b")
        elif self._peakMode == "weighted":
            print("Will search for the peak in the weighted dirty map", file=log)
            a, b = self._MeanDirty, self._peakWeightImage
            self._PeakSearchImage = numexpr.evaluate("a*b")
        else:
            print("Will search for the peak in the unweighted dirty map", file=log)
            self._PeakSearchImage = self._MeanDirty


        # ########################################
        # op=lambda x: np.abs(x)
        # AA=op(self._PeakSearchImage)
        # _,_,xx, yx = np.where(AA==np.max(AA))
        # print "--!!!!!!!!!!!!!!!!!!!!!!",xx, yx
        # W=np.float32(self.DicoDirty["WeightChansImages"])
        # W=W/np.sum(W)
        # print "W",W
        # print np.sum(self._CubeDirty[:,0,xx,yx].ravel()*W.ravel()),self._MeanDirty[0,0,xx,yx]
        # #print np.mean(self._CubeDirty[:,0,xx,yx])
        # print "--!!!!!!!!!!!!!!!!!!!!!!"
        # ########################################
            
        NPixStats = self.GD["Deconv"]["NumRMSSamples"]
        if NPixStats>0:
            self.IndStats=np.int64(np.linspace(0,self._PeakSearchImage.size-1,NPixStats))
        # self._MeanPSF=self.MSMachine._MeanPSF


        NPSF = self.PSFServer.NPSF
        #_,_,NPSF,_=self._PSF.shape
        _, _, NDirty, _ = self._CubeDirty.shape

        off = (NPSF-NDirty)//2
        self.DirtyExtent = (off, off+NDirty, off, off+NDirty)

#        if self._ModelImage is None:
#            self._ModelImage=np.zeros_like(self._CubeDirty)

        # if DoSetMask:
        #     if self._MaskArray is None:
        #         self._MaskArray=np.zeros(self._MeanDirty.shape,dtype=np.bool8)
        #     else:
        #         maskshape = (1,1,NDirty,NDirty)
        #         # check for mask shape
        #         if maskshape != self._MaskArray.shape:
        #             ma0 = self._MaskArray
        #             _,_,nx,ny = ma0.shape
        #             def match_shapes (n1,n2):
        #                 if n1<n2:
        #                     return slice(None), slice((n2-n1)/2,(n2-n1)/2+n1)
        #                 elif n1>n2:
        #                     return slice((n1-n2)/2,(n1-n2)/2+n2), slice(None)
        #                 else:
        #                     return slice(None), slice(None)
        #             sx1, sx2 = match_shapes(NDirty, nx) 
        #             sy1, sy2 = match_shapes(NDirty, ny) 
        #             self._MaskArray = np.zeros(maskshape, dtype=np.bool8)
        #             self._MaskArray[0,0,sx1,sy1] = ma0[0,0,sx2,sy2]
        #             print>>log,ModColor.Str("WARNING: reshaping mask image from %dx%d to %dx%d"%(nx, ny, NDirty, NDirty))
        #             print>>log,ModColor.Str("Are you sure you supplied the correct cleaning mask?")
        

    def GiveEdges(self,xc0,yc0,N0,xc1,yc1,N1):
        M_xc=xc0
        M_yc=yc0
        NpixMain=N0
        F_xc=xc1
        F_yc=yc1
        NpixFacet=N1
                
        ## X
        M_x0=M_xc-NpixFacet//2
        x0main=np.max([0,M_x0])
        dx0=x0main-M_x0
        x0facet=dx0
                
        M_x1=M_xc+NpixFacet//2
        x1main=np.min([NpixMain-1,M_x1])
        dx1=M_x1-x1main
        x1facet=NpixFacet-dx1
        x1main+=1
        ## Y
        M_y0=M_yc-NpixFacet//2
        y0main=np.max([0,M_y0])
        dy0=y0main-M_y0
        y0facet=dy0
        
        M_y1=M_yc+NpixFacet//2
        y1main=np.min([NpixMain-1,M_y1])
        dy1=M_y1-y1main
        y1facet=NpixFacet-dy1
        y1main+=1

        Aedge=[x0main,x1main,y0main,y1main]
        Bedge=[x0facet,x1facet,y0facet,y1facet]
        return Aedge,Bedge


    def SubStep(self,dx,dy,LocalSM):
        _,npol,_,_ = self._MeanDirty.shape
        x0,x1,y0,y1=self.DirtyExtent

        xc,yc=dx,dy
        #NpixFacet=self.SubPSF.shape[-1]
        #PSF=self.CubePSFScales[iScale]
        N0=self._MeanDirty.shape[-1]
        N1=LocalSM.shape[-1]

        # PSF=PSF[N1/2-1:N1/2+2,N1/2-1:N1/2+2]
        # N1=PSF.shape[-1]

        #Aedge,Bedge=self.GiveEdges(xc,yc,N0,N1/2,N1/2,N1)
        N0x,N0y=self._MeanDirty.shape[-2::]
        Aedge,Bedge=GiveEdgesDissymetric(xc,yc,N0x,N0y,N1//2,N1//2,N1,N1)

        #_,n,n=self.PSF.shape
        # PSF=self.PSF.reshape((n,n))
        # print "Fpol00",Fpol
        factor = -1.  # Fpol[0,0,0]*self.Gain
        # print "Fpol01",Fpol

        nch, npol, nx, ny = LocalSM.shape
        # print Fpol[0,0,0]
        # print Aedge
        # print Bedge

        #print>>log, "    Removing %f Jy at (%i %i) (peak of %f Jy)"%(Fpol[0,0,0]*self.Gain,dx,dy,Fpol[0,0,0])
        # PSF=self.PSF[0]

        x0d, x1d, y0d, y1d = Aedge
        x0p, x1p, y0p, y1p = Bedge

        # nxPSF=self.CubePSFScales.shape[-1]
        # x0,x1=nxPSF/2-self.SupWeightWidth,nxPSF/2+self.SupWeightWidth+1
        # y0,y1=nxPSF/2-self.SupWeightWidth,nxPSF/2+self.SupWeightWidth+1
        # x0p=x0+x0p
        # x1p=x0+x1p
        # y0p=y0+y0p
        # y1p=y0+y1p
        # Bedge=x0p,x1p,y0p,y1p

        # # import pylab
        # pylab.clf()
        # ax=pylab.subplot(1,3,1)
        # vmin,vmax=self._CubeDirty.min(),self._CubeDirty.max()
        # pylab.imshow(self._MeanDirty[0,0,x0d:x1d,y0d:y1d],interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.colorbar()
        # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
        # pylab.imshow(np.mean(LocalSM,axis=0)[0,x0p:x1p,y0p:y1p],interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.colorbar()
        # pylab.draw()
        
        # #print "Fpol02",Fpol
        # # NpParallel.A_add_B_prod_factor((self.Dirty),LocalSM,Aedge,Bedge,factor=float(factor),NCPU=self.NCPU)

# <<<<<<< HEAD

        
#         self._CubeDirty[:,:,x0d:x1d,y0d:y1d] -= LocalSM[:,:,x0p:x1p,y0p:y1p]
        
# =======
        # self._CubeDirty[:,:,x0d:x1d,y0d:y1d] -= LocalSM[:,:,x0p:x1p,y0p:y1p]
        cube, sm = self._CubeDirty[:,:,x0d:x1d,y0d:y1d], LocalSM[:,:,x0p:x1p,y0p:y1p]


        # if True:#self.DoPlot:
        #     AA0=cube[0,0,:,:].copy()
        #     vmin,vmax=np.min(AA0),np.max(AA0)
        #     AA1=sm[0,0,:,:].copy()
        #     import pylab
        #     pylab.clf()
        #     pylab.subplot(1,3,1)
        #     pylab.imshow(AA0,interpolation="nearest")
        #     pylab.colorbar()
        #     pylab.subplot(1,3,2)
        #     pylab.imshow(AA1,interpolation="nearest")
        #     pylab.colorbar()
        #     pylab.subplot(1,3,3)
        #     pylab.imshow((AA0-AA1),interpolation="nearest")
        #     pylab.colorbar()
        #     pylab.draw()
        #     pylab.show(block=False)
        #     pylab.pause(0.1)

        numexpr.evaluate('cube-sm',out=cube,casting="unsafe")
        #a-=b

        if self._MeanDirty is not self._CubeDirty:
            ### old code, got MeanDirty out of alignment with CubeDirty somehow
            ## W=np.float32(self.DicoDirty["WeightChansImages"])
            ## self._MeanDirty[0,:,x0d:x1d,y0d:y1d]-=np.sum(LocalSM[:,:,x0p:x1p,y0p:y1p]*W.reshape((W.size,1,1,1)),axis=0)
            meanimage = self._MeanDirty[:, :, x0d:x1d, y0d:y1d]

                
            
            # cube.mean(axis=0, out=meanimage) should be a bit faster, but we experienced problems with some numpy versions,
            # see https://github.com/cyriltasse/DDFacet/issues/325
            # So use array copy instead (which makes an intermediate array)
            if cube.shape[0] > 1:
                meanimage[...] = (cube*self._band_weights).sum(axis=0)

                # cube.mean(axis=0, out=meanimage)
            else:
                meanimage[0,...] = cube[0,...]

            # ## this is slower:
            # self._MeanDirty[0,:,x0d:x1d,y0d:y1d] = self._CubeDirty[:,:,x0d:x1d,y0d:y1d].mean(axis=0)


            # np.save("_MeanDirty",self._MeanDirty)
            # np.save("_CubeDirty",self._CubeDirty)
            # stop

            
        if self._peakMode == "sigma":
            a, b = self._MeanDirty[:, :, x0d:x1d, y0d:y1d], self._NoiseMap[:, :, x0d:x1d, y0d:y1d]
            numexpr.evaluate("a/b", out=self._PeakSearchImage[:,:,x0d:x1d,y0d:y1d])
        elif self._peakMode == "weighted":
            a, b = self._MeanDirty[:, :, x0d:x1d, y0d:y1d], self._peakWeightImage[:, :, x0d:x1d, y0d:y1d]
            numexpr.evaluate("a*b", out=self._PeakSearchImage[:, :, x0d:x1d, y0d:y1d])

                # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
        # pylab.imshow(self._MeanDirty[0,0,x0d:x1d,y0d:y1d],interpolation="nearest",vmin=vmin,vmax=vmax)#,vmin=vmin,vmax=vmax)
        # pylab.colorbar()
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # # print Aedge
        # #unc print Bedge
        # # print self.Dirty[0,x0d:x1d,y0d:y1d]

    def Plot(self,label=None):
        import pylab
        pylab.clf()
        nch=self._CubeDirty.shape[0]
        for ich in range(nch):
            pylab.subplot(1,nch,ich+1)
            pylab.imshow(self._CubeDirty[ich,0])
            pylab.colorbar()
        if label is not None:
            pylab.suptitle(label)
        pylab.draw()
        pylab.show(block=False)
        pylab.pause(0.1)

        
    def updateRMS(self):
        _,npol,npix,_ = self._MeanDirty.shape
        NPixStats = self.GD["Deconv"]["NumRMSSamples"]
        if NPixStats:
            #self.IndStats=np.int64(np.random.rand(NPixStats)*npix**2)
            self.IndStats=np.int64(np.linspace(0,self._PeakSearchImage.size-1,NPixStats))
        else:
            self.IndStats = slice(None)
        self.RMS=np.std(np.real(self._PeakSearchImage.ravel()[self.IndStats]))
        

    def setMask(self,Mask):
        self.CurrentNegMask=Mask

    def Deconvolve(self, ch=0, UpdateRMS=True):
        """
        Runs minor cycle over image channel 'ch'.
        initMinor is number of minor iteration (keeps continuous count through major iterations)
        Nminor is max number of minor iteration

        Returns tuple of: return_code,continue,updated
        where return_code is a status string;
        continue is True if another cycle should be executed;
        update is True if model has been updated (note update=False implies continue=False)
        """
        if self._niter >= self.MaxMinorIter:
            return "MaxIter", False, False

        _, npol, npix, _ = self._MeanDirty.shape
        xc = (npix)//2

        # m0,m1=self._CubeDirty.min(),self._CubeDirty.max()
        # pylab.clf()
        # pylab.subplot(1,2,1)
        # pylab.imshow(self.Dirty[0],interpolation="nearest",vmin=m0,vmax=m1)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)

        DoAbs = int(self.GD["Deconv"]["AllowNegative"])
        print("  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]" % (
            self._niter, self.MaxMinorIter, DoAbs), file=log)

        if UpdateRMS: self.updateRMS()
        RMS=self.RMS
        self.GainMachine.SetRMS(RMS)

        Fluxlimit_RMS = self.RMSFactor*RMS
        #print "startmax",self._MeanDirty.shape,self._MaskArray.shape

        
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
        
        x,y,MaxDirty = NpParallel.A_whereMax(self._PeakSearchImage,NCPU=self.NCPU,DoAbs=DoAbs,Mask=CurrentNegMask)

        # ThisFlux is evaluated against stopping criteria. In weighted mode, use the true flux. Else use sigma value.
        ThisFlux = self._MeanDirty[0,0,x,y] if self._peakMode == "weighted" else MaxDirty
        if DoAbs:
            ThisFlux = abs(ThisFlux)
        # in weighted or noisemap mode, look up the true max as well
        trueMaxDirty = MaxDirty if self._peakMode == "normal" else ThisFlux
        # return condition indicating cleaning is to be continued
        cont = True

        CondPeak=(self._previous_initial_peak is not None)
        CondDiverge=False
        if self._previous_initial_peak is not None:
            CondDiverge=(abs(ThisFlux) > self.GD["HMP"]["MajorStallThreshold"]*self._previous_initial_peak)
        CondPeakType=(self._peakMode!="sigma")

        if CondPeak and CondDiverge and CondPeakType:
            print(ModColor.Str("STALL! dirty image peak %10.6g Jy, was %10.6g at previous major cycle."
                        % (ThisFlux, self._previous_initial_peak), col="red"), file=log)
            print(ModColor.Str("This will be the last major cycle"), file=log)
            cont = False

        self._previous_initial_peak = abs(ThisFlux)
        #x,y,MaxDirty=NpParallel.A_whereMax(self._MeanDirty.copy(),NCPU=1,DoAbs=DoAbs,Mask=self._MaskArray.copy())
        #A=self._MeanDirty.copy()
        #A.flat[:]=np.arange(A.size)[:]
        #x,y,MaxDirty=NpParallel.A_whereMax(A,NCPU=1,DoAbs=DoAbs)
        #print "max",x,y
        #stop

        # print>>log,"npp: %d %d %g"%(x,y,MaxDirty)
        # xy = ma.argmax(ma.masked_array(abs(self._MeanDirty), self._MaskArray))
        # x1, y1 = xy/npix, xy%npix
        # MaxDirty1 = abs(self._MeanDirty[0,0,x1,y1])
        # print>>log,"argmax: %d %d %g"%(x1,y1,MaxDirty1)

        Fluxlimit_Peak = ThisFlux*self.PeakFactor
        # if previous peak is not set (i.e. first major cycle), use current dirty image peak instead
        Fluxlimit_PrevPeak = (self._prevPeak if self._prevPeak is not None else ThisFlux)*self.PrevPeakFactor
        Fluxlimit_Sidelobe = ((self.CycleFactor-1.)/4.*(
            1.-self.SideLobeLevel)+self.SideLobeLevel)*ThisFlux if self.CycleFactor else 0

        mm0, mm1 = self._PeakSearchImage.min(), self._PeakSearchImage.max()

        # work out upper peak threshold
        StopFlux = max(
            Fluxlimit_Peak,
            Fluxlimit_RMS,
            Fluxlimit_Sidelobe,
            Fluxlimit_Peak,
            Fluxlimit_PrevPeak,
            self.FluxThreshold)

        print("    Dirty image peak           = %10.6g Jy [(min, max) = (%.3g, %.3g) Jy]" % (
            trueMaxDirty, mm0, mm1), file=log)
        if self._peakMode == "sigma":
            print("      in sigma units           = %10.6g" % MaxDirty, file=log)
        elif self._peakMode == "weighted":
            print("      weighted peak flux is    = %10.6g Jy" % MaxDirty, file=log)
        print("      RMS-based threshold      = %10.6g Jy [rms = %.3g Jy; RMS factor %.1f]" % (
            Fluxlimit_RMS, RMS, self.RMSFactor), file=log)
        print("      Sidelobe-based threshold = %10.6g Jy [sidelobe  = %.3f of peak; cycle factor %.1f]" % (
            Fluxlimit_Sidelobe, self.SideLobeLevel, self.CycleFactor), file=log)
        print("      Peak-based threshold     = %10.6g Jy [%.3f of peak]" % (
            Fluxlimit_Peak, self.PeakFactor), file=log)
        print("      Previous peak-based thr  = %10.6g Jy [%.3f of previous minor cycle peak]" % (
            Fluxlimit_PrevPeak, self.PrevPeakFactor), file=log)
        print("      Absolute threshold       = %10.6g Jy" % (
            self.FluxThreshold), file=log)
        print("    Stopping flux              = %10.6g Jy [%.3f of peak ]" % (
            StopFlux, StopFlux/ThisFlux), file=log)
        rms=RMS
        # MaxModelInit=np.max(np.abs(self.ModelImage))
        # Fact=4
        # self.BookKeepShape=(npix/Fact,npix/Fact)
        # BookKeep=np.zeros(self.BookKeepShape,np.float32)
        # NPixBook,_=self.BookKeepShape
        # FactorBook=float(NPixBook)/npix

        T = ClassTimeIt.ClassTimeIt()
        T.disable()

        # #print x,y
        # print>>log, "npp: %d %d %g"%(x,y,ThisFlux)
        # xy = ma.argmax(ma.masked_array(abs(self._MeanDirty), self._MaskArray))
        # x, y = xy/npix, xy%npix
        # ThisFlux = abs(self._MeanDirty[0,0,x,y])
        # print>> log, "argmax: %d %d %g"%(x, y, ThisFlux)

        if ThisFlux < StopFlux:
            print(ModColor.Str(
                "    Initial maximum peak %10.6g Jy below threshold, we're done here" %
                ThisFlux, col="green"), file=log)
            return "FluxThreshold", False, False

        # self._MaskArray.fill(1)
        # self._MaskArray.fill(0)
        #self._MaskArray[np.abs(self._MeanDirty) > Fluxlimit_Sidelobe]=0

        #        DoneScale=np.zeros((self.MSMachine.NScales,),np.float32)

        PreviousMaxFlux = 1e30

        # pBAR= ProgressBar('white', width=50, block='=', empty=' ',Title="Cleaning   ", HeaderSize=20,TitleSize=30)
        # pBAR.disable()

        self.GainMachine.SetFluxMax(ThisFlux)
        # pBAR.render(0,"g=%3.3f"%self.GainMachine.GiveGain())
        PreviousFlux=ThisFlux

        divergence_factor = 1 + max(self.GD["HMP"]["AllowResidIncrease"],0)

        xlast,ylast,Flast=None,None,None
        
        def GivePercentDone(ThisMaxFlux):
            fracDone = 1.-(ThisMaxFlux-StopFlux)/(MaxDirty-StopFlux)
            return max(int(round(100*fracDone)), 100)
            
        x0 = y0 = None

        try:
            for i in range(self._niter+1, self.MaxMinorIter+1):
                self._niter = i

                # x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=1)
                x, y, peak = NpParallel.A_whereMax(
                    self._PeakSearchImage, NCPU=self.NCPU, DoAbs=DoAbs, Mask=CurrentNegMask)
                
                # nch,npol,nx0,ny0=self._CubeDirty.shape
                # peak=0.
                # ichb=0
                # for ich in range(nch):
                #     x0, y0, peak0 = NpParallel.A_whereMax(self._CubeDirty[ich].reshape((1,npol,nx0,ny0)),
                #                                           NCPU=self.NCPU, DoAbs=DoAbs, Mask=CurrentNegMask)
                #     if peak0>peak:
                #         peak=peak0
                #         x,y=x0,y0
                #         ichb=ich
                # print(ichb,peak)
                
                if self.GD["HMP"]["FractionRandomPeak"] is not None:
                    op=lambda x: x
                    if DoAbs: op=lambda x: np.abs(x)
                    _,_,indx,indy=np.where((op(self._PeakSearchImage)>=peak*self.GD["HMP"]["FractionRandomPeak"]) & np.logical_not(CurrentNegMask))
                    ii=np.int64(np.random.rand(1)[0]*indx.size)
                    x,y=indx[ii],indy[ii]
                    peak=op(self._PeakSearchImage[0,0,x,y])


                ThisFlux = float(self._MeanDirty[0,0,x,y] if self._peakMode == "weighted" else peak)
                if DoAbs:
                    ThisFlux = abs(ThisFlux)


                if xlast is not None:
                    if x==xlast and y==ylast and np.abs((Flast-peak)/Flast)<1e-6:
                        print(ModColor.Str("    [iter=%i] peak of %.3g Jy stuck"%(i, ThisFlux), col="red"), file=log)
                        return "Stuck", False, True
                xlast=x
                ylast=y
                Flast=peak

                
                #x,y=self.PSFServer.SolveOffsetLM(self._MeanDirty[0,0],x,y); ThisFlux=self._MeanDirty[0,0,x,y]
                self.GainMachine.SetFluxMax(ThisFlux)

                # #x,y=1224, 1994
                # print x,y,ThisFlux
                # x,y=np.where(np.abs(self.Dirty[0])==np.max(np.abs(self.Dirty[0])))
                # ThisFlux=self.Dirty[0,x,y]
                # print x,y,ThisFlux
                # stop

                T.timeit("max0")
                if np.abs(ThisFlux) > divergence_factor*np.abs(PreviousFlux):
                    print(ModColor.Str(
                        "    [iter=%i] peak of %.3g Jy diverging w.r.t. floor of %.3g Jy " %
                        (i, ThisFlux, PreviousFlux), col="red"), file=log)
                    return "Diverging", False, True
                fluxgain = np.abs(ThisFlux-self._prevPeak)/abs(ThisFlux) if self._prevPeak is not None else 1e+99
                if x == x0 and y == y0 and fluxgain < 1e-6:
                    print(ModColor.Str(
                        "    [iter=%i] stalled at peak of %.3g Jy, x=%d y=%d" % (i, ThisFlux, x, y), col="red"), file=log)
                    return "Stalled", False, True
                if np.abs(ThisFlux) < np.abs(PreviousFlux):
                    PreviousFlux = ThisFlux
                    
                self._prevPeak = ThisFlux
                x0, y0 = x, y

                ThisPNR=ThisFlux/rms

                if ThisFlux <= (StopFlux if StopFlux is not None else 0.0) or \
                    ThisPNR <= (self._PNRStop if self._PNRStop is not None else 0.0):
                    rms = np.std(np.real(self._PeakSearchImage.ravel()[self.IndStats]))
                    # pBAR.render(100,"peak %.3g"%(ThisFlux,))
                    if ThisFlux <= StopFlux:
                        print(ModColor.Str(
                            "    [iter=%i] peak of %.3g Jy lower than stopping flux, PNR %.3g" %
                            (i, ThisFlux, ThisFlux/rms), col="green"), file=log)
                    elif ThisPNR <= self._PNRStop:
                        print(ModColor.Str(
                            "    [iter=%i] PNR of %.3g lower than stopping PNR, peak of %.3g Jy" %
                            (i, ThisPNR, ThisFlux), col="green"), file=log)
                        

                    cont = cont and ThisFlux > self.FluxThreshold
                    if not cont:
                        print(ModColor.Str(
                            "    [iter=%i] absolute flux threshold of %.3g Jy has been reached, PNR %.3g" %
                            (i, self.FluxThreshold, ThisFlux/rms), col="green", Bold=True), file=log)
                    # DoneScale*=100./np.sum(DoneScale)
                    # for iScale in range(DoneScale.size):
                    #     print>>log,"       [Scale %i] %.1f%%"%(iScale,DoneScale[iScale])

                    # stop deconvolution if hit absolute treshold; update model
                    return "MinFluxRms", cont, True

    #            if (i>0)&((i%1000)==0):
    #                print>>log, "    [iter=%i] Peak residual flux %f Jy" % (i,ThisFlux)
    #             if (i>0)&((i%100)==0):
    #                 PercentDone=GivePercentDone(ThisFlux)
    #                 pBAR.render(PercentDone,"peak %.3g i%d"%(ThisFlux,self._niter))
                rounded_iter_step = 1 if i < 10 else ( 
                                        10 if i<200 else ( 
                                            100 if i < 2000 
                                                else 1000 ))
                # min(int(10**math.floor(math.log10(i))), 10000)
                if i >= 10 and i % rounded_iter_step == 0:
                    # if self.GD["Debug"]["PrintMinorCycleRMS"]:
                    rms = np.std(np.real(self._PeakSearchImage.ravel()[self.IndStats]))
                    if self._peakMode == "weighted":
                        print("    [iter=%i] peak residual %.3g, gain %.3g, rms %g, PNR %.3g (weighted peak %.3g at x=%d y=%d)" % 
                                (i, ThisFlux, fluxgain, rms, ThisFlux/rms, peak, x, y), file=log)
                    else:
                        print("    [iter=%i] peak residual %.3g, gain %.3g, rms %g, PNR %.3g (at x=%d y=%d)" % (i, ThisFlux, fluxgain, rms, ThisFlux/rms, x, y), file=log)
                    # else:
                    #     print >>log, "    [iter=%i] peak residual %.3g" % (
                    #         i, ThisFlux)
                    ClassMultiScaleMachine.CleanSolutionsDump.flush()
                    #self.Plot()

                nch, npol, _, _ = self._CubeDirty.shape
                Fpol = np.float32(
                    (self._CubeDirty[
                        :, :, x, y].reshape(
                            (nch, npol, 1, 1))).copy())
                #print "Fpol",Fpol
                dx = x-xc
                dy = y-xc

                T.timeit("stuff")

                # iScale=self.MSMachine.FindBestScale((x,y),Fpol)

                self.PSFServer.setLocation(x, y)

                PSF = self.PSFServer.GivePSF()
                MSMachine = self.DicoMSMachine[self.PSFServer.iFacet]

                T.timeit("stuff2")
                LocalSM = MSMachine.GiveLocalSM(x, y, Fpol)

                T.timeit("GiveLocalSM")
                # print iScale

                # if iScale=="BadFit": continue

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
                CurrentGain=np.float32(self.GD["Deconv"]["Gain"])

                numexpr.evaluate('LocalSM*CurrentGain', out=LocalSM)
                self.SubStep(x,y,LocalSM)
                T.timeit("SubStep")

                # self.Plot(label=i)
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

                #     Aedge,Bedge=self.GiveEdges(x,y,N0,Sup/2,Sup/2,Sup)
                #     x0d,x1d,y0d,y1d=Aedge
                #     x0p,x1p,y0p,y1p=Bedge

                #     for pol in range(npol):
                #         self.ModelImage[pol,x0d:x1d,y0d:y1d]+=Gauss[x0p:x1p,y0p:y1p]*pol[pol,0,0]*self.Gain

                # else:
                #     stop

                T.timeit("End")
        except KeyboardInterrupt:
            rms = np.std(np.real(self._PeakSearchImage.ravel()[self.IndStats]))
            print(ModColor.Str(
                "    [iter=%i] minor cycle interrupted with Ctrl+C, peak flux %.3g, PNR %.3g" %
                (self._niter, ThisFlux, ThisFlux/rms)), file=log)
            # DoneScale*=100./np.sum(DoneScale)
            # for iScale in range(DoneScale.size):
            #     print>>log,"       [Scale %i] %.1f%%"%(iScale,DoneScale[iScale])
            return "MaxIter", False, True   # stop deconvolution but do update model

        rms = np.std(np.real(self._PeakSearchImage.ravel()[self.IndStats]))
        print(ModColor.Str(
            "    [iter=%i] Reached maximum number of iterations, peak flux %.3g, PNR %.3g" %
            (self._niter, ThisFlux, ThisFlux/rms)), file=log)
        # DoneScale*=100./np.sum(DoneScale)
        # for iScale in range(DoneScale.size):
        #     print>>log,"       [Scale %i] %.1f%%"%(iScale,DoneScale[iScale])
        return "MaxIter", False, True   # stop deconvolution but do update model

    def Update(self, DicoDirty, **kwargs):
        """
        Method to update attributes from ClassDeconvMachine
        """
        # Update image dict
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
