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
from DDFacet.Other import MyLogger

log= MyLogger.getLogger("ClassMultiScaleMachine")
from DDFacet.Array import ModLinAlg
from DDFacet.Array import lsqnonneg
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
import scipy.optimize

from DDFacet.ToolsDir.GiveEdges import GiveEdges

import pickle
import cPickle
from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric
import numexpr

def writetofile(fname,val):
    file1 = open(fname,'a+')
    file1.write('%s\n' % (val))
    file1.close()
    return

def pickleadic(dname,dict):
    with open(dname, 'wb') as handle:
        pickle.dump(dict, handle)
    return

class CleanSolutionsDump(object):
    """
    This is a helper class to dump and load clean solutions from a pickle file. For writing, it maintains a singleton
    version of itself. For reading, it will bootstrap itself from a dump file (so it's entirely self-contained, and
    does not need importing DDFacet or anything)

    Usage, for writing:
        # repeat at every minor cycle
        CleanSolutionsDump.init(filename, "a", "b", "c")    # only inits the first time, if necessary
        CleanSolutionsDump.write(a, b, c)

    Usage, for reading (in e.g. ipython _without_ needing to import anything)
        import cPickle
        fobj = open(filename)
        dump = cPickle.load(fobj)
        dump.read(fobj)

        pylab.plot(dump.a, dump.b)


    Note that after reading, dump.a and dump.b and dump.c will be arrays of shape (NMinorCycle,...), where ...
    is the shape of the a, b, c components passed to write()
    """
    _dump = None

    @staticmethod
    def init(filename, *columns):
        """
        Initializes a singleton solutions dump object for writing, if one hasn't been initialized yet
        """
        if CleanSolutionsDump._dump is None:
            CleanSolutionsDump._dump = CleanSolutionsDump(columns)
            CleanSolutionsDump._dump._init_for_writing(filename)

    @staticmethod
    def flush():
        """
        Initializes a solutions dump object for writing, if one hasn't been initialized yet
        """
        if CleanSolutionsDump._dump is not None:
            CleanSolutionsDump._dump._flush()

    @staticmethod
    def write(*components):
        """
        Writes set of components to dump object
        """
        CleanSolutionsDump._dump._write(*components)

    @staticmethod
    def load(filename):
        """
        Loads dump object from file
        """
        fobj = open(filename)
        dump = cPickle.load(fobj)
        dump.read(fobj)
        return dump

    @staticmethod
    def close():
        """
        Closes and deletes the dump object
        """
        if CleanSolutionsDump._dump is not None:
            CleanSolutionsDump._dump._close()
            CleanSolutionsDump._dump = None


    def __init__(self, columns):
        """
        Creates a dump object with a list of columns.
        """
        self._columns = columns
        self._fobj = None

    def _init_for_writing(self, filename):
        """
        Initializes dump object for writing to the given filename
        """
        fobj = open(filename, "w")
        cPickle.dump(self, fobj)  # dump self as first entry of file object
        self._fobj = fobj

    def _flush(self):
        if self._fobj:
            self._fobj.flush()

    def _close(self):
        if self._fobj:
            self._fobj.close()
            self._fobj = None

    def _write(self, *components):
        if len(components) != len(self._columns):
            raise TypeError,"%d components given, but dump initialized with %d columns"%(len(components), len(self._columns))
        cPickle.dump(components, self._fobj, 2)

    def read(self, fobj):
        """
        Reads a dump from a file
        """
        self._complist=[]
        while True:
            try:
                self._complist.append(cPickle.load(fobj))
            except EOFError:
                break
        print "Loaded %dx%d component dump"%(len(self._complist), len(self._complist[0]))
        for ent in self._columns:
            setattr(self, ent, [])
        for comp in self._complist:
            for e,c in zip(self._columns, comp):
                getattr(self, e).append(c)
        for ent in self._columns:
            col = np.array([x for x in getattr(self, ent) if x is not None])
            setattr(self, ent, col)
            print "%s: shape %s" % (ent, col.shape)


class ClassMultiScaleMachine():

    def __init__(self,GD,cachedict,Gain=0.1,GainMachine=None,NFreqBands=1):
        """
        :param GD:
        :param cachedict: a SharedDict in which internal arrays will be stored
        :param Gain:
        :param GainMachine:
        :param NFreqBands:
        """
        self.SubPSF=None
        self.GainMachine=GainMachine
        self.CubePSFScales=None
        self.GD=GD
        self.MultiFreqMode=False
        self.Alpha=np.array([0.],float)
        self.NFreqBands = NFreqBands
        self.MultiFreqMode = NFreqBands>1
        self.SolveMode = self.GD["HMP"]["SolverMode"]
        self._kappa = self.GD["HMP"]["Kappa"]
        self._stall_threshold = self.GD["Debug"]["CleanStallThreshold"]
        self.GlobalWeightFunction=None
        self.IsInit_MultiScaleCube=False
        self.cachedict = cachedict
        self.ListScales = cachedict.get("ListScales", None)
        self.CubePSFScales = cachedict.get("CubePSFScales", None)
        self.DicoBasisMatrix = cachedict.get("BasisMatrix", None)
        if self.DicoBasisMatrix is not None:
            self.GlobalWeightFunction = self.DicoBasisMatrix["GlobalWeightFunction"]

        # image or FT basis matrix representation? Use Image for now
        # self.Repr = "FT"
        self.Repr = "IM"
        # setup dumping
        dump_stamps = self.GD["Debug"]["DumpCleanPostageStamps"]
        dump = self.GD["Debug"]["DumpCleanSolutions"] or (dump_stamps and True)
        # dump parameter is 0 to disable, 1 to enable with default column list, else col1,col2,... etc.
        if isinstance(dump, str):
            self._dump = bool(dump)
            self._dump_cols = dump.split(',')
        else:
            self._dump = dump
            self._dump_cols = None
        dump_stamps = self.GD["Debug"]["DumpCleanPostageStamps"]
        if dump_stamps:
            if isinstance(dump_stamps, (list, tuple)) and len(dump_stamps) == 3:
                self._dump_xyr = tuple(dump_stamps)
            else:
                self._dump_xyr = 0,0,0
            print>>log,ModColor.Str("Dumping minor cycle postage stamps at %d,%d r=%dpix"%self._dump_xyr)
        else:
            self._dump_xyr = None

    def getCacheDict(self):
        return self.cachedict

    def setModelMachine(self,ModelMachine):
        self.ModelMachine=ModelMachine



    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe

    def SetFacet(self,iFacet):
        self.iFacet=iFacet


    def SetPSF(self,PSFServer):#PSF,MeanPSF):
        #self.DicoPSF=DicoPSF
        self.PSFServer=PSFServer
        self.DicoVariablePSF=self.PSFServer.DicoVariablePSF
        PSF,MeanPSF=self.PSFServer.GivePSF()
        self._PSF=PSF#self.DicoPSF["ImageCube"]
        self._MeanPSF=MeanPSF
        
        _,_,NPSF,_=self._PSF.shape
        self.NPSF=NPSF


    def SetDirty(self,DicoDirty):

        self.DicoDirty=DicoDirty
        #self.NChannels=self.DicoDirty["NChannels"]
        PeakSearchImage=self.DicoDirty["MeanImage"]
        NPixStats = 1000
        IndStats=np.int64(np.linspace(0,PeakSearchImage.size-1,NPixStats))
        self.RMS=np.std(np.real(PeakSearchImage.ravel()[IndStats]))

        self._Dirty=self.DicoDirty["ImageCube"]
        self._MeanDirty=self.DicoDirty["MeanImage"]
        _,_,NDirty,_=self._Dirty.shape
        NPSF=self.NPSF
        off=(NPSF-NDirty)/2
        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)
        

#        print>>log, "!!!!!!!!!!!"
#        self._MeanDirtyOrig=self._MeanDirty.copy()
        self.ModelMachine.setModelShape(self._Dirty.shape)

        # nch,_,_,_=self._PSF.shape
        # for ich in range(nch):
        #     pylab.clf()
        #     pylab.subplot(1,2,1)
        #     pylab.imshow(self._PSF[ich,0,:,:],interpolation="nearest")
        #     pylab.subplot(1,2,2)
        #     pylab.imshow(self._Dirty[ich,0,:,:],interpolation="nearest")
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        # stop



    def FindPSFExtent(self, verbose=False):
        if self.SubPSF is not None: return
        PSF=self._MeanPSF
        _,_,NPSF,_=PSF.shape

        # for backwards compatibility -- if PSFBox is 0 or unset, use the "auto" method below
        method = self.GD["Deconv"]["PSFBox"] or "auto"

        if isinstance(method, int):
            dx = method
            method = "explicit"
        else:
            if method == "frombox":
                xtest = np.int64(np.linspace(NPSF / 2, NPSF, 100))
                box = 100
                itest = 0
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
                #print>>log, "PSF extends to [%i] from center, with rms=%.5f"%(dx0,std)
            elif method == "auto" or method == "sidelobe":
                dx0=2*self.OffsetSideLobe
                dx0=np.max([dx0,50])
                #print>>log, "PSF extends to [%i] from center"%(dx0)
            elif method == "full":
                dx0 = NPSF/2
            else:
                raise ValueError,"unknown PSFBox setting %s" % method

            dx0=np.max([dx0,200])
            dx0=np.min([dx0,NPSF/2])
            npix=2*dx0+1
            npix=ModToolBox.GiveClosestFastSize(npix,Odd=True)

            self.PSFMargin=(NPSF-npix)/2

            dx=np.min([NPSF/2, npix/2])

        self.PSFExtent = (NPSF/2-dx,NPSF/2+dx+1,NPSF/2-dx,NPSF/2+dx+1)
        x0,x1,y0,y1 = self.PSFExtent
        self.SubPSF = self._PSF[:,:,x0:x1,y0:y1]
        #print "!!!!!!!!!!!!!!!!!!!!!!!!!!!",self.SubPSF.shape
        if verbose:
            print>>log,"using %s PSF box of size %dx%d in minor cycle subtraction" % (method, dx*2+1, dx*2+1)

    def MakeMultiScaleCube(self, verbose=False):
        if self.IsInit_MultiScaleCube: return
        T=ClassTimeIt.ClassTimeIt("MakeMultiScaleCube")
        T.disable()
        #print>>log, "Making MultiScale PSFs..."
        LScales=self.GD["HMP"]["Scales"]
        ScaleStart=0
        if 0 in LScales: 
            ScaleStart=1
            #LScales.remove(0)
        LRatios=self.GD["HMP"]["Ratios"]
        NTheta=self.GD["HMP"]["NTheta"]
        
        NAlpha=1
        if self.MultiFreqMode:
            AlphaMin,AlphaMax,NAlpha=self.GD["HMP"]["Alpha"]
            NAlpha=int(NAlpha)
            AlphaL=np.linspace(AlphaMin,AlphaMax,NAlpha)
            Alpha=np.array([0.]+[al for al in AlphaL if not(al==0.)])
        else:
            Alpha=np.array([0.]) #not in multi-frequency synthesis mode. Assume no ((v-v0)/v) modulation of I_model

        _,_,nx,ny=self.SubPSF.shape
        NScales=len(LScales)
        self.NScales=NScales
        NRatios=len(LRatios)

        Ratios=np.float32(np.array([float(r) for r in LRatios if r!="" and r!="''" and float(r) != 1]))

        Scales=np.float32(np.array([float(ls) for ls in LScales if ls!="" and ls !="''"]))


        Support=31
        #Support=1

        #CubePSFScales=np.zeros((self.NFreqBands,NScales+NRatios*NTheta*(NScales-1),nx,ny))

        # Scale Zero


        ######################

        # #############################
        # # Moving up
        # AllFreqs=[]
        # AllFreqsMean=np.zeros((self.NFreqBands,),np.float32)
        # for iChannel in range(self.NFreqBands):
        #     AllFreqs+=self.DicoVariablePSF["freqs"][iChannel]
        #     AllFreqsMean[iChannel]=np.mean(self.DicoVariablePSF["freqs"][iChannel])
        # RefFreq=np.sum(AllFreqsMean.ravel()*self.DicoVariablePSF["WeightChansImages"].ravel())
        # self.ModelMachine.setRefFreq(RefFreq)#,AllFreqs)
        # self.RefFreq=RefFreq
        # self.PSFServer.RefFreq=RefFreq
        # #############################
        T.timeit("0")
        FreqBandsFluxRatio=self.PSFServer.GiveFreqBandsFluxRatio(self.iFacet,Alpha)
        T.timeit("1")
        # if self.iFacet==96: 
        #     print 96
        #     print FreqBandsFluxRatio
        # if self.iFacet==60: 
        #     print 60
        #     print FreqBandsFluxRatio

        #FreqBandsFluxRatio.fill(1.)

        #####################

#        print FreqBandsFluxRatio
        self.Alpha=Alpha
        nch,_,nx,ny=self.SubPSF.shape

        if self.CubePSFScales is None or self.ListScales is None:
            # print>>log,"computing scales"
            #self.ListSumFluxes = []
            Theta = np.arange(0., np.pi - 1e-3, np.pi / NTheta)

            ListParam = []
            for iScales in range(ScaleStart, NScales):
                ListParam.append((iScales, 1, 0))

            for iScales in range(ScaleStart, NScales):
                for ratio in Ratios:
                    for th in Theta:
                        ListParam.append((iScales, ratio, th))

            ncubes = NAlpha*(1+len(ListParam))
            self.CubePSFScales = self.cachedict.addSharedArray("CubePSFScales",
                                        (ncubes, nch, nx, ny), self.SubPSF.dtype)

            self.ListScales = self.cachedict.addSubdict("ListScales")
            ListPSFScales = []
            for iAlpha in range(NAlpha):
                FluxRatios=FreqBandsFluxRatio[iAlpha,:]
                FluxRatios=FluxRatios.reshape((FluxRatios.size,1,1))
                ThisMFPSF = self.CubePSFScales[len(self.ListScales)]
                ThisMFPSF[:] = self.SubPSF[:,0,:,:]
                ThisMFPSF *= FluxRatios
                ThisAlpha=Alpha[iAlpha]
                #print FluxRatios,ThisAlpha
                iSlice=0
                #self.ListSumFluxes.append(1.)

                d = self.ListScales.addSubdict(len(self.ListScales))
                d["ModelType"] = "Delta"
                d["Scale"] = iSlice
                d["Alpha"] = ThisAlpha
                d["CodeTypeScale"] = 0
                d["SumFunc"] = 1.
                d["ModelParams"] = (0,0,0)
                iSlice+=1

                #print ListParam
                for iScales,MinMajRatio,th in ListParam:
                    Major=Scales[iScales]/(2.*np.sqrt(2.*np.log(2.)))
                    Minor=Major/float(MinMajRatio)
                    PSFGaussPars=(Major,Minor,th)
                    #ThisSupport=int(np.max([Support,15*Major]))
                    #if ThisSupport%2==0: ThisSupport+=1
                    #Gauss=ModFFTW.GiveGauss(ThisSupport,CellSizeRad=1.,GaussPars=PSFGaussPars)
                    #ratio=np.sum(ModFFTW.GiveGauss(101,CellSizeRad=1.,GaussPars=PSFGaussPars))/np.sum(Gauss)
                    #Gauss*=ratio
                    #ThisPSF=ModFFTW.ConvolveGaussian(ThisMFPSF.reshape((nch,1,nx,ny)),CellSizeRad=1.,GaussPars=[PSFGaussPars]*self.NFreqBands)[:,0,:,:]#[0,0]


                    ThisPSF = self.CubePSFScales[len(self.ListScales),...]

                    _,Gauss = ModFFTW.ConvolveGaussianWrapper(ThisMFPSF.reshape((nch,1,nx,ny)),
                                                              Out=ThisPSF.reshape((nch,1,nx,ny)),
                                                              Sig=Major,
                                                              GaussPar=PSFGaussPars)

                    # import pylab
                    # pylab.clf()
                    # pylab.imshow(Gauss,interpolation="nearest")
                    # pylab.title("%s"%str(PSFGaussPars))
                    # pylab.draw()
                    # pylab.show(False)
                    # pylab.pause(0.1)
                    # import time
                    # time.sleep(1.)

                    Max=np.max(ThisPSF)
                    #ThisPSF/=Max
                    #fact=np.max(Gauss)/np.sum(Gauss)
                    SumGauss=np.sum(Gauss)
                    fact=1./SumGauss
                    #fact=1./SumGauss#/(np.mean(np.max(np.max(ThisPSF,axis=-1),axis=-1)))
                    #fact=1./Max
                    Gauss *= fact#*ratio
                    ThisPSF *= fact
                    #_,n,_=ThisPSF.shape
                    #Peak=np.mean(ThisPSF[:,n/2,n/2])
                    d = self.ListScales.addSubdict(len(self.ListScales))
                    d["ModelType"] = "Gaussian"
                    d["Model"]=Gauss
                    d["ModelParams"]=PSFGaussPars
                    d["Scale"]=iScales
                    d["Alpha"]=ThisAlpha
                    d["CodeTypeScale"]=iScales
                    d["SumFunc"]=SumGauss


                iSlice+=1


                # for iScale in range(ScaleStart,NScales):

                #     for ratio in Ratios:
                #         for th in Theta:
                #             Minor=Scales[iScale]/(2.*np.sqrt(2.*np.log(2.)))
                #             Major=Minor*ratio
                #             PSFGaussPars=(Major,Minor,th)
                #             ThisPSF=ModFFTW.ConvolveGaussian(ThisMFPSF.reshape((nch,1,nx,ny)),CellSizeRad=1.,GaussPars=[PSFGaussPars]*self.NFreqBands)[:,0,:,:]
                #             Max=np.max(ThisPSF)
                #             ThisPSF/=Max
                #             ListPSFScales.append(ThisPSF)
                #             # pylab.clf()
                #             # pylab.subplot(1,2,1)
                #             # pylab.imshow(CubePSFScales[0,:,:],interpolation="nearest")
                #             # pylab.subplot(1,2,2)
                #             # pylab.imshow(CubePSFScales[iSlice,:,:],interpolation="nearest")
                #             # pylab.title("Scale = %s"%str(PSFGaussPars))
                #             # pylab.draw()
                #             # pylab.show(False)
                #             # pylab.pause(0.1)
                #             iSlice+=1
                #             Gauss=ModFFTW.GiveGauss(ThisSupport,CellSizeRad=1.,GaussPars=PSFGaussPars)/Max
                #             #fact=np.max(Gauss)/np.sum(Gauss)
                #             #Gauss*=fact

                #             self.ListScales.append({"ModelType":"Gaussian",
                #                                     "Model":Gauss, 
                #                                     "ModelParams": PSFGaussPars,
                #                                     "Scale":iScale,
                #                                     "Alpha":ThisAlpha})

            assert(len(self.ListScales) == ncubes)
        # else:
        #     print>>log,"scales already loaded"
        T.timeit("1")
        # Max=np.max(np.max(CubePSFScales,axis=1),axis=1)
        # Max=Max.reshape((Max.size,1,1))
        # CubePSFScales=CubePSFScales/Max

        # for iChannel in range(self.NFreqBands):
        #     Flat=np.zeros((nch,nx,ny),ThisPSF.dtype)
        #     Flat[iChannel]=1
        #     ListPSFScales.append(Flat)


        self.ModelMachine.setListComponants(self.ListScales)

        self.ListTypeScales=[]
        #self.ListPeakPSFScales=[]
        self.FluxScales=[]
        for DicoScale in self.ListScales.itervalues():
            self.ListTypeScales.append(DicoScale["CodeTypeScale"])
            #self.ListPeakPSFScales.append()
            self.FluxScales.append(DicoScale["SumFunc"])

        AlphaMin,AlphaMax,NAlpha=self.GD["HMP"]["Alpha"]
        NAlpha=int(NAlpha)
        AlphaL=np.linspace(AlphaMin,AlphaMax,NAlpha)
        self.Alpha=np.array([0.]+[al for al in AlphaL if not(al==0.)])

        self.FluxScales=np.array(self.FluxScales)

        self.IndexScales=[]
        self.SumFluxScales=[]
        self.ListSizeScales=[]
        self.ListTypeScales=np.array(self.ListTypeScales)
        for iScale in range(self.NScales):
            indScale=np.where(self.ListTypeScales==iScale)[0]
            self.IndexScales.append(indScale.tolist())
            self.SumFluxScales.append(self.ListScales[indScale[0]]["SumFunc"])
            self.ListSizeScales.append(self.ListScales[indScale[0]]["ModelParams"][0])
            
        self.IndexScales=np.array(self.IndexScales)
        self.SumFluxScales=np.array(self.SumFluxScales)
        self.ListSizeScales=np.array(self.ListSizeScales)
        #self.NScales=self.ListTypeScales.size/NAlpha
        #print self.IndexScales
        T.timeit("init")
        #self.SumFuncScales=np.array(self.ListSumFluxes)
        self.FFTMachine=ModFFTW.FFTW_2Donly_np(self.CubePSFScales.shape, self.CubePSFScales.dtype)

        T.timeit("init1")

        self.nFunc=self.CubePSFScales.shape[0]
        self.AlphaVec=np.array([Sc["Alpha"] for Sc in self.ListScales.itervalues()])

        self.WeightWidth = self.GD["HMP"].get("Taper",0)
        self.SupWeightWidth = self.GD["HMP"].get("Support",0)

        if not self.WeightWidth:
            self.WeightWidth = np.max([6.,np.max(Scales)])
        if not self.SupWeightWidth:
            self.SupWeightWidth = np.max([3.*self.WeightWidth,15])

        if self.WeightWidth%2==0: self.WeightWidth+=1
        if self.SupWeightWidth%2==0: self.SupWeightWidth+=1

        if verbose:
            print>>log,"  using HMP taper width %d, support size %d"%(self.WeightWidth, self.SupWeightWidth)

        T.timeit("init2")
        if self.GlobalWeightFunction is None:
            CellSizeRad=1.
            PSFGaussPars=(self.WeightWidth,self.WeightWidth,0.)
            self.GlobalWeightFunction=ModFFTW.GiveGauss(self.SubPSF.shape[-1],CellSizeRad=1.,GaussPars=PSFGaussPars)
            T.timeit("givegauss")
            nch,npol,_,_=self._PSF.shape

            # N=self.SubPSF.shape[-1]
            # dW=N/2
            # Wx,Wy=np.mgrid[-dW:dW:1j*N,-dW:dW:1j*N]
            # r=np.sqrt(Wx**2+Wy**2)
            # print r
            # r0=self.WeightWidth
            # weight=(r/r0+1.)**(-1)
            self.GlobalWeightFunction=self.GlobalWeightFunction.reshape((1,1,self.SubPSF.shape[-1],self.SubPSF.shape[-1]))*np.ones((nch,npol,1,1),np.float32)
            # print "!!!!!!!!!!!"
            # self.GlobalWeightFunction.fill(1)
            
            #ScaleMax=np.max(Scales)
            #self.SupWeightWidth=ScaleMax#3.*self.WeightWidth
        T.timeit("other")
        self.IsInit_MultiScaleCube=True

        #print>>log, "   ... Done"

    def MakeBasisMatrix(self):
        # self.OPFT=np.real
        self.OPFT=np.abs
        nxPSF=self.CubePSFScales.shape[-1]
        # x0,x1=nxPSF//2-int(self.SupWeightWidth),nxPSF//2+int(self.SupWeightWidth)+1
        # y0,y1=nxPSF//2-int(self.SupWeightWidth),nxPSF//2+int(self.SupWeightWidth)+1

        # Aedge,Bedge=GiveEdgesDissymetric((nxPSF/2,nxPSF/2),(nxPSF,nxPSF),(nxPSF/2,nxPSF/2),(int(self.SupWeightWidth),int(self.SupWeightWidth)))
        # #x0d,x1d,y0d,y1d=Aedge
        # (x0,x1,y0,y1)=Bedge

        nxGWF,nyGWF=self.GlobalWeightFunction.shape[-2],self.GlobalWeightFunction.shape[-1]
        Aedge,Bedge=GiveEdgesDissymetric((nxPSF/2,nxPSF/2),(nxPSF,nxPSF),(nxGWF/2,nyGWF/2),(nxGWF,nyGWF),WidthMax=(int(self.SupWeightWidth),int(self.SupWeightWidth)))
        x0d,x1d,y0d,y1d=Aedge
        (x0,x1,y0,y1)=Bedge

        self.SubSubCoord=(x0,x1,y0,y1)
        self.SubCubePSF=self.CubePSFScales[:,:,x0:x1,y0:y1]
        self.SubWeightFunction=self.GlobalWeightFunction[:,:,x0:x1,y0:y1]
        _,nch,_,_ = self.SubCubePSF.shape
        
        
        # import pylab
        # pylab.subplot(2,2,1)
        # pylab.imshow(self.CubePSFScales[0,0,:,:],interpolation="nearest")
        # pylab.subplot(2,2,2)
        # pylab.imshow(self.GlobalWeightFunction[0,0,:,:],interpolation="nearest")
        # pylab.subplot(2,2,3)
        # pylab.imshow(self.SubCubePSF[0,0,:,:],interpolation="nearest")
        # pylab.subplot(2,2,4)
        # pylab.imshow(self.SubWeightFunction[0,0,:,:],interpolation="nearest")
        # pylab.draw()
        # pylab.show()
        # stop


        self.WeightMeanJonesBand=self.DicoVariablePSF["MeanJonesBand"][self.iFacet].reshape((nch,1,1,1))
        WeightMueller=self.WeightMeanJonesBand.ravel()
        self.WeightMuellerSignal = WeightMueller*self.DicoVariablePSF["WeightChansImages"].ravel()

        if self.DicoBasisMatrix is None:
            self.DicoBasisMatrix = self.GiveBasisMatrix()
        return self.DicoBasisMatrix


    def GiveBasisMatrix(self,SubSubSubCoord=None):
#        print>>log,"Calculating basisc function for SubSubSubCoord=%s"%(str(SubSubSubCoord))
        if SubSubSubCoord is None:
            CubePSF=self.SubCubePSF
            WeightFunction=self.SubWeightFunction
        else:
            x0s,x1s,y0s,y1s=SubSubSubCoord
            CubePSF=self.SubCubePSF[:,:,x0s:x1s,y0s:y1s]
            WeightFunction=self.SubWeightFunction[:,:,x0s:x1s,y0s:y1s]

        nFunc,nch,nx,ny=CubePSF.shape
        # Bias=np.zeros((nFunc,),float)
        # for iFunc in range(nFunc):
        #     Bias[iFunc]=np.sum(CubePSF[iFunc]*WeightFunction[:,0,:,:])

        # Bias/=np.sum(Bias)
        # self.Bias=Bias
        # stop
        #BM=(CubePSFNorm.reshape((nFunc,nch*nx*ny)).T.copy())

        # if called with None, we're filling the cache, so create a subdict in the cache dict
        if SubSubSubCoord is None:
            DicoBasisMatrix = self.cachedict.addSubdict("BasisMatrix")
            DicoBasisMatrix["CubePSF"] = CubePSF
            DicoBasisMatrix["WeightFunction"] = WeightFunction
            DicoBasisMatrix["GlobalWeightFunction"] = self.GlobalWeightFunction
        # else extracting subcube, so return a temporary dict
        else:
            DicoBasisMatrix = {"CubePSF": CubePSF,
                            "WeightFunction": WeightFunction,
                            "GlobalWeightFunction": self.GlobalWeightFunction}



        if self.Repr == "IM":
            BM = np.float64(CubePSF.reshape((nFunc,nch*nx*ny)).T)
            WVecPSF = np.float64(WeightFunction.reshape((WeightFunction.size,1)))
            BMT_BM = np.dot(BM.T,WVecPSF*BM)
            DicoBasisMatrix["BM"] = np.float32(BM)
            DicoBasisMatrix["BMT_BM_inv"] = np.float32(ModLinAlg.invSVD(BMT_BM))
            BMnorm = np.sum(BM ** 2, axis=0)
            DicoBasisMatrix["BMnorm"] = np.float32(1. / BMnorm.reshape((nFunc, 1)))

        if self.Repr == "FT":
            #fCubePSF=np.float32(self.FFTMachine.fft(np.complex64(CubePSF)).real)
            W=WeightFunction.reshape((1,nch,nx,ny))
            fCubePSF=np.float32(self.OPFT(self.FFTMachine.fft(np.complex64(CubePSF*W))))
            nch,npol,_,_=self._PSF.shape
            u,v=np.mgrid[-nx/2+1:nx/2:1j*nx,-ny/2+1:ny/2:1j*ny]

            r=np.sqrt(u**2+v**2)
            r0=1.
            UVTaper=1.-np.exp(-(r/r0)**2)

            UVTaper=UVTaper.reshape((1,1,nx,ny))*np.ones((nch,npol,1,1),np.float32)


            UVTaper.fill(1)

            UVTaper*=self.WeightMuellerSignal.reshape((nch,1,1,1))

            # fCubePSF[:,:,nx/2,ny/2]=0
            # import pylab
            # for iFunc in range(self.nFunc):
            #     Basis=fCubePSF[iFunc]
            #     pylab.clf()
            #     pylab.subplot(1,3,1)
            #     pylab.imshow(Basis[0]*UVTaper[0,0],interpolation="nearest")
            #     pylab.title(iFunc)
            #     pylab.subplot(1,3,2)
            #     pylab.imshow(Basis[1]*UVTaper[0,0],interpolation="nearest")
            #     pylab.subplot(1,3,3)
            #     pylab.imshow(Basis[2]*UVTaper[0,0],interpolation="nearest")
            #     pylab.draw()
            #     pylab.show(False)
            #     pylab.pause(0.1)



            fBM = np.float64((fCubePSF.reshape((nFunc,nch*nx*ny)).T))
            fBMT_fBM = np.dot(fBM.T,UVTaper.reshape((UVTaper.size,1))*fBM)
            DicoBasisMatrix["fBMT_fBM_inv"] = np.float32(ModLinAlg.invSVD(fBMT_fBM))
            DicoBasisMatrix["fBM"] = np.float32(fBM)
            DicoBasisMatrix["fWeightFunction"] = UVTaper

            # DeltaMatrix=np.zeros((nFunc,),np.float32)
            # #BM_BMT=np.dot(BM,BM.T)
            # #BM_BMT_inv=ModLinAlg.invSVD(BM_BMT)

            # BM_BMT_inv=np.diag(1./np.sum(BM*BM,axis=1))
            # nData,_=BM.shape
            # for iFunc in range(nFunc):
            #     ai=BM[:,iFunc].reshape((nData,1))
            #     DeltaMatrix[iFunc]=1./np.sqrt(np.dot(np.dot(ai.T,BM_BMT_inv),ai))
            # DeltaMatrix=DeltaMatrix.reshape((nFunc,1))
            # print>>log, "Delta Matrix: %s"%str(DeltaMatrix)

            #WeightFunction.fill(1.)


#        if self.GD["Debug"]["DumpCleanSolutions"] and not SubSubSubCoord:
#            BaseName = self.GD["Output"]["Name"]
#            pickleadic(BaseName+".DicoBasisMatrix.pickle",DicoBasisMatrix)

        return DicoBasisMatrix
        
        

    def giveSmallScaleBias(self):
        a=self.ListSizeScales
        y=[]
        for a in self.ListSizeScales:
            if a==0:
                b=1.
            else:
                a1=self.ListSizeScales[1]
                b=0.55**(-np.log(a1/a)/np.log(2.))
            y.append(b)
        return np.array(y)

    #@profile
    def GiveLocalSM(self,(x,y),Fpol):
        T= ClassTimeIt.ClassTimeIt("   GiveLocalSM")
        T.disable()

        N0=self._Dirty.shape[-1]
        N1=self.DicoBasisMatrix["CubePSF"].shape[-1]
        xc,yc=x,y

        #N1=CubePSF.shape[-1]

        nchan,npol,_,_=Fpol.shape

        JonesNorm=np.ones((nchan,npol,1,1),Fpol.dtype)
        #print self.DicoDirty.keys()
        #print Fpol
        FpolTrue=Fpol
        if self.DicoDirty["JonesNorm"] is not None:
            JonesNorm=(self.DicoDirty["JonesNorm"][:,:,x,y]).reshape((nchan,npol,1,1))
            
            FpolTrue=Fpol/np.sqrt(JonesNorm)
            #print JonesNorm

        # #print Fpol
        # print "JonesNorm",JonesNorm
        # FpolMean=np.mean(Fpol,axis=0).reshape((1,npol,1,1))

        #Aedge,Bedge=GiveEdges((xc,yc),N0,(N1/2,N1/2),N1)
        N0x,N0y=self._Dirty.shape[-2],self._Dirty.shape[-1]
        Aedge,Bedge=GiveEdgesDissymetric((xc,yc),(N0x,N0y),(N1/2,N1/2),(N1,N1))
        x0d,x1d,y0d,y1d=Aedge
        x0s,x1s,y0s,y1s=Bedge
        nxs,nys=x1s-x0s,y1s-y0s
        dirtyNormIm=self._Dirty[:,:,x0d:x1d,y0d:y1d]
        
        if (nxs!=self.DicoBasisMatrix["CubePSF"].shape[-2])|(nys!=self.DicoBasisMatrix["CubePSF"].shape[-1]):
            DicoBasisMatrix=self.GiveBasisMatrix((x0s,x1s,y0s,y1s))
        else:
            DicoBasisMatrix=self.DicoBasisMatrix

        CubePSF=DicoBasisMatrix["CubePSF"]

        nxp,nyp=x1s-x0s,y1s-y0s

        T.timeit("0")
        #MeanData=np.sum(np.sum(dirtyNorm*WCubePSF,axis=-1),axis=-1)
        #MeanData=MeanData.reshape(nchan,1,1)
        #dirtyNorm=dirtyNorm-MeanData.reshape((nchan,1,1,1))

        # dirtyNormIm=dirtyNormIm/FpolMean


        #print "0",np.max(dirtyNormIm)
        dirtyNormIm = np.float64(dirtyNormIm)
        dirtyNormIm /= np.sqrt(JonesNorm)
        #print "1",np.max(dirtyNormIm)


        if self.Repr=="FT":
            BM=DicoBasisMatrix["fBM"]
            WCubePSF=DicoBasisMatrix["fWeightFunction"]#*(JonesNorm)
            WCubePSFIm=DicoBasisMatrix["WeightFunction"]#*(JonesNorm)
            WVecPSF=WCubePSF.reshape((WCubePSF.size,1))
            dirtyNorm=np.float64(self.OPFT(self.FFTMachine.fft(np.complex64(dirtyNormIm*WCubePSFIm))))#.real)
            BMT_BM_inv=DicoBasisMatrix["fBMT_fBM_inv"]
        else:
            #print "0:",DicoBasisMatrix["WeightFunction"].shape,JonesNorm.shape
            WCubePSF=DicoBasisMatrix["WeightFunction"]#*(JonesNorm)
            WVecPSF=WCubePSF.reshape((WCubePSF.size,1))
            dirtyNorm=dirtyNormIm
            BM=DicoBasisMatrix["BM"]
            BMT_BM_inv=DicoBasisMatrix["BMT_BM_inv"]


        dirtyVec=dirtyNorm.reshape((dirtyNorm.size,1))
        T.timeit("1")



        #self.SolveMode="MatchingPursuit"
        # self.SolveMode="PI"
        #self.SolveMode="ComplementaryMatchingPursuit"
        #self.SolveMode="NNLS"

        # MeanFluxTrue=np.sum(FpolTrue.ravel()*self.WeightMuellerSignal)/np.sum(self.WeightMuellerSignal)
        MeanFluxTrue = Fpol.mean()/np.sqrt(JonesNorm).mean()
        
        if  self.SolveMode=="MatchingPursuit":
            #Sol=np.dot(BM.T,WVecPSF*dirtyVec)
            Sol=np.dot(BMT_BM_inv,np.dot(BM.T,WVecPSF*dirtyVec))
            #print Sol
            #indMaxSol1=np.where(np.abs(Sol)==np.max(np.abs(Sol)))[0]
            #indMaxSol0=np.where(np.abs(Sol)!=np.max(np.abs(Sol)))[0]
            indMaxSol1=np.where(np.abs(Sol)==np.max(np.abs(Sol)))[0]
            indMaxSol0=np.where(np.abs(Sol)!=np.max(np.abs(Sol)))[0]
            #indMaxSol1=np.where(np.abs(Sol)==np.max((Sol)))[0]
            #indMaxSol0=np.where(np.abs(Sol)!=np.max((Sol)))[0]

            Sol[indMaxSol0]=0
            Max=Sol[indMaxSol1[0]]
            Sol[indMaxSol1]=MeanFluxTrue#np.sign(Max)*MeanFluxTrue

            # D=self.ListScales[indMaxSol1[0]]
            # print "Type %10s (sc, alpha)=(%i, %f)"%(D["ModelType"],D["Scale"],D["Alpha"])
            LocalSM=self.CubePSFScales[indMaxSol1[0]]*MeanFluxTrue#FpolMean.ravel()[0]
            # LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)

        elif  self.SolveMode=="ComplementaryMatchingPursuit":
            #Sol=DicoBasisMatrix["DeltaMatrix"]*np.dot(BM.T,WVecPSF*dirtyVec)
            #Sol=DicoBasisMatrix["BMnorm"]*np.dot(BM.T,WVecPSF*dirtyVec)
            Sol=DicoBasisMatrix["BMnorm"]*np.dot(BM.T,WVecPSF*(dirtyVec/MeanFluxTrue-BM))
            #Sol=np.dot(BM.T,WVecPSF*dirtyVec)
            indMaxSol1=np.where(np.abs(Sol)==np.max(np.abs(Sol)))[0]
            indMaxSol0=np.where(np.abs(Sol)!=np.max(np.abs(Sol)))[0]

            Sol[indMaxSol0]=0
            Max=Sol[indMaxSol1[0]]
            Sol[indMaxSol1]=MeanFluxTrue#np.sign(Max)*MeanFluxTrue

            # D=self.ListScales[indMaxSol1[0]]
            # print "Type %10s (sc, alpha)=(%i, %f)"%(D["ModelType"],D["Scale"],D["Alpha"])
            # LocalSM=self.CubePSFScales[indMaxSol1[0]]*FpolMean.ravel()[0]
            LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)

        elif self.SolveMode=="PI":
            
            Sol=np.float32(np.dot(BMT_BM_inv,np.dot(BM.T,WVecPSF*dirtyVec)))
            #Sol.fill(1)
            
            #LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)*FpolMean.ravel()[0]
            
            
            # # # ############## debug
            # #Sol.fill(0)
            # #Sol[0]=1.
            # ConvSM=np.dot(BM,Sol.reshape((-1,1))).reshape((nchan,1,nxp,nyp))
            # print
            # print "=====",self.iFacet,x,y
            # print self.PSFServer.GiveFreqBandsFluxRatio(self.iFacet,self.Alpha)
            # print "Apparant:", Fpol.ravel()
            # print "Correct :",FpolTrue.ravel()
            # print "Cube app:", dirtyNormIm[:,0,xc-x0d,yc-y0d]
            # print "LocalSM :", ConvSM[:,0,xc-x0d,yc-y0d]
            # print "Weights :",self.DicoDirty["WeightChansImages"].ravel()
            # print "Data shape",dirtyVec.shape
            # # print dirtyVec
            # # #print "BM",BM.shape
            # # #print BM
            # print "Sum, Sol",np.sum(Sol),Sol.ravel()
            # # print "aaa",np.dot(BM,Sol)
            # # stop
            # #print "FpolTrue,WeightChansImages:",FpolTrue.ravel(),self.DicoDirty["WeightChansImages"].ravel()
            # print "MeanFluxTrue",MeanFluxTrue
            # print "coef",coef
            # import pylab
            # pylab.clf()
            # # iFunc=0
            # # #BM*=0.947
            # # pylab.plot(dirtyVec.ravel())
            # # pylab.plot(BM[:,iFunc].ravel())
            # # #pylab.plot(BM)
            # # pylab.plot(dirtyVec.ravel()-BM[:,iFunc].ravel())
            # pylab.subplot(1,3,1)
            # pylab.imshow(dirtyNormIm[0,0,:,:],interpolation="nearest")
            # pylab.colorbar()
            # pylab.subplot(1,3,2)
            # pylab.imshow(ConvSM[0,0,:,:],interpolation="nearest")
            # pylab.colorbar()
            # pylab.subplot(1,3,3)
            # pylab.imshow((dirtyNormIm-ConvSM)[0,0,:,:],interpolation="nearest")
            # pylab.colorbar()
            # pylab.draw()
            # pylab.show(False)
            # # ##########################
            # stop

            # regularized solution is just MeanFluxTrue with spi=0, and nulls for the other components
            # regularizatuion coefficient goes to 0 to use regularized solution, to 1 to use the "proper" solution


            # First coefficient: take care of cases where solution is too small
            #   if solution sum tends to be much less than mean flux, then coef1 -> 0 (use regularized solution)
            #   if it is larger than  mean flux, then coef1 -> 1
            coef1 = min(abs(Sol.sum()/MeanFluxTrue),1.)

            # Second coefficient: take care of cases where solution has components of alternating signs
            # this is characterized by a high std of the solution coefficients
            # 1/self._kappa determines the "maximum" stddev (relative to maximum solution amplitude) beyond
            # which coef2->0 to force a fully-regular solution
            # NB: this caused standard tests to fail, so I've set default Kappa=0 for now
            coef2 = max(1 - Sol.std()/abs(MeanFluxTrue) * self._kappa,0)

            coef = coef1*coef2

            Sol0 = Sol
            SolReg = np.zeros_like(Sol)
            SolReg[0] = MeanFluxTrue

            if np.sign(SolReg[0])!=np.sign(np.sum(Sol)):
                Sol=SolReg
            else:
                Sol=Sol*coef+SolReg*(1.-coef)
                # if np.abs(np.sum(Sol))>np.abs(MeanFluxTrue):
                #     Sol=SolReg

            # print "Sum, Sol",np.sum(Sol),Sol.ravel()
            
            Fact=(MeanFluxTrue/np.sum(Sol))
            #Sol*=Fact
            
            
            if abs(Sol).max() < self._stall_threshold:
                print>>log,"Stalled CLEAN!"
                print>>log,(self.iFacet, x, y, Fpol, FpolTrue, Sol, Sol0, SolReg, coef, coef1, coef2, MeanFluxTrue, self.WeightMuellerSignal)
                raise RuntimeError("CLEAN has stalled. This is a bug!")

            # print "Sum, Sol",np.sum(Sol),Sol.ravel()
            

            # LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)/Fact

            # # multiply basis functions by solutions (first axis is basis index)
            a, b = self.CubePSFScales, np.float32(Sol.reshape((Sol.size, 1, 1, 1)))
            scales = numexpr.evaluate('a*b')
            # # model is sum of basis functions
            LocalSM = scales.sum(axis=0) if Sol.size>1 else scales[0,...]

            if self._dump:
                postage_stamp = None
                # dump sub-images, if we come within a certain distance of x,y
                if self._dump_xyr:
                    xd, yd, radius = self._dump_xyr
                    if abs(x - xd) < radius and abs(y - yd) < radius:
                        postage_stamp = self._Dirty[:, :, xd-radius*2:xd+radius*2, yd-radius*2:yd+radius*2]
                columns = [ "iFacet", "x", "y", "Fpol", "FpolTrue", "Sol", "Sol0", "SolReg", "coef", "coef1", "coef2", "Fact", "MeanFluxTrue",
                            "WeightMuellerSignal", "postage_stamp" ]
                iFacet, WeightMuellerSignal = self.iFacet, self.WeightMuellerSignal
                if self._dump_cols:
                    columns += self._dump_cols
                CleanSolutionsDump.init(self.GD["Output"]["Name"] + ".clean.solutions", *columns)
                CleanSolutionsDump.write(*[ locals()[col] for col in columns ])

                    #print "Max abs model",np.max(np.abs(LocalSM))
            #print "Min Max model",LocalSM.min(),LocalSM.max()
        elif self.SolveMode=="NNLS":
            #HasReverted=False
            Peak=np.max(dirtyVec)
            # if Peak<0:
            #     dirtyVec=dirtyVec*-1
            #     HasReverted=True
                
            W=WVecPSF.copy()
            # print ":::::::::"
            # W.fill(1.)
            OrigDirty=dirtyVec.copy().reshape((nchan,1,nxp,nyp))[:,0]
            MeanOrigDirty=np.mean(OrigDirty,axis=0)
            xc0,yc0=np.where(np.abs(MeanOrigDirty) == np.max(np.abs(MeanOrigDirty)))
            PeakMeanOrigDirty=MeanOrigDirty[xc0[0],yc0[0]]
            dirtyVec=dirtyVec.copy()
            Mask=np.zeros(WVecPSF.shape,np.bool8)

            T=ClassTimeIt.ClassTimeIt()
            T.disable()
            NNLSStep=10
            for iIter in range(100):
                A=W*BM
                y=W*dirtyVec
                d=dirtyVec.reshape((nchan,1,nxp,nyp))[:,0]
                PeakMeanOrigResid=np.mean(d,axis=0)[xc0[0],yc0[0]]

                FactNorm=np.abs(PeakMeanOrigDirty/PeakMeanOrigResid)
                if np.isnan(FactNorm) or np.isinf(FactNorm):
                    #print "Cond1 %i"%iIter 
                    Sol=np.zeros((A.shape[1],),dtype=np.float32)
                    break
                T.timeit("0")
                if 1.<FactNorm<10.:
                    y*=FactNorm
                #print "  ",PeakMeanOrigDirty,PeakMeanOrigResid,PeakMeanOrigDirty/PeakMeanOrigResid

                if not(iIter%NNLSStep):
                    x,_=scipy.optimize.nnls(A, y.ravel())
                    T.timeit("1")
                    #x0=x.copy()
                    # Compute "dirty" solution and residuals
                    ConvSM=np.dot(BM,x.reshape((-1,1))).reshape((nchan,1,nxp,nyp))[:,0]
                Resid=d-ConvSM

                T.timeit("2")
                # # ### debug
                # print "x",x
                # VecConvSM=np.dot(BM,x.reshape((-1,1))).ravel()
                # x2=np.zeros_like(x)
                # x2[3]=1.#*0.95
                # #x0=x2
                # VecConvSM=np.dot(BM,x2.reshape((-1,1))).ravel()
                # import pylab
                # pylab.clf()
                # pylab.plot(dirtyVec.ravel())
                # pylab.plot(VecConvSM)
                # pylab.plot(dirtyVec.ravel()-VecConvSM)
                # #pylab.plot(dirtyVec.ravel()-VecConvSM2)
                # pylab.draw()
                # pylab.show(False)
                # stop
                # # ###########
                
                # Max_d=np.mean(np.max(np.max(d,axis=-1),axis=-1))
                # Max_ConvSM=np.mean(np.max(np.max(ConvSM,axis=-1),axis=-1))
                # #r=Max_d/Max_ConvSM
                # #x*=r
                # #ConvSM*=r



                #x,_=scipy.optimize.nnls(A, y.ravel())
                #ConvSM=np.dot(BM,x.reshape((-1,1))).reshape((nchan,1,nxp,nyp))[:,0]
                w=W.reshape((nchan,1,nxp,nyp))[:,0]
                m=Mask.reshape((nchan,1,nxp,nyp))[:,0]

                sig=self.RMS#np.std(Resid)
                MaxResid=np.max(Resid)
                #sig=np.sqrt(np.sum(w*Resid**2)/np.sum(w))
                #MaxResid=np.max(w*Resid)
                
                # Check if there is contamining nearby sources
                _,xc1,yc1=np.where((Resid>self.GD["HMP"]["OuterSpaceTh"]*sig)&(Resid==MaxResid))

                dirtyVecSub=d
                Sol=x

                # Compute flux in each spacial scale
                SumCoefScales=np.zeros((self.NScales,),np.float32)
                for iScale in range(self.NScales):
                    indAlpha=self.IndexScales[iScale]
                    SumCoefScales[iScale]=np.sum(Sol[indAlpha])
                #print "  SumCoefScales",SumCoefScales


                # If source is contaminating, substract it with the delta (with alpha=0)
                T.timeit("3")
                if xc1.size>0 and MaxResid>Peak/100.:
                    CentralPixel=(xc1[0]==xc0[0] and yc1[0]==yc0[0])
                    if CentralPixel: 
                        #print "CondCentralPix %i"%iIter 
                        break
                    F=Resid[:,xc1[0],yc1[0]]
                    dx,dy=nxp/2-xc1[0],nyp/2-yc1[0]
                    _,_,nxPSF,nyPSF=self.SubPSF.shape

                    #xc2,yc2=nxPSF/2+dx,nyPSF/2+dy
                    #ThisPSF=self.SubPSF[:,0,xc2-nxp/2:xc2+nxp/2+1,yc2-nyp/2:yc2+nyp/2+1]

                    N0x,N0y=d.shape[-2::]
                    Aedge,Bedge=GiveEdgesDissymetric((xc1[0],yc1[0]),(N0x,N0y),(nxPSF/2,nyPSF/2),(nxPSF,nyPSF))
                    x0d,x1d,y0d,y1d=Aedge
                    x0p,x1p,y0p,y1p=Bedge
                    ThisPSF=self.SubPSF[:,0,x0p:x1p,y0p:y1p]
                    _,nxThisPSF,nyThisPSF=ThisPSF.shape

                    #############
                    ThisDirty=ThisPSF*F.reshape((-1,1,1))
                    dirtyVecSub[:,x0d:x1d,y0d:y1d]=d[:,x0d:x1d,y0d:y1d]-ThisDirty
                    dirtyVec=dirtyVecSub.reshape((-1,1))
                    DoBreak=False

                else:
                    #print "NotContam %i"%iIter 
                    #print "  xc1.size>0, MaxResid>Peak/100.: ",xc1.size>0, MaxResid>Peak/100.

                    x,_=scipy.optimize.nnls(A, y.ravel())
                    DoBreak=True

                # ####### debug
                # import pylab
                # pylab.clf()
                # pylab.subplot(2,3,1)
                # pylab.imshow(OrigDirty[0],interpolation="nearest")
                # pylab.title("Dirty")
                # pylab.subplot(2,3,2)
                # pylab.imshow(d[0],interpolation="nearest")
                # pylab.title("Dirty iter=%i"%iIter)
                # pylab.colorbar()
                # pylab.subplot(2,3,3)
                # pylab.imshow(ConvSM[0],interpolation="nearest")
                # pylab.title("Model")
                # pylab.colorbar()
                # pylab.subplot(2,3,4)
                # pylab.imshow((Resid)[0],interpolation="nearest")
                # pylab.title("Residual")
                # pylab.colorbar()
                # pylab.subplot(2,3,5)
                # pylab.imshow(dirtyVecSub[0],interpolation="nearest")
                # pylab.title("NewDirty")
                # pylab.colorbar()
                # pylab.draw()
                # pylab.show(False)
                # pylab.pause(0.1)
                # #####################

                T.timeit("4")
                if DoBreak: break


                # if indTh[0].size>0:
                #     w[indTh]=0
                #     m[indTh]=1
                #     W=w.reshape((-1,1))
                #     Mask=m.reshape((-1,1))
                # else:
                #     break

        

            Sol=x
            #Sol.flat[:]/=self.SumFuncScales.flat[:]
            #print Sol

            #if HasReverted: Sol*=-1
            
            Mask=np.zeros((Sol.size,),np.float32)
            FuncScale=1.#self.giveSmallScaleBias()
            wCoef=SumCoefScales/self.SumFluxScales*FuncScale
            ChosenScale=np.argmax(wCoef)

            # print "==============="
            # print "%s -> %i"%(str(wCoef),ChosenScale)
            # print "Sol =  %s"%str(Sol)
            # print


            Mask[self.IndexScales[ChosenScale]]=1
            Sol.flat[:]*=Mask.flat[:]



            SolReg = np.zeros_like(Sol)
            SolReg[0] = MeanFluxTrue
            Peak=np.mean(np.max(np.max(ConvSM,axis=-1),axis=-1))

            if (np.sign(SolReg[0]) != np.sign(np.sum(Sol))) or (np.max(np.abs(Sol))==0):
                Sol = SolReg
                LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)
            else:
                coef = np.min([np.abs(Peak / MeanFluxTrue), 1.])
                Sol = Sol * coef + SolReg * (1. - coef)
                # if np.abs(np.sum(Sol))>np.abs(MeanFluxTrue):
                #     Sol=SolReg

                # Fact=(MeanFluxTrue/np.sum(Sol))
                LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)
                Peak=np.mean(np.max(np.max(LocalSM,axis=-1),axis=-1))

                nch,nx,ny=LocalSM.shape
                #print Peak

                #Sol=(Sol.reshape((-1,self.NScales))/(self.SumFluxScales.reshape((1,-1)))).flatten()

                
                #print "Sol1",Sol
                LocalSM*=(MeanFluxTrue/Peak)
                Sol*=(MeanFluxTrue/Peak)
                #print coef,MeanFluxTrue,Peak,Sol
            #print "Sol2",Sol


            if self._dump:
                columns = ["iFacet", "xc", "yc", "Fpol", "FpolTrue", "Sol", "WeightMuellerSignal"]
                iFacet, WeightMuellerSignal = self.iFacet, self.WeightMuellerSignal
                if self._dump_cols:
                    columns += self._dump_cols
                CleanSolutionsDump.init(self.GD["Output"]["Name"] + ".clean.solutions", *columns)
                CleanSolutionsDump.write(*[locals()[col] for col in columns])


            # P=set()
            # R=set(range(self.nFunc))
            # x=np.zeros((self.nFunc,1),np.float32)
            # s=np.zeros((self.nFunc,1),np.float32)
            # A=BM
            # y=dirtyVec
            # w=np.dot(A.T,y-np.dot(A,x))
            # print>>log, "init w: %s"%str(w.ravel())
            # while (len(R)>0):
            #     print>>log, "while j (len(R)>0)"
            #     j=np.argmax(w)
            #     print>>log, "selected j: %i"%j
            #     print>>log, "P: %s"%str(P)
            #     print>>log, "R: %s"%str(R)
            #     P.add(j)
            #     R.remove(j)
            #     print>>log, "P: %s"%str(P)
            #     print>>log, "R: %s"%str(R)
            #     LP=sorted(list(P))
            #     LR=sorted(list(R))
            #     Ap=A[:,LP]
            #     ApT_Ap_inv=ModLinAlg.invSVD(np.dot(Ap.T,Ap))
            #     sp=np.dot(ApT_Ap_inv,np.dot(Ap.T,y))
            #     s[LP,0]=sp[:,0]
            #     print>>log, "P: %s, s: %s"%(str(P),str(s.ravel()))
            #     while np.min(sp)<=0.:
            #         alpha=np.min([x[i,0]/(x[i,0]-s[i,0]) for i in LP if s[i,0]<0])
            #         print>>log, "  Alpha= %f"%alpha
            #         x=x+alpha*(s-x)
            #         print>>log, "  x= %s"%str(x)
            #         for j in LP:
            #             if x[j,0]==0: 
            #                 R.add(j)
            #                 P.remove(j)
            #         LP=sorted(list(P))
            #         LR=sorted(list(R))
            #         Ap=A[:,LP]
            #         ApT_Ap_inv=ModLinAlg.invSVD(np.dot(Ap.T,Ap))
            #         sp=np.dot(ApT_Ap_inv,np.dot(Ap.T,y))
            #         print>>log, "  sp= %s"%str(sp)
            #         s[LP,0]=sp[:,0]
            #         s[LR,0]=0.
            #     x=s
            #     w=np.dot(A.T,y-np.dot(A,x))
            #     print>>log, "x: %s, w: %s"%(str(x.ravel()),str(w.ravel()))
                    
                    
            # Sol=x
            # LocalSM=np.sum(self.CubePSFScales*Sol.reshape((Sol.size,1,1,1)),axis=0)
            


        nch,nx,ny = LocalSM.shape
        LocalSM = LocalSM.reshape((nch,1,nx,ny))
#        LocalSM *= np.sqrt(JonesNorm)
        numexpr.evaluate('LocalSM*sqrt(JonesNorm)',out=LocalSM)

        # print self.AlphaVec,Sol
        # print "alpha",np.sum(self.AlphaVec.ravel()*Sol.ravel())/np.sum(Sol)

        FpolMean=1.

        self.ModelMachine.AppendComponentToDictStacked((xc,yc),FpolMean,Sol)

        BM=DicoBasisMatrix["BM"]
        #print "MaxSM=",np.max(LocalSM)
        ConvSM=((np.dot(BM,Sol)).ravel())#*(WVecPSF.ravel())
        #Sol/=self.Bias

        #Sol[-self.NFreqBands::]=0
        # Sol=np.dot(BM.T,WVecPSF*dirtyVec)
        # Sol[Sol<0]=0

        #print Sol.flatten()

        T.timeit("2")
        
        #print>>log,( "Sol:",Sol)
        #print>>log, ("MaxLSM:",np.max(LocalSM))
        T.timeit("3")

        #print Sol

        T.timeit("4")

        nch,npol,_,_=self._Dirty.shape
        ConvSM=ConvSM.reshape((nch,npol,nxp,nyp))

        nFunc,_=BM.T.shape
        BBM=BM.T.reshape((nFunc,nch,npol,nxp,nyp))
        

        

        #print np.sum(Sol.flatten()*self.Alpha.flatten())/np.sum(Sol.flatten())

        T.timeit("5")



#         ##############################
#         #ConvSM*=FpolMean.ravel()[0]
#         import pylab

#         dv=1
# #        for iFunc in range(nFunc):#[0]:#range(nFunc):
#         for iFunc in [0]:#range(nFunc):

#             pylab.clf()
#             iplot=1
#             nxp,nyp=3,3
            
#             FF=ConvSM[:,0]#BBM[iFunc,:,0]
#             Resid=(dirtyNormIm[:,0]-FF[:])
#             #FF=BBM[iFunc,:,0]
#             Resid*=DicoBasisMatrix["WeightFunction"][:,0,:,:]
#             vmin,vmax=np.min([dirtyNormIm[0,0],ConvSM[0,0],dirtyNormIm[0,0]-FF[0]]),np.max([dirtyNormIm[0,0],ConvSM[0,0],dirtyNormIm[0,0]-FF[0]])

#             ax=pylab.subplot(nxp,nyp,iplot); iplot+=1
#             pylab.imshow(dirtyNormIm[0,0],interpolation="nearest",vmin=vmin,vmax=vmax)#)
#             pylab.colorbar()
#             pylab.subplot(nxp,nyp,iplot); iplot+=1
#             #pylab.imshow(dirtyNormIm[1,0],interpolation="nearest",vmin=vmin,vmax=vmax)#)
#             pylab.subplot(nxp,nyp,iplot); iplot+=1
#             #pylab.imshow(dirtyNormIm[2,0],interpolation="nearest",vmin=vmin,vmax=vmax)#)
            
#             pylab.subplot(nxp,nyp,iplot,sharex=ax,sharey=ax); iplot+=1
#             pylab.imshow(ConvSM[0,0],interpolation="nearest",vmin=vmin,vmax=vmax)#)#,vmin=-0.5,vmax=0.5)
#             pylab.colorbar()
#             pylab.subplot(nxp,nyp,iplot); iplot+=1
#             #pylab.imshow(ConvSM[1,0],interpolation="nearest",vmin=vmin,vmax=vmax)#)#,vmin=-0.5,vmax=0.5)
#             pylab.subplot(nxp,nyp,iplot); iplot+=1
#             #pylab.imshow(ConvSM[2,0],interpolation="nearest",vmin=vmin,vmax=vmax)#)#,vmin=-0.5,vmax=0.5)

#             pylab.subplot(nxp,nyp,iplot,sharex=ax,sharey=ax); iplot+=1
#             pylab.imshow(Resid[0],interpolation="nearest")#,vmin=vmin,vmax=vmax)#,vmin=-0.5,vmax=0.5)
#             pylab.colorbar()
#             pylab.subplot(nxp,nyp,iplot); iplot+=1
#             #pylab.imshow(Resid[1],interpolation="nearest")#,vmin=vmin,vmax=vmax)#,vmin=-0.5,vmax=0.5)
#             pylab.colorbar()
#             pylab.subplot(nxp,nyp,iplot); iplot+=1
#             #pylab.imshow(Resid[2],interpolation="nearest")#,vmin=vmin,vmax=vmax)#,vmin=-0.5,vmax=0.5)
#             pylab.colorbar()


#             # pylab.subplot(3,2,iplot)
#             # pylab.imshow(BBM[iFunc,0,0],interpolation="nearest",vmin=-0.5,vmax=0.5)
#             # #pylab.colorbar()
#             # pylab.subplot(3,2,iplot)
#             # pylab.imshow(BBM[iFunc,1,0],interpolation="nearest",vmin=-0.5,vmax=0.5)
#             # #pylab.colorbar()

#             #pylab.colorbar()
            
#             pylab.draw()
#             pylab.show(False)
#             pylab.pause(0.1)

#             # stop

        return LocalSM

            

#################

