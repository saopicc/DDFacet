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
import pylab
from DDFacet.Other import MyLogger
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassImageDeconvMachine")
from DDFacet.Array import NpParallel
from DDFacet.Array import NpShared
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
from DDFacet.Imager import ClassMultiScaleMachine
from pyrap.images import image
from DDFacet.Imager.ClassPSFServer import ClassPSFServer
import sys
from DDFacet.Other.progressbar import ProgressBar
from DDFacet.Imager import ClassGainMachine
from SkyModel.PSourceExtract import ClassIslands
from SkyModel.PSourceExtract import ClassIncreaseIsland

from DDFacet.Other import MyPickle

import multiprocessing
import time

MyLogger.setSilent("ClassArrayMethodGA")
MyLogger.setSilent("ClassIsland")



class ClassImageDeconvMachine():
    def __init__(self,Gain=0.3,
                 MaxMinorIter=100,NCPU=6,
                 CycleFactor=2.5,FluxThreshold=None,RMSFactor=3,PeakFactor=0,
                 GD=None,SearchMaxAbs=1,CleanMaskImage=None,IdSharedMem="",ModelMachine=None,
                 **kw    # absorb any unknown keywords arguments into this
                 ):
        #self.im=CasaImage

        self.SearchMaxAbs=SearchMaxAbs
        self.ModelImage=None
        self.MaxMinorIter=MaxMinorIter
        self.NCPU=NCPU
        self.Chi2Thr=10000
        self.MaskArray=None
        self.GD=GD
        self.IdSharedMem=IdSharedMem
        self.SubPSF=None
        self.MultiFreqMode=(self.GD["Freq"]["NBand"]>1)
        self.FluxThreshold = FluxThreshold 
        self.CycleFactor = CycleFactor
        self.RMSFactor = RMSFactor
        self.PeakFactor = PeakFactor
        self.GainMachine=ClassGainMachine.ClassGainMachine(GainMin=Gain)
        # reset overall iteration counter
        self._niter = 0
        self.PSFCross=None

        self.IslandDeconvMode=self.GD["SSD"]["IslandDeconvMode"]  # "GA" or "Moresane" or "Sasir"
        if ModelMachine is None:
            if self.IslandDeconvMode == "GA":
                from DDFacet.Imager.GA import ClassModelMachineGA
                self.ModelMachine = ClassModelMachineGA.ClassModelMachine(self.GD, GainMachine=self.GainMachine)
            elif self.IslandDeconvMode == "Moresane":
                from DDFacet.Imager.MORESANE import ClassModelMachineMORESANE
                self.ModelMachine = ClassModelMachineMORESANE.ClassModelMachine(self.GD, GainMachine=self.GainMachine)
            elif self.IslandDeconvMode == "Sasir":
                raise NotImplementedError("ClassModelMachineSASIR is not implemented")
        else:
            # Trusting the user to pass correct ModelMachine for deconv algo
            self.ModelMachine = ModelMachine

        if CleanMaskImage is not None:
            print>>log, "Reading mask image: %s"%CleanMaskImage
            MaskArray=image(CleanMaskImage).getdata()
            nch,npol,_,_=MaskArray.shape
            self._MaskArray=np.zeros(MaskArray.shape,np.bool8)
            for ch in range(nch):
                for pol in range(npol):
                    self._MaskArray[ch,pol,:,:]=np.bool8(1-MaskArray[ch,pol].T[::-1].copy())[:,:]
            self.MaskArray=self._MaskArray[0]
            self.IslandArray=np.zeros_like(self._MaskArray)
            self.IslandHasBeenDone=np.zeros_like(self._MaskArray)
        else:
            print>>log, "You have to provide a mask image for SSD deconvolution"




    def GiveModelImage(self,*args): return self.ModelMachine.GiveModelImage(*args)

    def setSideLobeLevel(self,SideLobeLevel,OffsetSideLobe):
        self.SideLobeLevel=SideLobeLevel
        self.OffsetSideLobe=OffsetSideLobe
        

    def SetPSF(self,DicoVariablePSF):
        self.PSFServer=ClassPSFServer(self.GD)
        DicoVariablePSF["CubeVariablePSF"]=NpShared.ToShared("%s.CubeVariablePSF"%self.IdSharedMem,DicoVariablePSF["CubeVariablePSF"])
        self.PSFServer.setDicoVariablePSF(DicoVariablePSF)
        #self.DicoPSF=DicoPSF
        self.DicoVariablePSF=DicoVariablePSF
        #self.NChannels=self.DicoDirty["NChannels"]
        self.ModelMachine.setRefFreq(self.PSFServer.RefFreq) #,self.PSFServer.AllFreqs)

    def Init(self,**kwargs):
        self.SetPSF(kwargs["PSFVar"])
        self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
        self.InitSSD()

    def InitSSD(self):
        pass

    def AdaptArrayShape(self,A,Nout):
        nch,npol,Nin,_=A.shape
        if Nin==Nout: 
            return A
        elif Nin<Nout:
            off=(Nout-Nin)/2
            B=np.zeros((nch,npol,Nout,Nout),A.dtype)
            B[:,:,off:off+Nin,off:off+Nin]=A
            return B


    def SetDirty(self,DicoDirty):
        DicoDirty["ImageCube"]=NpShared.ToShared("%s.Dirty.ImagData"%self.IdSharedMem,DicoDirty["ImageCube"])
        DicoDirty["MeanImage"]=NpShared.ToShared("%s.Dirty.MeanImage"%self.IdSharedMem,DicoDirty["MeanImage"])
        self.DicoDirty=DicoDirty
        self._Dirty=self.DicoDirty["ImageCube"]
        self._MeanDirty=self.DicoDirty["MeanImage"]
        NPSF=self.PSFServer.NPSF
        _,_,NDirty,_=self._Dirty.shape

        off=(NPSF-NDirty)/2

        _,_,NMask,_=self._MaskArray.shape
        if NMask!=NDirty:
            print>>log,"Adapt mask shape"
            self._MaskArray=self.AdaptArrayShape(self._MaskArray,NDirty)
            self.MaskArray=self._MaskArray[0]
            self.IslandArray=np.zeros_like(self._MaskArray)
            self.IslandHasBeenDone=np.zeros_like(self._MaskArray)

        self.DirtyExtent=(off,off+NDirty,off,off+NDirty)

        if self.ModelImage is None:
            self._ModelImage=np.zeros_like(self._Dirty)
        self.ModelMachine.setModelShape(self._Dirty.shape)
        if self.MaskArray is None:
            self._MaskArray=np.zeros(self._Dirty.shape,dtype=np.bool8)
            self.IslandArray=np.zeros_like(self._MaskArray)
            self.IslandHasBeenDone=np.zeros_like(self._MaskArray)

    def CalcCrossIslandPSF(self,ListIslands):
        print>>log,"  calculating global islands cross-contamination"
        PSF=np.mean(self.PSFServer.DicoVariablePSF["MeanFacetPSF"][:,0],axis=0)#self.PSFServer.DicoVariablePSF["MeanFacetPSF"][0,0]
        
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

        xMean=np.zeros((NIslands,),np.int32)
        yMean=xMean.copy()
        for iIsland in range(NIslands):
            x0,y0=np.array(ListIslands[iIsland]).T
            xc0,yc0=int(np.mean(x0)),int(np.mean(y0))
            xMean[iIsland]=xc0
            yMean[iIsland]=yc0

        PSFCross=np.zeros((NIslands,NIslands),np.float32)
        dx=xMean.reshape((NIslands,1))-xMean.reshape((1,NIslands))+xcPSF
        dy=yMean.reshape((NIslands,1))-yMean.reshape((1,NIslands))+xcPSF
        indPSF=np.arange(NIslands**2)
        Cx=((dx>=0)&(dx<nPSF))
        Cy=((dy>=0)&(dy<nPSF))
        C=(Cx&Cy)
        indPSF_sel=indPSF[C.ravel()]
        indPixPSF=dx.ravel()[C.ravel()]*nPSF+dy.ravel()[C.ravel()]
        PSFCross.flat[indPSF_sel]=np.abs(PSF.flat[indPixPSF.ravel()])



        
        self.PSFCross=PSFCross

    def GiveNearbyIsland(self,DicoIsland,iIsland):
        Th=0.05
        indNearbyIsland=np.where((self.PSFCross[iIsland])>Th)[0]


        #Th=0.3
        #Flux=self.CrossFluxContrib[iIsland,iIsland]
        #C0=(self.CrossFluxContrib[iIsland] > Flux*Th)
        #indNearbyIsland=np.where(C0)[0]

        ii=0
        #print DicoIsland.keys()
        #print>>log,"Looking around island #%i"%(iIsland)
        for jIsland in indNearbyIsland:
            #if jIsland in DicoIsland.keys():
            try:
                Island=DicoIsland[jIsland]
                #print>>log,"  merging island #%i -> #%i"%(jIsland,iIsland)
                del(DicoIsland[jIsland])
                SubIslands=self.GiveNearbyIsland(DicoIsland,jIsland)
                if SubIslands is not None:
                    Island+=SubIslands
                return Island
            except:
                continue


        #print>>log,"  could not find island #%i"%(iIsland)
                
        return None

# JG Mod
    def ToSquareIslands(self,ListIslands):
        NIslands=len(ListIslands)
        DicoSquareIslands={}
        Dirty=self.DicoDirty["MeanImage"]

        for iIsland in range(NIslands):
            x, y = np.array(ListIslands[iIsland]).T
            minx,maxx=x.min(),x.max()
            miny,maxy=y.min(),y.max()
            nxisl,nyisl=maxx-minx+1,maxy-miny+1                   # size of the smallest square around the island
            npix=np.max([nxisl,nyisl])

            if npix %2 == 0:
                npix+=1

            xc,yc=np.round((maxx+minx)/2),np.round((maxy+miny)/2) # compute island centre from minx,maxx,miny,maxy in the

            locxc,locyc=npix/2,npix/2                         # center of the square around island

            SquareIsland=np.zeros((npix,npix),np.float32)
            SquareIsland[0:npix,0:npix]=Dirty[0,0,xc-(npix-1)/2:xc+(npix-1)/2+1,yc-(npix-1)/2:yc+(npix-1)/2+1]

            SquareIsland_Mask=np.zeros((npix,npix),np.float32)
            SquareIsland_Mask[locxc+(x-xc),locyc+(y-yc)]=1 # put 1 on the Island
            #SquareIsland[locxc+(minx-xc):locxc+(maxx-xc)+1,locyc+(miny-yc):locyc+(maxy-yc)+1]=Dirty[0,0,minx:maxx+1,miny:maxy+1]

            DicoSquareIslands[iIsland]={"IslandCenter":(xc,yc), "IslandSquareData":SquareIsland, "IslandSquareMask":SquareIsland_Mask}

        return DicoSquareIslands

    def SquareIslandtoIsland(self, Model,ThisSquarePixList,ThisPixList):
        ### Build ThisPixList from Model, in the reference frame of the Dirty

        xc,yc = ThisSquarePixList['IslandCenter']  # island center in original dirty
        ListSquarePix_Data = ThisSquarePixList['IslandSquareData']  # square image of the dirty around Island center
        ListSquarePix_Mask = ThisSquarePixList['IslandSquareMask']  # Corresponding square mask image

        NIslandPix=len(ThisPixList)

        Mod_x,Mod_y=Model.shape
        SquarePix_x,SquarePix_y=ListSquarePix_Data.shape

        if Mod_x != SquarePix_x or Mod_y != SquarePix_y:
            raise NameError('Mismatch between output Model image dims and original Square image dims. Please check if the even to uneven correction worked.')

        FluxV = []
        NewThisPixList = []
        for tmpcoor in ThisPixList:
            currentx = tmpcoor[0]
            currenty = tmpcoor[1]
            x_loc_coor = (currentx - xc) + SquarePix_x / 2  # coordinates in the small Model image
            y_loc_coor = (currenty - yc) + SquarePix_y / 2  # coordinates in the small Model image
            if ListSquarePix_Mask[x_loc_coor, y_loc_coor] == 1:  # if it is not masked (e.g. part of the island)
                FluxV.append(ListSquarePix_Data[x_loc_coor, y_loc_coor])
                NewThisPixList.append([currentx, currenty])

        return np.array(FluxV), np.array(NewThisPixList)

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
        

        NDone=0
        NJobs=NIslands
        pBAR= ProgressBar(Title=" Group islands")
        pBAR.disable()
        pBAR.render(0, '%4i/%i' % (0,NJobs))

        Th=0.05
        ListIslandMerged=[]
        for iIsland in range(NIslands):
            NDone+=1
            intPercent=int(100*  NDone / float(NJobs))
            pBAR.render(intPercent, '%4i/%i' % (NDone,NJobs))

            ThisIsland=self.GiveNearbyIsland(DicoIsland,iIsland)
            
            # indiIsland=np.where((self.PSFCross[iIsland])>Th)[0]
            # ThisIsland=[]
            # #print "Island #%i: %s"%(iIsland,str(np.abs(self.PSFCross[iIsland])))
            # for jIsland in indiIsland:
            #     if not(jIsland in DicoIsland.keys()): 
            #         #print>>log,"    island #%i not there "%(jIsland)
            #         continue
            #     #print>>log,"  Putting island #%i in #%i"%(jIsland,iIsland)
            #     for iPix in range(len(DicoIsland[jIsland])):
            #         ThisIsland.append(DicoIsland[jIsland][iPix])
            #     del(DicoIsland[jIsland])


            if ThisIsland is not None:
                ListIslandMerged.append(ThisIsland)

        print>>log,"    have grouped %i --> %i islands"%(NIslands, len(ListIslandMerged))

        return ListIslandMerged



    def SearchIslands(self,Threshold):
        print>>log,"Searching Islands"
        Dirty=self.DicoDirty["MeanImage"]
        self.IslandArray[0,0]=(Dirty[0,0]>Threshold)|(self.IslandArray[0,0])
        #MaskImage=(self.IslandArray[0,0])&(np.logical_not(self._MaskArray[0,0]))
        #MaskImage=(np.logical_not(self._MaskArray[0,0]))
        MaskImage=(np.logical_not(self._MaskArray[0,0]))
        Islands=ClassIslands.ClassIslands(Dirty[0,0],MaskImage=MaskImage,
                                          MinPerIsland=0,DeltaXYMin=0)
        Islands.FindAllIslands()

        ListIslands=Islands.LIslands

        print>>log,"  found %i islands"%len(ListIslands)
        dx=self.GD["GAClean"]["NEnlargePars"]
        if dx>0:
            print>>log,"  increase their sizes by %i pixels"%dx
            IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
            for iIsland in range(len(ListIslands)):#self.NIslands):
                ListIslands[iIsland]=IncreaseIslandMachine.IncreaseIsland(ListIslands[iIsland],dx=dx)


        ListIslands=self.CalcCrossIslandFlux(ListIslands)


        # FluxIslands=[]
        # for iIsland in range(len(ListIslands)):
        #     x,y=np.array(ListIslands[iIsland]).T
        #     FluxIslands.append(np.sum(Dirty[0,0,x,y]))
        # ind=np.argsort(np.array(FluxIslands))[::-1]

        # ListIslandsSort=[ListIslands[i] for i in ind]
        

        # ListIslands=self.CalcCrossIslandFlux(ListIslandsSort)
        self.ListIslands=[]

        for iIsland in range(len(ListIslands)):
            x,y=np.array(ListIslands[iIsland]).T
            PixVals=Dirty[0,0,x,y]
            DoThisOne=False

            MaxIsland=np.max(np.abs(PixVals))

            if (MaxIsland>(3.*self.RMS))|(MaxIsland>Threshold):
                self.ListIslands.append(ListIslands[iIsland])

            # ###############################
            # if np.max(np.abs(PixVals))>Threshold:
            #     DoThisOne=True
            #     self.IslandHasBeenDone[0,0,x,y]=1
            # if ((DoThisOne)|self.IslandHasBeenDone[0,0,x[0],y[0]]):
            #     self.ListIslands.append(ListIslands[iIsland])
            # ###############################


        self.NIslands=len(self.ListIslands)
        print>>log,"  selected %i islands [out of %i] with peak flux > %.3g Jy"%(self.NIslands,len(ListIslands),Threshold)
        

        self.ListIslands_Keep=self.ListIslands

        Sz=np.array([len(self.ListIslands[iIsland]) for iIsland in range(self.NIslands)])
        ind=np.argsort(Sz)[::-1]

        ListIslandsOut=[self.ListIslands[ind[i]] for i in ind]
        self.ListIslands=ListIslandsOut

        # MORESANE MOD TO MAKE SQUARE IMAGE AROUND EACH ISLAND
        if self.IslandDeconvMode == "Moresane" or self.IslandDeconvMode == "Sasir":  # convert island to square image to pass to MORESANE & SASIR
            self.ListSquareIslands = self.ToSquareIslands(self.ListIslands)
                

    def setChannel(self,ch=0):
        self.Dirty=self._MeanDirty[ch]
        self.ModelImage=self._ModelImage[ch]
        self.MaskArray=self._MaskArray[ch]


    def GiveThreshold(self,Max):
        return ((self.CycleFactor-1.)/4.*(1.-self.SideLobeLevel)+self.SideLobeLevel)*Max if self.CycleFactor else 0

    def Deconvolve(self, *args, **kwargs):

        if self.GD['SSD']['Parallel'] == True:
            print>>log,"Using Deconvolve Parallel for GA"
            return self.DeconvolveParallel(*args,**kwargs)
        else:
            print>>log,"Using Deconvolve Serial for GA"
            return self.DeconvolveSerial(*args, **kwargs)

    def DeconvolveSerial(self, ch=0):
        """
        Runs minor cycle over image channel 'ch'.
        initMinor is number of minor iteration (keeps continuous count through major iterations)
        Nminor is max number of minor iteration

        Returns tuple of: return_code,continue,updated
        where return_code is a status string;
        continue is True if another cycle should be executed;
        update is True if model has been updated (note that update=False implies continue=False)
        """
        from DDFacet.Imager.GA.ClassEvolveGA import ClassEvolveGA
        from DDFacet.Imager.MORESANE.ClassMoresane import ClassMoresane
        from DDFacet.Imager.SASIR.ClassSasir import ClassSasir
        if self._niter >= self.MaxMinorIter:
            return "MaxIter", False, False

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

        DoAbs=int(self.GD["Deconv"]["AllowNegative"])
        print>>log, "  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter,self.MaxMinorIter,DoAbs)

        NPixStats=1000
        RandomInd=np.int64(np.random.rand(NPixStats)*npix**2)
        RMS=np.std(np.real(self.Dirty.ravel()[RandomInd]))
        self.RMS=RMS

        self.GainMachine.SetRMS(RMS)
        
        Fluxlimit_RMS = self.RMSFactor*RMS

        x,y,MaxDirty=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)
        #MaxDirty=np.max(np.abs(self.Dirty))
        #Fluxlimit_SideLobe=MaxDirty*(1.-self.SideLobeLevel)
        #Fluxlimit_Sidelobe=self.CycleFactor*MaxDirty*(self.SideLobeLevel)
        Fluxlimit_Peak = MaxDirty*self.PeakFactor
        Fluxlimit_Sidelobe = self.GiveThreshold(MaxDirty)

        mm0,mm1=self.Dirty.min(),self.Dirty.max()

        # work out uper threshold
        StopFlux = max(Fluxlimit_Peak, Fluxlimit_RMS, Fluxlimit_Sidelobe, Fluxlimit_Peak, self.FluxThreshold)

        print>>log, "    Dirty image peak flux      = %10.6f Jy [(min, max) = (%.3g, %.3g) Jy]"%(MaxDirty,mm0,mm1)
        print>>log, "      RMS-based threshold      = %10.6f Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor)
        print>>log, "      Sidelobe-based threshold = %10.6f Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor)
        print>>log, "      Peak-based threshold     = %10.6f Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor)
        print>>log, "      Absolute threshold       = %10.6f Jy"%(self.FluxThreshold)
        print>>log, "    Stopping flux              = %10.6f Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty)


        MaxModelInit=np.max(np.abs(self.ModelImage))

        
        # Fact=4
        # self.BookKeepShape=(npix/Fact,npix/Fact)
        # BookKeep=np.zeros(self.BookKeepShape,np.float32)
        # NPixBook,_=self.BookKeepShape
        # FactorBook=float(NPixBook)/npix
        
        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)

        if ThisFlux < StopFlux:
            print>>log, ModColor.Str("    Initial maximum peak %g Jy below threshold, we're done here" % (ThisFlux),col="green" )
            return "FluxThreshold", False, False

        self.SearchIslands(StopFlux)

        for iIsland in range(self.NIslands):
            ThisPixList=self.ListIslands[iIsland]
            if self.IslandDeconvMode == "Moresane" or self.IslandDeconvMode == "Sasir":  # convert island to square image to pass to MORESANE & SASIR
                ThisSquarePixList=self.ListSquareIslands[iIsland]

            print>>log,"  Fitting island #%4.4i with %i pixels"%(iIsland,len(ThisPixList))

            XY=np.array(ThisPixList,dtype=np.float32)
            xm,ym=np.int64(np.mean(np.float32(XY),axis=0))

            FacetID=self.PSFServer.giveFacetID2(xm,ym)
            PSF=self.DicoVariablePSF["CubeVariablePSF"][FacetID]

            # self.DicoVariablePSF["CubeMeanVariablePSF"][FacetID]
            
            # FreqsInfo={"freqs":self.DicoVariablePSF["freqs"],
            #            "WeightChansImages":self.DicoVariablePSF["WeightChansImages"]}

            FreqsInfo=self.PSFServer.DicoMappingDesc


            nchan,npol,_,_=self._Dirty.shape
            JonesNorm=(self.DicoDirty["NormData"][:,:,xm,ym]).reshape((nchan,npol,1,1))
            W=self.DicoDirty["WeightChansImages"]
            JonesNorm=np.sum(JonesNorm*W.reshape((nchan,1,1,1)),axis=0).reshape((1,npol,1,1))


            IslandBestIndiv=self.ModelMachine.GiveIndividual(ThisPixList)

            # COMMENT TO RUN IN DDF, UNCOMMENT TO TEST WITH TESTXXDeconv.py under XX folder
            ################################
            #DicoSave={"Dirty":self._Dirty,
            #          "PSF":PSF,
            #          "FreqsInfo":FreqsInfo,
                      #"DicoMappingDesc":self.PSFServer.DicoMappingDesc,
            #          "ListPixData":ThisPixList,
            #          "ListPixParms":ThisPixList,
            ##          "ListSquarePix": ThisSquarePixList, ## When Moresane or SASIR are used
            #          "IslandBestIndiv":IslandBestIndiv,
            #          "GD":self.GD,
            #          "FacetID":FacetID}
            #print "saving"
            #MyPickle.Save(DicoSave, "SaveTest")
            #print "saving ok"
            #ipdb.set_trace()

            ################################

            nch=nchan
            self.FreqsInfo=FreqsInfo
            WeightMeanJonesBand=self.FreqsInfo["MeanJonesBand"][FacetID].reshape((nch,1,1,1))
            WeightMueller=WeightMeanJonesBand.ravel()
            WeightMuellerSignal=WeightMueller*self.FreqsInfo["WeightChansImages"].ravel()


            if self.IslandDeconvMode=="GA":
                CEv=ClassEvolveGA(self._Dirty,PSF,FreqsInfo,ListPixParms=ThisPixList,
                                  ListPixData=ThisPixList,IslandBestIndiv=IslandBestIndiv,
                                  GD=self.GD,WeightFreqBands=WeightMuellerSignal)
                Model=CEv.main(NGen=100,DoPlot=False)

            if self.IslandDeconvMode=="Moresane" or self.IslandDeconvMode=="Sasir":

                # 0) Load Island info (center and square data)
                ListSquarePix_Center = ThisSquarePixList['IslandCenter']    # island center in original dirty
                ListSquarePix_Data = ThisSquarePixList['IslandSquareData']  # square image of the dirty around Island center
                ListSquarePix_Mask= ThisSquarePixList['IslandSquareMask']   # Corresponding square mask image
                xisland, yisland = ListSquarePix_Data.shape  # size of the square postage stamp around island

                if self.IslandDeconvMode == "Moresane":
                    # 1) Shape PSF and Dirty to have even number of pixels (required by Moresane)
                    # DEAL WITH SQUARE DATA OF ISLAND IF UNEVEN
                    PSF_monochan=np.squeeze(PSF[0,0,:,:])
                    # Single Channel for the moment
                    PSF_monochan_nx,PSF_monochan_ny=PSF_monochan.shape

                    cropped_square_to_even = False
                    if xisland % 2 != 0:
                        #    PSFCrop_even = np.zeros((xisland+1, xisland+1))
                        #    PSFCrop_even[:-1, :-1] = np.squeeze(PSFCrop)
                        Dirty_even = np.zeros((xisland - 1, xisland - 1))
                        Dirty_even[:, :] = ListSquarePix_Data[:-1, :-1]

                        PSF2_monochan=self.PSFServer.CropPSF(PSF_monochan,2*(xisland-1)+1) # PSF uneven cropped to double uneven dirty island
                        cropped_square_to_even = True
                    else:
                        Dirty_even = ListSquarePix_Data
                        PSF2_monochan=self.PSFServer.CropPSF(PSF_monochan,2*(xisland)+1) # PSF uneven cropped to double even dirty island

                    PSF2_monochan_nx, PSF2_monochan_ny = PSF2_monochan.shape

                    # DEAL WITH PSF IF UNEVEN (WILL ALWAYS BE UNEVEN EXCEPT IN PYMORESANE)
                    if PSF2_monochan_nx % 2 != 0:
                        PSF2_monochan_even = np.zeros((PSF2_monochan_nx-1, PSF2_monochan_nx-1))
                        PSF2_monochan_even = PSF2_monochan[:-1,:-1]
                    else:
                        PSF2_monochan_even = PSF2_monochan


                    # 2) Run the actual MinorCycle algo
                    #Moresane = ClassMoresane(Dirty_even, PSF_monochan_even, DictMoresaneParms, GD=self.GD)
                    Moresane= ClassMoresane(Dirty_even, PSF2_monochan_even, self.GD['MORESANE'], GD=self.GD)
                    Model_Square,Residuals=Moresane.main()

                    # 3) Apply Island mask to model to get rid of regions outside the island.
                    if cropped_square_to_even: # then restore the model to its original uneven dimension
                        Model_Square_uneven = np.zeros((xisland,xisland))
                        Model_Square_uneven[:-1, :-1] = Model_Square
                        Model_Square = Model_Square_uneven

                    Model_Square *= ListSquarePix_Mask # masking outside the island

                    # 4) Convert back to Island format ( "S" and ThisPixList )
                    NewModel, NewThisPixList=self.SquareIslandtoIsland(Model_Square, ThisSquarePixList, ThisPixList)

                    Model = NewModel
                    ThisPixList = NewThisPixList

                if self.IslandDeconvMode == "Sasir":
                    # DO SASIR STUFF
                    # INCOMPLETE

                    Sasir = ClassSasir(Dirty, PSF, self.GD['SASIR'], GD=self.GD)
                    Model_Square, Residuals = Sasir.main()

                    # INCOMPLETE

            # Common command for every MinorCycle deconvolution algo
            self.ModelMachine.AppendIsland(ThisPixList, Model)

        return "MaxIter", True, True   # stop deconvolution but do update model





    def DeconvolveParallel(self, ch=0):
        if self._niter >= self.MaxMinorIter:
            return "MaxIter", False, False

        self.setChannel(ch)

        _,npix,_=self.Dirty.shape
        xc=(npix)/2

        npol,_,_=self.Dirty.shape

        m0,m1=self.Dirty[0].min(),self.Dirty[0].max()

        DoAbs=int(self.GD["Deconv"]["AllowNegative"])
        print>>log, "  Running minor cycle [MinorIter = %i/%i, SearchMaxAbs = %i]"%(self._niter,self.MaxMinorIter,DoAbs)

        NPixStats=1000
        RandomInd=np.int64(np.random.rand(NPixStats)*npix**2)
        RMS=np.std(np.real(self.Dirty.ravel()[RandomInd]))
        self.RMS=RMS

        self.GainMachine.SetRMS(RMS)
        
        Fluxlimit_RMS = self.RMSFactor*RMS

        x,y,MaxDirty=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)
        #MaxDirty=np.max(np.abs(self.Dirty))
        #Fluxlimit_SideLobe=MaxDirty*(1.-self.SideLobeLevel)
        #Fluxlimit_Sidelobe=self.CycleFactor*MaxDirty*(self.SideLobeLevel)
        Fluxlimit_Peak = MaxDirty*self.PeakFactor
        Fluxlimit_Sidelobe = self.GiveThreshold(MaxDirty)

        mm0,mm1=self.Dirty.min(),self.Dirty.max()

        # work out uper threshold
        StopFlux = max(Fluxlimit_Peak, Fluxlimit_RMS, Fluxlimit_Sidelobe, Fluxlimit_Peak, self.FluxThreshold)

        print>>log, "    Dirty image peak flux      = %10.6f Jy [(min, max) = (%.3g, %.3g) Jy]"%(MaxDirty,mm0,mm1)
        print>>log, "      RMS-based threshold      = %10.6f Jy [rms = %.3g Jy; RMS factor %.1f]"%(Fluxlimit_RMS, RMS, self.RMSFactor)
        print>>log, "      Sidelobe-based threshold = %10.6f Jy [sidelobe  = %.3f of peak; cycle factor %.1f]"%(Fluxlimit_Sidelobe,self.SideLobeLevel,self.CycleFactor)
        print>>log, "      Peak-based threshold     = %10.6f Jy [%.3f of peak]"%(Fluxlimit_Peak,self.PeakFactor)
        print>>log, "      Absolute threshold       = %10.6f Jy"%(self.FluxThreshold)
        print>>log, "    Stopping flux              = %10.6f Jy [%.3f of peak ]"%(StopFlux,StopFlux/MaxDirty)


        MaxModelInit=np.max(np.abs(self.ModelImage))

        
        # Fact=4
        # self.BookKeepShape=(npix/Fact,npix/Fact)
        # BookKeep=np.zeros(self.BookKeepShape,np.float32)
        # NPixBook,_=self.BookKeepShape
        # FactorBook=float(NPixBook)/npix
        
        T=ClassTimeIt.ClassTimeIt()
        T.disable()

        x,y,ThisFlux=NpParallel.A_whereMax(self.Dirty,NCPU=self.NCPU,DoAbs=DoAbs,Mask=self.MaskArray)

        if ThisFlux < StopFlux:
            print>>log, ModColor.Str("    Initial maximum peak %g Jy below threshold, we're done here" % (ThisFlux),col="green" )
            return "FluxThreshold", False, False

        self.SearchIslands(StopFlux)
        


        # ================== Parallel part
        NCPU=self.NCPU
        work_queue = multiprocessing.Queue()


        ListBestIndiv=[]

        NJobs=self.NIslands
        T=ClassTimeIt.ClassTimeIt("    ")
        T.disable()
        for iIsland in range(self.NIslands):
            # print "%i/%i"%(iIsland,self.NIslands)
            ThisPixList=self.ListIslands[iIsland]
            XY=np.array(ThisPixList,dtype=np.float32)
            xm,ym=np.mean(np.float32(XY),axis=0)
            T.timeit("xm,ym")
            nchan,npol,_,_=self._Dirty.shape
            JonesNorm=(self.DicoDirty["NormData"][:,:,xm,ym]).reshape((nchan,npol,1,1))
            W=self.DicoDirty["WeightChansImages"]
            JonesNorm=np.sum(JonesNorm*W.reshape((nchan,1,1,1)),axis=0).reshape((1,npol,1,1))
            T.timeit("JonesNorm")

            IslandBestIndiv=self.ModelMachine.GiveIndividual(ThisPixList)
            T.timeit("GiveIndividual")
            ListBestIndiv.append(IslandBestIndiv)
            FacetID=self.PSFServer.giveFacetID2(xm,ym)
            T.timeit("FacetID")

            DicoOrder={"iIsland":iIsland,
                       "FacetID":FacetID,
                       "JonesNorm":JonesNorm}
            
            ListOrder=[iIsland,FacetID,JonesNorm.flat[0]]


            work_queue.put(ListOrder)
            T.timeit("Put")
            
        SharedListIsland="%s.ListIslands"%(self.IdSharedMem)
        ListArrayIslands=[np.array(self.ListIslands[iIsland]) for iIsland in range(self.NIslands)]
        NpShared.PackListArray(SharedListIsland,ListArrayIslands)
        T.timeit("Pack0")
        SharedBestIndiv="%s.ListBestIndiv"%(self.IdSharedMem)
        NpShared.PackListArray(SharedBestIndiv,ListBestIndiv)
        T.timeit("Pack1")
        

        workerlist=[]

        # List_Result_queue=[]
        # for ii in range(NCPU):
        #     List_Result_queue.append(multiprocessing.JoinableQueue())


        result_queue=multiprocessing.Queue()


        for ii in range(NCPU):
            W=WorkerDeconvIsland(work_queue, 
                                 result_queue,
                                 # List_Result_queue[ii],
                                 self.GD,
                                 IdSharedMem=self.IdSharedMem,
                                 FreqsInfo=self.PSFServer.DicoMappingDesc)
            workerlist.append(W)
            workerlist[ii].start()


        print>>log, "Evolving %i generations of %i sourcekin"%(self.GD["GAClean"]["NMaxGen"],self.GD["GAClean"]["NSourceKin"])
        pBAR= ProgressBar(Title=" Evolve pop.")
        #pBAR.disable()
        pBAR.render(0, '%4i/%i' % (0,NJobs))

        iResult=0
        while iResult < NJobs:
            DicoResult=None
            # for result_queue in List_Result_queue:
            #     if result_queue.qsize()!=0:
            #         try:
            #             DicoResult=result_queue.get_nowait()
                        
            #             break
            #         except:
                        
            #             pass
            #         #DicoResult=result_queue.get()
            if result_queue.qsize()!=0:
                try:
                    DicoResult=result_queue.get_nowait()
                except:
                    pass
                    #DicoResult=result_queue.get()


            if DicoResult is None:
                time.sleep(0.05)
                continue

            iResult+=1
            NDone=iResult
            intPercent=int(100*  NDone / float(NJobs))
            pBAR.render(intPercent, '%4i/%i' % (NDone,NJobs))

            if DicoResult["Success"]:
                iIsland=DicoResult["iIsland"]
                ThisPixList=self.ListIslands[iIsland]
                SharedIslandName="%s.FitIsland_%5.5i"%(self.IdSharedMem,iIsland)
                Model=NpShared.GiveArray(SharedIslandName)
                self.ModelMachine.AppendIsland(ThisPixList,Model)
                NpShared.DelArray(SharedIslandName)



        for ii in range(NCPU):
            workerlist[ii].shutdown()
            workerlist[ii].terminate()
            workerlist[ii].join()


        return "MaxIter", True, True   # stop deconvolution but do update model

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

    ###################################################################################
    ###################################################################################
    
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
        N0=self.Dirty.shape[-1]
        N1=LocalSM.shape[-1]
        Aedge,Bedge=self.GiveEdges((xc,yc),N0,(N1/2,N1/2),N1)
        factor=-1.
        nch,npol,nx,ny=LocalSM.shape
        x0d,x1d,y0d,y1d=Aedge
        x0p,x1p,y0p,y1p=Bedge
        self._Dirty[:,:,x0d:x1d,y0d:y1d]-=LocalSM[:,:,x0p:x1p,y0p:y1p]
        W=np.float32(self.DicoDirty["WeightChansImages"])
        self._MeanDirty[0,:,x0d:x1d,y0d:y1d]-=np.sum(LocalSM[:,:,x0p:x1p,y0p:y1p]*W.reshape((W.size,1,1,1)),axis=0)


#===============================================
#===============================================
#===============================================
#===============================================

class WorkerDeconvIsland(multiprocessing.Process):
    def __init__(self,
                 work_queue,
                 result_queue,
                 GD,
                 IdSharedMem=None,
                 FreqsInfo=None,
                 MultiFreqMode=False):
        multiprocessing.Process.__init__(self)
        self.MultiFreqMode=MultiFreqMode
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.kill_received = False
        self.exit = multiprocessing.Event()
        self.GD=GD
        self.IdSharedMem=IdSharedMem
        self.FreqsInfo=FreqsInfo
        self.CubeVariablePSF=NpShared.GiveArray("%s.CubeVariablePSF"%self.IdSharedMem)
        self._Dirty=NpShared.GiveArray("%s.Dirty.ImagData"%self.IdSharedMem)
        #self.WeightFreqBands=WeightFreqBands

    def shutdown(self):
        self.exit.set()

 
    def run(self):
        from DDFacet.Imager.GA.ClassEvolveGA import ClassEvolveGA
        while not self.kill_received:
            #gc.enable()
            try:
                iIsland,FacetID,JonesNorm = self.work_queue.get()
            except:
                break


            # iIsland=DicoOrder["iIsland"]
            # FacetID=DicoOrder["FacetID"]
            
            # JonesNorm=DicoOrder["JonesNorm"]

            SharedListIsland="%s.ListIslands"%(self.IdSharedMem)
            ThisPixList=NpShared.UnPackListArray(SharedListIsland)[iIsland].tolist()

            SharedBestIndiv="%s.ListBestIndiv"%(self.IdSharedMem)
            IslandBestIndiv=NpShared.UnPackListArray(SharedBestIndiv)[iIsland]
            
            PSF=self.CubeVariablePSF[FacetID]
            NGen=self.GD["GAClean"]["NMaxGen"]
            NIndiv=self.GD["GAClean"]["NSourceKin"]

            ListPixParms=ThisPixList
            ListPixData=ThisPixList
            dx=self.GD["GAClean"]["NEnlargeData"]
            if dx>0:
                IncreaseIslandMachine=ClassIncreaseIsland.ClassIncreaseIsland()
                ListPixData=IncreaseIslandMachine.IncreaseIsland(ListPixData,dx=dx)


            # if island lies inside image
            try:
                nch=self.FreqsInfo["MeanJonesBand"][FacetID].size
                WeightMeanJonesBand=self.FreqsInfo["MeanJonesBand"][FacetID].reshape((nch,1,1,1))
                WeightMueller=WeightMeanJonesBand.ravel()
                WeightMuellerSignal=np.sqrt(WeightMueller*self.FreqsInfo["WeightChansImages"].ravel())

                CEv=ClassEvolveGA(self._Dirty,
                                  PSF,
                                  self.FreqsInfo,
                                  ListPixParms=ListPixParms,
                                  ListPixData=ListPixData,
                                  IslandBestIndiv=IslandBestIndiv,#*np.sqrt(JonesNorm),
                                  GD=self.GD,
                                  WeightFreqBands=WeightMuellerSignal)
                Model=CEv.main(NGen=NGen,NIndiv=NIndiv,DoPlot=False)
            
                Model=np.array(Model).copy()#/np.sqrt(JonesNorm)
                #Model*=CEv.ArrayMethodsMachine.Gain
                
                del(CEv)
                
                NpShared.ToShared("%s.FitIsland_%5.5i"%(self.IdSharedMem,iIsland),Model)
                
                #print "Current process: %s [%s left]"%(str(multiprocessing.current_process()),str(self.work_queue.qsize()))
                
                self.result_queue.put({"Success":True,"iIsland":iIsland})
            except Exception,e:
                print "Exception : %s"%str(e)

                self.result_queue.put({"Success":False})

