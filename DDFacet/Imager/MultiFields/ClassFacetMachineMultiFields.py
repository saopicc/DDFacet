from DDFacet.Imager.ClassFacetMachineTessel import ClassFacetMachineTessel as ClassFacetMachine
import csv
import numpy as np
import copy
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms
from astropy.coordinates import SkyCoord
import astropy.units as u
from DDFacet.Other.AsciiReader import readMultiFieldFile


class DictImages(dict):
    def __init__(self,Dict=None):
        if Dict is None:
            Dict={}
        self.data = Dict
    
    def __getitem__(self, X):
        if isinstance(X,tuple) and len(X)==2:
            (key, subkey)=X
            if key == slice(None):
                return [self.data[k][subkey] for k in sorted(self.data.keys())]
            else:
                return self.data[key, subkey]
        else:
            key=X
            if key == slice(None):
                return [self.data[k] for k in sorted(self.data.keys())]
            else:
                return self.data[key]
            
            
        
    def __setitem__(self, key, value):
        self.data[key] = value

    def save(self,path):
        for k in sorted(self.data.keys()):
            ThisPath="%s_Field%i"%(path,k)
            self.data[k].save(ThisPath)


    def reload(self):
        for k in sorted(self.data.keys()):
            self.data[k].reload()

    def delete(self):
        for k in sorted(self.data.keys()):
            self.data[k].delete()
        
        
            
class ClassFacetMachineMultiFields():
    def __init__(self,
                 VS,
                 GD,
                 Precision="S",
                 PolMode=["I"],
                 Sols=None,
                 PointingID=0,
                 DoPSF=False,
                 Oversize=1,   # factor by which image is oversized
                 custom_id=None,
                 FieldID=None):
        custom_id0=custom_id
        self.GD=GD
        self.VS=VS
        # Oleg's "new" interface: set up which output images will be generated
        # --SaveImages abc means save defaults plus abc
        # --SaveOnly abc means only save abc
        # --SaveImages all means save all
        saveimages = self.GD["Output"]["Also"]
        saveonly = self.GD["Output"]["Images"]
        savecubes = self.GD["Output"]["Cubes"]
        allchars = set([chr(x) for x in range(128)])
        if saveimages.lower() == "all" or saveonly.lower() == "all":
            self._saveims = allchars
        else:
            self._saveims = set(saveimages) | set(saveonly)
        self._savecubes = allchars if savecubes.lower() == "all" else set(savecubes)

        self.BaseName =  self.GD["Output"]["Name"]       

        
        self.ListDicoFields=[]
        if self.GD["Image"]["MultiFieldFile"] is None:
            ThisDicoField={"GD":GD,
                           "FieldID":None,
                           "ra0dec0":None}
            self.ListDicoFields.append(ThisDicoField)
        else:
            Fields=readMultiFieldFile(self.GD["Image"]["MultiFieldFile"])
            for iField,ThisField in enumerate(Fields):
                coords = SkyCoord(ra=ThisField["ra"],
                                  dec=ThisField["dec"],
                                  unit=(u.hourangle, u.deg))
                ras=rad2hmsdms(coords.ra.rad,Type="ra").replace(" ",":")
                decs=rad2hmsdms(coords.dec.rad,Type="dec").replace(" ",":")
                NPix=int(ThisField["NPix"])
                ThisGD=copy.deepcopy(self.GD)
                ThisGD["Image"]["NPix"]=NPix
                ThisDicoField={"GD":ThisGD,
                               "FieldID":iField,
                               "ra0dec0":(coords.ra.rad,coords.dec.rad)}
                self.ListDicoFields.append(ThisDicoField)
            self.NFields=len(self.ListDicoFields)

        self.LFM=[]

        # MultiField mode
        for iField,DicoField in enumerate(self.ListDicoFields):
            ra0dec0=DicoField["ra0dec0"]
            FieldID=DicoField["FieldID"]
            ThisGD=DicoField["GD"]
            BaseName=ThisGD["Output"]["Name"]
            if FieldID is not None:
                custom_id="%s_Field%i"%(custom_id0,FieldID)
                BaseName="%s_Field%i"%(BaseName,FieldID)
                ThisGD["Output"]["Name"]=BaseName
                CounterName=" F#%i"%FieldID
                
            FM=ClassFacetMachine(VS,
                                 ThisGD,
                                 Precision=Precision,
                                 PolMode=PolMode,
                                 Sols=Sols,
                                 PointingID=PointingID,
                                 DoPSF=DoPSF,
                                 Oversize=Oversize,   # factor by which image is oversized
                                 custom_id=custom_id,
                                 CounterName=CounterName)
            self.LFM.append(FM)
        self.DoSmoothBeam=(self.GD["Beam"]["Smooth"] and self.GD["Beam"]["Model"])
            
    def GiveMainFacetOptions(self,GD):
        MainFacetOptions=GD["Image"].copy()
        MainFacetOptions.update(GD["CF"].copy())
        MainFacetOptions.update(GD["Image"].copy())
        MainFacetOptions.update(GD["Facets"].copy())
        MainFacetOptions.update(GD["RIME"].copy())
        MainFacetOptions.update(GD["Weight"].copy())
        del(MainFacetOptions['Precision'],
            MainFacetOptions['PolMode'],MainFacetOptions['Mode'],MainFacetOptions['Robust'])
        return MainFacetOptions
    
    def FitPSF(self):
        for FM in self.LFM:
            FM.FitPSF(*args,**kwargs)
            
    def Init(self,*args,**kwargs):
        for FM in self.LFM:
            FM.Init(*args,**kwargs)
        self.FullImShape = self.LFM[0].OutImShape
        self.OutImShape = self.LFM[0].OutImShape
        self.PaddedGridShape = self.PaddedFacetShape = self.LFM[0].PaddedGridShape
        self.FacetShape = self.LFM[0].FacetShape
        self.CellSizeRad_x,self.CellSizeRad_y=self.CellSizeRad = self.LFM[0].CellSizeRad
        self.FacetDirCat=np.concatenate([FM.FacetDirCat for FM in self.LFM])
        self.FacetDirCat=self.FacetDirCat.view(np.recarray)
        
        self.DicoImager={}
        for iFM,FM in enumerate(self.LFM):
            self.DicoImager["Field_%i"%iFM]=FM.DicoImager
            
            
    def appendMainField(self,*args,**kwargs):
        for iFM,FM in enumerate(self.LFM):
            MainFacetOptions=self.GiveMainFacetOptions(FM.GD)
            MainFacetOptions["ImageName"]=FM.GD["Output"]["Name"]
            MainFacetOptions["ra0dec0"]=self.ListDicoFields[iFM]["ra0dec0"]
            FM.appendMainField(**MainFacetOptions)

    def ToCasaImage(self,ListArray,Fits=True, ImageName=None,
                    beam=None, beamcube=None, Freqs=None, Stokes=["I"]):
        if isinstance(ListArray,list):
            if not len(ListArray)==self.NFields: stop
        elif isinstance(ListArray,np.ndarray):
            if not ListArray.shape[0]==self.NFields: stop

        for iFM,FM in enumerate(self.LFM):
            Array=ListArray[iFM]
            if Array.dtype.type is not np.float32:
                Array=np.float32(Array)
            ThisBeam=beam
            if beam is not None:
                ThisBeam=beam[iFM]
            ThisBeamCube=beamcube
            if beamcube is not None:
                ThisBeamCube=beamcube[iFM]
            FM.ToCasaImage(Array,Fits=Fits, ImageName="%s_Field%i"%(ImageName,iFM),
                           beam=ThisBeam, beamcube=ThisBeamCube, Freqs=Freqs, Stokes=Stokes)
    def releaseGrids(self,*args,**kwargs):
        for iFM,FM in enumerate(self.LFM):
            FM.releaseGrids(*args,**kwargs)

    def getNormDict(self):
        D={}
        for iFM,FM in enumerate(self.LFM):
            D[iFM]=FM.getNormDict()
        return L
        
            
    def StackAverageBeam(self,*args,**kwargs):
        for FM in self.LFM:
            FM.StackAverageBeam(*args,**kwargs)
            
    def putChunkInBackground(self,*args,**kwargs):
        for FM in self.LFM:
            FM.putChunkInBackground(*args,**kwargs)
            
    def finaliseSmoothBeam(self,*args,**kwargs):
        self.SmoothJonesNorm=DictImages()
        self.MeanSmoothJonesNorm=DictImages()
        for iFM,FM in enumerate(self.LFM):
            FM.finaliseSmoothBeam(*args,**kwargs)
            self.SmoothJonesNorm[iFM]=FM.SmoothJonesNorm
            self.MeanSmoothJonesNorm[iFM] = FM.MeanSmoothJonesNorm

    def FacetsToIm(self,*args,**kwargs):
        DicoImages=DictImages()
        for iFM,FM in enumerate(self.LFM):
            DicoImages[iFM]=FM.FacetsToIm(*args,**kwargs)
        self.DicoImages=DicoImages
        return DicoImages
    
    def applySparsification(self,*args,**kwargs):
        for FM in self.LFM:
            FM.applySparsification(*args,**kwargs)
            
    def awaitInitCompletion(self,*args,**kwargs):
        for FM in self.LFM:
            FM.awaitInitCompletion(*args,**kwargs)
            
    def setAverageBeamMachine(self,AverageBeamMachine):
        for FM in self.LFM:
            FM.setAverageBeamMachine(AverageBeamMachine)
            
    def setModelImage(self, ModelImage):
        L=[]
        for iFM,FM in enumerate(self.LFM):
            M=FM.setModelImage(ModelImage[iFM])
            L.append(M)
        return L

    def getChunkInBackground(self,*args,**kwargs):
        for iFM,FM in enumerate(self.LFM):
            FM.getChunkInBackground(*args,**kwargs)
            
    def collectDegriddingResults(self,*args,**kwargs):
        for iFM,FM in enumerate(self.LFM):
            FM.collectDegriddingResults(*args,**kwargs)
            
    def releaseModelImage(self,*args,**kwargs):
        for iFM,FM in enumerate(self.LFM):
            FM.releaseModelImage(*args,**kwargs)
            
            
    def initCFInBackground(self, other_fm=None):
        ThisOtherFM=None
        for iFM,FM in enumerate(self.LFM):
            if other_fm is not None:
                ThisOtherFM=other_fm.LFM[iFM]
            FM.initCFInBackground(other_fm=ThisOtherFM)
            
    def releaseCFs(self,*args,**kwargs):
        for FM in self.LFM:
            FM.releaseCFs(*args,**kwargs)
            
        
    def ReinitDirty(self,*args,**kwargs):
        for FM in self.LFM:
            FM.ReinitDirty(*args,**kwargs)

    def collectGriddingResults(self,*args,**kwargs):
        for FM in self.LFM:
            FM.collectGriddingResults(*args,**kwargs)

    def SaveDirtyProducts(self):
        if "d" in self._saveims:
            self.ToCasaImage(self.DicoImages[:,"MeanImage"],ImageName="%s.dirty"%self.BaseName,Fits=True,
                             Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        if "d" in self._savecubes:
            self.ToCasaImage(self.DicoImages[:,"ImageCube"],ImageName="%s.cube.dirty"%self.BaseName,
                             Fits=True,Freqs=self.VS.FreqBandCenters,Stokes=self.VS.StokesConverter.RequiredStokesProducts())

        if "n" in self._saveims:
            FacetNormReShape = self.FacetMachine.getNormDict()["FacetNormReShape"]
            self.ToCasaImage(FacetNormReShape,
                             ImageName="%s.NormFacets"%self.BaseName,
                             Fits=True)

        if self.DicoImages[0]["JonesNorm"] is not None:
            NormImage=self.DicoImages[:,"JonesNorm"]

            if self.DoSmoothBeam and self.SmoothJonesNorm is not None:
                NormImage=self.SmoothJonesNorm

            LDirty=self.DicoImages[:,"ImageCube"]
            LDirtyCorr=[]
            LMeanDirtyCorr=[]
            for iFM in range(self.NFields):
                DirtyCorr=LDirty[iFM]/np.sqrt(NormImage[iFM])
                nch,npol,nx,ny = DirtyCorr.shape
                LDirtyCorr.append(DirtyCorr)
                MeanDirtyCorr=np.mean(DirtyCorr, axis=0).reshape((1, npol, nx, ny))
                LMeanDirtyCorr.append(MeanDirtyCorr)
                
            if "D" in self._saveims:
                self.ToCasaImage(LMeanDirtyCorr,ImageName="%s.dirty.corr"%self.BaseName,Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            if "D" in self._savecubes:
                self.ToCasaImage(LDirtyCorr,ImageName="%s.cube.dirty.corr"%self.BaseName,
                                              Fits=True,Freqs=self.VS.FreqBandCenters,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())

            self.LJonesNorm = self.DicoImages[:,"JonesNorm"]
            self.LMeanJonesNorm=[]
            for iFM in range(self.NFields):
                JonesNorm=self.LJonesNorm[iFM]
                nch,npol,nx,ny = JonesNorm.shape
                MeanJonesNorm=np.mean(JonesNorm, axis=0).reshape((1, npol, nx, ny))
                self.LMeanJonesNorm.append(MeanJonesNorm)
                
            if self.DoSmoothBeam and self.FacetMachine.SmoothJonesNorm is not None:
                self.ToCasaImage(self.FacetMachine.MeanSmoothJonesNorm,ImageName="%s.MeanSmoothNorm"%self.BaseName,Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
                self.ToCasaImage(self.FacetMachine.SmoothJonesNorm,ImageName="%s.SmoothNorm"%self.BaseName,Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts(),
                                              Freqs=self.VS.FreqBandCenters)


            if "N" in self._saveims:
                self.ToCasaImage(self.LMeanJonesNorm,ImageName="%s.Norm"%self.BaseName,Fits=True,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
            if "N" in self._savecubes:
                self.ToCasaImage(self.LJonesNorm, ImageName="%s.cube.Norm" % self.BaseName,
                                              Fits=True, Freqs=self.VS.FreqBandCenters,
                                              Stokes=self.VS.StokesConverter.RequiredStokesProducts())
        else:
            self.MeanJonesNorm = None
            
