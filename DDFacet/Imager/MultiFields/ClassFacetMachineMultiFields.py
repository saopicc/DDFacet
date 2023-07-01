from DDFacet.Imager.ClassFacetMachineTessel import ClassFacetMachineTessel as ClassFacetMachine
import csv
import numpy as np
import copy
from astropy.coordinates import SkyCoord
import astropy.units as u
from DDFacet.Other.AsciiReader import readMultiFieldFile
import os
from DDFacet.Array import shared_dict
import glob
import cpuinfo

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
        os.system("mkdir -p %s"%path)
        for k in sorted(self.data.keys()):
            ThisPath=self.data[k].path.split("/")[-1]
            #ThisPath=path"%s/Field%i"%(path,k)
            #print("HHHHHH",path,ThisPath)
            ThisPath="%s/%s"%(path,ThisPath)
            self.data[k].save(ThisPath)


    def reload(self):
        for k in sorted(self.data.keys()):
            self.data[k].reload()

    def delete(self):
        for k in sorted(self.data.keys()):
            self.data[k].delete()

    def restore(self,DirName):
        ll=sorted(glob.glob("%s*"%DirName))
        for iField,l in enumerate(ll):
            ThisSHMName=l.split("/")[-1]

            print(l,ThisSHMName)
            D = shared_dict.create(ThisSHMName)
            D.restore(l)
            self.data[iField]=D
            

# class DictImages(shared_dict.SharedDict):
#     def __init__(self, *args, **kwargs):
#         shared_dict.SharedDict.__init__(self, *args, **kwargs)
    
#     def __getitem__(self, X):
#         if isinstance(X,tuple) and len(X)==2:
#             (key, subkey)=X
#             if key == slice(None):
#                 return [shared_dict.SharedDict.__getitem__(self, k)[subkey] for k in sorted(self.keys())]
#             else:
#                 return shared_dict.SharedDict.__getitem__(self, key)[subkey]
#         else:
#             key=X
#             if key == slice(None):
#                 return [shared_dict.SharedDict.__getitem__(self, k) for k in sorted(self.keys())]
#             else:
#                 return shared_dict.SharedDict.__getitem__(self, key)
            
            

            
        
            
class ClassFacetMachineMultiFields():
    def __init__(self,
                 VS,
                 GD,
                 Precision="S",
                 PolMode=["I"],
                 Sols=None,
                 #PointingID=0,
                 DoPSF=False,
                 Oversize=1,   # factor by which image is oversized
                 custom_id=None,
                 FieldID=None,
                 DicoFields=None):
        custom_id0=custom_id
        self.custom_id=custom_id
        self.DicoFields=DicoFields
        self.GD=GD
        self.VS=VS
        self.Type="MultiField"
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
        self.LMeanSmoothJonesNorm=None
        
        self.ListFMConf=[]
        if self.GD["Image"]["MultiFieldFile"] is None:
            FMConf={"GD":GD,
                    "FieldID":None}
            self.ListFMConf.append(FMConf)
        else:
            
            for iField,ThisField in enumerate(self.DicoFields):
                # coords = SkyCoord(ra=ThisField["ra"],
                #                   dec=ThisField["dec"],
                #                   unit=(u.hourangle, u.deg))
                # ras=rad2hmsdms(coords.ra.rad,Type="ra").replace(" ",":")
                # decs=rad2hmsdms(coords.dec.rad,Type="dec").replace(" ",":")
                ThisGD=copy.deepcopy(self.GD)
                if "NPix" in ThisField.keys():
                    NPix=int(ThisField["NPix"])
                    ThisGD["Image"]["NPix"]=NPix
                ThisGD["Image"]["ImageCenterRADEC"]=ThisField["ra"],ThisField["dec"]

                ThisFMConf={"GD":ThisGD,
                            "FieldID":iField}
                self.ListFMConf.append(ThisFMConf)
            self.NFields=len(self.ListFMConf)

        self.LFM=[]
        cpudict=cpuinfo.get_cpu_info()
        # MultiField mode
        for iField,FMConf in enumerate(self.ListFMConf):
            # ra0dec0=DicoField["ra0dec0"]
            FieldID=FMConf["FieldID"]
            ThisGD=FMConf["GD"]
            BaseName=ThisGD["Output"]["Name"]
            if FieldID is not None:
                custom_id=custom_id0#"%s_Field%i"%(custom_id0,FieldID)
                BaseName="%s_Field%i"%(BaseName,FieldID)
                ThisGD["Output"]["Name"]=BaseName
                
            FM=ClassFacetMachine(VS,
                                 ThisGD,
                                 Precision=Precision,
                                 PolMode=PolMode,
                                 Sols=Sols,
                                 iField=iField,
                                 #PointingID=PointingID,
                                 DoPSF=DoPSF,
                                 Oversize=Oversize,   # factor by which image is oversized
                                 custom_id=custom_id,
                                 cpudict=cpudict)
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

        # self.FullImShape = self.LFM[0].OutImShape
        # self.OutImShape = self.LFM[0].OutImShape
        # self.PaddedGridShape = self.PaddedFacetShape = self.LFM[0].PaddedGridShape
        # self.FacetShape = self.LFM[0].FacetShape
        # self.CellSizeRad_x,self.CellSizeRad_y=self.CellSizeRad = self.LFM[0].CellSizeRad
        # self.FacetDirCat=np.concatenate([FM.FacetDirCat for FM in self.LFM])
        # self.FacetDirCat=self.FacetDirCat.view(np.recarray)
        
        self.D_DicoImager={}
        self.NFacetsTotalFields=0
        for iFM,FM in enumerate(self.LFM):
            self.D_DicoImager[iFM]=FM.DicoImager
            self.NFacetsTotalFields+=len(FM.DicoImager)
            
    def appendMainField(self,*args,**kwargs):
        for iFM,FM in enumerate(self.LFM):
            MainFacetOptions=self.GiveMainFacetOptions(FM.GD)
            MainFacetOptions["ImageName"]=FM.GD["Output"]["Name"]
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
        
        self.LJonesNorm = self.DicoImages[:,"JonesNorm"]
        self.LMeanJonesNorm=[]
        for iFM in range(self.NFields):
            JonesNorm=self.LJonesNorm[iFM]
            nch,npol,nx,ny = JonesNorm.shape
            MeanJonesNorm=np.mean(JonesNorm, axis=0).reshape((1, npol, nx, ny))
            self.LMeanJonesNorm.append(MeanJonesNorm)
                
        # self.JonesNorm=DictImages()
        # self.MeanJonesNorm=DictImages()
        # self.FacetNorm=DictImages()
        # self.FacetNormReShape=DictImages()
        # for iFM,FM in enumerate(self.LFM):
        #     self.JonesNorm[iFM]=FM.JonesNorm
        #     self.MeanJonesNorm[iFM]=FM.MeanJonesNorm
        #     self.FacetNorm[iFM]=FM.FacetNorm
        #     self.FacetNormReShape[iFM]=FM.FacetNormReShape
            
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

    def BuildFacetNormImage(self):
        for iFM,FM in enumerate(self.LFM):
            FM.BuildFacetNormImage()
            
    def set_model_grid (self,*args,**kwargs):
        self.D_model_dict=DictImages()
        for iFM,FM in enumerate(self.LFM):
            FM.set_model_grid(*args,**kwargs)
            self.D_model_dict[iFM]=FM._model_dict
                                     
    def releaseCFs(self,*args,**kwargs):
        for FM in self.LFM:
            FM.releaseCFs(*args,**kwargs)
            
        
    def ReinitDirty(self,*args,**kwargs):
        for FM in self.LFM:
            FM.ReinitDirty(*args,**kwargs)

    def collectGriddingResults(self,*args,**kwargs):
        for FM in self.LFM:
            FM.collectGriddingResults(*args,**kwargs)

    def setNormImages(self,DicoDirty):
        self.JonesNorm=DictImages()
        self.MeanJonesNorm=DictImages()
        self.FacetNorm=DictImages()
        self.FacetNormReShape=DictImages()
        
        for iFM,FM in enumerate(self.LFM):
            FM.setNormImages(DicoDirty[iFM])
            self.JonesNorm[iFM]=FM.JonesNorm
            self.MeanJonesNorm[iFM]=FM.MeanJonesNorm
            self.FacetNorm[iFM]=FM.FacetNorm
            self.FacetNormReShape[iFM]=FM.FacetNormReShape
            
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
            
