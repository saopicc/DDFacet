from DDFacet.Imager.ClassFacetMachineTessel import ClassFacetMachineTessel as ClassFacetMachine
import csv
import numpy as np
import copy


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
                 
        self.GD=GD

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
            ThisGD=ThisDicoField["GD"]
            BaseName=ThisGD["Image"]["Name"]
            if FieldID is not None:
                custom_id="%s_%s"%(custom_id,FieldID)
                BaseName="%s_Field%i"%(BaseName,self.FieldID)
                ThisGD["Image"]["Name"]=BaseName
                
            FM=ClassFacetMachine(VS,
                                 ThisGD,
                                 Precision=Precision,
                                 PolMode=PolMode,
                                 Sols=Sols,
                                 PointingID=PointingID,
                                 DoPSF=DoPSF,
                                 Oversize=Oversize,   # factor by which image is oversized
                                 custom_id=custom_id)
            self.LFM.append(FM)

            
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


    def StackAverageBeam(self,*args,**kwargs):
        for FM in self.LFM:
            FM.StackAverageBeam(*args,**kwargs)

    def putChunkInBackground(self,*args,**kwargs):
        for FM in self.LFM:
            FM.putChunkInBackground(*args,**kwargs)
            
    def finaliseSmoothBeam(self,*args,**kwargs):
        for FM in self.LFM:
            FM.finaliseSmoothBeam(*args,**kwargs)
            
    def FacetsToIm(self,*args,**kwargs):
        for FM in self.LFM:
            FM.FacetsToIm(*args,**kwargs)
            
    def applySparsification(self,*args,**kwargs):
        for FM in self.LFM:
            FM.applySparsification(*args,**kwargs)
            
    def awaitInitCompletion(self,*args,**kwargs):
        for FM in self.LFM:
            FM.awaitInitCompletion(*args,**kwargs)
            
    def setAverageBeamMachine(self,AverageBeamMachine):
        for FM in self.LFM:
            FM.setAverageBeamMachine(AverageBeamMachine)
            
    def initCFInBackground(self):
        for FM in self.LFM:
            FM.initCFInBackground()
        
    def appendMainField(self,*args,**kwargs):
        for FM in self.LFM:
            FM.appendMainField(*args,**kwargs)
        
    def ReinitDirty(self,*args,**kwargs):
        for FM in self.LFM:
            FM.ReinitDirty(*args,**kwargs)

    def collectGriddingResults(self,*args,**kwargs):
        for FM in self.LFM:
            FM.collectGriddingResults(*args,**kwargs)
