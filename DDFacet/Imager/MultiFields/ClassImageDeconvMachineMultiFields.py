import numpy as np
from DDFacet.Other import logger
import pickle as cPickle
log=logger.getLogger("ImageDeconvMachineMultiField")
import six
import copy
from DDFacet.Other import ModColor

class ClassImageDeconvMachineMultiFields():

    def __init__(self,GD,VS=None,DicoFields=None,
                 LModelMachine=None,LMaskMachine=None,NMajor=None):
        self.GD=GD
        self.VS=VS
        self.NMajor=NMajor
        self.LMaskMachine=LMaskMachine
        # If we do the deconvolution construct a model according to what is in MinorCycleConfig
        MinorCycleConfig=dict(self.GD["Deconv"])
        MinorCycleConfig["NCPU"] = self.GD["Parallel"]["NCPU"]
        MinorCycleConfig["NBand"]=MinorCycleConfig["NFreqBands"]=self.VS.NFreqBands
        MinorCycleConfig["GD"] = self.GD
        MinorCycleConfig["ImagePolDescriptor"] = self.VS.StokesConverter.RequiredStokesProducts()
        self.DicoFields=DicoFields
        self.NFields=len(self.DicoFields)
        self.LImageDeconvMachine=[]
        for iField in range(self.NFields):
            M=copy.deepcopy(MinorCycleConfig)
            MM=LModelMachine[iField]
            M["RefFreq"] = MM.RefFreq
            M["ModelMachine"] = MM
            self.LImageDeconvMachine.append(self.giveImageDeconvMachineSingleField(iField,M))
            
        
    def giveImageDeconvMachineSingleField(self,iField,MinorCycleConfig):
        # Specify which deconvolution algorithm to use
        if self.GD["Deconv"]["Mode"] == "HMP":
            if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                raise NotImplementedError("Multi-polarization CLEAN is not supported in MSMF")
            from DDFacet.Imager.MSMF import ClassImageDeconvMachineMSMF
            DeconvMachine=ClassImageDeconvMachineMSMF.ClassImageDeconvMachine(MainCache=self.VS.maincache, **MinorCycleConfig)
            print("Using MSMF algorithm", file=log)
        elif self.GD["Deconv"]["Mode"]=="SSD":
            if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                raise NotImplementedError("Multi-polarization is not supported in SSD")
            from DDFacet.Imager.SSD import ClassImageDeconvMachineSSD
            DeconvMachine=ClassImageDeconvMachineSSD.ClassImageDeconvMachine(MainCache=self.VS.maincache,
                                                                                  **MinorCycleConfig)
            print("Using SSD with %s Minor Cycle algorithm"%self.GD["SSDClean"]["IslandDeconvMode"], file=log)
        elif self.GD["Deconv"]["Mode"]=="SSD2":
            if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                raise NotImplementedError("Multi-polarization is not supported in SSD")
            from DDFacet.Imager.SSD2 import ClassImageDeconvMachineSSD
            DeconvMachine=ClassImageDeconvMachineSSD.ClassImageDeconvMachine(MainCache=self.VS.maincache,
                                                                                  MinorCycleConfig=MinorCycleConfig,
                                                                                  **MinorCycleConfig)
            DeconvMachine.setMaxMajorIter(self.NMajor)
            print("Using SSD2 with %s Minor Cycle algorithm"%self.GD["SSDClean"]["IslandDeconvMode"], file=log)
            if self.NMajor>3:
                print(ModColor.Str("  Your number of major iterations (%i) seem too high for SSD2, we advice using a maximum of 3..."%self.NMajor), file=log)
        elif self.GD["Deconv"]["Mode"] == "Hogbom":
            if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                raise NotImplementedError("Multi-polarization CLEAN is not supported in Hogbom")
            from DDFacet.Imager.HOGBOM import ClassImageDeconvMachineHogbom
            DeconvMachine=ClassImageDeconvMachineHogbom.ClassImageDeconvMachine(**MinorCycleConfig)
            print("Using Hogbom algorithm", file=log)
        elif self.GD["Deconv"]["Mode"]=="MultiSlice":
            if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                raise NotImplementedError("Multi-polarization is not supported in MultiSlice")
            from DDFacet.Imager.MultiSliceDeconv import ClassImageDeconvMachineMultiSlice
            DeconvMachine=ClassImageDeconvMachineMultiSlice.ClassImageDeconvMachine(MainCache=self.VS.maincache, **MinorCycleConfig)
            print("Using MultiSlice algorithm", file=log)
        elif self.GD["Deconv"]["Mode"]=="WSCMS":
            if MinorCycleConfig["ImagePolDescriptor"] != ["I"]:
                raise NotImplementedError("Multi-polarization is not supported in WSCMS")
            from DDFacet.Imager.WSCMS import ClassImageDeconvMachineWSCMS
            DeconvMachine = ClassImageDeconvMachineWSCMS.ClassImageDeconvMachine(MainCache=self.VS.maincache,
                                                                                      **MinorCycleConfig)
            print("Using WSCMS algorithm", file=log)
        else:
            raise NotImplementedError("Unknown --Deconvolution-Mode setting '%s'" % self.GD["Deconv"]["Mode"])
        DeconvMachine.setMaskMachine(self.LMaskMachine[iField])
        return DeconvMachine

    def Init(self,PSFVar=None, PSFAve=None,
             approx=None, cache=None,
             GridFreqs=None, DegridFreqs=None,
             RefFreq=None, MaxBaseline=None,
             FacetMachine=None, BaseName=None):
        for iField in range(self.NFields):
            self.LImageDeconvMachine[iField].Init(PSFVar=PSFVar[iField],
                                                  PSFAve=PSFVar[iField]["PSFSidelobesAvg"],
                                                  approx=approx, cache=cache,
                                                  GridFreqs=GridFreqs,
                                                  DegridFreqs=DegridFreqs,
                                                  RefFreq=RefFreq, MaxBaseline=MaxBaseline,
                                                  FacetMachine=FacetMachine.LFM[iField],
                                                  BaseName=FacetMachine.LFM[iField].ImageName)


    def Update(self,DicoDirty):
        for iField in range(self.NFields):
            self.LImageDeconvMachine[iField].Update(DicoDirty[iField])

    def Deconvolve(self):
        LrepMinor=[]
        Lcontinue_deconv=[]
        Lupdate_model=[]
        
        for iField in range(self.NFields):
            log.print(ModColor.Str("=============== Deconv Field #%i / %i ============="%(iField,self.NFields),col="blue"))
            repMinor, continue_deconv, update_model = self.LImageDeconvMachine[iField].Deconvolve()
            LrepMinor.append(repMinor)
            Lcontinue_deconv.append(continue_deconv)
            Lupdate_model.append(update_model)
        return LrepMinor, Lcontinue_deconv, Lupdate_model

    def GiveModelImage(self,model_freqs):
        L=[]
        for iField in range(self.NFields):
            M=self.LImageDeconvMachine[iField].GiveModelImage(model_freqs)
            L.append(M)
        return L

    def ToFile(self,*args,**kwargs):
        pass
