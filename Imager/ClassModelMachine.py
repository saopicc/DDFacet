import numpy as np
import pylab
from DDFacet.Other import MyLogger
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassModelMachine")
from DDFacet.Array import NpParallel
from DDFacet.Array import ModLinAlg
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyPickle

from DDFacet.ToolsDir.GiveEdges import GiveEdges

import ClassModelMachine
from DDFacet.ToolsDir import ModFFTW
import scipy.ndimage


class ClassModelMachine():
    def __init__(self,GD=None,Gain=None):
        self.GD=GD
        if Gain==None:
            self.Gain=self.GD["ImagerDeconv"]["Gain"]
        else:
            self.Gain=Gain
        self.DicoSMStacked={}
        self.DicoSMStacked["Comp"]={}

    def setRefFreq(self,RefFreq,AllFreqs):
        self.RefFreq=RefFreq
        self.DicoSMStacked["RefFreq"]=RefFreq
        self.DicoSMStacked["AllFreqs"]=np.array(AllFreqs)
        
    def ToFile(self,FileName):
        print>>log, "Saving dico model to %s"%FileName
        D=self.DicoSMStacked
        D["ListScales"]=self.ListScales
        D["ModelShape"]=self.ModelShape
        MyPickle.Save(D,FileName)

    def FromFile(self,FileName):
        print>>log, "Reading dico model from %s"%FileName
        self.DicoSMStacked=MyPickle.Load(FileName)
        self.RefFreq=self.DicoSMStacked["RefFreq"]
        self.ListScales=self.DicoSMStacked["ListScales"]
        self.ModelShape=self.DicoSMStacked["ModelShape"]

    def setModelShape(self,ModelShape):
        self.ModelShape=ModelShape

    def AppendComponentToDictStacked(self,key,Fpol,Sols):
        DicoComp=self.DicoSMStacked["Comp"]
        if not(key in DicoComp.keys()):
            #print>>log, ModColor.Str("Add key %s"%(str(key)))
            DicoComp[key]={}
            DicoComp[key]["SolsArray"]=np.zeros((Sols.size,),np.float32)
            DicoComp[key]["SumWeights"]=0.

        Weight=1.
        SolNorm=Sols.ravel()*self.Gain*np.mean(Fpol)


        DicoComp[key]["SumWeights"] += Weight
        DicoComp[key]["SolsArray"]  += Weight*SolNorm
        # print>>log, "Append %s: %s %s"%(str(key),str(DicoComp[key]["SolsArray"]),str(SolNorm))
        
    def setListComponants(self,ListScales):
        self.ListScales=ListScales


    def GiveSpectralIndexMap(self,CellSizeRad=1.,GaussPars=[(1,1,0)]):

        
        dFreq=1e6
        f0=self.DicoSMStacked["AllFreqs"].min()
        f1=self.DicoSMStacked["AllFreqs"].max()
        M0=self.GiveModelImage(f0)
        M0=ModFFTW.ConvolveGaussian(M0,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
        M1=self.GiveModelImage(f1)
        M1=ModFFTW.ConvolveGaussian(M1,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
        Mask=((M1!=0)&(M0!=0))
        alpha=np.zeros_like(M0)
        alpha[Mask]=(np.log(M0[Mask])-np.log(M1[Mask]))/(np.log(f0/f1))
        return alpha

    def GiveModelImage(self,Freq):
        RefFreq=self.DicoSMStacked["RefFreq"]
        DicoComp=self.DicoSMStacked["Comp"]
        _,npol,nx,ny=self.ModelShape
        N0=nx
        ModelImage=np.zeros((1,npol,nx,ny),dtype=np.float32)
        DicoSM={}
        for key in DicoComp.keys():
            Sol=DicoComp[key]["SolsArray"]#/self.DicoSMStacked[key]["SumWeights"]
            x,y=key

            #print>>log, "%s : %s"%(str(key),str(Sol))

            for iFunc in range(Sol.size):
                ThisComp=self.ListScales[iFunc]
                ThisAlpha=ThisComp["Alpha"]
                Flux=Sol[iFunc]*(Freq/RefFreq)**(ThisAlpha)
                if ThisComp["ModelType"]=="Delta":
                    for pol in range(npol):
                       ModelImage[0,pol,x,y]+=Flux
                
                elif ThisComp["ModelType"]=="Gaussian":
                    Gauss=ThisComp["Model"]
                    Sup,_=Gauss.shape
                    x0,x1=x-Sup/2,x+Sup/2+1
                    y0,y1=y-Sup/2,y+Sup/2+1
                
                
                    Aedge,Bedge=GiveEdges((x,y),N0,(Sup/2,Sup/2),Sup)
                    x0d,x1d,y0d,y1d=Aedge
                    x0p,x1p,y0p,y1p=Bedge
                
                    for pol in range(npol):
                        ModelImage[0,pol,x0d:x1d,y0d:y1d]+=Gauss[x0p:x1p,y0p:y1p]*Flux
        
        # vmin,vmax=np.min(self._MeanDirtyOrig[0,0]),np.max(self._MeanDirtyOrig[0,0])
        # vmin,vmax=-1,1
        # #vmin,vmax=np.min(ModelImage),np.max(ModelImage)
        # pylab.clf()
        # ax=pylab.subplot(1,3,1)
        # pylab.imshow(self._MeanDirtyOrig[0,0],interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
        # pylab.imshow(self._MeanDirty[0,0],interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.colorbar()
        # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
        # pylab.imshow( ModelImage[0,0],interpolation="nearest",vmin=vmin,vmax=vmax)
        # pylab.colorbar()
        # pylab.draw()
        # pylab.show(False)
        # print np.max(ModelImage[0,0])
        # # stop

 
        return ModelImage
        
    def CleanNegComponants(self,box=20,sig=3):
        print>>log, "Cleaning model dictionary from negative componants with (box, sig) = (%i, %i)"%(box,sig)
        ModelImage=self.GiveModelImage(self.DicoSMStacked["RefFreq"])[0,0]
        
        Min=scipy.ndimage.filters.minimum_filter(ModelImage,(box,box))
        Min[Min>0]=0
        Min=-Min
        Lx,Ly=np.where((ModelImage<sig*Min)&(ModelImage!=0))
        
        for icomp in range(Lx.size):
            key=Lx[icomp],Ly[icomp]
            try:
                del(self.DicoSMStacked["Comp"][key])
            except:
                print>>log, "  Componant at (%i, %i) not in dict "%key
