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

from DDFacet.ToolsDir.GiveEdges import GiveEdges

import ClassModelMachine


class ClassModelMachine():

    def __init__(self,GD):
        self.GD=GD
        self.Gain=self.GD["ImagerDeconv"]["Gain"]
        self.DicoSMStacked={}
        self.DicoSMStacked["Comp"]={}

    def setRefFreq(self,RefFreq):
        self.RefFreq=RefFreq
        self.DicoSMStacked["RefFreq"]=RefFreq
        
    def ToFile(self,FileName):
        MyPickle.Save(self.DicoSMStacked,FileName)

    def FromFile(self,FileName):
        self.DicoSMStacked=MyPickle.Load(FileName)


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


    def GiveModelImage(self,Freq):
        RefFreq=self.DicoSMStacked["RefFreq"]
        DicoComp=self.DicoSMStacked["Comp"]
        _,npol,nx,ny=self.ModelShape
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
                
                    _,_,N0,_=self._Dirty.shape
                
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
        

