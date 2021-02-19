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

from DDFacet.ToolsDir import ModFFTW
import numpy as np
import random
from deap import tools
from DDFacet.Other import ClassTimeIt
from itertools import repeat
from DDFacet.ToolsDir import ClassSpectralFunctions


class ClassParamMachine():
    def __init__(self,ListPixParms,ListPixData,FreqsInfo,iFacet=0,
                 NOrderPoly=2,
                 SolveParamType=["Poly"]):

        self.ListPixParms=ListPixParms
        self.ListPixData=ListPixData
        self.NPixListParms=len(self.ListPixParms)
        self.NPixListData=len(self.ListPixData)
        self.iFacet=iFacet
        self.NOrderPoly=NOrderPoly
        SolveParam=[]
        if "Poly" in SolveParamType:
            for iOrder in range(NOrderPoly): SolveParam.append("Poly%i"%iOrder)
        if "GSig" in SolveParamType: 
            SolveParam.append("GSig")
            
        self.PolyOrder=NOrderPoly#np.sum([("Poly" in key) for key in SolveParam])
        
        self.SolveParam=SolveParam

        
        self.MultiFreqMode=False
        if self.PolyOrder>1:
            self.MultiFreqMode=True

        self.NParam=len(self.SolveParam)
        
        self.DicoIParm={}
        DefaultValues={"Poly0":{"Mean":0.,
                            "Sigma":{
                                "Type":"PeakFlux",
                                "Value":0.1}
                        },
                       "Poly1":{"Mean":-0.6,
                                "Sigma":{
                                    "Type":"Abs",
                                    "Value":0.1}
                                },
                       "GSig":{"Mean":0.,
                                "Sigma":{
                                    "Type":"Abs",
                                    "Value":1}
                                }
                   }

        for order in range(2,10):
            DefaultValues["Poly%i"%order]={"Mean":0.,
                                           "Sigma":{
                                               "Type":"Abs",
                                               "Value":0.1}
            }
            

        
        for Type in DefaultValues.keys():
            self.DicoIParm[Type]={}
            self.DicoIParm[Type]["Default"]=DefaultValues[Type]
            self.DicoIParm[Type]["iSlice"]=None
            
        for iParm,Type in zip(range(self.NParam),self.SolveParam):
            self.DicoIParm[Type]["iSlice"]=iParm

        self.NFreqBands=len(FreqsInfo["freqs"])
        self.SetSquareGrids()

    def setFreqs(self,DicoMappingDesc):
        self.DicoMappingDesc=DicoMappingDesc
        if self.DicoMappingDesc is None: return
        self.SpectralFunctionsMachine=ClassSpectralFunctions.ClassSpectralFunctions(self.DicoMappingDesc,RefFreq=self.DicoMappingDesc["RefFreq"])#,BeamEnable=False)
        
    def GiveIndivZero(self):
        return np.zeros((self.NParam,self.NPixListParms),np.float32)


    def GiveInitList(self,toolbox):
        ListPars=[]
        for Type in self.SolveParam:
            DicoSigma=self.DicoIParm[Type]["Default"]["Sigma"]
            MeanVal=self.DicoIParm[Type]["Default"]["Mean"]
            if Type=="Poly0":
                toolbox.register("attr_float_unif_S", random.uniform, 0., 0.1)
                ListPars+=[toolbox.attr_float_unif_S]*self.NPixListParms
            elif "Poly" in Type:
                NameAttr="attr_float_normal_%s"%Type
                toolbox.register(NameAttr, random.gauss, MeanVal, 0)#DicoSigma["Value"])
                attr=getattr(toolbox,NameAttr)
                #ListPars+=[toolbox.attr_float_normal_Alpha]*self.NPixListParms
                ListPars+=[attr]*self.NPixListParms
            elif Type=="GSig":
                toolbox.register("attr_float_normal_GSig", random.uniform, 0, 1)#DicoSigma["Value"])
                ListPars+=[toolbox.attr_float_normal_GSig]*self.NPixListParms
        return ListPars

    #    def ReinitPop(self,pop,SModelArray,AlphaModel=None,GSigModel=None,PutNoise=True):
    def ReinitPop(self,pop,ListPolyModelArray,GSigModel=None,PutNoise=True):
        
        
        for Type in self.SolveParam:
            

            for i_indiv,indiv in zip(range(len(pop)),pop):
                
                PolyModelArray=ListPolyModelArray[i_indiv]
                
                SModelArray=PolyModelArray[0,:]
                
                DicoSigma=self.DicoIParm[Type]["Default"]["Sigma"]
                MeanVal=self.DicoIParm[Type]["Default"]["Mean"]
                if DicoSigma["Type"]=="Abs":
                    SigVal=DicoSigma["Value"]
                elif DicoSigma["Type"]=="PeakFlux":
                    SigVal=DicoSigma["Value"]*np.max(np.abs(SModelArray))


                
                SubArray=self.ArrayToSubArray(indiv,Type=Type)
                if Type=="Poly0":
                    SubArray[:]=PolyModelArray[0,:]
                    S=np.abs(SubArray[:]).copy()
                    if np.max(S)>0:
                        S/=np.max(S)
                        
                    if (i_indiv!=0) and PutNoise:
                        #SubArray[:]+=np.random.randn(SModelArray.size)*SigVal
                        #SubArray[:]+=np.random.randn(SModelArray.size)*SigVal*(SubArray[:]!=0.) # will not put noise in zero-valued pixels
                        
                        SubArray[:]+=np.random.randn(SModelArray.size)*SigVal*S # will not put noise in zero-valued pixels
                        
                elif "Poly" in Type:
                    iOrder=int(Type[4:])
                    AlphaModel=None
                    
                    AlphaModel=PolyModelArray[iOrder]
                    
                    # if PolyModelArray.shape[0]>=iOrder:
                    #     AlphaModel=PolyModelArray[iOrder]
                    # else:
                    #     AlphaModel=MeanVal*np.ones((SModelArray.size,),np.float32)
                        
                    SubArray[:]=AlphaModel[:]
                    if (i_indiv!=0) and PutNoise: 
                        SubArray[:]+=np.random.randn(SModelArray.size)*SigVal

                # elif Type=="Poly1":
                #     if AlphaModel is None:
                #         AlphaModel=MeanVal*np.ones((SModelArray.size,),np.float32)
                #     SubArray[:]=AlphaModel[:]
                #     if (i_indiv!=0) and PutNoise: 
                #         SubArray[:]+=np.random.randn(SModelArray.size)*SigVal
                # elif "Poly" in Type:
                #     ParmModel=MeanVal*np.ones((SModelArray.size,),np.float32)
                #     SubArray[:]=ParmModel[:]
                #     if (i_indiv!=0) and PutNoise: 
                #         SubArray[:]+=np.random.randn(SModelArray.size)*SigVal

                elif Type=="GSig":
                    if GSigModel==None:
                        GSigModel=MeanVal*np.ones((SModelArray.size,),np.float32)
                    SubArray[:]=GSigModel[:]
                    #SubArray[:]=0

                    #SubArray[49]=1.
                    #SubArray[:]+=np.random.randn(SModelArray.size)*SigVal
                    #SubArray[SubArray<0]=0
                    #if i_indiv!=0: 
                    #   SubArray[:]+=np.random.randn(SModelArray.size)*SigVal
                    SubArray[SubArray<0]=0
                    #SubArray.fill(0)

                    # SubArray[:]=np.zeros_like(AlphaModel)[:]#+np.random.randn(SModelArray.size)*SigVal
                    # print SubArray[:]

                # SubArray=self.ArrayToSubArray(indiv,Type="S")
                # SubArray.fill(0)
                # SubArray[49]=1.
                # SubArray=self.ArrayToSubArray(indiv,Type="GSig")
                # SubArray.fill(0)
                # SubArray[49]=1.

                
    def giveIndexParm(self,Type):
        return self.DicoIParm[Type]["iSlice"]


    def ArrayToSubArray(self,A,Type):
        iSlice=self.DicoIParm[Type]["iSlice"]
        if iSlice is not None:
            ParmsArray=A.reshape((self.NParam,self.NPixListParms))[iSlice]
        elif "DataModel" in self.DicoIParm[Type].keys():
            ParmsArray=self.DicoIParm[Type]["DataModel"].flatten().copy()
        else:
            ParmsArray=np.zeros((self.NPixListParms,),np.float32)
            ParmsArray.fill(self.DicoIParm[Type]["Default"]["Mean"])

        return ParmsArray

    # def SubArrayToArray(self,A,Type):
    #     iSlice=self.DicoIParm[Type]["iSlice"]
    #     if iSlice is not None:
    #         ParmsArray=A.reshape((self.NParam,self.AM.NPixListParms))[iSlice]
    #     else:
    #         ParmsArray=np.zeros((self.AM.NPixListParm,),np.float32)
    #         ParmsArray.fill(self.DicoIParm[Type]["Default"])
    #     return ParmsArray

    def SetSquareGrid(self,Type):
        if Type=="Data":
            ArrayPix=np.array(self.ListPixData)
        else:
            ArrayPix=np.array(self.ListPixParms)

        x,y=ArrayPix.T
        nx=x.max()-x.min()+1
        ny=y.max()-y.min()+1
        NPixSquare=np.max((nx,ny))
        if NPixSquare%2==0: NPixSquare+=1
        xx,yy=np.mgrid[0:NPixSquare,0:NPixSquare]

        MappingIndexToXYPix=(xx[x-x.min(),y-y.min()],yy[x-x.min(),y-y.min()])
        xx=np.int32(xx.flatten())
        yy=np.int32(yy.flatten())
        return {"XY":(xx,yy),
                "NPixSquare":NPixSquare,
                "ArrayPix":ArrayPix,
                "MappingIndexToXYPix":MappingIndexToXYPix,
                "x0y0":(x.min(),y.min())}

    def SetSquareGrids(self):
        self.SquareGrids={"Data":self.SetSquareGrid("Data"),
                          "Parms":self.SetSquareGrid("Parms")}

    # def IndToArray(self,V,key=None):
    #     A=np.zeros((1,1,nx,nx),np.float32)
    #     A.ravel()[:]=np.array(V).ravel()[:]
    #     return A
    
    # def ArrayToInd(self,A):
    #     return A.ravel()#.tolist()

    def ModelToSquareArray(self,Ain,TypeInOut=("Parms","Data"),DomainOut="Freqs"):

        TypeIn,TypeOut=TypeInOut

        if DomainOut=="Parms":
            NSlice=self.NParam
        elif DomainOut=="Freqs":
            NSlice=self.NFreqBands

        NPixSquare=self.SquareGrids[TypeOut]["NPixSquare"]
        A=np.zeros((NSlice,1,NPixSquare,NPixSquare),Ain.dtype)
        x0y0_in  = self.SquareGrids[TypeIn]["x0y0"]
        x0y0_out = self.SquareGrids[TypeOut]["x0y0"]
        dx=x0y0_in[0]-x0y0_out[0]
        dy=x0y0_in[1]-x0y0_out[1]

        ArrayPix=self.SquareGrids[TypeIn]["ArrayPix"]

        Ain=Ain.reshape((NSlice,Ain.size//NSlice))

        x,y=ArrayPix.T
        #print "=============",TypeInOut,A.shape,Ain.shape
        #print "before",A
        xpos=x-x.min()+dx
        ypos=y-y.min()+dy
        nch,npol,nx,_=A.shape
        for iSlice in range(NSlice):
            
            #A[iChannel,0][xpos,ypos]=Ain[iChannel,0]#.copy().flatten()[:]
            ind=xpos*nx + ypos
            A[iSlice].flat[ind]=Ain[iSlice].flat[:]

            # print "ichannel:",iSlice
            # print "in  ",Ain[iSlice].ravel()
            # print "out ",A[iSlice,0,xpos,ypos]
            # R=Ain[iSlice].ravel()-A[iSlice,0,xpos,ypos]
            # print "diff",R
            # if np.max(R)!=0: stop

        return A

    def SquareArrayToModel(self,A,TypeInOut=("Data","Parms")):
        TypeIn,TypeOut=TypeInOut
        x0y0_in  = self.SquareGrids[TypeIn]["x0y0"]
        x0y0_out = self.SquareGrids[TypeOut]["x0y0"]
        dx=x0y0_in[0]-x0y0_out[0]
        dy=x0y0_in[1]-x0y0_out[1]
        if TypeOut=="Data":
            NPixOut=self.NPixListData
        else:
            NPixOut=self.NPixListParms
        
        ArrayPix=self.SquareGrids[TypeOut]["ArrayPix"]
        x,y=ArrayPix.T

        ArrayModel=A[:,0,x-x.min()-dx,y-y.min()+dy].ravel()

        return ArrayModel
        
    def PrintIndiv(self,indiv):
        S=self.ArrayToSubArray(indiv,"S")
        for iPix in range(self.NPixListParms):
            if S[iPix]==0: continue
            print("iPix = %i"%iPix)
            for ParamType in self.SolveParam:
                Q=self.ArrayToSubArray(indiv,ParamType)
                print("  %s = %f"%(ParamType,Q[iPix]))


    def GiveModelArray(self,A):
        
        MA=np.zeros((self.NFreqBands,self.NPixListParms),np.float32)

        # S=self.ArrayToSubArray(A,"Poly0")
        # Alpha=self.ArrayToSubArray(A,"Poly1")
        # A.reshape((self.NParam,self.NPixListParms))
        
        PolyArray=np.zeros((self.NPixListParms,self.PolyOrder),np.float32)
        for iOrder in range(self.PolyOrder):
            PolyArray[:,iOrder]=self.ArrayToSubArray(A,"Poly%i"%iOrder)
            
        for iBand in range(self.NFreqBands):
            MA[iBand]=self.SpectralFunctionsMachine.IntExpFuncPoly(PolyArray,iChannel=iBand,iFacet=self.iFacet)



        if "GSig" in self.SolveParam:
            GSig=self.ArrayToSubArray(A,"GSig")
            #GSig[49]=1.
            ArrayPix=np.array(self.ListPixParms)
            x,y=ArrayPix.T
            MAOut=np.zeros_like(MA)
            
            for iPix in range(self.NPixListParms):
                if S[iPix]==0: continue

                sig=GSig[iPix]
                if sig==0: 
                    for iBand in range(self.NFreqBands):
                        MAOut[iBand,iPix]+=MA[iBand,iPix]
                    continue#np.abs(sig)<=0.5: continue


                d=np.sqrt((x[iPix]-x)**2+(y[iPix]-y)**2)
                v=np.exp(-d**2/(2.*sig**2))
                Sv=np.sum(v)
                #v[v<0.05*SMax]=0
                for iBand in range(self.NFreqBands):
                    SMax=MA[iBand,iPix]#S[iPix]
                    a=SMax/Sv#(2.*np.pi*sig**2)
                    MAOut[iBand]+=np.ones_like(MA[iBand])*a*v
            MA=MAOut

            #print MA.sum(axis=1)



        return MA
