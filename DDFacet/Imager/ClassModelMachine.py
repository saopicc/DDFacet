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
log=MyLogger.getLogger("ClassModelMachine")
from DDFacet.Other import MyPickle


class ClassModelMachine():
    """
    Interface to ClassModelMachine (in progress)
    GiveModelImage(FreqIn)
        Input:
            FreqIn      = The frequencies at which to return the model image

    ToFile(FileName,DicoIn)
        Input:
            FileName    = The name of the file to write to
            DicoIn      = The dictionary to write to file. If None it writes the current dict in DicoSMStacked to file

    FromFile(FileName)
        Input:
            FileName    = The name of the file to read dict from

    FromDico(DicoIn)
        Input:
            DicoIn      = The dictionary to read in

    """
    def __init__(self,GD=None,Gain=None,GainMachine=None):
        self.GD=GD
        # if Gain is None:
        #     self.Gain=self.GD["ImagerDeconv"]["Gain"]
        # else:
        #     self.Gain=Gain
        self.RefFreq=None
# =======
#         if Gain is None:
#             self.Gain=self.GD["Deconv"]["Gain"]
#         else:
#             self.Gain=Gain
# >>>>>>> issue-255
        self.GainMachine=GainMachine
        self.DicoSMStacked={}
        self.DicoSMStacked["Comp"]={}

    def setRefFreq(self,RefFreq):
        if self.RefFreq is not None:
            print>>log,ModColor.Str("Reference frequency already set to %f MHz"%(self.RefFreq/1e6))
            return
        self.RefFreq=RefFreq
        self.DicoSMStacked["RefFreq"]=RefFreq
        #self.DicoSMStacked["AllFreqs"]=np.array(AllFreqs)

    def ToFile(self,FileName,DicoIn=None):
        print>>log, "Saving dico model to %s"%FileName
        if DicoIn is None:
            D=self.DicoSMStacked
        else:
            D=DicoIn

        D["ListScales"]=self.ListScales
        D["ModelShape"]=self.ModelShape
        MyPickle.Save(D,FileName)

    def FromFile(self,FileName):
        print>>log, "Reading dico model from %s"%FileName
        self.DicoSMStacked=MyPickle.Load(FileName)
        self.FromDico(self.DicoSMStacked)


    def FromDico(self,DicoSMStacked):
        self.DicoSMStacked=DicoSMStacked
        self.RefFreq=self.DicoSMStacked["RefFreq"]
        self.ListScales=self.DicoSMStacked["ListScales"]
        self.ModelShape=self.DicoSMStacked["ModelShape"]

    # def setModelShape(self,ModelShape):
    #     self.ModelShape=ModelShape
    #
    # def AppendComponentToDictStacked(self,key,Fpol,Sols,pol_array_index=0):
    #     """
    #     Adds component to model dictionary (with key l,m location tupple). Each
    #     component may contain #basis_functions worth of solutions. Note that
    #     each basis solution will have multiple Stokes components associated to it.
    #     Args:
    #         key: the (l,m) centre of the component
    #         Fpol: Weight of the solution
    #         Sols: Nd array of solutions with length equal to the number of basis functions representing the component.
    #         pol_array_index: Index of the polarization (assumed 0 <= pol_array_index < number of Stokes terms in the model)
    #     Post conditions:
    #     Added component list to dictionary (with keys (l,m) coordinates). This dictionary is stored in
    #     self.DicoSMStacked["Comp"] and has keys:
    #         "SolsArray": solutions ndArray with shape [#basis_functions,#stokes_terms]
    #         "SumWeights": weights ndArray with shape [#stokes_terms]
    #     """
    #     nchan, npol, nx, ny = self.ModelShape
    #     if not (pol_array_index >= 0 and pol_array_index < npol):
    #         raise ValueError("Pol_array_index must specify the index of the slice in the "
    #                          "model cube the solution should be stored at. Please report this bug.")
    #     DicoComp=self.DicoSMStacked["Comp"]
    #     if not(key in DicoComp.keys()):
    #         DicoComp[key]={}
    #         for p in range(npol):
    #             DicoComp[key]["SolsArray"]=np.zeros((Sols.size,npol),np.float32)
    #             DicoComp[key]["SumWeights"]=np.zeros((npol),np.float32)
    #
    #     Weight=1.
    #     Gain=self.GainMachine.GiveGain()
    #
    #     SolNorm=Sols.ravel()*Gain*np.mean(Fpol)
    #
    #     DicoComp[key]["SumWeights"][pol_array_index] += Weight
    #     DicoComp[key]["SolsArray"][:,pol_array_index] += Weight*SolNorm
    #
    # def setListComponants(self,ListScales):
    #     self.ListScales=ListScales
    #
    #
    # def GiveSpectralIndexMap(self,CellSizeRad=1.,GaussPars=[(1,1,0)],DoConv=True,MaxSpi=100,MaxDR=1e+6):
    #     dFreq=1e6
    #     f0=self.DicoSMStacked["AllFreqs"].min()
    #     f1=self.DicoSMStacked["AllFreqs"].max()
    #     M0=self.GiveModelImage(f0)
    #     M1=self.GiveModelImage(f1)
    #     if DoConv:
    #         M0=ModFFTW.ConvolveGaussian(M0,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
    #         M1=ModFFTW.ConvolveGaussian(M1,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
    #
    #     # compute threshold for alpha computation by rounding DR threshold to .1 digits (i.e. 1.65e-6 rounds to 1.7e-6)
    #     minmod = float("%.1e"%(abs(M0.max())/MaxDR))
    #     # mask out pixels above threshold
    #     mask=(M1<minmod)|(M0<minmod)
    #     print>>log,"computing alpha map for model pixels above %.1e Jy (based on max DR setting of %g)"%(minmod,MaxDR)
    #     with np.errstate(invalid='ignore'):
    #         alpha = (np.log(M0)-np.log(M1))/(np.log(f0/f1))
    #     alpha[mask] = 0
    #     # mask out |alpha|>MaxSpi. These are not physically meaningful anyway
    #     mask = alpha>MaxSpi
    #     alpha[mask]  = MaxSpi
    #     masked = mask.any()
    #     mask = alpha<-MaxSpi
    #     alpha[mask] = -MaxSpi
    #     if masked or mask.any():
    #         print>>log,ModColor.Str("WARNING: some alpha pixels outside +/-%g. Masking them."%MaxSpi,col="red")
    #     return alpha
    #
    # def GiveModelList(self):
    #     """
    #     Iterates through components in the "Comp" dictionary of DicoSMStacked,
    #     returning a list of model sources in tuples looking like
    #     (model_type, coord, flux, ref_freq, alpha, model_params).
    #     model_type is obtained from self.ListScales
    #     coord is obtained from the keys of "Comp"
    #     flux is obtained from the entries in Comp["SolsArray"]
    #     ref_freq is obtained from DicoSMStacked["RefFreq"]
    #     alpha is obtained from self.ListScales
    #     model_params is obtained from self.ListScales
    #     If multiple scales exist, multiple sources will be created
    #     at the same position, but different fluxes, alphas etc.
    #     """
    #     DicoComp = self.DicoSMStacked["Comp"]
    #     ref_freq = self.DicoSMStacked["RefFreq"]
    #
    #     # Assumptions:
    #     # DicoSMStacked is a dictionary of "Solution" dictionaries
    #     # keyed on (l, m), corresponding to some point or
    #     # gaussian source. Depending on whether this is multi-scale,
    #     # there may be multiple solutions (intensities) for a source in SolsArray.
    #     # Components associated with the source for each scale are
    #     # located in self.ListScales.
    #
    #     def _model_map(coord, component):
    #         """
    #         Given a coordinate and component obtained from DicoMap
    #         returns a tuple with the following information
    #         (ModelType, coordinate, vector of STOKES solutions per basis function, alpha, shape data)
    #         """
    #         sa = component["SolsArray"]
    #
    #         return map(lambda (i, sol, ls): (ls["ModelType"],               # type
    #                                          coord,                         # coordinate
    #                                          sol,                           # vector of STOKES parameters
    #                                          ref_freq,                      # reference frequency
    #                                          ls.get("Alpha", 0.0),          # alpha
    #                                          ls.get("ModelParams", None)),  # shape
    #            map(lambda i: (i, sa[i], self.ListScales[i]), range(sa.size)))
    #
    #     # Lazily iterate through DicoComp entries and associated ListScales and SolsArrays,
    #     # assigning values to arrays
    #     source_iter = itertools.chain.from_iterable(_model_map(coord, comp)
    #         for coord, comp in DicoComp.iteritems())
    #
    #     # Create list with iterator results
    #     return [s for s in source_iter]
    #
    # def GiveModelImage(self,FreqIn=None):
    #
    #     RefFreq=self.DicoSMStacked["RefFreq"]
    #     if FreqIn is None:
    #         FreqIn=np.array([RefFreq])
    #
    #     #if type(FreqIn)==float:
    #     #    FreqIn=np.array([FreqIn]).flatten()
    #     #if type(FreqIn)==np.ndarray:
    #
    #     FreqIn=np.array([FreqIn.ravel()]).flatten()
    #
    #     DicoComp=self.DicoSMStacked["Comp"]
    #     _,npol,nx,ny=self.ModelShape
    #     N0=nx
    #
    #     nchan=FreqIn.size
    #     ModelImage=np.zeros((nchan,npol,nx,ny),dtype=np.float32)
    #     DicoSM={}
    #     for key in DicoComp.keys():
    #         for pol in range(npol):
    #             Sol=DicoComp[key]["SolsArray"][:,pol]#/self.DicoSMStacked[key]["SumWeights"]
    #             x,y=key
    #
    #             #print>>log, "%s : %s"%(str(key),str(Sol))
    #
    #             for iFunc in range(Sol.size):
    #                 ThisComp=self.ListScales[iFunc]
    #                 ThisAlpha=ThisComp["Alpha"]
    #                 for ch in range(nchan):
    #                     Flux=Sol[iFunc]*(FreqIn[ch]/RefFreq)**(ThisAlpha)
    #                     if ThisComp["ModelType"]=="Delta":
    #                         ModelImage[ch,pol,x,y]+=Flux
    #
    #                     elif ThisComp["ModelType"]=="Gaussian":
    #                         Gauss=ThisComp["Model"]
    #                         Sup,_=Gauss.shape
    #                         x0,x1=x-Sup/2,x+Sup/2+1
    #                         y0,y1=y-Sup/2,y+Sup/2+1
    #
    #
    #                         Aedge,Bedge=GiveEdges((x,y),N0,(Sup/2,Sup/2),Sup)
    #                         x0d,x1d,y0d,y1d=Aedge
    #                         x0p,x1p,y0p,y1p=Bedge
    #
    #                         ModelImage[ch,pol,x0d:x1d,y0d:y1d]+=Gauss[x0p:x1p,y0p:y1p]*Flux
    #
    #     # vmin,vmax=np.min(self._MeanDirtyOrig[0,0]),np.max(self._MeanDirtyOrig[0,0])
    #     # vmin,vmax=-1,1
    #     # #vmin,vmax=np.min(ModelImage),np.max(ModelImage)
    #     # pylab.clf()
    #     # ax=pylab.subplot(1,3,1)
    #     # pylab.imshow(self._MeanDirtyOrig[0,0],interpolation="nearest",vmin=vmin,vmax=vmax)
    #     # pylab.subplot(1,3,2,sharex=ax,sharey=ax)
    #     # pylab.imshow(self._MeanDirty[0,0],interpolation="nearest",vmin=vmin,vmax=vmax)
    #     # pylab.colorbar()
    #     # pylab.subplot(1,3,3,sharex=ax,sharey=ax)
    #     # pylab.imshow( ModelImage[0,0],interpolation="nearest",vmin=vmin,vmax=vmax)
    #     # pylab.colorbar()
    #     # pylab.draw()
    #     # pylab.show(False)
    #     # print np.max(ModelImage[0,0])
    #     # # stop
    #
    #
    #     return ModelImage
    #
    # def CleanNegComponants(self,box=20,sig=3,RemoveNeg=True):
    #     print>>log, "Cleaning model dictionary from negative componants with (box, sig) = (%i, %i)"%(box,sig)
    #     ModelImage=self.GiveModelImage(self.DicoSMStacked["RefFreq"])[0,0]
    #
    #     Min=scipy.ndimage.filters.minimum_filter(ModelImage,(box,box))
    #     Min[Min>0]=0
    #     Min=-Min
    #
    #     if RemoveNeg==False:
    #         Lx,Ly=np.where((ModelImage<sig*Min)&(ModelImage!=0))
    #     else:
    #         print>>log, "  Removing neg componants too"
    #         Lx,Ly=np.where( ((ModelImage<sig*Min)&(ModelImage!=0)) | (ModelImage<0))
    #
    #     for icomp in range(Lx.size):
    #         key=Lx[icomp],Ly[icomp]
    #         try:
    #             del(self.DicoSMStacked["Comp"][key])
    #         except:
    #             print>>log, "  Componant at (%i, %i) not in dict "%key
    #
    # def CleanMaskedComponants(self,MaskName):
    #     print>>log, "Cleaning model dictionary from masked componants using %s"%(MaskName)
    #     im=image(MaskName)
    #     MaskArray=im.getdata()[0,0].T[::-1]
    #     for (x,y) in self.DicoSMStacked["Comp"].keys():
    #         if MaskArray[x,y]==0:
    #             del(self.DicoSMStacked["Comp"][(x,y)])

# ### Not being used anymore
#     def DelAllComp(self):
#         for key in self.DicoSMStacked["Comp"].keys():
#             del(self.DicoSMStacked["Comp"][key])
#
#
#     def PutBackSubsComps(self):
#         #if self.GD["Data"]["RestoreDico"] is None: return
#
#         SolsFile=self.GD["DDESolutions"]["DDSols"]
#         if not(".npz" in SolsFile):
#             Method=SolsFile
#             ThisMSName=reformat.reformat(os.path.abspath(self.GD["Data"]["MS"]),LastSlash=False)
#             SolsFile="%s/killMS.%s.sols.npz"%(ThisMSName,Method)
#         DicoSolsFile=np.load(SolsFile)
#         SourceCat=DicoSolsFile["SourceCatSub"]
#         SourceCat=SourceCat.view(np.recarray)
#         #RestoreDico=self.GD["Data"]["RestoreDico"]
#         RestoreDico=DicoSolsFile["ModelName"][()][0:-4]+".DicoModel"
#
#         print>>log, "Adding previously substracted components"
#         ModelMachine0=ClassModelMachine(self.GD)
#
#
#         ModelMachine0.FromFile(RestoreDico)
#
#
#
#         _,_,nx0,ny0=ModelMachine0.DicoSMStacked["ModelShape"]
#
#         _,_,nx1,ny1=self.ModelShape
#         dx=nx1-nx0
#
#
#
#         for iSource in range(SourceCat.shape[0]):
#             x0=SourceCat.X[iSource]
#             y0=SourceCat.Y[iSource]
#
#             x1=x0+dx
#             y1=y0+dx
#
#             if not((x1,y1) in self.DicoSMStacked["Comp"].keys()):
#                 self.DicoSMStacked["Comp"][(x1,y1)]=ModelMachine0.DicoSMStacked["Comp"][(x0,y0)]
#             else:
#                 self.DicoSMStacked["Comp"][(x1,y1)]+=ModelMachine0.DicoSMStacked["Comp"][(x0,y0)]
