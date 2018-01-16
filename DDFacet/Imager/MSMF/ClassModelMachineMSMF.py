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

import itertools

import numpy as np
from DDFacet.Other import MyLogger
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import ModColor
log=MyLogger.getLogger("ClassModelMachineMSMF")
from DDFacet.Array import NpParallel
from DDFacet.Array import ModLinAlg
from DDFacet.ToolsDir import ModFFTW
from DDFacet.ToolsDir import ModToolBox
from DDFacet.Other import ClassTimeIt
from DDFacet.Other import MyPickle
from DDFacet.Other import reformat
from DDFacet.Imager import ClassFrequencyMachine
from DDFacet.ToolsDir.GiveEdges import GiveEdges
from DDFacet.ToolsDir.GiveEdges import GiveEdgesDissymetric
from DDFacet.Imager import ClassModelMachine as ClassModelMachinebase
from DDFacet.ToolsDir import ModFFTW
import scipy.ndimage
from SkyModel.Sky import ModRegFile
from pyrap.images import image
from SkyModel.Sky import ClassSM
import os

class ClassModelMachine(ClassModelMachinebase.ClassModelMachine):
    def __init__(self,*args,**kwargs):
        ClassModelMachinebase.ClassModelMachine.__init__(self, *args, **kwargs)
        self.Test = True
        # self.GD=GD
        # if Gain is None:
        #     self.Gain=self.GD["Deconv"]["Gain"]
        # else:
        #     self.Gain=Gain
        # self.GainMachine=GainMachine
        # self.DicoSMStacked={}
        # self.DicoSMStacked["Comp"]={}
        self.DicoSMStacked={}
        self.DicoSMStacked["Type"]="HMP"

    def setRefFreq(self,RefFreq,Force=False):#,AllFreqs):
        if self.RefFreq is not None and not Force:
            print>>log,ModColor.Str("Reference frequency already set to %f MHz"%(self.RefFreq/1e6))
            return

        self.RefFreq=RefFreq
        self.DicoSMStacked["RefFreq"]=RefFreq
        #self.DicoSMStacked["AllFreqs"]=np.array(AllFreqs)

    def setFreqMachine(self,GridFreqs, DegridFreqs):
        # Initiaise the Frequency Machine
        self.FreqMachine = ClassFrequencyMachine.ClassFrequencyMachine(GridFreqs, DegridFreqs, self.DicoSMStacked["RefFreq"], self.GD)

    def ToFile(self,FileName,DicoIn=None):
        print>>log, "Saving dico model to %s"%FileName
        if DicoIn is None:
            D=self.DicoSMStacked
        else:
            D=DicoIn

        D["GD"]=self.GD
        D["Type"]="HMP"
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



    def setModelShape(self,ModelShape):
        self.ModelShape=ModelShape

    def AppendComponentToDictStacked(self,key,Fpol,Sols,pol_array_index=0):
        """
        Adds component to model dictionary (with key l,m location tupple). Each
        component may contain #basis_functions worth of solutions. Note that
        each basis solution will have multiple Stokes components associated to it.
        Args:
            key: the (l,m) centre of the component
            Fpol: Weight of the solution
            Sols: Nd array of solutions with length equal to the number of basis functions representing the component.
            pol_array_index: Index of the polarization (assumed 0 <= pol_array_index < number of Stokes terms in the model)
        Post conditions:
        Added component list to dictionary (with keys (l,m) coordinates). This dictionary is stored in
        self.DicoSMStacked["Comp"] and has keys:
            "SolsArray": solutions ndArray with shape [#basis_functions,#stokes_terms]
            "SumWeights": weights ndArray with shape [#stokes_terms]
        """
        nchan, npol, nx, ny = self.ModelShape
        if not (pol_array_index >= 0 and pol_array_index < npol):
            raise ValueError("Pol_array_index must specify the index of the slice in the "
                             "model cube the solution should be stored at. Please report this bug.")
        try:
            DicoComp=self.DicoSMStacked["Comp"]
        except:
            self.DicoSMStacked["Comp"]={}
            DicoComp=self.DicoSMStacked["Comp"]


        comp = DicoComp.get(key)
        if comp is None:
            DicoComp[key] = comp = dict(
                                    SolsArray = np.zeros((Sols.size,npol),np.float32),
                                    SumWeights = np.zeros((npol),np.float32))
# =======
#         DicoComp = self.DicoSMStacked["Comp"]
#         entry = DicoComp.setdefault(key, {})
#         if not entry:
#             entry["SolsArray"]  = np.zeros((Sols.size, npol),np.float32)
#             entry["SumWeights"] = np.zeros((npol),np.float32)
# >>>>>>> issue-255

        Weight=1.
        #Gain=self.GainMachine.GiveGain()
        Gain=self.GD["Deconv"]["Gain"]

        SolNorm=Sols.ravel()*Gain*np.mean(Fpol)

        comp["SumWeights"][pol_array_index] += Weight
        comp["SolsArray"][:,pol_array_index] += Weight*SolNorm
# =======

#         entry["SumWeights"][pol_array_index] += Weight
#         entry["SolsArray"][:,pol_array_index] += Weight*SolNorm
# >>>>>>> issue-255

    def setListComponants(self,ListScales):
        self.ListScales=ListScales

    # def GiveSpectralIndexMap(self, threshold=0.1, save_dict=True):
    #     # Get the model image
    #     IM = self.GiveModelImage(self.FreqMachine.Freqsp)
    #     nchan, npol, Nx, Ny = IM.shape
    #     # Fit the alpha map
    #     self.FreqMachine.FitAlphaMap(IM[:, 0, :, :],
    #                                  threshold=threshold)  # should set threshold based on SNR of final residual
    #     if save_dict:
    #         FileName = self.GD['Output']['Name'] + ".Dicoalpha"
    #         print>> log, "Saving componentwise SPI map to %s" % FileName
    #         MyPickle.Save(self.FreqMachine.alpha_dict, FileName)
    #     #return self.FreqMachine.weighted_alpha_map.reshape((1, 1, Nx, Ny))
    #     return self.FreqMachine.alpha_map.reshape((1, 1, Nx, Ny))

    def GiveSpectralIndexMap(self,CellSizeRad=1.,GaussPars=[(1,1,0)],DoConv=True,MaxSpi=100,MaxDR=1e+6,threshold=None):
        dFreq=1e6
        #f0=self.DicoSMStacked["AllFreqs"].min()
        #f1=self.DicoSMStacked["AllFreqs"].max()
        RefFreq=self.DicoSMStacked["RefFreq"]
        f0=RefFreq/1.5
        f1=RefFreq*1.5
    
        M0=self.GiveModelImage(f0)
        M1=self.GiveModelImage(f1)
        if DoConv:
            #M0=ModFFTW.ConvolveGaussian(M0,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
            #M1=ModFFTW.ConvolveGaussian(M1,CellSizeRad=CellSizeRad,GaussPars=GaussPars)
            #M0,_=ModFFTW.ConvolveGaussianWrapper(M0,Sig=GaussPars[0][0]/CellSizeRad)
            #M1,_=ModFFTW.ConvolveGaussianWrapper(M1,Sig=GaussPars[0][0]/CellSizeRad)
            M0,_=ModFFTW.ConvolveGaussianScipy(M0,Sig=GaussPars[0][0]/CellSizeRad)
            M1,_=ModFFTW.ConvolveGaussianScipy(M1,Sig=GaussPars[0][0]/CellSizeRad)
            

        #print M0.shape,M1.shape
        # compute threshold for alpha computation by rounding DR threshold to .1 digits (i.e. 1.65e-6 rounds to 1.7e-6)
        if threshold is not None:
            minmod = threshold
        elif not np.all(M0==0):
            minmod = float("%.1e"%(np.max(np.abs(M0))/MaxDR))
        else:
            minmod=1e-6
    
        # mask out pixels above threshold
        mask=(M1<minmod)|(M0<minmod)
        print>>log,"computing alpha map for model pixels above %.1e Jy (based on max DR setting of %g)"%(minmod,MaxDR)
        M0[mask]=minmod
        M1[mask]=minmod
        #with np.errstate(invalid='ignore'):
        #    alpha = (np.log(M0)-np.log(M1))/(np.log(f0/f1))
        # print
        # print np.min(M0),np.min(M1),minmod
        # print
        alpha = (np.log(M0)-np.log(M1))/(np.log(f0/f1))
        alpha[mask] = 0
    
        # mask out |alpha|>MaxSpi. These are not physically meaningful anyway
        mask = alpha>MaxSpi
        alpha[mask]  = MaxSpi
        masked = mask.any()
        mask = alpha<-MaxSpi
        alpha[mask] = -MaxSpi
        if masked or mask.any():
            print>>log,ModColor.Str("WARNING: some alpha pixels outside +/-%g. Masking them."%MaxSpi,col="red")
        return alpha

    def GiveModelList(self):
        """
        Iterates through components in the "Comp" dictionary of DicoSMStacked,
        returning a list of model sources in tuples looking like
        (model_type, coord, flux, ref_freq, alpha, model_params).

        model_type is obtained from self.ListScales
        coord is obtained from the keys of "Comp"
        flux is obtained from the entries in Comp["SolsArray"]
        ref_freq is obtained from DicoSMStacked["RefFreq"]
        alpha is obtained from self.ListScales
        model_params is obtained from self.ListScales

        If multiple scales exist, multiple sources will be created
        at the same position, but different fluxes, alphas etc.

        """
        DicoComp = self.DicoSMStacked["Comp"]
        ref_freq = self.DicoSMStacked["RefFreq"]

        # Assumptions:
        # DicoSMStacked is a dictionary of "Solution" dictionaries
        # keyed on (l, m), corresponding to some point or
        # gaussian source. Depending on whether this is multi-scale,
        # there may be multiple solutions (intensities) for a source in SolsArray.
        # Components associated with the source for each scale are
        # located in self.ListScales.

        def _model_map(coord, component):
            """
            Given a coordinate and component obtained from DicoMap
            returns a tuple with the following information
            (ModelType, coordinate, vector of STOKES solutions per basis function, alpha, shape data)
            """
            sa = component["SolsArray"]

            return map(lambda (i, sol, ls): (ls["ModelType"],               # type
                                             coord,                         # coordinate
                                             sol,                           # vector of STOKES parameters
                                             ref_freq,                      # reference frequency
                                             ls.get("Alpha", 0.0),          # alpha
                                             ls.get("ModelParams", None)),  # shape
               map(lambda i: (i, sa[i], self.ListScales[i]), range(sa.size)))

        # Lazily iterate through DicoComp entries and associated ListScales and SolsArrays,
        # assigning values to arrays
        source_iter = itertools.chain.from_iterable(_model_map(coord, comp)
            for coord, comp in DicoComp.iteritems())

        # Create list with iterator results
        return [s for s in source_iter]

    def GiveModelImage(self,FreqIn=None,out=None,DoAbs=False):
        """
        Renders a model image at the specified frequency(ies)
        Args:
            FreqIn: scalar or vector of frequencies
            out: if not None, image to be rendered into. Must have correct shape.

        Returns:
            Model image
        """
        if DoAbs:
            f_apply=np.abs
        else:
            f_apply=lambda x:x
        RefFreq=self.DicoSMStacked["RefFreq"]
        if FreqIn is None:
            FreqIn=np.array([RefFreq])

        #if type(FreqIn)==float:
        #    FreqIn=np.array([FreqIn]).flatten()
        #if type(FreqIn)==np.ndarray:

        FreqIn=np.array([FreqIn.ravel()]).flatten()

        _,npol,nx,ny=self.ModelShape
        N0x=nx
        N0y=ny

        nchan=FreqIn.size
        if out is not None:
            if out.shape != (nchan,npol,nx,ny) or out.dtype != np.float32:
                raise RuntimeError("supplied image has incorrect type (%s) or shape (%s)" % (out.dtype, out.shape))
            ModelImage = out
        else:
            ModelImage = np.zeros((nchan,npol,nx,ny),dtype=np.float32)

        if "Comp" not in  self.DicoSMStacked.keys():
            return ModelImage

        DicoComp=self.DicoSMStacked["Comp"]

        DicoSM={}
        for key in DicoComp.keys():
            for pol in range(npol):
                Sol=DicoComp[key]["SolsArray"][:,pol]#/self.DicoSMStacked[key]["SumWeights"]
                x,y=key

                #print>>log, "%s : %s"%(str(key),str(Sol))

                for iFunc in range(Sol.size):
                    ThisComp=self.ListScales[iFunc]
                    ThisAlpha=ThisComp["Alpha"]
                    for ch in range(nchan):
                        Flux=Sol[iFunc]*(FreqIn[ch]/RefFreq)**(ThisAlpha)
                        Flux=f_apply(Flux)
                        if ThisComp["ModelType"]=="Delta":
                            ModelImage[ch,pol,x,y]+=Flux

                        elif ThisComp["ModelType"]=="Gaussian":
                            Gauss=ThisComp["Model"]
                            Sup,_=Gauss.shape
                            x0,x1=x-Sup/2,x+Sup/2+1
                            y0,y1=y-Sup/2,y+Sup/2+1


                            Aedge,Bedge=GiveEdgesDissymetric((x,y),(N0x,N0y),(Sup/2,Sup/2),(Sup,Sup))
                            x0d,x1d,y0d,y1d=Aedge
                            x0p,x1p,y0p,y1p=Bedge

                            ModelImage[ch,pol,x0d:x1d,y0d:y1d]+=Gauss[x0p:x1p,y0p:y1p]*Flux

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

    def CleanNegComponants(self,box=20,sig=3,RemoveNeg=True):
        print>>log, "Cleaning model dictionary from negative components with (box, sig) = (%i, %i)"%(box,sig)
        ModelImage=self.GiveModelImage(self.DicoSMStacked["RefFreq"])[0,0]

        Min=scipy.ndimage.filters.minimum_filter(ModelImage,(box,box))
        Min[Min>0]=0
        Min=-Min

        if RemoveNeg==False:
            Lx,Ly=np.where((ModelImage<sig*Min)&(ModelImage!=0))
        else:
            print>>log, "  Removing neg components too"
            Lx,Ly=np.where( ((ModelImage<sig*Min)&(ModelImage!=0)) | (ModelImage<0))

        for icomp in range(Lx.size):
            key=Lx[icomp],Ly[icomp]
            try:
                del(self.DicoSMStacked["Comp"][key])
            except:
                print>>log, "  Component at (%i, %i) not in dict "%key

    def CleanMaskedComponants(self,MaskName):
        print>>log, "Cleaning model dictionary from masked components using %s"%(MaskName)
        im=image(MaskName)
        MaskArray=im.getdata()[0,0].T[::-1]
        for (x,y) in self.DicoSMStacked["Comp"].keys():
            if MaskArray[x,y]==0:
                del(self.DicoSMStacked["Comp"][(x,y)])

    def ToNPYModel(self,FitsFile,SkyModel,BeamImage=None):
        #R=ModRegFile.RegToNp(PreCluster)
        #R.Read()
        #R.Cluster()
        #PreClusterCat=R.CatSel
        #ExcludeCat=R.CatExclude


        AlphaMap=self.GiveSpectralIndexMap(DoConv=False)
        ModelMap=self.GiveModelImage()
        nch,npol,_,_=ModelMap.shape

        for ch in range(nch):
            for pol in range(npol):
                ModelMap[ch,pol]=ModelMap[ch,pol][::-1]#.T
                AlphaMap[ch,pol]=AlphaMap[ch,pol][::-1]#.T

        if BeamImage is not None:
            ModelMap*=(BeamImage)

        im=image(FitsFile)
        pol,freq,decc,rac=im.toworld((0,0,0,0))

        Lx,Ly=np.where(ModelMap[0,0]!=0)

        X=np.array(Lx)
        Y=np.array(Ly)

        #pol,freq,decc1,rac1=im.toworld((0,0,1,0))
        dx=abs(im.coordinates().dict()["direction0"]["cdelt"][0])

        SourceCat=np.zeros((100000,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('Sref',np.float),('I',np.float),('Q',np.float),\
                                           ('U',np.float),('V',np.float),('RefFreq',np.float),('alpha',np.float),('ESref',np.float),\
                                           ('Ealpha',np.float),('kill',np.int),('Cluster',np.int),('Type',np.int),('Gmin',np.float),\
                                           ('Gmaj',np.float),('Gangle',np.float),("Select",np.int),('l',np.float),('m',np.float),("Exclude",bool),
                                           ("X",np.int32),("Y",np.int32)])
        SourceCat=SourceCat.view(np.recarray)

        IndSource=0

        SourceCat.RefFreq[:]=self.DicoSMStacked["RefFreq"]
        _,_,nx,ny=ModelMap.shape

        for iSource in range(X.shape[0]):
            x_iSource,y_iSource=X[iSource],Y[iSource]
            _,_,dec_iSource,ra_iSource=im.toworld((0,0,y_iSource,x_iSource))
            SourceCat.ra[iSource]=ra_iSource
            SourceCat.dec[iSource]=dec_iSource
            SourceCat.X[iSource]=(nx-1)-X[iSource]
            SourceCat.Y[iSource]=Y[iSource]

            #print self.DicoSMStacked["Comp"][(SourceCat.X[iSource],SourceCat.Y[iSource])]
            # SourceCat.Cluster[IndSource]=iCluster
            Flux=ModelMap[0,0,x_iSource,y_iSource]
            Alpha=AlphaMap[0,0,x_iSource,y_iSource]
            # print iSource,"/",X.shape[0],":",x_iSource,y_iSource,Flux,Alpha
            SourceCat.I[iSource]=Flux
            SourceCat.alpha[iSource]=Alpha


        SourceCat=(SourceCat[SourceCat.ra!=0]).copy()
        np.save(SkyModel,SourceCat)
        self.AnalyticSourceCat=ClassSM.ClassSM(SkyModel)

    def DelAllComp(self):
        for key in self.DicoSMStacked["Comp"].keys():
            del(self.DicoSMStacked["Comp"][key])


    def PutBackSubsComps(self):
        #if self.GD["Data"]["RestoreDico"] is None: return

        SolsFile=self.GD["DDESolutions"]["DDSols"]
        if not(".npz" in SolsFile):
            Method=SolsFile
            ThisMSName=reformat.reformat(os.path.abspath(self.GD["Data"]["MS"]),LastSlash=False)
            SolsFile="%s/killMS.%s.sols.npz"%(ThisMSName,Method)
        DicoSolsFile=np.load(SolsFile)
        SourceCat=DicoSolsFile["SourceCatSub"]
        SourceCat=SourceCat.view(np.recarray)
        #RestoreDico=self.GD["Data"]["RestoreDico"]
        RestoreDico=DicoSolsFile["ModelName"][()][0:-4]+".DicoModel"

        print>>log, "Adding previously subtracted components"
        ModelMachine0=ClassModelMachine(self.GD)


        ModelMachine0.FromFile(RestoreDico)



        _,_,nx0,ny0=ModelMachine0.DicoSMStacked["ModelShape"]

        _,_,nx1,ny1=self.ModelShape
        dx=nx1-nx0



        for iSource in range(SourceCat.shape[0]):
            x0=SourceCat.X[iSource]
            y0=SourceCat.Y[iSource]

            x1=x0+dx
            y1=y0+dx

            if not((x1,y1) in self.DicoSMStacked["Comp"].keys()):
                self.DicoSMStacked["Comp"][(x1,y1)]=ModelMachine0.DicoSMStacked["Comp"][(x0,y0)]
            else:
                self.DicoSMStacked["Comp"][(x1,y1)]+=ModelMachine0.DicoSMStacked["Comp"][(x0,y0)]

