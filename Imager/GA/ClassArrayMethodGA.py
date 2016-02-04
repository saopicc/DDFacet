from DDFacet.ToolsDir import ModFFTW
import numpy as np
import random
from deap import tools
import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
from DDFacet.Other import ClassTimeIt
from itertools import repeat
import collections


from ClassParamMachine import ClassParamMachine
from DDFacet.ToolsDir.GeneDist import ClassDistMachine


class ClassArrayMethodGA():
    def __init__(self,Dirty,PSF,ListPixParms,ListPixData,FreqsInfo,GD=None,PixVariance=1.e-2):

        self.Dirty=Dirty
        self.PSF=PSF
        self.ListPixParms=ListPixParms
        self.ListPixData=ListPixData
        self.NPixListParms=len(ListPixParms)
        self.NPixListData=len(ListPixData)
        self.GD=GD
        self.WeightMaxFunc=collections.OrderedDict()
        self.WeightMaxFunc["Chi2"]=1.
        self.WeightMaxFunc["MinFlux"]=1.
        #self.WeightMaxFunc["L0"]=1.
        self.MaxFunc=self.WeightMaxFunc.keys()
        
        self.NFuncMin=len(self.MaxFunc)
        self.WeightsEA=[1.]#*len(self.MaxFunc)
        #self.WeightsEA=[1.]*len(self.MaxFunc)
        self.MinVar=np.array([0.01,0.01])
        self.PixVariance=PixVariance
        self.FreqsInfo=FreqsInfo
        
        self.NFreqBands,self.npol,self.NPixPSF,_=PSF.shape
        self.PM=ClassParamMachine(self,self.GD)
        self.PM.setFreqs(FreqsInfo)

        self.SetConvMatrix()
        self.NParms=self.NPixListParms*self.PM.NParam
        self.DataTrue=None
        pylab.figure(3,figsize=(5,3))
        pylab.clf()
        pylab.figure(4,figsize=(5,3))
        pylab.clf()

    def SetConvMatrix(self):
        PSF=self.PSF
        NPixPSF=PSF.shape[-1]
        if self.ListPixData==None:
            x,y=np.mgrid[0:NPixPSF:1,0:NPixPSF:1]
            self.ListPixData=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()
        if self.ListPixParms==None:
            x,y=np.mgrid[0:NPixPSF:1,0:NPixPSF:1]
            self.ListPixParms=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()


        M=np.zeros((self.NFreqBands,1,self.NPixListData,self.NPixListParms),np.float32)
        xc=yc=NPixPSF/2
        self.DirtyArray=np.zeros((self.NFreqBands,1,self.NPixListData),np.float32)

        for iBand in range(self.NFreqBands):
            for iPix in range(self.NPixListData):
                x0,y0=self.ListPixData[iPix]
                self.DirtyArray[iBand,0,iPix]=self.Dirty[iBand,0,x0,y0]
                for jPix,(x,y) in zip(range(self.NPixListParms),self.ListPixParms):
                    i,j=(x-x0)+xc,(y-y0)+yc
                    if (i>=0)&(i<NPixPSF)&(j>=0)&(j<NPixPSF):
                        M[iBand,0,iPix,jPix]=PSF[iBand,0,i,j]

        self.DirtyArrayMean=np.mean(self.DirtyArray,axis=0).reshape((1,1,self.NPixListData))
        self.DirtyArrayAbsMean=np.mean(np.abs(self.DirtyArray),axis=0).reshape((1,1,self.NPixListData))
        self.CM=M
    
    def Convolve(self,A):
        sh=A.shape
        ConvA=np.zeros((self.NFreqBands,1,self.NPixListData),np.float32)
        for iBand in range(self.NFreqBands):
            AThisBand=A[iBand]
            CF=self.CM[iBand,0]
            ConvA[iBand,0]=np.dot(CF,AThisBand.reshape((AThisBand.size,1))).reshape((self.NPixListData,))
        return ConvA

    def ToConvArray(self,V):
        A=self.PM.GiveModelArray(V)
        #A=ModFFTW.ConvolveGaussian(A,CellSizeRad=1,GaussPars=[(1.,1.,0.)])
        A=self.Convolve(A)
        return A

    def setBestIndiv(self,BestIndiv):
        self.BestContinuousFitNess=BestIndiv.ContinuousFitNess

    def GiveDecreteFitNess(self,ContinuousFitNess):
        # M=np.concatenate([0.01*np.abs(self.BestContinuousFitNess),self.MinVar]).reshape((2,self.NFuncMin))
        # Sig=np.max(M,axis=0)
        # Sig.fill(.01)
        Sig=self.MinVar

        sh=ContinuousFitNess.shape
        #d=(ContinuousFitNess-self.BestContinuousFitNess)/Sig
        d=(ContinuousFitNess)/Sig
        DecreteFitNess=np.array(np.int64(np.round(d))).reshape(sh)
        #DecreteFitNess=np.array(d).reshape(sh)

        # print "=============================="
        # print "Best",self.BestContinuousFitNess
        # print "In  ",ContinuousFitNess
        # print "Sig ",Sig
        # print "Out ",DecreteFitNess
        return DecreteFitNess
    
    def GiveFitness(self,individual):

        # individual.fill(-0.8)
        A=self.ToConvArray(individual)
        fitness=0.
        Resid=self.DirtyArray-A
        
        S=self.PM.ArrayToSubArray(individual,"S")
        
        ContinuousFitNess=[]
        for FuncType in self.MaxFunc:
            if FuncType=="Chi2":
                chi2=-np.sum((Resid)**2)/(self.PixVariance*Resid.size)
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(chi2*W)
            if FuncType=="L0":
                # ResidNonZero=S[S!=0]
                # W=self.WeightMaxFunc[FuncType]
                # l0=-(ResidNonZero.size)
                l0=-self.GiveCompacity(S)

                ContinuousFitNess.append(l0*W)
            if FuncType=="MinFlux":
                SNegArr=np.abs(S[S<0])[()]
                FNeg=-np.sum(SNegArr)/(np.sqrt(self.PixVariance)*Resid.size)
                W=self.WeightMaxFunc[FuncType]
                ContinuousFitNess.append(FNeg*W)

        return np.sum(ContinuousFitNess),

        ContinuousFitNess=np.array(ContinuousFitNess)
        DecreteFitNess=self.GiveDecreteFitNess(ContinuousFitNess)
        rep=DecreteFitNess.tolist()

        # ContinuousFitNess=np.array(ContinuousFitNess)
        # setattr(individual,"ContinuousFitNess",ContinuousFitNess)
        # if "BestContinuousFitNess" in dir(self):
        #     DecreteFitNess=self.GiveDecreteFitNess(ContinuousFitNess)
        #     rep=DecreteFitNess.tolist()
        # else:
        #     rep=ContinuousFitNess.tolist()#np.sum(ContinuousFitNess)

        return rep
        # return l0,FNeg,chi2#,STot
        # print chi2,l0,FNeg
        # return FNeg,chi2,l0
        # return fitness,#l0
    
    def testMovePix(self):
        A=np.random.randn(self.PM.NParam,self.NPixListParms)
        A.fill(0.)
        A[:,10]=1.
        print A.shape

        ArrayModel=self.PM.GiveModelArray(A)
        A0=self.PM.ModelToSquareArray(ArrayModel,TypeInOut=("Parms","Parms"),DomainOut="Parms").copy()

        for reg in np.linspace(0,0.99,8):


            import pylab
            pylab.clf()
            pylab.imshow(A0[0,0],interpolation="nearest",vmax=1.)
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)


            
            A1=self.MovePix(A.copy().ravel(),10,InReg=reg)

            ArrayModel=self.PM.GiveModelArray(A1)
            A1=self.PM.ModelToSquareArray(ArrayModel,TypeInOut=("Parms","Parms"),DomainOut="Parms").copy()

            #pylab.subplot(1,2,1)
            #pylab.subplot(1,2,2)
            pylab.imshow(A1[0,0],interpolation="nearest",vmax=1.)
            pylab.draw()
            pylab.show(False)
            pylab.pause(0.1)
    
    def MovePix(self,indiv,iPix,alpha=None,FluxWeighted=True,InReg=None):
    
        
        if alpha==None:
            alpha=random.random()
        dx,dy=np.mgrid[-1:1:3*1j,-1:1:3*1j]
        Dx=np.int32(np.concatenate((dx.flatten()[0:4],dx.flatten()[5::])))
        Dy=np.int32(np.concatenate((dy.flatten()[0:4],dy.flatten()[5::])))

        
        
        # ArrayModel=self.PM.GiveModelArray(indiv)
        # ArrayModel_S=self.PM.ArrayToSubArray(indiv,Type="S")
        ArrayModel_S=indiv # ArrayModel_S.reshape((1,ArrayModel_S.size))*np.ones((2,1))
        A=self.PM.ModelToSquareArray(ArrayModel_S,TypeInOut=("Parms","Parms"),DomainOut="Parms")
        
        nf,npol,nx,nx=A.shape
        #A=np.mean(A,axis=0).reshape(1,npol,nx,nx)

        mapi,mapj=self.PM.SquareGrids["Parms"]["MappingIndexToXYPix"]
        i0,j0=mapi[iPix],mapj[iPix]
        FluxWeighted=False
        if FluxWeighted:
            iAround=i0+Dx
            jAround=j0+Dy
            cx=((iAround>=0)&(iAround<nx))
            cy=((jAround>=0)&(jAround<nx))
            indIN=(cx&cy)
            iAround=iAround[indIN]
            jAround=jAround[indIN]
            sAround=A[0,0,iAround,jAround].copy()
            #sInt=np.sum(sAround)
            #sAround[sAround==0]=sInt*0.05
            X=np.arange(iAround.size)
            DM=ClassDistMachine()
            DM.setRefSample(X,W=sAround,Ns=10)
            ind=int(round(DM.GiveSample(1)[0]))
            ind=np.max([0,ind])
            ind=np.min([ind,iAround.size-1])
            ind=indIN[ind]
        else:
            if InReg==None:
                reg=random.random()
            else:
                reg=InReg
            ind=int(reg*8)
        
        i1=i0+Dx[ind]
        j1=j0+Dy[ind]
        f0=alpha*A[0,0,i0,j0]
        
        _,_,nx,ny=A.shape
        condx=((i1>0)&(i1<nx))
        condy=((j1>0)&(j1<ny))
        if condx&condy:
            A[0,0,i1,j1]+=f0
            A[0,0,i0,j0]-=f0
            AParm=self.PM.SquareArrayToModel(A,TypeInOut=("Parms","Parms"))

            ArrayModel_S.flat[:]=AParm.flat[:]
        return indiv
    
    
    def mutGaussian(self,individual, pFlux, p0, pMove):
        #return individual,
        T=ClassTimeIt.ClassTimeIt()
        T.disable()
        size = len(individual)
        #mu = repeat(mu, size)
        #sigma = repeat(sigma, size)

        T.timeit("start0")
        #A0=IndToArray(individual).copy()
        Ps=np.array([pFlux, p0, pMove])
        _p0=p0/np.sum(Ps)
        _pMove=pMove/np.sum(Ps)
        _pFlux=pFlux/np.sum(Ps)
    
        T.timeit("start1")
    
        Af=self.PM.ArrayToSubArray(individual,"S")
        index=np.arange(Af.size)
        ind=np.where(Af!=0.)[0]
        NNonZero=(ind.size)
        if NNonZero==0: return individual,

        T.timeit("start2")
    
        RType=random.random()
        T.timeit("start3")
        if RType < _pFlux:
            Type=0
            N=1
        elif RType < _pFlux+_pMove:
            Type=1
            N=np.max([(NNonZero/10),1])
        else:
            Type=2
            N=1
    
        indR=sorted(list(set(np.int32(np.random.rand(N)*NNonZero).tolist())))
        indSel=ind[indR]
        #Type=0

        #print "Type:",Type

        for iPix in indSel:
            #print iPix,Type
            if Type==0:
                for TypeParm in self.PM.SolveParam:
                    A=self.PM.ArrayToSubArray(individual,TypeParm)
                    if TypeParm=="S":
                        ds=0.1*np.abs(self.DirtyArrayAbsMean.ravel()[iPix]-np.abs(A[iPix]))
                    else:
                        if "Sigma" in self.PM.DicoIParm[TypeParm]["Default"].keys():
                            ds=self.PM.DicoIParm[TypeParm]["Default"]["Sigma"]
                        else:
                            ds=A[iPix]
                    A[iPix] += random.gauss(0, 1.)*ds


            if Type==1:
                for TypeParm in self.PM.SolveParam:
                    A=self.PM.ArrayToSubArray(individual,TypeParm)
                    A[iPix] = 0.#1e-3
    
            if Type==2:
                individual=self.MovePix(individual,iPix)
                    
    
                
            # if individual[i]==0:
            #     if random.random() < indpb/100.:
            #         individual[i] += random.gauss(m, s)
    
        T.timeit("for")
    
        # if Type==2:
        #     A1=IndToArray(individual).copy()
        #     v0,v1=A0.min(),A0.max()
        #     import pylab
        #     pylab.clf()
        #     #pylab.subplot(1,2,1)
        #     pylab.imshow(A0[0,0],interpolation="nearest",vmin=v0,vmax=v1)
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
        #     #pylab.subplot(1,2,2)
        #     pylab.imshow(A1[0,0],interpolation="nearest",vmin=v0,vmax=v1)
        #     pylab.draw()
        #     pylab.show(False)
        #     pylab.pause(0.1)
    
    
        return individual,
    
    
    def Plot(self,pop,iGen):
        for iChannel in range(self.NFreqBands):
            self.PlotChannel(pop,iGen,iChannel=iChannel)

        V = tools.selBest(pop, 1)[0]

        pylab.figure(3,figsize=(5,3))
        #pylab.clf()
        S=self.PM.ArrayToSubArray(V,"S")
        Al=self.PM.ArrayToSubArray(V,"Alpha")

        pylab.subplot(1,2,1)
        pylab.plot(S)
        pylab.subplot(1,2,2)
        pylab.plot(Al)
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        

    def GiveCompacity(self,S):
        DM=ClassDistMachine()
        #S.fill(1)
        #S[0]=100
        DM.setRefSample(np.arange(S.size),W=np.sort(S),Ns=100,xmm=[0,S.size-1])#,W=sAround,Ns=10)
        #DM.setRefSample(S)#,W=sAround,Ns=10)
        xs,ys=DM.xyCumulD
        dx=xs[1]-xs[0]
        I=2.*(S.size-np.sum(ys)*dx)/S.size-1.
        return I
        # pylab.figure(4,figsize=(5,3))
        # pylab.plot(xp,yp)
        # pylab.title("%f"%I)
        # pylab.draw()
        # pylab.show(False)
        # pylab.pause(0.1)
        # stop


    def PlotChannel(self,pop,iGen,iChannel=0):

        best_ind = tools.selBest(pop, 1)[0]
        V=best_ind

        #print self.PM.ArrayToSubArray(V,"Alpha")

        # A=self.PM.GiveModelArray(V)
        # A=self.Convolve(A)

        ConvModelArray=self.ToConvArray(V)
        IM=self.PM.ModelToSquareArray(ConvModelArray,TypeInOut=("Data","Data"))
        Dirty=self.PM.ModelToSquareArray(self.DirtyArray,TypeInOut=("Data","Data"))


        vmin,vmax=np.min([Dirty.min(),0]),Dirty.max()
    
        fig=pylab.figure(iChannel+1,figsize=(5,3))
        pylab.clf()
    
        ax0=pylab.subplot(2,3,1)
        im0=pylab.imshow(Dirty[iChannel,0],interpolation="nearest",vmin=vmin,vmax=vmax)
        pylab.title("Data")
        ax0.axes.get_xaxis().set_visible(False)
        ax0.axes.get_yaxis().set_visible(False)
        divider0 = make_axes_locatable(ax0)
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im0, cax=cax0)
    
        ax1=pylab.subplot(2,3,2)
        im1=pylab.imshow(IM[iChannel,0],interpolation="nearest",vmin=vmin,vmax=vmax)
        pylab.title("Convolved Model")
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im1, cax=cax1)
    
        ax2=pylab.subplot(2,3,3)
        R=Dirty[iChannel,0]-IM[iChannel,0]
        im2=pylab.imshow(R,interpolation="nearest")#,vmin=vmin,vmax=vmax)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)
        pylab.title("Residual Data")
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im2, cax=cax2)
    
    
        #pylab.colorbar()
        if self.DataTrue!=None:
            DataTrue=self.DataTrue
            vmin,vmax=DataTrue.min(),DataTrue.max()
            ax3=pylab.subplot(2,3,4)
            im3=pylab.imshow(DataTrue[iChannel,0],interpolation="nearest",vmin=vmin,vmax=vmax)
            ax3.axes.get_xaxis().set_visible(False)
            ax3.axes.get_yaxis().set_visible(False)
            pylab.title("True Sky")
            divider3 = make_axes_locatable(ax3)
            cax3 = divider3.append_axes("right", size="5%", pad=0.05)
            pylab.colorbar(im3, cax=cax3)
    
    
        ax4=pylab.subplot(2,3,5)
        ModelArray=self.PM.GiveModelArray(V)
        IM=self.PM.ModelToSquareArray(ModelArray)
        print "            difference between channels: ", np.max(np.abs(IM[0,0]-IM[1,0]))

        im4=pylab.imshow(IM[iChannel,0],interpolation="nearest",vmin=vmin,vmax=vmax)
        ax4.axes.get_xaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)
        pylab.title("Best individual")
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im4, cax=cax4)

        PSF=self.PSF
        vmin,vmax=PSF.min(),PSF.max()
        ax5=pylab.subplot(2,3,6)
        im5=pylab.imshow(PSF[iChannel,0],interpolation="nearest",vmin=vmin,vmax=vmax)
        ax5.axes.get_xaxis().set_visible(False)
        ax5.axes.get_yaxis().set_visible(False)
        pylab.title("PSF")
        divider5 = make_axes_locatable(ax5)
        cax5 = divider5.append_axes("right", size="5%", pad=0.05)
        pylab.colorbar(im5, cax=cax5)
    
    
        pylab.suptitle('Population generation %i [%f]'%(iGen,best_ind.fitness.values[0]),size=16)
        #pylab.tight_layout()
        pylab.draw()
        pylab.show(False)
        pylab.pause(0.1)
        #fig.savefig("png/fig%4.4i.png"%iGen)

    
