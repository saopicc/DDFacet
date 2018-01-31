import numpy as np
from DDFacet.Other import ClassTimeIt
import random

class ClassMutate():
    def __init__(self,PM):
        self.PM=PM

    def MovePix(self,indiv,iPix,Flux,FluxWeighted=True,InReg=None):
        if Flux==0: return indiv
        
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

        # else:
        #     if InReg is None:
        #         reg=random.random()
        #     else:
        #         reg=InReg
        #     ind=int(reg*8)

        
        
        i1=i0+Dx[InReg]
        j1=j0+Dy[InReg]

        f1=Flux # alpha*A[0,0,i0,j0]
        
        _,_,nx,ny=A.shape
        condx=((i1>=0)&(i1<nx))
        condy=((j1>=0)&(j1<ny))
        iS=self.PM.giveIndexParm("S")
        iAlpha=self.PM.giveIndexParm("Alpha")
        if condx&condy:
            f0=A[iS,0,i1,j1]
            #fa=A[0,0,i0,j1]
            
            A[iS,0,i1,j1]+=f1
            A[iS,0,i0,j0]-=f1

            if iAlpha is not None:
                f2=A[iS,0,i1,j1]
                a0=A[iAlpha,0,i0,j0]
                a1=A[iAlpha,0,i1,j1]

                a2=(f0/f2)*(2**a0)+(f1/f2)*(2**a1)
                if 1e-3<a2<1e3:
                    a2=np.log(a2)/np.log(2.)
                    # print a0,a1,a2
                    A[iAlpha,0,i1,j1]=a2

                
            AParm=self.PM.SquareArrayToModel(A,TypeInOut=("Parms","Parms"))
            ArrayModel_S.flat[:]=AParm.flat[:]

        return indiv
    
    def setData(self,DicoData):
        self.DicoData=DicoData




    def mutGaussian(self,individual, pFlux, p0, pMove, pScale, pOffset,FactorAccelerate=1.):
        #return individual,
        T= ClassTimeIt.ClassTimeIt()
        T.disable()
        size = len(individual)
        #mu = repeat(mu, size)
        #sigma = repeat(sigma, size)

        T.timeit("start0")
        #A0=IndToArray(individual).copy()
        # Ps=np.array([pFlux, p0, pMove])
        # _p0=p0/np.sum(Ps)
        # _pMove=pMove/np.sum(Ps)
        # _pFlux=pFlux/np.sum(Ps)
   
        
 
        T.timeit("start1")
        
        Af=self.PM.ArrayToSubArray(individual,"S")
        index=np.arange(Af.size)
        ind=np.where(Af!=0.)[0]
        #AbsAf=np.abs(Af)
        #Max=np.max(AbsAf)
        #ind=np.where(AbsAf>1e-2*Max)[0]
        NNonZero=(ind.size)
        if NNonZero==0: return individual,

        T.timeit("start2")
    
        PMat=np.array([0.,pFlux, p0, pMove, pScale, pOffset])
        PMat/=np.sum(PMat)
        PMat=np.cumsum(PMat)
        

        RType=random.random()
        P0=PMat[0:-1]
        P1=PMat[1::]
        Type=np.where((RType>P0)&(RType<P1))[0]

        T.timeit("start3")
        
        # randomly change values
        if Type==0:
            NMax=int(np.max([3.,10]))
            N=int(random.uniform(1, NMax))
            indR=sorted(list(set(np.int32(np.random.rand(N)*NNonZero).tolist())))
            indSel=ind[indR]
        # zero a pixel
        elif Type==1:
            N=np.max([(NNonZero/10),1])
            indR=sorted(list(set(np.int32(np.random.rand(N)*NNonZero).tolist())))
            indSel=ind[indR]
        # move a pixel
        elif Type==2:
            NMax=int(np.max([3.,self.PM.NPixListParms/10]))
            NMax=np.min([NMax,10])
            N=int(random.uniform(1, NMax))
            indR=sorted(list(set(np.int32(np.random.rand(N)*NNonZero).tolist())))
            indSel=ind[indR]
        elif Type==3:
            FactorScale=1.+np.random.randn(1)[0]*0.01
            indSel=ind#np.arange(Af.size)
        elif Type==4:
            SMin=np.min(np.abs(Af[Af!=0.]))
            Offset=np.random.randn(1)[0]*SMin
            indSel=ind#np.arange(Af.size)
        #print pFlux, p0, pMove
        #print "PPPPPP",PMat,RType,Type
        
        #Type=0

        #print "Type:",Type,RType
        #print individual.shape[1]
        for iPix in indSel:
            #print "    --- Type %i [%i]"%(Type,iPix)

            # randomly change value of parameter
            if Type==0:
                iTypeParm=int(np.random.rand(1)[0]*len(self.PM.SolveParam))
                
                for TypeParm in self.PM.SolveParam:
                #for TypeParm in [self.PM.SolveParam[iTypeParm]]:
                    A=self.PM.ArrayToSubArray(individual,TypeParm)
                    #if TypeParm=="GSig": continue
                    if TypeParm=="S":
                        ds=0.1*np.abs(self.DicoData["DirtyArrayParmsMean"].ravel()[iPix]-Af[iPix])
                    else:
                        if "Sigma" in self.PM.DicoIParm[TypeParm]["Default"].keys():
                            ds=self.PM.DicoIParm[TypeParm]["Default"]["Sigma"]["Value"]
                        else:
                            ds=A[iPix]
                    #print "Mutating %f"%A[iPix],TypeParm
                    A[iPix] += random.gauss(0, 1.)*ds*FactorAccelerate
                    #print "      ---> %f"%A[iPix]

            # zero a pixel
            if Type==1:
                for TypeParm in self.PM.SolveParam:
                    A=self.PM.ArrayToSubArray(individual,TypeParm)
                    A[iPix] = 0.#1e-3
                    #A[iPix] *= np.random.rand(1)[0]#*2
    
            # move a pixel
            if Type==2:
                Flux=random.random()*Af[iPix]*FactorAccelerate
                #Flux=0.5*Af[iPix]#*FactorAccelerate
                Flux=np.min([Af[iPix],Flux])
                InReg=int(random.random()*8)
                #i0=individual.copy()
                individual=self.MovePix(individual,iPix,Flux,InReg=InReg)
                #i1=individual.copy()
                #stop
                # if random.random()<0.5:
                #     Flux=random.random()*Af[iPix]
                #     InReg=random.random()*8
                #     individual=self.MovePix(individual,iPix,Flux,InReg=InReg)
                # else:
                #     Flux=random.random()*0.3*Af[iPix]
                #     for iReg in [1,3,5,7]:
                #         individual=self.MovePix(individual,iPix,Flux,InReg=iReg)
                    
            if Type==3:
                Af[iPix]*=FactorScale
            if Type==4:
                Af[iPix]+=Offset

        if "GSig" in self.PM.SolveParam:
            GSig=self.PM.ArrayToSubArray(individual,"GSig")
            GSig[GSig<0]=0
            #GSig.fill(0)
            # GSig[49]=1
            # #print GSig
            # # if individual[i]==0:
            # #     if random.random() < indpb/100.:
            # #         individual[i] += random.gauss(m, s)
    
        T.timeit("for Type=%i"%Type)
    
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

    def mutNormal(self,individual,ds):
        Af=self.PM.ArrayToSubArray(individual,"S")
        dS=np.random.randn(*Af.shape)*ds
        Af += dS
        Af[Af<0]=0
        return individual,


    
    def testMovePix(self):
        A=np.random.randn(self.PM.NParam,self.PM.NPixListParms)
        A.fill(0.)
        A[:,10]=1.
        print A.shape

        ArrayModel=self.PM.GiveModelArray(A)
        A0=self.PM.ModelToSquareArray(ArrayModel,TypeInOut=("Parms","Parms"),DomainOut="Parms").copy()

        import pylab
        for reg in np.linspace(0,0.99,8):

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
