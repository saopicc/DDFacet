import numpy as np
from DDFacet.ToolsDir import ClassSpectralFunctions

class ClassPSFServer():
    def __init__(self,GD):
        self.GD=GD

    def setDicoVariablePSF(self,DicoVariablePSF):
        # NFacets=len(DicoVariablePSF.keys())
        # NPixMin=1e6
        # for iFacet in sorted(DicoVariablePSF.keys()):
        #     _,npol,n,n=DicoVariablePSF[iFacet]["PSF"][0].shape
        #     if n<NPixMin: NPixMin=n

        # nch=self.GD["MultiFreqs"]["NFreqBands"]
        # CubeVariablePSF=np.zeros((NFacets,nch,npol,NPixMin,NPixMin),np.float32)
        # CubeMeanVariablePSF=np.zeros((NFacets,1,npol,NPixMin,NPixMin),np.float32)
        # for iFacet in sorted(DicoVariablePSF.keys()):
        #     _,npol,n,n=DicoVariablePSF[iFacet]["PSF"][0].shape
        #     for ch in range(nch):
        #         i=n/2-NPixMin/2
        #         j=n/2+NPixMin/2+1
        #         CubeVariablePSF[iFacet,ch,:,:,:]=DicoVariablePSF[iFacet]["PSF"][ch][0,:,i:j,i:j]
        #     CubeMeanVariablePSF[iFacet,0,:,:,:]=DicoVariablePSF[iFacet]["MeanPSF"][0,:,i:j,i:j]

        self.DicoVariablePSF=DicoVariablePSF
        self.CubeVariablePSF=DicoVariablePSF["CubeVariablePSF"]
        self.CubeMeanVariablePSF=DicoVariablePSF["CubeMeanVariablePSF"]
        self.NFacets,nch,npol,NPixMin,_=self.CubeVariablePSF.shape
        self.ShapePSF=nch,npol,NPixMin,NPixMin
        self.NPSF=NPixMin
        
        DicoMappingDesc={"freqs":DicoVariablePSF["freqs"],
                         "WeightChansImages":DicoVariablePSF["WeightChansImages"],
                         "SumJonesChan":DicoVariablePSF["SumJonesChan"],
                         "SumJonesChanWeightSq":DicoVariablePSF["SumJonesChanWeightSq"],
                         "ChanMappingGrid":DicoVariablePSF["ChanMappingGrid"],
                         "MeanJonesBand":DicoVariablePSF["MeanJonesBand"]}

        self.DicoMappingDesc=DicoMappingDesc

        self.SpectralFunctionsMachine=ClassSpectralFunctions.ClassSpectralFunctions(DicoMappingDesc)
        self.RefFreq=self.SpectralFunctionsMachine.RefFreq
        self.AllFreqs=self.SpectralFunctionsMachine.AllFreqs
        #print "PSFServer:",self.RefFreq, self.AllFreqs
        #self.CalcJacobian()

    def setLocation(self,xp,yp):
        self.iFacet=self.giveFacetID2(xp,yp)

    def giveFacetID(self,xp,yp):
        dmin=1e6
        for iFacet in range(self.NFacets):
            d=np.sqrt((xp-self.DicoVariablePSF[iFacet]["pixCentral"][0])**2+(yp-self.DicoVariablePSF[iFacet]["pixCentral"][1])**2)
            if d<dmin:
                dmin=d
                ClosestFacet=iFacet
        return ClosestFacet
                
    def giveFacetID2(self,xp,yp):
        dmin=1e6
        for iFacet in range(self.NFacets):
            CellSizeRad=self.DicoVariablePSF[iFacet]["CellSizeRad"]
            _,_,nx,_=self.DicoVariablePSF[iFacet]["OutImShape"]
            l=CellSizeRad*(xp-nx/2)
            m=CellSizeRad*(yp-nx/2)
            lSol,mSol=self.DicoImager[iFacet]["lmSol"]

            d=np.sqrt((l-lSol)**2+(m-mSol)**2)

            if d<dmin:
                dmin=d
                ClosestFacet=iFacet

        return ClosestFacet



    def setFacet(self,iFacet):
        #print "set facetloc"
        self.iFacet=iFacet

    def CalcJacobian(self):
        self.CubeJacobianMeanVariablePSF={}
        dx=31
        # CubeMeanVariablePSF shape = NFacets,1,npol,NPixMin,NPixMin
        _,_,_,nx,_=self.CubeMeanVariablePSF.shape

        # import pylab
        for iFacet in range(self.NFacets):
            ThisCubePSF=self.CubeMeanVariablePSF[iFacet,0,0][nx/2-dx-1:nx/2+dx+1+1,nx/2-dx-1:nx/2+dx+1+1]
            Jx=(ThisCubePSF[:-2,:]-ThisCubePSF[2::,:])/2
            Jy=(ThisCubePSF[:,:-2]-ThisCubePSF[:,2::])/2
            Jx=Jx[:,1:-1]
            Jy=Jy[1:-1,:]
            J=np.zeros((Jx.size,2),np.float32)
            J[:,0]=Jx.ravel()
            J[:,1]=Jy.ravel()
            self.CubeJacobianMeanVariablePSF[iFacet]={}
            self.CubeJacobianMeanVariablePSF[iFacet]["J"]=J
            self.CubeJacobianMeanVariablePSF[iFacet]["JHJ"]=np.dot(J.T,J)
            
            

            # pylab.clf()
            # pylab.subplot(1,3,1)
            # pylab.imshow(ThisCubePSF,interpolation="nearest")
            # pylab.subplot(1,3,2)
            # pylab.imshow(Jx,interpolation="nearest")
            # pylab.subplot(1,3,3)
            # pylab.imshow(Jy,interpolation="nearest")
            # pylab.draw()
            # pylab.show(False)
            # pylab.pause(0.1)
            # stop

    def SolveOffsetLM(self,Dirty,xc0,yc0):
        iFacet=self.iFacet
        Lambda=1.
        nIter=30
        beta=np.zeros((2,1),np.float32)
        J=self.CubeJacobianMeanVariablePSF[iFacet]["J"]
        s,_=J.shape
        nx=int(np.sqrt(s))
        JHJ=self.CubeJacobianMeanVariablePSF[iFacet]["JHJ"]
        JHJ1inv=np.linalg.inv(JHJ+Lambda*np.diag(np.diag(JHJ)))

        dx=nx/2
        
        _,_,_,nx_psf,_=self.CubeMeanVariablePSF.shape
        xc_psf=nx_psf/2
        ThisCubePSF=self.CubeMeanVariablePSF[iFacet,0,0][xc_psf-dx:xc_psf+dx+1,xc_psf-dx:xc_psf+dx+1]
        xc,yc=xc0,yc0
        Val=Dirty[xc,yc]
        for Iter in range(nIter):
            D=Dirty[xc-dx:xc+dx+1,yc-dx:yc+dx+1]
            Val=Dirty[xc,yc]
            Model=(ThisCubePSF/np.max(ThisCubePSF))*Val
            R=(D-Val*Model).reshape((D.size,1))
            delta=np.dot(JHJ1inv,np.dot(J.T,R))
            delta_x=int(round(delta[0,0]))
            delta_y=int(round(delta[1,0]))
            if (delta_x==0)&(delta_y==0):
                break
            xc+=delta_x
            yc+=delta_y
            print delta_x,delta_y
        return xc,yc
            
            

    def GivePSF(self):
        return self.CubeVariablePSF[self.iFacet],self.CubeMeanVariablePSF[self.iFacet]

    def GiveFreqBandsFluxRatio(self,*args,**kwargs):
        return self.SpectralFunctionsMachine.GiveFreqBandsFluxRatio(*args,**kwargs)
