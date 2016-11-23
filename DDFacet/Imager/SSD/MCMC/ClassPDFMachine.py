import numpy as np
from DDFacet.ToolsDir.GeneDist import ClassDistMachine
import scipy.interpolate

class ClassPDFMachine():
    def __init__(self,ConvMachine):
        self.ConvMachine=ConvMachine
        self.DM=ClassDistMachine()

    def setPDF(self,indivIn,std,Chi2_0=None,NReal=100):
        indiv=indivIn.copy()
        indiv.fill(0)
        LTries=[]
        for iReal in range(NReal):
            #print iReal
            LTries.append(self.ConvMachine.ConvolveFFT(indiv,OutMode="Data",AddNoise=1.).ravel())
        ATries=np.array(LTries)
        
        ThisStd=np.mean(np.std(ATries,axis=0))
        ATries*=std/ThisStd

        AChi2=np.sum(ATries**2,axis=1)
        
        if Chi2_0 is not None:
            AChi2*=Chi2_0/np.mean(AChi2)
        self.MeanChi2=np.mean(AChi2)


        xd,yd=self.DM.giveDist(AChi2,Ns=20)
        dx=xd[1]-xd[0]
        yd/=dx
        pdf=scipy.interpolate.interp1d(xd, yd,"cubic")
        x=np.linspace(xd.min(),xd.max(),1000)
        
        y=pdf(x)
        self.MaxPDF=np.max(y)
        #x,y=xd, yd

        self.InterpPDF=pdf
        self.InterpX0=xd.min()
        self.InterpX1=xd.max()

        # import pylab
        # pylab.clf()
        # pylab.plot(x,y,ls="",marker=".",color="blue")
        # pylab.plot(xd, yd,ls="",marker=".",color="green")
        # pylab.scatter([self.MeanChi2], [0],marker=".",color="black")
        # pylab.draw()
        # pylab.show()

        

    def pdfScalar(self,Chi2):
        if not(self.InterpX0<Chi2<self.InterpX1):
            return 1e-20
        p=self.InterpPDF(Chi2)
        if p<=self.MaxPDF/4.:
            return 1e-20
        return p

    def logpdfScalar(self,Chi2):
        return np.log(self.pdfScalar(Chi2))

    def pdf(self,Chi2):
        if type(Chi2)==list or type(Chi2)==np.ndarray:
            ans=[]
            for chi2 in Chi2:
                ans.append(self.pdfScalar(chi2))
            if type(Chi2)==np.ndarray:
                ans=np.array(ans)
            return ans
        else:
            return self.pdfScalar(Chi2)

    def logpdf(self,Chi2):
        if type(Chi2)==list or type(Chi2)==np.ndarray:
            ans=[]
            for chi2 in Chi2:
                ans.append(self.logpdfScalar(chi2))
            if type(Chi2)==np.ndarray:
                ans=np.array(ans)
            return ans
        else:
            return self.logpdfScalar(Chi2)
            
