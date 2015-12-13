
import numpy as np

class ClassGainMachine():
    def __init__(self,
                 GainMax=0.9,
                 GainMin=0.1,
                 SigmaScale=5.,
                 Sigma0=1.,
                 Mode="Dynamic"):
        self.SigmaScale=SigmaScale
        self.Sigma0=Sigma0
        self.Mode=Mode
        self.GainMax=GainMax
        self.GainMin=GainMin
        self.CurrentGain=GainMin

    def SetRMS(self,rms):
        self.rms=rms
        
    def SetFluxMax(self,ThisFlux):
        if self.Mode=="Dynamic":
            x=np.abs(ThisFlux)
            ExpGain=self.GainMax*np.exp(-(x/(self.rms)-self.Sigma0)/self.SigmaScale)
            self.CurrentGain=np.min([self.GainMax,ExpGain])
            self.CurrentGain=np.max([self.CurrentGain,self.GainMin])
            #print "========="
            #print "Flux = %f\nRMS  = %f\nsig0 = %f\nsigs = %f\n"%(ThisFlux,self.rms,self.Sigma0,self.SigmaScale)
        else:
            self.CurrentGain=self.GainMin

    def GiveGain(self):
        return self.CurrentGain

    def Update(self,ThisFluxMax):
        self.ListFlux.append(ThisFluxMax)
        self.CurrentIter+=1
        if not((self.CurrentIter%self.IterUpdate)==0): return
            
        ArrayFlux = np.array(self.ListFlux)
        Slopes = np.abs(ArrayFlux[1::]-ArrayFlux[0:-1])
        


        if len(self.ListFlux)>LookBackStep:
            MeanSlope=np.median(Slopes[-LookBackStep::]) # cleaned Jy/Iteration
        else:
            MeanSlope=np.median(Slopes) # cleaned Jy/Iteration
            
        if MeanSlope<ThresholdAccelerate:
            self.CurrentGain*=1.1

        if self.CurrentGain>self.GainMax:
            self.CurrentGain=self.GainMax
