#!/usr/bin/env python
"""
killMS, a package for calibration in radio interferometry.
Copyright (C) 2013-2017  Cyril Tasse, l'Observatoire de Paris,
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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import numpy as np
from killMS.Array import ModLinAlg

from DDFacet.Other import logger
log = logger.getLogger("ClassJonesDomains")


class ClassJonesDomains():
    def __init__(self):
        pass

    def GiveFreqDomains(self,ChanFreqs,ChanWidth,NChanJones=0):
        if NChanJones==0:
            NChanJones=ChanFreqs.size
        ChanEdges=np.linspace(ChanFreqs.min()-ChanWidth/2.,ChanFreqs.max()+ChanWidth/2.,NChanJones+1)
        FreqDomains=[[ChanEdges[iF],ChanEdges[iF+1]] for iF in range(NChanJones)]
        return np.array(FreqDomains)

    def AddFreqDomains(self,DicoJones,ChanFreqs,ChanWidth):
        FreqDomains=self.GiveFreqDomains(ChanFreqs,ChanWidth)
        DicoJones["FreqDomains"]=FreqDomains


    def AverageInFreq(self,DicoJones,FreqDomainsOut):
        FreqsIn=np.mean(DicoJones["FreqDomains"],axis=1)
        NChanIn=FreqsIn.size
        FreqsOut=np.mean(FreqDomainsOut,axis=1)
        NChanOut=FreqsOut.size

        MeanFreqJonesChan=(FreqDomainsOut[:,0]+FreqDomainsOut[:,1])/2.
        DFreq=np.abs(FreqsIn.reshape((NChanIn,1))-FreqsOut.reshape((1,NChanOut)))
        JonesChanMapping=np.argmin(DFreq,axis=1)
        nt,nd,na,_,_,_=DicoJones["Jones"].shape
        MeanJones=np.zeros((nt,nd,na,NChanOut,2,2),dtype=DicoJones["Jones"].dtype)
        for ich in range(NChanOut):
            indCh=np.where(JonesChanMapping==ich)[0]
            MeanJones[:,:,:,ich,:,:]=np.mean(DicoJones["Jones"][:,:,:,indCh,:,:],axis=3)

        DicoJones["Jones"]=MeanJones
        DicoJones["FreqDomains"]=FreqDomainsOut


    def AddVisToJonesMapping(self,Jones,VisTimes,VisFreqs):
        print("Building VisToJones time mapping...", file=log)
        DicoJonesMatrices=Jones
        G=DicoJonesMatrices["Jones"]
        ind=np.zeros((VisTimes.size,),np.int32)
        nt,nd,na,_,_,_=G.shape
        ii=0
        for it in range(nt):
            t0=DicoJonesMatrices["t0"][it]
            t1=DicoJonesMatrices["t1"][it]
            indMStime=np.where((VisTimes>=t0)&(VisTimes<t1))[0]
            indMStime=np.ones((indMStime.size,),np.int32)*it
            # print "================="
            # print t0,t1,t1-t0
            # print it,indMStime.size,np.max(ind)
            ind[ii:ii+indMStime.size]=indMStime[:]
            ii+=indMStime.size
        Jones["Map_VisToJones_Time"]=ind

        print("Building VisToJones freq mapping...", file=log)
        FreqDomains=Jones["FreqDomains"]
        NChanJones=FreqDomains.shape[0]
        MeanFreqJonesChan=(FreqDomains[:,0]+FreqDomains[:,1])/2.
        NVisChan=VisFreqs.size
        DFreq=np.abs(VisFreqs.reshape((NVisChan,1))-MeanFreqJonesChan.reshape((1,NChanJones)))
        VisToJonesChanMapping=np.argmin(DFreq,axis=1)
        Jones["Map_VisToJones_Freq"]=VisToJonesChanMapping
        print("  VisToJonesChanMapping %s"%str(VisToJonesChanMapping), file=log)
    


    def GetMergedFreqDomains(self,DicoJ0,DicoJ1):
        print("Compute frequency domains of merged Jones arrays", file=log)
        FreqDomain0=DicoJ0["FreqDomains"]
        FreqDomain1=DicoJ1["FreqDomains"]
        f0=FreqDomain0.flatten().tolist()
        f1=FreqDomain1.flatten().tolist()
        ff=np.array(sorted(list(set(f0+f1))))
        
        dff=np.abs(ff[1::]-ff[0:-1])
        LF=[]
        MaskSkip=np.zeros((ff.size,),bool)
        for iFreq in range(ff.size):
            if MaskSkip[iFreq]: continue
            df=np.abs(ff-ff[iFreq])
            ind=np.where(df<1.)[0]
            MaskSkip[ind]=1
            LF.append(ff[iFreq])


        fmin=np.max([FreqDomain0.min(),FreqDomain1.min()])
        fmax=np.min([FreqDomain0.max(),FreqDomain1.max()])
            
        ff=np.array(LF)
        nf=ff.size
        FreqDomainOut=np.zeros((nf-1,2),np.float64)
        FreqDomainOut[:,0]=ff[0:-1]
        FreqDomainOut[:,1]=ff[1::]


        fm=np.mean(FreqDomainOut,axis=1)
        ind=np.where((fm>=fmin)&(fm<fmax))[0]
        FreqDomainOut=FreqDomainOut[ind]

        print("  There are %i channels in the merged Jones array"%FreqDomainOut.shape[0], file=log)
        return FreqDomainOut
        


    def MergeJones(self,DicoJ0,DicoJ1):

        print("Merging Jones arrays", file=log)

        FreqDomainOut=self.GetMergedFreqDomains(DicoJ0,DicoJ1)

        DicoOut={}
        DicoOut["t0"]=[]
        DicoOut["t1"]=[]
        DicoOut["tm"]=[]
        DicoOut["FreqDomains"]=FreqDomainOut
        T0=np.min([DicoJ0["t0"][0],DicoJ1["t0"][0]])
        it=0
        CurrentT0=T0
        
        while True:
            T0=CurrentT0
            
            dT0=DicoJ0["t1"]-T0
            dT0=dT0[dT0>0]
            dT1=DicoJ1["t1"]-T0
            dT1=dT1[dT1>0]
            if(dT0.size==0)&(dT1.size==0):
                break
            elif dT0.size==0:
                dT=dT1[0]
            elif dT1.size==0:
                dT=dT0[0]
            else:
                dT=np.min([dT0[0],dT1[0]])
                
            DicoOut["t0"].append(CurrentT0)
            T1=T0+dT
            DicoOut["t1"].append(T1)
            Tm=(T0+T1)/2.
            DicoOut["tm"].append(Tm)
            CurrentT0=T1
            it+=1
    
            
        DicoOut["t0"]=np.array(DicoOut["t0"])
        DicoOut["t1"]=np.array(DicoOut["t1"])
        DicoOut["tm"]=np.array(DicoOut["tm"])
        
        
        nt0=DicoJ0["t0"].size
        nt1=DicoJ1["t0"].size
        
        fm0=np.mean(DicoJ0["FreqDomains"],axis=1)
        fm1=np.mean(DicoJ1["FreqDomains"],axis=1)
        fmOut=np.mean(DicoOut["FreqDomains"],axis=1)

        _,nd,na,_,_,_=DicoJ0["Jones"].shape
        nt=DicoOut["tm"].size

        nchOut=fmOut.size

        DicoOut["Jones"]=np.zeros((nt,nd,na,nchOut,2,2),np.complex64)
        DicoOut["Jones"][:,:,:,:,0,0]=1.
        DicoOut["Jones"][:,:,:,:,1,1]=1.

        iG0_t=np.argmin(np.abs(DicoOut["tm"].reshape((nt,1))-DicoJ0["tm"].reshape((1,nt0))),axis=1)
        iG1_t=np.argmin(np.abs(DicoOut["tm"].reshape((nt,1))-DicoJ1["tm"].reshape((1,nt1))),axis=1)
        
#        print>>log,fmOut,DicoJ0["FreqDomain"],DicoJ1["FreqDomain"]
        for ich in range(nchOut):

#            print>>log,fmOut[ich]
            indChG0=np.where((fmOut[ich]>=DicoJ0["FreqDomains"][:,0]) & (fmOut[ich]<DicoJ0["FreqDomains"][:,1]))[0]
            if indChG0.size==0: continue
            indChG0=indChG0[0]
            
            indChG1=np.where((fmOut[ich]>=DicoJ1["FreqDomains"][:,0]) & (fmOut[ich]<DicoJ1["FreqDomains"][:,1]))[0]
            if indChG1.size==0: continue
            indChG1=indChG1[0]
            


            for itime in range(nt):
                G0=DicoJ0["Jones"][iG0_t[itime],:,:,indChG0,:,:]
                G1=DicoJ1["Jones"][iG1_t[itime],:,:,indChG1,:,:]
                DicoOut["Jones"][itime,:,:,ich,:,:]=ModLinAlg.BatchDot(G0,G1)
            
        
        return DicoOut
    
    

