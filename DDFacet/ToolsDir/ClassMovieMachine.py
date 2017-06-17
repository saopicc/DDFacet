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

import ClassFacetMachine
import numpy as np
import ToolsDir
import os



class MovieMachine():
    def __init__(self,ParsetFile="ParsetDDFacet.txt",PointingID=0,pngBaseDir="png",TimeCode=(0,-1,1)):
        
        MDC,GD=ToolsDir.GiveMDC.GiveMDC(ParsetFile)
        self.MDC=MDC
        self.GD=GD
        self.PointingID=PointingID
        Imager=ClassFacetMachine.ClassFacetMachine(MDC,GD,Precision="S",Parallel=True)#,Sols=SimulSols)
        MainFacetOptions=GD.DicoConfig["Facet"]["MainFacetOptions"]
        Imager.appendMainField(**MainFacetOptions)
        self.Imager=Imager
        self.Stack=False
        self.pngBaseDir=pngBaseDir
        os.system("rm -rf %s"%self.pngBaseDir)
        os.system("mkdir -p %s"%self.pngBaseDir)
        import pylab
        self.fig=pylab.figure(1)
        self.CurrentPNGNum=0
        self.HaveData=False
        self.TStart,self.TEnd,self.Tincr=TimeCode

        


        MS=self.MDC.giveMS(self.PointingID)

        self.NTimes=MS.F_times.shape[0]


    def ReadData(self):
        
        self.DC=self.MDC.giveSinglePointingData(self.PointingID)
        MS=self.DC.MS
        MS.ReadData()
        self.MS=MS
        self.HaveData=True
    
    def ToPngImage(self,ImageIn=None,NamePNGOut=None):
        import pylab
        pylab.clf()
        pylab.imshow(self.Image[0,0],interpolation="nearest",cmap="gray")
        pylab.draw()
        pylab.show(False)
        
        FileOut="%s/Snap%5.5i.png"%(self.pngBaseDir,self.CurrentPNGNum)
        print "... saving %s"%FileOut
        self.fig.savefig(FileOut)
        MS=self.DC.MS
        NpFileOut="%s/Snap%5.5i"%(self.pngBaseDir,self.CurrentPNGNum)

        Cell=self.GD.DicoConfig["Facet"]["MainFacetOptions"]["Cell"]
        Cell*=(np.pi/180)/3600
        np.savez(NpFileOut,Image=self.Image[0,0],Time=self.ThisCurrentTime,Freq=MS.Freq_Mean,CellSize=Cell)
        self.CurrentPNGNum+=1
            
    def MakeSnap(self,it0,it1):
        if not(self.HaveData):
            self.ReadData()

        DicoData=self.MS.GiveDataChunk(it0,it1)
        
        times,uvw,vis,flag,A0A1=DicoData['times'],DicoData['uvw'],DicoData['data'],DicoData['flags'],DicoData['A0A1']

        self.Imager.putChunk(times,uvw,vis,flag,A0A1,doStack=self.Stack)
        
        self.Image=self.Imager.FacetsToIm()
        self.ThisCurrentTime=np.mean(times)
        #self.Imager.ToCasaImage()

    def MainLoop(self):
        for i in range(0,self.NTimes-1,self.Tincr):
            print "Step %i/%i"%(i,self.NTimes)
            self.MakeSnap(i,i+self.Tincr)
            self.ToPngImage()
        

