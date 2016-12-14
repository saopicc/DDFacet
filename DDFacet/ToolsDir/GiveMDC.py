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

from ClassData import ClassMultiPointingData,ClassSinglePointingData,ClassGlobalData
from ClassME import MeasurementEquation
import numpy as np
import ClassSM
import ClassMS
import os

def GiveMDC(ParsetFile="ParsetNew.txt",
            freqs=None,GD=None,DoReadData=True):
    

    MS=[]
    SM=[]
    if GD is None:
        GD=ClassGlobalData(ParsetFile)

        
    
    ListMSCat=GD.DicoConfig["Files"]["FileMSCat"]["Name"]
    NPointing=len(ListMSCat)

    ListMS=[]
    for i in range(NPointing):
        ThisCat=ListMSCat[i]
        if ".npy" in ThisCat:
            ListMS.append(np.load(ThisCat)["dirMSname"][0])
        else:
            ListMS.append(ThisCat)
            
    
    if "FileSourceCat" in GD.DicoConfig["Files"].keys():
        ListSM=GD.DicoConfig["Files"]["FileSourceCat"]
        ThereIsSM=True
    else:
        ListSM=[None for i in range(NPointing)]
        ThereIsSM=False


    for MSname,SMname in zip(ListMS,ListSM):
        MS0=ClassMS.ClassMS(MSname,Col=GD.DicoConfig["Files"]["ColName"],DoReadData=DoReadData)
        if ThereIsSM:
            SM0=ClassSM.ClassSM(SMname)
            SM0.AppendRefSource((MS0.rac,MS0.decc))
            SM.append(SM0)
        MS0.DelData()
        MS.append(MS0)

    MDC=ClassMultiPointingData(GD)


    for ID in range(NPointing):
        MDC.setMS(MS[ID],PointingID=ID)
        if ThereIsSM:
            MDC.setSM(SM[ID],PointingID=ID)
        if freqs is None: freqs=MS[ID].ChanFreq.flatten()
        MDC.setFreqs(freqs,PointingID=ID)
        MDC.setMappingBL(PointingID=ID)

    #MME=MeasurementEquation()
    #HYPERCAL_DIR=os.environ["HYPERCAL_DIR"]
    #execfile("%s/HyperCal/Scripts/ScriptSetMultiRIME.py"%HYPERCAL_DIR)
    return MDC,GD#,MME
