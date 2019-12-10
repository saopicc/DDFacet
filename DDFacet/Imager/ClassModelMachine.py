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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import numpy as np
from DDFacet.Other import logger
log=logger.getLogger("ClassModelMachine")
from DDFacet.Other import MyPickle
from DDFacet.Other import ModColor


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
            print(ModColor.Str("Reference frequency already set to %f MHz"%(self.RefFreq/1e6)), file=log)
            return
        self.RefFreq=RefFreq
        self.DicoSMStacked["RefFreq"]=RefFreq
        #self.DicoSMStacked["AllFreqs"]=np.array(AllFreqs)

    def ToFile(self,FileName,DicoIn=None):
        print("Saving dico model to %s"%FileName, file=log)
        if DicoIn is None:
            D=self.DicoSMStacked
        else:
            D=DicoIn

        D["ListScales"]=self.ListScales
        D["ModelShape"]=self.ModelShape
        MyPickle.Save(D,FileName)

    def FromFile(self,FileName):
        print("Reading dico model from %s"%FileName, file=log)
        self.DicoSMStacked=MyPickle.Load(FileName)
        self.FromDico(self.DicoSMStacked)


    def FromDico(self,DicoSMStacked):
        self.DicoSMStacked=DicoSMStacked
        self.RefFreq=self.DicoSMStacked["RefFreq"]
        self.ListScales=self.DicoSMStacked["ListScales"]
        self.ModelShape=self.DicoSMStacked["ModelShape"]