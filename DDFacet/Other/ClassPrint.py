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

import os
import sys
import ModColor

class ClassPrint():
    def __init__(self,HW=20,quote='"'):
        self.LeftW=HW
        self.WV0=50
        self.WV1=30
        self.WidthHelp=30
        self.quote=quote
        self.proto=" - %s %s %s : "
        self.Lproto=len(self.proto.replace("%s",""))


    def getWidth(self):
        rows, columns = os.popen('stty size', 'r').read().split()
        return int(columns)

    def Print(self,par,value,value2=None,dest=sys.stdin):
        parout=" - %s %s"%(par,"."*(self.LeftW-len(par)))
        if value2 is None:
            valueOut=value
        else:
            valueOut="%s%s"%(value.ljust(self.WV0),(""" "%s" """%value2).rjust(self.WV1))
        print>>dest,"%s = %s"%(parout,valueOut)
        
    def Print2(self,par,value,helpit,col="white"):
        WidthTerm=self.getWidth()
        Lpar=len(str(par))
        Lval=len(str(value))
        SFill="."*max(self.LeftW-self.Lproto-Lpar-Lval,2)
        WidthHelp=WidthTerm-(self.Lproto+Lpar+Lval+len(SFill))
        Spar="%s"%ModColor.Str(str(par),col=col,Bold=False)
        Sval="%s"%ModColor.Str(str(value),col=col,Bold=False)
        if helpit=="":
            helpit="Help yourself"
        Shelp="%s"%helpit
        if WidthHelp<0:
             print self.proto%(Spar,SFill,Sval)+Shelp
             return
        Lhelp=len(str(helpit))
        listStrHelp=range(0,Lhelp,WidthHelp)
        
        if listStrHelp[-1]!=Lhelp: listStrHelp.append(Lhelp)
        
        
        
        print self.proto%(Spar,SFill,Sval)+Shelp[0:WidthHelp]
        for i in range(1,len(listStrHelp)-1):
            parout="%s: %s"%(" "*(self.LeftW-2),Shelp[listStrHelp[i]:listStrHelp[i+1]])
            print parout
