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

import ClassPrint
import ModColor
import MyPickle


def test():
    P= ClassPrint.ClassPrint()
    Obj,ValObj= MyPickle.Load("test")
    #return Obj
    #ValObj,_=Obj.parse_args()
    #return ValObj
    LGroups=Obj.option_groups
    for Group in LGroups:
        print Group.title

        option_list=Group.option_list
        for o in option_list:
            lopt=o._long_opts[0]
            oname=lopt.split("--")[-1]
            V=getattr(ValObj,oname)
            if V!="":

                P.Print(oname,V)
                # strName=%s
                # print "       "oname,V
        print

def test2():
    Obj,ValObj= MyPickle.Load("test")
    PrintOptParse(Obj,ValObj,RejectGroup=["CohJones"])

def PrintOptParse(Obj,ValObj,RejectGroup=[]):
    P= ClassPrint.ClassPrint(HW=30)
    LGroups=Obj.option_groups
    print ModColor.Str(" Selected Options:")

    for Group in LGroups:
        Skip=False
        for Name in RejectGroup:
            if Name in Group.title:
                Skip=True

        if Skip: continue
        print ModColor.Str(Group.title, col="green")

        option_list=Group.option_list
        for o in option_list:
            lopt=o._long_opts[0]
            oname=lopt.split("--")[-1]


            V=getattr(ValObj,oname)
#            if (V!="")&(V is not None):
            if True:#V!="":
                if V=="": V="''"

                #P.Print(oname,V)
                default=o.default
                H=o.help
                # if H is not None:
                #     H=o.help.replace("%default",str(default))
                #     P.Print2(oname,V,H)
                # else:
                #     P.Print(oname,V)
                P.Print(oname,V)

                # strName=%s
                # print "       "oname,V
        print
