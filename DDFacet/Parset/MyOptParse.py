#!/usr/bin/env python
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

import collections
import optparse as OptParse
import sys
from collections import OrderedDict


import ReadCFG
from DDFacet.Other import ClassPrint
from DDFacet.Other import ModColor


#global Parset
#Parset=ReadCFG.Parset("/media/tasse/data/DDFacet/Parset/DefaultParset.cfg")
#D=Parset.DicoPars 

class MyOptParse():
    def __init__(self,usage='Usage: %prog <options>',version='%prog version 1.0',
                 description="""Questions and suggestions: cyril.tasse@obspm.fr""",
                 defaults=None, attributes=None):
        self.opt = OptParse.OptionParser(usage=usage,version=version,description=description)
        self.DefaultDict = defaults or OrderedDict()
        self.AttrDict = attributes or {}
        self.CurrentGroup = None
        self.DicoGroupDesc = collections.OrderedDict()
        # if True, then the section currently being added to by add_option() is "local", in the sense
        # that its options are only exposed as --section-name, not --name.
        self._local_scope = False

    def OptionGroup(self,Name,key=None):
        if self.CurrentGroup is not None:
            self.Finalise()
        self.CurrentGroup = OptParse.OptionGroup(self.opt, Name)
        self.CurrentGroupKey = key
        self.DicoGroupDesc[key] = Name
        # every group has global scope by default, until overridden by _Local=1
        self._local_scope = False

    def add_option(self,name,default):
        if name == "_Help":
            return
        # if name == "_Local":
        #     self._local_scope = default
        #     return
        # get optional attributes of this item
        attrs = self.AttrDict.get(self.CurrentGroupKey,{}).get(name,{})
        # items that are aliases of other items do not get their own option
        if attrs.get('alias_of'):
            return
        # if default is None:
        #     default = self.DefaultDict[self.CurrentGroupKey][name]
        opttype = attrs.get('type', str)
        metavar = attrs.get('options') or attrs.get('metavar', None)
        action = None
        if opttype is bool:
            opttype = str
            metavar = "0|1"
        # handle doc string
        if 'doc' in attrs:
            help = attrs['doc']
            if '%default' not in help and not action:
                help += " (default: %default)"
        else:
            help = "(default: %default)"

        option_names = [ '--%s-%s' % (self.CurrentGroupKey, name) ]
        # ## if option has an alias, enable --section-name and --alias. Else enable --section-name and --name.
        # if not attrs.get('local') or not self._local_scope:
        #     if 'alias' in attrs:
        #         option_names.append( "--%s" % attrs['alias'])
        #     elif not attrs.get('local'):
        #         option_names.append("--%s" % name)

        self.CurrentGroup.add_option(*option_names,
            help=help, type=opttype, default=default, metavar=metavar, action=action,
            dest=self.GiveKeyDest(self.CurrentGroupKey,name))

    def GiveKeyDest(self,GroupKey,Name):
        return "_".join([GroupKey,Name])

    def GiveKeysOut(self,KeyDest):
        return KeyDest.split("_",1)

    def Finalise(self):
        self.opt.add_option_group(self.CurrentGroup)

    def ReadInput(self):
        self.options, self.arguments = self.opt.parse_args()
        self.GiveDicoConfig()
        self.DicoConfig = self.DefaultDict
    
    def GiveArguments(self):
        return self.arguments

    def ExitWithError(self,message):
        self.opt.error(message)

    def GiveDicoConfig(self):
        """
        Converts options into a parset-like dict.
        If attrs is supplied, this is used to look up option aliases (in which case
        duplicate values are assinged to the aliased keys)
        """
        DicoDest = vars(self.options)
        for key, value in DicoDest.iteritems():
            GroupName, Name = self.GiveKeysOut(key)
            GroupDict = self.DefaultDict.setdefault(GroupName, OrderedDict())
            attrs = self.AttrDict.get(GroupName, {}).get(Name, {})
            if type(value) is str:
                value, _ = ReadCFG.parse_config_string(value, name=Name, extended=False, type=attrs.get('type'))
            GroupDict[Name] = value
            alias = attrs.get('alias') or attrs.get('alias_of')
            if alias:
                self.DefaultDict[GroupName][alias] = value

        return self.DefaultDict

    def ToParset (self, ParsetName):
        Dico = self.GiveDicoConfig()
        f = open(ParsetName,"w")
        for MainKey in Dico.keys():
            f.write('[%s]\n'%MainKey)
            D=Dico[MainKey]
            for SubKey in D.keys():
                attrs = self.AttrDict.get(MainKey, {}).get(SubKey, {})
                if SubKey[0] != "_" and not attrs.get('cmdline_only') and not attrs.get('alias_of'):
                    f.write('%s = %s \n'%(SubKey,str(D[SubKey])))
            f.write('\n')
        f.close()
                


    def Print(self, RejectGroups=[], dest=sys.stdout):
        P= ClassPrint.ClassPrint(HW=50)
        print>>dest, ModColor.Str(" Selected Options:")
    
        for Group,V in self.DefaultDict.items():
            if Group in RejectGroups:
                continue
    
            try:
                GroupTitle=self.DicoGroupDesc[Group]
            except:
                GroupTitle=Group
            print>>dest, ModColor.Str("[%s] %s"%(Group,GroupTitle), col="green")
    
            option_list=self.DefaultDict[Group]
            for oname in option_list:
                if oname[0] != "_":   # skip "internal" values such as "_Help"
                    V = self.DefaultDict[Group][oname]
                    attrs = self.AttrDict.get(Group).get(oname, {})
                    if not attrs.get('alias_of') and not attrs.get("cmdline_only"): # and V!="":
                        if V=="": V="''"
                        P.Print(oname,V,dest=dest)
            print



def test():
    OP=MyOptParse()
    
    OP.OptionGroup("* Data","Data")
    OP.add_option('MSName',help='Input MS')
    OP.add_option('ColName')
    OP.Finalise()
    OP.ReadInput()
    Dico=OP.GiveDicoConfig()
    OP.Print()
    
    return OP.DefaultDict
    

if __name__=="__main__":
    test()

