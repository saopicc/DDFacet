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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import optparse
import sys
import pickle
import DDFacet.DDF
from pyrap.tables import table
import numpy as np
from astropy.time import Time

def read_options():
    desc="""DDFacet """
    
    opt = optparse.OptionParser(usage='Usage: %prog --Parset=somename.MS <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--Parset',help='Input Parset [no default]',default='')
    opt.add_option_group(group)

    group = optparse.OptionGroup(opt, "* Data selection options")
    group.add_option('--RunDir',help='Run directory [no default]',default='MOVIE')
    group.add_option('--EnablePlot',help='Enable matplotlib',default=0)
    #group.add_option('--TimeCode',help='TimeCode',default=[0,-1,1])
    group.add_option('--Incr',type=int,help='Increment in time-steps',default=10)
    opt.add_option_group(group)
    
    options, arguments = opt.parse_args()
    f = open("last_param.obj","wb")
    pickle.dump(options,f)
    return options

from DDFacet.Parset import ReadCFG
#from DDFacet.Parset import MyOptParse
import os

def main(options=None):
    

    if options is None:
        f = open("last_param.obj",'rb')
        options = pickle.load(f)
    
    if options.EnablePlot==0:
        import matplotlib 
        matplotlib.use('agg')

    TCode=(0,-1,int(options.Incr))

    RunDir=options.RunDir
    print(RunDir)

    Parset = ReadCFG.Parset(options.Parset)
    GD=Parset.DicoPars
    TempParset=os.path.abspath(options.Parset)


    
    # default_values = Parset.value_dict
    # attrs = Parset.attr_dict

    # desc = """Questions and suggestions: cyril.tasse@obspm.fr"""

    # OP = MyOptParse.MyOptParse(usage='Usage: %prog [parset file] <options>', 
    #                            description=desc, defaults=default_values, attributes=attrs)

    # # create options based on contents of parset
    # for section in Parset.sections:
    #     values = default_values[section]
    #     # "_Help" value in each section is its documentation string
    #     OP.OptionGroup(values.get("_Help", section), section)
    #     for name, value in getattr(default_values[section], "iteritems", default_values[section].items)():
    #         if not attrs[section][name].get("no_cmdline"):
    #             OP.add_option(name, value)

    # OP.Finalise()
    # OP.ReadInput()
    # TempParset="_TemplateMovie.parset"options.Parset
    # GD=OP.DicoConfig
    
    MSName=GD["Data"]["MS"]
    
    MSName=os.path.abspath(MSName)
    #GD["Output"]["Images"] = d
    
    #OP.ToParset(TempParset)
    
    t=table(MSName)
    TIME=times=t.getcol("TIME")
    tt=np.sort(np.unique(times))
    dt=(tt[1]-tt[0])
    
    dt_frame=dt*options.Incr
    tStart=tt[0]
    tEnd=tt[-1]
    NFrames=int((tEnd-tStart+dt)/dt_frame)
    tt=np.linspace(tStart-dt/2,tEnd+dt/2,NFrames+1)

    # for iTime in range(tt.size-1):
    #     t0,t1=tt[iTime],tt[iTime+1]
    #     mjd = t0 / (3600.0 * 24.0)
    #     s0=Time(mjd, format='mjd').isot
    #     mjd = t1 / (3600.0 * 24.0)
    #     s1=Time(mjd, format='mjd').isot
    #     print(s0,s1)
    #     return

    FIELD_ID=t.getcol("FIELD_ID")
    DATA_DESC_ID=t.getcol("DATA_DESC_ID")
    os.system("mkdir -p %s"%RunDir)
    os.chdir(RunDir)
    for iTime in range(tt.size-1):
        t0,t1=tt[iTime],tt[iTime+1]
        ind=np.where((FIELD_ID==0) & (DATA_DESC_ID==0) & (TIME>t0) & (TIME<t1))[0]
        if ind.size==0: continue
        print("=========================================================")
        print("================== %i / %i"%(iTime,tt.size))
        print("=========================================================")
        ss="""DDF.py %s --Data-MS %s --Output-Name %s.snap%5.5i --Deconv-Mode Hogbom --Cache-Reset 0 --Debug-Pdb 0 --Mask-External None --Output-Images d --Freq-NBand 1 --Output-Mode Dirty --Selection-TaQL "TIME>%.2f && TIME<%.2f" """%(TempParset,MSName,GD["Output"]["Name"],iTime,t0,t1)
        print(ss)
        os.system(ss)
        
    # import ClassMovieMachine
    # MM=ClassMovieMachine.MovieMachine(ParsetFile=options.Parset,PointingID=options.PointingID,pngBaseDir=options.pngBaseDir,TimeCode=TCode)
    # MM.MainLoop()


if __name__=="__main__":
    options=read_options()
    main(options)
    

