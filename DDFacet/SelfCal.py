import os
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
def main():


    MSListName="MSList6.txt"
    LMS=["BOOTES24_SB100-109.2ch8s.ms","BOOTES24_SB110-119.2ch8s.ms","BOOTES24_SB120-129.2ch8s.ms","BOOTES24_SB130-139.2ch8s.ms","BOOTES24_SB140-149.2ch8s.ms","BOOTES24_SB150-159.2ch8s.ms"]
    BaseNameStart="KAFCA.6Freqs.MF.Beam.CompDeg"

    NSelfCal=4
    BaseName_i=BaseNameStart#"%s.%2.2i"%(BaseName,iSelfCal)

    DDF="DDF.py --MSName=%s --MaxMinorIter=50000 --Gain=0.1 --Npix=16000 --Robust=0 --Weighting=Briggs --Cell=2 --NFacets=11 --ImageName=%s --wmax=50000 --Nw=100 --ColName=CORRECTED_DATA --NCPU=32 --TChunk=10 --OverS=11 --Sup=7 --Scales=[0] --CycleFactor=2. --ScaleAmpGrid=0 --Mode=Clean --DeleteDDFProducts=1 --CompDeGridMode=1 --DDModeGrid=P --BeamMode=LOFAR"%(MSListName,BaseName_i)
        

    os.system(DDF)
 

    for iSelfCal in range(NSelfCal):

        for MSName in LMS:
            KMS="killMS.py --MSName=%s --dt=1 --InCol=CORRECTED_DATA --BaseImageName=%s --SolverType=KAFCA --InitLM=0 --InitLMdt=20 --DoPlot=0 --evPStep=40 --UVMinMax=0.2,300 --NCPU=32 --Resolution=0 --Weighting=Natural --Decorrelation=0 --OverS=17 --BeamMode=LOFAR"%(MSName,BaseName_i)

            os.system(KMS)

        BaseName_i+=".S"

        DDF="DDF.py --MSName=%s --MaxMinorIter=50000 --Gain=0.1 --Npix=16000 --Robust=0 --Weighting=Briggs --Cell=2 --NFacets=11 --ImageName=%s --wmax=50000 --Nw=100 --ColName=CORRECTED_DATA --NCPU=32 --TChunk=10 --OverS=11 --Sup=7 --Scales=[0] --CycleFactor=2. --ScaleAmpGrid=0 --Mode=Clean --DeleteDDFProducts=1 --CompDeGridMode=1 --DDModeGrid=AP --DDSols=KAFCA --BeamMode=LOFAR"%(MSListName,BaseName_i)
        

        os.system(DDF)
        


