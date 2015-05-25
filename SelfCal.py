import os

def main():


    MSName="/data/tasse/BOOTES/BOOTES24_SB100-109.2ch8s.ms"
    BaseNameStart="MultiFreq2.ImRepr.KAFCA.Briggs"

    NSelfCal=4
    BaseName_i=BaseNameStart#"%s.%2.2i"%(BaseName,iSelfCal)

    for iSelfCal in range(NSelfCal):


        KMS="killMS.py --MSName=%s --dt=1 --InCol=CORRECTED_DATA --BaseImageName=%s --SolverType=KAFCA --InitLM=0 --InitLMdt=20 --DoPlot=0 --evPStep=40 --UVMinMax=0.2,300 --NCPU=32 --Resolution=0 --Weighting=Natural --Decorrelation=0 --OverS=11"%(MSName,BaseName_i)

        os.system(KMS)

        BaseName_i+=".S"

        DDF="DDF.py --MSName=%s --MaxMinorIter=20000 --Gain=0.1 --Npix=16000 --Robust=0 --Weighting=Briggs --Cell=2 --NFacets=11 --ImageName=%s --wmax=50000 --Nw=100 --ColName=CORRECTED_DATA --NCPU=32 --TChunk=10 --OverS=11 --Sup=7 --Scales=[0] --CycleFactor=2. --ScaleAmpGrid=0 --Mode=Clean --DeleteDDFProducts=1 --CompDeGridMode=1 --DDModeGrid=P --DDSols=KAFCA"%(MSName,BaseName_i)
        

        os.system(DDF)
        


