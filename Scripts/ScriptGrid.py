import NpShared


for iFacet in DicoImager.keys():

    GridMachine=DicoImager[iFacet]["GridMachine"]
    
    if UseShared:
        uvwThis=NpShared.GiveArray("%s.uvw"%PrefixShared)
        visThis=NpShared.GiveArray("%s.data"%PrefixShared)
        flagsThis=NpShared.GiveArray("%s.flags"%PrefixShared)
        times=NpShared.GiveArray("%s.times"%PrefixShared)
        A0=NpShared.GiveArray("%s.A0"%PrefixShared)
        A1=NpShared.GiveArray("%s.A1"%PrefixShared)
        A0A1=A0,A1
        W=NpShared.GiveArray("%s.Weights"%PrefixShared)
    else:
        uvwThis=uvw.copy()
        visThis=vis.copy()
        flagsThis=flags.copy()

    DicoJonesMatrices=None
    if ApplyCal:
        DicoJonesMatrices=NpShared.SharedToDico("killMSSolutionFile")
        DicoClusterDirs=NpShared.SharedToDico("DicoClusterDirs")
        DicoJonesMatrices["DicoClusterDirs"]=DicoClusterDirs
    # A0A1=self.A0A1
    # times=self.times
    # W=self.W
    Dirty=GridMachine.put(times,uvwThis,visThis,flagsThis,A0A1,W,DoNormWeights=False, DicoJonesMatrices=DicoJonesMatrices)#,doStack=False)
    DicoImager[iFacet]["Dirty"]=Dirty
    DicoImager[iFacet]["Weights"]=GridMachine.SumWeigths


if not(UseShared):
    del(uvw,vis,times,flags,A0A1,W)
