import NpShared


for iFacet in DicoImager.keys():

    GridMachine=DicoImager[iFacet]["GridMachine"]
    
    if UseShared:
        uvwThis=NpShared.GiveArray("%s.uvw"%PrefixShared)
        visThis=NpShared.GiveArray("%s.predict_data"%PrefixShared)
        flagsThis=NpShared.GiveArray("%s.flags"%PrefixShared)
        times=NpShared.GiveArray("%s.times"%PrefixShared)
        A0=NpShared.GiveArray("%s.A0"%PrefixShared)
        A1=NpShared.GiveArray("%s.A1"%PrefixShared)
        A0A1=A0,A1
    else:
        uvwThis=uvw.copy()
        visThis=vis.copy()
        flagsThis=flags.copy()

    ModelIm=DicoImager[iFacet]["ModelFacet"]


    #visThis[Row0:Row1]+=1

    vis=GridMachine.get(times,uvwThis,visThis,flagsThis,A0A1,ModelIm,(Row0,Row1))

    if not(UseShared):
        DicoImager[iFacet]["PredictVis"]=vis

#    visThis[:,:,:]+=vis[:,:,:]


if not(UseShared):
    del(uvw,vis,times,flags,A0A1)
