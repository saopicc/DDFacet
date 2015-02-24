
for iFacet in DicoImager.keys():

    fields=["Dirty","Weights","PredictVis","ModelFacet"]

    for f in fields:
        if f in DicoImager[iFacet].keys():
            del(DicoImager[iFacet][f])

