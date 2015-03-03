import ClassDDEGridMachine

for iFacet in DicoImager.keys():

    DicoImager[iFacet]["GridMachine"]=ClassDDEGridMachine.ClassDDEGridMachine(GD,MDC,
                                                                              RaDec=DicoImager[iFacet]["RaDec"],
                                                                              lmShift=DicoImager[iFacet]["lmShift"],
                                                                              **DicoImager[iFacet]["DicoConfigGM"])
    GridMachine=DicoImager[iFacet]["GridMachine"]
    if GridMachine.DoDDE:
        GridMachine.setSols(self.SolsTimes,self.SolsXi)
        GridMachine.CalcAterm()
        Xp=GridMachine.MME.Xp
        for Term in Xp.KeyOrderKeep:
            T=Xp.giveModelTerm(Term)
            if hasattr(T,"DelForPickle"): T.DelForPickle()
