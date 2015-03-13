import MyPickle
import ClassPrint
import ModColor

def test():
    P=ClassPrint.ClassPrint()
    Obj,ValObj=MyPickle.Load("test")
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
    Obj,ValObj=MyPickle.Load("test")
    PrintOptParse(Obj,ValObj,RejectGroup=["CohJones"])

def PrintOptParse(Obj,ValObj,RejectGroup=[]):
    P=ClassPrint.ClassPrint(HW=30)
    LGroups=Obj.option_groups
    print ModColor.Str(" Selected Options:")

    for Group in LGroups:
        Skip=False
        for Name in RejectGroup:
            if Name in Group.title:
                Skip=True

        if Skip: continue
        print ModColor.Str(Group.title,col="green")

        option_list=Group.option_list
        for o in option_list:
            lopt=o._long_opts[0]
            oname=lopt.split("--")[-1]


            V=getattr(ValObj,oname)
            if (V!="")&(V!=None):

                #P.Print(oname,V)
                default=o.default
                H=o.help
                if H!=None:
                    H=o.help.replace("%default",str(default))
                    P.Print2(oname,V,H)
                else:
                    P.Print(oname,V)

                # strName=%s
                # print "       "oname,V
        print
