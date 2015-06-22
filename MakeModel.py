#!/usr/bin/env python

import optparse
import pickle

SaveName="last_MakeModel.obj"

def read_options():
    desc="""Questions and suggestions: cyril.tasse@obspm.fr"""
    global options
    opt = optparse.OptionParser(usage='Usage: %prog --ms=somename.MS <options>',version='%prog version 1.0',description=desc)

    group = optparse.OptionGroup(opt, "* Data-related options", "Won't work if not specified.")
    group.add_option('--SkyModel',help='List of targets [no default]',default='')
    group.add_option('--BaseImageName',help='List of targets [no default]',default='')
    group.add_option('--NCluster',help=' Default is %default',default="0")
    group.add_option('--DoPlot',help=' Default is %default',default="1")
    group.add_option('--DoSelect',help=' Default is %default',default="0")
    group.add_option('--DoPrint',help=' Default is %default',default="0")
    group.add_option('--CMethod',help=' Clustering method [1,2,3,4]. Default is %default',default="4")
    group.add_option('--PreClusterFile',help=' PreClusterFile. Default is %default',default="")
    
    opt.add_option_group(group)


    options, arguments = opt.parse_args()
    f = open(SaveName,"wb")
    pickle.dump(options,f)
    
def main(options=None):
    if options==None:
        f = open(SaveName,'rb')
        options = pickle.load(f)

    SkyModel=options.SkyModel

    if options.BaseImageName!="":
        from DDFacet.Imager.ClassModelMachine import ClassModelMachine
        MM=ClassModelMachine(Gain=0.1)
        DicoModel="%s.DicoModel"%options.BaseImageName
        FitsFile="%s.model.fits"%options.BaseImageName
        MM.FromFile(DicoModel)
        MM.CleanNegComponants(box=15,sig=1)
        MM.ToNPYModel(FitsFile)
        SkyModel="tmpSourceCat.npy"

    NCluster=int(options.NCluster)
    DoPlot=(int(options.DoPlot)==1)
    DoSelect=(int(options.DoSelect)==1)
    CMethod=int(options.CMethod)

    if DoPlot==0:
        import matplotlib
        matplotlib.use('agg')

    from Sky import ClassSM

    SM=ClassSM.ClassSM(SkyModel,ReName=True,
                       DoREG=True,SaveNp=True,
                       SelSource=DoSelect,ClusterMethod=CMethod)

    PreCluster=options.PreClusterFile
    SM.Cluster(NCluster=NCluster,DoPlot=DoPlot,PreCluster=PreCluster)
    SM.MakeREG()
    SM.Save()

    if options.DoPrint=="1":
        SM.print_sm2()




if __name__=="__main__":
    read_options()
    f = open(SaveName,'rb')
    options = pickle.load(f)

    main(options=options)
