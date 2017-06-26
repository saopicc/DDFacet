import numpy as np
import ModTigger
import ModSMFromNp

from SkyModel.Other import rad2hmsdms
from SkyModel.Other import ModColor
from SkyModel.Array import RecArrayOps
from ClassClusterClean import ClassClusterClean
from ClassClusterTessel import ClassClusterTessel
from ClassClusterRadial import ClassClusterRadial
from ClassClusterKMean import ClassClusterKMean
#import ModClusterRadial
from pyrap.images import image
import scipy.linalg
from ModBBS2np import ReadBBSModel
import ModRegFile
import time
from DDFacet.ToolsDir import ModCoord

class ClassSM():
    def __init__(self,infile,infile_cluster="",killdirs=[],invert=False,DoPrintCat=False,\
                     ReName=False,DoREG=False,SaveNp=False,NCluster=0,DoPlot=True,Tigger=False,\
                     FromExt=None,ClusterMethod=1,SelSource=False):
        self.ClusterMethod=ClusterMethod
        self.infile_cluster=infile_cluster
        self.TargetList=infile
        self.Type="Catalog"
        if ".npy" in infile:
            Cat=np.load(infile)
            Cat=Cat.view(np.recarray)
        elif Tigger:
            Cat=ModTigger.ReadTiggerModel(infile)
        elif FromExt!=None:
            
            Cat=ModSMFromNp.ReadFromNp(FromExt)
        else:
            Cat=ReadBBSModel(infile,infile_cluster=infile_cluster)
        self.SourceCat=Cat
        self.killdirs=killdirs
        self.invert=invert
        self.infile=infile
        self.REGFile=None
        self.Dirs=sorted(list(set(self.SourceCat.Cluster.tolist())))
        self.NDir=np.max(self.SourceCat.Cluster)+1
        self.NSources=Cat.shape[0]

        try:
            SourceCat=self.SourceCat
            indIMax=np.argmax(SourceCat.I)
            self.rarad=SourceCat.ra[indIMax]#np.sum(SourceCat.I*SourceCat.ra)/np.sum(SourceCat.I)
            self.decrad=np.sum(SourceCat.I*SourceCat.dec)/np.sum(SourceCat.I)
            self.CoordMachine = ModCoord.ClassCoordConv(self.rarad, self.decrad)
        except:
            pass


        self.BuildClusterCat()
        self.NDir=np.max(self.SourceCat.Cluster)+1
        self.NSources=Cat.shape[0]
        self.SetSelection()
        self.PrintBasics()


        #self.print_sm2()

    def SetSelection(self):
        Cat=self.SourceCat
        killdirs=self.killdirs
        invert=self.invert
        self.SourceCatKeepForSelector=self.SourceCat.copy()

        # if DoPrintCat:
        #     self.print_sm(Cat)

        self.SourceCat.kill=1

        if killdirs!=[]:
            self.SourceCat.kill=0
            for i in range(len(self.SourceCat)):
                for StrPiece in killdirs:
                    if StrPiece=="": continue
                    if StrPiece in self.SourceCat.Name[i]: self.SourceCat.kill[i]=1
        if invert:
            ind0=np.where(self.SourceCat.kill==0)[0]
            ind1=np.where(self.SourceCat.kill==1)[0]
            self.SourceCat.kill[ind0]=1
            self.SourceCat.kill[ind1]=0

    def Save(self):
        infile=self.infile
        print ModColor.Str(" SkyModel PROPERTIES: ")
        print "   - SkyModel File Name: %s"%ModColor.Str(infile,col="green")
        if self.REGFile!=None: print "   - ds9 region file: %s"%ModColor.Str(self.REGFile,col="green")
        npext=""
        if not(".npy" in infile): npext=".npy"
        self.NpFile="%s%s"%(infile,npext)
        np.save(infile,self.SourceCat)

        FileClusterCat="%s.ClusterCat.npy"%(self.infile)
        print "   - ClusterCat File Name: %s"%ModColor.Str(FileClusterCat,col="green")
        np.save(FileClusterCat,self.ClusterCat)

        self.PrintBasics()

    def PrintBasics(self):
        infile=self.infile
        npext=""
        if not(".npy" in infile): npext=".npy"
        print "   - Numpy catalog file: %s"%ModColor.Str("%s%s"%(infile,npext),col="green")

        #print "Oufile: %s"%self.infile_cluster
        #if infile_cluster!="":
        #print "   - Cluster File Name: %s"%self.infile_cluster
        print "   - Number of Sources  = ",self.SourceCat.shape[0]
        print "   - Number of Directions  = ",self.NDir
        print

    def Cluster(self,NCluster=1,DoPlot=True,PreCluster="",FromClusterCat=""):


        if PreCluster!="":
            R=ModRegFile.RegToNp(PreCluster)
            R.Read()
            #R.Cluster()
            PreClusterCat=R.CatSel
            ExcludeCat=R.CatExclude
        else:
            PreClusterCat=None
            ExcludeCat=None

        if ExcludeCat!=None:
            for j in range(ExcludeCat.shape[0]):
                d=np.sqrt((self.SourceCat.ra-ExcludeCat.ra[j])**2+(self.SourceCat.dec-ExcludeCat.dec[j])**2)
                self.SourceCat.Exclude[d<ExcludeCat.Radius[j]]=True

        if NCluster==0:
            self.SourceCat.Cluster=np.arange(self.SourceCat.shape[0])
        elif NCluster==1:
            self.SourceCat.Cluster=0
        else:
            self.cluster(NCluster,DoPlot,PreClusterCat=PreClusterCat,FromClusterCat=FromClusterCat)#,ExcludeCat=ExcludeCat)
            

        self.SourceCat=self.SourceCat[self.SourceCat.Exclude==False]

        ClusterList=sorted(list(set(self.SourceCat.Cluster.tolist())))
        self.NDir=len(ClusterList)
        for iCluster,iNewCluster in zip(ClusterList,range(self.NDir)):
            ind=np.where(self.SourceCat.Cluster==iCluster)[0]
            self.SourceCat.Cluster[ind]=iNewCluster
            self.REGName=False

        self.Rename()

        self.REGFile=None
        self.MakeREG()

        self.Dirs=sorted(list(set(self.SourceCat.Cluster.tolist())))
        self.WeightDirKeep=np.zeros((self.NDir,),float)
        for diri in self.Dirs:
            ind=np.where(self.SourceCat.Cluster==diri)[0]
            self.WeightDirKeep[diri]=np.sqrt(np.sum(self.SourceCat.Sref[ind]))
        self.WeightDir=self.WeightDirKeep.copy()

        self.ExistToSub=False
        self.ExistToSub=(np.count_nonzero(self.SourceCat.kill==-1)>0)
        self.BuildClusterCat()

    def Rename(self):
        for diri in range(self.NDir):
            ind=np.where(self.SourceCat.Cluster==diri)[0]
            #CatSel=self.SourceCat[self.SourceCat.Cluster==diri]
            Names=["c%is%i."%(diri,i) for i in range(ind.shape[0])]
            self.SourceCat.Name[ind]=Names
        self.REGName=True



    def cluster(self,nk=10,DoPlot=False,PreClusterCat=None,FromClusterCat=""):

        # pylab.clf()
    
        #s.fill(0.)
        #s[0]=1

        cos=np.cos
        sin=np.sin


        self.SourceCat.Cluster=-1
        indSubSel=np.arange(self.SourceCat.shape[0])
        NPreCluster=0

        # #######################################
        # if (PreClusterCat!=None)&(FromClusterCat==""):
        #     N=PreClusterCat.shape[0]
        #     Ns=self.SourceCat.ra.shape[0]
        #     for iReg in range(N):
        #         #d=np.sqrt((self.SourceCat.ra-PreClusterCat.ra[iReg])**2+(self.SourceCat.dec-PreClusterCat.dec[iReg])**2)
        #         ra1=self.SourceCat.ra
        #         ra2=PreClusterCat.ra[iReg]
        #         d1=self.SourceCat.dec
        #         d2=PreClusterCat.dec[iReg]
        #         cosD = sin(d1)*sin(d2) + cos(d1)*cos(d2)*cos(ra1-ra2)
        #         d=np.arccos(cosD)
        #         self.SourceCat.Cluster[d<PreClusterCat.Radius[iReg]]=PreClusterCat.Cluster[iReg]
        #         self.SourceCat.Exclude[d<PreClusterCat.Radius[iReg]]=False
        #     print self.SourceCat.Cluster
        #     indPreCluster=np.where(self.SourceCat.Cluster!=-1)[0]
        #     NPreCluster=np.max(PreClusterCat.Cluster)+1
        #     SourceCatPreCluster=self.SourceCat[indPreCluster]
        #     indSubSel=np.where(self.SourceCat.Cluster==-1)[0]
        #     print "number of preselected clusters: %i"%NPreCluster
        
        # if nk==-1:
        #     print "Removing non-clustered sources"%NPreCluster
        #     self.SourceCat=self.SourceCat[self.SourceCat.Cluster!=-1]
        #     print self.SourceCat.Cluster
        #     return
        # #######################################

        SourceCat=self.SourceCat[indSubSel]

        indIMax=np.argmax(SourceCat.I)

        self.rarad=SourceCat.ra[indIMax]#np.sum(SourceCat.I*SourceCat.ra)/np.sum(SourceCat.I)
        self.decrad=np.sum(SourceCat.I*SourceCat.dec)/np.sum(SourceCat.I)

        self.CoordMachine = ModCoord.ClassCoordConv(self.rarad, self.decrad)


        x,y,s=SourceCat.ra,SourceCat.dec,SourceCat.I
        x,y=self.radec2lm_scalar(x,y)
        
        SourceCat.Cluster=0

        if self.ClusterMethod==1:
            CM=ClassClusterClean(x,y,s,nk,DoPlot=DoPlot)
        elif self.ClusterMethod==2:
            CM=ClassClusterTessel(x,y,s,nk,DoPlot=DoPlot)
        elif self.ClusterMethod==3:
            CM=ClassClusterRadial(x,y,s,nk,DoPlot=DoPlot)
        elif self.ClusterMethod==4:
            if PreClusterCat!=None:
                l0,m0=self.radec2lm_scalar(PreClusterCat.ra,PreClusterCat.dec)
                CM=ClassClusterKMean(x,y,s,nk,DoPlot=DoPlot,PreCluster=(l0,m0))
            else:
                CM=ClassClusterClean(x,y,s,nk,DoPlot=0)#DoPlot)
                DictNode=CM.Cluster()
                ra,dec=[],[]
                ListL,ListM=[],[]
                for idDir in DictNode.keys():
                    LDir=DictNode[idDir]["ListCluster"]
                    #ra0,dec0=np.mean(self.SourceCat.ra[LDir]),np.mean(self.SourceCat.dec[LDir])
                    # print idDir,ra0,dec0,self.SourceCat.ra[LDir].min(),self.SourceCat.ra[LDir].max()
                    # if not np.isnan(ra0):
                    #     ra.append(ra0)
                    #     dec.append(dec0)
                    This_l,This_m=self.radec2lm_scalar(np.array(self.SourceCat.ra[LDir]),np.array(self.SourceCat.dec[LDir]))
                    ListL.append(np.mean(This_l))
                    ListM.append(np.mean(This_m))

                # l0,m0=self.radec2lm_scalar(np.array(ra),np.array(dec))
                l0,m0=np.array(ListL),np.array(ListM)
                nk=l0.size

                CM=ClassClusterKMean(x,y,s,nk,DoPlot=DoPlot,InitLM=(l0,m0))


        REGFile="%s.tessel.reg"%self.TargetList

        #print FromClusterCat
        if FromClusterCat=="":
            DictNode=CM.Cluster()
        else:

            DictNode={}
            if not("reg" in FromClusterCat):
                SourceCatRef=np.load(FromClusterCat)
                SourceCatRef=SourceCatRef.view(np.recarray)
            else:
                R=ModRegFile.RegToNp(FromClusterCat)
                R.Read()
                SourceCatRef=R.CatSel

            ClusterList=sorted(list(set(SourceCatRef.Cluster.tolist())))
            xc,yc=self.radec2lm_scalar(SourceCatRef.ra,SourceCatRef.dec)
            lc=np.zeros((len(ClusterList),),dtype=np.float32)
            mc=np.zeros((len(ClusterList),),dtype=np.float32)
            for iCluster in ClusterList:
                indC=np.where(SourceCatRef.Cluster==iCluster)[0]
                lc[iCluster]=np.sum(SourceCatRef.I[indC]*xc[indC])/np.sum(SourceCatRef.I[indC])
                mc[iCluster]=np.sum(SourceCatRef.I[indC]*yc[indC])/np.sum(SourceCatRef.I[indC])
            Ns=x.size
            Nc=lc.size
            D=np.sqrt((x.reshape((Ns,1))-lc.reshape((1,Nc)))**2+(y.reshape((Ns,1))-mc.reshape((1,Nc)))**2)
            Cid=np.argmin(D,axis=1)

            #pylab.clf()
            for iCluster in ClusterList:
                ind=np.where(Cid==iCluster)[0]
                DictNode["%3.3i"%iCluster]={}
                DictNode["%3.3i"%iCluster]["ListCluster"]=ind.tolist()
                # pylab.scatter(x[ind],y[ind],c=np.ones((ind.size,))*iCluster,vmin=0,vmax=Nc,lw=0)
                # pylab.draw()
                # pylab.show(False)
            


        try:
            CM.ToReg(REGFile,self.rarad,self.decrad)
        except:
            pass

        iK=NPreCluster
        self.NDir=len(DictNode.keys())
        
        # print self.SourceCat.Cluster.min(),self.SourceCat.Cluster.max()
        
        for key in DictNode.keys():
            ind=np.array(DictNode[key]["ListCluster"])
            if ind.size==0: 
                print "Direction %i is empty"%int(key)
                continue
            self.SourceCat["Cluster"][indSubSel[ind]]=iK
            iK+=1



        # if PreClusterCat!=None:
        #     SourceCat=np.concatenate((SourceCatPreCluster,SourceCat))
        #     SourceCat=SourceCat.view(np.recarray)

        #print self.SourceCat.Cluster.min(),self.SourceCat.Cluster.max()
        #self.SourceCat=SourceCat



    def AppendRefSource(self,(rac,decc)):
        S0=1e-10
        CatCopy=self.SourceCat[0:1].copy()
        CatCopy['Name']="Reference"
        CatCopy['ra']=rac
        CatCopy['dec']=decc
        CatCopy['Sref']=S0
        CatCopy['I']=S0
        CatCopy['Q']=S0
        CatCopy['U']=S0
        CatCopy['V']=S0
        CatCopy['Cluster']=0
        CatCopy['Type']=0
        self.SourceCat.Cluster+=1
        self.SourceCat=np.concatenate([CatCopy,self.SourceCat])
        self.SourceCat=self.SourceCat.view(np.recarray)
        self.NDir+=1
        self.NSources+=1

        CatCopy=self.ClusterCat[0:1].copy()
        CatCopy['Name']="Reference"
        CatCopy['ra']=rac
        CatCopy['dec']=decc
        CatCopy['SumI']=0
        CatCopy['Cluster']=0
        self.ClusterCat.Cluster+=1
        self.ClusterCat=np.concatenate([CatCopy,self.ClusterCat])
        self.ClusterCat=self.ClusterCat.view(np.recarray)

    def AppendFromSMClass(self,SM):
        #self.SourceCat=np.concatenate([self.SourceCat, SM.SourceCat])
        #self.SourceCat=self.SourceCat.view(np.recarray)
        ClusterCatCopy=SM.ClusterCat.copy()
        ClusterCatCopy.Cluster=ClusterCatCopy.Cluster+np.max(self.ClusterCat.Cluster)+1
        self.ClusterCat=np.concatenate([self.ClusterCat, ClusterCatCopy])
        self.ClusterCat=self.ClusterCat.view(np.recarray)
        
        self.NDir=self.ClusterCat.shape[0]
        self.NSources=self.SourceCat.shape[0]

    def SaveNP(self):
        infile=self.NpFile
        np.save(self.NpFile,self.SourceCat)
        print "   - Numpy catalog file: %s"%ModColor.Str(self.NpFile,col="green")

    def SelectSourceMouse(self):
        from ClassSelectMouse2 import ClassSelectMouse
        M=ClassSelectMouse()
        ra=self.SourceCat.ra*180/np.pi
        dec=self.SourceCat.dec*180/np.pi
        ra-=np.mean(ra)
        dec-=np.mean(dec)
        M.DefineXY((ra,dec),np.log10(self.SourceCat.I))
        self.SourceCat.Select=M.Start()

    def BuildClusterCat(self):
        ClusterCat=np.zeros((len(self.Dirs),),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('SumI',np.float),("Cluster",int)])
        ClusterCat=ClusterCat.view(np.recarray)
        icat=0
        for d in self.Dirs:
            cat=self.SourceCat[self.SourceCat.Cluster==d]
            l,m=self.CoordMachine.radec2lm(cat.ra, cat.dec)
            lmean,mmean=np.sum(l*cat.I)/np.sum(cat.I),np.sum(m*cat.I)/np.sum(cat.I)

            ramean,decmean=self.CoordMachine.lm2radec(np.array([lmean]),np.array([mmean]))
            ClusterCat.ra[icat]=ramean
            ClusterCat.dec[icat]=decmean
            ClusterCat.SumI[icat]=np.sum(cat.I)
            ClusterCat.Cluster[icat]=d
            icat+=1
        #print ClusterCat.ra
        self.ClusterCat=ClusterCat



    def radec2lm_scalar(self,ra,dec,rarad0=None,decrad0=None):
        if rarad0==None:
            rarad0=self.rarad
            decrad0=self.decrad
        l = np.cos(dec) * np.sin(ra - rarad0)
        m = np.sin(dec) * np.cos(decrad0) - np.cos(dec) * np.sin(decrad0) * np.cos(ra - rarad0)
        return l,m





    def Calc_LM(self,rac,decc):
        Cat=self.SourceCat
        if not("l" in Cat.dtype.fields.keys()):
            Cat=RecArrayOps.AppendField(Cat,('l',float))
            Cat=RecArrayOps.AppendField(Cat,('m',float))
        Cat.l,Cat.m=self.radec2lm_scalar(self.SourceCat.ra,self.SourceCat.dec,rac,decc)
        self.SourceCat=Cat
        self.SourceCatKeepForSelector=self.SourceCat.copy()

        Cat=self.ClusterCat
        if not("l" in Cat.dtype.fields.keys()):
            Cat=RecArrayOps.AppendField(Cat,('l',float))
            Cat=RecArrayOps.AppendField(Cat,('m',float))
        Cat.l,Cat.m=self.radec2lm_scalar(self.ClusterCat.ra,self.ClusterCat.dec,rac,decc)
        self.ClusterCat=Cat
    # def Calc_LM(self,rac,decc):
    #     Cat=self.SourceCat
    #     if not("l" in Cat.dtype.fields.keys()):
    #         Cat=RecArrayOps.AppendField(Cat,('l',float))
    #         Cat=RecArrayOps.AppendField(Cat,('m',float))
    #     Cat.l,Cat.m=self.radec2lm_scalar(self.SourceCat.ra,self.SourceCat.dec,rac,decc)
    #     self.SourceCat=Cat
    #     self.SourceCatKeepForSelector=self.SourceCat.copy()
        

    def MakeREG(self):
        self.REGFile="%s.reg"%self.TargetList
        f=open(self.REGFile,"w")
        self.REGName=True
        f.write("# Region file format: DS9 version 4.1\n")
        f.write('global color=green dashlist=8 3 width=1 font="helvetica 7 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        for i in range(self.SourceCat.shape[0]):
            #ss="fk5;ellipse(213.202544,49.871826,0.003909,0.003445,181.376961) # text={P1C1}"
            ra=self.SourceCat.ra[i]*180./np.pi
            dec=self.SourceCat.dec[i]*180./np.pi
            Type=self.SourceCat.Type[i]
            Gmaj=self.SourceCat.Gmaj[i]*180./np.pi*(2.*np.sqrt(2.*np.log(2)))
            Gmin=self.SourceCat.Gmin[i]*180./np.pi*(2.*np.sqrt(2.*np.log(2)))
            if Gmin==0.: Gmin=1./3600
            PA=(self.SourceCat.Gangle[i]+np.pi/2.)*180./np.pi
            rad=20./2600

            #ss="fk5;ellipse(%f,%f,%f,%f,%f) # text={%s}"%(ra,dec,Gmaj,Gmin,0,str(i))
            if self.REGName:
                if Type==1:
                    ss="fk5;ellipse(%f,%f,%f,%f,%f) # text={%s} color=green width=2 "%(ra,dec,Gmaj,Gmin,PA,self.SourceCat.Name[i])
                else:
                    ss="fk5;point(%f,%f) # text={%s} point=circle 5 color=red width=2"%(ra,dec,self.SourceCat.Name[i])
            else:
                if Type==1:
                    ss="fk5;ellipse(%f,%f,%f,%f,%f) # color=green width=2 "%(ra,dec,Gmaj,Gmin,PA)
                else:
                    ss="fk5;point(%f,%f) # point=circle 5 color=red width=2"%(ra,dec)
                

            f.write(ss+"\n")
        f.close()

    def RestoreCat(self):
        self.SourceCat=self.SourceCatKeepForSelector.copy()
        self.Dirs=sorted(list(set(self.SourceCat.Cluster.tolist())))
        self.NDir=len(self.Dirs)
        self.NSources=self.SourceCat.shape[0]
        selDir=np.array(sorted(list(set(self.SourceCat.Cluster.tolist()))))
        # self.WeightDir=self.WeightDirKeep[selDir].copy()


    def SelectSubCat(self,Selector):
        self.Selector=Selector
        self.SourceCat=(self.SourceCatKeepForSelector[self.Selector]).copy()
        self.Dirs=sorted(list(set(self.SourceCat.Cluster.tolist())))
        self.NDir=len(self.Dirs)
        self.NSources=self.SourceCat.shape[0]
        selDir=np.array(sorted(list(set(self.SourceCat.Cluster.tolist()))))
        # self.WeightDir=self.WeightDirKeep[selDir].copy()


        





    def print_sm2(self):
        CatIn=self.SourceCat
        ind=np.argsort(CatIn.Cluster)
        Cat=CatIn[ind]
        TEMPLATE = ('  %(Cluster)5s %(name)10s %(RA)15s %(DEC)15s %(Flux)10s %(alpha)10s %(RefFreq)10s %(Kill)6s ')
        print
        
        print " TARGET LIST: "
        print TEMPLATE % {
                'Cluster': "K".center(5),
                'name': "Name".center(10),
                'RA': "RA".center(15),
                'DEC': "DEC".center(15),
                'Flux': "Flux".rjust(10),
                'alpha': "alpha".rjust(10),
                'RefFreq': "RefFreq".rjust(10),
                'Kill': "Kill" }

        for i in range(Cat.shape[0]):
            SName=Cat.Name[i]
            SRa=rad2hmsdms.rad2hmsdms(Cat.ra[i],Type="ra").replace(" ",":")
            SDec=rad2hmsdms.rad2hmsdms(Cat.dec[i]).replace(" ",".")
            SI="%6.3f"%Cat.I[i]
            SAlpha="%4.2f"%Cat.alpha[i]
            SRefFreq="%5.1f"%(Cat.RefFreq[i]/1.e6)
            SKill="%i"%Cat.kill[i]
            SCluster="%2.2i"%Cat.Cluster[i]
            StrOut = TEMPLATE % {
                'Cluster': SCluster.center(5),
                'name': SName.center(10),
                'RA': SRa,
                'DEC': SDec,
                'Flux': SI,
                'alpha': SAlpha,
                'RefFreq':SRefFreq,
                'Kill':SKill }
            print StrOut
                

    def print_sm(self,Cat):
        if self.infile_cluster=="":
            print " TARGET LIST: "
            format="%13s%20s%20s%10s%10s%10s"#%10s"
            print format%("Name","Ra","Dec","Flux","alpha","RefFreq")#,"Kill")
            for i in range(Cat.shape[0]):
    
    
                SName=Cat.Name[i]
                SRa=rad2hmsdms.rad2hmsdms(Cat.ra[i]/15).replace(" ",":")
                SDec=rad2hmsdms.rad2hmsdms(Cat.dec[i]).replace(" ",".")
                SI=Cat.I[i]
                SAlpha=Cat.alpha[i]
                SRefFreq=Cat.RefFreq[i]
                SKill=str(Cat.kill[i]==1)
                #print "%13s%20s%20s%10.4f%10s%10.2e%8s"%(SName,SRa,SDec,SI,SAlpha,SRefFreq,SKill)
                print "%13s%20s%20s%10.4f%10s%10.2e"%(SName,SRa,SDec,SI,SAlpha,SRefFreq)#,SKill)
        else:
            format="%10s%10s%15s%15s%10s%10s%10s"#%8s"
            print
            print " TARGET LIST: "
            print format%("Group","Name","Ra","Dec","Flux","alpha","RefFreq")#,"kill")
            for i in range(np.max(Cat.Cluster)+1):
                ind=np.where(Cat.Cluster==i)[0]
                for j in range(ind.shape[0]):
                    jj=ind[j]
                    SName=Cat.Name[jj]
                    SRa=rad2hmsdms.rad2hmsdms(Cat.ra[jj]/15).replace(" ",":")
                    SDec=rad2hmsdms.rad2hmsdms(Cat.dec[jj]).replace(" ",".")
                    SI=Cat.I[jj]
                    SAlpha=Cat.alpha[jj]
                    SRefFreq=Cat.RefFreq[jj]
                    SKill=str(Cat.kill[jj]==1)
                    SGroup=str(i)
                    #print "%10s%10s%15s%15s%8.4f%10s%10.2e%8s"%(SGroup,SName,SRa,SDec,SI,SAlpha,SRefFreq)#,SKill)
                    print "%10s%10s%15s%15s%8.4f%10s%10.2e"%(SGroup,SName,SRa,SDec,SI,SAlpha,SRefFreq)#,SKill)
                    

    
    
    
    
