import numpy as np

def test():
    R=RegToNp()
    R.Read()
    R.Cluster()

from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms


class PolygonNpToReg():
    def __init__(self,ListPol,RegFile):
        self.ListPol=ListPol
        self.RegFile=RegFile
        
    def makeRegPolyREG(self):
        f=open(self.RegFile,"w")
        ListPol=self.ListPol

        
        f.write("""# Region fiale format: DS9 version 4.1\n""")
        f.write("""global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n""")
        f.write("fk5\n")
        for PolyGon in ListPol:
            Lra,Ldec=PolyGon.T
            for iLine in range(Lra.size-1):
                ra0,dec0=Lra[iLine],Ldec[iLine]
                ra1,dec1=Lra[iLine+1],Ldec[iLine+1]
                sRA0=rad2hmsdms(ra0,Type="ra").replace(" ",":")
                sRA1=rad2hmsdms(ra1,Type="ra").replace(" ",":")
                sDEC0=rad2hmsdms(dec0,Type="dec").replace(" ",":")
                sDEC1=rad2hmsdms(dec1,Type="dec").replace(" ",":")
                f.write("line(%s,%s,%s,%s) # line=0 0\n"%(sRA0,sDEC0,sRA1,sDEC1))
        f.close()



    
class RegToNp():
    def __init__(self,RegName="/data/tasse/BOOTES/FirstPealed.reg"):
        self.REGFile=RegName
        
    def Read(self):
        f=open(self.REGFile,"r")

        Cat=np.zeros((1000,),dtype=[("ra",np.float32),("dec",np.float32),("I",np.float32),("Radius",np.float32),("Exclude",np.bool8),
                                    ("Cluster",np.int16),("ID",np.int16)])
        Cat=Cat.view(np.recarray)
        Cat.Cluster=-1
        Cat.Cluster=np.arange(Cat.shape[0])
        Cat.ID=np.arange(Cat.shape[0])
        Cat.I=1
        Ls=f.readlines()
        f.close()
        iCat=0
        for L in Ls:
            if "circle" in L:
                Exclude=False
                if "#" in L: 
                    Exclude=True
                    L,_=L.split("#")

                L=L.replace("\n","")
                _,L=L.split("(")
                L,_=L.split(")")

                sra,sdec,srad=L.split(",")

                srah,sram,sras=sra.split(":")
                ra=15.*(float(srah)+float(sram)/60+float(sras)/3600.)
                ra*=np.pi/180

                sdech,sdecm,sdecs=sdec.split(":")
                sgndec=1.
                if ("-" in sdec):
                    sgndec=-1.
                dech=np.abs(float(sdech))
                dec=sgndec*(dech+float(sdecm)/60+float(sdecs)/3600.)
                dec*=np.pi/180
                
                

                rad=(float(srad[0:-1])/3600.)*np.pi/180

                Cat.ra[iCat]=ra
                Cat.dec[iCat]=dec
                Cat.Radius[iCat]=rad
                Cat.Exclude[iCat]=Exclude
                iCat+=1

        Cat=(Cat[Cat.ra!=0]).copy()
        self.CatSel=Cat[Cat.Exclude==0]
        self.CatExclude=Cat[Cat.Exclude==1]

    def Cluster(self,RadMaxMinutes=1.):

        Cat=self.CatSel
        N=Cat.shape[0]
        
        #print self.CatSel

        r0=(RadMaxMinutes/60.)*np.pi/180
        iCluster=0
        for i in range(N):
            #print "Row %i: Cluster %i"%(i, iCluster)

            if (Cat.Cluster[i]==-1):
                #print "    Put row %i"%(i)
                Cat.Cluster[i] = iCluster
                iCluster+=1
                ThisICluster=Cat.Cluster[i]
            
                
            for j in range(N):
                d=np.sqrt((Cat.ra[i]-Cat.ra[j])**2+(Cat.dec[i]-Cat.dec[j])**2)
                ri=Cat.Radius[i]
                rj=Cat.Radius[j]
                if (d<(ri+rj+r0))&(Cat.Cluster[j]==-1):
                    #print "    Put row %i"%(j)
                    Cat.Cluster[j]=Cat.Cluster[i]
            
            #print "incrementing iCluster: %i"%iCluster


        #print "cats:"
        #print self.CatSel
        #print self.CatExclude
        #stop
        # import pylab
        # pylab.clf()
        # pylab.scatter(Cat.ra,Cat.dec,c=Cat.Cluster)
        # pylab.draw()
        # pylab.show()
        # print Cat.Cluster
