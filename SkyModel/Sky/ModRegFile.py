from __future__ import division, absolute_import, print_function
from __future__ import division, absolute_import, print_function
import numpy as np
import regions

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
        regs = regions.read_ds9(self.REGFile)

        Cat=np.zeros((len(regs),),dtype=[("ra",np.float32),("dec",np.float32),
                                    ("I",np.float32),("Radius",np.float32),
                                    ("Type","<S200"),("Exclude",np.bool8),
                                    ("dx",np.float32),("dy",np.float32),
                                    ("ra1",np.float32),("dec1",np.float32),
                                    ("Cluster",np.int16),("ID",np.int16)])
        Cat=Cat.view(np.recarray)
        Cat.Cluster=-1
        Cat.Cluster=np.arange(Cat.shape[0])
        Cat.ID=np.arange(Cat.shape[0])
        Cat.I=1
        
        for iCat, reg in enumerate(regs):
            # Excluse region if color is any kind of red 
            # This is a bit hacky, because the color can be a word, or it can be
            # a '#RRGGBB' hex code 
            color = reg.visual['color']
            if color.startswith("#") and len(color) >= 7:
                exclude = color[1:3] != "00"   # exclude if RR part of tuple is !=0
            else:
                exclude = "red" in color
                
            if type(reg) is regions.CircleSkyRegion:
                
                Cat.ra[iCat]  = reg.center.ra.rad
                Cat.dec[iCat] = reg.center.dec.rad
                Cat.Radius[iCat] = reg.radius.to("rad").value
                Cat.Type[iCat]   = "Circle"
            
            # treat ellipse and box the same (Cyril ignored ellipses)
            elif type(reg) is regions.RectangleSkyRegion or type(reg) is regions.EllipseSkyRegion:
                
                Cat.ra[iCat]    = reg.center.ra.rad
                Cat.dec[iCat]   = reg.center.dec.rad
                Cat.dx[iCat]    = reg.width.to("rad").value
                Cat.dy[iCat]    = reg.height.to("rad").value
                Cat.Type[iCat]  = "Box"
                
            elif type(reg) is regions.LineSkyRegion:
                
                exclude = False  # never exclude lines? Cyril had it that way
                Cat.ra[iCat]    = reg.start.ra.rad
                Cat.dec[iCat]   = reg.start.dec.rad
                Cat.ra1[iCat]   = reg.end.ra.rad
                Cat.dec1[iCat]  = reg.end.dec.rad
                Cat.Type[iCat]  = "Line"

            Cat.Exclude[iCat] = exclude
                
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
