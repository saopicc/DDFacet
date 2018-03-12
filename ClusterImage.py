#!/usr/bin/env python
import numpy as np
import pylab
from DDFacet.Other import MyLogger
from DDFacet.Other import MyPickle
log=MyLogger.getLogger("ClusterImage")
import pyfits
import Sky.ClassClusterDEAP
from DDFacet.ToolsDir.rad2hmsdms import rad2hmsdms
import optparse
import pickle
SaveFile="ClusterImage.last"

def test():
    
    CIM=ClusterImage("image_dirin_SSD_m.app.restored.fits")
    CIM.setCatName("image_dirin_SSD_m.app.restored.pybdsm.srl.fits")
    #CIM.GroupSources()
    CIM.Cluster()

def angDist(a0,a1,d0,d1):
    s=np.sin
    c=np.cos
    return np.arccos(s(d0)*s(d1)+c(d0)*c(d1)*c(a0-a1))


def read_options():
    desc=""" cyril.tasse@obspm.fr"""
    
    opt = optparse.OptionParser(usage='Task to build a boolean mask file from a restored fits image, Usage: %prog <options>',version='%prog version 1.0',description=desc)
    group = optparse.OptionGroup(opt, "* Data-related options")
    group.add_option('--RestoredIm',type="str",help="Name of the restored image",default=None)
    group.add_option('--SourceCat',type="str",help="Name of the source catalog",default="")
    group.add_option('--AvoidPolygons',type="str",help="Name of the avoidace polygon file",default="")
    opt.add_option_group(group)

    
    options, arguments = opt.parse_args()

    f = open(SaveFile,"wb")
    pickle.dump(options,f)




class ClusterImage():
    def __init__(self,**kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
        print>>log,"Reading %s"%self.RestoredIm
        f=pyfits.open(self.RestoredIm)
        decc,rac=f[0].header["CRVAL1"],f[0].header["CRVAL2"]
        rac,decc=f[0].header["CRVAL1"],f[0].header["CRVAL2"]
        self.dPix=abs(f[0].header["CDELT1"])
        self.NPix=abs(f[0].header["NAXIS1"])
        rac*=np.pi/180
        decc*=np.pi/180
        sRA =rad2hmsdms(rac,Type="ra").replace(" ",":")
        sDEC=rad2hmsdms(decc,Type="dec").replace(" ",":")
        print>>log,"Image center: %s %s"%(sRA,sDEC)
        self.rarad=rac
        self.decrad=decc

        if self.SourceCat!="":
            self.setCatName(self.SourceCat)
        
    def radec2lm(self,ra,dec):
        l = np.cos(dec) * np.sin(ra - self.rarad)
        m = np.sin(dec) * np.cos(self.decrad) - np.cos(dec) * np.sin(self.decrad) * np.cos(ra - self.rarad)
        return l,m
        
    def setCatName(self,CatName):
        print>>log,"Reading source catalog %s"%CatName
        self.CatName=CatName
        self.c=pyfits.open(CatName)[1]
        self.c.data.RA*=np.pi/180.
        self.c.data.DEC*=np.pi/180.

        Cat=np.zeros((len(self.c.data.RA),),dtype=[("ra",np.float32),
                                                   ("dec",np.float32),
                                                   ("S",np.float32),
                                                   ("Maj",np.float32)])
        Cat=Cat.view(np.recarray)
        Cat.ra=self.c.data.RA
        Cat.dec=self.c.data.DEC
        Cat.S=self.c.data.Total_flux
        Cat.Maj=self.c.data.Maj
        self.Cat=Cat
        
    def GroupSources(self,RadiusArcmin=2.):
        Rad=RadiusArcmin/60.*np.pi/180.
        c=self.c
        self.DicoPos={}
        
        Ns=c.data.RA.size
        print>>log,"Merging sources..."
        for iS in range(Ns):
            self.DicoPos[iS]={}
            self.DicoPos[iS]["RA"]=c.data.RA[iS]
            self.DicoPos[iS]["DEC"]=c.data.DEC[iS]
            self.DicoPos[iS]["Maj"]=c.data.Maj[iS]
            self.DicoPos[iS]["Total_flux"]=c.data.Total_flux[iS]
            self.DicoPos[iS]["Checked"]=False

        self.DicoAssoc={}
        for iS in range(Ns):
            RA=self.DicoPos[iS]["RA"]
            DEC=self.DicoPos[iS]["DEC"]
            Maj=self.DicoPos[iS]["Maj"]
            S=self.DicoPos[iS]["Total_flux"]
            
            #print>>log,"======================"
            #print>>log,"Inspecting %i"%iS
            if len(self.DicoAssoc)==0:
                self.DicoAssoc[0]={"RA":[RA],
                                   "DEC":[DEC],
                                   "S":[S],
                                   "Maj":[Maj]}
                continue

            HasAssociatedThis=False
            for iSc in self.DicoAssoc.keys():
                ra,dec=np.array(self.DicoAssoc[iSc]["RA"]),np.array(self.DicoAssoc[iSc]["DEC"])
                dra,ddec=RA-ra,DEC-dec
                d=angDist(ra,RA,dec,DEC)#np.sqrt((dra)**2+(ddec)**2)
                if d.min()<Rad:
                    #print>>log,"  Associating %i <- %i"%(iSc,iS)
                    self.DicoAssoc[iSc]["RA"].append(RA)
                    self.DicoAssoc[iSc]["DEC"].append(DEC)
                    self.DicoAssoc[iSc]["S"].append(S)
                    self.DicoAssoc[iSc]["Maj"].append(Maj)
                    HasAssociatedThis=True
                    break
            if not HasAssociatedThis:
                ThisInd=max(self.DicoAssoc.keys())+1
                self.DicoAssoc[ThisInd]={"RA":[RA],
                                         "DEC":[DEC],
                                         "S":[S],
                                         "Maj":[Maj]}

        
        raOut=[]
        decOut=[]
        SOut=[]
        MajOut=[]
        for iS in self.DicoAssoc.keys():
            ra=self.DicoAssoc[iS]["RA"]
            dec=self.DicoAssoc[iS]["DEC"]
            S=self.DicoAssoc[iS]["S"]
            Maj=self.DicoAssoc[iS]["Maj"]
            raOut.append(np.mean(ra))
            decOut.append(np.mean(dec))
            SOut.append(np.sum(S))
            MajOut.append(np.min(Maj))

        Cat=np.zeros((len(raOut),),dtype=[("ra",np.float32),
                                          ("dec",np.float32),
                                          ("S",np.float32),
                                          ("Maj",np.float32)])
        Cat=Cat.view(np.recarray)
        Cat.ra=raOut
        Cat.dec=decOut
        Cat.S=SOut
        Cat.Maj=MajOut
        self.Cat=Cat
        print>>log,"Have merged %i -> %s"%(Ns,Cat.ra.size)

    
    def Cluster(self):

        l,m=self.radec2lm(self.Cat.ra,self.Cat.dec)
        CC=Sky.ClassClusterDEAP.ClassCluster(l,m)
        if self.AvoidPolygons!="":
            print>>log,"Reading polygon file: %s"%self.AvoidPolygons
            PolyList=MyPickle.Load(self.AvoidPolygons)
            LPoly=[]
            for Poly in PolyList:
                ra,dec=Poly.T
                l,m=self.radec2lm(ra,dec)
                Poly[:,0]=l
                Poly[:,1]=m
                
            CC.setAvoidPolygon(PolyList)

        CC.Cluster()

def main(options=None):
        
    if options==None:
        f = open(SaveFile,'rb')
        options = pickle.load(f)
    
    CIM=ClusterImage(**options.__dict__)
    #CIM.GroupSources()
    CIM.Cluster()
    
    
        
if __name__=="__main__":
    read_options()
    f = open(SaveFile,'rb')
    options = pickle.load(f)
    main(options=options)
