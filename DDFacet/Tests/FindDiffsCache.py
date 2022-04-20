#!/usr/bin/env python
"""
Useful test function to debug two branches that does the diff of two caches
"""
import numpy as np
import os
import glob
from DDFacet.Other import ModColor
import sys
from DDFacet.Array import shared_dict
import DDFacet.Other.MyPickle
import traceback

def test():
    
    ll=sorted(glob.glob("L242820_SB150_uv.dppp.pre-cal_125145DD7t_150MHz.pre-cal.ms.tsel.F0.D0.ddfcache"))
    for l in ll:
        C0=l
        C1="%s.master"%l
        CheckDir(C0,C1)

def PNameValue(Name="key",Value=0):
    if float(Value)>0:
        print("%s = %s"%(Name,ModColor.Str(str(Value))))
        return "Different"
    else:
        print("%s = %s"%(Name,str(Value)))
        return "Same"

D={"A":"a",
   "B":np.arange(5),
   "C":{"A":5,
        "B":np.arange(2)}}

def giveStrDiffObj(D0,D1):
    if isinstance(D0,np.ndarray):
        if D0.dtype==np.bool:
            D0=np.float32(D0)
            D1=np.float32(D1)
            diff=np.abs(D0-D1).max()
        else:
            diff=np.abs(D0-D1).max()
        return str(diff)
    else:
        return D0-D1
    
        
def DiffObjPrint(Obj0,Obj1,LLevel=[]):


    def giveStrTitle(Ls):
        s=""
        for k in Ls:
            kk=k
            s+=kk.ljust(15)+":"
        return s
    if isinstance(Obj0,list) or isinstance(Obj0,tuple) or isinstance(Obj0,float) or isinstance(Obj0,int) :
        try:
            Obj0=np.array(Obj0,dtype=object)
            Obj1=np.array(Obj1,dtype=object)
        except:
            print("could not convert to array: %s "%str(Obj0))
    
    if isinstance(Obj0,np.ndarray) or isinstance(Obj0,float):
        A0=Obj0
        A1=Obj1
        Sdiff=giveStrDiffObj(A0,A1)
        s="%s [%s] "%(giveStrTitle(LLevel),str(A0.shape))
        r=PNameValue(Name=s,Value=Sdiff)
        if len(A0.shape)>=4:#(r=="Different") and (float(Sdiff)>1e-3):
            print("Different")
            nx,ny=A0.shape[-2:]
            op=np.real
            a0=op(A0.reshape((A0.size//(nx*ny),nx,ny))[0])
            a1=op(A1.reshape((A1.size//(nx*ny),nx,ny))[0])#.T[::-1,:]
            import pylab
            pylab.clf()
            ax=pylab.subplot(1,3,1)
            pylab.imshow(np.log10(np.abs(a0)),interpolation="nearest")
            pylab.colorbar()
            pylab.subplot(1,3,2,sharex=ax,sharey=ax)
            pylab.imshow(np.log10(np.abs(a1)),interpolation="nearest")
            pylab.colorbar()
            pylab.subplot(1,3,3,sharex=ax,sharey=ax)
            pylab.imshow(a0-a1,interpolation="nearest")
            pylab.colorbar()
            pylab.suptitle(str(s)+" %f"%np.max(a0-a1))
            
            pylab.draw()
            pylab.show()
            pylab.pause(0.1)
        else:
            print("Same")
            
    elif isinstance(Obj0,dict):
        for k in sorted(Obj0.keys()):
            
            A0=Obj0[k]
            try:
                A1=Obj1[k]
                DiffObjPrint(A0,A1,LLevel=LLevel+[str(k)])
            except Exception:
                traceback.print_exc()
                s="%s [missing key %s]"%(giveStrTitle(LLevel),k)

                
    else:
        stop    
        
        
def CheckDir(R0,R1):
    print("===============================================")
    LFiles=glob.glob("%s/*"%R0)
    LFiles=[File.split("/")[-1] for File in LFiles]
    print("Comparing: ",LFiles)
    for File in LFiles: 
        F0="%s/%s"%(R0,File)
        F1="%s/%s"%(R1,File)
        CheckFiles(F0,F1)

def CheckFiles(F0,F1):
    if os.path.isdir(F0):
        print("Exploring subdirs: %s"%F0)
        if "SubDico" not in F0 and "SubArray" not in F0:
            CheckDir(F0,F1)
    else:
        #if (os.path.isfile(F0) and os.path.isfile(F1))==0: continue
        if os.path.getsize(F0)==0 or os.path.getsize(F1)==0:
            return
        #if "Dirty" in F0 or "CF" in F0:
        if False:#"CF" in F0:
            print("Skipping ",F0)
            return
        elif F0[-5:]==".hash":
            return
            # os.system("diff %s %s"%(F0,F1))
        else:
            print("=======")
            print("Directly comparing as dicts:",F0,F1)
            if F0[-3:]=="PSF":
                D0 = shared_dict.create("PSF0")
                D0.restore(F0)
                D0.reload()
                D1 = shared_dict.create("PSF1")
                D1.restore(F1)
                D1.reload()
            elif F0[-5:]=="Dirty":
                D0 = shared_dict.create("Dirty0")
                D0.restore(F0)
                D0.reload()
                D1 = shared_dict.create("Dirty1")
                D1.restore(F1)
                D1.reload()
            elif F0[-10:]=="DicoPickle":
                D0=DDFacet.Other.MyPickle.FileToDicoNP(F0)
                D1=DDFacet.Other.MyPickle.FileToDicoNP(F1)
            else:
                D0=np.load(F0,allow_pickle=True)
                D1=np.load(F1,allow_pickle=True)
                
            if str(type(D0))=="<class 'numpy.lib.npyio.NpzFile'>":
                D0s={}
                D1s={}
                for k in D0.keys():
                    D0s[k]=D0[k]
                for k in D1.keys():
                    D1s[k]=D1[k]
                D0=D0s
                D1=D1s
            DiffObjPrint(D0,D1,LLevel=[F0.split("/")[-1]])
                
if __name__=="__main__":
    CheckDir(*sys.argv[1:])
