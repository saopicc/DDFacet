"""
Useful test function to debug two branches that does the diff of two caches
"""
import numpy as np
import os
import glob

C0="ssd0000.tsel.MS.keep.F0.D0.ddfcache"
C1="ssd0000.tsel.MS.F0.D0.ddfcache"
#SubDirList=[x[0] for x in os.walk(C0)]

def test():
    CheckDir(C0,C1)

def CheckDir(R0,R1):
    print("===============================================")
    LFiles=glob.glob("%s/*"%R0)
    LFiles=[File.split("/")[-1] for File in LFiles]
    print("LFiles",LFiles)
    for File in LFiles: 
        F0="%s/%s"%(R0,File)
        F1="%s/%s"%(R1,File)
        if os.path.isdir(F0):
            print("Exploring subdir: %s"%F0)
            CheckDir(F0,F1)
        else:
            if os.path.getsize(F0)==0 or os.path.getsize(F1)==0: continue
            if F0[-5:]==".hash":
                pass
                #os.system("diff %s %s"%(F0,F1))
            else:
                D0=np.load(F0,allow_pickle=True)
                D1=np.load(F1,allow_pickle=True)
                print("=======")
                print(F0,F1)
                if str(type(D0))=="<class 'numpy.lib.npyio.NpzFile'>":
                    for k in list(D0.keys()):
                        d0=D0[k]
                        d1=D1[k]
                        if d0.size==0:
                            print("   %s: zero size"%k)
                        else:
                            print("   %s:"%k,np.abs(d0-d1).max())
                elif D0.dtype==np.bool:
                    D0=np.float32(D0)
                    D1=np.float32(D1)
                    print("Array diffs: %s.%s - %s.%s"%(str(D0.dtype),str(D0.shape),str(D1.dtype),str(D1.shape)))
                    print("         ",np.abs(D0-D1).max())
                else:
                    print("Array diffs: %s.%s - %s.%s"%(str(D0.dtype),str(D0.shape),str(D1.dtype),str(D1.shape)))
                    print("         ",np.abs(D0-D1).max())
