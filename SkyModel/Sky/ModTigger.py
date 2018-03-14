import numpy as np

def ReadTiggerModel(infile,header=["name", "ra_d", "dec_d", "i", "emaj_d", "emin_d", "pa_d"]):

    


    Cat=np.zeros((1000,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('Sref',np.float),('I',np.float),('Q',np.float),\
                                ('U',np.float),('V',np.float),('RefFreq',np.float),('alpha',np.float),('ESref',np.float),\
                                ('Ealpha',np.float),('kill',np.int),('Cluster',np.int),('Type',np.int),('Gmin',np.float),\
                                ('Gmaj',np.float),('Gangle',np.float)])
    Cat=Cat.view(np.recarray)
    Cat.RefFreq=1.

    icat=0
    fin=file(infile,"r")
    Lines=fin.readlines()
    fin.close()
    F=header

    for Lin in Lines:
        L=Lin.replace("\n","").split(" ")

        ok=0
        donekey=np.zeros((len(F),),dtype=np.bool)
        #print L
        for i in range(len(L)):
            if L[0][0]=="#": break
            ok=1
            donekey[i]=True
            L[i]=L[i].replace(" ","")
            if len(L[i])==0: continue
            #print "%3i, %30s, %s"%(icat,F[i],L[i])
            if F[i]=="name":
                SName=L[i]
                Cat.Name[icat]=L[i]
                continue
            if (F[i]=="ra_d"):
                Cat.ra[icat]=float(L[i])*np.pi/180
                continue
            if (F[i]=="dec_d"):
                Cat.dec[icat]=float(L[i])*np.pi/180
                continue
            if F[i]=="i":
                SI=L[i]
                Cat.I[icat]=float(L[i])
                continue
            if F[i]=="q":
                SI=L[i]
                Cat.Q[icat]=float(L[i])
                continue
            if F[i]=="u":
                SI=L[i]
                Cat.U[icat]=float(L[i])
                continue
            if F[i]=="v":
                SI=L[i]
                Cat.V[icat]=float(L[i])
                continue
            if F[i]=="type":
                SType=L[i]
                Cat.Type[icat]=(SType!="POINT")
                continue

            if F[i]=="emaj_d":
                Smaj=L[i]
                Cat.Gmaj[icat]=(float(Smaj))*(np.pi/180.)/(2.*np.sqrt(2.*np.log(2)))
                continue
            if F[i]=="emin_d":
                Smin=L[i]
                Cat.Gmin[icat]=(float(Smin))*(np.pi/180.)/(2.*np.sqrt(2.*np.log(2)))
                continue
            if F[i]=="pa_d":
                Sangle=L[i]
                Cat.Gangle[icat]=(float(Sangle)*np.pi/180.)#+np.pi/2
                continue

    
    

    Cat=Cat[Cat.ra!=0.]
    Cat.Type[Cat.Gmaj>0.]=1

    Cat.Sref=Cat.I
    return Cat


