import csv
import numpy as np

def ReadBBSModel(infile,infile_cluster=""):
    ifile  = open(infile, "rb")
    reader = csv.reader(ifile)
    F = reader.next()
    F[0]=F[0].lower().replace(" ","").split("(")[-1]
    F[-1]=F[-1].lower().replace(" ","").split(")")[0]
    dtype_str=[]
    default=[]
    killhere=0
    for i in range(len(F)):
        ss=F[i].lower()
        if ss.count("=")>0:
            default.append(ss.split("=")[1].replace("'",""))
            F[i]=ss.split("=")[0].replace(" ","")
        else:
            F[i]=ss.replace(" ","")
            default.append("")
        if F[i]=='kill': killhere=1
    #for i in range(len(F)):
    #    F[i]=F[i].lower().replace(" ","")
        

    Cat=np.zeros((10000,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('Sref',np.float),('I',np.float),('Q',np.float),\
                                 ('U',np.float),('V',np.float),('RefFreq',np.float),('alpha',np.float),('ESref',np.float),\
                                 ('Ealpha',np.float),('kill',np.int),('Cluster',np.int),('Type',np.int),('Gmin',np.float),\
                                 ('Gmaj',np.float),('Gangle',np.float),("Select",np.int),('l',np.float),('m',np.float),("Exclude",bool)])
    Cat=Cat.view(np.recarray)
    Cat.Select=1
    Cat.Exclude=0

    for i in range(len(default)):
        if default[i]!="":
            if F[i]=="spectralindex":
                salpha=default[i].replace("[","").replace("]","")
                if salpha=="": salpha=0.
                Cat.alpha[:]=float(salpha)
                SAlpha_default=salpha
            if F[i]=="referencefrequency":
                Cat.RefFreq[:]=float(default[i])
                SRefFreq_default=default[i]
    
    if F.count('kill')==0:
        Cat.kill=0
        SKill="False"

    icat=0
    while True:
        try:
            L=reader.next()
        except:
            break

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
            if F[i]=="ra":
                SRa=L[i]
                fact=1.
                separ='.'
                sgn=1.
                if "-" in SRa:
                    sgn=-1.
                    SRa=SRa.replace("-","")

                if L[i].count(":")>0:
                    rah,ram,ras=L[i].split(":")
                    ra=sgn*15.*(float(rah)+float(ram)/60.+float(ras)/3600.)*np.pi/180.
                else:
                    rah,ram,ras,rass=L[i].split(".")
                    ncoma=10**len(rass)
                    ra=sgn*(float(rah)+float(ram)/60.+(float(ras)+float(rass)/ncoma)/3600.)*np.pi/180.
                Cat.ra[icat]=ra
                continue
            if F[i]=="dec":
                SDec= L[i]
                sgn=1.
                if "-" in SDec:
                    sgn=-1.
                    SDec=SDec.replace("-","")
                decd,decm,decs,decss=SDec.split(".")
                ncoma=10**len(decss)
                dec=sgn*(float(decd)+float(decm)/60.+(float(decs)+float(decss)/ncoma)/3600.)*np.pi/180.
                Cat.dec[icat]=dec
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
            if F[i]=="referencefrequency":
                SRefFreq=L[i]
                if len(SRefFreq.replace(" ",""))>0:
                    Cat.RefFreq[icat]=float(L[i])
                continue
            if F[i]=="spectralindex":
                SAlpha=L[i].replace("[","").replace("]","")
                if "," in SAlpha:
                    ss=float(SAlpha.split(",")[0])
                else:
                    ss=float(SAlpha)
                Cat.alpha[icat]=ss
                continue
            if F[i]=="kill":
                SKill=L[i]
                #print Cat.Name[icat],L[i]
                Cat.kill[icat]=int(L[i])
                continue
            if F[i]=="type":
                SType=L[i]
                Cat.Type[icat]=(SType!="POINT")
                continue
            if F[i]=="majoraxis":
                Smaj=L[i]
                Cat.Gmaj[icat]=(float(Smaj)/3600.)*(np.pi/180.)/(2.*np.sqrt(2.*np.log(2)))
                continue
            if F[i]=="minoraxis":
                Smin=L[i]
                Cat.Gmin[icat]=(float(Smin)/3600.)*(np.pi/180.)/(2.*np.sqrt(2.*np.log(2)))
                continue
            if F[i]=="orientation":
                Sangle=L[i]
                Cat.Gangle[icat]=(float(Sangle)*np.pi/180.)#+np.pi/2
                continue


#Gmin',np.float),('Gmaj',np.float),('Gangle
#MajorAxis, MinorAxis, Orientation

        if (len(L)==0): continue
        if (L[0][0]=="#")|(L[0][0]==" "): continue
        for i in range(donekey.shape[0]):
            if donekey[i]==False:
                if F[i]=="referencefrequency":
                    SRefFreq=SRefFreq_default
                if F[i]=="spectralindex":
                    SAlpha=SAlpha_default

        icat+=1
    

    
    ifile.close()
    Cat=Cat[Cat.ra!=0.]
    # print Cat.Name
    # print Cat.kill

    if infile_cluster!="":
        ifile  = open(infile_cluster, "rb")
        reader = csv.reader(ifile)
        while True:
            try:
                F=reader.next()
            except:
                break
            F=F[0].split(" ")
            cluster=int(F[0])
            for i in range(1,len(F)):
                if F[i]=='': continue
                ind=np.where(Cat.Name==F[i])[0]
                Cat.Cluster[ind[0]]=cluster
    else:
        Cat.Cluster=range(Cat.shape[0])

    # if (killhere==0)|(len(killdirs)>0):
    #     if (killdirs!=[]):
    #         killnum=1
    #         notkillnum=0
    #         Cat.kill=0
    #         if invert==True:
    #             Cat.kill=1
    #             killnum=0
    #             notkillnum=1
    #         if type(killdirs[0]) is int:
    #             Cat.kill=0
    #             for i in range(len(killdirs)):
    #                 ind=np.where(Cat.Cluster==killdirs[i])[0]
    #                 Cat.kill[ind]=killnum
    #         if type(killdirs[0]) is str:
    #             for i in range(len(killdirs)):
    #                 for j in range(Cat.shape[0]):
    #                     if Cat.Name[j].count(killdirs[i])>0:
    #                         Cat.kill[j]=killnum
    #     else: Cat.kill[:]=1

    Cat.Sref=Cat.I
    return Cat
