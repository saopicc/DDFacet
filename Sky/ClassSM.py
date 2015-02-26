import csv
import numpy as np
import rad2hmsdms
import ModColor
import ModTigger
import ModSMFromNp




class ClassSM():
    def __init__(self,infile,infile_cluster="",killdirs=[],invert=False,solveFor=[],DoPrintCat=False,\
                     ReName=False,DoREG=False,SaveNp=False,NCluster=0,DoPlot=True,Tigger=False,\
                     FromExt=None,ClusterMethod=1,SelSource=False):
        self.ClusterMethod=ClusterMethod
        self.infile_cluster=infile_cluster
        self.TargetList=infile
        if ".npy" in infile:
            Cat=np.load(infile)
            Cat=Cat.view(np.recarray)
        elif Tigger:
            Cat=ModTigger.ReadTiggerModel(infile)
        elif FromExt!=None:
            Cat=ModSMFromNp.ReadFromNp(FromExt.ra,FromExt.dec,FromExt.s)
        else:
            Cat=self.ReadBBSModel(infile,infile_cluster=infile_cluster)
        self.SourceCat=Cat

        self.NDir=np.max(self.SourceCat.Cluster)+1
        self.NSources=Cat.shape[0]
        if DoPrintCat:
            self.print_sm(Cat)

        if killdirs!=[]:
            self.SourceCat.kill=0
            for i in range(len(self.SourceCat)):
                for StrPiece in killdirs:
                    if StrPiece in self.SourceCat.Name[i]: self.SourceCat.kill[i]=1
        if invert:
            ind0=np.where(self.SourceCat.kill==0)[0]
            ind1=np.where(self.SourceCat.kill==1)[0]
            self.SourceCat.kill[ind0]=1
            self.SourceCat.kill[ind1]=0

        # for i in range(self.SourceCat.shape[0]):
        #     print "%s: %i"%(self.SourceCat.Name[i],self.SourceCat.kill[i])

        if NCluster!=0:
            self.cluster(NCluster,DoPlot)
            #print self.SourceCat.Cluster
            ClusterList=sorted(list(set(self.SourceCat.Cluster.tolist())))
            self.NDir=len(ClusterList)
            for iCluster,iNewCluster in zip(ClusterList,range(self.NDir)):
                ind=np.where(self.SourceCat.Cluster==iCluster)[0]
                self.SourceCat.Cluster[ind]=iNewCluster
            #print self.SourceCat.Cluster
            #print

        self.REGName=False
        if ReName:
            for diri in range(self.NDir):
                ind=np.where(self.SourceCat.Cluster==diri)[0]
                #CatSel=self.SourceCat[self.SourceCat.Cluster==diri]
                Names=["c%is%i."%(diri,i) for i in range(ind.shape[0])]
                self.SourceCat.Name[ind]=Names
            self.REGName=True

        self.REGFile=None
        if DoREG:
            self.MakeREG()

        self.Dirs=sorted(list(set(self.SourceCat.Cluster.tolist())))
        self.WeightDirKeep=np.zeros((self.NDir,),float)
        for diri in self.Dirs:
            ind=np.where(self.SourceCat.Cluster==diri)[0]
            self.WeightDirKeep[diri]=np.sqrt(np.sum(self.SourceCat.Sref[ind]))
        self.WeightDir=self.WeightDirKeep.copy()

        self.ExistToSub=False
        self.ExistToSub=(np.count_nonzero(self.SourceCat.kill==-1)>0)

        self.SourceCatKeepForSelector=self.SourceCat.copy()

        self.BuildClusterCat()

        if SelSource:
            self.SelectSourceMouse()
        #print self.SourceCat.Select

        print ModColor.Str(" SkyModel PROPERTIES: ")
        print "   - SkyModel File Name: %s"%ModColor.Str(infile,col="green")
        if self.REGFile!=None: print "   - ds9 region file: %s"%ModColor.Str(self.REGFile,col="green")
        npext=""
        if not(".npy" in infile): npext=".npy"
        self.NpFile="%s%s"%(infile,npext)
        if SaveNp:
            
            np.save(infile,self.SourceCat)
            print "   - Numpy catalog file: %s"%ModColor.Str("%s%s"%(infile,npext),col="green")

        #print "Oufile: %s"%self.infile_cluster
        #if infile_cluster!="":
        #print "   - Cluster File Name: %s"%self.infile_cluster
        print "   - Number of Sources  = ",Cat.shape[0]
        print "   - Number of Directions  = ",self.NDir
        print
    
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
            ClusterCat.ra[icat]=np.sum(cat.ra*cat.I)/np.sum(cat.I)
            ClusterCat.dec[icat]=np.sum(cat.dec*cat.I)/np.sum(cat.I)
            ClusterCat.SumI[icat]=np.sum(cat.I)
            ClusterCat.Cluster[icat]=d
            icat+=1
        self.ClusterCat=ClusterCat
        


    def radec2lm_scalar(self,ra,dec):
        l = np.cos(dec) * np.sin(ra - self.rarad)
        m = np.sin(dec) * np.cos(self.decrad) - np.cos(dec) * np.sin(self.decrad) * np.cos(ra - self.rarad)
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



    def MakeREG(self):
        self.REGFile="%s.reg"%self.TargetList
        f=open(self.REGFile,"w")

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
        self.WeightDir=self.WeightDirKeep[selDir].copy()


    def SelectSubCat(self,Selector):
        self.Selector=Selector
        self.SourceCat=(self.SourceCatKeepForSelector[self.Selector]).copy()
        self.Dirs=sorted(list(set(self.SourceCat.Cluster.tolist())))
        self.NDir=len(self.Dirs)
        self.NSources=self.SourceCat.shape[0]
        selDir=np.array(sorted(list(set(self.SourceCat.Cluster.tolist()))))
        self.WeightDir=self.WeightDirKeep[selDir].copy()


        

    def cluster(self,nk=10,DoPlot=False):

        import pylab
        import time
        # pylab.clf()
    
        #s.fill(0.)
        #s[0]=1
        self.rarad=np.sum(self.SourceCat.I*self.SourceCat.ra)/np.sum(self.SourceCat.I)
        self.decrad=np.sum(self.SourceCat.I*self.SourceCat.dec)/np.sum(self.SourceCat.I)
        x,y,s=self.SourceCat.ra,self.SourceCat.dec,self.SourceCat.I
        x,y=self.radec2lm_scalar(x,y)
        
        if self.ClusterMethod==2:
            import ModCluster
            self.SourceCat.Cluster=0
            DictNode=ModCluster.tessel(x,y,s,nk,DoPlot=DoPlot)
            iK=0
            self.NDir=len(DictNode.keys())
            for key in DictNode.keys():
                ind=np.array(DictNode[key]["ListCluster"])
                self.SourceCat.Cluster[ind]=iK
                iK+=1
            return

        if self.ClusterMethod==3:
            self.SourceCat.Cluster=0
            import ModClusterRadial
            DictNode=ModClusterRadial.RadialCluster(x,y,s,nk,DoPlot=DoPlot)
            iK=0
            self.NDir=len(DictNode.keys())
            for key in DictNode.keys():
                ind=np.array(DictNode[key]["ListCluster"])
                #print ind
                self.SourceCat.Cluster[ind]=iK
                iK+=1
            return

        x0,x1=np.min(x),np.max(x)#0.9*np.min(x),1.1*np.max(x)
        y0,y1=np.min(y),np.max(y)#0.9*np.min(y),1.1*np.max(y)
    
        Dx=1e-2#5*np.abs(x1-x0)
        Dy=1e-2#5*np.abs(y1-y0)
        x0,x1=x0-Dx,x1+Dx
        y0,y1=y0-Dy,y1+Dy
        # for i in range(x.shape[0]):
        #     print "(%6.2f, %6.2f)"%(x[i]*1000,y[i]*1000)

        #xx,yy=np.mgrid[x0:x1:40j,y0:y1:40j]
        Np=100
        xx,yy=np.mgrid[x0:x1:Np*1j,y0:y1:Np*1j]
        
        sigx=1.5*(x0-x1)/nk
        sigy=1.5*(y0-y1)/nk
        sigx=1.*(x0-x1)/nk
        sigy=1.*(y0-y1)/nk
    
        xnode=[]
        ynode=[]
        ss=np.zeros(xx.shape,dtype=xx.dtype)
        ss0=np.zeros(xx.shape,dtype=xx.dtype)
        dx,dy=(x1-x0)/Np,(y1-y0)/Np
    
    
        C=1./(2.*np.pi*sigx*sigy)
        for i in range(x.shape[0]):
            ss+=s[i]*C*np.exp(-((xx-x[i])**2/sigx**2+(yy-y[i])**2/sigy**2))*dx*dy
            ss0+=C*np.exp(-((xx-x[i])**2/sigx**2+(yy-y[i])**2/sigy**2))*dx*dy
            
        # xr=np.random.rand(1000)
        # yr=np.random.rand(1000)
        # C=1./(2.*np.pi*sigx*sigy)
        # for i in range(xr.shape[0]):
        #     ss0+=C*np.exp(-((xx-xr[i])**2/sigx**2+(yy-yr[i])**2/sigy**2))
        # ss0/=xr.shape[0]
        # ssn=ss/ss0#(ss-ss0)/ss0
    
        ss/=np.sqrt(ss0)

        #ssn=ss
        # pylab.scatter(xx,yy,marker="s",c=ss,s=50)
        # pylab.scatter(x,y,c=s,s=np.sqrt(s)*50)
        # pylab.colorbar()
        # pylab.draw()
        # #time.sleep(1)
        # #return
        vmin,vmax=np.min(ss),np.max(ss)
        #print
        #print vmin,vmax
        if DoPlot:
            pylab.ion()
            pylab.scatter(xx,yy,marker="s",c=ss,s=50,vmin=vmin,vmax=vmax)
            pylab.xlim(x0,x1)
            pylab.ylim(y0,y1)
            pylab.draw()
            pylab.show()
        for i in range(nk):
            time.sleep(1)
            ind=np.where(ss==np.max(ss))
            xnode.append(xx[ind])
            ynode.append(yy[ind])
            ss-=ss[ind]*np.exp(-((xx-xnode[-1])**2/sigx**2+(yy-ynode[-1])**2/sigy**2))


            if DoPlot:
                pylab.clf()
                pylab.scatter(xx,yy,marker="s",c=ss,s=50)#,vmin=vmin,vmax=vmax)
                #pylab.scatter(xnode[-1],ynode[-1],marker="s",vmin=vmin,vmax=vmax)
                pylab.scatter(xnode,ynode,marker="s",color="red",vmin=vmin,vmax=vmax)
                #pylab.scatter(x,y,marker="o")#,vmin=vmin,vmax=vmax)
                #pylab.colorbar()
                pylab.xlim(x0,x1)
                pylab.ylim(y0,y1)
                pylab.draw()
                pylab.show()
                pylab.pause(0.1)

        pylab.ioff()

        xnode=np.array(xnode)
        ynode=np.array(ynode)
        KK={}
        keys=[]
        for i in range(nk):
            key="%3.3i"%i
            keys.append(key)
            KK[key]=[]
    
    
        for i in range(x.shape[0]):
            d=np.sqrt((x[i]-xnode)**2+(y[i]-ynode)**2)
            ind=np.where(d==np.min(d))[0][0]
            (KK[keys[ind]]).append(i)

        if DoPlot:
            import ModCluster
            pylab.clf()
            Dx=Dy=0.01
            extent=(np.min(x)-Dx,np.max(x)+Dx,np.min(y)-Dy,np.max(y)+Dy)
            ModCluster.PlotTessel(xnode,ynode,extent)
        #self.infile_cluster="%s.cluster"%self.TargetList
        #f=file(self.infile_cluster,"w")
    



        for key in keys:
            ll=KK[key]
                
            self.SourceCat.Cluster[ll]=int(key)
            ss="%s "*len(ll)
            #print self.SourceCat.Name[ll]
            lout=[key]+self.SourceCat.Name[ll].tolist()
            #sout=("Cluster%s "+ss)%(tuple(lout))
            sout=("%s "+ss)%(tuple(lout))
            #print sout
            ind=ll
            #f.write(sout+"\n")
            cc=np.ones((len(ll),))*int(key[-2::])

            if DoPlot: pylab.scatter(x[ind],y[ind],c=cc,s=np.sqrt(s[ind])*50,vmin=0,vmax=nk)
        if DoPlot:
            pylab.tight_layout()
            pylab.draw()
            pylab.show()
        self.NDir=np.max(self.SourceCat.Cluster)+1
        #f.close()
        #import os
        #os.system("cat %s"%self.infile_cluster)


    def CorrPA(self,imin):
        from pyrap.images import image
        import scipy.linalg
        R2arc=1.#3600*180/np.pi

        im=image(imin)
        PMaj=(im.imageinfo()["restoringbeam"]["major"]["value"]/3600.)*(np.pi/180.)/np.sqrt(2.*np.log(2))
        PMin=(im.imageinfo()["restoringbeam"]["minor"]["value"]/3600.)*(np.pi/180.)/np.sqrt(2.*np.log(2))
        PPA=(im.imageinfo()["restoringbeam"]["positionangle"]["value"]*np.pi/180.)

        # self.SourceCat.Gmaj[:]=10.0*PMaj
        # self.SourceCat.Gmin[:]=1.0*PMin
        # self.SourceCat.Gangle[:]=1.0*PPA

        ind=self.SourceCat.Gmaj<=PMaj
        self.SourceCat.Gmaj[ind]=PMaj
        self.SourceCat.Gmin[ind]=PMin
        self.SourceCat.Gangle[ind]=PPA
        ind=self.SourceCat.Gmin<=PMin
        self.SourceCat.Gmin[ind]=PMin
        
        


        P_a,P_b,P_c=self.Give_GaussABC(PMin*R2arc,PMaj*R2arc,PPA)



        SigMaj=np.zeros_like(self.SourceCat.Gmaj)
        SigMin=np.zeros_like(self.SourceCat.Gmaj)
        PA=np.zeros_like(self.SourceCat.Gmaj)
        for i in range(self.SourceCat.Gmaj.shape[0]):
            # print
            # print self.SourceCat.Name[i],self.SourceCat.Gmin[i]*R2arc,self.SourceCat.Gmaj[i]*R2arc,self.SourceCat.Gangle[i]
            # print PMin*R2arc,PMaj*R2arc,PPA
            M_a,M_b,M_c=self.Give_GaussABC(self.SourceCat.Gmin[i]*R2arc,self.SourceCat.Gmaj[i]*R2arc,self.SourceCat.Gangle[i])
            da=M_a-P_a
            db=M_b-P_b
            dc=M_c-P_c
            M=np.abs(np.array([[da,db],[db,dc]]))
            #print M
            u,l,d=scipy.linalg.svd(M)
            #print u
            #print l
            #Theta,SigMin0,SigMaj0= -np.angle(-(u[0,0]+1j*u[0,1]))/np.pi,np.sqrt(np.abs(u[0,0]/l[0]))/np.sqrt(2.),\
            #    np.sqrt(l[0]/l[1])*np.sqrt(np.abs(u[0,0]/l[0]))/np.sqrt(2.)


            Theta=-np.angle(-(u[0,0]+1j*u[0,1]))/np.pi

            if l[0]!=0:
                a=1./l[0]
                SigMin0=1./np.sqrt(a/2.)
            else:
                SigMin0=0.
            if l[1]!=0:
                b=1./l[1]
                SigMaj0=1./np.sqrt(b/2.)
            else:
                SigMaj0=0.
            if SigMin0>SigMaj0:
                c=SigMaj0
                SigMaj0=SigMin0
                SigMin0=c
                Theta+=np.pi/2

            #print "Smin=%f Smaj=%s PA=%f"%(SigMin0*3600*180/np.pi, SigMaj0*3600*180/np.pi, Theta)

            SigMin[i]=SigMin0
            SigMaj[i]=SigMaj0
            PA[i]=Theta


        # import pylab
        # pylab.clf()
        # pylab.plot(self.SourceCat.Gmaj,SigMaj,ls="",marker="+")
        # pylab.plot(self.SourceCat.Gmin,SigMin,ls="",marker="+")
        # pylab.draw()
        # pylab.show()
        self.SourceCat.Gmaj=SigMaj
        self.SourceCat.Gmin=SigMin
        self.SourceCat.Gangle=Theta
        #print self.SourceCat.Gmaj
        self.SourceCat.Type[self.SourceCat.Gmaj==0]=0



    def ReadBBSModel(self,infile,infile_cluster=""):
        ifile  = open(infile, "rb")
        reader = csv.reader(ifile)
        F=reader.next()
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
            
    
        Cat=np.zeros((1000,),dtype=[('Name','|S200'),('ra',np.float),('dec',np.float),('Sref',np.float),('I',np.float),('Q',np.float),\
                                    ('U',np.float),('V',np.float),('RefFreq',np.float),('alpha',np.float),('ESref',np.float),\
                                    ('Ealpha',np.float),('kill',np.int),('Cluster',np.int),('Type',np.int),('Gmin',np.float),\
                                    ('Gmaj',np.float),('Gangle',np.float),("Select",np.int)])
        Cat=Cat.view(np.recarray)
        Cat.Select=1
    
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
                    if L[i].count(":")>0:
                        rah,ram,ras=L[i].split(":")
                        ra=15.*(float(rah)+float(ram)/60.+float(ras)/3600.)*np.pi/180.
                    else:
                        rah,ram,ras,rass=L[i].split(".")
                        ncoma=10**len(rass)
                        ra=(float(rah)+float(ram)/60.+(float(ras)+float(rass)/ncoma)/3600.)*np.pi/180.
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


    def Give_GaussABC(self,m0in,m1in,ang):
        m0=1./m0in
        m1=1./m1in
        a=0.5*((np.cos(ang)/m0)**2+(np.sin(ang)/m1)**2)
        b=0.25*(-np.sin(2*ang)/(m0**2)+np.sin(2.*ang)/(m1**2))
        c=0.5*((np.sin(ang)/m0)**2+(np.cos(ang)/m1)**2)
        return a,b,c

    def print_sm2(self):
        CatIn=self.SourceCat
        ind=np.argsort(CatIn.Cluster)
        Cat=CatIn[ind]
        TEMPLATE = ('  %(Cluster)5s %(name)10s %(RA)15s %(DEC)15s %(Flux)10s %(alpha)10s %(RefFreq)10s')# %(Kill)6s ')
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
                    

    
    
    
    
