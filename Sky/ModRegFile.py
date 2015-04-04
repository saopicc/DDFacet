import numpy as np

def test():
    R=RegToNp()
    R.Read()


class RegToNp():
    def __init__(self,RegName="ds9.reg"):
        self.REGFile=RegName
        
    def Read(self):
        f=open(self.REGFile,"r")

        Cat=np.zeros((1000,),dtype=[("ra",np.float32),("dec",np.float32),("Radius",np.float32),("Exclude",np.bool8)])
        Cat=Cat.view(np.recarray)

        Ls=f.readlines()
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
                dec=(float(sdech)+float(sdecm)/60+float(sdecs)/3600.)
                dec*=np.pi/180
                
                rad=(float(srad[0:-1])/3600.)*np.pi/180

                Cat.ra[iCat]=ra
                Cat.dec[iCat]=dec
                Cat.Radius[iCat]=rad
                Cat.Exclude[iCat]=Exclude
                iCat+=1

        Cat=(Cat[Cat.ra!=0]).copy()
        return Cat


        # while True:
        #     print L

        # # f.write('global color=green dashlist=8 3 width=1 font="helvetica 7 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        # # for i in range(self.SourceCat.shape[0]):
        # #     #ss="fk5;ellipse(213.202544,49.871826,0.003909,0.003445,181.376961) # text={P1C1}"
        # #     ra=self.SourceCat.ra[i]*180./np.pi
        # #     dec=self.SourceCat.dec[i]*180./np.pi
        # #     Type=self.SourceCat.Type[i]
        # #     Gmaj=self.SourceCat.Gmaj[i]*180./np.pi*(2.*np.sqrt(2.*np.log(2)))
        # #     Gmin=self.SourceCat.Gmin[i]*180./np.pi*(2.*np.sqrt(2.*np.log(2)))
        # #     if Gmin==0.: Gmin=1./3600
        # #     PA=(self.SourceCat.Gangle[i]+np.pi/2.)*180./np.pi
        # #     rad=20./2600

        # #     #ss="fk5;ellipse(%f,%f,%f,%f,%f) # text={%s}"%(ra,dec,Gmaj,Gmin,0,str(i))
        # #     if self.REGName:
        # #         if Type==1:
        # #             ss="fk5;ellipse(%f,%f,%f,%f,%f) # text={%s} color=green width=2 "%(ra,dec,Gmaj,Gmin,PA,self.SourceCat.Name[i])
        # #         else:
        # #             ss="fk5;point(%f,%f) # text={%s} point=circle 5 color=red width=2"%(ra,dec,self.SourceCat.Name[i])
        # #     else:
        # #         if Type==1:
        # #             ss="fk5;ellipse(%f,%f,%f,%f,%f) # color=green width=2 "%(ra,dec,Gmaj,Gmin,PA)
        # #         else:
        # #             ss="fk5;point(%f,%f) # point=circle 5 color=red width=2"%(ra,dec)
                

        #     f.write(ss+"\n")
        f.close()
