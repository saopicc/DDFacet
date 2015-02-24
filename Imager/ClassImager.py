

# import multiprocessing
# class WorkerImaging(multiprocessing.Process):
#     def __init__(self,
#             work_queue,
#             result_queue):
#         multiprocessing.Process.__init__(self)
#         self.work_queue = work_queue
#         self.result_queue = result_queue
#         self.kill_received = False
#         self.exit = multiprocessing.Event()
#     def shutdown(self):
#         self.exit.set()
#     def run(self):
#         while not self.kill_received:
#             try:
#                 job = self.work_queue.get()
#             except:
#                 break
#             Row0,Row1,x0=job
#             xi,xii=estimate_xi_pseudo(Row0,Row1)
#             self.result_queue.put([xi,Row0,Row1])

import ClassGridMachine
import numpy as np
import ClassMS
import pylab
import ClassCasaImage

def testFacet():

    wmax=10000
    MS=ClassMS.ClassMS("MSTest.MS",Col="CORRECTED_DATA")
    Imager=ClassFacetImager(Support=5,NFacets=2,OverS=5,wmax=wmax,Nw=101,Npix=1024,Cell=20.,ChanFreq=MS.ChanFreq.flatten(),
                            Padding=2,RaDecRad=MS.radec)
    #MS.data.fill(100)
    Imager.GiveDirtyimage(MS.uvw,MS.data,MS.flag_all)
    return Imager

def PlotResult(Imager):
    pylab.clf()
    i=0
    for key in Imager.DicoImager.keys():
        pylab.subplot(2,2,i+1); i+=1
        Dirty=Imager.DicoImager[key]["Dirty"]
        
        pylab.imshow(Dirty[0,0].T.real,interpolation="nearest")
        pylab.draw()
        pylab.pause(0.1)
        pylab.show(False)


class ClassFacetImager():
    def __init__(self,
                 Npix=512,Cell=10.,
                 NFacets=5,
                 ChanFreq=np.array([6.23047e7],dtype=np.float64),
                 Support=11,OverS=5,Padding=1.2,
                 WProj=False,wmax=10000,Nw=11,RaDecRad=(0.,0.)):
# ,
#                  ImageName="Image",RaDecRad=(0.,0.),
#                  DoPSF=True,lmShift=None):

        lrad=Npix*(Cell/3600.)*0.5*np.pi/180.
        lfacet=lrad/NFacets
        lcenter_max=lrad-lfacet
        lFacet,mFacet,=np.mgrid[-lcenter_max:lcenter_max:(NFacets)*1j,-lcenter_max:lcenter_max:(NFacets)*1j]
        NpixFacet=Npix/NFacets
        lFacet=lFacet.flatten()
        mFacet=mFacet.flatten()
        self.Npix=Npix
        self.CasaImage=ClassCasaImage.ClassCasaimage("ImageTotal",Npix,Cell,RaDecRad)
        self.cMain = self.CasaImage.im.coordinates()

        # import pylab
        # pylab.clf()
        # pylab.scatter(lFacet,mFacet)
        # pylab.plot([-lrad,lrad,lrad,-lrad,-lrad],[-lrad,-lrad,lrad,lrad,-lrad])
        # pylab.draw()
        # pylab.show(False)
        # pylab.clf()
        self.DicoImager={}
        for iFacet in range(lFacet.size):
            self.DicoImager[iFacet]={}
            lmShift=(lFacet[iFacet],mFacet[iFacet])
            xc,yc=RaDecRad[0]+lFacet[iFacet],RaDecRad[1]+mFacet[iFacet]
            TransfRaDec=[RaDecRad,(xc,yc)]
            lfacet=NpixFacet*(Cell/3600.)*0.5*np.pi/180.
            # x0=xc-lfacet
            # x1=xc+lfacet
            # y0=yc-lfacet
            # y1=yc+lfacet
            # pylab.scatter([xc],[yc])
            # pylab.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0])
            # pylab.draw()
            self.DicoImager[iFacet]["lmShift"]=lmShift
            GridMachine=ClassGridMachine.ClassGridMachine(Npix=NpixFacet,Cell=Cell,ChanFreq=ChanFreq,DoPSF=False,
                                                          Support=Support,OverS=OverS,
                                                          wmax=wmax,Nw=Nw,WProj=True,
                                                          Padding=Padding,TransfRaDec=TransfRaDec,ImageName="Image_%i"%iFacet)
            self.DicoImager[iFacet]["GridMachine"]=GridMachine

        



    def GiveDirtyimage(self,uvwIn,visIn,flag,doStack=False):
        uvw=uvwIn.copy()
        vis=visIn.copy()
        Stack=self.CasaImage.im.getdata()
        NpPoints=np.zeros(Stack.shape,dtype=np.int64)
        Npix=self.Npix
        pylab.clf()
        for iFacet in self.DicoImager.keys():
            GridMachine=self.DicoImager[iFacet]["GridMachine"]
            GridMachine.put(uvw,vis,flag,doStack=False)
            self.DicoImager[iFacet]["Dirty"]=GridMachine.getDirtyIm()
            img=GridMachine.CasaImage.im
            ImStack=img.regrid( [0,1], self.cMain, outshape=(int(Npix),int(Npix)))
            DataAtStack=ImStack.getdata()
            Stack+=DataAtStack
            DataAtStack[DataAtStack!=0]=1
            NpPoints+=DataAtStack
         
        NpPoints[NpPoints==0]=1
        Stack/=NpPoints
        pylab.imshow(Stack.T.real,interpolation="nearest")
        pylab.draw()
        pylab.pause(0.1)
        pylab.show(False)
           
