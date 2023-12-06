'''
DDFacet, a facet-based radio imaging package
Copyright (C) 2013-2016  Cyril Tasse, l'Observatoire de Paris,
SKA South Africa, Rhodes University

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import numpy as np
from DDFacet.Imager import ClassFacetMachine
from DDFacet.Other import MyPickle
from scipy.spatial import Voronoi, ConvexHull
from SkyModel.Sky import ModVoronoi
from DDFacet.Other import reformat
from DDFacet.Other import ModColor
from DDFacet.Imager.ModModelMachine import ClassModModelMachine
from DDFacet.Other import Exceptions

from DDFacet.Data.ClassJones import _parse_solsfile, _which_solsfile
import os, glob

from DDFacet.Data.ClassJones import _parse_solsfile
import os

from DDFacet.ToolsDir.ModToolBox import EstimateNpix
import tables

from matplotlib.path import Path
from SkyModel.Sky import ModVoronoiToReg
import Polygon
from DDFacet.ToolsDir import rad2hmsdms
from DDFacet.Other import logger
log = logger.getLogger("ClassFacetMachineTessel")
from pyrap.images import image


class ClassFacetMachineTessel(ClassFacetMachine.ClassFacetMachine):
    """
    This class extends the basic infrastructure set out in ClassFacetMachine to
    split the sky into a Voronoi tesselated sky. The projection, unprojection and resampling logic
    remains the same.
    """

    def __init__(self, *args, **kwargs):
        ClassFacetMachine.ClassFacetMachine.__init__(self, *args, **kwargs)

    def setFacetsLocs(self):
        NFacets = self.NFacets
        Npix = self.GD["Image"]["NPix"]
        Padding = self.GD["Facets"]["Padding"]
        self.Padding = Padding
        Npix, _ = EstimateNpix(float(Npix), Padding=1)
        self.Npix = Npix
        self.OutImShape = (self.nch, self.npol, self.Npix, self.Npix)

        RadiusTot = self.CellSizeRad * self.Npix / 2
        self.RadiusTot = RadiusTot

        lMainCenter, mMainCenter = 0., 0.
        self.lmMainCenter = lMainCenter, mMainCenter
        self.CornersImageTot = np.array(
            [[lMainCenter - RadiusTot, mMainCenter - RadiusTot],
             [lMainCenter + RadiusTot, mMainCenter - RadiusTot],
             [lMainCenter + RadiusTot, mMainCenter + RadiusTot],
             [lMainCenter - RadiusTot, mMainCenter + RadiusTot]])

        # MSName = self.GD["Data"]["MS"]
        # if ".txt" in MSName:
        #     f = open(MSName)
        #     Ls = f.readlines()
        #     f.close()
        #     MSName = []
        #     for l in Ls:
        #         ll = l.replace("\n", "")
        #         MSName.append(ll)
        #     MSName = MSName[0]

        MSName = self.VS.ListMS[0].MSName

        SolsFile = self.GD["DDESolutions"]["DDSols"]
        if isinstance(SolsFile, list):
            SolsFile = self.GD["DDESolutions"]["DDSols"][0]

        if SolsFile=="": SolsFile=None


        if SolsFile and (not (".npz" in SolsFile)) and (not (".h5" in SolsFile)):
            Method = SolsFile
            # ThisMSName = reformat.reformat(
            #     os.path.abspath(MSName), LastSlash=False)
            # SolsFile = "%s/killMS.%s.sols.npz" % (ThisMSName, Method)
            SolsDir=self.GD["DDESolutions"]["SolsDir"]
            if SolsDir is None or SolsDir=="":
                ThisMSName = reformat.reformat(os.path.abspath(MSName), LastSlash=False)
                SolsFile = "%s/killMS.%s.sols.npz" % (ThisMSName, Method)
            else:
                _MSName=reformat.reformat(os.path.abspath(MSName).split("/")[-1])
                DirName=os.path.abspath("%s%s"%(reformat.reformat(SolsDir),_MSName))
                if not os.path.isdir(DirName):
                    os.makedirs(DirName)
                SolsFile = "%s/killMS.%s.sols.npz"%(DirName,SolsFile)

        
        
        
        if SolsFile is not None and not ".h5" in SolsFile:
            DoCheckParset=False
            if "killMS" in SolsFile:
                try:
                    from killMS.Parset import ReadCFG as ReadCFGkMS
                    PName=SolsFile[0:-4]+".parset"
                    DoCheckParset=os.path.isfile(PName)
                    if not DoCheckParset:
                        log.print(ModColor.Str("File %s does not exist - can't check parset consistencies"%PName))
                except:
                    log.print(ModColor.Str("Cannot import killMS - can't check parset consistencies"))
                    DoCheckParset=False
                    
            if DoCheckParset:
                print("reading %s"%PName,file=log)
                GDkMS=ReadCFGkMS.Parset(PName).DicoPars
                DoCheckParset=False
                
            
                Ls=[]
            
                def CheckField(D0,k0,D1,k1):
                    try:
                        v0=GDkMS[D0][k0]
                        v1=self.GD[D1][k1]
                        if v0=="": v0=None
                        if v1=="": v1=None
                        if v0!=v1:
                            Ls.append("!!! kMS parameter [[%s][%s] = %s] differs from DDF [[%s][%s] = %s]"%(D0,k0,str(v0),D1,k1,str(v1)))
                    except:
                        pass
                
            
                CheckField("Beam",'BeamModel',"Beam","Model")
                CheckField("Beam",'NChanBeamPerMS',"Beam","NBand")
                CheckField("Beam",'BeamAt', "Beam","At") # tessel/facet
                CheckField("Beam",'LOFARBeamMode', "Beam","LOFARBeamMode")     # A/AE
                CheckField("Beam",'DtBeamMin', "Beam","DtBeamMin")
                CheckField("Beam",'CenterNorm', "Beam","CenterNorm")
                CheckField("Beam",'FITSFile', "Beam","FITSFile")
                CheckField("Beam",'FITSParAngleIncDeg', "Beam","FITSParAngleIncDeg")
                CheckField("Beam",'FITSLAxis', "Beam","FITSLAxis")
                CheckField("Beam",'FITSMAxis', "Beam","FITSMAxis")
                CheckField("Beam",'FITSFeed', "Beam","FITSFeed") 
                # CheckField("Beam",'FITSVerbosity', "Beam","FITSVerbosity")
                CheckField("Beam","FeedAngle", "Beam","FeedAngle")
                CheckField("Beam",'ApplyPJones', "Beam","ApplyPJones")
                CheckField("Beam",'FlipVisibilityHands', "Beam","FlipVisibilityHands")
                CheckField("Beam",'FITSFeedSwap',"Beam","FITSFeedSwap")
    
                CheckField("ImageSkyModel",'MaxFacetSize',"Facets","DiamMax")
                CheckField("ImageSkyModel",'MinFacetSize',"Facets","DiamMin")
                CheckField("SkyModel",'Decorrelation',"RIME","DecorrMode")
    
                if len(Ls)>0:
                    log.print(ModColor.Str("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
                    log.print(ModColor.Str("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
                    log.print(ModColor.Str("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
                    log.print(ModColor.Str("!!! The following parameters are different in kMS/DDF, and you may think whether this has an effect or not..."))
                    for l in Ls:
                        log.print(ModColor.Str(l))
                    log.print(ModColor.Str("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
                    log.print(ModColor.Str("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
                    log.print(ModColor.Str("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"))
                else:
                    log.print(ModColor.Str("All kMS/DDF beam parameters are the same...",col="green"))
    
        if (self.GD["Facets"]["CatNodes"] is not None) and (SolsFile is not None):
            log.print(ModColor.Str("Both --Facets-CatNodes and --DDESolutions-DDSols are specified which might have different clusterings..."))

            
                
#        if "CatNodes" in self.GD.keys():
        regular_grid = False
        if self.GD["Facets"]["CatNodes"] is not None:
            print("Taking facet directions from Nodes catalog: %s" % self.GD["Facets"]["CatNodes"], file=log)
            ClusterNodes = np.load(self.GD["Facets"]["CatNodes"])
            ClusterNodes = ClusterNodes.view(np.recarray)
            raNode = ClusterNodes.ra
            decNode = ClusterNodes.dec
            lFacet, mFacet = self.CoordMachine.radec2lm(raNode, decNode)
        elif SolsFile is not None and ".npz" in SolsFile:
            print("Taking facet directions from solutions file: %s" % SolsFile, file=log)
            ClusterNodes = np.load(SolsFile)["ClusterCat"]
            ClusterNodes = ClusterNodes.view(np.recarray)
            raNode = ClusterNodes.ra
            decNode = ClusterNodes.dec
            lFacet, mFacet = self.CoordMachine.radec2lm(raNode, decNode)
        elif SolsFile is not None and ".h5" in  SolsFile:

            h5files, apply_solsets, apply_map = _parse_solsfile(SolsFile)
            print("Parsing h5file pattern {}".format(h5files), file=log)
            import glob
            h5file = glob.glob(h5files)[0]
            print( "Taking facet directions from H5parm: {}, solsets: {}".format(h5file, apply_solsets), file=log)
            with tables.open_file(h5file) as H:
                lm, radec = [], []
                for solset in apply_solsets:
                    _solset = getattr(H.root, solset)
                    raNode, decNode = _solset.source[:]["dir"].T
                    lFacet, mFacet = self.CoordMachine.radec2lm(raNode, decNode)
                    radec.append(np.stack([raNode, decNode], axis=1))
                    lm.append(np.stack([lFacet, mFacet], axis=1))
            # Nd+Nd+...,2
            lm = np.concatenate(lm, axis=0)
            radec = np.concatenate(radec, axis=0)
            lFacet, mFacet = lm[:, 0], lm[:, 1]
            raNode, decNode = radec[:, 0], radec[:, 1]
        else:
            print("Taking facet directions from regular grid", file=log)
            regular_grid = True
            CellSizeRad = (self.GD["Image"][
                           "Cell"] / 3600.) * np.pi / 180
            lrad = Npix * CellSizeRad * 0.5

            NpixFacet = Npix // NFacets
            lfacet = NpixFacet * CellSizeRad * 0.5
            lcenter_max = lrad - lfacet

            lFacet, mFacet, = np.mgrid[
                -lcenter_max: lcenter_max: (NFacets) * 1j, -
                lcenter_max: lcenter_max: (NFacets) * 1j]
            lFacet = lFacet.flatten()
            mFacet = mFacet.flatten()
        print("  There are %i Jones-directions" % lFacet.size, file=log)


        self.lmSols = lFacet.copy(), mFacet.copy()

        raSols, decSols = self.CoordMachine.lm2radec(
            lFacet.copy(), mFacet.copy())
        self.radecSols = raSols, decSols

        NodesCat = np.zeros(
            (raSols.size,),
            dtype=[('ra', float),
                   ('dec', float),
                   ('l', float),
                   ('m', float)])
        NodesCat = NodesCat.view(np.recarray)
        NodesCat.ra = raSols
        NodesCat.dec = decSols
        # print>>log,"Facet RA %s"%raSols
        # print>>log,"Facet Dec %s"%decSols
        NodesCat.l = lFacet
        NodesCat.m = mFacet

        ## saving below
        # NodeFile = "%s.NodesCat.%snpy" % (self.GD["Output"]["Name"], "psf." if self.DoPSF else "")
        # print>> log, "Saving Nodes catalog in %s" % NodeFile
        # np.save(NodeFile, NodesCat)

        self.DicoImager = {}

        xy = np.zeros((lFacet.size, 2), np.float32)
        xy[:, 0] = lFacet
        xy[:, 1] = mFacet

        regFile = "%s.tessel0.reg" % self.ImageName
        NFacets = self.NFacets = lFacet.size
        rac, decc = self.MainRaDec
        VM = ModVoronoiToReg.VoronoiToReg(rac, decc)

        if NFacets > 2:

            vor = Voronoi(xy, furthest_site=False)
            regions, vertices = ModVoronoi.voronoi_finite_polygons_2d(
                vor, radius=1.)

            PP = Polygon.Polygon(self.CornersImageTot)

            LPolygon=[]
            ListNode=[]
            for region,iNode in zip(regions,range(NodesCat.shape[0])):
                PP1=Polygon.Polygon(np.array(vertices[region]))
                ThisP = np.array(PP & PP1)
                # x,y=np.array(PP1).T
                # xp,yp=np.array(PP).T
                # stop
                # import pylab
                # pylab.clf()
                # #pylab.plot(x,y)
                # pylab.plot(xp,yp)
                # pylab.draw()
                # pylab.show()
                # #pylab.pause(0.1)

                if ThisP.size>0:
                    LPolygon.append(ThisP[0])
                    ListNode.append(iNode)
            NodesCat=NodesCat[np.array(ListNode)].copy()



# =======
#             LPolygon = [
#                 np.array(PP & Polygon.Polygon(np.array(vertices[region])))[0]
#                 for region in regions]
# >>>>>>> issue-255

        elif NFacets == 1:
            l0, m0 = lFacet[0], mFacet[0]
            LPolygon = [self.CornersImageTot]
        # VM.ToReg(regFile,lFacet,mFacet,radius=.1)

        NodeFile = "%s.NodesCat.npy" % self.GD["Output"]["Name"]
        print("Saving Nodes catalog in %s (Nfacets:%i)" % (NodeFile, NFacets), file=log)
        np.save(NodeFile, NodesCat)

        
        for iFacet, polygon0 in zip(range(len(LPolygon)), LPolygon):
            # polygon0 = vertices[region]
            P = polygon0.tolist()

        # VM.PolygonToReg(regFile,LPolygon,radius=0.1,Col="red")

        # stop

        ###########################################
        # SubDivide
        def GiveDiam(polygon):
            lPoly, mPoly = polygon.T
            l0 = np.max([lMainCenter - RadiusTot, lPoly.min()])
            l1 = np.min([lMainCenter + RadiusTot, lPoly.max()])
            m0 = np.max([mMainCenter - RadiusTot, mPoly.min()])
            m1 = np.min([mMainCenter + RadiusTot, mPoly.max()])
            dl = l1 - l0
            dm = m1 - m0
            diam = np.max([dl, dm])
            return diam, (l0, l1, m0, m1)

        DiamMax = self.GD["Facets"]["DiamMax"] * np.pi / 180
        # DiamMax=4.5*np.pi/180
        DiamMin = self.GD["Facets"]["DiamMin"] * np.pi / 180
        
        def ClosePolygon(polygon):
            P = polygon.tolist()
            polygon = np.array(P + [P[0]])
            return polygon

        def GiveSubDivideRegions(polygonFacet, DMax):

            polygonFOV = self.CornersImageTot
            # polygonFOV=ClosePolygon(polygonFOV)
            PFOV = Polygon.Polygon(polygonFOV)

            # polygonFacet=ClosePolygon(polygonFacet)
            P0 = Polygon.Polygon(polygonFacet)
            P0Cut = Polygon.Polygon(P0 & PFOV)
            
            if P0Cut.nPoints() == 0:
                return []

            polygonFacetCut = np.array(P0Cut[0])
            # polygonFacetCut=ClosePolygon(polygonFacetCut)

            diam, (l0, l1, m0, m1) = GiveDiam(polygonFacetCut)
            if diam < DMax:
                return [polygonFacetCut]

            Nl = int((l1 - l0) / DMax) + 1
            Nm = int((m1 - m0) / DMax) + 1
            dl = (l1 - l0) / Nl
            dm = (m1 - m0) / Nm
            lEdge = np.linspace(l0, l1, Nl + 1)
            mEdge = np.linspace(m0, m1, Nm + 1)
            lc = (lEdge[0:-1] + lEdge[1::]) / 2
            mc = (mEdge[0:-1] + mEdge[1::]) / 2
            LPoly = []
            Lc, Mc = np.meshgrid(lc, mc)
            Lc = Lc.ravel().tolist()
            Mc = Mc.ravel().tolist()

            DpolySquare = np.array([[-dl, -dm],
                                    [dl, -dm],
                                    [dl, dm],
                                    [-dl, dm]]) * 0.5
            
            for lc, mc in zip(Lc, Mc):
                polySquare = DpolySquare.copy()  # ClosePolygon(DpolySquare.copy())
                polySquare[:, 0] += lc
                polySquare[:, 1] += mc
                # polySquare=ClosePolygon(polySquare)
                P1 = Polygon.Polygon(polySquare)
                
                POut = (P0Cut & P1)
                if POut.nPoints() == 0:
                    continue

                DoPlot=0
                if len(POut)>1:
                    DoPlot=1
                    log.print(ModColor.Str("WARNING: There are more than one polygon in the intersection"))
                    sFile="Poly%i.npz"%self.iSave
                    log.print(ModColor.Str("   Saving polygon file for eventual debugging as: %s"%sFile))
                    np.savez(sFile,
                             polySquare=polySquare,
                             polygonFacetCut=polygonFacetCut)
                    log.print(ModColor.Str("   Taking the largest polygon..."))
                    A=0
                    for iP in range(len(POut)):
                        Aa=Polygon.Polygon(POut[iP]).area()
                        if Aa>A:
                            A=Aa
                            polyOut = np.array(POut[iP])
                else:
                    polyOut = np.array(POut[0])
                    
                # polyOut=ClosePolygon(polyOut)
                LPoly.append(polyOut)

                # ################################
                # def CloseLoop(x,y):
                #     x=np.concatenate([x,np.array([x[0]])])
                #     y=np.concatenate([y,np.array([y[0]])])
                #     return x,y
                # if False:#DoPlot:
                #     # np.savez("Poly%i.npz"%self.iSave,
                #     #          polySquare=polySquare,
                #     #          polygonFacetCut=polygonFacetCut)
                #     import pylab
                #     # pylab.clf()
                #     x,y=polygonFacetCut.T
                #     x,y=CloseLoop(x,y)
                #     #pylab.plot(x,y,color="blue")
                #     x,y=polygonFacet.T
                #     x,y=CloseLoop(x,y)
                #     #pylab.plot(x,y,color="blue",ls=":",lw=3)
                #     x,y=np.array(PFOV[0]).T
                #     x,y=CloseLoop(x,y)
                #     pylab.plot(x,y,color="black")
                #     x,y=polySquare.T
                #     x,y=CloseLoop(x,y)
                #     pylab.plot(x,y,color="green",ls=":",lw=3)
                #     x,y=polyOut.T
                #     x,y=CloseLoop(x,y)
                #     pylab.plot(x,y,color="red",ls="--",lw=3)
                #     pylab.scatter(x,y,c="red")
                #     pylab.xlim(self.RadiusTot,-self.RadiusTot)
                #     pylab.ylim(-self.RadiusTot,self.RadiusTot)
                #     pylab.title("iSave=%i"%self.iSave)
                #     pylab.draw()
                #     pylab.show(block=True)
                #     #pylab.pause(0.5)
                # #############################
                
                self.iSave+=1

            return LPoly

        def PlotPolygon(P, *args, **kwargs):
            for poly in P:
                x, y = ClosePolygon(np.array(poly)).T
                pylab.plot(x, y, *args, **kwargs)

        LPolygonNew = []

        self.iSave=0
        for iFacet in range(len(LPolygon)):
            polygon = LPolygon[iFacet]
            ThisDiamMax = DiamMax
            SubReg = GiveSubDivideRegions(polygon, ThisDiamMax)

            LPolygonNew += SubReg

        regFile = "%s.FacetMachine.tessel.ReCut.reg" % self.ImageName
        # VM.PolygonToReg(regFile,LPolygonNew,radius=0.1,Col="green",labels=[str(i) for i in range(len(LPolygonNew))])

        DicoPolygon = {}
        # import pylab
        # Lx,Ly=[],[]
        for iFacet in range(len(LPolygonNew)):
            DicoPolygon[iFacet] = {}
            poly = LPolygonNew[iFacet]
            DicoPolygon[iFacet]["poly"] = poly
            diam, (l0, l1, m0, m1) = GiveDiam(poly)
            DicoPolygon[iFacet]["diam"] = diam
            DicoPolygon[iFacet]["diamMin"] = np.min([(l1 - l0), (m1 - m0)])
            xc, yc = np.mean(poly[:, 0]), np.mean(poly[:, 1])
            DicoPolygon[iFacet]["xyc"] = xc, yc
            dSol = np.sqrt((xc - lFacet) ** 2 + (yc - mFacet) ** 2)
            ind=np.where(dSol == np.min(dSol))[0][0:1]
            DicoPolygon[iFacet]["iSol"] = ind
            # if ind.size==2: stop
            # Lx.append(xc)
            # Ly.append(yc)
            # print(iFacet,DicoPolygon[iFacet]["iSol"])
            

        # for iFacet in list(DicoPolygon.keys()):
        #     pylab.clf()
        #     pylab.scatter(lFacet,mFacet,c="red")
        #     pylab.scatter(Lx,Ly,c="blue")
        #     ind=DicoPolygon[iFacet]["iSol"]
        #     pylab.scatter(lFacet[ind],mFacet[ind],c="green")
        #     pylab.scatter(Lx[iFacet],Ly[iFacet],c="black")
        #     pylab.draw()
        #     pylab.show(block=False)
        #     pylab.pause(0.1)
        #     if ind.size==2: stop


            
        for iFacet in sorted(DicoPolygon.keys()):
            diam = DicoPolygon[iFacet]["diamMin"]
            #print(iFacet,diam,DiamMin)
            if diam < DiamMin:
                dmin = 1e6
                xc0, yc0 = DicoPolygon[iFacet]["xyc"]
                HasClosest = False
                for iFacetOther in sorted(DicoPolygon.keys()):
                    if iFacetOther == iFacet:
                        continue
                    iSolOther = DicoPolygon[iFacetOther]["iSol"]
                    # print "  ",iSolOther,DicoPolygon[iFacet]["iSol"]
                    if iSolOther != DicoPolygon[iFacet]["iSol"]:
                        continue
                    xc, yc = DicoPolygon[iFacetOther]["xyc"]
                    d = np.sqrt((xc - xc0) ** 2 + (yc - yc0) ** 2)
                    if d < dmin:
                        dmin = d
                        iFacetClosest = iFacetOther
                        HasClosest = True
                if (HasClosest):
                    log.print("Merging facet #%i to #%i" % (
                        iFacet, iFacetClosest))
                    P0 = Polygon.Polygon(DicoPolygon[iFacet]["poly"])
                    P1 = Polygon.Polygon(DicoPolygon[iFacetClosest]["poly"])
                    P2 = (P0 | P1)
                    POut = []
                    for iP in range(len(P2)):
                        POut += P2[iP]

                    poly = np.array(POut)
                    hull = ConvexHull(poly)
                    Contour = np.array(
                        [hull.points[hull.vertices, 0],
                         hull.points[hull.vertices, 1]])
                    poly2 = Contour.T

                    del (DicoPolygon[iFacet])
                    DicoPolygon[iFacetClosest]["poly"] = poly2
                    DicoPolygon[iFacetClosest]["diam"] = GiveDiam(poly2)[0]
                    DicoPolygon[iFacetClosest]["xyc"] = np.mean(
                        poly2[:, 0]), np.mean(
                        poly2[:, 1])

        # stop
        LPolygonNew = []
        for iFacet in sorted(DicoPolygon.keys()):
            # if DicoPolygon[iFacet]["diam"]<DiamMin:
            #     print>>log, ModColor.Str("  Facet #%i associated to direction #%i is too small, removing it"%(iFacet,DicoPolygon[iFacet]["iSol"]))
            #     continue
            LPolygonNew.append(DicoPolygon[iFacet]["poly"])

        # for iFacet in range(len(regions)):
        #     polygon=LPolygon[iFacet]
        #     ThisDiamMax=DiamMax
        #     while True:
        #         SubReg=GiveSubDivideRegions(polygon,ThisDiamMax)
        #         if SubReg==[]:
        #             break
        #         Diams=[GiveDiam(poly)[0] for poly in SubReg]

        #         if np.min(Diams)>DiamMin: break
        #         ThisDiamMax*=1.1
        #     LPolygonNew+=SubReg
        #     print

        regFile = "%s.tessel.%sreg" % (self.GD["Output"]["Name"], "psf." if self.DoPSF else "")
        # labels=["[F%i.C%i]"%(i,DicoPolygon[i]["iSol"]) for i in range(len(LPolygonNew))]
        # VM.PolygonToReg(regFile,LPolygonNew,radius=0.1,Col="green",labels=labels)

        # VM.PolygonToReg(regFile,LPolygonNew,radius=0.1,Col="green")

        # pylab.clf()
        # x,y=LPolygonNew[11].T
        # pylab.plot(x,y)
        # pylab.draw()
        # pylab.show()
        # stop
        ###########################################

        NFacets = len(LPolygonNew)

        NJonesDir=NodesCat.shape[0]
        self.JonesDirCat = np.zeros(
            (NodesCat.shape[0],),
            dtype=[('Name', '|S200'),
                   ('ra', float),
                   ('dec', float),
                   ('SumI', float),
                   ("Cluster", int),
                   ("l", float),
                   ("m", float),
                   ("I", float)])
        self.JonesDirCat = self.JonesDirCat.view(np.recarray)
        self.JonesDirCat.I = 1
        self.JonesDirCat.SumI = 1

        self.JonesDirCat.ra=NodesCat.ra
        self.JonesDirCat.dec=NodesCat.dec
        self.JonesDirCat.l=NodesCat.l
        self.JonesDirCat.m=NodesCat.m
        self.JonesDirCat.Cluster = range(NJonesDir)

        print("Sizes (%i facets):" % (self.JonesDirCat.shape[0]), file=log)
        print("   - Main field :   [%i x %i] pix" % (self.Npix, self.Npix), file=log)


        l_m_Diam = np.zeros((NFacets, 4), np.float32)
        l_m_Diam[:, 3] = np.arange(NFacets)

        Np = 10000
        D = {}
        for iFacet in range(NFacets):
            D[iFacet] = {}
            polygon = LPolygonNew[iFacet]
            D[iFacet]["Polygon"] = polygon
            lPoly, mPoly = polygon.T

            ThisDiam, (l0, l1, m0, m1) = GiveDiam(polygon)

            # ###############################
            # # Find barycenter of polygon
            # X=(np.random.rand(Np))*ThisDiam+l0
            # Y=(np.random.rand(Np))*ThisDiam+m0
            # XY = np.dstack((X, Y))
            # XY_flat = XY.reshape((-1, 2))
            # mpath = Path( polygon )
            # XY = np.dstack((X, Y))
            # XY_flat = XY.reshape((-1, 2))
            # mask_flat = mpath.contains_points(XY_flat)
            # mask=mask_flat.reshape(X.shape)
            # ###############################
            ThisPolygon = Polygon.Polygon(polygon)
            lc, mc = ThisPolygon.center()
            dl = np.max(np.abs([l0 - lc, l1 - lc]))
            dm = np.max(np.abs([m0 - mc, m1 - mc]))
            ###############################
            # lc=np.sum(X*mask)/np.sum(mask)
            # mc=np.sum(Y*mask)/np.sum(mask)
            # dl=np.max(np.abs(X[mask==1]-lc))
            # dm=np.max(np.abs(Y[mask==1]-mc))
            diam = 2 * np.max([dl, dm])

            ######################
            # lc=(l0+l1)/2.
            # mc=(m0+m1)/2.
            # dl=l1-l0
            # dm=m1-m0
            # diam=np.max([dl,dm])

            l_m_Diam[iFacet, 0] = lc
            l_m_Diam[iFacet, 1] = mc
            l_m_Diam[iFacet, 2] = diam

        self.SpacialWeigth = {}
        self.DicoImager = {}

        # sort facets by size, unless we're in regular grid mode
        if not regular_grid:
            indDiam = np.argsort(l_m_Diam[:, 2])[::-1]
            l_m_Diam = l_m_Diam[indDiam]

        for iFacet in range(l_m_Diam.shape[0]):
            self.DicoImager[iFacet] = {}
            self.DicoImager[iFacet]["Polygon"] = D[l_m_Diam[iFacet, 3]]["Polygon"]
            

        if self.GD["Facets"].get("FluxPaddingAppModel",None) is not None:
            NameModel=self.GD["Facets"]["FluxPaddingAppModel"]
            log.print("Computing individual facet flux density for facet-dependent padding...")
            ModelImage=image(NameModel).getdata()
            nch,npol,_,_=ModelImage.shape
            ModelImage=np.mean(ModelImage[:,0,:,:],axis=0)
            ModelImage=(ModelImage.T[::-1]).copy()
            _,_,nx,ny=self.OutImShape
            Dx=self.CellSizeRad * nx/2
            Dy=self.CellSizeRad * ny/2

            from DDFacet.Other import ClassTimeIt
            T=ClassTimeIt.ClassTimeIt("FluxPaddingAppModel")
            T.disable()
            lg, mg = X, Y = np.mgrid[-Dx:Dx:nx * 1j, -Dy:Dy:ny * 1j]
            for iFacet in self.DicoImager.keys():
                vertices = self.DicoImager[iFacet]["Polygon"]
                lp,mp=vertices.T

                # indx=((lg>=lp.min())&(lg<lp.max()))
                # indy=((mg>=mp.min())&(mg<mp.max()))
                # ind=(indx&indy)
                
                lg1, mg1 = np.mgrid[-Dx:Dx:nx * 1j], np.mgrid[-Dy:Dy:ny * 1j]
                indx=((lg1>=lp.min())&(lg1<lp.max()))
                indy=((mg1>=mp.min())&(mg1<mp.max()))
                ind1=(indx.reshape((-1,1))*indy.reshape((1,-1)))
                # print(np.allclose(ind,ind1))
                # stop
                ind=ind1
                
                ModelImage_s=ModelImage[ind]
                X,Y=lgs,mgs=lg[ind],mg[ind]
                
                XY = np.dstack((X, Y))
                XY_flat = XY.reshape((-1, 2))
                T.timeit("build s")
                
                mpath = Path(vertices)  # the vertices of the polygon
                mask_flat = mpath.contains_points(XY_flat)
                T.timeit("contains")
                mask = mask_flat.reshape(X.shape)
                Ft=np.max(ModelImage_s.flat[mask.ravel()])
                # log.print("Flux Facet [%.3i] = %f"%(iFacet,Ft))
                self.DicoImager[iFacet]["MaxFlux"]=Ft
                Ft=np.sum(ModelImage_s.flat[mask.ravel()])
                self.DicoImager[iFacet]["TotalFlux"]=Ft
                T.timeit("rest")

            # lg, mg = X, Y = np.mgrid[-Dx:Dx:nx * 1j, -Dy:Dy:ny * 1j]
            # XY = np.dstack((X, Y))
            # XY_flat = XY.reshape((-1, 2))
            # for iFacet in self.DicoImager.keys():
            #     vertices = self.DicoImager[iFacet]["Polygon"]
            #     mpath = Path(vertices)  # the vertices of the polygon
            #     mask_flat = mpath.contains_points(XY_flat)
            #     T.timeit("contains")
            #     mask = mask_flat.reshape(X.shape)
            #     Ft=np.max(ModelImage.flat[mask.ravel()])
            #     log.print("Flux Facet [%.3i] = %f"%(iFacet,Ft))
            #     self.DicoImager[iFacet]["MaxFlux"]=Ft
            #     Ft=np.sum(ModelImage.flat[mask.ravel()])
            #     self.DicoImager[iFacet]["TotalFlux"]=Ft
            #     T.timeit("rest")

        ###############
                
        for iFacet in range(l_m_Diam.shape[0]):
            x0 = round(l_m_Diam[iFacet, 0] / self.CellSizeRad)
            y0 = round(l_m_Diam[iFacet, 1] / self.CellSizeRad)
            # if x0 % 2 == 0:
            #     x0 += 1
            # if y0 % 2 == 0:
            #     y0 += 1
            l0 = x0 * self.CellSizeRad
            m0 = y0 * self.CellSizeRad
            diam = round(
                l_m_Diam[
                    iFacet,
                    2] / self.CellSizeRad) * self.CellSizeRad
            # self.AppendFacet(iFacet,l0,m0,diam)
            self.AppendFacet(iFacet, l0, m0, diam)



            

        # self.MakeMasksTessel()

        NpixMax = np.max([self.DicoImager[iFacet]["NpixFacet"]
                          for iFacet in sorted(self.DicoImager.keys())])
        NpixMaxPadded = np.max(
            [self.DicoImager[iFacet]["NpixFacetPadded"]
             for iFacet in sorted(self.DicoImager.keys())])
        self.PaddedGridShape = (1, 1, NpixMaxPadded, NpixMaxPadded)
        self.FacetShape = (1, 1, NpixMax, NpixMax)

        dmin = 1
        for iFacet in range(len(self.DicoImager)):
            l, m = self.DicoImager[iFacet]["l0m0"]
            d = np.sqrt(l ** 2 + m ** 2)
            if d < dmin:
                dmin = d
                iCentralFacet = iFacet
        self.iCentralFacet = iCentralFacet
        self.NFacets = len(self.DicoImager)
        # regFile="%s.tessel.reg"%self.GD["Output"]["Name"]
        labels = [
            (self.DicoImager[i]["lmShift"][0],
             self.DicoImager[i]["lmShift"][1],
             "[F%i_S%i]" % (i, self.DicoImager[i]["iSol"]))
            for i in range(len(LPolygonNew))]
        VM.PolygonToReg(
            regFile,
            LPolygonNew,
            radius=0.1,
            Col="green",
            labels=labels)

        self.WriteCoordFacetFile()

        self.FacetDirections=set([self.DicoImager[iFacet]["RaDec"] for iFacet in range(len(self.DicoImager))])
        #DicoName = "%s.DicoFacet" % self.GD["Images"]["ImageName"]
        DicoName = "%s.%sDicoFacet" % (self.GD["Output"]["Name"], "psf." if self.DoPSF else "")


        # Find the minimum l,m in the facet (for decorrelation calculation)
        for iFacet in self.DicoImager.keys():
            #Create smoothned facet tessel mask:
            Npix = self.DicoImager[iFacet]["NpixFacetPadded"]
            l0, l1, m0, m1 = self.DicoImager[iFacet]["lmExtentPadded"]
            X, Y = np.mgrid[l0:l1:Npix//10 * 1j, m0:m1:Npix//10 * 1j]
            XY = np.dstack((X, Y))
            XY_flat = XY.reshape((-1, 2))
            vertices = self.DicoImager[iFacet]["Polygon"]
            mpath = Path(vertices)  # the vertices of the polygon
            mask_flat = mpath.contains_points(XY_flat)
            mask = mask_flat.reshape(X.shape)
            mpath = Path(self.CornersImageTot)
            mask_flat2 = mpath.contains_points(XY_flat)
            mask2 = mask_flat2.reshape(X.shape)
            mask[mask2 == 0] = 0
            R=np.sqrt(X**2+Y**2)
            R[mask==0]=1e6
            indx,indy=np.where(R==np.min(R))
            lmin,mmin=X[indx[0],indy[0]],Y[indx[0],indy[0]]
            self.DicoImager[iFacet]["lm_min"]=lmin,mmin


            
        self.FacetDirCat = np.zeros((len(self.DicoImager),),
                                    dtype=[('Name', '|S200'),
                                           ('ra', float),
                                           ('dec', float),
                                           ('SumI', float),
                                           ("Cluster", int),
                                           ("l", float),
                                           ("m", float),
                                           ("I", float)])
        self.FacetDirCat = self.FacetDirCat.view(np.recarray)
        self.FacetDirCat.I = 1
        self.FacetDirCat.SumI = 1
        for iFacet in self.DicoImager.keys():
            l,m=self.DicoImager[iFacet]["lmShift"]
            ra,dec=self.DicoImager[iFacet]["RaDec"]
            self.FacetDirCat.ra[iFacet]=ra
            self.FacetDirCat.dec[iFacet]=dec
            self.FacetDirCat.l[iFacet]=l
            self.FacetDirCat.m[iFacet]=m
            self.FacetDirCat.Cluster[iFacet] = iFacet

        print("Saving DicoImager in %s" % DicoName, file=log)
        MyPickle.Save(self.DicoImager, DicoName)

    def WriteCoordFacetFile(self):
        FacetCoordFile = "%s.facetCoord.%stxt" % (self.GD["Output"]["Name"], "psf." if self.DoPSF else "")
        print("Writing facet coordinates in %s" % FacetCoordFile, file=log)
        f = open(FacetCoordFile, 'w')
        ss = "# (Name, Type, Ra, Dec, I, Q, U, V, ReferenceFrequency='7.38000e+07', SpectralIndex='[]', MajorAxis, MinorAxis, Orientation) = format"
        for iFacet in range(len(self.DicoImager)):
            ra, dec = self.DicoImager[iFacet]["RaDec"]
            sra = rad2hmsdms.rad2hmsdms(ra, Type="ra").replace(" ", ":")
            sdec = rad2hmsdms.rad2hmsdms(dec).replace(" ", ".")
            ss = "%s, %s, %f, %f" % (sra, sdec,ra,dec)
            f.write(ss+'\n')
        f.close()
