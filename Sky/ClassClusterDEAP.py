#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import random

import numpy
import numpy as np

from deap import algorithms
import DeapAlgo as algorithms
from deap import base
from deap import creator
from deap import tools
import pylab
from scipy.spatial import Voronoi
import ModVoronoi
from DDFacet.Other import MyLogger
log=MyLogger.getLogger("ClusterDEAP")
from DDFacet.Other import ClassTimeIt
#from scoop import futures
import multiprocessing
import scipy.stats
import Polygon
import ClassMetricDEAP

def test():
    Np=1000
    x=np.random.randn(Np)
    y=np.random.randn(Np)
    
    CC=ClassCluster(x,y)
    CC.Cluster()
    
def evalOneMax(individual):
    return sum(individual),



def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2


    
def Mutate(Indiv,indpb=0.05,AmpRad=0.017453292519943295):
    N=Indiv.size/2
    Ind=Indiv.reshape((2,N))
    #i=int(np.random.rand(1)[0]*N)
    for i in range(N):
        r=int(np.random.rand(1)[0]*N)
        if r>indpb: continue
        ra,dec=Ind
        ra[i]+=np.random.randn(1)[0]*AmpRad#*0.03
        dec[i]+=np.random.randn(1)[0]*AmpRad#*0.03
    return Indiv,


def giveFitness(Indiv,x=None,y=None,S=None,Polygons=None,PolyCut=None,BigPolygon=None): 
    T=ClassTimeIt.ClassTimeIt("Fitness")
    T.disable()
    CMD=ClassMetricDEAP.ClassMetricDEAP(Indiv,x=x,y=y,S=S,Polygons=Polygons,PolyCut=PolyCut,BigPolygon=BigPolygon)
    fluxPerFacet=CMD.fluxPerFacet()
    NPerFacet=CMD.NPerFacet()
    aspectRatioPerFacet=CMD.aspectRatioPerFacet()
    meanDistancePerFacet=CMD.meanDistancePerFacet()
    overlapPerFacet=CMD.overlapPerFacet()

    Fitness=0
    Fitness+= -np.std(fluxPerFacet)
    Fitness+= -np.std(NPerFacet)
    Fitness+= -1e5*np.count_nonzero(NPerFacet==0)
    A=aspectRatioPerFacet
    Fitness+= -np.mean(A[A>0])
    Fitness+= -np.mean(meanDistancePerFacet)*10
    Fitness+= -np.sum(overlapPerFacet)*1e5
    
    return Fitness,


class ClassCluster():
    def __init__(self,x,y,S,nNode=50,RandAmpDeg=1.,NGen=300,NPop=1000,DoPlot=True,PolyCut=None,
                 NCPU=1,BigPolygon=None):
        self.DoPlot=DoPlot
        self.PolyCut=PolyCut
        self.BigPolygon=BigPolygon
        self.x=x
        self.y=y
        self.S=S
        self.NGen=NGen
        self.NPop=NPop
        self.nNode=nNode
        self.RandAmpRad=RandAmpDeg*np.pi/180
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()

        if NCPU>1:
            pool = multiprocessing.Pool(NCPU)
            toolbox.register("map", pool.map)
        

        #x0=np.min([self.x.min(),self.y.min()])
        #x1=np.max([self.x.max(),self.y.max()])
        
        toolbox.register("attr_float", random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2*nNode)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # toolbox.register("individual",
        #                  tools.initCycle,
        #                  creator.Individual,
        #                  Obj, n=1)
        # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        

        toolbox.register("mate", cxTwoPointCopy)
        #toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("mutate", Mutate, indpb=0.05, AmpRad=self.RandAmpRad)
        toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox=toolbox

        self.Polygons=None#[np.array([[0,0],[0,1],[1,1.]])*0.5]

    def setAvoidPolygon(self,PolyList):
        self.Polygons=PolyList
        

    def reinitPop(self,pop):
        print>>log,"Initialise population"
        x0,x1=self.x.min(),self.x.max()
        y0,y1=self.y.min(),self.y.max()
        for Indiv in pop:
            x,y=Indiv.reshape((2,self.nNode))
            x[:]=np.random.uniform(x0,x1,self.nNode)
            y[:]=np.random.uniform(y0,y1,self.nNode)
            #x.fill(0)
            #y.fill(0)
            
    def Cluster(self):
        random.seed(64)
        toolbox=self.toolbox

        toolbox.register("evaluate", giveFitness, x=self.x, y=self.y, S=self.S, Polygons=self.Polygons,
                         PolyCut=self.PolyCut, BigPolygon=self.BigPolygon)

        pop = toolbox.population(n=self.NPop)
        self.reinitPop(pop)

        # Numpy equality function (operators.eq) between two arrays returns the
        # equality element wise, which raises an exception in the if similar()
        # check of the hall of fame. Using a different equality function like
        # numpy.array_equal or numpy.allclose solve this issue.
        hof = tools.HallOfFame(1, similar=numpy.array_equal)
        #print>>log,"Declare HOF"
        
        # stats = tools.Statistics(lambda ind: ind.fitness.values)
        # stats.register("avg", numpy.mean)
        # stats.register("std", numpy.std)
        # stats.register("min", numpy.min)
        # stats.register("max", numpy.max)

        print>>log,"Clustering input catalog in %i directions"%(self.nNode)
        print>>log,"  Start evolution of %i generations of %i individuals"%(self.NGen,self.NPop)
        PlotMachine=False
        if self.DoPlot:
            PlotMachine=ClassPlotMachine(self.x,self.y,self.S,self.Polygons,self.PolyCut)
        
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=self.NGen,
                            #stats=stats,
                            halloffame=hof,PlotMachine=PlotMachine)

        
        CMD=ClassMetricDEAP.ClassMetricDEAP(hof[-1],
                                            x=self.x,
                                            y=self.y,
                                            S=self.S,
                                            Polygons=self.Polygons,
                                            PolyCut=self.PolyCut)
        LPolygon=CMD.ListPolygons
        return hof[-1],LPolygon

class ClassPlotMachine():
    def __init__(self,x,y,S,Polygons=None,PolyCut=None):
        self.x=x
        self.y=y
        self.S=S
        self.Polygons=Polygons
        self.PolyCut=PolyCut

    def Plot(self,hof):
        indiv=hof[-1]
        N=indiv.size/2
        xc,yc=indiv.reshape((2,N))

        CMD=ClassMetricDEAP.ClassMetricDEAP(indiv,x=self.x,y=self.y,S=self.S,Polygons=self.Polygons,PolyCut=self.PolyCut)
        ListPolygons=CMD.ListPolygons
        fluxPerFacet=CMD.fluxPerFacet()

        import matplotlib
        fig1=pylab.figure(1)
        pylab.clf()
        pylab.subplot(1,1,1, aspect='equal')
        for iR,Poly in enumerate(ListPolygons):
            polygon = Poly
            if Poly.size==0: continue
            x,y=polygon.T
            pylab.fill(*zip(*polygon), alpha=0.1)#, color=cms[iR])
            pylab.text(np.mean(x),np.mean(y),"%.2f"%fluxPerFacet[iR])
            

        if self.Polygons is not None:
            for Polygon in self.Polygons:
                pylab.fill(*zip(*Polygon), color="black")#alpha=0.4)

            #pylab.plot(xp,yp)
        pylab.scatter(self.x,self.y,s=5,color="blue")
        pylab.scatter(xc,yc,color="red")

        dx=0.01
        pylab.xlim(self.x.max()+dx,self.x.min() - dx)
        pylab.ylim(self.y.min() - dx, self.y.max()+dx)
        # mng = pylab.get_current_fig_manager()
        # #mng.frame.Maximize(True)
        # mng.window.showMaximized()
        pylab.pause(0.1)
        pylab.draw()
        pylab.show(False)
            

        ######################
        
        aspectRatioPerFacet=CMD.aspectRatioPerFacet()
        meanDistancePerFacet=CMD.meanDistancePerFacet()
        overlapPerFacet=CMD.overlapPerFacet()
        NPerFacet=CMD.NPerFacet()

        fig2=pylab.figure(2)
        pylab.clf()
        pylab.subplot(2,2,1)
        pylab.hist(fluxPerFacet,bins=100)

        pylab.subplot(2,2,2)
        pylab.hist(aspectRatioPerFacet,bins=100)

        pylab.subplot(2,2,3)
        pylab.hist(NPerFacet,bins=100)

        pylab.subplot(2,2,4)
        pylab.hist(meanDistancePerFacet,bins=100)



            
        pylab.pause(0.1)
        pylab.draw()
        pylab.show(False)
        
        # II=np.unique(ind)
        # NPerNode=np.zeros((xc.size,),np.float32)
        # for iC in II:
        #     NPerNode[iC]=np.count_nonzero(ind==iC)

            
