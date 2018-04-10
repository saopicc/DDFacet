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
import random

#from deap import base
from deap import base
from deap import creator
from deap import tools
import numpy
import algorithms
import numpy as np


import ClassArrayMethodGA

def FilterIslandsPix(ListIn,Npix):
    ListOut=[]
    for x,y in ListIn:
        Cx=((x>=0)&(x<Npix))
        Cy=((y>=0)&(y<Npix))
        if (Cx&Cy):
            ListOut.append([x,y])
    return ListOut


class ClassEvolveGA():
    def __init__(self,Dirty,PSF,FreqsInfo,ListPixData=None,ListPixParms=None,IslandBestIndiv=None,GD=None,WeightFreqBands=None):
        _,_,NPixPSF,_=PSF.shape
        if ListPixData is None:
            x,y=np.mgrid[0:NPixPSF:1,0:NPixPSF:1]
            ListPixData=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()
        if ListPixParms is None:
            x,y=np.mgrid[0:NPixPSF:1,0:NPixPSF:1]
            ListPixParms=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()
        self.IslandBestIndiv=IslandBestIndiv

        _,_,Npix,_=Dirty.shape
        ListPixData=FilterIslandsPix(ListPixData,Npix)
        ListPixParms=FilterIslandsPix(ListPixParms,Npix)

        self.ArrayMethodsMachine=ClassArrayMethodGA.ClassArrayMethodGA(Dirty,PSF,ListPixParms,ListPixData,FreqsInfo,IslandBestIndiv=IslandBestIndiv,GD=GD,WeightFreqBands=WeightFreqBands)
        
        self.InitEvolutionAlgo()
        #self.ArrayMethodsMachine.testMovePix()
        #stop

    def InitEvolutionAlgo(self):
        if "FitnessMax" not in dir(creator):
            creator.create("FitnessMax", base.Fitness, weights=self.ArrayMethodsMachine.WeightsEA)
        if "Individual" not in dir(creator):
            creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        #Obj=[toolbox.attr_float_unif]*self.ArrayMethodsMachine.NParms
        #Obj=[toolbox.attr_float_unif]*self.ArrayMethodsMachine.NParms
        #Obj=[toolbox.attr_float_normal]*self.ArrayMethodsMachine.NParms

        Obj=self.ArrayMethodsMachine.PM.GiveInitList(toolbox)

        toolbox.register("individual", tools.initCycle, creator.Individual,
                         Obj, n=1)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        
        
        toolbox.register("evaluate", self.ArrayMethodsMachine.GiveFitness)
        # toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        # toolbox.register("mate", tools.cxOrdered)
        # toolbox.register("mutate", tools.mutGaussian, indpb=0.3,  mu=0.0, sigma=.1)
        #toolbox.register("mutate", self.ArrayMethodsMachine.mutGaussian, pFlux=0.3, p0=0.3, pMove=0.3)
        toolbox.register("mutate", self.ArrayMethodsMachine.mutGaussian, pFlux=0.1, p0=0.5, pMove=0.1)

        toolbox.register("select", tools.selTournament, tournsize=3)
        #toolbox.register("select", tools.selRoulette)

        self.toolbox=toolbox



    def main(self,NGen=1000,NIndiv=100,DoPlot=True):
        #os.system("rm png/*.png")
        #random.seed(64)
        #np.random.seed(64)
        toolbox=self.toolbox
        # pool = multiprocessing.Pool(processes=6)
        # toolbox.register("map", pool.map)
        self.pop = toolbox.population(n=NIndiv)
        self.hof = tools.HallOfFame(1, similar=numpy.array_equal)
        #self.hof = tools.ParetoFront(1, similar=numpy.array_equal)

        # stats = tools.Statistics(lambda ind: ind.fitness.values)
        # stats.register("avg", numpy.mean)
        # stats.register("std", numpy.std)
        # stats.register("min", numpy.min)
        # stats.register("max", numpy.max)


        if self.IslandBestIndiv is not None:
            if np.max(np.abs(self.IslandBestIndiv))==0:
                #print "deconv"
                SModelArray=self.ArrayMethodsMachine.DeconvCLEAN()
                #AlphaModel=np.zeros_like(SModelArray)#-0.6
                #AlphaModel[SModelArray==np.max(SModelArray)]=0
                self.ArrayMethodsMachine.PM.ReinitPop(self.pop,SModelArray)#,AlphaModel=AlphaModel)
            else:
                SModelArray=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.IslandBestIndiv,"S")
                AlphaModel=None
                if "Alpha" in self.ArrayMethodsMachine.PM.SolveParam:
                    AlphaModel=self.ArrayMethodsMachine.PM.ArrayToSubArray(self.IslandBestIndiv,"Alpha")
                
                self.ArrayMethodsMachine.PM.ReinitPop(self.pop,SModelArray,AlphaModel=AlphaModel)


        self.pop, log= algorithms.eaSimple(self.pop, toolbox, cxpb=0.3, mutpb=0.1, ngen=NGen, 
                                           halloffame=self.hof, 
                                           #stats=stats,
                                           verbose=False, 
                                           ArrayMethodsMachine=self.ArrayMethodsMachine,DoPlot=DoPlot)

        
        # #:param mu: The number of individuals to select for the next generation.
        # #:param lambda\_: The number of children to produce at each generation.
        # #:param cxpb: The probability that an offspring is produced by crossover.
        # #:param mutpb: The probability that an offspring is produced by mutation.

        # mu=70
        # lambda_=50
        # cxpb=0.3
        # mutpb=0.5
        # ngen=1000

        # self.pop, log= algorithms.eaMuPlusLambda(self.pop, toolbox, mu, lambda_, cxpb, mutpb, ngen,
        #                               stats=None, halloffame=None, verbose=__debug__,
        #                               ArrayMethodsMachine=self.ArrayMethodsMachine)

        V = tools.selBest(self.pop, 1)[0]

        return V
