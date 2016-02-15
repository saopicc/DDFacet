
import random

#from deap import base
from deap import base
from deap import creator
from deap import tools
import numpy
import algorithms
import multiprocessing

from DDFacet.ToolsDir import ModFFTW
import numpy as np
import pylab
import os


import ClassArrayMethodGA


class ClassEvolveGA():
    def __init__(self,Dirty,PSF,FreqsInfo,ListPixData=None,ListPixParms=None,IslandBestIndiv=None,GD=None):
        _,_,NPixPSF,_=PSF.shape
        if ListPixData==None:
            x,y=np.mgrid[0:NPixPSF:1,0:NPixPSF:1]
            ListPixData=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()
        if ListPixParms==None:
            x,y=np.mgrid[0:NPixPSF:1,0:NPixPSF:1]
            ListPixParms=np.array([x.ravel().tolist(),y.ravel().tolist()]).T.tolist()
        self.ArrayMethodsMachine=ClassArrayMethodGA.ClassArrayMethodGA(Dirty,PSF,ListPixParms,ListPixData,FreqsInfo,IslandBestIndiv=IslandBestIndiv,GD=GD)
        self.InitEvolutionAlgo()
        #self.ArrayMethodsMachine.testMovePix()
        #stop

    def InitEvolutionAlgo(self):

        creator.create("FitnessMax", base.Fitness, weights=self.ArrayMethodsMachine.WeightsEA)
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

        # stats = tools.Statistics(lambda ind: ind.fitness.values)
        # stats.register("avg", numpy.mean)
        # stats.register("std", numpy.std)
        # stats.register("min", numpy.min)
        # stats.register("max", numpy.max)

        SModelArray=self.ArrayMethodsMachine.DeconvCLEAN()
        self.ArrayMethodsMachine.PM.ReinitPop(self.pop,SModelArray)


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
