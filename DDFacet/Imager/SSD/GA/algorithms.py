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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DDFacet.compatibility import range

import random

from DDFacet.Other import ClassTimeIt
from deap import tools
import copy

def varAnd(population, toolbox, cxpb, mutpb, ArrayMethodsMachine,MutConfig):
    T= ClassTimeIt.ClassTimeIt("VarAnd")
    T.disable()
    offspring = [toolbox.clone(ind) for ind in population]
    T.timeit("   clone")

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
            del offspring[i-1].fitness.values, offspring[i].fitness.values
    T.timeit("   crossover")
    
    #P0=copy.deepcopy(offspring)
    
    offspring=ArrayMethodsMachine.mutatePop(offspring,mutpb,MutConfig)
    #P1=offspring
    #stop
    # for i in range(len(offspring)):
    #     if random.random() < mutpb:
    #         #print "Mutate %i"%i
    #         ArrayMethodsMachine
    #         offspring[i], = toolbox.mutate(offspring[i])
    #         del offspring[i].fitness.values


    # offspring2 = toolbox.map(toolbox.mutate, offspring)

    # offspring = [I[0] for I in offspring2]

    T.timeit("   mutate")
    
    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__,ArrayMethodsMachine=None,DoPlot=True,
             StopFitness=1e-2,
             MutConfig=None):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    
    invalid_ind = [ind for ind in population]# if not ind.fitness.valid]
    #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    fitnesses = ArrayMethodsMachine.GiveFitnessPop(population)[0]

    #print fitnesses[0]
    #stop
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)


    T= ClassTimeIt.ClassTimeIt()
    T.disable()
    best_ind0 = tools.selBest(population, 1)[0]

    #print best_ind0
    #print "Best indiv 0",best_ind0
    #print "Best indiv 0 fitness",best_ind0.fitness
    best_ind=best_ind0
    # from operator import attrgetter
    # k=1
    # popsorted=sorted(population, key=attrgetter("fitness"), reverse=True)#[:k]
    # V0=[ind.fitness.values[0] for ind in popsorted]
    # V1=[ind.fitness.values[1] for ind in popsorted]
    # A=np.array(V0)+np.array(V1)
    # print A[0:-1]-A[1::]
    # print np.argmax(A)
    # stop
    if (ArrayMethodsMachine is not None)&(DoPlot):
        ArrayMethodsMachine.Plot(population,0)
    
    # Begin the generational process
    for gen in range(1, ngen+1):
        #print gen
        # Select the next generation individuals

        offspring = toolbox.select(population, len(population))
        T.timeit("select")

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb, ArrayMethodsMachine, MutConfig)
        offspring[0]=best_ind
        T.timeit("vary")
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring]# if not ind.fitness.valid]
        #fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        fitnesses = ArrayMethodsMachine.GiveFitnessPop(invalid_ind)[0]

        T.timeit("fitness")
        for ind, fit in zip(invalid_ind, fitnesses):
            #print fit
            ind.fitness.values = fit
#        best_ind0 = tools.selBest(population, 1)[0]
#        print "Best indiv 0b fitness",best_ind0.fitness
        T.timeit("export")
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
#        best_ind0 = tools.selBest(population, 1)[0]
#        print "Best indiv 0c fitness",best_ind0.fitness
        T.timeit("hof")
            
        # Replace the current population by the offspring
        population[:] = offspring
        best_ind = tools.selBest(population, 1)[0]
        #ArrayMethodsMachine.setBestIndiv(best_ind)

        # print "Diff ",np.max(best_ind-best_ind0)
        # best_ind0=best_ind
        # from operator import attrgetter
        # k=1
        # popsorted=sorted(population, key=attrgetter("fitness"), reverse=True)#[:k]
        # V0=[ind.fitness.values[0] for ind in popsorted]
        # V1=[ind.fitness.values[1] for ind in popsorted]
        # A=np.array(V0)+np.array(V1)
        # print "IND=",np.argmax(A)

        if (ArrayMethodsMachine is not None)&(DoPlot):
            if gen%50==0:
                ArrayMethodsMachine.Plot(population,gen)

        # if gen%10==0:
        #     ArrayMethodsMachine.Plot(population,gen)
        #     stop
        # ArrayMethodsMachine.Plot(population,gen)
        # stop
        
        #print best_ind
        #print "Best indiv fitness",best_ind.fitness

        
        #BestFitNess=best_ind.fitness.values[0]
        #if BestFitNess<StopFitness:
        #    break

        #if gen%30==0:
        #    Plot(population,gen)
        #Plot(population,gen)
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        T.timeit("stats")
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        T.timeit("record")
        if verbose:
            print(logbook.stream)        

    #print "Best indiv1 fitness",best_ind.fitness
    return population, logbook

#from ModArrayOps_np import *


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, ("The sum of the crossover and mutation "
        "probabilities must be smaller or equal to 1.0.")
    
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))
    
    return offspring

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__,ArrayMethodsMachine=None):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    ArrayMethodsMachine.Plot(population,0)
    # Begin the generational process
    for gen in range(1, ngen+1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)


        if ArrayMethodsMachine is not None:
            if gen%30==0:
                ArrayMethodsMachine.Plot(population,gen)
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook
    
def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                    stats=None, halloffame=None, verbose=__debug__):
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen+1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)


    return population, logbook

def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None, 
                     verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(ngen):
        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        if halloffame is not None:
            halloffame.update(population)
        
        # Update the strategy with the evaluated individuals
        toolbox.update(population)
        
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook
