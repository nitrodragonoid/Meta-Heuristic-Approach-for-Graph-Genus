import numpy as np
from operator import attrgetter
from random import randint, random, sample
import matplotlib.pyplot as plt
import math

class Chromosone:
    def __init__(self, solution, fitness=0) -> None:
        self.solution = solution
        self.fitness = fitness

class EA:
    """
    Generic EA Class
    """
    def __init__(self, populationSize, nOffsprings, nGenerations, mutationRate, nIterations, isMaximization=True) -> None:
        self.populationSize = populationSize
        self.nOffsprings = nOffsprings
        self.nGenerations = nGenerations
        self.mutationRate = mutationRate
        self.nIterations = nIterations 
        self.isMaximization = isMaximization 

    def __initPopulation(self):
        pass

    def fitness(self):
        pass

    def crossover(self):
        pass

    def mutation(self):
        pass

    def selection(self, scheme, nSelected=2, isSurvivorSelection=False):
        self.selected = []

        if scheme.lower() == 'random':
            for i in range(nSelected):
                self.selected.append(self.population[randint(0, len(self.population)-1)])
        
        elif scheme.lower() == 'truncation':
            for n in range(nSelected):
                tempPopulation = self.population.copy() # copy to preserve original population order
                tempPopulation.sort(key=lambda chromosone: chromosone.fitness, reverse=self.isMaximization)

                self.selected = [tempPopulation[i] for i in range(nSelected)]

        elif scheme.lower() == 'fps':
            # Sort Population 
            population = self.population.copy()
            population.sort(key=lambda chromosone: chromosone.fitness, reverse=True)

            fitnessSum = sum([chromosone.fitness for chromosone in population])
            normalisedFitnessValues = [chromosone.fitness/fitnessSum for chromosone in population]

            if not self.isMaximization:
                normalisedFitnessValues.reverse()

            cumilativeFitnessValues = [0]
            cumValue = 0
            for fitness in normalisedFitnessValues:
                cumValue += fitness
                cumilativeFitnessValues.append(cumValue)

            for n in range(nSelected):
                randomN = random()

                i = 0
                while randomN > cumilativeFitnessValues[i] and not randomN <= cumilativeFitnessValues[i + 1] and i < len(cumilativeFitnessValues) - 1: 
                    i += 1
                
                self.selected.append(population[i])

        elif scheme.lower() == 'rbs':
            # population = list(enumerate(self.population))
            # population.sort(key=lambda chromosone:chromosone[1].fitness, reverse=True)

            # Sort Population according to its fitness value low to high
            population = self.population.copy()
            population.sort(key=lambda chromosone: chromosone.fitness, reverse=False)
            population2 = population.copy()

            # Set ranks to the sorted population
            population = list(enumerate(population))

            # Normalise rank values
            rankSum = sum([chromosone[0] for chromosone in population])
            normalisedRankValues = [chromosone[0]/rankSum for chromosone in population]

            if not self.isMaximization:
                normalisedRankValues.reverse()

            cumilativeRankValues = [0]
            cumValue = 0
            for rank in normalisedRankValues:
                cumValue += rank
                cumilativeRankValues.append(cumValue)

            for n in range(nSelected):
                randomN = random()

                i = 0
                while randomN > cumilativeRankValues[i] and not randomN <= cumilativeRankValues[i + 1] and i < len(cumilativeRankValues) - 1: 
                    i += 1
                
                self.selected.append(population2[i])

        elif scheme.lower() == 'bt':
            for n in range(nSelected):
                contestants = sample(self.population, nSelected)
                contestants.sort(key=lambda contestant: contestant.fitness, reverse=self.isMaximization)
                winner = contestants[0]

                self.selected.append(winner)


        if isSurvivorSelection:
            self.population = self.selected

    def solve(self, selection='bt', survivor='truncation', title=''):
        self.BSF = [[] for gen in range(self.nGenerations)]
        self.ASF = [[] for gen in range(self.nGenerations)]

        self.__initPopulation()
        self.fitness()

        for i in range(self.nIterations):
            self.__initPopulation()
            self.fitness()

            for gen in range(self.nGenerations):                                    
                self.selection(selection, self.nOffsprings)
                self.crossover()
                self.mutation()
                self.selection(survivor, self.populationSize, True)
                self.fitness()
                BSF = max(self.population, key=attrgetter('fitness')).fitness
                ASF = sum([chromosone.fitness for chromosone in self.population]) / len(self.population)
                self.BSF[gen].append(BSF)
                self.ASF[gen].append(ASF)

                print(f'Iteration {i}, Generation {gen}, BSF: {BSF}, ASF: {ASF}')

        self.avgBSF = [sum(self.BSF[gen])/len(self.BSF[gen]) for gen in range(self.nGenerations)]
        self.avgASF = [sum(self.ASF[gen])/len(self.ASF[gen]) for gen in range(self.nGenerations)]

        print('avgBSF', self.avgBSF)
        print('avgASF', self.avgASF)
    
        generations = [gen for gen in range(self.nGenerations)]

        plt.plot(generations, self.avgBSF, label = 'Average Best So Far')
        plt.plot(generations, self.avgASF, label = 'Average Average Fitness')
        
        plt.xlabel('Generations')
        plt.ylabel('Optimum')
        plt.title(title if title else f'{selection} vs {survivor}')
        plt.legend()
        plt.show()

    def plotGraph(self, x, y, labels, title):
        for i in range(len(y)):      
            plt.plot(x, y[i], label = labels[i])
        
        plt.xlabel('Generations')
        plt.ylabel('Optimum')
        plt.title(title)
        plt.legend()
        plt.show()
