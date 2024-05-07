import numpy as np
from pprint import pprint
from EA import *

class Knapsack(EA):
    def __init__(self, knapsackFile, populationSize, nOffsprings, nGenerations, mutationRate, nIterations) -> None:
        EA.__init__(self, populationSize, nOffsprings, nGenerations, mutationRate, nIterations)
        
        self.profit = np.loadtxt(knapsackFile, skiprows = 1, usecols = 0)
        self.weight = np.loadtxt(knapsackFile, skiprows = 1, usecols = 1)
        self.capacity = np.loadtxt(knapsackFile, usecols = 1)[0]
        self.nItems = int(np.loadtxt(knapsackFile, usecols = 0)[0])
        
        # Randomly Initialise a Population
        self.__initPopulation()

    def __initPopulation(self):
        self.population = []
        for i in range(self.populationSize):
            solution = [0 if randint(0, 1) == 0 else 1 for n in range(self.nItems)]
            fitness = self.calculateProfit(solution) if self.calculateWeight(solution) < self.capacity else 0
            self.population.append(Chromosone(solution, fitness))

    def calculateWeight(self, chromosone):
        return sum([self.weight[i] * chromosone[i] for i in range(len(chromosone))])
    
    def calculateProfit(self, chromosone):
        return sum([self.profit[i] * chromosone[i] for i in range(len(chromosone))])

    def fitness(self):
        for chromosone in range(len(self.population)):
            profitSum = self.calculateProfit(self.population[chromosone].solution)
            weightSum = self.calculateWeight(self.population[chromosone].solution)

            if weightSum <= self.capacity:
                self.population[chromosone].fitness = profitSum
            else:
                self.population[chromosone].fitness = 0
               
    def crossover(self):
        self.offsprings = []

        for parent in range(0, len(self.selected)-1, 2): # Interval of 2 to select 2 new parents for crossover
            p1 = self.selected[parent]
            p2 = self.selected[parent + 1]

            crossoverPoint1 = randint(1, self.nItems-1) 
            crossoverPoint2 = randint(crossoverPoint1, self.nItems-1) 

            # Ref Crossover from the slides
            offspring1 = [0 for i in range(self.nItems)]
            offspring2 = offspring1.copy()

            for bit in range(crossoverPoint1, crossoverPoint2+1):
                offspring1[bit] = p1.solution[bit]
                offspring2[bit] = p2.solution[bit]

            for bit in range(crossoverPoint2+1, self.nItems):
                offspring1[bit] = p2.solution[bit]
                offspring2[bit] = p1.solution[bit]

            for bit in range(0, crossoverPoint1):
                offspring1[bit] = p2.solution[bit]
                offspring2[bit] = p1.solution[bit]

            self.offsprings.append(Chromosone(offspring1, self.calculateProfit(offspring1)))
            self.offsprings.append(Chromosone(offspring2, self.calculateProfit(offspring2)))

    def mutation(self):
        for i in range(len(self.offsprings)):
            randomN = random()

            if randomN < self.mutationRate:
                index1 = randint(0, self.nItems-1)
                index2 = randint(0, self.nItems-1)

                self.offsprings[i].solution[index1] = 0 if (self.offsprings[i].solution[index1] == 1) else  1
                self.offsprings[i].solution[index2] = 0 if (self.offsprings[i].solution[index2] == 1) else  1
                
        while len(self.offsprings) > 0:    
            self.population.append(self.offsprings.pop())

    def solve(self, selection='bt', survivor='truncation', title=''):
        self.BSF = [[] for gen in range(self.nGenerations)]
        self.ASF = [[] for gen in range(self.nGenerations)]

        self.__initPopulation()
        self.fitness()

        for i in range(self.nIterations):
            self.__initPopulation()
            self.fitness()

            for gen in range(self.nGenerations):  
                                              
                for n in range(self.nOffsprings//2):
                    self.selection(selection)
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

        self.plotGraph(generations, [self.avgBSF,self.ASF] , ['Average Best So Far', 'Average Average Fitness'], title if title else f'{selection} vs {survivor}' )
     

knapsackFile = 'f8_l-d_kp_23_10000'
populationSize = 30
nOffsprings = 10
nGenerations = 300
mutationRate = 0.5
nIterations = 2

knapsack = Knapsack(knapsackFile, populationSize, nOffsprings, nGenerations, mutationRate, nIterations)

# knapsack.solve('fps', 'random', 'FPS and Random')
# knapsack.solve('bt', 'truncation', 'Binary Tournament and Truncation')
knapsack.solve('truncation', 'truncation', 'Truncation and Truncation')
# knapsack.solve('random', 'random', 'Random and Random')
# knapsack.solve('fps', 'truncation', 'FPS and Truncation')
# knapsack.solve('rbs', 'bt', 'RBS and Binary Tournament')
# knapsack.solve('random', 'truncation', 'Random and Truncation')
# knapsack.solve('fps', 'fps', 'FPS and FPS')
# knapsack.solve('rbs', 'rbs', 'RBS and RBS')
