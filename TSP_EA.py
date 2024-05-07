from EA import *

class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

class TSP(EA):
    def __init__(self, tspFile, populationSize, nOffsprings, nGenerations, mutationRate, nIterations) -> None:
        EA.__init__(self, populationSize, nOffsprings, nGenerations, mutationRate, nIterations, False)
        
        # Load data
        self.__loadData(tspFile)

        # Total Cities
        self.nCities = len(self.nodes)

        # Randomly Initialise a Population
        self.__initPopulation()


    def __loadData(self, tspFile):
         # Load File 
        self.nodes = []

        with open(tspFile, 'r') as file:
            lines = file.readlines()

            for line in lines[lines.index("NODE_COORD_SECTION\n") + 1:]:
                if not line.strip():
                    continue

                if line.startswith('EOF'):
                    break

                data = line.split()
                id = data[0]
                x = float(data[1])
                y = float(data[2])

                node = Node(id, x, y)
                self.nodes.append(node)

        

    def __initPopulation(self):
        self.population = []
        for i in range(self.populationSize):
            # Create a Random Path
            solution = sample(self.nodes, len(self.nodes))
            # for node in solution:
            #     print(f'{node.id}', end=' ')
            # print()
            # Calculate its fitness
            fitness = self.calculateFitness(solution)
            # print(fitness)

            self.population.append(Chromosone(solution, fitness))

    def fitness(self):
        for chromosone in range(len(self.population)):
            fitness = self.calculateFitness(self.population[chromosone].solution)
            self.population[chromosone].fitness = fitness

    def calculateFitness(self, solution):
        fitness = 0 
        for i in range(len(solution)-1):
            dx = solution[i].x - solution[i+1].x
            dy = solution[i].y - solution[i+1].y
            fitness += math.sqrt(dx**2 + dy**2)
        return fitness
               
    def crossover(self):
        self.offsprings = []

        for parent in range(0, len(self.selected)-1, 2): # Interval of 2 to select 2 new parents for crossover
            p1 = self.selected[parent]
            p2 = self.selected[parent + 1]

            crossoverPoint1 = randint(1, self.nCities-1) 
            crossoverPoint2 = randint(crossoverPoint1, self.nCities-1) 

            # Ref Crossover from the slides
            offspring1 = [0 for i in range(self.nCities)]
            offspring2 = offspring1.copy()

            for city in range(crossoverPoint1, crossoverPoint2+1):
                offspring1[city] = p1.solution[city]
                offspring2[city] = p2.solution[city]

            for city in range(crossoverPoint2+1, self.nCities):
                offspring1[city] = p2.solution[city]
                offspring2[city] = p1.solution[city]

            for city in range(0, crossoverPoint1):
                offspring1[city] = p2.solution[city]
                offspring2[city] = p1.solution[city]

            self.offsprings.append(Chromosone(offspring1, self.calculateFitness(offspring1)))
            self.offsprings.append(Chromosone(offspring2, self.calculateFitness(offspring2)))

    def mutation(self):     
        for i in range(len(self.offsprings)):
            randomN = random()

            if randomN < self.mutationRate:
                if randomN < self.mutationRate:
                    # Swap 2 Random Nodes
                    index1, index2 = sample(range(len(self.offsprings[i].solution)), 2)
                    self.offsprings[i].solution[index1], self.offsprings[i].solution[index2] = self.offsprings[i].solution[index2], self.offsprings[i].solution[index1]
                
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
                self.selection(selection, nOffsprings)
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

tspFile = 'qa194.tsp'
populationSize = 100
nOffsprings = 60
nGenerations = 1000
mutationRate = 0.5
nIterations = 5


tsp = TSP(tspFile, populationSize, nOffsprings, nGenerations, mutationRate, nIterations)


# tsp.solve('bt', 'truncation', 'Binary Tournament and Truncation')
# tsp.solve('fps', 'random', 'FPS and Random')
# tsp.solve('truncation', 'truncation', 'Truncation and Truncation')
# tsp.solve('random', 'random', 'Random and Random')
# tsp.solve('fps', 'truncation', 'FPS and Truncation')
# tsp.solve('rbs', 'bt', 'RBS and Binary Tournament')
tsp.solve('random', 'truncation', 'Random and Truncation')
tsp.solve('fps', 'rbs', 'Random and RBS')
tsp.solve('fps', 'fps', 'FPS and FPS')
