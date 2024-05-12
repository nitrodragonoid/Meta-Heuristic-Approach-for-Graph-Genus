import random
import math
from numpy.random import choice   
import matplotlib.pyplot as plt
import numpy as np
# import sage

def permutation(lst):
    
    if len(lst) == 0:
        return []
 
    if len(lst) == 1:
        return [lst]
 
 
    l = [] 
    for i in range(len(lst)):
       m = lst[i]
 
       remLst = lst[:i] + lst[i+1:]
 
       for p in permutation(remLst):
           l.append([m] + p)
    return l

class EA:
    def __init__(self, graph = {
    0: [1,3,2],
    1: [2,3,0],
    2: [0,3,1],
    3: [2,0,1]
}, size = 30, generations = 50 , offsprings =  10, rate = 0.5, iteration = 10, mutation = 1, parent_scheme = 1, surviver_scheme = 1, tournament_size = 2, data = {1:(1,1)}):
        
        self.graph = graph
        self.size = size
        self.population = {}
        self.generation = generations
        self.offsprings = offsprings
        self.mutation_rate = rate
        self.iterations = iteration
        self.parent_scheme = parent_scheme
        self.surviver_scheme = surviver_scheme
        self.tournament_size = tournament_size

    # def get_data(self, file):
    #     data = {}
    #     cities = []
    #     f = open(file, "r")
    #     l = 1
    #     for x in f:
    #         if x == "EOF":
    #             break
    #         if l > 7:
    #             info = x.split(" ")
    #             data[int(info[0])] = (float(info[1]),float(info[2]))
    #             cities.append(int(info[0]))
    #         l+=1
    #     self.cities = cities
    #     self.data = data
    #     return self.data
    
    def getDirEdges(self, G):
        E = set()
        for v in G.keys():
            for u in G[v]:
                if (v,u) not in E:
                    E.add((v,u))
        return E
    
    def getEdges(self,G):
        E = set()
        for v in G.keys():
            for u in G[v]:
                if (u,v) not in E:
                    E.add((v,u))
        return E
    
    def compute_fitness(self,rotation):
        D = list(self.getDirEdges(self.graph))
        faces = 0 
        unused = D
        arc = unused[0]
        start = arc
        while len(unused) > 0:
            unused.remove(arc)
            arc = (arc[1], rotation[arc[1]][(rotation[arc[1]].index(arc[0]) + 1)%len(rotation[arc[1]])])
            # if arc[0] != rotation[arc[1]][-1]:
            #     arc = (arc[1], rotation[arc[1]][rotation[arc[1]].index(arc[0]) + 1])
            # else:
            #     arc = (arc[1], rotation[arc[1]][0])

            if arc == start:
                faces+=1
                if len(unused) <= 0:
                    break
                arc = unused[0]
                start = arc
        return faces
    # V-E+F = 2-2g
    def initialize_population(self):
        self.population = {}
        for i in range(self.size):
            ind  = []
            for v in list(self.graph.keys()):
                N = self.graph[v].copy()
                random.shuffle(N)
                ind.append(tuple(N))
            ind = tuple(ind)
            self.population[ind] = self.compute_fitness(ind)
            
            # print(self.population[tuple(ind)])
        return self.population
        
    def crossover(self, p1, p2):
        parent_1 = list(p1)
        parent_2 = list(p2)
        ind = []
        start = random.randint(0, (len(parent_1)//2)-1)
        
        for i in range(start):
            ind.append(parent_2[i])
        for j in range(start, len(parent_1)):
            ind.append(parent_1[j])
        # for i in range(len(parent_1)//2):
        #     ind.append(parent_1[i+start])
            
        # for j in range(len(parent_2)):
        #     if parent_2[j] not in ind:
        #         ind.append(parent_2[j])
        if len(ind) != len(parent_1):
            print("error")
            return False
        
        return tuple(ind)
    
    # selection schemes
    def fitness_proportional(self):
        
        total_weight = 0
        for ind in self.population:
            total_weight += self.population[ind]
            
        individuals =  list(self.population.keys())
        
        c = []
        for i in range(len(individuals)):
            c.append(i)
        
        relative_fitness= [(self.population[i])/total_weight for i in individuals]
        
        win = choice(c, 1, p=relative_fitness)

        return individuals[win[0]]
    
    
    def ranked(self):
        pop = self.population.copy()
        sor = sorted(pop.keys(), key=lambda x: pop[x], reverse=True)  # Sort by fitness, descending

        individuals = list(self.population.keys())
        
        n = len(individuals)
        total_weight = (n*(n+1)) / 2
        
        c = list(range(n))
        relative_fitness = [(i+1) / total_weight for i in range(n)]  # Weights are now correctly ordered
        
        win = choice(c, 1, p=relative_fitness)
        return individuals[win[0]]


    def tournament(self, size):
        participants = {}
        for i in range(size):
            p = random.choice(list(self.population.keys()))
            participants[p] = self.population[p]
            
        min = math.inf
        winner = p
        for i in participants:
            if participants[i] < min:
                min = participants[i]
                winner = i
        return winner
    
    def truncation(self,sols):
        winner = sols[0]
        max = 0
        for i in sols:
            if self.population[i] > max:
                max = self.population[i]
                winner = i
        return winner
    
    def random_selection(self):
        return random.choice(list(self.population.keys()))
    
    
    
    # selection schemes for parents selection 
    def create_offsprings_fitness_proportional(self):
            
        total_weight = 0
        for ind in self.population:
            total_weight += self.population[ind]
            
        individuals =  list(self.population.keys())
                
        c = []
        for i in range(len(individuals)):
            c.append(i)
        
        relative_fitness= [(self.population[i])/total_weight for i in individuals]
        
        
        for o in range(self.offsprings):
            
            win = choice(c, 1, p=relative_fitness)
            parent_1 = individuals[win[0]]
            win = choice(c, 1, p=relative_fitness)
            parent_2 = individuals[win[0]]

            child = self.mutation(self.crossover(parent_1, parent_2))
            # c = self.crossover(parent_1, parent_2)
            # print("here",c)
            # child = self.mutation(c)
            self.population[child] = self.compute_fitness(child)
                
        return self.population
    
        # pop = self.population.copy()
        # sor = sorted(pop.keys(), key=lambda x: pop[x], reverse=True)  # Sort by fitness, descending

        # individuals = list(self.population.keys())
        
        # n = len(individuals)
        # total_weight = (n*(n+1)) / 2
        
        # c = list(range(n))
        # relative_fitness = [(i+1) / total_weight for i in range(n)]  # Weights are now correctly ordered
        
        # win = choice(c, 1, p=relative_fitness)
        # return individuals[win[0]]
    def create_offsprings_ranked(self):
        
        pop = self.population.copy()
        sor = sorted(pop.keys(), key=lambda x: pop[x], reverse=True)  # Sort by fitness, descending

        individuals = list(self.population.keys())
        
        n = len(individuals)
        total_weight = (n*(n+1))/2
        
        c = []
        for i in range(len(individuals)):
            c.append(i)
        
        relative_fitness = [(i+1)/total_weight for i in c]
        
        
        for o in range(self.offsprings):
            
            win = choice(c, 1, p=relative_fitness)
            parent_1 = individuals[win[0]]
            
            win = choice(c, 1, p=relative_fitness)
            parent_2 = individuals[win[0]]
            # num = random.randint(0, math.floor(current))
            
            # parent_1 = list(self.population.keys())[0]
            # parent_2 = list(self.population.keys())[0]
            # for ran in ranks:
            #     if num >= ran[0] and num <= ran[1]:
            #         parent_1 = ranks[ran]
            #         # print(parent_1)
            #         break
                    
            # num = random.randint(0, math.floor(current))
        
            # for ran in ranks:
            #     if num >= ran[0] and num <= ran[1]:
            #         parent_2 = ranks[ran]
            #         break
                              
            child =  self.mutation(self.crossover(parent_1, parent_2))
            self.population[child] = self.compute_fitness(child)
                
        return self.population
    
    def create_offsprings_tournament(self, size):
        for o in range(self.offsprings):
            parent_1 = self.tournament(size)
            parent_2 = self.tournament(size)
            
                           
            child = self.mutation(self.crossover(parent_1, parent_2))
            self.population[child] = self.compute_fitness(child)
                
        return self.population
    
    def create_offsprings_truncation(self):
        arr = list(self.population.keys()).copy()
        for o in range(self.offsprings):
            parent_1 = self.truncation(arr)
            arr.remove(parent_1)
            parent_2 = self.truncation(arr)
            
            child =  self.mutation(self.crossover(parent_1, parent_2))
            self.population[child] = self.compute_fitness(child)
                
        return self.population
    
    def create_offsprings_random_selection(self):
        for o in range(self.offsprings):
            parent_1 = self.random_selection()
            parent_2 = self.random_selection()
            
            
            child =  self.mutation(self.crossover(parent_1, parent_2))
            self.population[child] = self.compute_fitness(child)
                
        return self.population


    # selection schemes for surviver selection
    def survivers_fitness_proportional(self):

        
        total_weight = 0
        for ind in self.population:
            total_weight += self.population[ind]
            
        individuals =  list(self.population.keys())
        
        c = []
        for i in range(len(individuals)):
            c.append(i)
        
        relative_fitness= [(self.population[i])/total_weight for i in individuals]
        
        win = choice(c, 1, p=relative_fitness)

        
        new = {}
        for s in range(self.generation):
            
            win = choice(c, 1, p=relative_fitness)
            
            sur = individuals[win[0]]
            new[sur] = self.population[sur]

        self.population = new
        return self.population
    
    def survivers_ranked(self):
        pop = self.population.copy()
        sor = sorted(pop.keys(), key=lambda x: pop[x])
        sor.reverse()
        
        # ranks = {}
        # current = 0
        # for i in range(len(sor)):
        #     ranks[(current,current+ i+1)] = sor[i]
        #     current+= i
        # num = random.randint(0, math.floor(current))
        
        individuals =  list(self.population.keys())
        
        n = len(individuals)
        total_weight = (n*(n+1))/2
        
        
        c = []
        for i in range(len(individuals)):
            c.append(i)
        
        relative_fitness = [(i+1)/total_weight for i in c]
        
        new = {}
        
        for s in range(self.generation):
            
            win = choice(c, 1, p=relative_fitness)
            
            sur = individuals[win[0]]
            new[sur] = self.population[sur]
            # win = choice(c, 1, p=relative_fitness)
            # parent_1 = individuals[win[0]]
        
            # for ran in ranks:
            #     if num >= ran[0] and num <= ran[1]:
            #         new[ranks[ran]] = self.population[ranks[ran]]
        
        self.population = new
        return self.population
    
    def survivers_tournament(self, size):
        new = {}
        
        for s in range(self.generation):
            
            survivor = self.tournament(size)
            new[survivor] = self.population[survivor]
        
        self.population = new
        return self.population
    
    def survivers_truncation(self):
        new = {}
        arr = list(self.population.keys()).copy()
        if len(arr) == self.generation:
            return self.population
        for s in range(self.size):
            
            survivor = self.truncation(arr)
            arr.remove(survivor)
            new[survivor] = self.population[survivor]
        
        self.population = new
        return self.population
    
    def survivers_random_selection(self):
        new = {}
        
        for s in range(self.generation):
            
            survivor = self.random_selection()
            new[survivor] = self.population[survivor]
        
        self.population = new
        return self.population
    
    
    # mutation schemes
    def mutation(self,individual):
        r = random.randint(0,100)
        num = 100 * self.mutation_rate
        # print(individual)
        mutated = list(individual)
        if r <= num:
            k = random.randint(0,len(mutated)-1)
            rot = list(mutated[k])
            i = random.randint(0,len(rot)-1)
            j = random.randint(0,len(rot)-1)
            
            temp = rot[i]
            rot[i] = rot[j]
            rot[j] =  temp
            # print("mut",mutated)
        
            mutated[k] = tuple(rot)
            
        return tuple(mutated)
    
    
    
    def insert_mutation(self,individual):
        r = random.randint(0,100)
        num = 100 * self.mutation_rate
        mutated = list(individual)
        if r <= num:
            i = random.randint(0,len(mutated)-1)
            j = random.randint(0,len(mutated)-1)
            city = mutated[i]
            mutated.remove(city)
            mutated.insert(j,city)
            
        return tuple(mutated)
    
    
    def best(self):
        winner = list(self.population.keys())[0]
        max = 0
        for i in self.population:
            if self.population[i] > max:
                max = self.population[i]
                winner = i
        
        return winner, max
    
    
    def tsp_brute_force(self):
        arr = self.cities.copy()
        
        sols = permutation(arr)
        
        min_fit = math.inf
        best = sols[0]
        for i in sols:
            fitness = self.compute_fitness(i)
            if fitness <= min_fit:
                min_fit = fitness
                best = i
                
        print(best,":",min_fit)
        return best,min_fit
    
    def EulerianCharacteristics(self, G, F):
        E = len(self.getEdges(G))
        V = len(list(G.keys()))
        g = math.ceil((2-F+E-V)/2)
        return g
    
    def evolution(self):
        self.initialize_population()
        best_so_far = []
        avg_so_far = []
        for g in range(self.generation):
            print(g, self.best())
            if g == 0:
                embedding, faces = self.best()
                print("Embbedding is:",embedding)
                print("Number of faces are:",faces)
                print("Genus od embedding is:",self.EulerianCharacteristics(self.graph,faces))
        
            if self.parent_scheme == 1:
                self.create_offsprings_fitness_proportional()
            elif self.parent_scheme == 2:
                self.create_offsprings_ranked()
            elif self.parent_scheme == 3:
                self.create_offsprings_tournament(self.tournament_size)
            elif self.parent_scheme == 4:
                self.create_offsprings_truncation()
            else:
                self.create_offsprings_random_selection()
            

            if self.parent_scheme == 1:
                self.survivers_fitness_proportional()
            elif self.parent_scheme == 2:
                self.survivers_ranked()
            elif self.parent_scheme == 3:
                self.survivers_tournament(self.tournament_size)
            elif self.parent_scheme == 4:
                self.survivers_truncation()
            else:
                self.survivers_random_selection()

            current_best, current_max_fitness = self.best()
            current_avg_fitness = self.average_fitness()
            if current_avg_fitness <= 0:
                print("masla")
                for i in self.population:
                    print(self.population[i],":",i)
                    return
            best_so_far.append(self.EulerianCharacteristics(self.graph, current_max_fitness))
            # avg_so_far.append(self.EulerianCharacteristics(self.graph, current_avg_fitness))

        embedding, faces = self.best()
        print("Embbedding is:",embedding)
        print("Number of faces are:",faces)
        print("Genus od embedding is:",self.EulerianCharacteristics(self.graph,faces))

        generations = np.arange(1, self.generation + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(generations, best_so_far, label='Best Fitness So Far', color='red')
        # plt.plot(generations, avg_so_far, label='Average Fitness So Far', color='blue')
        plt.title('Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (genus from embedding)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def average_fitness(self):
        total = 0
        for i in self.population:
            total += self.population[i]
        return total/self.size
        
    def test(self,k):
        l = k+1
        fitprop = []
        fitprop_average = []
        for i in range(self.generation):
            r = []
            r1 = []
            for j in range(k+1):
                r.append(0)
                r1.append(0)
            fitprop.append(r)
            fitprop_average.append(r1)
        
        
        for i in range(k):
            self.initialize_population()
            for g in range(self.generation):
            # print(g)
                self.create_offsprings_fitness_proportional()

                self.survivers_fitness_proportional()
                
                b = self.best()[1]
                fitprop[g][i] = b
                fitprop[g][k] += b
                
                a = self.average_fitness()
                fitprop_average[g][i] = a
                fitprop_average[g][k] += a
        for g in range(self.generation):
            fitprop[g][k] = fitprop[g][k]/k
            fitprop_average[g][k] = fitprop_average[g][k]/k
        
        fitprop_x = []
        y = []
        for i in range(self.generation):
            y.append(i+1)
            fitprop_x.append(fitprop[i][k])
            
            
        ranked = []
        ranked_average = []
        for i in range(self.generation):
            r = []
            r1 = []
            for j in range(k+1):
                r.append(0)
                r1.append(0)
            ranked.append(r)
            ranked_average.append(r1)
        
        for i in range(k):
            self.initialize_population()
            for g in range(self.generation):
            # print(g)
                self.create_offsprings_ranked()

                self.survivers_ranked()
                
                b = self.best()[1]
                ranked[g][i] = b
                ranked[g][k] += b
                a = self.average_fitness()
                ranked_average[g][i] = a
                ranked_average[g][k] += a
        for g in range(self.generation):
            ranked[g][k] = ranked[g][k]/k
            ranked_average[g][k] = ranked_average[g][k]/k
        
        ranked_x = []
        for i in range(self.generation):
            ranked_x.append(ranked[i][k])
            
        tournament = []
        tournament_average = []
        for i in range(self.generation):
            r = []
            r1 = []
            for j in range(k+1):
                r.append(0)
                r1.append(0)
            tournament.append(r)
            tournament_average.append(r1)
        
        for i in range(k):
            self.initialize_population()
            for g in range(self.generation):
            # print(g)
                self.create_offsprings_tournament(self.tournament_size)

                self.survivers_tournament(self.tournament_size)
                
                b = self.best()[1]
                tournament[g][i] = b
                tournament[g][k] += b
                
                a = self.average_fitness()
                tournament_average[g][i] = a
                tournament_average[g][k] += a
        for g in range(self.generation):
            tournament[g][k] = tournament[g][k]/k
            tournament_average[g][k] = tournament_average[g][k]/k
        
        tournament_x = []
        for i in range(self.generation):
            tournament_x.append(tournament[i][k])
            
            
            
        truncation = []
        truncation_average = []
        for i in range(self.generation):
            r = []
            r1 = []
            for j in range(k+1):
                r.append(0)
                r1.append(0)
            truncation.append(r)
            truncation_average.append(r1)
        
        for i in range(k):
            self.initialize_population()
            for g in range(self.generation):
            # print(g)
                self.create_offsprings_truncation()

                self.survivers_truncation()
                
                b = self.best()[1]
                truncation[g][i] = b
                truncation[g][k] += b
                
                a = self.average_fitness()
                truncation_average[g][i] = a
                truncation_average[g][k] += a
        for g in range(self.generation):
            truncation[g][k] = truncation[g][k]/k
            truncation_average[g][k] = truncation_average[g][k]/k
        
        truncation_x = []
        for i in range(self.generation):
            truncation_x.append(truncation[i][k])
            
            
            
        random = []
        random_average = []
        for i in range(self.generation):
            r = []
            r1 = []
            for j in range(k+1):
                r.append(0)
                r1.append(0)
            random.append(r)
            random_average.append(r1)
        
        for i in range(k):
            self.initialize_population()
            for g in range(self.generation):
            # print(g)
                self.create_offsprings_random_selection()

                self.survivers_random_selection()
                
                b = self.best()[1]
                random[g][i] = b
                random[g][k] += b
                
                a = self.average_fitness()
                random_average[g][i] = a
                random_average[g][k] += a
        for g in range(self.generation):
            random[g][k] = random[g][k]/k
            random_average[g][k] = random_average[g][k]/k
        
        random_x = []
        for i in range(self.generation):
            random_x.append(random[i][k])
        
        print("Fitness propotional best fitness table")
        for i in range(len(random)):
            print(fitprop[i])
            
        print("Fitness propotional average fitness table")
        for i in range(len(random)):
            print(fitprop_average[i])
            
            
        print("Ranked best fitness table")
        for i in range(len(random)):
            print(ranked[i])
            
        print("Ranked average fitness table")
        for i in range(len(random)):
            print(ranked_average[i])
            
            
        print("Tournament best fitness table")
        for i in range(len(random)):
            print(tournament[i])
            
        print("Tournament average fitness table")
        for i in range(len(random)):
            print(tournament_average[i])
            
            
        print("Truncation best fitness table")
        for i in range(len(random)):
            print(truncation[i])
            
        print("Truncation average fitness table")
        for i in range(len(random)):
            print(truncation_average[i])
            
            
        print("Random best fitness table")
        for i in range(len(random)):
            print(random[i])   
            
        print("Random average fitness table")
        for i in range(len(random)):
            print(random_average[i])    
        
        # print(truncation_x)
        ypoints_1 = np.array(fitprop_x)
        ypoints_2 = np.array(ranked_x)
        ypoints_3 = np.array(tournament_x)
        ypoints_4 = np.array(truncation_x)
        ypoints_5 = np.array(random_x)
        xpoints = np.array(y)
        
        # plt.plot(X, y, color='r', label='sin') 
        # plt.plot(X, z, color='g', label='cos')

        plt.plot(xpoints, ypoints_1, color='r', label='fitness proportional')
        plt.plot(xpoints, ypoints_2, color='b', label='ranked')
        plt.plot(xpoints, ypoints_3, color='g', label='tournament')
        plt.plot(xpoints, ypoints_4, color='y', label='truncation')
        plt.plot(xpoints, ypoints_5, color='k', label='random')
        plt.show()
                

            
            
            
                

        
# size = 30, generations = 50 , offsprings =  10, rate = 0.5, iteration = 10, mutation = 1, parent_scheme = 1, surviver_scheme = 1
       
petersen = {0: [5,6,9],
    1: [3,4,6],
    2: [4,5,7],
    3: [1,5,8],
    4: [1,2,9],
    5: [2,3,0],
    6: [1,7,0],
    7: [2,6,8],
    8: [3,7,9],
    9: [4,8,0]}

k7 = {
    0: [1,2,3,4,5,6],
    1: [0,2,3,4,5,6],
    2: [0,1,3,4,5,6],
    3: [0,1,2,4,5,6],
    4: [0,1,2,3,5,6],
    5: [0,1,2,3,4,6],
    6: [0,1,2,3,4,5],
}
       
       
k8 = {
    0: [1,2,3,4,5,6,7],
    1: [0,2,3,4,5,6,7],
    2: [0,1,3,4,5,6,7],
    3: [0,1,2,4,5,6,7],
    4: [0,1,2,3,5,6,7],
    5: [0,1,2,3,4,6,7],
    6: [0,1,2,3,4,5,7],
    7: [0,1,2,3,4,5,6]
}
       
       
k4 = {
    0: [1,3,2],
    1: [2,3,0],
    2: [0,3,1],
    3: [2,0,1]
}
 
k5 = {
    0: [1,3,2,4],
    1: [2,3,0,4],
    2: [0,3,1,4],
    3: [2,0,1,4],
    4: [1,3,2,0]
}

Test = EA( petersen, size = 100, generations = 1000, offsprings =  20, rate = 0.7, parent_scheme = 1, surviver_scheme = 4, tournament_size= 10)
# Test.get_data("qa194.tsp")
Test.evolution()

Test = EA( k8, size = 100, generations = 1000, offsprings =  20, rate = 0.7, parent_scheme = 1, surviver_scheme = 4, tournament_size= 10)
# Test.get_data("qa194.tsp")
Test.evolution()

