# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:20:05 2019

@author: farukkutlu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gmplot

# Read the distance-matrix as 81x81 data-frame using pandas.
df1 = pd.read_excel('distancematrix.xls', header=2)
df1.drop(['İL ADI', 'İL PLAKA NO'], axis=1, inplace=True)
# Read the Coordinates-matrix as 81x2 data-frame using pandas.
df2 = pd.read_excel('Coordinates.xlsx', header=0)
# Converting and saving the data-frames to numpy-arrays of (81x81) and (81x2).
distances = df1.values
coordinates = df2.values
#np.savetxt('Distances Array.txt', distances)
#np.savetxt('Coordinates Array.txt', coordinates)
# Creating the necessary functions.
""" The get_path function always return with a list starting with 5.
        which means all the paths start with Ankara.   """
def get_path(n):
    l = list(range(n))
    np.random.shuffle(l)
    l.pop(l.index(5))
    l.insert(0, 5)
    return l

def get_path_length(path):
    path = np.append(path,path[0])
    total_length = 0.0
    for i in range(len(path)-1):
        j, k = path[i], path[i+1]
        total_length += distances[j, k]
    return total_length

def create_a_population(n, m):
    population, fitness = [], []
    for i in range(m):
        gene = get_path(n)
        path_length = get_path_length(gene)
        population.append(get_path(n))
        fitness.append(1e6/path_length)
    return np.array(population), np.array(fitness)

def get_fitness(pop):
    fitness = []
    for i in range(len(pop)):
        path_length = get_path_length(pop[i])
        fitness.append(1e6/path_length)
    avg_fit = sum(fitness)/len(fitness)
    return fitness, avg_fit

def selection(pop, fit, m):
    # Sorting according to fitnesses.
    if type(pop) != list:
        pop = pop.tolist()
    sort_pop = [(x,y) for x,y in reversed(sorted(zip(fit, pop)))]
    sorted_paths = [sort_pop[i][1] for i in range(len(sort_pop))]
    # Creating the mating pool.
    n, pool = len(pop), []
    elites = sorted_paths[:m]
    pool = [sorted_paths[np.random.choice(list(range(n)))] for i in range(n-m)]
    return elites, pool

def cross_over(g1, g2):
    m = (np.random.randint(len(g1)),np.random.randint(len(g1)))
    min_, max_ = min(m), max(m)
    daughter_1 = g2[min_:max_]
    daughter_2 = []
    daughter_2 = [kr for kr in g1 if kr not in daughter_1]
    child = daughter_1.copy()
    child.extend(daughter_2)
    return child

def mutate_gene(gene):
    newgene = gene.copy()
    j, k = np.random.randint(len(gene)), np.random.randint(len(gene)-1)
    x = newgene.pop(j)
    y = newgene.pop(k)
    newgene.insert(j, y)
    newgene.insert(k, x)
    return newgene

def mutation(pop, chance):
    mutated_pop = []
    for gene in pop:
        c = np.random.randint(100)
        if c <= chance*100:
            newgene = mutate_gene(gene)
            mutated_pop.append(newgene)
        else:
            mutated_pop.append(gene)    
    return mutated_pop

def breed(elites, pool, mut_chance):
    n, m = len(elites+pool), len(pool)
    pick_1 = [pool[np.random.choice(list(range(m)))] for i in range(int(n*0.4))]
    pick_2 = [pool[np.random.choice(list(range(m)))] for i in range(int(n*0.2))]
    news = []
    for i in range(len(pick_1)):
        news.append(cross_over(elites[i], pick_1[i]))
    elites.extend(news)
    elites.extend(pick_2)
    elites = mutation(elites, mut_chance)
    fitness, avg_fit = get_fitness(elites)
    return elites, fitness, avg_fit

def plot_path(path):
    path = np.append(path,path[0])
    latitude_list = coordinates[:,0]
    longitude_list = coordinates[:,1]
    path_latitude = latitude_list[path]
    path_longitude = longitude_list[path]
    gmap = gmplot.GoogleMapPlotter(coordinates[39,0], coordinates[39,1], 7)
    gmap.apikey="AIzaSyD_07xnO2NqogYSQoddVloGMufVKcx42lk"
    gmap.scatter( path_latitude, path_longitude, '# FF0000', 
                  size = 40, marker = False )
    gmap.plot(path_latitude, path_longitude,  
           'cornflowerblue', edge_width = 2.5)
    gmap.draw( "Path Map.html" )
    return None

def genetic_algorithm(n, n_pop, n_gen, mut_chance):
    pop, fit = create_a_population(n, n_pop)
    m = int(len(pop)*0.4)
    avg_fitnesses = []
    for i in range(n_gen):
        elites, pool = selection(pop, fit, m)
        pop, fit, avg_fit = breed(elites, pool, mut_chance)
        avg_fitnesses.append(avg_fit)
        if (i+1)%10 == 0:
            print(str(i+1) + 'th generation complete.')  
    sort_pop = [(x,y) for x,y in reversed(sorted(zip(fit, pop)))]
    return sort_pop[0], pop, fit, np.array(avg_fitnesses)

"""    The inputs of the program are entered below.    """

# For all cities, n = coordinates.shape[0]
n = 81              # Number of cities starting from plate number 01 (Adana)
pop_size = 200     # Number of the paths in populations.
gnr_no = 1000       # number of generations to terminate the program
mut_chn = 0.05      # mutation occurance chance
# Calling the main function (genetic_algorithm)
first, pop, fit, avg_fit = genetic_algorithm(n, pop_size, gnr_no, mut_chn)
# Best fitness and the fittest (shortest) path.
best_fit, fittest = first

"""    Printing the output.    """
print('\x1b[36;40m' + str(1e6/best_fit) +'\x1b[33;40m'
    + ' km estimated for the shortest round trip.' + '\x1b[36;40m')
print(str(fittest) + '\x1b[33;40m' + ' is the shortest route.'+'\x1b[0m')

"""    Creating a GoogleMaps plot in the working directory as a .html file. """
plot_path(fittest)

"""    Plotting the Generations vs Avg. Fitness results.    """
gen_count = np.arange(1, gnr_no+1, 1)
plt.figure(figsize=(6,6))
plt.plot(gen_count, 1e6/avg_fit, '-')
plt.title('Generations vs Total Distance', loc='center', fontsize=15)
plt.ylabel('Total Distance (km)', fontsize=12)
plt.xlabel('Generation', fontsize=12)
plt.show()
