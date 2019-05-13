# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:20:05 2019

@author: farukkutlu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
def get_path(distances, n=81):
    path, l, short, ind = [5], list(range(n)), 10000, 0
    l.pop(5)
    for j in range(len(l)):
        for i in l:
            if distances[path[-1],l[l.index(i)]] < short:
                short = distances[path[-1],l[l.index(i)]]
                ind = i
        path.append(ind)
        l.pop(l.index(ind))
        short = 10000
    return path

def get_path_length(path):
    path = np.append(path,path[0])
    total_length = 0.0
    for i in range(len(path)-1):
        j, k = path[i], path[i+1]
        total_length += distances[j, k]
    return total_length

def plot_path(path):
    path = np.append(path,path[0])
    latitude_list = coordinates[:,0]
    longitude_list = coordinates[:,1]
    path_latitude = latitude_list[path]
    path_longitude = longitude_list[path]
    lon, lat = np.array(path_longitude), np.array(path_latitude)
    img = plt.imread("harita.png")
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, 1347, 0, 721])
    ax.plot(lon*70-70*25.8, lat*84-84*34, '--', linewidth=2, color='firebrick')
#    plt.savefig('MyMap.png', dpi=1200)
    plt.show()
    return None

shortest_path = get_path(distances, 81)
total_length = get_path_length(shortest_path)
print(total_length)
plot_path(shortest_path)

