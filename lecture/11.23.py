# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:39:48 2020

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
#센트로이드를 두묶음으로 나누다 =>2개필요
#리스트 어펜트 

n_class = 10
std = 1
n_point = 100

dataset = np.empty(shape=(0, 2))
    
for class_idx in range(n_class):
    centers = np.random.uniform(-10, 10, size=(2, ))
    
    x_data = np.random.normal(loc=centers[0], scale=std,
                              size=(n_point, 1))
    y_data = np.random.normal(loc=centers[1], scale=std,
                              size=(n_point, 1))
    
    data = np.hstack((x_data, y_data))
    dataset = np.vstack((dataset, data))
#%%
import numpy as np
import matplotlib.pyplot as plt

def make_dataset(n_class, std, n_point):
    dataset = np.empty(shape=(0, 2))
    
    for class_idx in range(n_class):
        centers = np.random.uniform(-10, 10, size=(2, ))
        
        x_data = np.random.normal(loc=centers[0], scale=std,
                                  size=(n_point, 1))
        y_data = np.random.normal(loc=centers[1], scale=std,
                                  size=(n_point, 1))
        
        data = np.hstack((x_data, y_data))
        dataset = np.vstack((dataset, data))
    return dataset

n_class, std, n_point = 10, 1, 100

dataset = make_dataset(n_class, std, n_point)
dataset = dataset.tolist()

#%%
# dataset generation
n_class, std, n_point = 2, 1, 100

dataset = np.empty(shape=(0, 2))

for class_idx in range(n_class):
    centers = np.random.uniform(-10, 10, size=(2, ))
    
    x_data = np.random.normal(loc=centers[0], scale=std,
                              size=(n_point, 1))
    y_data = np.random.normal(loc=centers[1], scale=std,
                              size=(n_point, 1))
    
    data = np.hstack((x_data, y_data))
    
    dataset = np.vstack((dataset, data))
  
dataset = dataset.tolist()
centroid = np.random.uniform(-5, 5, size=(n_class, 2)).tolist()
#%%
#거리를 구한다
# print(dataset)
# print(centroid)
centroid1 =centroid[0]
centroid2 =centroid[1]
cluster1, cluster2 =[], []
for data in dataset:
    x, y =data
    distance1 = (centroid1[0]- x_data)**2 + (centroid1[1] - y_data)**2
    distance2 = (centroid2[0] - x_data)**2 + (centroid2[1] - y_data)**2

    if distance1 < distance2:
        cluster1.append(data)
    else:
        cluster2.append(data)
        
print(len(cluster1))
print(len(cluster2))
#%%

import numpy as np
import matplotlib.pyplot as plt

def make_dataset(n_class, std, n_point):
    dataset = np.empty(shape=(0, 2))
    
    for class_idx in range(n_class):
        centers = np.random.uniform(-5, 5, size=(2, ))
        
        x_data = np.random.normal(loc=centers[0], scale=std,
                                  size=(n_point, 1))
        y_data = np.random.normal(loc=centers[1], scale=std,
                                  size=(n_point, 1))
        
        data = np.hstack((x_data, y_data))
        dataset = np.vstack((dataset, data))
    return dataset

n_class, std, n_point = 2, 1, 100

dataset = make_dataset(n_class, std, n_point)
dataset = dataset.tolist()

# print(dataset)
centroid = np.random.uniform(-10, 10, size=(n_class, 2)).tolist()
centroid1 = centroid[0]
centroid2 = centroid[1]


# print(dataset)
for i in range(2):

    cluster1, cluster2 =[], []
    
    for data in dataset:
        x, y = data
        distance1 =(centroid1[0] - x)**2 + (centroid1[1] -y)**2
        distance2 =(centroid2[0] - x)**2 + (centroid2[1] -y)**2
        
        if distance1 < distance2 :  #centroid1
            cluster1.append(data)
        else:
            cluster2.append(data)   #centroid2
    
    cnt = 0  
    x_sum, y_sum = 0, 0  
    # print(len(cluster2))
    for i in cluster1:
        x, y = i
        
        x_sum += x
        y_sum += y
        cnt += 1
    # print(cnt)
    
    centroid1[0] = x_sum / cnt
    centroid1[1] = y_sum / cnt
    print(centroid1)
            
    cnt = 0  
    x_sum, y_sum = 0, 0  
    # print(len(cluster2))
    for i in cluster2:
        x, y = i
        
        x_sum += x
        y_sum += y
        cnt += 1
    # print(cnt)
   
    
    centroid2[0] = x_sum / cnt
    centroid2[1] = y_sum / cnt    
    print(centroid2)
        
    
    
    