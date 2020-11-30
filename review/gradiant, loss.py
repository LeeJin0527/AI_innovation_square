# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:18:48 2020

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

n_point = 100

x = np.random.normal(0, 1, size =(n_point, ))
y = 3*x + 0.5*np.random.normal(0, 1, size = (n_point))

fig, axes  = plt.subplots(1, 2, figsize = (20, 10))
print(axes[0].scatter(x,y))
print(axes[0].axvline(x=0, linewidth = 3, color='black',alpha=0.5))
print(axes[0].axhline(y=0, linewidth = 3, color='black',alpha=0.5))

a = -3
n_iter = 300
learning_rate = 0.01
cmap = cm.get_cmap('rainbow', lut = n_iter)
losses = []

for i in range(n_iter):
    predictions = a*x
    mse = np.mean((y-predictions)**2)
    dl_da = -2*np.mean(x*(y-predictions))
    a = a- learning_rate *dl_da
    
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range
   
    axes[0].plot(x_range, y_range, color = cmap(i))
    losses.append(mse)

axes[1].plot(losses)
    