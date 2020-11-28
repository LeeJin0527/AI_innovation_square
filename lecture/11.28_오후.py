# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:23:59 2020

@author: Administrator
"""

import numpy as np 
import matplotlib.pyplot as plt

n_point = 100

x = np.random.normal(0, 1, size = n_point)
y = 3*x +0.2*np.random.normal(0, 1, size = n_point)

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x,y)
ax.axvline(x=0, linewidth = 3 , color='black',alpha=0.5)
ax.axvline(y=0, linewidth = 3 , color='black',alpha=0.5)

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

n_point = 100

x = np.random.normal(0, 1, size=(n_point, ))
y = 3*x + 0.*np.random.normal(0, 1, size=(n_point, ))

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x, y)
ax.axvline(x=0, linewidth=3, color='black', alpha=0.5)
ax.axhline(y=0, linewidth=3, color='black', alpha=0.5)

a = -10
n_iter = 200
learning_rate = 0.01
cmap = cm.get_cmap('rainbow', lut=n_iter)
for i in range(n_iter):
    predictions = a*x
    dl_da = -2*np.mean((y-predictions)*x)
    a = a- learning_rate*dl_da
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range
    ax.plot(x_range, y_range, color=cmap(i))

#%%
n_point = 100
x = np.random.normal(0, 1, size=(n_point, ))
x_noise = x + 0.2*np.random.normal(0, 1, size=(n_point, ))
    
y = (x_noise >= 0).astype(np.int)

fig, ax = plt.subplots(figsize=(20, 10))
ax.scatter(x, y)

a, b = np.random.normal(0, 1, size=(2, ))
n_iter = 1000
learning_rate = 0.1
cmap = cm.get_cmap('rainbow', lut=n_iter)
losses = []
for i in range(n_iter):
    
    

    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range + b
    y_range = 1/(1 + np.exp(-y_range))
    ax.plot(x_range, y_range, color=cmap(i))

    losses.append(mse)

ax.grid()
#%%
n_point = 100
x = np.random.normal(0, 1, size=(n_point, ))
x_noise = x + 0.2*np.random.normal(0, 1, size=(n_point, ))
    
y = (x_noise >= 0).astype(np.int)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].scatter(x, y)

a, b = np.random.normal(0, 1, size=(2, ))
n_iter = 1000
learning_rate = 0.1
cmap = cm.get_cmap('rainbow', lut=n_iter)
losses = []
for i in range(n_iter):
    affine = a*x + b
    predictions = 1/ (1+ np.exp(-affine))
    bce = - (y*np.log(predictions)+(1-y)*np.log(1-predictions))
    #업데이트 필요 
    a = a- 
    b = b - alqns
    

    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range + b
    y_range = 1/(1 + np.exp(-y_range))
    axes[0].plot(x_range, y_range, color=cmap(i))

    losses.append(bce)

axes[0].grid()
axes[1].plot(losses)

#%%리니어리그레션, 로지스틱리그레션