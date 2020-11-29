# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:48:02 2020

@author: Administrator
"""

#%%sigmoid ,tanh, relu
import numpy as np
import matplotlib.pyplot as plt


#np.exp ,np.maximum
affine = np.array([-5, 2, 6, 8, 1 ])


def sigmoid(affine): return 1 / (1 + np.exp(-affine))
def tanh(affine): return (np.exp(affine) - np.exp(-affine)) / \
    (np.exp(affine) + np.exp(-affine))
def relu(affine) : return  np.maximum(0,affine)


x_range = np.linspace(-5, 5, 300)
a_sigmoid = sigmoid(x_range)
a_tanh = tanh(x_range)
a_relu = relu(x_range)

plt.style.use('seaborn')
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(x_range, a_sigmoid, label='Sigmoid')
ax.plot(x_range, a_tanh, label='Tanh')
ax.plot(x_range, a_relu, label='ReLU')

ax.tick_params(labelsize=20)
ax.legend(fontsize=40)

