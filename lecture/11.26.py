# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:21:32 2020

@author: Administrator
"""

import numpy as np

### list slicing

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(a)

b = a[1:7:2]
print(b)


#%% NumPy Indexing

a = np.random.uniform(0, 10, (3, 3)).astype(np.int)
print(a, '\n')
print(a[0,0], a[0, 1], a[0, 2])
print(a[1,0], a[1, 1], a[1, 2])
print(a[2,0], a[2, 1], a[2, 2], '\n')

for row_idx in range(a.shape[0]):
    for col_idx in range(a.shape[1]):
        print(a[row_idx, col_idx])
#%% NumPy Slicing
a = np.random.uniform(0, 10, (5, 5)).astype(np.int)
print(a, '\n')

print(a[0 , 0:4])
print(a[1 , 0:4])
print(a[0:2, 0:2])
print(a[0,:])
print(a[:,0])

#%% 1D Correlation
signal = np.random.normal(0, 1, (100, ))
filter_ = np.array([1, 5, 3, 2, 1])

n_signal, n_filter = signal.shape[0], filter_.shape[0]

correlations = np.empty(shape=(0, 0))
for co_idx in range(n_signal - n_filter):
    signal_segment = signal[co_idx : co_idx + n_filter]
    correlation = np.sum(signal_segment * filter_)
    correlations = np.hstack((correlations, correlation))

#%% 1D vs 2D Correlations

signal_segment = np.random.uniform(0, 10, (9,)).astype(np.int)
filter_ = np.random.uniform(0, 10, (9,)).astype(np.int)

print(signal_segment)
print(filter_)

correlation = np.sum(signal_segment * filter_)


signal_segment = signal_segment.reshape(3, 3)
filter_ = filter_.reshape(3, 3)
print(signal_segment)
print(filter_)

correlation = np.sum(signal_segment * filter_)
#%%
filter_ = np.ones(shape=(3, 3)) /9
l_filter = filter_.shape[0]

H, W = img_gray.shape
img_convolved = np.zeros(shape=(H-l_filter+1, W-l_filter+1))

for row_idx in range(img_gray.shape[0] - l_filter):
    for col_idx in range(img_gray.shape[1] - l_filter):
        img_segment = img_gray[row_idx:row_idx+l_filter,
                               col_idx:col_idx+l_filter]
        convolution = np.sum(img_segment * filter_)
        img_convolved[row_idx, col_idx] = convolution


fig, axes = plt.subplots(1, 3, figsize=(25, 12))
axes[0].imshow(img_gray, 'gray')
axes[1].imshow(img_convolved, 'gray')
#%%

filter_ = np.ones(shape=(11, 11)) /121
l_filter = filter_.shape[0]

H, W = img_gray.shape
img_convolved = np.zeros(shape=(H-l_filter+1, W-l_filter+1))

for row_idx in range(img_gray.shape[0] - l_filter):
    for col_idx in range(img_gray.shape[1] - l_filter):
        img_segment = img_gray[row_idx:row_idx+l_filter,
                               col_idx:col_idx+l_filter]
        convolution = np.sum(img_segment * filter_)
        img_convolved[row_idx, col_idx] = convolution
axes[2].imshow(img_convolved, 'gray')
fig.tight_layout()
#%%
filter_ = np.ones(shape=(3, 3))/9

l_filter = filter_.shape[0]

H, W = img_gray.shape
img_convolved = np.zeros(shape=(H-l_filter+1, W-l_filter+1))

for row_idx in range(img_gray.shape[0] - l_filter):
    for col_idx in range(img_gray.shape[1] - l_filter):
        img_segment = img_gray[row_idx:row_idx+l_filter,
                               col_idx:col_idx+l_filter]
        convolution = np.sum(img_segment * filter_)
        img_convolved[row_idx, col_idx] = convolution

fig, axes = plt.subplots(1, 2, figsize=(25, 12))
axes[0].imshow(img_gray, 'gray')
axes[1].imshow(img_convolved, 'gray')
