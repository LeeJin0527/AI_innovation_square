# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:15:45 2020

@author: Administrator
"""

import numpy as np 
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1],rgb[:,:,2] #빨, 초, 파 빼기
    gray = 0.2989 *r + 0.5870 * g +0.1140 *b
    return gray

def conv2d(img_gray, filter_):
    filter_len = filter_.shape[0] #3
    H, W =img_gray.shape  #478,600
  
    
    img_convolved = np.zeros(shape = (H-filter_len +1, W-filter_len + 1))
    # print(img_convolved)
    for row_idx in range(H - filter_len):
        for col_idx in range(W - filter_len):
            img_segment = img_gray[row_idx : row_idx +filter_len, col_idx :col_idx +filter_len]
            convolution = np.sum(img_segment * filter_)
            img_convolved[row_idx, col_idx] = convolution
            
    return img_convolved


img = plt.imread('C:/Users/Administrator/Desktop/AI/나은.gif')
img_gray = rgb2gray(img)
# print(img_gray)

filter_ = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])

img_convolved = conv2d(img_gray, filter_)

fig , axes = plt.subplots(1, 2, figsize=(20, 12))
axes[0].imshow(img_gray, 'gray')
axes[1].imshow(img_convolved,'gray')