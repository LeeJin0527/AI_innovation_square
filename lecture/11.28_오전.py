# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 09:32:24 2020

@author: Administrator
"""

import numpy as np 
def get_random_scores(n_student):
    scores = np.random.uniform(low = 0 ,high = 100.0, size=(n_student,))
    scores = scores.astype(np.int)
    
    return scores

#클래스와 함수 구분
def test_function():
    pass


class TestClass:
    pass

n_student= 100
scores = get_random_scores(n_student)

#%%
import numpy as np

 
def get_random_scores(n_student):
    scores = np.random.uniform(low = 0 ,high = 100.0, size=(n_student,))
    scores = scores.astype(np.int)
    
    return scores

#유지보수면에서 좋음
def get_mean(scores):
    sum_ = 0
    for cnt, score in enumerate(scores):
        sum_ += score
    mean = sum_ /(cnt+1)
    return mean


def get_variance(scores):
    mean = get_mean(scores)
    
    square_sum = 0
    for cnt, score in enumerate(scores):
        square_sum += score**2
    variance = square_sum /(cnt+1) -mean**2
    return variance


#실행시간이 빠름
def get_mean_variance(scores):
    scores_sum = 0
    scores_square_sum = 0
    
    for score in scores:
        scores_sum += score
        scores_square_sum += score**2
        
        scores_mean = scores_sum / n_student
        scores_var = (scores_square_sum / n_student) - (scores_mean**2)
        return scores_mean ,  scores_var
    

#리턴값을 전체 다 넣는다 
mean = get_mean(scores)

variance = get_variance(scores)
mean, var = get_mean_variance(scores)
#%%
n_student = 100
scores = get_random_scores(n_student)
max_ = scores.max()
min_ = scores.min()
max_idx = scores.argmax()

min_idx = scores.argmin()

print(max_)
print(min_)
print(max_idx)
print(min_idx)

#%%
n_student = 100
scores = get_random_scores(n_student)
def get_M_m(scores, M, m):
    max_,min_ = None, None
    
    for idx, score in enumerate(n_student):
        if max_ == None or max_ < score:
            max_idx = idx
            max_ = score
        
        if min_ == None or min_ < score:
            min_idx = idx
            min_ = score

        
    return max_, max_idx, min_, min_idx


#%%
n_student = 100
scores = get_random_scores(n_student)

def get_M_m(scores, M, m):
    max_score, min_score = None, None
    for idx, score in enumerate(scores):
        if max_score == None or max_score < score:
            max_idx = idx
            max_score = score
        
        if min_score == None or min_score > score:
            min_idx = idx
            min_score = score
    
    if M == True and m == True:
        return max_score, max_idx, min_score, min_idx
    if M == True and m == False:
        return max_score, max_idx
    if M == False and m == True:
        return min_score, min_idx

M, M_idx, m, m_idx = get_M_m(scores, M=True, m=True)
M, M_idx = get_M_m(scores, M=True, m=False)
m, m_idx = get_M_m(scores, M=False, m=True)

#%%
def calculator(input1, input2, operand):
    if operand == '+':
        return input1 + input2
    elif operand == '-':
        return input1 - input2
    elif operand == '*':
        return input1 * input2
    elif operand == '/':
        return input1 / input2
    else:
        print("Unknown Operand")

result = calculator(10, 20, '+')
result = calculator(10, 20, '-')
result = calculator(10, 20, '*')
result = calculator(10, 20, '/')
result = calculator(10, 20, '^')

#%%
from termcolor import colored
print(colored('Hello world','red','on_white',attrs=['blink']))

#%%
from termcolor import colored

print(colored('Hello World!', 'red', 'on_white', attrs=['blink']))

template = colored('[INFO]', 'cyan', attrs=['blink'])
print(template + ' Dataset is Loading')

template = colored('[INFO]', 'cyan', attrs=['blink'])
print(template + ' Model is Loading')
#%%
from tqdm import tqdm
import time

for i in tqdm(range(10000)):
    time.sleep(0.01)

#%%
n_point = 100
ground_truth = np.random.normal(0,1,(n_point,))
predictions= np.random.normal(0,1,(n_point,))


def get_mse(ground_truth, predictions):
    mse  = np.mean((ground_truth -predictions) **2) 
    print(mse)

#%%
import numpy as np
import matplotlib.pyplot as plt

# def rgb2gray(rgb):
#     r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#     gray = 0.2989 *r +0.5870 * g +0.1140 *b
#     return gray
# img = plt.imread('C:/Users/Administrator/.spyder-py3/마카롱.jpg')
# img_gray = rgb2gray(img)

# filter_ = np.ones(shape=(11, 11))
# filter_ = filter_ / filter_.size

# fig, ax = plt.subplots(figsize =(7, 12))
# ax.imshow(img_gray, 'gray')

#2dconvolution
import numpy as np
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def conv2d(img_gray, filter_):
    filter_len = filter_.shape[0]
    H, W = img_gray.shape
    
    img_convolved = np.zeros(shape=(H - filter_len + 1, W - filter_len + 1))
    
    for row_idx in range(H - filter_len):
        for col_idx in range(W - filter_len):
            img_segment = img_gray[row_idx : row_idx + filter_len, 
                                   col_idx : col_idx + filter_len]
            convolution = np.sum(img_segment * filter_)
            img_convolved[row_idx, col_idx] = convolution

    return img_convolved

img = plt.imread('C:/Users/Administrator/.spyder-py3/마카롱.jpg')
img_gray = rgb2gray(img)

filter_ = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])

img_convolved = conv2d(img_gray, filter_)

fig, axes = plt.subplots(1, 2, figsize=(20, 12))
axes[0].imshow(img_gray, 'gray')
axes[1].imshow(img_convolved, 'gray')

#%%
def get_random_scores(n_student, n_subject):
    scores = np.random.uniform(low=0, high=100.0, size=(n_student, n_subject))
    scores = scores.astype(np.int)
    return scores

def get_mean(scores):
    sum_ = 0
    for cnt, score in enumerate(scores):
        sum_ += score
    mean = sum_ / (cnt+1)
    return mean


def get_mean(scores):
    
    mean = np.mean(scores, axis = 0)
    return mean

#%%
def get_mean(scores):
    # scores: 1-D(Vector) or 2-D(Matrix)
    if len(scores.shape) == 1: # vector
        mean = np.mean(scores)
    else: # matrix
        mean = np.mean(scores, axis = 0)
    return mean


def get_mean(scores):
    mean = np.mean(scores, axis=0)
    return mean


def get_mean(scores):
    sum_ = 0
    for cnt, score in enumerate(scores):
        sum_ += score
    mean = sum_ / (cnt+1)
    return mean

scores = get_random_scores(100, 5)
print(scores.shape)
mean = get_mean(scores)
print(mean)
#%%
labels = np.array([1, 0, 1, 0, 1, 0, 1])
predictions = np.array([[0.9, 0.1],
                        [0.1, 0.9],
                        [0.8, 0.2],
                        [0.7, 0.3],
                        [0.9, 0.1],
                        [0.4, 0.6],
                        [0.3, 0.7]])

# labels = np.array([[0, 1], [1, 0],
#                    [0, 1], [1, 0],
#                    [0, 1], [1, 0],
#                    [0, 1]])
# predictions = np.array([[0.9, 0.1],
#                         [0.1, 0.9],
#                         [0.8, 0.2],
#                         [0.7, 0.3],
#                         [0.9, 0.1],
#                         [0.4, 0.6],
#                         [0.3, 0.7]])



# bce_loss = -1*np.mean((labels*np.log(predictions)))
# print(bce_loss)


#%%
def get_bce(labels, predictions):
    if len(labels.shape) == 1:
        tmp_list = []
        for label in labels:
            if label == 0:
                tmp_list.append([1, 0])
            elif label == 1:
                tmp_list.append([0, 1])
        labels = np.array(tmp_list)
    
    losses = -1*np.sum(labels*np.log(predictions), axis=1)
    loss = np.mean(losses)
    return loss
labels = np.array([1, 0, 1, 0, 1, 0, 1])
predictions = np.array([[0.9, 0.1],
                        [0.1, 0.9],
                        [0.8, 0.2],
                        [0.7, 0.3],
                        [0.9, 0.1],
                        [0.4, 0.6],
                        [0.3, 0.7]])
bce = get_bce(labels, predictions)
print(bce)

labels = np.array([[0, 1], [1, 0],
                   [0, 1], [1, 0],
                   [0, 1], [1, 0],
                   [0, 1]])
predictions = np.array([[0.9, 0.1],
                        [0.1, 0.9],
                        [0.8, 0.2],
                        [0.7, 0.3],
                        [0.9, 0.1],
                        [0.4, 0.6],
                        [0.3, 0.7]])
bce = get_bce(labels, predictions)
print(bce)
#%%sigmoid, tanh, relu
affine = np.array([-5, 2, 6, 8, 1])

sigmoid = 1/ (1+ np.exp(-affine))

print(sigmoid)

#%%
def sigmoid(affine): return 1/ (1+ np.exp(-affine))
def tanf(affine) : return (np.exp(affine) - np.exp(-affine)) / (np.exp(affine) + np.exp(-affine)) 
def relu(affine) : return np.maximum(0,affine)
print(relu)


#%%
import matplotlib.pyplot as plt

# np.exp, np.maximum
affine = np.array([-5, 2, 6, 8, 1])

def sigmoid(affine): return 1/(1 + np.exp(-affine))
def tanh(affine): return (np.exp(affine) - np.exp(-affine)) / \
    (np.exp(affine) +  np.exp(-affine))
def relu(affine) : return np.maximum(0,affine)

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

