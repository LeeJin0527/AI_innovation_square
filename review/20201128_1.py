# 평균, 분산 구하기(국어, 수학, 영어)

# M, m, M_idx, m_idx

# MSE error revisited

# 2D convolution

# BCE(integer format, one-hot vector format)

# sigmoid, tanh, ReLU


''' 수학 
1. derivatives
2. chain rule
3. linear regression
4. logitstic regression
'''
# linear regression

# logistic regression


#%%
import numpy as np


def get_random_scores(n_student):
    scores = np.random.uniform(low=0, high=100.0, size=(n_student, ))
    scores = scores.astype(np.int)
    return scores


def get_mean(scores):
    sum_ = 0
    for cnt, score in enumerate(scores):
        sum_ += score
    mean = sum_ / (cnt+1)
    return mean
    
    
def get_variance(scores):
    mean = get_mean(scores)
    
    square_sum = 0
    for cnt, score in enumerate(scores):
        square_sum += score**2
        
    variance = square_sum/(cnt+1) - mean**2
    return variance


def get_mean_variance(scores):
    scores_sum = 0
    scores_squared_sum = 0
    
    for score in scores:
        scores_sum += score
        scores_squared_sum += score**2
        
    scores_mean = scores_sum / n_student
    scores_var = (scores_squared_sum / n_student) - (scores_mean**2)
    return scores_mean, scores_var


n_student = 100
scores = get_random_scores(n_student)
mean = get_mean(scores)
var = get_variance(scores)

mean, var = get_mean_variance(scores)

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

print(colored('Hello World!', 'red', 'on_white', attrs=['blink']))

template = colored('[INFO]', 'cyan', attrs=['blink'])
print(template + ' Dataset is Loading')

template = colored('[INFO]', 'cyan', attrs=['blink'])
print(template + ' Model is Loading')

#%%
from tqdm import tqdm
import time

template = colored('[INFO]', 'cyan', attrs=['blink'])
print(template + ' Dataset is Loading')
for i in tqdm(range(10000)):
    time.sleep(0.01)

#%% MSE error revisited

n_point = 100
ground_truths = np.random.normal(0, 1, (n_point, ))
predictions = np.random.normal(0, 1, (n_point, ))

def get_mse(ground_truths, predictions)
    mse_error = np.mean((ground_truths - predictions)**2)
    return mse_error

#%% 
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
    
img = plt.imread('./test_image.jpg')
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
#%% BCE(integer format, one-hot vector format)
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

#%% sigmoid, tanh, relu
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



































