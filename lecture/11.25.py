# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:32:43 2020

@author: Administrator
"""

#ndarray

import numpy as np

python_list =[1, 2, 3]
ndarray = np.array(python_list)

print(python_list)
print(ndarray)

print(type(python_list))
print(type(ndarray))

#리스트 기능
print(dir(python_list))

#numpy 기능
print(dir(ndarray))

#%% ndarray 만들기 
python_list = [1, 2, 3]
ndarray = np.array(python_list)

#np.zeros()
ndarray2 = np.zeros(shape=(10,))
print(ndarray2)

ndarray3 = np.ones(shape=(10,))
print(ndarray3)

#np.full()
ndarray4 = np.full(shape=(10,),fill_value = 3.14)
print(ndarray4)


#%% # 특정값으로 초기화 하기 
ndarray5 = 3.14*np.ones(shape=(10,))
print(ndarray4)

#%% 자리만 잡고 있는 상태 
ndarray6 = np.empty(shape=(10,))
print(ndarray6)

#%% ndarray 만들기.2
tmp = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
#tmp와 똑같은 모양을 가진 벡터 생성
ndarray7 = np.zeros_like(tmp)
print(ndarray7)

ndarray8 = np.ones_like(tmp)
print(ndarray8)

ndarray10 = np.empty_like(tmp)
print(ndarray10)

#%% Matrix 만들기 

python_list = [[1, 2], [3, 4]]
ndarray1 = np.array(python_list)

# print(ndarray1)


ndarray2 = np.zeros(shape=(2,2,2))
print(ndarray2)

ndarray3 = np.ones(shape=(2,2))
print(ndarray3)

#np.full()
ndarray4 = np.full(shape=(2,2),fill_value = 3.14)
print(ndarray4)

#%% vector vs matrix
ndarray1 = np.ones(shape=(5,))
ndarray2 = np.ones(shape=(5,1))

print(ndarray1)


print(ndarray2)
#%%

ndarray4 = np.full(shape=(2,2),fill_value = 3.14)

'''shape, dtype, size , itemsize'''

print(ndarray4.shape)
print(ndarray4.dtype)
print(ndarray4.size)
print(ndarray4.itemsize)

print(ndarray4.size*ndarray4.itemsize)
#%%

import numpy as np
a = np.array([1, 2, 3])
b = np.array([5, 6, 7])

print( a + b)

dot_product = np.sum(a*b)
print(dot_product)

#%%
python_list = [ [1, 2],[ 3, 4]]
ndarray1 =np.array(python_list)

print(ndarray1)
print(ndarray1[0]) 
#첫번째 학생의 데이터를 가져와
print(ndarray1[1])
#두번째 학생의 데이터를 가져와


#%% 평균 구하기 (수학, 영어)

# scores = np.random.randint(1, 100, size=(100,3))
# # print(scores.shape)
# # print(scores)
# n_student = scores.shape[0] #100
# n_class = scores.shape[1] #3

# class1_sum, class2_sum, class3_sum = 0, 0, 0
# for scores in scores:
#     # print(score)
    # class1_val, class2_val, class3_val =score
    
#%% 평균 구하기 
scores = np.random.randint(1, 100, size=(100,3))
n_student = scores.shape[0] 
class_sum = np.zeros(shape=(3, ))
for score in scores:
    class_sum += score
class_mean = class_sum / n_student

print(class_mean)

#%% 분산구하기 

scores = np.random.randint(1, 100, size=(100,3))
n_student = scores.shape[0] 

class_sum = np.zeros(shape=(3, ))
class_sum2 = np.zeros(shape=(3, ))
for score in scores:
    class_sum += score**2
    class_sum2 += score
class_mean = class_sum / n_student
class_mean2 = (class_sum2 / n_student)**2


var = class_mean - class_mean2
print(var)

#%%
import matplotlib.pyplot as plt
n_point = 100
mse_sum = 0 
x= np.random.normal(0, 2, size=(n_point,))
y= 3*x
pred = 2*x
# print(x)

for i in range(n_point):
    mse_sum +=(y[i] - pred[i])**2
mse_error = mse_sum / n_point 
print(mse_error)
#%%API

scores = np.random.randint(1, 100, size=(100,3))
score_mean = np.mean(scores)
print(score_mean)

score_var = np.var(scores)
print(score_var)

score_std = np.std(scores)
print(score_std)

score_max = np.max(scores)
print(score_max)
score_min = np.min(scores)
print(score_min)

#max 의 index
score_argmax = np.argmax(scores)
print(score_argmax)

score_argmin = np.argmin(scores)
print(score_argmin)

#%%
scores = np.random.randint(1, 100, size=(100,3))
sum1 = np.sum(scores)
print(sum1)

sum2 = np.sum(scores, axis =0) #과목차원
sum3 = np.sum(scores, axis =1) #학생차원
print(sum2)
# print(sum3)

# print(sum2.shape)
# print(sum3.shape)

#%%
#%% 분산 구하기
scores = np.random.randint(0, 100, size=(100, 3)) 

n_student, n_class = scores.shape

scores_sum = np.zeros(shape = (scores.shape[1], ))
scores_squared_sum = np.zeros_like(scores_sum)

for score in scores:
    scores_sum += score
    scores_squared_sum += score**2

scores_mean = scores_sum / n_student
scores_variance = scores_squared_sum / n_student - scores_mean**2

print('Variance : ', scores_variance)

#%% MSE error revisited
import matplotlib.pyplot as plt

n_point = 100
x = np.random.normal(0, 2, size=(n_point,))
y = 3*x
predictions = 2*x

# answer.1
sub_square_sum = 0
for data_idx in range(n_point):
    sub_square_sum = (y[data_idx] - predictions[data_idx])**2
mse_error = sub_square_sum / n_point

# answer.2
sub_squares = (y - predictions)**2
mse_error = 0
for sub_square in sub_squares:
    mse_error += sub_square
mse_error /= n_point

# answer.3
sum_, cnt = 0, 0
for data_idx in range(n_point):
    sum_ += (y[data_idx] - predictions[data_idx])**2
    cnt += 1
mse_error = sum_ / cnt

# answer.4
sum_ = 0
for cnt, data in enumerate(range(n_point)):
    sum_ += (y[data_idx] - predictions[data_idx])**2
mse_error = sum_ / (cnt + 1)


#%% APIs
scores = np.random.randint(0, 100, size=(100, ))
score_mean = np.mean(scores)
print(score_mean)
score_mean = scores.mean()
print(score_mean)

score_var = np.var(scores)
print(score_var)
score_var = scores.var()
print(score_var)

score_std = np.std(scores)
print(score_std)
score_std = scores.std()
print(score_std)

score_max = np.max(scores)
print(score_max)
score_max = scores.max()
print(score_max)

score_min = np.min(scores)
print(score_min)
score_min = scores.min()
print(score_min)

score_argmax = np.argmax(scores)
print(score_argmax)
score_argmax = scores.argmax()
print(score_argmax)

score_argmin = np.argmin(scores)
print(score_argmin)
score_argmin = scores.argmin()
print(score_argmin)

#%% APIs for matrices
scores = np.random.randint(0, 100, size=(100, 3))

mean1 = np.mean(scores)
mean2 = np.mean(scores, axis=0)
mean3 = np.mean(scores, axis=1)

print(scores.shape)
print("axis=0: ", mean2.shape)
print("axis=1: ", mean3.shape)

print("mean1: ", mean1)
print("mean2: ", mean2)
print("mean3: ", mean3)


#%%

import numpy as np

#reshape
a = np.array([1, 2, 3, 4])
print(a.shape)

a = a. reshape(4, 1)
print(a.shape)
print(a, '\n')
a = a. reshape(1, 4)
print(a.shape)
print(a, '\n')
a = a. reshape(2, 2)
print(a.shape)
print(a, '\n')


#%% reshape + -1 value
a = np.random.uniform(0, 20, size=(20, ))

a = a.reshape((4,-1))
print(a)


a = a.reshape((-1,4))
print(a.shape)


a = a.reshape((1,-1)) 
'''to row vector'''
print(a.shape)


a = a.reshape((-1, 1)) 
'''to column vector'''
print(a.shape)
#%%
a = np.array([1, 2, 3]).reshape((1,-1))
b = np.array([10, 20, 30]).reshape((-1,1))
print(a)
print(b)

print(a + b)

#%%

#%% Broadcasting

a = np.array([1, 2, 3, 4]).reshape((1, -1))
b = np.array([10, 20, 30]).reshape((-1, 1))
print(a.shape)
print(b.shape)

c = a / b
print(c.shape)

print(a, '\n')

print(b, '\n')
print(c, '\n')

#%%
a = np.random.uniform(0, 5, size=(2, 3)).astype(np.int)
print(a, '\n')

b = np.array([1, 2]).reshape(-1, 1)
print(a + b, '\n')

b = np.array([1, 2, 3]).reshape(1, -1)
print(a + b, '\n')
#%%
#간격
a= np.arange(5, 100, 2)
print(a)

#%%
#몇개의 총 점을 만들것인지
a = np.linspace(-10, 10 , 20)
print(a)

#%%
#MSE error
import numpy as np
n_point = 100
x = np.random.normal(0, 2, size=(n_point,))
y = 3*x
predictions = 2*x

diff_square = (y-predictions)**2
mse_error = np.mean(diff_square)
print(mse_error)
#%%
scores = np.random.uniform(0,100,size=(100,))
cutoff = 80
# print(scores)
scores = scores.astype(np.int)
print(scores)
student_pass =(scores> cutoff ).astype(np.int)
print(student_pass)

pass_percentage = np.sum(student_pass)/n_student*100
print(pass_percentage, '%')
#%%

#짝수 홀수 
scores = np.random.uniform(0,100,size=(100,)).astype(np.int)

is_odds= (scores % 2).astype(np.int)
print(is_odds)

odd_percentage = np.sum(is_odds)/n_student*100
print(odd_percentage, '%')
#%% dataset 생성
n_class, std, n_point = 2, 1, 100

dataset = np.empty(shape=(0, 2))

for class_idx in range(n_class):
    centers = np.random.uniform(-3, 3, size=(2, ))
    
    x_data = np.random.normal(loc=centers[0], scale=std,
                              size=(n_point, 1))
    y_data = np.random.normal(loc=centers[1], scale=std,
                              size=(n_point, 1))
    
    data = np.hstack((x_data, y_data))
    dataset = np.vstack((dataset, data))
    
dataset = dataset

# print(dataset)
# k-means clustering/

centroids = np.random.uniform(-5, 5, size=(n_class, 2))

template = "Shape -- dataset:{}\t centroids:{}"
print(template.format(dataset.shape, centroids.shape))

for i in range(9):
    clusters = dict()
    for cluster_idx in range(n_class):
        clusters[cluster_idx] = np.empty(shape=(0, 2))
    
    for data in dataset:
        data = data.reshape(1, -1)
        
        distances = np.sum((data - centroids)**2, axis=1)
        min_idx = np.argmin(distances)
        
        clusters[min_idx] = np.vstack((clusters[min_idx], data))
    
    for cluster_idx in range(n_class):
        cluster = clusters[cluster_idx]
        centroid = np.mean(cluster, axis=0)
        centroids[cluster_idx] = centroid

#%%
#과제 
#cross- correlation

 
