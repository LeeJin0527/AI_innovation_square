# %% sorting

# answer.1
a = [50, 30, 10, 60, 20, 70, 15]
buffer = None

for first_index in range(len(a)):
    for second_index in range(first_index+1, len(a)):
        if a[first_index] > a[second_index]:
            a[first_index], a[second_index] = \
                a[second_index], a[first_index]
print(a)

# answer.2
a = [50, 30, 10, 60, 20, 70, 15]

sorted_list = list()
tmp_unsorted_list = a
n_numbers = len(a)

for iteration in range(n_numbers):
    tmp_min = None
    tmp_rest_list = []
    
    for element in tmp_unsorted_list:
        if tmp_min == None or element < tmp_min:
            tmp_min = element
    
    for element in tmp_unsorted_list:
        if element == tmp_min:
            sorted_list.append(tmp_min)
        else:
            tmp_rest_list.append(element)
            
    tmp_unsorted_list = tmp_rest_list
    
print('unsorted list : \n', a)
print('sorted list   : \n', sorted_list)

#%% 평균은 넘겠지
import numpy as np
import matplotlib.pyplot as plt


scores = np.random.normal(loc=50, scale=10, size=(100, ))
n_student = len(scores)

score_sum = 0
for score in scores:
    score_sum += score
mean = score_sum / n_student

over_cnt = 0
for score in scores:
    if score > mean:
        over_cnt += 1
over_percentage = over_cnt / n_student * 100

print(str(over_percentage) + '%')

#%% Vector Additions
a = [1, 2, 3]
b = [4, 5, 6]
c = list()

for index in range(len(a)):
    c.append(a[index] + b[index])

print(c)

#%% Hadamard Product
a = [1, 2, 3]
b = [4, 5, 6]
c = list()

for index in range(len(a)):
    c.append(a[index] * b[index])

print(c)

#%% matrix multiplication

A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
B = [[10, 20], [30, 40], [50, 60]]

C = [[0, 0], [0, 0], [0, 0]]

n_row = len(A)
n_col = len(B[0])
n_iter = len(A[0])

row_idx, col_idx = 0, 1

for row_idx in range(n_row):
    for col_idx in range(n_col):
        for i in range(n_iter):
            C[row_idx][col_idx] += A[row_idx][i]*B[i][col_idx]

print(C)

import numpy as np
A = np.array(A)
B = np.array(B)

C = np.matmul(A, B)
print(C)

#%% dot product

a = [1, 2, 3]
b = [4, 5, 6]

dot_product = 0
for i in range(len(a)):
    dot_product += a[i]*b[i]

print(dot_product)

#%% vector norm
import math

a =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 19]

square_a_norm = 0
for element in range(len(a)):
    square_a_norm += a[element] ** 2

a_norm = math.sqrt(square_a_norm) 

print(a_norm)
    
#%% unit vector

import math

a = [3, 4, 5]

square_sum = 0
for element in a:
    square_sum += element**2

vector_norm = math.sqrt(square_sum)

unit_vectors = []
for element in a:
    unit_vector = element / vector_norm
    unit_vectors.append(unit_vector)
    
print(unit_vectors)

#%% dot product with unit vectors
a = [20, 80]
b = [5, 20]

c = [10000, 100]

square_sum = 0
for element in a:
    square_sum += element**2

vector_norm = math.sqrt(square_sum)

a_unit_vector = []
for element in a:
    element = element / vector_norm
    a_unit_vector.append(element)
    
square_sum = 0
for element in b:
    square_sum += element**2

vector_norm = math.sqrt(square_sum)

b_unit_vector = []
for element in b:
    element = element / vector_norm
    b_unit_vector.append(element)

square_sum = 0
for element in c:
    square_sum += element**2

vector_norm = math.sqrt(square_sum)

c_unit_vector = []
for element in c:
    element = element / vector_norm
    c_unit_vector.append(element)

'''
a_unit_vector, b_unit_vector, c_unit_vector
'''

dp_ab = 0
for i in range(len(a_unit_vector)):
    dp_ab += a_unit_vector[i]*b_unit_vector[i]

dp_ac = 0
for i in range(len(a)):
    dp_ac += a_unit_vector[i]*c_unit_vector[i]

dp_bc = 0
for i in range(len(a)):
    dp_bc += b_unit_vector[i]*c_unit_vector[i]

print("dp_ab", dp_ab)
print("dp_ac", dp_ac)
print("dp_bc", dp_bc)

#%% 

a = [10, 50, 20, 30, 10]
a.sort()
print(a)

#%% Special Method

a = 10
b = 20

print(a + b)
print(a.__add__(b))

print(a - b)
print(a.__sub__(b))

print(a * b)
print(a.__mul__(b))

# %%
class RGB_pixel:
    def __init__(self, rgb):
        self.tmp = rgb
        
    def __add__(self, operand2):
        return "Hello World!"

        
pixel1 = RGB_pixel([100, 150, 20])
pixel2 = RGB_pixel([50, 30, 80])
print(pixel1 + pixel2)

#%% print 함수 사용하기

a, b = 10, 20
# print(a, b)

# string + int
print("a: ", a)
print("b: ", b)

# escape character
print(a, '\n', b)
print(a, '\t', b)

# string + integer
print(str(a) + ' Hello')

#%% string formatting
template = 'format value: {}'.format(10)
print(template)
template = 'format value: {}'.format("Hello")
print(template)

#%%

template = 'format value: {:d}'.format(10)
print(template)

template = 'format value: {:f}'.format(10.3234)
print(template)

template = 'format value: {:s}'.format("Hello")
print(template)

template = 'format value: {:x}'.format(15)
print(template)
template = 'format value: {:X}'.format(15)
print(template)

# %% padding

template = 'format value: {:10d}'.format(13847)
print(template)

# %% padding + alignment

template = 'format value: {:<10d}'.format(13847) # 왼쪽 정렬
print(template)

template = 'format value: {:^10d}'.format(13847) # 가운데 정렬
print(template)

template = 'format value: {:>10d}'.format(13847) # 오른쪽 정렬
print(template)

#%% floating point
template = 'format value: {:^10.3f}'.format(3.141592)
print(template)

#%% 
template = 'A: {:^10d} B: {:^10.3f}'.format(10, 20.321432)
print(template)

# %%
template = 'A: {0} B: {1}'.format(10, 20)
print(template)

#%%
template = '{name}: {age}'.format(name='ShinKS', age=29)
print(template)

#%%
template = 'Name: {:^8s}\t Age: {:^4d}\t Height: {:^5.2f}'
print(template.format('Shin', 29, 195.5432432))


































