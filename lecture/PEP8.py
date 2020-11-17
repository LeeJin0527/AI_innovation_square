# -*- coding: utf-8 -*-

math_score = 10 #PEP 8 Naming convention

SCORE = 10 #constant

#Dynamic Typing
score = 10 #주석
print(score) 
score = 10.53
print(score) 
score = 'Hello'
print(score) 

#%%  #셀 나누기
print("hello hello hello ")

#%%  연산자(대입연산자)

a = 10
b = 20 
c = 10.53
d = 'jin'

a,b = 10, 20

#%% #사칙 연산자

a = 20
b = 5

print(a + b)
print(a - b)
print(a * b)
print(a / b)

print(a // b)
print(a % b)

# %%

print(a.__add__(b))
print(int.__add__(a,b))

#%%
score1, score2, score3, score4 = 50, 40, 60 ,30 

mean =(score1 +score2 +score3+score4 )/4
#print(mean)

variance =(score1**2 +score2**2 +score3**2+score4**2)/4- mean**2

print(variance)
#분산




#%%
#Mean Squared Error 구하기
y1, y2, y3, y4 =10, 20 ,30 ,40
p1, p2, p3, p4 =40, 30, 20 ,40

diff= (y1 - p1)**2 +(y2 -p2)**2+(y3 -p3)**2 +(y4 -p4)**2
num = 4

mse_error= diff / num
print(mse_error)





 #%% 복합대입연산자
 
a = 0
a += 1
a /= 1

 #%%
test_list =[]
for i in range(10):
    test_list.append(2*i)
print(test_list)
    
#%% 이게 더 의미 있음 
#의미 전달이 중요함 
test_list = [i**2 for i in range(10)]
print(test_list)
#%%
#동전 손상==> 5:5 깨짐
#추측을 잘했다: 수치로 나타내야함 
#cross entropy :두개의 확률비교하는 도구
#실제 정답이랑 추측한거랑 얼마나 차이

import math 

y = 0.7 #정답
p = 0.5 #내가 예측한 값

loss = -(y*math.log(p) + (1-y)*math.log(1-p))
print(loss) 

p=0.6
loss = -(y*math.log(p) + (1-y)*math.log(1-p))
print(loss)
p=0.7
loss = -(y*math.log(p) + (1-y)*math.log(1-p))
print(loss)

#%% List => 값들을 한번에 들고 다니기 위해서 개발됨


scores = [20, 40, 50, 60, 10]
#%% indexing
print(scores[0])
print(scores[1])
print(scores[2])
print(scores[3])
print(scores[4])

#%%
print(scores[-1])
print(scores[-2])
#%% 빈 리스트 만들기 
test_list = []
print(test_list)

test_list2 = list()
print(test_list2) #추천

#%%
#중요 값 수정하기 
'''mutable object'''

scores = [20, 40, 50, 60, 10]
print(scores)
scores[0] = 100
print(scores)

#%% immutable
test_tuple = (1, 2, 3, 4, 5)
print(test_tuple[0])
test_tuple[0] = 100
print(test_tuple) #수정불가

#%%
scores = [50, 40, 60, 30]
scores2 = [60, 50, 70, 40]

n_data = 4
score_sum = 0
score_sum1 = 0

var_sum = 0
var_sum1 = 0

add = 10

for score in scores:
    score_sum += score
   
for score in scores2:
    score_sum1 += score

ave= score_sum / n_data
ave2 = score_sum1 /n_data

print(ave)
print(ave2)

    

    
for score in scores:
    var_sum += score**2   
    
variance = var_sum/n_data - ave**2

print(variance)


for score in scores2:
    var_sum1 += score**2

variance2 = var_sum1/n_data - ave2**2
print(variance2)

#for문은 프로그램을 느리게 만든다 
