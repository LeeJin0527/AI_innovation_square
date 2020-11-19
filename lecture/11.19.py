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
#%%
#%% 3의 배수들의 합,평균을 구하기 

three_mid_sum, three_cnt = 0, 0

for i in range(100):
    if i % 3 ==0:
        three_mid_sum += i
        three_cnt += 1
    else:
        continue

print("3의 배수의 합",three_mid_sum)
print("3의 배수의 평균",three_mid_sum/three_cnt)


#%% 3으로 나눴을 때 0인 값, 1인 값 , 2인 값 들의 총합과 평균 
three_mid_sum, three_cnt = 0, 0
two_mid_sum, two_cnt = 0, 0
one_mid_sum, one_cnt = 0, 0
for i in range(100):
    if i % 3 ==0:
        three_mid_sum += i
        three_cnt += 1
    elif i% 3 == 2:
        two_mid_sum += i
        two_cnt += 1
    else:
        one_mid_sum += i
        one_cnt += 1
        
        
print("3의 배수의 합",three_mid_sum)
print("3의 배수의 평균",three_mid_sum/ three_cnt)
print('\n')
print("3으로 나눴을 때 나머지가 1인 값 합",two_mid_sum)
print("3으로 나눴을 때 나머지가 1인 값 평균",two_mid_sum/ two_cnt)
print('\n')
print("3으로 나눴을 때 나머지가 2인 값 합", one_mid_sum)
print("3으로 나눴을 때 나머지가 2인 값 평균", one_mid_sum/ one_cnt)
print('\n')



#%%
#초를 넣었을 때 분과 초를 출력

second = 120
minute = second / 60
second1 = second % 60
if second > 60:
    print(int(minute),"분", second1,"초")

else:
    print(second,"초")
#%%

second = 3129478932

hour = second //60 **2
minute = second // 60
sec = second % 60

print (hour, minute, sec)


#%%
#사분면 코드 만들기 

point_coordinate = [1, 0]

if point_coordinate[0] > 0 and point_coordinate[1] > 0:
    
    print("제 1사분면")
    
elif point_coordinate[0] < 0 and point_coordinate[1] > 0:
    print("제 2사분면")
    
elif point_coordinate[0] < 0 and point_coordinate[1] < 0:
    print("제 3사분면")
    
elif point_coordinate[0] > 0 and point_coordinate[1] < 0:
    print("제 4사분면")
    
elif  point_coordinate[1] == 0 and point_coordinate[0]:
    print("x축")
    
elif  point_coordinate[0] == 0 and point_coordinate[1]:
    print("y축")

else:
    print("원점")
    


#%%
# answer.1
second = 3601

hour = second // 3600
minute = (second - hour*3600) // 60
second = (second - hour*3600) % 60

print(hour, '시간', minute, '분', second, '초')

# answer.2
input_second = 1234567890
counted_hour = input_second //3600
counted_minute = (input_second - counted_hour * 3600) // 60
counted_second = input_second - counted_hour * 3600 - counted_minute * 60
print(counted_hour ,'시간 ',counted_minute,'분 ',counted_second ,'초 ' )
# answer.3
time_in_sec = 30

result_time = [0, 0, 0] # [hour, min, sec]

time_min, result_sec = time_in_sec // 60, time_in_sec % 60
result_time[-1] = result_sec

if time_min >= 60:
    result_hour, result_min = time_min // 60, time_min % 60

result_time[0], result_time[1] = result_hour, result_min

print(result_time[0], ' hour ', result_time[1],
      ' min. ', result_time[2], ' sec.')

# answer.4
input_second = 3670

hour = input_second // 3600
remaining_second = input_second - hour*3600

minute = remaining_second // 60
remaining_second = remaining_second - minute * 60

second = remaining_second

print(hour, '시', minute, '분', second, '초')

#%% 피보나치 수열

first_number, second_number = 0, 1
print(first_number)
print(second_number)

 

for i in range(10):
    current_num = first_number + second_number
    
    first_number = second_number
    second_number = current_num
    print(current_num)
    
#%%
first_number, second_number, third_number = 0, 1, 2

print(first_number)
print(second_number)
print(third_number)

for i in range(10):
    current_num = first_number + second_number + third_number
    
    first_number = second_number
    
    second_number = third_number
    
    third_number = current_num
    print(current_num)

 
 #%%

prices = [6000, 14000, 17000, 25000, 3000, 300]
discounted_prices=[]
total_sum = 0
for price in prices:
    if price >= 10000:
        discounted_prices.append(0.95*price)
    elif price >= 15000:
        discounted_prices.append(0.90*price)
    elif price >= 20000:
        discounted_prices.append(0.85*price)
    else:
        discounted_prices.append(price)
#print(discounted_prices)

for discounted_price in discounted_prices:
    total_sum += discounted_price
print(total_sum)

#%%
 
import numpy as np

test_input = np.random.normal(loc=0, scale = 1, size =(100, ))
print(test_input)

test_input = np.random.uniform(0,  100, size =(100, )).astype(np.int)
print(test_input)
#print(type(test_input))

#%%
import numpy as np

test_input = np.random.normal(loc=0, scale=1, size=(100, ))
print(test_input)

#%%

import numpy as np

scores = np.random.uniform(0, 100, size=(100,)).astype(np.int32)
#print(scores)

max_score = None
#current_score = 0
for score in scores:
    if max_score == None or score > max_score:
        max_score = score
print(max_score)
        


print(type(None))
#%%
import numpy as np

scores = np.random.uniform(0, 100, size=(100,)).astype(np.int32)
#print(scores)

min_score = None
#current_score = 0
for score in scores:
    if min_score == None or score < min_score:
        min_score = score
print(min_score)
        


print(type(None))

#%%
for i in range(5,10,2):
    print(i)
#%% list comprehension
#map 대체
#0부터 9까지 도는데 i가 짝수 일때만 넣어줘 
test_list = [2 * i for i in range(10) if i%2 == 0 ]

print(test_list)





