# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:47:59 2020

@author: Administrator
"""

#%%
#평균과 분산 구하기 



scores = [10, 20, 30 ]

n_student = len(scores)

mid_sum = 0

var_mid_sum = 0

for score in scores:
    mid_sum += score
    
ave = mid_sum / n_student

for score in scores:
    var_mid_sum += score**2
    
#print(var_mid_sum)

var_sum = var_mid_sum / n_student

var = var_sum - ave**2

print(ave)

print(var)
    
#%%
# 평균 구하기
scores = [10, 20, 30]

sum_scores = 0
squared_sum_scores = 0
n_student = len(scores)

for score in scores:
    sum_scores += score
    squared_sum_scores += score**2
    
mean_scores = sum_scores / n_student
variance_scores = squared_sum_scores / n_student - mean_scores**2

print('Average  : {}'.format(mean_scores))
print('Variance : {}'.format(variance_scores))

# %%
''' range() '''
math_scores = [10, 20, 30, 40, 50]
english_scores = [20, 30, 40, 50, 60]

math_score_sum, english_score_sum = 0, 0
for index in range(len(scores)):
    math_score_sum += math_scores[index]
    english_score_sum += english_scores[index]

math_mean = math_score_sum / len(math_scores)
english_mean = english_score_sum / len(english_scores)

print(math_mean, english_mean)



#%%
math_scores = [10, 20, 30, 40, 50]
english_scores = [20, 30, 40, 50, 60]

math_score_sum, english_score_sum = 0, 0
for index in range(len(math_scores)):
    math_score_sum += math_scores[index]
    english_score_sum += english_scores[index]

math_mean = math_score_sum / len(math_scores)
english_mean = english_score_sum / len(english_scores)

print(math_mean, english_mean)



#%% 별 찍기

star="*"

for index in range(6):
    print((5-index)*" ",index*"*",sep='')
    
#%% list의 연산 
test_list1 = [1, 2, 3]
test_list2 = [4, 5, 6]

print(test_list1 + test_list2)
  
#%%

a = 10
b = 10

print( a >= b)
print( a <= b)





#%% Boolean data type

#0 빼고는 다 True, 음수도 True이다 .
# True, False

a = 3.0
print(bool(a))

a = 0.0
print(bool(a))

#list ==> bool

a = []
print(bool(a))

a = [0]
print(bool(a))

#%%

score = 70

if score > 60:
    print("합격! ")

elif score > 40 and score <= 60:
    print("재시험")

else:
    print("불합격")
    
print("시험보느라 고생 많았어")

#%%

''' 학점 매기기 '''
''' 90~100 :A
    70~90 :B
    50~70 :C
    0~50: F
'''
print("점수 입력:")
score = int(input())

if score >= 90 and score <= 100:
    print("A")

elif score >= 70 and score <= 89:
    print("B")
    
elif score >= 50 and score <= 69:
    print("C")
    
else:
    print("F")
    
#%%


print("점수 입력:")
score = int(input())

if score >= 90 and score <= 100:
    print("A")

elif score >= 70 and score <= 89:
    print("B")
    
elif score >= 50 and score <= 69:
    print("C")
    
elif score >= 0 and score <= 49:
    print("F")

else:
    raise ValueError

#%%

''' 학점 매기기 '''
''' 95~100 :A+
    90~94 :A0
    80~89 :B+
    70~79 :B0
    60~69:C+
    50~59:C0
    0~49: F
'''
print("점수 입력:")
score = int(input())

if score >= 90 and score <= 100:
    if score >= 95 and score <= 100:
        print("A+")
    else:
        print("A0")
        
if score >= 70 and score <= 89:
    if score >= 80 and score <= 89:
        print("B+")
    else:
        print("B0")
    
elif score >= 50 and score <= 69:
    if score >= 60 and score <= 69:
        print("C+")
    else:
        print("C0")
 
    
elif score >= 0 and score <= 49:
    print("F")

else:
    raise ValueError
#%% 큰 수 출력하기 

a, b = 30, 20

if a > b:
    print(a)

elif a == b:
    print("같다")
    
else:
    print(b)
    
    
#%% 양수 음수 판별

a = -40

if a > 0:
    print("절댓값은",a)
elif a < 0:
    print("절댓값은",-a)
else:
    print("절댓값은",a)

#%%
#만원이상 구매시 5퍼센트 할인
#이만원이상 구매시 10퍼센트 할인
#삼만원이상 구매시 15퍼센트 할인

print("가격:")
score = int(input())


if score >= 10000 and score < 20000:
    print(score-score*0.05)

elif score >= 20000 and score < 30000:
    print(score-score*0.1)
    
elif score >= 30000 :
    print(score-score*0.15)

else:
    raise ValueError
    
#%%
print("가격:")
score = int(input())


if score >= 10000 and score < 20000:
    print(score*0.95)

elif score >= 20000 and score < 30000:
    print(score*0.9)
    
elif score >= 30000 :
    print(score*0.85)

else:
    raise ValueError



#%% 연산 기호에 따라 값 출력하기 

a, b =10, 20 

c = input()

if a =='+':
    
    
elif a =='-':

elif a =='*':

elif a =='/':

elif a =='//':    

elif a =='%': 
    
else:
    
    
    #%%
scores = [10, 30, 60, 80, 20, 50]
'''반평균이 70점 이상이면 우수반, 아니면 보충반'''
score_sum = 0
stu_num = len(scores)
for score in scores:
    score_sum += score
    
ave = score_sum / stu_num

if ave >= 70:
    print("우수반")

if ave >= 50 and ave <= 69:
    print("중급반")
    
else:
    print("보충반")
#%% 홀수들의 합, 짝수들의 합 구하기 
odd_sum, odd_cnt = 0, 0
even_sum, even_cnt = 0, 0

for i in range(100):
    if i % 2 ==0:
        even_sum += i
        n_even += 1
    else:
        odd_sum += i
        n_odd += 1
        
print(odd_sum / even_cnt)
print(even_sum / odd_cnt)

#%% 3의 배수들의 합,평균을 구하기 



#%% 3으로 나눴을 때 0인 값, 1인 값 , 2인 값 들의 총합과 평균 










    
    