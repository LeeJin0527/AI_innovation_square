#%% 3의 배수들의 합, 평균
#%% 3으로 나눴을 때 0인 값, 1인 값, 2인 값들의 총합과 평균

#%% 과제.1
three_sum = 0
total_number = 0
for i in range (100):
    if i%3 == 0 :
        three_sum += i
        total_number += 1
print ("총합", three_sum)
print ("평균", three_sum/total_number)

# %% 과제.2
n_iteration = 100
zero_sum, zero_mean, zero_cnt = 0, 0, 0
one_sum, one_mean, one_cnt = 0, 0 ,0
two_sum, two_mean, two_cnt = 0, 0, 0

for i in range(n_iteration):
    if i % 3 == 0:
        zero_sum += i
        zero_cnt += 1
    elif i % 3 == 1:
        one_sum += i
        one_cnt += 1
    else:
        two_sum += i 
        two_cnt += 1

zero_mean = zero_sum / zero_cnt
print("zero_sum : ", zero_sum)
print("zero_mean : ", zero_mean)

one_mean = one_sum / one_cnt
print("one_sum : ", one_sum)
print("one_mean : ", one_mean)

two_mean = two_sum / two_cnt
print("two_sum : ", two_sum)
print("two_mean : ", two_mean)

# %% 분, 초 구하기

# answer.1
second = 129
print(second//60, '분 ',second%60,'초 입니다' )

# answer.2
minute = 0
second = 123

minute = (second//60)
second = (second%60)

print (minute, "분", second, "초")

# answer.3
time_in_sec = 182
time_min, time_sec = time_in_sec // 60, time_in_sec % 60

print(time_min, ' min. ', time_sec, ' sec.')

# answer.4
input_second = 131
counted_second = input_second  % 60
counted_minute = (input_second - counted_second ) / 60
print('입력하신 시간은',counted_minute, '분  ',counted_second,'초 입니다.' )

# answer.5
second = 70
minute = second // 60
second1 = second % 60
if second > 60:
    print(minute,"분", second1,"초")
else:
    print(second1,"초")


# %% 시분초 구하기

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

#%% 사분면 구하기

# answer.1
point = [0, -3]
if point[0] > 0 and point[1] > 0:
    print ("1사분면")
elif point[0] < 0 and point[1] > 0:
    print("2사분면")
elif point[0] < 0 and point[1] < 0:   
    print("3사분면")
elif point[0] > 0 and point[1] < 0:
    print("4사분면")
else:
    print('좌표계에 점이 위치')

# answer.2
point_coordinate = [0, 0]
if point_coordinate [0] > 0:
    if point_coordinate [1] > 0:
        print('1사분면에 위치')
    elif point_coordinate[1] < 0:
         print('4사분면에 위치')
elif point_coordinate [0] < 0:     
    if point_coordinate [1] > 0:
        print('2사분면에 위치')
    elif point_coordinate[1] < 0:
         print('3사분면에 위치')  
else:
    print('좌표계에 점이 위치')

# answer.3
point_coordinate = [2, 3]
x = int(point_coordinate[0])
y = int(point_coordinate[1])

if ((x > 0) and (y > 0)):
    print("1사분면")
elif ((x < 0) and (y > 0)):
    print("2사분면")
elif ((x < 0) and (y < 0)):
    print("3사분면")
elif ((x > 0) and (y < 0)):
    print("4사분면")

# %% 사분면, x축, y축, 원점에 있는지 구하기

# answer.1
point_coordinate = [0, 0]
x = int(point_coordinate[0])
y = int(point_coordinate[1])
if ((x == 0) and (y != 0)):
    print('y축 위에 있습니다')
elif ((x != 0) and (y == 0)):
    print('x축 위에 있습니다')
elif ((x ==0) and (y == 0)):
    print('원점 위에 있습니다')
    
elif x > 0:
    if y > 0:
        print('1사분면')
    elif y < 0:
        print('4사분면')
elif x < 0:
    if y > 0:
        print('2사분면')
    elif y < 0:
        print('3사분면')    
else:
    print('invalid value')


# answer.2
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

# %% 피보나치 수열

n_iter = 10
first_number, second_number = 0, 1

print(first_number)
print(second_number)

for i in range(n_iter):
    current_number = first_number + second_number
    
    first_number = second_number
    second_number = current_number
    print(current_number)

#%% 3개씩 더하는 피보나치 수열
first_number, second_number, third_number = 0, 1, 2
print(first_number)
print(second_number)
print(third_number)

for i in range(10):
    current_number= first_number + second_number + third_number
    first_number = second_number
    second_number= third_number
    third_number = current_number 
    print(current_number)

# %% 최댓값, 최솟값 구하기
    
# answer.1
import numpy as np
scores = np.random.uniform(-50, -10, size=(100, )).astype(np.int32)
max_score = scores[0]

for score in scores:
     if max_score < score:
        max_score = score
print(max_score)

# answer.2
print(scores)
max_score, min_score = None, None
for score in scores:
    if max_score == None or score > max_score:
        max_score = score
        
    if min_score == None or score < min_score:
        min_Score = score
        
print(max_score, min_score)


#%% range의 확장 - 시작값, step 조절하기

start_idx = 3
n_iter = 10

for i in range(start_idx, start_idx + n_iter, 3):
    print(i)


list_ = list()
for i in range(1, 100, 2):
    list_.append(i)
print(list_)


#%% list comprehension 소개

test_list = [i for i in range(10) if i % 2 == 0]












