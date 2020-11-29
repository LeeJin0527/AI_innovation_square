# 단순한 list comprehension
test_list = []
for i in range(100):
    test_list.append(i)

# print(test_list, '\n')

test_list = [2*i for i in range(100)]
# print(test_list)

test_list = [3*i for i in range(100)]
print(test_list)
# 0 포함해서 3의 배수를 가지는 list 만들기
test_list = [5*i for i in range(100)]
print(test_list)
# 0 포함해서 5의 배수를 가지는 list 만들기

# %% list comprehension + if
test_list = [i for i in range(100) if i % 2 == 0]
print(test_list)

#%% list comprehension applications

test_list = [str(i) for i in range(100)]
test_list = [float(i) for i in range(100)]
test_list = [i+1 for i in range(100)]
test_list = [i%4 for i in range(100)]
test_list = [bool(i%2) for i in range(100)]
test_list = [i**2 for i in range(100)]

#%% list comprehension + 2 for loops + if
test_list = [i*j for i in range(10) for j in range(10)]
print(test_list)

test_list = list()
for i in range(10):
    for j in range(10):
        test_list.append(i*j)
        
test_list = [i*j for i in range(10) 
                 for j in range(10)
                 if i*j % 2 == 0]

#%%

a = [1, 2, 3, 4]
print(type(a))
print(a, '\n')
# print(dir(a))

for i in range(4):
    print(a.pop())
    print(a, '\n')

#%% object의 개념

# %% list in list 개념

# [수학, 영어] 점수
scores = [[10, 20], [30, 40], [50, 60]]

means = []
for score in scores:
    math_score = score[0]
    english_score = score[1]
      
print(means[0]) # 수학 평균
print(means[1]) # 영어 평균

#%% Q) 평균 구하기

# answer.1
scores = [[10, 20], [30, 40], [50, 60]]

# 리스트 0 = 수학평균, 리스트 1 = 영어평균
means, sums = [0, 0], [0, 0]

for score in scores:
    sums[0] += score[0]
    sums[1] += score[1]

means[0] = sums[0] / len(scores)
means[1] = sums[1] / len(scores)

print(means)

# answer.2
scores = [[10, 20], [30, 40], [50, 60]]

means = []
n_students = len(scores)

math_score_sum = 0
english_score_sum = 0

for score in scores:
    math_score_sum += score[0]
    english_score_sum += score[1]
    
means.append(math_score_sum / n_students)
means.append(english_score_sum / n_students)

print('Mean score (math) : ', means[0],
      '\nMean score (english) : ', means[1])

# answer.3
scores = [[10,20], [30,40], [50,60]]
n_student = len(scores)
sum_math_score, sum_english_score =0, 0

for score in scores:
    sum_math_score += score[0]
    sum_english_score += score[1]
mean = [sum_math_score/n_student, sum_english_score/n_student]

print(mean[0])
print(mean[1])

#%% Unpacking 개념

a, b = [1, 2]
print('a: ', a, '\n', 'b: ', b)

scores = [[10, 20], [30, 40], [50, 60]]
# 리스트 0 = 수학평균, 리스트 1 = 영어평균
means, sums = [0, 0], [0, 0]

for math_score, english_score in scores:
    sums[0] += math_score
    sums[1] += english_score

means[0] = sums[0] / len(scores)
means[1] = sums[1] / len(scores)

print(means)

# %% 평균 구하기
names_scores = [['A', 100],
                ['B', 50],
                ['C', 30]]

sum_score = 0
# _는 필요없는 값을 저장하기 위한 변수로 약속
for _, score in names_scores:
    sum_score += score

mean_score = sum_score / len(names_scores)
print(mean_score)

#%% centroid 구하기

coordinates = [[-2, 3], [4, 6], [-10, 30]]

xs, ys = [0, 0]

for x, y in coordinates:
    xs += x
    ys += y
    
centroid = [xs / len(coordinates), ys / len(coordinates)]
print(centroid)

#%% Euclidean distance
import math

coordinates = [[-2, 3], [4, 6], [-10, 30]]
centroid = [5, -1]

distances = list()
for x, y in coordinates:
    square_sum = (x - centroid[0])**2 + (y - centroid[1])**2
    distance = math.sqrt(square_sum)
    
    distances.append(distance)

print(distances)

# %% dictionary

means = {'math':10, 'english':20, 'physics':30}
print(means)

print(means['math'])
print(means['english'])
print(means['physics'])

# %% dictionary 만들기

means = dict()
print(means)
means['math'] = 20
print(means)
means['english'] = 30
print(means)
means['physics'] = 40
print(means)

# %% dictionary로 평균 구하기

scores = [[10, 20], [30, 40], [50, 60]]
means = dict()
sums = [0, 0]
subjects = ['math', 'english']

for math_score, eng_score in scores:
    sums[0] += math_score
    sums[1] += eng_score

for index in range(len(sums)):
    subject = subjects[index]
    means[subject] = sums[index] / len(scores)

print(means)
