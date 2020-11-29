import numpy as np

python_list = [1, 2, 3, 4, 5]
ndarray1 = np.array(python_list)

print(ndarray1)
print(ndarray1[0])
print(ndarray1[1])
print(ndarray1[2])

#%% ndarray indexing
python_list = [[1, 2], [3, 4], [5, 6]]
ndarray1 = np.array(python_list)

print(ndarray1)
print('student1: ', ndarray1[0])
print('student2: ', ndarray1[1])
print('student3: ', ndarray1[2])

#%% 평균 구하기(국어, 수학, 영어)
scores = np.random.randint(0, 100, size=(100, 3))
n_student = scores.shape[0]
n_class = scores.shape[1]
# print(scores.shape)
# print(scores)

# answer.1
class1_sum, class2_sum, class3_sum = 0, 0, 0
for score in scores:
    class1_val, class2_val, class3_val = score
    class1_sum += class1_val
    class2_sum += class2_val
    class3_sum += class3_val

class1_mean = class1_sum / n_student
class2mean = class2_sum / n_student
class3_mean = class3_sum / n_student

# answer.2
class_sum = np.zeros(shape=(3, ))
for score in scores:
    class_sum += score
class_mean = class_sum / n_student
print(class_mean)

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






























