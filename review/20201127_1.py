import math

#%% Binary Cross Entropy Loss 구하기

# answer.1
import math
ground_truth = 0.7
prediction1 = 0.5
prediction2 = 0.9
prediction3 = 0.4

loss1 = -(ground_truth*math.log(prediction1) + (1 - ground_truth)*math.log(1-prediction1))
loss2 = -(ground_truth*math.log(prediction2) + (1 - ground_truth)*math.log(1-prediction2))
loss3 = -(ground_truth*math.log(prediction3) + (1 - ground_truth)*math.log(1-prediction3))

# answer.2
predictions = [prediction1, prediction2, prediction3]
binary_cross_entropy = []

for prediction in predictions:
    binary_cross_entropy.append(ground_truth * math.log(prediction)\
                                + (1 - ground_truth)*math.log(1 - prediction))
print(binary_cross_entropy)

#%% Binary Cross Entropy Loss 구하기(평균)
import math
ground_truth = 0.7
prediction1 = 0.5
prediction2 = 0.9
prediction3 = 0.4

N = 3
loss1 = -(ground_truth*math.log(prediction1) + (1 - ground_truth)*math.log(1-prediction1))
loss2 = -(ground_truth*math.log(prediction2) + (1 - ground_truth)*math.log(1-prediction2))
loss3 = -(ground_truth*math.log(prediction3) + (1 - ground_truth)*math.log(1-prediction3))

bce_loss = (loss1 + loss2 + loss3)/N
print(bce_loss)

#%% BCE + NumPy
import numpy as np

labels = np.array([0.9, 0.7, 0.4])
predictions = np.array([0.7, 0.5, 0.7])

bce_loss = -1*np.mean(labels*np.log(predictions) + \
                      (1-labels)*np.log(1-predictions))
print(bce_loss)

#%% BCE in Deep Learning
import numpy as np

# 2 classes
labels = np.array([1, 0, 1])
labels = np.array([[0, 1], [1, 0], [0, 1]])

# 3 classes
labels = np.array([1, 0, 2])
labels = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

#%%
# 2 classes
labels = np.array([1, 0, 1])
predictions = np.array([0.9, 0.1, 0.6])
bce_loss = -1*np.mean(labels*np.log(predictions) + \
                      (1-labels)*np.log(1-predictions))
print(bce_loss)

labels = np.array([[0, 1], [1, 0], [0, 1]])
predictions = np.array([[0.1, 0.9],
                        [0.9, 0.1],
                        [0.4, 0.6]])

inner = labels*np.log(predictions)
inner = np.sum(inner, axis=1)
bce_loss = -1*np.mean(inner)
print(bce_loss)

#%% accuracy
labels = np.array([1, 0, 1, 0, 1, 0, 1])
predictions = np.array([0, 1, 0, 0, 0, 1, 1])

accuracy = np.sum((labels == predictions).astype(np.int))/labels.shape[0]*100
accuracy = np.around(accuracy, 2)
print(accuracy, '%')

labels = np.array([1, 0, 1, 0, 1, 0, 1])
predictions = np.array([0.1, 0.9, 0.2, 0.3, 0.1, 0.6, 0.7])
predictions = (predictions >= 0.5).astype(np.int)
accuracy = np.sum((labels == predictions).astype(np.int))/labels.shape[0]*100
accuracy = np.around(accuracy, 2)
print(accuracy, '%')

labels = np.array([1, 0, 1, 0, 1, 0, 1])
predictions = np.array([[0.9, 0.1],
                        [0.1, 0.9],
                        [0.8, 0.2],
                        [0.7, 0.3],
                        [0.9, 0.1],
                        [0.4, 0.6],
                        [0.3, 0.7]])
predictions = np.argmax(predictions, axis=1)
accuracy = np.sum((labels == predictions).astype(np.int))/labels.shape[0]*100
accuracy = np.around(accuracy, 2)
print(accuracy, '%')









































