import numpy as np

# MSE error
n_point = 100
x = np.random.normal(0, 2, size=(n_point,))
y = 3*x
predictions = 2*x

diff_square = (y - predictions)**2
mse_error = np.mean(diff_square)

#%%
n_student = 100
cutoff = 80
scores = np.random.uniform(0, 100, size=(n_student, ))

student_pass = (scores > cutoff).astype(np.int)
pass_percentage = np.sum(student_pass) / n_student * 100
print(pass_percentage, '%')

#%%
scores = np.random.uniform(0, 100, size=(n_student, )).astype(np.int)
print(scores)
is_odds = (scores % 2).astype(np.int)
print(is_odds)
odd_percentage = np.sum(is_odds) / n_student * 100
print(odd_percentage)

#%%
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


signal = np.random.normal(loc=0, scale=1, size=(100, ))
filter_ = np.array([1, 5, 3, 2, 1, 5])

































