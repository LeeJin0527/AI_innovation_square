import numpy as np
import matplotlib.pyplot as plt

# dataset generation
n_class, std, n_point = 2, 1, 100

dataset = np.empty(shape=(0, 2))

# fig, ax = plt.subplots(figsize=(10, 10))
for class_idx in range(n_class):
    centers = np.random.uniform(-3, 3, size=(2, ))
    
    x_data = np.random.normal(loc=centers[0], scale=std,
                              size=(n_point, 1))
    y_data = np.random.normal(loc=centers[1], scale=std,
                              size=(n_point, 1))
    
    data = np.hstack((x_data, y_data))
    dataset = np.vstack((dataset, data))
    
    # ax.scatter(x_data, y_data)


dataset = dataset.tolist()
centroids = np.random.uniform(-5, 5, size=(n_class, 2)).tolist()


# fig, axes = plt.subplots(3, 3, figsize=(15, 15))



for iteration in range(9):
    clusters = []
    for _ in range(n_class):
        clusters.append([])
        
    for data in dataset:
        x, y = data
        
        distances = []
        for centroid in centroids:
            centroid_x, centroid_y = centroid
            distance = (centroid_x - x)**2 + (centroid_y - y)**2
            distances.append(distance)
        
    cnt, m = 0, None
    for distance in distances:
        if m == None or distance < m:
            m = distance
            idx = cnt
        cnt += 1
        
    clusters[idx].append(data)
    cnt = 0
    for cluster in clusters:
        x_sum, y_sum = 0, 0
        iter_cnt = 0
        for clustered_data in cluster:
            x_sum += clustered_data[0]
            y_sum += clustered_data[1]
            iter_cnt += 1
        x_mean = x_sum / iter_cnt
        y_mean = y_sum / iter_cnt
        
        centroids[cnt] = [x_mean, y_mean]
        cnt += 1
    
            
        



kmeans = KMeans(n_clusters=2)
kmeans.fit(X)


#%%
a = 10
def outer():
    a = 20
    def inner():
        global a
        print(a)
        
    inner()
        
outer()









