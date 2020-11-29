import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
    
#%% enumerate
'''
Q) 뭐가 불편한가요
A) idx 만들고 += 해주기 귀찮아요
'''

scores = [10, 30, 50, 20, 10]

idx = 0
for score in scores:
    print(idx, score)
    idx += 1

for idx, score in enumerate(scores):
    print(idx, score)

#%%

scores = [10, 30, 50, 20, 10]
M, idx = None, None
iter_cnt = 0

for score in scores:
    if M == None or score > M:
        M = score
        M_idx = iter_cnt
    iter_cnt += 1

print(M_idx, M)

###
M, M_idx = None, None
for idx, score in enumerate(scores):
    if M == None or score > M:
        M = score
        M_idx = idx
print(M_idx, M)

###
scores = [10, 30, 50, 20, 10]

score_sum = 0
for idx, score in enumerate(scores):
    score_sum += score

score_mean = score_sum / (idx + 1)
print(score_mean)

#%% dictionary key, value
test_dict = {'a':1, 'b':2, 'c':3}
for key in test_dict:
    print(key, test_dict[key], '\n')

for key, value in test_dict.items():
    print(key, value, '\n')
    
#%% K-Means Clustering
    
##### dataset generation
n_class, std, n_point = 5, 1, 300
dataset = np.empty(shape=(0, 2))

# fig, ax = plt.subplots(figsize=(10, 10))
for class_idx in range(n_class):
    center = np.random.uniform(-10, 10, size=(2, ))
    
    x_data = np.random.normal(loc=center[0], scale=std,
                              size=(n_point, 1))
    y_data = np.random.normal(loc=center[1], scale=std,
                              size=(n_point, 1))
    
    class_data = np.hstack((x_data, y_data))
    dataset = np.vstack((dataset, class_data))
    
    # ax.scatter(x_data, y_data)
    
# ax.tick_params(labelsize=20)
# ax.set_xlabel("X data", fontsize=30)
# ax.set_ylabel("Y data", fontsize=30)
# ax.grid()
dataset = dataset


##### Centroid initialization
n_cluster = 5
centroid_indices = np.random.choice(len(dataset), size=(n_cluster, ))
centroids = dataset[centroid_indices]
cmap = cm.get_cmap(name='rainbow', lut=n_cluster)

dataset = dataset.tolist()
centroids = centroids.tolist()
# centroids = np.random.uniform(-5, 5, size=(n_cluster, 2))
 
# fig, axes = plt.subplots(3, 3, figsize=(15, 15))
# for ax_idx, ax in enumerate(axes.flat):

# fig, ax = plt.subplots(figsize=(15, 15))
# ax.scatter(dataset[:,0], dataset[:,1])
# ax.scatter(centroids[:,0], centroids[:,1], color='r', s=100)

fig, axes = plt.subplots(4, 4, figsize=(20, 10))


for cluter_cnt in range(9):
# for ax_idx, ax in enumerate(axes.flat):
    
    ##### cluster initialization
    cluster_dict = dict() 
    for class_idx in range(n_cluster):
        cluster_dict[class_idx] = list()
    
    #### clustering
    for data_point in dataset:
        x, y = data_point
        
        distances = list()
        for cluster_idx in range(n_cluster):
            centroid = centroids[cluster_idx]
            centroid_x, centroid_y = centroid
            
            distance = (centroid_x - x)**2 + (centroid_y - y)**2
            distances.append(distance)
            
          
        m, m_idx = None, None
        distance_idx = 0
        for distance in distances:
            if m == None or distance < m:
                m = distance
                m_idx = distance_idx
            distance_idx += 1
        
        cluster_dict[m_idx].append(data_point)
            
    ##### centroid update
    for cluster_idx, cluster in cluster_dict.items():
        x_sum, y_sum, iter_cnt = 0, 0, 0
        for data_point in cluster:
            x_sum += data_point[0]
            y_sum += data_point[1]
            iter_cnt += 1
        x_mean = x_sum / iter_cnt
        y_mean = y_sum / iter_cnt
        
        centroids[cluster_idx] = [x_mean, y_mean]
      
    for cluster_idx in range(n_cluster):
        cluster = cluster_dict[cluster_idx]
        centroid = centroids[cluster_idx]
        
        x_data = [x for (x, y) in cluster]
        y_data = [y for (x, y) in cluster]
        
        ax.scatter(x_data, y_data, alpha=0.3, color=cmap(cluster_idx))
        ax.scatter(centroid[0], centroid[1], s=100, color='r')
    
    ax.tick_params(left=False, labelleft=False,
                   bottom=False, labelbottom=False)
    

fig.tight_layout()


#%%







































