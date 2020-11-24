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
#%%
clusters = []
for _ in range(n_class):
    clusters.append([])

# for iteration in range(9):
cluster1, cluster2 = [], []
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
#%%
print(20)

#%%
def a(**k):
    print(list(k.keys()))
    print(list(list(k.values())))
    
a(a=10,b=20)
print(a)

#%%
A= [2**i for i in range(10)]
print(A[2])

#%%
def func():
    yield 1
    yield 1
    yield 1
print(type(func()))

#%%
class Test:
    def __init__(self,a):
        self.__a = a
        
    @property
    def a(self):
        return self.__a
    @a.setter
    def a(self,value):
        self.__a = value
a1,a2,a3 =Test(3),Test(2),Test(3)
a3.a =a1.a +a2.a
print(a3.a)
#%%
class GF:
    pass
class F(GF):
    pass
class S(F):
    pass

k,g,s=F(),GF(),S()
print(isinstance(s,GF),isinstance(s,F))

#%%
import numpy as np
import matplotlib.pyplot as plt

##### dataset generation
n_class, std, n_point = 10, 1, 100
dataset = np.empty(shape=(0, 2))

# fig, ax = plt.subplots(figsize=(10, 10))
for class_idx in range(n_class):
    center = np.random.uniform(-15, 15, size=(2, ))
    
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
dataset = dataset.tolist()


####centroid initialization
n_cluster = 2
centroids = np.random.uniform(-3, 3, size=(n_cluster,2 )).tolist()
# print(centroids)
for i in range(10):
    
    cluster1, cluster2 =[],[]
    for data_point in dataset:
        x,y = data_point
        distance1 = (centroids[0][0] - x)**2 + (centroids[0][1] - y)**2
        distance2 = (centroids[1][0] - x)**2 + (centroids[1][1] - y)**2
        
        if distance1 < distance2:
            cluster1.append(data_point)
            
               
        else:
            cluster2.append(data_point)
           
            
    # print(cluster1[0])           
        
           
           
    x_sum,y_sum, cnt = 0, 0, 0
    for i in cluster1:
        x,y =i
        x_sum += x 
        y_sum += y 
        cnt += 1
        
    centroids[0][0] = x_sum / cnt
    centroids[0][1]= y_sum / cnt
    print(centroids[0])
    
    x_sum,y_sum, cnt = 0, 0, 0
    for i in cluster2:
        x,y =i
        x_sum += x 
        y_sum += y 
        cnt += 1
        
    centroids[1][0] = x_sum / cnt
    centroids[1][1]= y_sum / cnt
    print(centroids[1])
    
 #%%
test_dict = {'a':1, 'b':2}

for key in test_dict:
    print(key,test_dict[key],'\n')
for key,value in test_dict.items():
    print(key,value,'\n')
    
    
    



#%%
## cluster 2개 짜리 9번 만들기 
# for cluter_cnt in range(9): #카운트 9
    ##### cluster initialization
cluster_dict = dict()  #클러스터 딕셔너리 형태
#센트로이드 개수마다 클러스터_dic리스트 형태로 만든다
for class_idx in range(n_cluster):  #2
    cluster_dict[class_idx] = list() #각 포문을 리스트로 만듦

#### clustering
#데이터셋 
for data_point in dataset:
    x, y = data_point #x,y로 나눔 
    #거리 리스트로 만든다 
    distances = list()  #거리 리스트
    #각 센트로이드마다 센트로이드 x,y좌표 만듦
    for cluster_idx in range(n_cluster): 
        centroid = centroids[cluster_idx]  
        centroid_x, centroid_y = centroid
        #각 센트로이드마다 거리 리스트로 만든다 그럼 리스트 두개 나오겟지 
        distance = (centroid_x - x)**2 + (centroid_y - y)**2
        distances.append(distance)
    # print(distances)
    m, m_idx = None, None
    distance_idx = 0
    for distance in distances:
        if m == None or distance < m:
            m = distance
            m_idx = distance_idx
            # print(type(m_idx))
        distance_idx += 1
    
    cluster_dict[m_idx].append(data_point)
        
    #### centroid update
    for cluster_idx, cluster in cluster_dict.items():
        x_sum, y_sum, iter_cnt = 0, 0, 0
        for data_point in cluster:
            x_sum += data_point[0]
            y_sum += data_point[1]
            iter_cnt += 1
        x_mean = x_sum / iter_cnt
        y_mean = y_sum / iter_cnt
        
    centroids[cluster_idx] = [x_mean, y_mean]
    print(centroids)


#%% enumerate
scores = [10, 30, 50, 20, 10]

idx = 0 
for score in scores:
    print(idx,score)
    
    idx += 1
    
#%%
scores = [10, 30, 50, 20, 10]
M, idx = None, None
for score in scores:
    if M ==None or score > M:
        M = score
        idx = iter_cnt
    iter_cnt += 1
    
print(idx, M)

#%%
for idx, score in enumerate(scores):
    print(idx, score)
#%%

import numpy as np 
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

print(a % b)

#%%
import numpy as np 
a = dir()
print(a)














