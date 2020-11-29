import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

n_point = 100

x = np.random.normal(0, 1, size=(n_point, ))
y = 3*x + 0.5*np.random.normal(0, 1, size=(n_point, ))

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].scatter(x, y)
axes[0].axvline(x=0, linewidth=3, color='black', alpha=0.5)
axes[0].axhline(y=0, linewidth=3, color='black', alpha=0.5)

#%%
a = -3
n_iter = 300
learning_rate = 0.01
cmap = cm.get_cmap('rainbow', lut=n_iter)
losses = []
for i in range(n_iter):
    predictions = a*x
    mse = np.mean((y - predictions)**2)
    dl_da = -2*np.mean(x*(y - predictions))
    a = a - learning_rate*dl_da
    
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range
    axes[0].plot(x_range, y_range, color=cmap(i))

    losses.append(mse)

axes[1].plot(losses)

#%%
n_point = 1000

x = np.random.normal(0, 1, size=(n_point, ))
y = 3*x + 2 + 0.5*np.random.normal(0, 1, size=(n_point, ))

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
axes[0].scatter(x, y)
axes[0].axvline(x=0, linewidth=3, color='black', alpha=0.5)
axes[0].axhline(y=0, linewidth=3, color='black', alpha=0.5)

a, b = np.random.normal(0, 1, size=(2, ))
n_iter = 300
learning_rate = 0.01
cmap = cm.get_cmap('rainbow', lut=n_iter)
losses = []
a_list, b_list = [], []
for i in range(n_iter):
    a_list.append(a)
    b_list.append(b)
    
    predictions = a*x + b
    mse = np.mean((y - predictions)**2)
    
    dl_da = -2*np.mean(x*(y - predictions))
    dl_db = -2*np.mean((y - predictions))
    
    a = a - learning_rate*dl_da
    b = b - learning_rate*dl_db
    
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range + b
    axes[0].plot(x_range, y_range, color=cmap(i))

    losses.append(mse)

axes[1].plot(losses)
axes[2].plot(a_list, label='a')
axes[2].plot(b_list, label='b')
axes[2].tick_params(labelsize=20)
axes[2].grid(axis='y')
axes[2].legend(fontsize=20)

#%%
n_point = 1000

x1 = np.random.normal(0, 1, size=(n_point, ))
x2 = np.random.normal(0, 1, size=(n_point, ))
y = 3*x1 + 2*x2 - 1

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, y)

fig.tight_layout()

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

a, b, c = np.random.normal(0, 1, size=(3, ))
n_iter = 300
learning_rate = 0.01
cmap = cm.get_cmap('rainbow', lut=n_iter)
losses = []
a_list, b_list, c_list = [], [], []

for i in range(n_iter):
    a_list.append(a)
    b_list.append(b)
    c_list.append(c)
    
    predictions = a*x1 + b*x2 + c
    mse = np.mean((y - predictions)**2)
    
    dl_da = -2*np.mean(x1*(y - predictions))
    dl_db = -2*np.mean(x2*(y - predictions))
    dl_dc = -2*np.mean((y - predictions))
    
    a = a - learning_rate*dl_da
    b = b - learning_rate*dl_db
    c = c - learning_rate*dl_dc
    
    losses.append(mse)

axes[1].plot(losses)
axes[2].plot(a_list, label='a')
axes[2].plot(b_list, label='b')
axes[2].plot(c_list, label='c')
axes[2].tick_params(labelsize=20)
axes[2].grid(axis='y')
axes[2].legend(fontsize=20)

#%%
n_point = 100
x = np.random.normal(0, 1, size=(n_point, ))
x_noise = x + 0.2*np.random.normal(0, 1, size=(n_point, ))
    
y = (x_noise >= 0).astype(np.int)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].scatter(x, y)

a, b = np.random.normal(0, 1, size=(2, ))
n_iter = 1000
learning_rate = 0.1
cmap = cm.get_cmap('rainbow', lut=n_iter)
losses = []
for i in range(n_iter):
    
    bce = 

    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = a*x_range + b
    y_range = 1/(1 + np.exp(-y_range))
    axes[0].plot(x_range, y_range, color=cmap(i))

    losses.append(bce)

axes[0].grid()
axes[1].plot(losses)




























