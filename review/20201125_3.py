import numpy as np

#%% reshape
a = np.array([1, 2, 3, 4])
print(a.shape)
print(a, '\n')

a = a.reshape((4, 1))
print(a.shape)
print(a, '\n')

a = a.reshape((1, 4))
print(a.shape)
print(a, '\n')

a = a.reshape((2, 2))
print(a.shape)
print(a, '\n')


#%% reshape + -1 value
a = np.random.uniform(0, 20, size=(20, ))
print(a.shape)

a = a.reshape((4, 5))
print(a.shape)
a = a.reshape((4, -1))
print(a.shape)
a = a.reshape((-1, 5))
print(a.shape)

a = a.reshape((2, -1))
print(a.shape)

#%% reshape to row/col vector
a = a.reshape((1, -1)) ''' to row vector '''
print(a.shape)
a = a.reshape((-1, 1)) ''' to column vector '''
print(a.shape)

#%% Broadcasting

a = np.array([1, 2, 3, 4]).reshape((1, -1))
b = np.array([10, 20, 30]).reshape((-1, 1))
print(a.shape)
print(b.shape)

c = a / b
print(c.shape)

print(a, '\n')
print(b, '\n')
print(c, '\n')

#%%
a = np.random.uniform(0, 5, size=(2, 3)).astype(np.int)
print(a, '\n')

b = np.array([1, 2]).reshape(-1, 1)
print(a + b, '\n')

b = np.array([1, 2, 3]).reshape(1, -1)
print(a + b, '\n')

























