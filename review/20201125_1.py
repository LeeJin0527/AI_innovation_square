import numpy as np

### What's ndarray?
python_list = [1, 2, 3]
ndarray = np.array(python_list)

print("python_list: ", python_list)
print("type(python_list): ", type(python_list), '\n')

print("ndarray: ", ndarray)
print("type(ndarray): ", type(ndarray))

print('dir(python):\n', dir(python_list), '\n')
print('dir(ndarray):\n', dir(ndarray), '\n')

#%% ndarray 만들기.1

# python list => ndarray
python_list = [1, 2, 3]
ndarray = np.array(python_list)

# np.zeros()
ndarray2 = np.zeros(shape=(10,))
# print(ndarray2)

# np.ones()
ndarray3 = np.ones(shape=(10,))
# print(ndarray3)

# np.full()
ndarray4 = np.full(shape=(10,), fill_value=3.14)
# print(ndarray4)

# np.full() with np.ones()
ndarray5 = 3.14*np.ones(shape=(10,))
# print(ndarray5)

# np.empty()
ndarray6 = np.empty(shape=(10,))
# print(ndarray6)

#%% ndarray 만들기.2
tmp = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
print(tmp)

ndarray7 = np.zeros_like(tmp)
print(ndarray7)

ndarray8 = np.ones_like(tmp)
print(ndarray8)

# ndarray9 = np.full_like(tmp)
ndarray9 = 3.14*np.ones_like(tmp)
print(ndarray9)

ndarray10 = np.empty_like(tmp)
print(ndarray10)

#%% Matrix 만들기.1 - Python list => ndarray

python_list = [[1, 2], [3, 4]]
ndarray1 = np.array(python_list)

print("python_list: ", python_list, '\n')
print("ndarray1: \n", ndarray1)

#%% Matrix 만들기.2

ndarray2 = np.zeros(shape=(2, 2, 2))
print(ndarray2)

ndarray2 = np.zeros(shape=(2, 2))
print(ndarray2)

ndarray3 = np.ones(shape=(2, 2))
print(ndarray3)

ndarray4 = np.full(shape=(2, 2), fill_value=3.14)
print(ndarray4)

ndarray5 = 3.14*np.ones(shape=(2, 2))
print(ndarray5)

ndarray6 = np.empty(shape=(2, 2))
print(ndarray6)

#%% vector vs matrix
ndarray1 = np.ones(shape=(5,))
ndarray2 = np.ones(shape=(5, 1))

print(ndarray1)
print(ndarray2)

#%% ndarray information
ndarray1 = np.full(shape=(2, 2), fill_value=3.14)

'''shape, dtype, size, itemsize '''
print(ndarray1)
print("ndarray1.shape: ", ndarray1.shape)
print("ndarray1.dtype: ", ndarray1.dtype)
print("ndarray1.size: ", ndarray1.size)
print("ndarray1.itemsize: ", ndarray1.itemsize)

print("ndarray data size: ",
      ndarray1.size*ndarray1.itemsize, 'B')

#%% shape => n_row, n_col
ndarray1 = np.full(shape=(100, 2), fill_value=3.14)
print("ndarray1.shape: ", ndarray1.shape)

print("n_row", ndarray1.shape[0])
print("n_col", ndarray1.shape[1])

#%% arange
a = np.arange(5, 100, 2)
print(a)


#%% linspace
a = np.linspace(-10, 10, 21)
print(a)

























