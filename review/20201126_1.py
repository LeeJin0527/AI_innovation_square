import numpy as np

### list slicing

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(a)

b = a[1:7:2]
print(b)


#%% NumPy Indexing

a = np.random.uniform(0, 10, (3, 3)).astype(np.int)
print(a, '\n')
print(a[0,0], a[0, 1], a[0, 2])
print(a[1,0], a[1, 1], a[1, 2])
print(a[2,0], a[2, 1], a[2, 2], '\n')

for row_idx in range(a.shape[0]):
    for col_idx in range(a.shape[1]):
        print(a[row_idx, col_idx])
        
#%% NumPy Slicing
a = np.random.uniform(0, 10, (5, 5)).astype(np.int)
print(a, '\n')

print(a[0 , 0:4])
print(a[1 , 0:4])
print(a[0:2, 0:2])
print(a[0,:])
print(a[:,0])

#%% 1D Correlation
signal = np.random.normal(0, 1, (100, ))
filter_ = np.array([1, 5, 3, 2, 1])

n_signal, n_filter = signal.shape[0], filter_.shape[0]

correlations = np.empty(shape=(0, 0))
for co_idx in range(n_signal - n_filter):
    signal_segment = signal[co_idx : co_idx + n_filter]
    correlation = np.sum(signal_segment * filter_)
    correlations = np.hstack((correlations, correlation))

#%% 1D vs 2D Correlations

signal_segment = np.random.uniform(0, 10, (9,)).astype(np.int)
filter_ = np.random.uniform(0, 10, (9,)).astype(np.int)

print(signal_segment)
print(filter_)

correlation = np.sum(signal_segment * filter_)


signal_segment = signal_segment.reshape(3, 3)
filter_ = filter_.reshape(3, 3)
print(signal_segment)
print(filter_)

correlation = np.sum(signal_segment * filter_)

#%% image read
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

img = plt.imread('./test_image.jpg')
img_gray = rgb2gray(img)

fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(img_gray, 'gray')

ax.tick_params(left=False, labelleft=False,
                bottom=False, labelbottom=False)
fig.tight_layout()

#%% smoothing filter
filter_ = np.ones(shape=(3, 3))/9

l_filter = filter_.shape[0]

H, W = img_gray.shape
img_convolved = np.zeros(shape=(H-l_filter+1, W-l_filter+1))

for row_idx in range(img_gray.shape[0] - l_filter):
    for col_idx in range(img_gray.shape[1] - l_filter):
        img_segment = img_gray[row_idx:row_idx+l_filter,
                               col_idx:col_idx+l_filter]
        convolution = np.sum(img_segment * filter_)
        img_convolved[row_idx, col_idx] = convolution

fig, axes = plt.subplots(1, 2, figsize=(25, 12))
axes[0].imshow(img_gray, 'gray')
axes[1].imshow(img_convolved, 'gray')

#%% sobel x, y filters
filter_ = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

l_filter = filter_.shape[0]

H, W = img_gray.shape
img_convolved = np.zeros(shape=(H-l_filter+1, W-l_filter+1))

for row_idx in range(img_gray.shape[0] - l_filter):
    for col_idx in range(img_gray.shape[1] - l_filter):
        img_segment = img_gray[row_idx:row_idx+l_filter,
                               col_idx:col_idx+l_filter]
        convolution = np.sum(img_segment * filter_)
        img_convolved[row_idx, col_idx] = convolution

fig, axes = plt.subplots(1, 3, figsize=(25, 12))
axes[0].imshow(img_gray, 'gray')
axes[1].imshow(img_convolved, 'gray')

filter_ = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

l_filter = filter_.shape[0]

H, W = img_gray.shape
img_convolved = np.zeros(shape=(H-l_filter+1, W-l_filter+1))

for row_idx in range(img_gray.shape[0] - l_filter):
    for col_idx in range(img_gray.shape[1] - l_filter):
        img_segment = img_gray[row_idx:row_idx+l_filter,
                               col_idx:col_idx+l_filter]
        convolution = np.sum(img_segment * filter_)
        img_convolved[row_idx, col_idx] = convolution
axes[2].imshow(img_convolved, 'gray')
fig.tight_layout()

#%% Laplacian filter
filter_ = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])

l_filter = filter_.shape[0]

H, W = img_gray.shape
img_convolved = np.zeros(shape=(H-l_filter+1, W-l_filter+1))

for row_idx in range(img_gray.shape[0] - l_filter):
    for col_idx in range(img_gray.shape[1] - l_filter):
        img_segment = img_gray[row_idx:row_idx+l_filter,
                               col_idx:col_idx+l_filter]
        convolution = np.sum(img_segment * filter_)
        img_convolved[row_idx, col_idx] = convolution

fig, axes = plt.subplots(1, 2, figsize=(25, 12))

axes[0].imshow(img_gray, 'gray')
axes[1].imshow(img_convolved, 'bwr')   
#%% sharpening filter
filter_ = np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]])

l_filter = filter_.shape[0]

H, W = img_gray.shape
img_convolved = np.zeros(shape=(H-l_filter+1, W-l_filter+1))

for row_idx in range(img_gray.shape[0] - l_filter):
    for col_idx in range(img_gray.shape[1] - l_filter):
        img_segment = img_gray[row_idx:row_idx+l_filter,
                               col_idx:col_idx+l_filter]
        convolution = np.sum(img_segment * filter_)
        img_convolved[row_idx, col_idx] = convolution

#%% 연습: 얼굴 블러처리하기
filter_ = np.ones(shape=(31, 31))/(31*31)

l_filter = filter_.shape[0]

H, W = img_gray.shape
img_convolved = np.zeros(shape=(H-l_filter+1, W-l_filter+1))

for row_idx in range(img_gray.shape[0] - l_filter):
    for col_idx in range(img_gray.shape[1] - l_filter):
        
        if row_idx >= 150 and row_idx <= 470 and col_idx >= 280 and col_idx <= 540:
            img_segment = img_gray[row_idx:row_idx+l_filter,
                                   col_idx:col_idx+l_filter]
            convolution = np.sum(img_segment * filter_)
            img_convolved[row_idx, col_idx] = convolution
            
        else:
            img_convolved[row_idx, col_idx] = img_gray[row_idx, col_idx]

fig, axes = plt.subplots(1, 2, figsize=(25, 12))
axes[0].imshow(img_gray, 'gray')
axes[1].imshow(img_convolved, 'gray')





















