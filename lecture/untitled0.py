#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:57:17 2021

@author: ijin
"""

#%%
import numpy as np 
A = np.array([
    [1,-1,2],
    [3,2,2],
    [4,1,2],
     ])

print(A)


#%%
c = np.random.rand(3,3)
print(c)

#%%
d = np.zeros((2,4))
print(d)

#%%
print(A[0][2])

#%%
print(A+c)
print(5*A)
print(A@c)

#%%
#transpose
A_trans = np.transpose(A)
print(A_trans)
print(A.T)

#identity
I = np.identity(3)
print(I)

print(A@I)

#inverse
A_inverse = np.linalg.pinv(A)
print(A_inverse)

print(A@A_inverse)


#%%
#선형시스템

'''
2*X0 -4*X1 +X2 = 3
3*X0+X1-6X2 = 10 
X0+X1+X2=5

[3 1 -6]
[1 1 1 ]

[2 -4 1] [X0     [ 3
[3 1 -6]  X1   =  10
[1 1 1 ]  X2]     5 ] 

'''
