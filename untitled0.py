# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:16:22 2020

@author: mjkiqce3
"""
import numpy as np
X=np.arange(25)
y=np.abs(X-7)+np.random.normal(0,2,len(X))
np.savetxt('inn1.out',X, fmt = '%4.6f', newline = '\n')
np.savetxt('out1.out',y, fmt = '%4.6f', newline = '\n')

t = np.arange(2000)
intercept = -45.0
slope = 0.7
v = intercept + slope*t + np.random.normal(0, 1, 2000)
np.savetxt('inn2.out',t, fmt = '%4.6f', newline = '\n')
np.savetxt('out2.out',v, fmt = '%4.6f', newline = '\n')

t = np.arange(1900, 2000)
v = t % 20
np.savetxt('inn3.out',t, fmt = '%4.6f', newline = '\n')
np.savetxt('out3.out',v, fmt = '%4.6f', newline = '\n')


t1 = [t for t in range(100)]
v1 = [v for v in np.random.normal(3, 1, 100)]
t2 = [t for t in range(99, 199)]
v2 = [v for v in np.random.normal(20, 1, 100)]
t = t1 + t2
v = v1 + v2

np.savetxt('inn4.out',t, fmt = '%4.6f', newline = '\n')
np.savetxt('out4.out',v, fmt = '%4.6f', newline = '\n')



