# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 00:39:10 2021

@author: Azum
"""

from sklearn import preprocessing
import numpy as np


a1 = np.random.randint(-10,10,10)

a2 = np.random.randint(-10,10,10)

stacking = np.vstack((a1,a2))

normstack = preprocessing.normalize(stacking, norm='l2')

al1 = normstack[0,:]
al2 = normstack[1,:]

euclidean_distance_not_normalized = np.linalg.norm(a1-a2)
euclidean_distance_normalized = np.linalg.norm(al1-al2)
