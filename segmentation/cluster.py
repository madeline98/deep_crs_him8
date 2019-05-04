# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:10:44 2019

@author: RMC
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pylab import imread,subplot,imshow,show
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from PIL import Image

b7 = np.load("D:/deep_crs_him8-master/b7_30.npy")[:100]
grayed = []
dataset = []
for i in b7:
    #img = Image.fromarray(i)
    #i = img.convert('L')
    #grayed.append(i)
    #k = np.array(i)
    q = i.reshape(1, i.shape[0]*i.shape[1])
    dataset.append(q[0])

#plt.imshow(grayed[10])


kmeans = KMeans(n_clusters=10, random_state=0).fit(dataset)
print (kmeans.labels_)

'''
sums = np.zeros((100, 100))
for x in range(len(grayed)):
    for i in range(100):
        for j in range(100):
            sums[i][j] += dataset[x][i][j]
#print(sums)
for idx in range(100):
    for idy in range(100):
        sums[idx][idy] /= len(dataset)
#print(sums)
plt.imshow(sums)'''



  