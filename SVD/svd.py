import numpy as np

## Calculate the mean
from sklearn.decomposition import IncrementalPCA
avg = np.zeros((160000))
nalltest = 0
for i in range(7,17):
    data = np.load("../../sat_precip/b{}_30.npy".format(i))
    data = data.reshape(data.shape[0],-1)
    avg += np.sum(data, axis=0)
    nalltest += data.shape[0]
avg = avg / nalltest

## Calculate the PCA operator
ranformer = IncrementalPCA(n_components = 100, batch_size=data.shape[0])
for i in range(7,17):
    data = np.load("../../sat_precip/b{}_30.npy".format(i))
    data = data.reshape(data.shape[0],-1)
    data = data - avg
    tranformer.partial_fit(data)
    print("loaded {}".format(i))

# For saving/loading already calculated PCA
import pickle
with open('pca.pickle', 'wb') as handle:
    pickle.dump(tranformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('avg.pickle', 'wb') as handle:
    pickle.dump(avg, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('pca.pickle', 'rb') as handle:
    tranformer = pickle.load(handle)
with open('avg.pickle', 'rb') as handle:
    avg = pickle.load(handle)

## Start comparing matrix
from sklearn.metrics import mean_squared_error
distance_matrix = np.zeros((10,10))
for i in range(7,17):
    for j in range(i+1,17):
        if i != j:
            data = np.load("../../sat_precip/b{}_30.npy".format(i))
            data = data.reshape(data.shape[0],-1)
            data1 = tranformer.transform(data-avg)
            data = np.load("../../sat_precip/b{}_30.npy".format(j))
            data = data.reshape(data.shape[0],-1)
            data2 = tranformer.transform(data-avg)
            distance = mean_squared_error(data1[z], data2[z])
            distance_matrix[i-7,j-7] = distance
            print(str(i)+"-> "+str(j)+" = "+str(distance))

from scipy.spatial.distance import euclidean
distance_matrix2 = np.zeros((10,10))
for i in range(7,17):
    for j in range(i+1,17):
        data = np.load("../../sat_precip/b{}_30.npy".format(i))
        data = data.reshape(data.shape[0],-1)
        data1 = tranformer.transform(data-avg)
        data = np.load("../../sat_precip/b{}_30.npy".format(j))
        data = data.reshape(data.shape[0],-1)
        data2 = tranformer.transform(data-avg)
        distance = 0
        for z in range(data.shape[0]):
            distance += euclidean(data1[z], data2[z])
        distance_matrix2[i-7,j-7] = distance
        print(str(i)+"-> "+str(j)+" = "+str(distance))
from scipy.spatial.distance import cosine
distance_matrix3 = np.zeros((10,10))
for i in range(7,17):
    for j in range(i+1,17):
        if i != j:
            data = np.load("../../sat_precip/b{}_30.npy".format(i))
            data = data.reshape(data.shape[0],-1)
            data1 = tranformer.transform(data-avg)
            data = np.load("../../sat_precip/b{}_30.npy".format(j))
            data = data.reshape(data.shape[0],-1)
            data2 = tranformer.transform(data-avg)
            distance = 0
            for z in range(data.shape[0]):
                distance += cosine(data1[z], data2[z])
            distance_matrix3[i-7,j-7] = distance
            print(str(i)+"-> "+str(j)+" = "+str(distance))
np.savetxt("mse.csv", distance_matrix, delimiter=",")
np.savetxt("euclidian.csv", distance_matrix2, delimiter=",")
np.savetxt("cosine.csv", distance_matrix3, delimiter=",")
