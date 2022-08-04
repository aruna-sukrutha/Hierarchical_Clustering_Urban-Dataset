#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import random
import math

#Reading centroid dataset with pandas as dataframe
df = pd.read_csv("urbanGB.centr.txt",sep="\n")
centroid=[]
for i in df:
    a=i.split(",") 
    a[0]=float(a[0])
    a[1]=float(a[1])
    centroid.append(a)
df=df.values.tolist()
for data in df:
    a=data[0].split(",")
    #explicit type converting string data points into float
    a[0]=float(a[0]) 
    a[1]=float(a[1])
    #adding data to centroid list in form of array
    centroid.append(a)
print("Given Centroid data points are:\n",centroid)
print("\nLength of centroid list is:",len(centroid))

#reading centroid data as table using dataframe
pd.DataFrame(centroid)


# In[5]:


def measuring_distance(a,b):
    #euclidean formular to find distance between two points a,b
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2) 
def distance_matrix(centroid):
    n=len(centroid)
    output=np.ones((n,n))
    for i in range(n):
        for j in range(n):
            output[i][j]=measuring_distance(centroid[i],centroid[j])
    return output
print("Distance matrix:\n",distance_matrix(centroid))


# In[6]:


#Average linkage
cluster_distance=distance_matrix(centroid)
#print("Distance matrix:\n",cluster_distance)
init_cluster=[[i] for i in range(len(cluster_distance))]
while len(cluster_distance)>2:
    #find min value other than zero
    m = np.where(cluster_distance==np.min(cluster_distance[np.nonzero(cluster_distance)]))[0] 
    for i in range(len(cluster_distance)):
        if i not in [m[0],m[1]]:
            #average linkage
            cluster_distance[m[0]][i]=(cluster_distance[m[0]][i]+cluster_distance[m[1]][i])/2
            cluster_distance[i][m[0]]=cluster_distance[m[0]][i]
            #deleting min value row, column of distance matrix
    cluster_distance=np.delete(cluster_distance,m[1],axis=0)
    cluster_distance=np.delete(cluster_distance,m[1],axis=1)
    #OUTPUTS
    print("\nNumber of items in the first cluster:",init_cluster[m[0]])
    print("\nNumber of items in the second cluster:",init_cluster[m[1]])
    init_cluster[m[0]].append([init_cluster.pop(m[1])])
    print("\nDistance between these two clusters before they joined:",np.min(cluster_distance[np.nonzero(cluster_distance)]))
print("\nAverage linkage final clusters:\n",init_cluster)

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
X = pdist(cluster_distance)
Z = linkage(X, 'average')
#print(Z)
print("\nThe final distance matrix is:\n",cluster_distance)
pd.DataFrame(cluster_distance)

