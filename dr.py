# 
# DR: Deterministic removal, Random addition 
# @Auth G. I. Choudhary 
# 25.4.2022
# common parameters:
# X: data set
# C: centroids

import numpy as np
import random as rand
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

__version__="V1.1"
class dr(KMeans):
    def get_version():
        return __version__
    def __init__(self, n_init=1, **kwargs):
        """ n_init: number of times k-means run initially
            kwargs: arguments for scikit-learns KMeans """
        super().__init__(n_init=n_init,  init='random', **kwargs)

    

    def _lloyd(self,C,X):
        """perform Lloyd's algorithm"""
        self.init = C # set cluster centers
        self.n_clusters = len(C) # set k-value
        super().fit(X) # Lloyd's algorithm, sets self.inertia_ (a.k.a. phi)

    def fit(self, X, iterations):
        """ DD: deterministic removal of centroid and deterministic addition of centroid """
        
        # run k-means++ (unless 'init' parameter specifies differently)
        super().fit(X) # requires self.n_clusters >= 1
        
        
        # handle trivial case k=1
        if self.n_clusters == 1:
            return self
        
        # memorize best error and codebook so far
        E_best = self.inertia_
        C_best = self.cluster_centers_
        l_best = self.labels_
       
        
        tmp = self.n_init, self.init # store for compatibility with sklearn
        
        for i in range(0,iterations):
            C = self.cluster_centers_
            
            cindx = rand.choice(self.getClosestCentroids(C))
            
            #swap centroid with a random data point from high distortion cluster
            C[cindx] = X[rand.choice(range(0,len(X)))]
            
            self.cluster_centers_ = C
            
            # add m centroids ("breathe in") and run Lloyd's algorithm
            self._lloyd(C,X)
            #print(self.cluster_centers_)
            if self.inertia_ < E_best*(1-self.tol):
                # improvement! update memorized best error and codebook so far
                E_best = self.inertia_
                C_best = self.cluster_centers_
                l_best = self.labels_   
                
                
        self.n_init, self.init = tmp # restore for compatibility with sklearn
        self.inertia_ = E_best
        self.cluster_centers_ = C_best
        self.labels_ = l_best
                
        return self
    
    def getClosestCentroids(self, C):
        """get closest pair of centroid"""
        
        # mutual distances among centroids (kxk-matrix)
        c_dist = cdist(C, C, metric="sqeuclidean")
        
        # index of nearest neighbor for each centroid
        nearest_neighbor=c_dist.argpartition(kth=1)[:,1]
        
        #indexes of closest centroids
        distance_neighbors = []
        for i in nearest_neighbor:
             distance_neighbors.append(c_dist[i][nearest_neighbor[i]])
        minIndx =  np.argmin(distance_neighbors)
        
        return [minIndx, nearest_neighbor[minIndx]]