# 
# Random Swap 
# G. I. Choudhary
# 24.4.2022
# common parameters:
# X: data set
# C: centroids

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import random as rand

__version__="V1.0"
class rs(KMeans):
    def get_version():
        return __version__
    
    def __init__(self, n_init=1, **kwargs):
        """ n_init: number of times k-means run initially
            kwargs: arguments for scikit-learns KMeans """
            
        super().__init__(n_init=n_init,  init='random', **kwargs)
        
    def get_error(self, X, C):
        """compute error per centroid"""
        # squared distances between data and centroids
        dist = cdist(X, C, metric="sqeuclidean")
        # indices to nearest centroid
        dist_min = np.argmin(dist,axis=1)
        # distances to nearest centroid
        d1 = dist[np.arange(len(X)), dist_min]
        # aggregate error for each centroid
        return np.array([np.sum(d1[dist_min==i]) for i in range(len(C))])

    def _lloyd(self,C,X):
        """perform Lloyd's algorithm"""
        self.init = C # set cluster centers
        self.n_clusters = len(C) # set k-value
        super().fit(X) # Lloyd's algorithm, sets self.inertia_ (a.k.a. phi)

    def fit(self, X, dim,  gt, benchMark, avg, k, iterations):
        """ compute k-means clustering via breathing k-means (if m > 0) """
        
        # run k-means (unless 'init' parameter specifies differently)
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
            C[rand.choice(range(0,len(C)))] = X[rand.randint(0, len(X)-1)]
            self.cluster_centers_ = C
            
            self._lloyd(C,X)
            
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