__authors__ = [1637620, 1638322, 1638529]
__group__ = 'Grup09'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

#Mahalanobis, Euclidean
class KNN:
    def __init__(self, train_data, labels, distance = 'Euclidean'):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################
        self.distance = distance #Euclidean or Mahanalobis

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #if not train_data.type == float:
        #    train_data.astype(float)
    

        self.train_data = train_data.reshape((train_data.shape[0], np.prod(train_data.shape[1:])))
        if not type(self.train_data) == float:
            self.train_data.astype(float)
        

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.test_data = test_data.reshape((test_data.shape[0], np.prod(test_data.shape[1:])))
        if self.distance == 'Euclidean':
            Mdist = cdist(self.test_data,self.train_data)

        elif self.distance == 'Mahalanobis': #reduir el tamany del train_data perque peta al fer la inversa
            C = np.cov(self.train_data, rowvar=False)
            C = np.linalg.inv(C)
            Mdist = cdist(self.test_data,self.train_data, metric='mahalanobis', VI = C)
        
        elif self.distance == 'Cosine Similarity': #Basat en la similitud de les característiques no en les distàncies com a tal
            self.test_data = self.test_data / np.linalg.norm(self.test_data, axis = 1, keepdims=True)
            self.train_data = self.train_data / np.linalg.norm(self.train_data, axis = 1, keepdims=True)
            Mdist = 1 - cdist(self.test_data, self.train_data, metric='cosine')
            indx = np.argsort(Mdist,axis=1)
            self.neighbors = self.labels[indx[:,-k:]] #últimes k columnes (més altes)
            return 0
        
        elif self.distance == 'Jaccard':
            Mdist = 1 - cdist(self.test_data, self.train_data, metric = jaccard_distance)
            indx = np.argsort(Mdist,axis=1)
            self.neighbors = self.labels[indx[:,-k:]] #últimes k columnes (més altes)
            return 0
        
        elif self.distance == 'Canberra':
            Mdist = cdist(self.test_data, self.train_data, metric = canberra_distance)
        
        elif self.distance == 'Manhattan':
            Mdist = cdist(self.test_data, self.train_data, metric = 'cityblock')
        
        elif self.distance == 'Chebyshev':
            Mdist = cdist(self.test_data, self.train_data, metric = chebyshev_distance)

        
        indx = np.argsort(Mdist,axis=1)        
        self.neighbors = self.labels[indx[:,:k]] #Agafa les primers k columnes


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
     
        values = []
        percentage = []
        k = len(self.neighbors[0])
        for i,nrow in enumerate(self.neighbors):
            _, indx, recount = np.unique(nrow, return_inverse=True, return_counts=True)
            first_count = recount[indx]
            max_indx = np.argmax(first_count)
            values.append(self.neighbors[i][max_indx])

            percentage.append(np.max(first_count)/k)
        
        return values


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()


def jaccard_distance(u, v):
    intersection = np.sum(np.minimum(u, v))
    union = np.sum(np.maximum(u, v))
    return 1.0 - intersection / union

def canberra_distance(u, v):
    numerator = np.abs(u - v)
    denominator = np.abs(u) + np.abs(v)
    return np.sum(numerator / denominator)

def chebyshev_distance(u, v):
    return np.max(np.abs(u - v))