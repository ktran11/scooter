<<<<<<< HEAD
import numpy as np# We recommend to use numpy arrays
=======

import numpy as np   
>>>>>>> a5b6113feca8a20664b624fa45114a069ba85d67
from os.path import isfile
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
import pandas as pd # for using pandas daraframe
from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.feature_selection import GenericUnivariateSelect, f_regression
from sklearn.ensemble import BaggingRegressor

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
<<<<<<< HEAD
=======
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import BaggingRegressor


>>>>>>> a5b6113feca8a20664b624fa45114a069ba85d67

class model (BaseEstimator):

    
    def __init__(self):
<<<<<<< HEAD
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples= 38563
        self.num_feat=59
        self.num_labels=1
        self.is_trained=False
        self.preprocess = GenericUnivariateSelect(f_regression, 'k_best', param=self.num_feat)
        self.mod = BaggingRegressor(n_estimators=60,bootstrap=True,bootstrap_features=False,warm_start = False,n_jobs=1,oob_score=True)
=======
        self.num_train_samples=38563
        self.num_feat=59
        self.num_labels=1
        self.is_trained=False
        self.preprocess_Scaler=StandardScaler()
        self.PCA=PCA(0.99)
        self.SelectKbest=SelectKBest(score_func=f_regression,k="all")
        self.mod = BaggingRegressor(n_estimators=60,bootstrap=True,bootstrap_features=False,warm_start = False,n_jobs=1)
>>>>>>> a5b6113feca8a20664b624fa45114a069ba85d67
    

        
    def fit(self, X, y):
<<<<<<< HEAD
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        norm = np.linalg.norm(X)
        pca = PCA(0.80)
        pca.fit(X/norm)
        x_pca =pca. transform(X/norm)
        if (not self.is_trained):
            self.num_feat = pca.n_components_
        X_preprocess=self.preprocess.fit_transform(X,y)

        self.mod.fit(X_preprocess, y)
        self.is_trained = True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
    
        y=np.random.rand(X.shape[0])
        X_preprocess = self.preprocess.fit_transform(X,y)
        y = self.mod.predict(X)


=======
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        X_scaled=self.preprocess_Scaler.fit_transform(X)
        X_PCA=self.PCA.fit_transform(X_scaled)
        X_new=self.SelectKbest.fit_transform(X_PCA,y)
        self.mod.fit(X_new,y)
        self.is_trained = True

    def predict(self, X):
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        y = np.zeros([num_test_samples, self.num_labels])
        X_scaled=self.preprocess_Scaler.transform(X)
        X_PCA=self.PCA.transform(X_scaled)
        X_new=self.SelectKbest.transform(X_PCA)
        y = self.mod.predict(X_new)
>>>>>>> a5b6113feca8a20664b624fa45114a069ba85d67
        return y

    def save(self, path="./"):
        pass

    def load(self, path="./"):
        pass


def test():
    mod = model()
    X_random = np.random.rand(mod.num_train_samples,mod.num_feat)
    Y_random = np.random.rand(mod.num_train_samples,)
    mod.fit(X_random,Y_random)
    mod.predict(X_random)
   
if __name__ == "__main__":
    test()
