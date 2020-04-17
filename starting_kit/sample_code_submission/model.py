import numpy as np# We recommend to use numpy arrays
from os.path import isfile
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
import pandas as pd # for using pandas daraframe
from sklearn.preprocessing import StandardScaler # for standardizing the Data
from sklearn.feature_selection import GenericUnivariateSelect, f_regression
from sklearn.ensemble import BaggingRegressor

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator

class model (BaseEstimator):

    
    def __init__(self):
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
    
    def fit(self, X, y):
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
