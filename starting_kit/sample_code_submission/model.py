
import numpy as np   
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import BaggingRegressor



class model (BaseEstimator):
    def __init__(self):
        self.num_train_samples=38563
        self.num_feat=59
        self.num_labels=1
        self.is_trained=False
        self.preprocess_Scaler=StandardScaler()
        self.PCA=PCA(0.99)
        self.SelectKbest=SelectKBest(score_func=f_regression,k="all")
        self.mod = BaggingRegressor(n_estimators=60,bootstrap=True,bootstrap_features=False,warm_start = False,n_jobs=1)
    

        
    def fit(self, X, y):
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
        return y

    def save(self, outname='model'):
        ''' Placeholder function.
            Save the trained model to avoid re-training in the future.
        '''
        pass
        
    def load(self):
        ''' Placeholder function.
            Load a previously saved trained model to avoid re-training.
        '''
        pass
