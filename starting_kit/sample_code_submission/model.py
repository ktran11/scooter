import numpy as np   
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler 
from sklearn.feature_selection import SelectKBest, f_regression




class model (BaseEstimator):
    def __init__(self):
        self.num_train_samples=38563
        self.num_feat=59
        self.num_labels=1
        self.is_trained=False
        self.preprocess_Scaler=StandardScaler()
        self.imp = QuantileTransformer(output_distribution='normal', random_state=0)
        self.SelectKbest = SelectKBest(score_func=f_regression,k=8)
        self.mod = BaggingRegressor(n_estimators = 50,n_jobs =1, warm_start = False)
    

        
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
        X_imp = self.imp.fit_transform(X_scaled)
        X_selected=self.SelectKbest.fit_transform(X_imp,y)
        self.mod.fit(X_selected,y)
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
        X_imp = self.imp.transform(X_scaled)
        X_selected=self.SelectKbest.transform(X_imp)
        y = self.mod.predict(X_selected)
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
        
def test():
    mod = model()
    X_random = np.random.rand(mod.num_train_samples,mod.num_feat)
    Y_random = np.random.rand(mod.num_train_samples,)
    mod.fit(X_random,Y_random)
    mod.predict(X_random)
    # 1 - créer un data X_random et y_random fictives: utiliser https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.rand.html
    # 2 - Tester l'entrainement avec mod.fit(X_random, y_random)
    # 3 - Test la prediction: mod.predict(X_random)
    # Pour tester cette fonction *test*, il suffit de lancer la commande ```python sample_code_submission/model.py```

if __name__ == "__main__":
    test()

