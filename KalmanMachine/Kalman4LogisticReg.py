###################################################################################
# Sequential second order method for logistic regression :                        #
# The extended Kalman filter = online natural gradient                            #
# The quadratic Kalman filter= online version of the bounded variational approach #
# (A sikit learn like API is used but we support only the binary case and no bias #
#   ie classes_={0,1} and intercept_=0 )                                          #
###################################################################################

# import the Sikitlearn base model
from sklearn.base import BaseEstimator
from .KUtils import sigmoid
import numpy as np
import numpy.random
import numpy.linalg as LA

# The extended Kalman filter
class BayesianLogisticRegression(object):
    
    def __init__(self, theta0, Cov0, passNumber):
        self._theta0=theta0 # the initial guess
        self._Cov0=Cov0 # the initial covariance (ie uncertainty on the initial guess)
        self._passNumber=passNumber # the number of pass on datas
        self._theta=np.copy(self._theta0) # the current mean
        self._Cov=np.copy(self._Cov0) # the current covariance
        self._history_theta=None # the mean history
        self._history_Cov=None  # the covariance history
     
    def init_history(self,X,y):
        N,d=X.shape
        self._history_theta = np.zeros((N*self._passNumber+1,d))
        self._history_theta[0,:]=self._theta0.flatten()
        self._history_Cov = np.zeros((N*self._passNumber+1,d*d))
        self._history_Cov[0,:]=self._Cov0.flatten()
    
    # virtual methods
    def fit(self,X,y):        
        return self
    
    def predict(self,X):
        return
    
    def predict_proba(self,X):
        return
    
    @property
    def theta(self):
        return self._theta
    
    @property
    def Cov(self):
        return self._Cov
    
    @property
    def history_theta(self):
        return self._history_theta
    
    @property
    def history_Cov(self):
        return self._history_Cov
    



class EKFLogisticRegression(BayesianLogisticRegression):
    
    def __init__(self, theta0, Cov0, passNumber=1):
        super().__init__(theta0, Cov0, passNumber)
        print(self._passNumber)
    
    def fit(self,X,y):
        N,d=X.shape
        super().init_history(X,y)
        nbIter=0
        for numeroPass in range(1,self._passNumber+1):   
            for t in range(0,N): 
                # get sample
                yt=y[t].reshape(1,1)
                xt=X[t].reshape(d,1)
            
                # compute R
                mu=sigmoid(xt.T.dot(self._theta)) 
                R=max(mu*(1-mu),1e-12)
                H=R*xt.T

                # prediction error
                err=yt-mu
        
                # computation of optimal gain
                S=R+H.dot(self._Cov).dot(H.T)
                K=self._Cov.dot(H.T).dot(LA.inv(S))
        
                # update state and covariance of state
                self._theta=self._theta+K.dot(err)
                self._Cov=self._Cov-K.dot(H).dot(self._Cov)
                self._history_theta[nbIter+1,:]=self._theta.flatten()
                self._history_Cov[nbIter+1,:]=self._Cov.flatten()
            
                nbIter=nbIter+1
        
            if numeroPass>1:
                # To manage different pass, shuffle the dataset
                DataSet=list(zip(X,y))
                random.shuffle(DataSet)
                X,y = zip(*DataSet)
                
        return self
    
    
    def predict(self,X):
        return sigmoid(X.dot(self._theta))>0.5
    
    def predict_proba(self,X):
        return sigmoid(X.dot(self._theta))
