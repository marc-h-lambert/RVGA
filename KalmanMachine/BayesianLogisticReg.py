###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Bayesian logistic regression based on the Gaussian approximation of posterior   #
# BayesianLogisticRegression:  the general API                                    #
# LaplaceLogisticRegression: batch Laplace method                                 #
# OnlineBayesianLogisticRegression : the API for sequential algorithms            #
#                                    defined in Kalman4LogisticRegression.py      #
# (A sikit learn like API is used but we support only the binary case and no bias #
#   ie classes_={0,1} and intercept_=0 )                                          #
###################################################################################

import numpy as np
import numpy.random
import numpy.linalg as LA
from .KUtils import sigmoid, graphix, negbayesianlogisticPdf
from matplotlib import cm
from scipy import optimize
import math

# Bayesian Logistic Regression (with a Gaussian model)
class BayesianLogisticRegression(object):
    
    def __init__(self,theta0, Cov0):
        super().__init__()
        self._theta0=theta0 # the initial guess
        self._Cov0=Cov0 # the initial covariance (ie uncertainty on the initial guess)
        self._theta=np.copy(self._theta0) # the current mean
        self._Cov=np.copy(self._Cov0) # the current covariance
     
    # virtual methods
    def fit(self,X,y):        
        return self
    
    def predict(self,X):
        return np.multiply(self.predict_proba(X)>0.5,1)
    
    #prediction of N outputs for inputs X=(N,d)
    #use approximation of the integration over a Gaussian
    def predict_proba(self,X):
        N,d=X.shape
        beta=math.sqrt(8/math.pi)
        vec_nu=np.diag(X.dot(self._Cov).dot(X.T))
        k=beta/np.sqrt(vec_nu+beta**2).reshape(N,1)
        return sigmoid(k*X.dot(self._theta))
    
    def plotPredictionMap(self,ax,size=6):
        N=100
        x=np.zeros([2,1])
        theta1=np.linspace(-size/2,size/2,N)
        theta2=np.linspace(-size/2,size/2,N)
        probaOutput=np.zeros((N,N)) 
        xv,yv=np.meshgrid(theta1,theta2)
        for i in np.arange(0,N):
            for j in np.arange(0,N):
                x[0]=xv[i,j]
                x[1]=yv[i,j]
                probaOutput[i,j]=self.predict_proba(x.T)
        contr=ax.contourf(xv,yv,probaOutput,20,zorder=1,cmap='jet')
        ax.set_xlim(-size/2, size/2)
        ax.set_ylim(-size/2, size/2)
        return contr

    @property
    def theta(self):
        return self._theta
    
    @property
    def Cov(self):
        return self._Cov

# Batch Laplace version of Bayesian Logistic Regression (with a Gaussian model)
class LaplaceLogisticRegression(BayesianLogisticRegression):
    
    # !!! implement only for theta0=0 and Cov0=sigma0^2 I, 
    # otherwise using sikit.logreg method produce biased maximum posterior
    def __init__(self, theta0, Cov0):
        super().__init__(theta0, Cov0)

    def fit(self,X,y):
        N,d=X.shape
        sol=optimize.minimize(negbayesianlogisticPdf, self._theta0, args=(self._theta0,self._Cov0,X,y.reshape(N,),1,),method='L-BFGS-B')
        self._theta=sol.x
        
        # the Hessian 
        L=sigmoid(X.dot(self._theta))
        K=(L*(1-L)).reshape(N,1,1)
        A=X[...,None]*X[:,None]
        H=np.sum(K*A,axis=0)+LA.inv(self._Cov0)
        self._Cov=LA.inv(H)
        return self
    
    def plotEllipsoid(self,ax,u=0,v=1,labelize=True):
        d=self._theta.shape[0]
        thetaproj,Covproj=graphix.projEllipsoid(self._theta,self._Cov.reshape(d,d),u,v)
        if labelize:
            graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,col='r',linewidth=1.2,zorder=3,linestyle='-',label='Laplace')
        else:
            graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,col='r',linewidth=1.2,zorder=3,linestyle='-')
        ax.scatter(self._theta[0],self._theta[1],color='r')
                
        
# Stochastic version of Bayesian Logistic Regression (with a Gaussian model)
class OnlineBayesianLogisticRegression(BayesianLogisticRegression):
    
    def __init__(self, theta0, Cov0, passNumber=1):
        super().__init__(theta0, Cov0)
        self._passNumber=passNumber # the number of pass on datas
        self._history_theta=None # the mean history
        self._history_Cov=None  # the covariance history
    
    # virtual method
    def update(self,xt,yt):
        pass
        
    def fit(self,X,y):
        N,d=X.shape
        self._history_theta = np.zeros((N*self._passNumber+1,d))
        self._history_theta[0,:]=self._theta0.flatten()
        self._history_Cov = np.zeros((N*self._passNumber+1,d*d))
        self._history_Cov[0,:]=self._Cov0.flatten()
        nbIter=0
        for numeroPass in range(1,self._passNumber+1):   
            for t in range(0,N): 
                # get new observation
                yt=y[t].reshape(1,1)
                xt=X[t].reshape(d,1)
                
                # update theta and Cov using the new observation
                self.update(xt,yt)
                
                self._history_theta[nbIter+1,:]=self._theta.flatten()
                self._history_Cov[nbIter+1,:]=self._Cov.flatten()
            
                nbIter=nbIter+1
        
            if numeroPass>1:
                # To manage different pass, shuffle the dataset
                DataSet=list(zip(X,y))
                random.shuffle(DataSet)
                X,y = zip(*DataSet)
                
        return self
        
    @property
    def history_theta(self):
        return self._history_theta
    
    @property
    def history_Cov(self):
        return self._history_Cov
    
    # u, v: coordinates of projection
    def plotEllipsoid(self,ax,u=0,v=1,nbLevels=6,labelize=True):
    
        colorMap = cm.get_cmap('Greys', 256)
        colorCovs = colorMap(np.linspace(0.5, 1, nbLevels+2))

        d=self._history_theta[0].shape[0]
        
        # print ellipsoids at first step
        if nbLevels>0:
            thetaproj,Covproj=graphix.projEllipsoid(self._history_theta[1],self._history_Cov[1].reshape(d,d),u,v)
            if labelize:
                graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,colorCovs[0],linewidth=1.2,zorder=3,label='first iteration',linestyle='-')
            else:
                graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,colorCovs[0],linewidth=1.2,zorder=3,linestyle='-')
                
        # print ellipsoids at intermediate step
        l=int(min(self._history_theta.shape[0],100)/(nbLevels+1))
        idx=l   
        col=0
        for i in range(0,nbLevels):
            col=col+1
            thetaproj,Covproj=graphix.projEllipsoid(self._history_theta[idx],self._history_Cov[idx].reshape(d,d),u,v)
            graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,colorCovs[col])
            idx=idx+l
    
        # print ellipsoids at last step
        thetaproj,Covproj=graphix.projEllipsoid(self._history_theta[-1],self._history_Cov[-1].reshape(d,d),u,v)
        if labelize:
            graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,colorCovs[-1],label='Last iteration',linewidth=1.2,linestyle='-')
        else:
            graphix.plot_ellipsoid2d(ax,thetaproj,Covproj,colorCovs[-1],linewidth=1.2,linestyle='-')
            
