###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Define metrics to assess linear and logistic regression    :                    #
# - the posterior including the Laplace approximation                             #                                    #
# - the KL divergence                                                             #                                                                 #
###################################################################################

import numpy as np 
import numpy.linalg as LA
import math
from .KUtils import graphix, bayesianlogisticPdf,neglogisticPdf
from .BayesianLogisticReg import LaplaceLogisticRegression
from scipy import optimize

############## METRICS FOR LINEAR REGRESSION ##########################        
#TO DO

############## METRICS FOR LOGISTIC REGRESSION ##########################    
        
class PosteriorLogReg(object):
    
    def __init__(self,theta0, Cov0):
        self._theta0=theta0 # the initial guess
        self._Cov0=Cov0 # the initial covariance (ie uncertainty on the initial guess)
        self._X=None
        self._Y=None
        self._Z=None
        self._mle=None
        self._map=None
        self._lapCov=None
        self._xv=None
        self._yv=None
        self._pdf=None
        
    # compute the posterior (un-normalized by default)
    def logPdf(self,theta):
        return bayesianlogisticPdf(theta,self._theta0, self._Cov0,self._X,self._Y,self._Z)
        
    def fit(self,X,Y,Z=1,Nsamples=30):
        self._X=X
        self._Y=Y
        self._Z=Z
        N,d=X.shape
        
        
        # solve with Laplace to find the center of the grid
        lap = LaplaceLogisticRegression(self._theta0,self._Cov0).fit(X, Y.reshape(N,))
        self._map=lap.theta
        self._lapCov=lap.Cov
        sol=optimize.minimize(neglogisticPdf, self._theta0, args=(X,Y.reshape(N,),Z,),method='L-BFGS-B')
        self._mle=sol.x
        
        if d==2:
            std=np.max(np.sqrt(np.linalg.eigvals(self._lapCov))).item()*3
    
            # generate the true posterior for different values of theta
            theta1=np.linspace(self._map[0]-std,self._map[0]+std,Nsamples)
            theta2=np.linspace(self._map[1]-std,self._map[1]+std,Nsamples)
            self._pdf=np.zeros((Nsamples,Nsamples))    
            self._xv,self._yv=np.meshgrid(theta1,theta2)
            for i in np.arange(0,Nsamples):
                for j in np.arange(0,Nsamples):
                    theta=np.zeros((2,1))
                    theta[0]=self._xv[i,j]
                    theta[1]=self._yv[i,j]
                    self._pdf[i,j]=math.exp(self.logPdf(theta))
                
        return self
    
    def plot(self,ax,labelize=True,showMleMap=True):
        ax.contour(self._xv,self._yv,self._pdf,6,zorder=1)
        if showMleMap: 
            if labelize:
                ax.scatter(self._mle[0],self._mle[1],marker='o',color='g',label='MLE',linewidth=2)
                ax.scatter(self._map[0],self._map[1],marker='o',color='r',label='MAP',linewidth=2)
                graphix.plot_ellipsoid2d(ax,self._map,self._lapCov,col='r',linewidth=1.2,zorder=3,linestyle='-',label='Laplace')
            else:
                ax.scatter(self._mle[0],self._mle[1],marker='o',color='g')
                ax.scatter(self._map[0],self._map[1],marker='o',color='r')
                graphix.plot_ellipsoid2d(ax,self._map,self._lapCov,col='r',linewidth=1.2,zorder=3,linestyle='-')

class KL:
    
    @staticmethod
    def divergence(theta,Cov,normalSamples,truePosterior):
        d=theta.shape[0]
        theta=theta.reshape(d,1)
        entropy=0.5*math.log(LA.det(Cov))+d/2*(1+math.log(2*math.pi))
        A=0
        thetaVec=theta+np.linalg.cholesky(Cov).dot(normalSamples.T)
        cmpt=0
        nbSamplesKL=normalSamples.shape[0]
        for i in range(0,nbSamplesKL):
            thetai=thetaVec[:,i].reshape(d,1)
            A=A-truePosterior.logPdf(thetai)
        KL=A/nbSamplesKL-entropy
        return KL.item()

    @staticmethod
    def history(onlineBayes,truePosterior,nbSamplesKL,seed):
        N,d=onlineBayes.history_theta.shape
        history_kl=np.zeros([N,1])
        np.random.seed(seed)
        normalSamples=np.random.multivariate_normal(np.zeros(d,),np.identity(d),size=(nbSamplesKL,))
        for t in range(0,N):
            thetat=onlineBayes.history_theta[t].reshape(d,1)
            Covt=onlineBayes.history_Cov[t].reshape(d,d)
            history_kl[t]=KL.divergence(thetat,Covt,normalSamples,truePosterior)
        return history_kl  
    
        
    

