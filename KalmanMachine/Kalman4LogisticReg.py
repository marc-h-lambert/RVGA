###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Online second order method for logistic regression :                            #
# The extended Kalman filter = online natural gradient                            #
# --> see "Online natural gradient as a Kalman filter, Yann Olivier 2018"         #                
# The quadratic Kalman filter= online version of the bounded variational approach #
# --> see "A variational approach to Bayesian logistic regression models \        #
#     and their extensions, Jaakkola and Jordan 1997"                             #
# The recursive VGA implicit                                                      #
# The recursive VGA explicit                                                      #
# --> see "The recursive variational Gaussian approximation (R-VGA),              #
#     Marc Lambert, Silvere Bonnabel and Francis Bach 2020"                       #                                                                                                                  
###################################################################################

from .KUtils import sigmoid, sigp, sigpp
import numpy as np
import numpy.random
import numpy.linalg as LA
from .BayesianLogisticReg import OnlineBayesianLogisticRegression
import math
from math import log, exp
from scipy import optimize

class EKFLogReg(OnlineBayesianLogisticRegression):
    
    def __init__(self, theta0, Cov0, passNumber=1):
        super().__init__(theta0, Cov0, passNumber)
    
    def update(self,xt,yt):
        # intermediate variables
        nu=xt.T.dot(self._Cov.dot(xt))
        Pu=self._Cov.dot(xt)
            
        a=xt.T.dot(self._theta)
        
        m=sigp(a)
        m=max(m,1e-100)
        
        # update state
        self._Cov=self._Cov-np.outer(Pu,Pu)/(1/m+nu)
        self._theta=self._theta+self._Cov.dot(xt)*(yt-sigmoid(a))
        
    # equivalent formulas using the Kalman form
    # def update(self,xt,yt):
    #     # compute R
    #     mu=sigmoid(xt.T.dot(self._theta)) 
    #     R=max(mu*(1-mu),1e-100)
    #     H=R*xt.T

    #     # prediction error
    #     err=yt-mu
        
    #     # computation of optimal gain
    #     S=R+H.dot(self._Cov).dot(H.T)
    #     K=self._Cov.dot(H.T).dot(LA.inv(S))
        
    #     # update state and covariance of state
    #     self._theta=self._theta+K.dot(err)
    #     self._Cov=self._Cov-K.dot(H).dot(self._Cov)
    
            
class QKFLogReg(OnlineBayesianLogisticRegression):
    
    def __init__(self, theta0, Cov0, passNumber=1):
        super().__init__(theta0, Cov0, passNumber)
    
    @staticmethod
    def eta(x):
        return -1/(2*x)*(sigmoid(x)-0.5)

    def update(self,xt,yt):
        # compute matrix R
        ksi=math.sqrt(xt.T.dot(self._Cov+np.outer(self._theta,self._theta)).dot(xt))
        invR=np.ones([1,1])*(-2*QKFLogReg.eta(ksi))
        R=LA.inv(invR)
            
        # compute gain K
        H=xt.T
        S=R+H.dot(self._Cov).dot(H.T)
        K=self._Cov.dot(H.T).dot(LA.inv(S))
                            
        #update theta
        self._theta=self._theta+K.dot(R.dot(yt-0.5)-H.dot(self._theta))
                
        #update Cov
        self._Cov=self._Cov-K.dot(H).dot(self._Cov)
      
    # equivalent formulas from Jordan paper:
    # def update(self,xt,yt):
    #     ksi=math.sqrt(xt.T.dot(self._Cov+np.outer(self._theta,self._theta)).dot(xt))
    #     print(ksi)
    #     P=LA.inv(LA.inv(self._Cov)-2*QKFLogReg2.eta(ksi)*np.outer(xt,xt))
        
    #     self._theta=P.dot(LA.inv(self._Cov).dot(self._theta)+(yt-0.5)*xt)
    #     self._Cov=P
            
class RVGALogReg(OnlineBayesianLogisticRegression):
    
    def __init__(self, theta0, Cov0, passNumber=1):
        super().__init__(theta0, Cov0, passNumber)
    
    beta=math.sqrt(8/math.pi)
            
    def fun2D(x,alpha0,nu0,y):
        k=RVGALogReg.beta/math.sqrt(exp(x[1])+RVGALogReg.beta**2)
        f=x[0]+nu0*sigmoid(x[0]*k)-alpha0-nu0*y
        g=exp(x[1])-nu0/(1+nu0*k*sigp(x[0]*k))
        return [f,g]

    def jac(x,alpha0,nu0,y):
        k=RVGALogReg.beta/math.sqrt(exp(x[1])+RVGALogReg.beta**2)
        kp=-0.5*RVGALogReg.beta*exp(x[1])/((exp(x[1])+RVGALogReg.beta**2)**(3/2))
        f_a=1+nu0*k*sigp(x[0]*k)
        f_gamma=nu0*x[0]*kp*sigp(x[0]*k)
        g_a=nu0**2*k**2*sigpp(x[0]*k)/((1+nu0*k*sigp(x[0]*k))**2)
        g_gamma=exp(x[1])+nu0**2*kp*(sigp(x[0]*k)+k*x[0]*sigpp(x[0]*k))/((1+nu0*k*sigp(x[0]*k))**2)
        return np.array([[f_a,f_gamma],[g_a,g_gamma]])
        
    def optim2D(alpha0,nu0,y):
        
        alphaMin=alpha0+nu0*y-nu0
        alphaMax=alpha0+nu0*y
        nuMin=nu0*(1-nu0/(4+nu0))
        nuMax=nu0
        a=(alphaMin+alphaMax)/2
        gamma=log((nuMin+nuMax)/2)
        #a,nu=alpha0,nu0
        sol=optimize.root(RVGALogReg.fun2D, [a,gamma], tol=1e-6, args=(alpha0,nu0,y,),jac=RVGALogReg.jac,method='hybr')

        return sol.x[0],exp(sol.x[1]) ,sol.nfev
    
    def update(self,xt,yt):
        
        # init parameters 
        nu0=xt.T.dot(self._Cov.dot(xt))
        alpha0=xt.T.dot(self._theta)
            
        a,nu,nbInnerLoop=RVGALogReg.optim2D(np.asscalar(alpha0), np.asscalar(nu0), np.asscalar(yt))
            
        #updates
        k=RVGALogReg.beta/math.sqrt(nu+RVGALogReg.beta**2)
        self._theta=self._theta+self._Cov.dot(xt)*(yt-sigmoid(k*a))
        s=1/(nu0+1/(k*sigp(k*a)))
        self._Cov=self._Cov-s*np.outer(self._Cov.dot(xt),self._Cov.dot(xt))
    
class RVGALogRegExplicite(OnlineBayesianLogisticRegression):
    
    def __init__(self, theta0, Cov0, passNumber=1):
        super().__init__(theta0, Cov0, passNumber)
    
    def update(self,xt,yt):
        beta=math.sqrt(8/math.pi)
        
        # intermediate variables
        nu=xt.T.dot(self._Cov.dot(xt))
        Pu=self._Cov.dot(xt)
            
        # compute sigma(a)
        k=beta/math.sqrt(nu+beta**2)
        a=xt.T.dot(self._theta)
        
        m=k*sigp(k*a)
        m=max(m,1e-100)
        
        # update state
        self._Cov=self._Cov-np.outer(Pu,Pu)/(1/m+nu)
        self._theta=self._theta+self._Cov.dot(xt)*(yt-sigmoid(k*a))
        
        
