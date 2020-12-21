###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Mathematical functions                                                          #
###################################################################################

import numpy.linalg as LA 
import numpy as np
import math

############## Graphix tools ########################## 
# the logistic function
def sigmoid(x):
    x=np.clip(x,-100,100)
    return 1/(1+np.exp(-x))

# the first derivative of the logistic function
def sigp(x):
    return sigmoid(x)*(1-sigmoid(x))

# the second derivative of the logistic function
def sigpp(x):
    return sigmoid(x)*(1-sigmoid(x))*(1-2*sigmoid(x))

# a quadratic form
def quadratic2D(x, y, H,v,c):
    q11=H[0,0]
    q22=H[1,1]
    q12=H[0,1]
    v1=v[0]
    v2=v[1]
    return 0.5*q11*x**2 + 0.5*q22*y**2 + q12*x*y-v1*x-v2*y+c

# the likelihood density for logistic regression, Y supposed in {0,1} 
def logisticPdf(theta,X,Y,Z):
    N,d=X.shape
    theta=theta.reshape(d,1)
    # we use the relation logloss=sigmoid(y.theta.u)
    # and log sigmoid(y.theta.u) = - log(1+exp(-y.theta.u)=- logexp(0,-y.theta.u)
    Yb=2*Y.reshape(N,1)-1
    log_pdf_likelihood=np.sum(-np.logaddexp(0, -Yb*X.dot(theta)),axis=0)
    return log_pdf_likelihood-math.log(Z)

# the bayesian density for logistic regression, Y supposed in {0,1}   
def bayesianlogisticPdf(theta,theta0, Cov0,X,Y,Z):
    N,d=X.shape
    # compute log prior:
    theta=theta.reshape(d,1)
    theta0=theta0.reshape(d,1)
    log_pdf_prior=-0.5*(theta-theta0).T.dot(LA.inv(Cov0)).dot(theta-theta0)\
        -0.5*d*math.log(2*math.pi)-0.5*math.log(LA.det(Cov0))
    return logisticPdf(theta,X,Y,Z)+log_pdf_prior

def negbayesianlogisticPdf(theta,theta0, Cov0,X,Y,Z):
    return -bayesianlogisticPdf(theta,theta0, Cov0,X,Y,Z)

def neglogisticPdf(theta,X,Y,Z):
    return -logisticPdf(theta,X,Y,Z)

class graphix: 
    # plot a 2D ellipsoid
    def plot_ellipsoid2d(ax,origin,Cov,col='r',zorder=1,label='',linestyle='dashed',linewidth=1):
        L=LA.cholesky(Cov)
        theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
        x = np.cos(theta)
        y = np.sin(theta)
        x,y=origin.reshape(2,1) + L.dot([x, y])
        ax.plot(x, y,linestyle=linestyle,color=col,zorder=zorder,label=label,linewidth=linewidth)
    
    # project a ND ellipsoid (mean-covariance) in plane (i,j)
    def projEllipsoid(theta,P,i,j):
        thetaproj=np.array([theta[i],theta[j]])
        Pproj=np.zeros((2,2))
        Pproj[0,0]=P[i,i]
        Pproj[0,1]=P[i,j]
        Pproj[1,0]=P[j,i]
        Pproj[1,1]=P[j,j]
        return thetaproj,Pproj
    
    
    