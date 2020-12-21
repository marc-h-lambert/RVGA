###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Validation tests for THE KALMAN MACHINE LIBRARY                                 #                     #
###################################################################################

import numpy as np
from KalmanMachine.KDataGenerator import LogisticRegObservations
from KalmanMachine.BayesianLogisticReg import LaplaceLogisticRegression
from KalmanMachine.Kalman4LogisticReg import EKFLogReg,QKFLogReg,RVGALogReg,RVGALogRegExplicite
from KalmanMachine.KEvalPosterior import PosteriorLogReg, KL
import matplotlib.pyplot as plt
import math
from plot4latex import set_size

########### COMPUTE FIGURE 1 TO VISUALIZE COVS FOR 3 ALGOS ##########
def plotDatas(RegObs,plotcov=False,plotNormal=False,num=1):
    fig = plt.figure(num,figsize=set_size(twoColumns=True))
    ax=fig.add_subplot(121)
    RegObs.plot(ax,plotcov=plotcov,plotNormal=plotNormal)
    #ax.set_xlim(-5,5)
    #ax.set_ylim(-5,5)
    ax.set_aspect('equal')
    ax.set_title('inputs X with separator (blue) and covariances (red)')
    ax=fig.add_subplot(122)
    RegObs.plotOutputs(ax)
    ax.set_title(r'ouputs means $sigma(x_i^Ttheta)$')
    
def plotCov(rvga,rvgae,ekf,qkf,lap,posterior,num=1,nbLevels=0):
    print('Plot covariances ... ')
    fig = plt.figure(num,figsize=set_size(twoColumns=True))
    ax=fig.add_subplot(141)
    rvga.plotEllipsoid(ax,nbLevels=nbLevels,labelize=True)
    lap.plotEllipsoid(ax,labelize=True)
    if not posterior is None:
        posterior.plot(ax,showMleMap=False)
    ax.set_title('RVGA (implicit)')
    #ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    #ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    ax=fig.add_subplot(142)
    rvgae.plotEllipsoid(ax,nbLevels=nbLevels,labelize=False)
    lap.plotEllipsoid(ax,labelize=False)
    if not posterior is None:
        posterior.plot(ax,showMleMap=False)
    ax.set_title('RVGA (explicit)')
    #ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    #ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    ax=fig.add_subplot(143)
    ekf.plotEllipsoid(ax,nbLevels=nbLevels,labelize=False)
    lap.plotEllipsoid(ax,labelize=False)
    if not posterior is None:
        posterior.plot(ax,showMleMap=False)
    ax.set_title('EKF')
    #ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    #ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    ax=fig.add_subplot(144)
    qkf.plotEllipsoid(ax,nbLevels=nbLevels,labelize=False)
    lap.plotEllipsoid(ax,labelize=False)
    if not posterior is None:
        posterior.plot(ax,showMleMap=False)
    ax.set_title('QKF')
    #ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    #ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    fig.legend(loc="lower center", ncol=4)
    
########### COMPUTE FIGURE 2 TO PLOT KL FOR 3 ALGOS ##########
def plotKL(ax,rvga,rvgae,ekf,qkf,lap,truePosterior,seed,labelize=True):
    N,d=rvga.history_theta.shape
    nbSamplesKL=300
    np.random.seed(seed)
    normalSamples=np.random.multivariate_normal(np.zeros(d,),np.identity(d),size=(nbSamplesKL,))
    kl_lap=KL.divergence(lap.theta,lap.Cov,normalSamples,truePosterior)
    
    print('Compute KL for RVGA implicit ... ')
    kl_histo_RVGA=KL.history(rvga,truePosterior,nbSamplesKL,seed)
    print('Compute KL for RVGA explicit ... ')
    kl_histo_RVGAE=KL.history(rvgae,truePosterior,nbSamplesKL,seed)
    print('Compute KL for EKF ... ')
    kl_histo_EKF=KL.history(ekf,truePosterior,nbSamplesKL,seed)
    print('Compute KL for QKF ... ')
    kl_histo_QKF=KL.history(qkf,truePosterior,nbSamplesKL,seed)
    
    Iters=np.arange(0,rvga.history_theta.shape[0])
    if labelize:
        ax.semilogy(Iters,kl_histo_RVGA,label='RVGA',color='b')
        ax.semilogy(Iters,kl_histo_EKF,label='EKF',color='orchid')
        ax.semilogy(Iters,kl_histo_QKF,label='QKF',color='g')
        ax.semilogy(Iters,kl_histo_RVGAE,label='RVGA-exp',linestyle='dashed',color='k')
        ax.semilogy(Iters,np.ones(N,)*kl_lap,label='Laplace',color='r')
        ax.set_ylabel('KL error')
    else:
        ax.semilogy(Iters,kl_histo_RVGA,color='b')
        ax.semilogy(Iters,kl_histo_EKF,color='orchid')
        ax.semilogy(Iters,kl_histo_QKF,color='g')
        ax.semilogy(Iters,kl_histo_RVGAE,linestyle='dashed',color='k')
        ax.semilogy(Iters,np.ones(N,)*kl_lap,color='r')

def plotPredictionMap(RegObs,rvga,rvgae,ekf,qkf,lap,num=3):
    print('Compute prediction maps ... ')
    fig = plt.figure(num,figsize=set_size(twoColumns=True))
    ax=fig.add_subplot(151)
    ax.set_title('RVGA')
    rvga.plotPredictionMap(ax)
    RegObs.plot(ax)
    ax=fig.add_subplot(152)
    ax.set_title('RVGA-exp')
    rvgae.plotPredictionMap(ax)
    RegObs.plot(ax)
    ax=fig.add_subplot(153)
    ax.set_title('EKF')
    ekf.plotPredictionMap(ax)
    RegObs.plot(ax)
    ax=fig.add_subplot(154)
    ax.set_title('QKF')
    qkf.plotPredictionMap(ax)
    RegObs.plot(ax)
    ax=fig.add_subplot(155)
    ax.set_title('Laplace')
    lap.plotPredictionMap(ax)
    RegObs.plot(ax)
    
if __name__=="__main__":
    ################### GENERATE DATA ####################### 
    N=100
    d=2
    c=2
    seed=10
    
    meansShift=5
    RegObs=LogisticRegObservations(meansShift,N,d,c,seed,scale=1,rotate=True,normalize=True)
    y,X=RegObs.datas
    
    ################### RUN KALMAN ####################### 
    sigma0=10
    alpha0=0
    theta0=alpha0*np.ones([d,1])/math.sqrt(d)
    Cov0=np.identity(d)*sigma0**2
    
    # True posterior
    posterior=PosteriorLogReg(theta0,Cov0).fit(X,y.reshape(N,))
    
    ekf = EKFLogReg(theta0,Cov0).fit(X, y.reshape(N,))
    qkf = QKFLogReg(theta0,Cov0).fit(X, y.reshape(N,))
    rvga = RVGALogReg(theta0,Cov0).fit(X, y.reshape(N,))
    rvgae= RVGALogRegExplicite(theta0,Cov0).fit(X, y.reshape(N,))
    lap = LaplaceLogisticRegression(theta0,Cov0).fit(X, y.reshape(N,))
    
    plotDatas(RegObs,plotcov=True,plotNormal=True,num=1)
    
    if d==2:
        plotCov(rvga,rvgae,ekf,qkf,lap,posterior,num=2,nbLevels=4)
    else:
        plotCov(rvga,rvgae,ekf,qkf,lap,None,num=2,nbLevels=4)
    plt.suptitle('covariances over iterations')
    plt.savefig('./outputs/Cov_outputs')
    
    fig = plt.figure(3,figsize=set_size(twoColumns=False))
    ax=fig.add_subplot(111)
    plotKL(ax,rvga,rvgae,ekf,qkf,lap,posterior,seed)
    ax.set_title('Evolution of the KL divergence with iterations')
    plt.legend(loc='upper right')
    fig.text(0.5, 0.04, 'number of iterations \n ({} pass x {} samples)'.format(1,N), ha='center')
    
    plotPredictionMap(RegObs,rvga,rvgae,ekf,qkf,lap,num=4)
    plt.savefig('./outputs/Map_outputs')
    plt.suptitle('outputs probabilities')
    plt.show()
    
    