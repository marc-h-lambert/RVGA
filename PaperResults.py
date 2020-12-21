###################################################################################
# THE KALMAN MACHINE LIBRARY                                                      #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "The recursive variational Gaussian approximation (R-VGA)"                      #
# Authors: Marc Lambert, Silvere Bonnabel and Francis Bach                        #
###################################################################################

import numpy as np
from KalmanMachine.KDataGenerator import LogisticRegObservations
from KalmanMachine.BayesianLogisticReg import LaplaceLogisticRegression
from KalmanMachine.Kalman4LogisticReg import EKFLogReg,QKFLogReg,RVGALogReg,RVGALogRegExplicite
from TestKalmanMachine import plotCov, plotKL
from KalmanMachine.KEvalPosterior import PosteriorLogReg
import matplotlib.pyplot as plt
from plot4latex import set_size
import math

def XP_2D(sigma0,mu0,N,s,c,seed,name,num,nbLevels=0):
    d=2
    ################### GENERATE DATA ####################### 
    RegObs=LogisticRegObservations(s,N,d,c,seed,scale=1,rotate=True,normalize=True)
    y,X=RegObs.datas
    
    ################### RUN KALMAN ####################### 
    theta0=mu0*np.ones([d,1])/math.sqrt(d)
    Cov0=np.identity(d)*sigma0**2
    
    # True posterior
    posterior=PosteriorLogReg(theta0,Cov0).fit(X,y.reshape(N,))
    
    # Kalman algorithms
    ekf = EKFLogReg(theta0,Cov0).fit(X, y.reshape(N,))
    qkf = QKFLogReg(theta0,Cov0).fit(X, y.reshape(N,))
    rvga = RVGALogReg(theta0,Cov0).fit(X, y.reshape(N,))
    rvgae= RVGALogRegExplicite(theta0,Cov0).fit(X, y.reshape(N,))
    lap = LaplaceLogisticRegression(theta0,Cov0).fit(X, y.reshape(N,))
    
    # Plot figure 1: Datas + KL
    fig = plt.figure(num,figsize=set_size(twoColumns=True))
    ax=fig.add_subplot(121)
    ax.set_title('Distributions of inputs')
    RegObs.plot(ax,plotcov=False,plotNormal=False)
    #ax.set_xticks(fontsize=8)
    #ax.set_yticks(fontsize=8)
    ax.set_xlabel(r'$\sigma_0=${0}, $||\mu_0||={1:.1f}$ , N={2}, d={3}, s={4:.1f}, c={5}'.format(sigma0,mu0,N,d,s,c))
    
    ax=fig.add_subplot(122)
    ax.set_title('KL divergence',loc='left')
    plotKL(ax,rvga,rvgae,ekf,qkf,lap,posterior,seed)
    ax.set_ylabel('KL error')
    #ax.set_xticks(fontsize=8)
    #ax.set_yticks(fontsize=8)
    fig.legend(loc="upper right", ncol=2)
    #plt.savefig('./outputs/KL_mu0{}_sigma0{}_N{}_d{}_s{}_c{}'.format(int(mu0),int(sigma0),int(N),int(d),int(s),int(c)))
    plt.savefig('./outputs/KL_2d_'+name)

    # Plot figure 2: Covariances
    num=num+1
    plotCov(rvga,rvgae,ekf,qkf,lap,posterior,num,nbLevels)
    #plt.savefig('./outputs/Covariances_mu0{}_sigma0{}_N{}_d{}_s{}_c{}'.format(int(mu0),int(sigma0),int(N),int(d),int(s),int(c)))
    plt.savefig('./outputs/Cov_2d_'+name)
    
def XP_HighDim(axs,sigma0,mu0,N,d,s,c,seed,label=True):
    ################### GENERATE DATA ####################### 
    RegObs=LogisticRegObservations(s,N,d,c,seed,scale=1,rotate=True,normalize=True)
    y,X=RegObs.datas
    
    ################### RUN KALMAN ####################### 
    theta0=mu0*np.ones([d,1])/math.sqrt(d)
    Cov0=np.identity(d)*sigma0**2
    posterior=PosteriorLogReg(theta0,Cov0).fit(X,y.reshape(N,))
    # Kalman algorithms
    ekf = EKFLogReg(theta0,Cov0).fit(X, y.reshape(N,))
    qkf = QKFLogReg(theta0,Cov0).fit(X, y.reshape(N,))
    rvga = RVGALogReg(theta0,Cov0).fit(X, y.reshape(N,))
    rvgae= RVGALogRegExplicite(theta0,Cov0).fit(X, y.reshape(N,))
    lap = LaplaceLogisticRegression(theta0,Cov0).fit(X, y.reshape(N,))
    
    # Plot figure 1: Datas + KL
    if axs.shape[0]>1:
        RegObs.plotOutputs(axs[0])
        #axs[0].set_xticks([0,1])
        if label:
            axs[0].set_ylabel('output means')

        plotKL(axs[1],rvga,rvgae,ekf,qkf,lap,posterior,seed,label)
    else:
        plotKL(axs[0],rvga,rvgae,ekf,qkf,lap,posterior,seed,label)
    #plt.xticks(fontsize=6)
    #plt.yticks(fontsize=6)

if __name__=="__main__":
    #Test=['LD1','LD2','LD3','HD1','HD2','HD3','HD4']
    Test=['LD1']
    
    # the images results are generated in the local directory ./outputs
    # XP in 2D
    num=1
    N=100
    if 'LD1' in Test:
        XP_2D(1,0.1,N,2,2,10,'mu01',num)
        num=num+2
        XP_2D(10,1,N,2,2,10,'mu1',num)
        num=num+2
        XP_2D(100,10,N,2,2,10,'mu10',num)
        num=num+2
    if 'LD2' in Test:
        XP_2D(10,0,N,5,2,10,'hard',num)
        num=num+2
    if 'LD3' in Test:
        XP_2D(10,0,N,2,1,1,'s2',num,nbLevels=4)
        num=num+2
        XP_2D(10,0,N,5,1,1,'s5',num,nbLevels=4)
        num=num+2
        XP_2D(10,0,N,10,1,1,'s10',num,nbLevels=4)
        num=num+2
        
    # XP in High Dim 
    # TEST HD1 sensitivity to dimension with Sharp prior sigma0=1
    if 'HD1' in Test:
        sigma0=1
        mu0=0
        d_list=[30,70,100]
        N=500
        s=2
        c=0
        seed=10
        
        fig = plt.figure(num,figsize=set_size(ratio=0.5))
        num=num+1
        ax1=fig.add_subplot(131)
        XP_HighDim(np.array([ax1]),sigma0,mu0,N,d_list[0],s,c,seed,label=True)
        ax1.set_title('d={}'.format(d_list[0]))
        ax2=fig.add_subplot(132)
        XP_HighDim(np.array([ax2]),sigma0,mu0,N,d_list[1],s,c,seed,label=False)
        ax2.set_title('d={}'.format(d_list[1]))
        ax3=fig.add_subplot(133)
        XP_HighDim(np.array([ax3]),sigma0,mu0,N,d_list[2],s,c,seed,label=False)
        ax3.set_title('d={}'.format(d_list[2]))
        
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        fig.legend(loc="lower right", ncol=3)
        fig.text(0.5, 0.04, 'number of iterations \n ({} pass x {} samples)'.format(1,N), ha='center')
        fig.suptitle('Sharp prior:'+r'$\sigma_0={}$'.format(sigma0))    
        plt.savefig('./outputs/KL_HighDim_sharpPrior')
    
    if 'HD2' in Test:
        # sensitivity to dimension with Flat prior sigma0=1
        sigma0=30
        mu0=0
        d_list=[30,70,100]
        N=500
        s=2
        c=0
        seed=10
        
        fig = plt.figure(num,figsize=set_size(ratio=0.5))
        num=num+1
        ax1=fig.add_subplot(131)
        XP_HighDim(np.array([ax1]),sigma0,mu0,N,d_list[0],s,c,seed,label=True)
        ax1.set_title('d={}'.format(d_list[0]))
        ax2=fig.add_subplot(132)
        XP_HighDim(np.array([ax2]),sigma0,mu0,N,d_list[1],s,c,seed,label=False)
        ax2.set_title('d={}'.format(d_list[1]))
        ax3=fig.add_subplot(133)
        XP_HighDim(np.array([ax3]),sigma0,mu0,N,d_list[2],s,c,seed,label=False)
        ax3.set_title('d={}'.format(d_list[2]))
        
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        fig.legend(loc="lower right", ncol=3)
        fig.text(0.5, 0.04, 'number of iterations \n ({} pass x {} samples)'.format(1,N), ha='center')
        fig.suptitle('Flat prior:'+r'$\sigma_0={}$'.format(sigma0))    
        plt.savefig('./outputs/KL_HighDim_flatPrior')
    
    if 'HD3' in Test:
        # sensitivity to separation s with isotropic covariance (c=0)
        s_list=[1,3,5]
        c=0
        sigma0=1
        mu0=0
        d=100
        N=200
        seed=10
            
        fig = plt.figure(num,figsize=set_size(ratio=1))
        num=num+1
        ax1=fig.add_subplot(231)
        ax1b=fig.add_subplot(234)
        XP_HighDim(np.array([ax1,ax1b]),sigma0,mu0,N,d,s_list[0],c,seed,label=True)
        ax1.set_title('s={}'.format(s_list[0]))
        ax2=fig.add_subplot(232)
        ax2b=fig.add_subplot(235)
        XP_HighDim(np.array([ax2,ax2b]),sigma0,mu0,N,d,s_list[1],c,seed,label=False)
        ax2.set_title('s={}'.format(s_list[1]))
        ax3=fig.add_subplot(233)
        ax3b=fig.add_subplot(236)
        XP_HighDim(np.array([ax3,ax3b]),sigma0,mu0,N,d,s_list[2],c,seed,label=False)
        ax3.set_title('s={}'.format(s_list[2]))
        
        fig.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
        fig.legend(loc="lower right", ncol=3)
        fig.text(0.5, 0.04, 'number of iterations \n ({} pass x {} samples)'.format(1,N), ha='center')
        plt.savefig('./outputs/KL_HighDim_shift')
    
    if 'HD4' in Test:
        # sensitivity to separation s with ill-conditioned covariance (c=2)
        s=2
        c_list=[0,1,2]
        sigma0=10
        mu0=0
        d=100
        N=200
        seed=10
            
        fig = plt.figure(num,figsize=set_size(ratio=1))
        num=num+1
        ax1=fig.add_subplot(231)
        ax1b=fig.add_subplot(234)
        XP_HighDim(np.array([ax1,ax1b]),sigma0,mu0,N,d,s,c_list[0],seed,label=True)
        ax1.set_title('c={}'.format(c_list[0]))
        ax2=fig.add_subplot(232)
        ax2b=fig.add_subplot(235)
        XP_HighDim(np.array([ax2,ax2b]),sigma0,mu0,N,d,s,c_list[1],seed,label=False)
        ax2.set_title('c={}'.format(c_list[1]))
        ax3=fig.add_subplot(233)
        ax3b=fig.add_subplot(236)
        XP_HighDim(np.array([ax3,ax3b]),sigma0,mu0,N,d,s,c_list[2],seed,label=False)
        ax3.set_title('c={}'.format(c_list[2]))
        
        fig.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=0.5, hspace=None)
        fig.legend(loc="lower right", ncol=3)
        fig.text(0.5, 0.04, 'number of iterations \n ({} pass x {} samples)'.format(1,N), ha='center')
        plt.savefig('./outputs/KL_HighDim_cond')

    
    