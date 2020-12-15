import numpy.linalg as LA 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from KalmanMachine.KDataGenerator import LinearRegObservations,LogisticRegObservations
from KalmanMachine.KEvaluation import LinearRegEvaluation, LogisticRegEvaluation

     ###################### TEST LINEAR REG ########################
def testLinReg():
    N=10
    d=2
    c=0
    sigma=-1
    seed=3
    RegObs=LinearRegObservations(sigma,N,d,c,seed,scale=1,rotate=True)
    Y,U=RegObs.datas
    
    print('the optimal is:\n',RegObs.optim)
    print('the loss with theta=theta_opt is',LinearRegEvaluation.loss(Y,U,RegObs.optim))
    print(RegObs.covInputs)
    
    fig,(ax1,ax2)=plt.subplots(1,2,num=2,figsize=(11,5))
    print('Approximated Hessian matrix=\n',np.diag(LA.linalg.eig(RegObs.covInputs)[0]))
    print('Empirical Hessian matrix=\n',np.diag(LA.linalg.eig(U.T.dot(U)/N)[0]))
    ax1.hist2d(U[:,0],U[:,1],50)
    ax1.set_title('Empirical distributions of inputs $u_i$ (density)')
    LinearRegEvaluation.plotLossShape(ax2,RegObs.optim,U.T.dot(U)/N,nbcontours=20)
    ax2.set_title('Hessian')
    plt.show()
    
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.view_init(90, -90)
    ax.set_zticks([])
    ax.scatter(U[:,0],U[:,1],Y,c=Y.reshape(N,),cmap='jet',marker='.',s=20)
    plt.title('outputs value in function of inputs $u_i$')
    plt.show()
 
import math
def testLogReg():
    ###################### TEST LOGISTIC REG ########################
    N=10
    d=100
    c=0
    seed=3
    meansShift=2
    RegObs2=LogisticRegObservations(meansShift,N,d,c,seed,scale=1,rotate=False,normalize=False)
    Y,U=RegObs2.datas
    
    #print('the optimal is:\n',RegObs2.optim)
    loss=LogisticRegEvaluation.loss(Y,U,RegObs2.optim)
    score=LogisticRegEvaluation.score(Y,U,RegObs2.optim)
    print('The loss for the optimal is ',loss)
    print('The percent of well classified datas is',score)
    
if __name__=="__main__":
    testLogReg()
   
    
    