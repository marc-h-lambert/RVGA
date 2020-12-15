import numpy as np
from KalmanMachineLib.KDataGenerator import LogisticRegObservations
from KalmanMachineLib.Kalman4LogisticReg import EKFLogisticRegression
from KalmanMachineLib.KEvaluation import LogisticRegEvaluation

if __name__=="__main__":
    N=10
    d=100
    c=0
    seed=3
    meansShift=2
    RegObs=LogisticRegObservations(meansShift,N,d,c,seed,scale=1,rotate=False,normalize=False)
    y,X=RegObs.datas
    
    theta0=np.zeros((d,1))
    Cov0=np.identity(d)*10
    passNumber=1
    ekf = EKFLogisticRegression(theta0,Cov0,passNumber).fit(X, y.reshape(N,))
    loss=LogisticRegEvaluation.loss(y,X,ekf.theta)
    score=LogisticRegEvaluation.score(y,X,ekf.theta)
    print('ekf loss={} and score={} %'.format(loss,score*100))
