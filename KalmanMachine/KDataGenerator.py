################################################################################
# Generate N synthetic noisy observations in dimension d for :                 #
# - the linear regression problem (with Gaussian inputs and an ouput noise)    #
# - the logistic regression problem (with two Gaussian inputs for Y=0 and Y=1) #
# The Gaussian covariance on inputs are parametrized by c, scale and rotate    #
################################################################################
import numpy.linalg as LA 
import numpy as np
from scipy.stats import special_ortho_group
    
class observations(object):
    def __init__(self, N,d,c,scale,rotate,normalize,seed):
        self._N = N # the number of observations
        self._d = d # the dimension of inputs
        self._c = c # a parameter driving the condition number of covariance of inputs 
        self._scale = scale # the inputs scale (1 by default)
        self._rotate = rotate #true if the covariance of inputs are rotated
        self._normalize = normalize #true if the covariance of inputs are rotated
        self._seed = seed # the random seed (to reproduce results)
    
    @property
    def N(self):
        return self._N
    
    @property
    def d(self):
        return self._d
    
    def covariance(self):
        Cov_u=np.diag(1/np.arange(1,self._d+1)**self._c)*self._scale**2
        if self._d>1 and self._rotate:
            np.random.seed(self._seed)
            Q = special_ortho_group.rvs(dim=self._d)
            Cov_u=np.transpose(Q).dot(Cov_u).dot(Q)
        return Cov_u
    
    @property
    def datas(self):
        pass
    
    @property
    def optim(self):
        pass
    
    @property
    def covInputs(self):
        pass
    
        
class LinearRegObservations(observations):
    def __init__(self, sigma,N,d,c,seed,scale=1,rotate=True,normalize=False):
        super().__init__(N,d,c,scale,rotate,normalize,seed)
        self.__sigma = sigma # the outputs noise
        self.__CovInputs=self.covariance()
        self.__meanInputs=np.zeros((d,))
    
        # generate the inputs
        np.random.seed(seed)
        X=np.random.multivariate_normal(self.__meanInputs,self.__CovInputs,(N))
        if normalize:
            X=X/LA.norm(np.std(X,axis=0))
        self.__inputs=X
        
        # generate a random optimal of norm 1
        np.random.seed(seed)
        theta=np.random.uniform(-1,1,d)
        self.__thetaOpt=theta/LA.norm(theta) 
    
        # generate the outputs
        Y=X.dot(self.__thetaOpt)
        if self.__sigma>0: #if sigma <0 --> no noise model 
            np.random.seed(seed)
            B=np.random.normal(0,self.__sigma,N).reshape(N,)
            Y=Y+B
        
        self.__outputs=Y
    
    @property
    def datas(self):
        return self.__outputs,self.__inputs
    
    @property
    def optim(self):
        return self.__thetaOpt
    
    @property
    def covInputs(self):
        return self.__CovInputs

class LogisticRegObservations(observations):
    def __init__(self, meansShift,N,d,c,seed,scale=1,rotate=True,normalize=False):
        super().__init__(N,d,c,scale,rotate,normalize,seed)
        self.__meansShift = meansShift # the distance between the means
        self.__CovInputs=self.covariance()
        
        # we normalize the means
        np.random.seed(seed)
        mean_dir=np.random.rand(d,)
        theta=mean_dir/LA.norm(mean_dir)
        self.__meanInputs0 = theta*self.__meansShift/2
        self.__meanInputs1 = -theta*self.__meansShift/2 
        
        invCov=LA.inv(self.__CovInputs)
        #gamma=0.5*self.__meanInputs0.T.dot(invCov).dot(self.__meanInputs0)-0.5*self.__meanInputs1.T.dot(invCov).dot(self.__meanInputs1)
        #print('gamma=(must be 0)',gamma)
        self.__thetaOpt=invCov.dot(self.__meanInputs1-self.__meanInputs0)
    
        # generate the inputs
        np.random.seed(seed)
        X0=np.random.multivariate_normal(self.__meanInputs0,self.__CovInputs,int(N/2))
        if normalize:
            X0=X0/LA.norm(np.std(X0,axis=0))
        np.random.seed(seed+1)
        X1=np.random.multivariate_normal(self.__meanInputs1,self.__CovInputs,int(N/2)) 
        if normalize:
            X1=X1/LA.norm(np.std(X1,axis=0))
        X=np.concatenate((X0,X1))
            
        # generate the outputs
        Y0=np.ones((int(N/2),1))*0
        Y1=np.ones((int(N/2),1))*1
        Y=np.concatenate((Y0,Y1))
        DataSet=list(zip(Y,X))
        np.random.seed(seed+2)
        np.random.shuffle(DataSet)
        Y,U= zip(*DataSet)
        self.__outputs,self.__inputs = np.array(Y),np.array(U)
        
    @property
    def datas(self):
        return self.__outputs,self.__inputs
    
    @property
    def optim(self):
        return self.__thetaOpt
    
    @property
    def covInputs(self):
        return self.__CovInputs
        
    