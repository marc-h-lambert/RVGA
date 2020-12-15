import numpy.linalg as LA 
import numpy as np
import matplotlib.pyplot as plt

############## Graphix tools ########################## 
def sigmoid(x):
    x=np.clip(x,-100,100)
    return 1/(1+np.exp(-x))

def sigp(x):
    return sigmoid(x)*(1-sigmoid(x))

def sigpp(x):
    return sigmoid(x)*(1-sigmoid(x))*(1-2*sigmoid(x))

def cross(A):
    if (len(A.shape)==1):
        return np.outer(A,A)
    else:
        return A.dot(A.T)

def cross2(A,B):
    if (len(A.shape)==1):
        return np.outer(A,B)
    else:
        return A.dot(B)

def quadratic2D(x, y, H,v,c):
    q11=H[0,0]
    q22=H[1,1]
    q12=H[0,1]
    v1=v[0]
    v2=v[1]
    return 0.5*q11*x**2 + 0.5*q22*y**2 + q12*x*y-v1*x-v2*y+c

class graphix: 
    def plot_ellipsoid2d(origin,Cov,col='r',zorder=1,label='',linestyle='dashed',linewidth=1):
        L=LA.cholesky(Cov)
        theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
        x = np.cos(theta)
        y = np.sin(theta)
        x,y=origin.reshape(2,1) + L.dot([x, y])
        plt.plot(x, y,linestyle=linestyle,color=col,zorder=zorder,label=label,linewidth=linewidth) 
    
    
    