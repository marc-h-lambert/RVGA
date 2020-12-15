################################################################################
# Define metrics to assess linear and logistic regression    :                 #
# - the loss with different plots                                              #
# - the rmse                                                                   #
################################################################################

import numpy as np 
from scipy.stats import special_ortho_group
import numpy.linalg as LA
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .KUtils import quadratic2D

############## METRICS FOR LINEAR REGRESSION ##########################        
class LinearRegEvaluation:
    
    @staticmethod 
    def loss(Y,U,theta):
        N,d=U.shape
        H=0.5*U.T.dot(U)/N
        Y=Y.reshape(N,1)
        v=np.mean(Y*U,axis=0)
        c=0.5*np.mean(Y**2)
        return theta.T.dot(H).dot(theta)-v.T.dot(theta)+c

    @staticmethod 
    def lossOnPath(U,Y,history_theta,thetaOpt):
        Niter=history_theta.shape[0]
        loss=np.zeros([Niter,1])
        N,d=U.shape
        H=0.5*U.T.dot(U)/N
        for i in np.arange(0,Niter):
            error=history_theta[i]-thetaOpt
            lossU=error.dot(H).dot(error.T)
            loss[i]=lossU
        return loss
    
    @staticmethod 
    def plotLoss(ax,U,Y,x_opt,numberPass,list_histo_x,list_labels,\
                plotOptimal=True,title='Least mean square loss',axesLegend=True):
        N,d=U.shape
        NIter=N*numberPass
        t=np.arange(0,NIter)

        ax.axvline(d,color='r',linestyle='-.') 
        ax.axvline(N,color='r',linestyle='-.')

        list_col=['b','g','y','m','c','grey','ivory','olive','brown']
        idx=0
        for history_x in list_histo_x:
            loss=lossOnPath(U,Y,history_x[t],x_opt)
            ax.semilogy(t,loss,'-.',color=list_col[idx],label=list_labels[idx])
            idx=idx+1

        if plotOptimal:
            loss=lossOnPath(U,Y,list_histo_x[0][0:1],x_opt)
            ax.semilogy(t[1:],loss[0]/t[1:],label='$1/t$')

        ax.set_title(title)
        if axesLegend:
            ax.set_xlabel('number of iterations ({} pass x {} samples)'.format(numberPass,N))
            ax.set_ylabel('Loss(x)-Loss($x^*$)')
            ax.legend(loc='upper right')

    @staticmethod 
    def plotLossShape(ax,optimal,Hessian,color='k',nbcontours=20,spread=0.5):
        xmin, xmax, xstep = optimal[0]-spread,  optimal[0]+spread, 0.01
        ymin, ymax, ystep = optimal[1]-spread,  optimal[1]+spread, 0.01
        x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
        z = quadratic2D(x, y, Hessian,Hessian.dot(optimal), 0)

        ax.contour(x, y, z, nbcontours, cmap=plt.cm.jet)
        ax.plot(*optimal, 'r*', markersize=18)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
    
    @staticmethod 
    def plotLossPath(ax,path,optimal,Hessian,color='k',nbcontours=20):
    
        xmin, xmax, xstep = np.min(path[:,0])-0.5,  np.max(path[:,0])+0.5, (np.max(path[:,0])-np.min(path[:,0]))/100
        ymin, ymax, ystep = np.min(path[:,1])-0.5,  np.max(path[:,1])+0.5, (np.max(path[:,1])-np.min(path[:,1]))/100
        x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
        z = quadratic2D(x, y, Hessian,Hessian.dot(optimal), 0)

        ax.contour(x, y, z, nbcontours, cmap=plt.cm.jet)
        ax.quiver(path[:-1,0], path[:-1,1], path[1:,0]-path[:-1,0], path[1:,1]-path[:-1,1], scale_units='xy', angles='xy', scale=1, color=color)
        ax.plot(*optimal, 'r*', markersize=18)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))

############## METRICS FOR LOGISTIC REGRESSION ##########################    
class LogisticRegEvaluation:
    
    @staticmethod 
    def loss(Y,U,x_opt):
        N,d=U.shape
        Y=Y.reshape(N,)
        x_opt=x_opt.reshape(d,)
        V=(2*Y-1)*U.dot(x_opt)
        V=np.clip(V,-50,50)
        return np.sum(np.log(1+np.exp(-V)))
    
    @staticmethod 
    def score(Y,U,x_opt):
        N,d=U.shape
        Y=Y.reshape(N,)
        x_opt=x_opt.reshape(d,)
        V=(2*Y-1)*U.dot(x_opt)
        return np.sum(V>0)/N
    
    @staticmethod 
    def lossOnPath(U,Y,history_x,x_opt):
        Niter=history_x.shape[0]
        loss=np.zeros([Niter,1])
        Loss_opt=loss(Y,U,x_opt)
        for i in np.arange(0,Niter):
            Loss_x=loss(Y,U,history_x[i])
            loss[i]=Loss_x-Loss_opt
        return loss
    
    @staticmethod 
    def scoreOnPath(U,Y,history_x,x_opt):
        Niter=history_x.shape[0]
        loss=np.zeros([Niter,1])
        N,d=U.shape
        Y=Y.reshape(N,)
        x_opt=x_opt.reshape(d,)
        Loss_opt=score(Y,U,x_opt)
        for i in np.arange(0,Niter):
            Loss_x=score(Y,U,history_x[i])
            loss[i]=Loss_x
        return loss
    
    @staticmethod 
    def showClassifier(ax,U,Y,theta0,theta_opt,list_theta,list_labels):
        ax.scatter(U[np.where((Y==0)),0],U[np.where((Y==0)),1])
        ax.scatter(U[np.where((Y==1)),0],U[np.where((Y==1)),1])
        ax.set_ylim((-2,2))
        
        theta=np.arange(-2,2,0.001)
        y0=-theta0[0]/theta0[1]*theta
        y_opt=-theta_opt[0]/theta_opt[1]*theta
        ax.plot(theta,y0,'-.r',label='initial')
        ax.plot(theta,y_opt,'r',label='optimal')
    
        list_col=['b','g','y','m','c']
        idx=0
        for theta_estim in list_theta:
            y=-theta_estim[0]/theta_estim[1]*theta
            ax.plot(theta,y,color=list_col[idx],label=list_labels[idx])
            idx=idx+1
            
    @staticmethod 
    def plotloss(ax,U,Y,x_opt,numberPass,list_histo_x,list_labels,\
                title='logistic loss',plotOptimal=True,axesLegend=True):
        N,d=U.shape
        t=np.arange(0,N*numberPass)

        list_col=['b','g','y','m','c','grey','ivory','olive','brown']
        idx=0
        for history_x in list_histo_x:
            loss=lossOnPath(U,Y,history_x[t],x_opt)
            ax.plot(t,loss,'-.',color=list_col[idx],label=list_labels[idx],linewidth=3)
            idx=idx+1

        if plotOptimal:
            Loss_opt=loss(Y,U,x_opt)
            loss=loss(Y,U,list_histo_x[0][0])-Loss_opt
            ax.plot(t[1:],loss/t[1:],label='$1/t$',linewidth=3)

        ax.set_title(title)
        ax.axvline(d,color='r',linestyle='-.',linewidth=3) 
        ax.axvline(N,color='r',linestyle='-.',linewidth=3)
        if axesLegend:
            ax.set_xlabel('number of iterations ({} pass x {} samples)'.format(numberPass,N))
            ax.set_ylabel('Loss(x)-Loss($x^*$)')
            ax.legend(loc='upper right')
            
    @staticmethod 
    def plotScore(ax,U,Y,x_opt,numberPass,list_histo_x,list_labels,\
                title='percent of good classification',\
                plotOptimal=True,axesLegend=True):
        N,d=U.shape
        t=np.arange(0,N*numberPass)

        list_col=['b','g','y','m','c','grey','ivory','olive','brown']
        idx=0
        for history_x in list_histo_x:
            score=scoreOnPath(U,Y,history_x[t],x_opt)
            ax.plot(t,score,'-.',color=list_col[idx],label=list_labels[idx],linewidth=3)
            idx=idx+1

        if plotOptimal:
            ax.plot(t,np.ones(N*numberPass)*LogScore(Y,U,x_opt),'-.k',label='batch',linewidth=3)
            
            ax.set_title(title)
            ax.axvline(d,color='r',linestyle='-.',linewidth=3) 
            ax.axvline(N,color='r',linestyle='-.',linewidth=3)
            if axesLegend:
                ax.set_xlabel('number of iterations ({} pass x {} samples)'.format(numberPass,N))
                ax.set_ylabel('precision')
                ax.set_ylim(0.4,1)
                ax.legend(loc='lower right')
            
    @staticmethod 
    def lossMesh2D(Y,U,x,y):
        N,d=U.shape
        #meshgrid of shape (2,dimx,dimy)
        X=np.array([x,y])
        # [N,2] tensor [2,dimx,dimy]= [N,dimx,dimy]
        UX=np.tensordot(U,X,axes=(1,0))
        A=np.ones([UX.shape[1],UX.shape[2]])
        # [N,1] outer [dimx,dimy] = [N,dimx.dimy]
        Y=np.outer(Y,A)
        Y=Y.reshape(N,UX.shape[1],UX.shape[2])
        YUX=(2*Y-1)*UX
        YUX=np.clip(YUX,-20,20)
        return np.sum(np.log(1+np.exp(-YUX)),axis=0)

    @staticmethod 
    def plotLossShape(ax,optimal,U,Y,color='k',nbcontours=20,spread=2,Projection3D=False):
        x0=optimal[0]
        y0=optimal[1]
        xmin, xmax, xstep = x0-spread,  x0+spread, 0.1*spread
        ymin, ymax, ystep = y0-spread,  y0+spread, 0.1*spread
        x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
        z = lossMesh2D(Y,U,x,y)
        z0 = loss(Y,U,optimal)
        #z=z/np.max(z)
        
        if Projection3D:
            ax.plot_surface(x,y,z)
            ax.scatter(x0,y0,z0, 'r*')
        else:
            ax.contour(x, y, z, nbcontours, cmap=plt.cm.jet)
            ax.plot(*optimal, 'r*', markersize=18,linewidth=3)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))

    @staticmethod 
    def plotlossPath(ax,path,optimal,U,Y,color='k',nbcontours=20):
    
        xmin, xmax, xstep = np.min(path[:,0])-0.5,  np.max(path[:,0])+0.5, (np.max(path[:,0])-np.min(path[:,0]))/100
        ymin, ymax, ystep = np.min(path[:,1])-0.5,  np.max(path[:,1])+0.5, (np.max(path[:,1])-np.min(path[:,1]))/100
        x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
        z = lossMesh2D(Y,U,x,y)
    
        ax.contour(x, y, z, nbcontours, cmap=plt.cm.jet)
        ax.quiver(path[:-1,0], path[:-1,1], path[1:,0]-path[:-1,0], path[1:,1]-path[:-1,1], scale_units='xy', angles='xy', scale=1, color=color)
        ax.plot(*optimal, 'r*', markersize=18,linewidth=3)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))

############## RMSE ERRORS ##########################
class RMSE:
    @staticmethod 
    def deviations(history_x,history_P,x_opt):
        N,d=history_x.shape
        rmse=np.sqrt(np.sum((history_x-x_opt.T)**2,axis=1)/d)
        TensorCov=history_P.reshape(N,d,d)
        TraceCov=np.asarray([np.sqrt(np.trace(TensorCov[i])/d) for i in range(0,N)])
        return rmse,TraceCov

    @staticmethod 
    def plot(ax,x_opt,numberPass,list_histo_x,list_histo_P,list_labels,plotRMSE=True,\
             title='Covariance matrix on x (rmse error in dash line)',axesLegend=True):
        NIter,d=list_histo_x[0].shape
        #N,d=U.shape
        #NIter=N*numberPass
        N=int(NIter/numberPass)
        t=np.arange(0,NIter)

        ax.axvline(d,color='r',linestyle='-.',linewidth=3) 
        ax.axvline(N,color='r',linestyle='-.',linewidth=3)

        list_col=['b','g','y','m','c','grey','ivory','olive','brown']

        for idx in range(0,len(list_histo_x)):
            history_x=list_histo_x[idx]
            history_P=list_histo_P[idx]
            label=list_labels[idx]
            col=list_col[idx]
            rmse, sqTrCov=deviations(history_x[t],history_P[t,:],x_opt)
            ax.semilogy(t,sqTrCov,color=col,label=label,linewidth=3)
            if plotRMSE:
                ax.semilogy(t,rmse,'-.',color=col,linewidth=3)

        if axesLegend:
            ax.set_title(title)
            ax.set_xlabel('number of iterations ({} pass x {} samples)'.format(numberPass,N))
            ax.set_ylabel('$\sqrt{\mathrm{Tr}\mathrm{Cov}(x)/d}$')
            ax.legend(loc='upper right')
        
    
