import math
import random
import numpy as np
import timeit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from mpmath import *
mp.dps = 15; mp.pretty = True
#from numba import jit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gvecs
from sklearn.utils import shuffle



def qmc_r1sl_sample(n, d, shift=None, gvs=gvecs.cbcpt_dn1_100):
    """
    Generate a set of rank-1 shifted-lattice points

    :param n:
        Minimum number of points to generate (will be rounded up to the next largest generating vector)
    :param d:
        Dimension of points to generate
    :param shift:
        Shift of lattice (must have `dim` dimensions)
    :param gvs:
        The generating vectors of the Rank-1 Lattice
    :return:
        Array of lattice points
    """
    gv_n = list(filter(lambda i: i >= n, gvs.keys()))[0]  # get key of first gen vec >= n
    gv = np.array(gvs[gv_n][:d])
    if shift is None:
        shift = np.zeros(d)
    res = np.array([[math.modf(x)[0] for x in ((i * gv) % gv_n) / float(gv_n) + shift] for i in range(gv_n)])
    return res

class PNNI:
    def __init__(self,node,activ,num_net,dim,fix,I,cons,datasize):
        self.node=node
        self.activ=activ
        self.num_net=num_net
        self.dim=dim
        self.fix=fix
        self.I=I
        self.cons=cons
        self.datasize=datasize
    def tf(self,x,y,epoch):
        if self.activ==1:
            activ='sigmoid'
        else:
            activ='relu'
        with tf.device('/GPU:0'):
            model = keras.Sequential([
                keras.layers.Dense(self.node, activation=activ),
                keras.layers.Dense(1)
            ])
            model.compile(
                loss=tf.keras.losses.mae,
                optimizer='adam',
                metrics=['mae']
            )
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                    patience=5, min_lr=0.001)
            model.fit(x, y, epochs=epoch, verbose=0)
            print('Finish fitting')
        return model

    def sk(self,x,y):
        if self.activ==1:
            activ='logistic'
        else:
            activ='relu'
        # mlpregression with relu activation with changed hyperparameter lbfgs
        mlp_sec1_l = MLPRegressor(hidden_layer_sizes=(k,),
                               max_iter = 1000000000,activation = activ,
                               solver = 'adam',shuffle=True,warm_start=True,alpha=0.001,
                               learning_rate_init=0.0001, tol=1e-15)
        # fitting NN
        mlp_sec1_l.fit(x, y)
        return mlp_sec1_l

    # sign of sigmoid integration
    def addIndex(self,l):
        ret = []
        for v in l:
            ret.append(v+[0])
            ret.append(v+[1])
        return ret
    def sigs(self,l):
        return int((-1)**(sum(l)-len(l)))
    def sig_lims(self,n):
        ret = [[]]
        for i in range(n):
            ret = self.addIndex(ret)
        if n%2==0:
            signs = [self.sigs(x) for x in ret]
        else:
            signs = [-1*self.sigs(x) for x in ret]
        return ret, signs
    def re_lims(self,n):
        ret = [[]]
        for i in range(n):
            ret = self.addIndex(ret)
        signs = [self.sigs(x) for x in ret]
        return ret, signs
    # numerical calculation of integral
    # Sigmoid
    def sig_inte(self,dim,fix,inp,wi,wo,bi,bo):
        vecpoly = np.vectorize(fp.polylog, signature='(),()->()', otypes=[float])
        if fix > 0:
            dim=dim-fix
            cons=np.dot(inp,wi[dim:])
            poly =[]
            for i in range(len(self.sig_lims(dim)[1])):
                weight=np.array(wi[:dim]).T*self.sig_lims(dim)[0][i]
                temp=self.sig_lims(dim)[1][i]*vecpoly(dim,-np.exp(-(bi+cons)-sum(weight.T)))
                poly.append(np.array(temp))
            sumpoly=np.sum(poly,axis=0)
            result=(bo+np.sum((1+sumpoly/np.prod(wi[:dim],axis=0))*wo.T,axis=1))
            return result
        else:
            poly =[]
            for i in range(len(self.sig_lims(dim)[1])):
                weight=np.array(wi[:dim]).T*self.sig_lims(dim)[0][i]
                temp=self.sig_lims(dim)[1][i]*vecpoly(dim,-np.exp(-bi-sum(weight.T)))
                poly.append(np.array(temp))
            sumpoly=np.sum(poly,axis=0)
            result=(bo+np.sum((1+sumpoly/np.prod(wi[:dim],axis=0))*wo.T))[0]
            return result
    # Relu
    def re_inte(self,dim,fix,inp,wi,wo,bi,bo):
        def relu(x):
            return np.clip(x,a_min=0.0, a_max=None)
        if fix > 0:
            temp=0
            dim=dim-fix # redefine dimension
            cons=np.dot(inp,wi[dim:]) # fixed variable makes to constant value
            for i in range(len(self.re_lims(dim)[1])):
                weight=np.array(wi[:dim]).T*self.re_lims(dim)[0][i] # weight input with boundaries
                temp+=self.re_lims(dim)[1][i]*1/np.prod(np.arange(dim+2)[1:])*relu(bi+cons+sum(weight.T))**(dim+1)/np.prod(wi[:dim],axis=0) # sum all weight input and bias input including boundaries
            result=(bo+np.sum((temp)*wo.T,axis=1)) # add bias output with weight output
            return result
        else:
            poly =[]
            temp=0
            for i in range(len(self.re_lims(dim)[1])):
                weight=np.array(wi[:dim]).T*self.re_lims(dim)[0][i]
                temp+=self.re_lims(dim)[1][i]*1/np.prod(np.arange(dim+2)[1:])*relu(bi+sum(weight.T))**(dim+1)/np.prod(wi,axis=0)
            result=(bo+np.sum((1+temp)*wo.T[0]))[0]
            return result
    def tf_coe(self,mlpr):
        weight_input = []
        weight_output = mlpr.layers[1].get_weights()[0]
        bias_input = mlpr.layers[0].get_weights()[1]
        bias_output = mlpr.layers[1].get_weights()[1]
        for i in range(self.dim):
            weight_input.append(mlpr.layers[0].get_weights()[0][i])
        return weight_input, weight_output, bias_input, bias_output
    def sk_coe(self,mlpr):
        weight_input = []
        weight_output = mlpr.coefs_[1]
        bias_input = mlpr.intercepts_[0]
        bias_output = mlpr.intercepts_[1]
        for i in range(self.dim):
            weight_input.append(mlpr.coefs_[0][i])
        return weight_input, weight_output, bias_input, bias_output
    def data(self):
        intdim=self.dim-self.fix
        qmcx=qmc_r1sl_sample(self.datasize, intdim)
        qmcf=qmc_r1sl_sample(self.datasize, self.fix)*(-27)+np.array(-3)
        qmc=np.concatenate((qmcx,qmcf),axis=1)
        qmcxs= shuffle(qmcx, random_state=0)
        qmcfs= shuffle(qmcf, random_state=1)
        con=np.concatenate((qmcxs,qmcfs),axis=1)
        X=con
        Y=[]
        Y2=[]
        for i in range(X.shape[0]):
            Y2.append(self.I(X[i,:intdim],X[i,intdim:],[])/(self.I(np.array([1/2]*(intdim)),X[i,intdim:],[]))) # make hypercube
            Y.append(self.I(X[i,:intdim],X[i,intdim:],[])) # normal Y values
        Y=np.array(Y)
        Y2=np.array(Y2)
        return X,Y,Y2
    def get_result(self,method,norm,epoch):
        X,Y,Y2=self.data()
        if norm == 0:
            Y2=Y
        if method == 1:
            if self.activ == 1:
                ANS={}
                for k in range(self.num_net):
                    wi, wo, bi, bo = self.tf_coe(self.tf(X,Y2,epoch))
                    ans=[]
                    for i in range(len(self.cons)):
                        ans.append(self.sig_inte(self.dim,self.fix,self.cons[i],wi,wo,bi,bo)[0])
                    ANS[k]=np.array(ans)
                res=sum(ANS.values())/self.num_net
            else:
                ANS={}
                for k in range(self.num_net):
                    wi, wo, bi, bo = self.tf_coe(self.tf(X,Y2,epoch))
                    ans=[]
                    for i in range(len(self.cons)):
                        ans.append(self.re_inte(self.dim,self.fix,self.cons[i],wi,wo,bi,bo)[0])
                    ANS[k]=np.array(ans)
                res=sum(ANS.values())/self.num_net
        else:
            if self.activ == 1:
                ANS={}
                for k in range(self.num_net):
                    wi, wo, bi, bo = self.sk_coe(self.sk(X,Y2,epoch))
                    ans=[]
                    for i in range(len(self.cons)):
                        ans.append(self.sig_inte(self.dim,self.fix,self.cons[i],wi,wo,bi,bo)[0])
                    ANS[k]=np.array(ans)
                res=sum(ANS.values())/self.num_net
            else:
                ANS={}
                for k in range(self.num_net):
                    wi, wo, bi, bo = self.sk_coe(self.sk(X,Y2,epoch))
                    ans=[]
                    for i in range(len(self.cons)):
                        ans.append(self.re_inte(self.dim,self.fix,self.cons[i],wi,wo,bi,bo)[0])
                    ANS[k]=np.array(ans)
                res=sum(ANS.values())/self.num_net
        return res