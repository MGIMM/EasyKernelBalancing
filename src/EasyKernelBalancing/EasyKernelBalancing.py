import numpy as np
from tqdm import tqdm
from numpy import linalg as LA

class KernelBalancing:
    def __init__(self,
                 X,
                 A,
                 Y,
                 _lambda
                 ):
        self.X = X
        self.A = A
        self.Y = Y 
        self._lambda = _lambda
        self.N = X.shape[0]
        self.M = Y.shape[0]
        self.KXX = np.zeros((self.N,self.N))
        self.KXY = np.zeros((self.N,self.M))
        self.KYY = np.zeros((self.M,self.M))
    def K(self,
          x,
          y):
        """
        x,y : np.array
        """
        return np.exp(-LA.norm(x - y)**2*.5)
        #return LA.norm(x - y)**2
        
    def getGramMatrix(self,
                      KXX = None,
                      KXY = None,
                      KYY = None):
        if KXX==None:
            print("Calculating Gram Matrix...")
            for i in range(self.N):
                for j in range(self.N):
                    self.KXX[i,j] = self.K(self.X[i],self.X[j])
            for i in range(self.N):
                for j in range(self.M):
                    self.KXY[i,j] = self.K(self.X[i],self.Y[j])
            for i in range(self.M):
                for j in range(self.M):
                    self.KYY[i,j] = self.K(self.Y[i],self.Y[j])
    def H(self,
          W):
        return -2.*self.A.dot(self.KXY).dot(W[:-1]) + W[:-1].dot(self.KYY).dot(W[:-1]) + self._lambda*np.sum(W[:-1]**2)
        
                
    def nabla_k(self,
                k,
                W):
        k = int(k)
        return -2.*np.dot(np.transpose(self.A),self.KXY[:,k]) + 2.*self._lambda  * W[k] + 2.*np.sum(self.KYY[k,:]*W[:-1]) + 2.*(np.sum(W[:-1])-1.)*W[-1]*W[k]
    
    def nabla_H(self, W):
        gradH = np.zeros(self.M+1)
        for i in range(self.M):
            gradH[i] = self.nabla_k(i,W)
        gradH[-1] = 1. - np.sum(W[:-1])
        return gradH
    def getOptimalWeights(self,
                          learning_rate = 0.001,
                          max_iter = 1000,
                          epsilon = 1e-7):
        W = np.zeros(self.M+1)
        W[:-1] = np.zeros(self.M) + 1./self.M
        W[-1] = 1.
        H_trace = []
        H_old = self.H(W) 
        for i in tqdm(range(max_iter)):
            H_trace += [H_old]
            W -= learning_rate* self.nabla_H(W = W)
            for d in range(self.M+1):
                W[d] = np.max([W[d],0.])
            
            H_new = self.H(W) 
            if  np.abs(H_new- H_old) <= epsilon:
                print("The preset tolorance is reached.")
                break
            H_old = H_new
        print("final objective value:", H_new)
        return W[:-1], np.array(H_trace)
    

