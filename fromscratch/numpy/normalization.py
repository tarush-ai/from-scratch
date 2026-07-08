import numpy as np
import os, sys
from config import RegularConfig

class Normalization:
    def __init__(self, num, path):
        self.config = RegularConfig()
        self.local_module = sys.modules[__name__]
        self.norm = getattr(self.local_module, self.config.norm_method, None)(num, path)
    
    def forward(self, pred, X):
        return self.norm.forward(pred, X)

    def backward(self, dL_dnorm):
        return self.norm.backward(dL_dnorm)


class LayerNorm:
    def __init__(self, num, path):
        numpath = "norm" + str(num)
        self.relpath = os.path.join(path, numpath)
        os.makedirs(self.relpath, exist_ok=True)
        
        self.config = RegularConfig()
        
        self.gamma_path = os.path.join(self.relpath, "gamma.npy")
        self.beta_path = os.path.join(self.relpath, "beta.npy")

        if os.path.exists(self.gamma_path):
            self.gamma = np.load(self.gamma_path)
        else:
            self.gamma = np.ones(self.config.d_model) 
            np.save(self.gamma_path, self.gamma)
        if os.path.exists(self.beta_path):
            self.beta = np.load(self.beta_path)
        else:
            self.beta = np.zeros(self.config.d_model)
            np.save(self.beta_path, self.beta)
    
    def forward(self, pred, X):
        self.pred = pred
        self.X_in = X
        X = pred + X
        self.X = X
        
        self.means = []
        self.stddevs = []
        self.normalized = []
        
        lnorm = []
        for xt in X:
            mean = np.mean(xt)
            stddev = np.std(xt)
            norm_xt = (xt - mean) / np.sqrt(stddev ** 2 + self.config.epsilon)
            
            self.means.append(mean)
            self.stddevs.append(stddev)
            self.normalized.append(norm_xt)
            
            norm = self.gamma * norm_xt + self.beta
            lnorm.append(norm)
        return np.array(lnorm)

    def backward(self, dL):
        dL_dgamma = np.sum(dL * np.array(self.normalized), axis=0)
        dL_dbeta = np.sum(dL, axis=0)
        
        self.gamma -= self.config.lr * dL_dgamma
        self.beta -= self.config.lr * dL_dbeta
        
        dL_dnorm = dL * self.gamma
        
        dL_dX = np.zeros_like(self.X)
        for i in range(len(self.X)):
            mean = self.means[i]
            stddev = self.stddevs[i]
            var = stddev ** 2
            
            N = self.config.d_model
            x_centered = self.X[i] - mean
            
            dL_dvar = np.sum(dL_dnorm[i] * x_centered * -0.5 * (var + self.config.epsilon) ** (-1.5))
            dL_dmean = np.sum(dL_dnorm[i] * -1.0 / np.sqrt(var + self.config.epsilon)) + dL_dvar * np.sum(-2.0 * x_centered) / N
            
            dL_dX[i] = dL_dnorm[i] / np.sqrt(var + self.config.epsilon) + dL_dvar * 2.0 * x_centered / N + dL_dmean / N
        
        dL_dpred = dL_dX
        dL_dX_in = dL_dX
        
        return dL_dpred, dL_dX_in
    
    def save_weights(self):
        np.save(self.gamma_path, self.gamma)
        np.save(self.beta_path, self.beta)
        