import numpy as np
import os
from functional import Util        
from config import RegularConfig

class FFN:
    def __init__(self, path):
        self.relpath = os.path.join(path, "ffn")
        self.config = RegularConfig()
        os.makedirs(self.relpath, exist_ok=True)

        self.layers = [Linear(path, layer, True) for layer in range(self.config.num_layers-1)]
        self.layers.append(Linear(path, self.config.num_layers-1, False))

    def forward(self, X):
        for i in self.layers:
            X = i.forward(X)
        return X
    
    def backward(self, dL_dh):
        for i in self.layers:
            dL_dh = i.backward(dL_dh)
        return dL_dh

class Linear:
    def __init__(self, path, layer_num, activ):
        self.relpath = os.path.join(path, f"ffn/{layer_num}")
        os.makedirs(self.relpath, exist_ok=True)
        self.config = RegularConfig()

        self.activ = activ
        self.util = Util()
        self.Wpath = os.path.join(self.relpath, "W.npy")
        self.bpath = os.path.join(self.relpath, "b.npy")
        if os.path.exists(self.Wpath):
            self.W = np.load(self.Wpath)
        else:
            self.W = np.random.normal(0, 0.02, (self.config.d_model, self.config.d_ff))
            np.save(self.Wpath, self.W)
        
        if os.path.exists(self.bpath):
            self.b = np.load(self.bpath)
        else:
            self.b = np.random.normal(0, 0.02, (self.config.d_ff,))
            np.save(self.bpath, self.b)
    
    def forward(self, X):
        self.X = X
        self.h = X @ self.W + self.b
        if self.activ:
            output = self.util.relu(self.h)
        else: 
            output = self.h
        return output

    def backward(self, dL_dh):
        if self.activ:
            dL_dh2 = dL_dh * self.util.reluprime(self.h)
        else:
            dL_dh2 = dL_dh
        dL_dW = self.h.T @ dL_dh2
        dL_db = np.sum(dL_dh2, axis=0)
        
        self.W -= self.config.lr * dL_dW
        self.b -= self.config.lr * dL_db

        return dL_dh2
    
    def save_weights(self):
        np.save(self.Wpath, self.W)
        np.save(self.bpath, self.b)