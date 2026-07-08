import numpy as np
from config import RegularConfig

class Util:
    def __init__(self):
        self.config = RegularConfig()
        self.activ = getattr(self, self.config.activation_function, None)
        self.activ_deriv = getattr(self, self.config.activation_function + 'prime', None)

    def softmax(self, n, axis):
        max_val = np.max(n, axis, keepdims=True)
        exp = np.exp(n-max_val)
        return exp / np.sum(exp, axis, keepdims=True)

    def activation(self, n):
        return self.activ(n)
    
    def activationprime(self, n):
        return self.activ_deriv(n)
    #this may not work 100% of the time; we will need a more robust version as things progress

    #relu
    def relu(self, n):
        return np.maximum(0, n) #no need for nonvectorized, slow time/space complex double loop nightmare version

    #derivative of relu for backpropogation
    def reluprime(self, n):
        return (n > 0).astype(float) #again, no need for manual reimplementation

    def mask(self, n, m):
        return np.triu(np.full((n,m), -np.inf), k=1)