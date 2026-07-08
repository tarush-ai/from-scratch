import numpy as np
from config import RegularConfig
from functional import Util
import os, sys

class Attention:
    def __init__(self, path):
        self.config = RegularConfig()
        current_module = sys.modules[__name__]
        self.attnclass = getattr(current_module, self.config.attention_type, None)(path)

    def forward(self, X):
        return self.attnclass.forward(X)

    def backward(self, dL_dattn):
        return self.attnclass.backward(dL_dattn)

class AttentionClass:
    def __init__(self, path):
        self.relpath = os.path.join(path, "attention")
        os.makedirs(self.relpath, exist_ok=True)

        self.config = RegularConfig()
        self.util = Util()

        self.Wq_path = os.path.join(self.relpath, "Wq.npy")
        self.Wk_path = os.path.join(self.relpath, "Wk.npy")
        self.Wv_path = os.path.join(self.relpath, "Wv.npy")
        self.Wo_path = os.path.join(self.relpath,  "Wo.npy")
        
        if os.path.exists(self.Wq_path):
            self.Wq = np.load(self.Wq_path)
        else:
            self.Wq = np.random.normal(0, 0.02, (self.config.d_model, self.config.d_k))
            np.save(self.Wq_path, self.Wq)
        
        if os.path.exists(self.Wk_path):
            self.Wk = np.load(self.Wk_path)
        else:
            self.Wk = np.random.normal(0, 0.02, (self.config.d_model, self.config.d_k))
            np.save(self.Wk_path, self.Wk)
        
        if os.path.exists(self.Wv_path):
            self.Wv = np.load(self.Wv_path)
        else:
            self.Wv = np.random.normal(0, 0.02, (self.config.d_model, self.config.d_v))
            np.save(self.Wv_path, self.Wv)
        
        if os.path.exists(self.Wo_path):
            self.Wo = np.load(self.Wo_path)
        else:
            self.Wo = np.random.normal(0, 0.02, (self.config.d_model, self.config.d_model))
            np.save(self.Wo_path, self.Wo)
    
    def forward(self, X):
        pass

    def backward(self, dL_dattn):
        pass

class MultiHeadAttention(AttentionClass):
    def __init__(self, path):
        super().__init__(path)
        self.heads = [AttentionHead() for i in range(self.config.h)]

    def forward(self, X):
        self.X = X
        Q = X @ self.Wq #t * d_k
        K = X @ self.Wk #t * d_k
        V = X @ self.Wv #t * d_v
        
        self.Q = Q
        self.K = K
        self.V = V
    
        passes = []
        for i, head in enumerate(self.heads):
            startindex = i * self.config.d_head
            endindex = (i+1) * self.config.d_head 
            Qh = Q[:, startindex:endindex]
            Kh = K[:, startindex:endindex]
            Vh = V[:, startindex:endindex]

            passes.append(head.forward(Qh, Kh, Vh))
        
        self.A = np.concatenate(passes, axis=1) #t * d_model

        MHA = self.A @ self.Wo
        return MHA
        #add in shape assertions later down the line; ensures that training is going smoothly

    def backward(self, dL_dmha):
        dL_dWo = self.A.T @ dL_dmha
        self.Wo -= self.config.lr * dL_dWo
        
        dL_dA = dL_dmha @ self.Wo.T
        
        dL_dheads = []
        for i in range(self.config.h):
            startindex = i * self.config.d_head
            endindex = (i+1) * self.config.d_head
            dL_dheads.append(dL_dA[:, startindex:endindex])
        
        dL_dQ = np.zeros_like(self.Q)
        dL_dK = np.zeros_like(self.K)
        dL_dV = np.zeros_like(self.V)
        
        for i, head in enumerate(self.heads):
            startindex = i * self.config.d_head
            endindex = (i+1) * self.config.d_head
            
            Qh = self.Q[:, startindex:endindex]
            Kh = self.K[:, startindex:endindex]
            Vh = self.V[:, startindex:endindex]
            
            dL_dQh, dL_dKh, dL_dVh = head.backward(dL_dheads[i], Qh, Kh, Vh)
            
            dL_dQ[:, startindex:endindex] = dL_dQh
            dL_dK[:, startindex:endindex] = dL_dKh
            dL_dV[:, startindex:endindex] = dL_dVh
        
        dL_dWq = self.X.T @ dL_dQ
        dL_dWk = self.X.T @ dL_dK
        dL_dWv = self.X.T @ dL_dV
        
        self.Wq -= self.config.lr * dL_dWq
        self.Wk -= self.config.lr * dL_dWk
        self.Wv -= self.config.lr * dL_dWv
        
        dL_dX = dL_dQ @ self.Wq.T + dL_dK @ self.Wk.T + dL_dV @ self.Wv.T
        
        return dL_dX
    
    def save_weights(self):
        np.save(self.Wq_path, self.Wq)
        np.save(self.Wk_path, self.Wk)
        np.save(self.Wv_path, self.Wv)
        np.save(self.Wo_path, self.Wo)

class AttentionHead():
    def __init__(self):    
        self.util = Util()
        self.config = RegularConfig()
    
    def forward(self, Qh, Kh, Vh):
        scores = Qh @ Kh.T
        scores /= np.sqrt(self.config.d_head)
        self.scores_pre_mask = scores.copy()
        scores += self.util.mask(len(scores), len(scores[0]))
        self.scores = scores
        
        self.attention_weights = self.util.softmax(scores, -1)
        attentionh = self.attention_weights @ Vh
        return attentionh
    
    def backward(self, dL_dattentionh, Qh, Kh, Vh):
        dL_dVh = self.attention_weights.T @ dL_dattentionh
        
        dL_dweights = dL_dattentionh @ Vh.T
        
        dL_dscores = np.zeros_like(self.scores)
        for i in range(len(self.attention_weights)):
            s = self.attention_weights[i:i+1, :]
            ds = dL_dweights[i:i+1, :]
            dL_dscores[i] = (ds - np.sum(ds * s, axis=1, keepdims=True)) * s
        
        dL_dscores /= np.sqrt(self.config.d_head)
        
        dL_dQh = dL_dscores @ Kh
        dL_dKh = dL_dscores.T @ Qh
        
        return dL_dQh, dL_dKh, dL_dVh    

class SingleHeadAttention(AttentionClass):
    def __init__(self,path):
        super().__init__(path)

    def forward(self, X):
        self.X = X
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        
        scores = Q @ K.T
        scores /= np.sqrt(self.config.d_k)
        scores_norm = scores
        scores_norm += self.util.mask(len(scores), len(scores[0]))
        scores_soft = self.util.softmax(scores_norm, axis=-1) #I think?
        output = scores_soft @ V
        
        return output

    def backward(self):
        #to be implemented
        pass

class GroupedQueryAttention(AttentionClass):
    #to be implemented once I know what it is lol
    pass


