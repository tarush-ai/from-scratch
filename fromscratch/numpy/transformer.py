import numpy as np
import os
from attention import Attention
from normalization import Normalization
from ffn import FFN
from functional import Util 
from config import RegularConfig

class PostNormTransformer:
    def __init__(self, tokenizer):
        self.relpath = os.path.join(self.config.PROJECT_ROOT, "data", "weights")
        self.tokenizer = tokenizer
        self.util = Util()
        self.config = RegularConfig()
        self.blocks = [PreNormTransformerBlock(num, self.relpath) for num in range(self.config.num_tblocks)]
        self.Wo_path = os.path.join(self.relpath, "Wo.npy")
        self.actual_vocab_size = len(tokenizer.vocab)
        self.lr = self.config.lr

        if os.path.exists(self.Wo_path):
            self.Wo = np.load(self.Wo_path)
        else:
            self.Wo = np.random.normal(0, 0.02, (self.config.d_model, self.actual_vocab_size))
            np.save(self.Wo_path, self.Wo)
        

    def forward(self, encoded, apply_soft=False):
        self.emb = self.tokenizer.embed(encoded)
        self.pos = self.tokenizer.positional(encoded)
        X = self.emb + self.pos
        self.X = X
        Y = X
        for block in self.blocks:
           Y = block.forward(Y)

        logits = Y @ self.Wo
        if apply_soft:
            return self.util.softmax(logits, -1) #not training
        return logits

    def backward(self, dL_dlogits, input_tokens):
        dL_dWo = self.blocks[-1].output.T @ dL_dlogits
        self.Wo -= self.lr * dL_dWo
        
        dL = dL_dlogits @ self.Wo.T
        for i in range(len(self.blocks)-1, -1, -1):
            dL = self.blocks[i].backward(dL)
        dL_dX = dL
        dL_dE = np.zeros_like(self.tokenizer.E)
        for i in range(len(self.X)):
            dL_dE[input_tokens[i]] += dL_dX[i]
        
        self.tokenizer.E -= self.config.lr * dL_dE
    
    def save_weights(self):
        np.save(self.Wo_path, self.Wo)
        for block in self.blocks:
            block.save_weights()

class PreNormTransformerBlock:
    def __init__(self, num, path):
        block = "block" + str(num)
        self.relpath = os.path.join(path, block)
        self.attentionblock = Attention(self.relpath)

        self.normone = Normalization(1, self.relpath)
        self.ffn = FFN(self.relpath)
        self.normtwo = Normalization(2, self.relpath)

    def forward(self, X):
        self.X = X
        MHA = self.attentionblock.forward(X)
        AddNorm = self.normone.forward(MHA, X)
        FFN = self.ffn.forward(AddNorm)
        output = self.normtwo.forward(FFN, AddNorm)
        self.output = output
        return output

    def backward(self, dL):
        dL_dffn, dL_daddnorm_norm2 = self.normtwo.backward(dL)
        dL_daddnorm_ffn = self.ffn.backward(dL_dffn)
        dL_daddnorm = dL_daddnorm_norm2 + dL_daddnorm_ffn
        dL_dmha, dL_dX_norm1 = self.normone.backward(dL_daddnorm)
        dL_dX_attention = self.attentionblock.backward(dL_dmha)
        
        dL_dX = dL_dX_norm1 + dL_dX_attention
        return dL_dX
    
    def save_weights(self):
        self.attentionblock.save_weights()
        self.normone.save_weights()
        self.ffn.save_weights()
        self.normtwo.save_weights()
