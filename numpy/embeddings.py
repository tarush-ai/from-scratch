import numpy as np
from config import RegularConfig

class Embeddings:
   def __init__(self):
      self.config = RegularConfig()
   
   def embed(self, encoded, E):
        X = np.array([E[i] for i in encoded])
        return X

   def positional(self, encoded):
      positionals = np.zeros((len(encoded), self.config.d_model))
      for pos in range(len(encoded)):
         for i in range(0, self.config.d_model, 2):
               denominator = self.config.n ** (i / self.config.d_model)
               positionals[pos, i] = np.sin(pos / denominator)
               if i + 1 < self.config.d_model:
                  positionals[pos, i + 1] = np.cos(pos / denominator)
      return positionals