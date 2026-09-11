import numpy as np
from config import RegularConfig
import os

class Embeddings:
   def __init__(self, path):
      self.relpath = os.path.join(path, "attention")
      os.makedirs(self.relpath, exist_ok=True)
      
      self.config = RegularConfig()
      self.E_path = os.path.join(self.relpath, "embeddings")

      if os.path.exists(self.E_path):
         self.E = np.load(self.E_path)
      else:
         self.E = np.random.normal(0,0.02,(self.config.vocab_length, self.config.d_model))   
      
      
   def embed(self, encoded):
        X = np.array([self.E[i] for i in encoded])
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