import os

class RegularConfig:
   def __init__(self):  
      self.d_model = 128
      self.h = 4
      self.lr = 0.001
      self.num_blocks = 3
      self.project_root = os.path.dirname(os.path.abspath(__file__))
      self.min_freq = 30
      self.n = 10000
      self.epsilon = 10**-8
      self.max_tokens_inference = 50
      self.temperature = 1.5
      self.vocab_length = 1000

   @property
   def d_k(self):
      return self.d_model * self.h

   @property
   def d_v(self):
      return self.d_model

   @property
   def d_ff(self):
      return self.d_model * self.h

   @property
   def max_seq_length(self):
      return self.d_model * 2

   @property 
   def d_head(self):
      return self.d_k

