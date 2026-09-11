import os

class RegularConfig:
   def __init__(self):  
      #architectural decisions
      self.transformer_type = "PreNormTransformerEndNorm"
      self.attention_type = "MultiHeadAttention"
      self.ffn_type = "RegularFFN"
      self.norm_method = "LayerNorm"

      #dimensions
      self.d_model = 128
      self.h = 4

      # hyperparameters
      self.lr = 0.001
      self.num_layers = 2
      self.num_tblocks = 3
      self.min_freq = 30
      self.n = 10000
      self.epsilon = 10**-8
      self.max_tokens_inference = 50
      self.temperature = 1.5
      self.vocab_length = 1000
      self.activation_function = "relu"

      #weight reading
      self.project_root = os.path.dirname(os.path.abspath(__file__))
      self.PROJECT_ROOT = self.project_root
      
      

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
      return self.d_model // self.h
   
   @property
   def num_heads(self):
      return self.h
   
   @property
   def num_blocks(self):
      return self.num_tblocks
   
   @property 
   def vocab_size(self):
      return self.vocab_length