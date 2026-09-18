import os

class RegularConfig:
   def __init__(self):
      # paths
      self.project_root = os.path.dirname(os.path.abspath(__file__))
      self.PROJECT_ROOT = self.project_root
      self.weights_base_path = os.path.join(self.project_root, "weights")

      # architecture dispatch
      self.attention_type = "MultiHeadAttention"
      self.torch_attention_type = "GQATrain"
      self.transformer_type = "StandardPreNormTransformer"
      self.ffn_type = "RegularFFN"
      self.norm_method = "LayerNorm"
      self.norm_type = self.norm_method

      # core dimensions
      self.d_model = 128
      self.h = 4

      # attention heads
      self.num_heads = self.h
      self.d_head = self.d_model // self.h
      self.h_q = self.h
      self.h_kv = 2
      self.d_k = self.d_model * self.h
      self.d_v = self.d_model

      # feed-forward
      self.d_ff = self.d_model * self.h

      # sequence length
      self.max_seq_length = self.d_model * 2
      self.max_seq_len = self.max_seq_length

      # vocabulary
      self.vocab_length = 1000
      self.vocab_size = self.vocab_length
      self.min_freq = 30
      self.n = 10000

      # depth
      self.num_layers = 2
      self.num_tblocks = 3
      self.num_blocks = self.num_tblocks

      # training hyperparameters
      self.lr = 0.001
      self.epsilon = 10**-8
      self.epsn = self.epsilon
      self.activation_function = "relu"

      # inference
      self.max_tokens_inference = 50
      self.temperature = 1.5
