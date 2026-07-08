import numpy as np

class CrossEntropy:
   def ce_loss(self, targets, logits):
         logits_max = np.max(logits, axis=-1, keepdims=True)
         logits_shifted = logits - logits_max
         
         log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=-1, keepdims=True))
         log_probs = logits_shifted - log_sum_exp
         
         target_log_probs = log_probs[np.arange(len(targets)), targets]
         
         return -np.mean(target_log_probs)
    
   def ce_loss_gradient(self, output_tokens, logits):
      probs = self.util.softmax(logits, axis=-1)

      grad = probs.copy()
      grad[np.arange(len(output_tokens)), output_tokens] -= 1
      
      grad = grad / len(output_tokens)
      return grad