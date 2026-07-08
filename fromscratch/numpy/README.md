# NumPy

This is the NumPy version of everything. Directory of relevant files:

```
from_scratch
   |--numpy
      |-- __init__.py
      |-- attention.py:      Implementation of Single Head and Multi Head attention (GQA and other attention types to come)
      |-- embeddings.py:     Implementation of embeddings and positionals (Other embeddings types to come)
      |-- ffn.py:            Implementation of a simple MLP FFN 
      |-- functional.py:     Implementation of activations, softmax, and functional classes (Other functions to come)
      |-- losses.py:         Implementation of CE loss (other loss approaches to come)
      |-- normalization.py:  Implementation of LayerNorm (other normalization approaches like RMSNorm to come)
      |-- optimizers.py:     Currently blank; future implementation of optimizers like Adam / Muon and more to come
      |-- README.md:         This document! 
      |-- tokenizer.py:      Implementation of manual BPE tokenizer; more tokenization approaches/optimizations to come
      |-- transformer.py:    Putting it all together. Implementation of Attention is All You Need PostNorm transformer, many more varieties to come
   ...
   |--config.py:             Regular configuration used for AureliusGPT; slightly mismatched with some of the files but more updates/configurations to come
```

I'm also going to put some training code in here that can actually be leveraged to train NumPy transformers (although I'm not fully sure that it will be useful).

Building in NumPy has given me a strong understanding of gradient flows and the reason for architectural optimizations compared to beforehand. 
It forces me to run manual backpropogation, which involves me using a whiteboard and/or watching some videos to build intuition. 
Many of this code (in the first checked in version) is from AureliusGPT, and honestly, I think I was moving a bit too fast to fully build intuition.
As I rederive all of these classes and components, I will likely either check in the whiteboarding on GitHub using Git LFS or put together a blog post visible on tarush.ai. 
I don't know if this will help anyone, but I hope my learning journey is useful in some way. 
