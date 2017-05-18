# pyTorch_NCE
An implementation of the Noise Contrastive Estimation algorithm for pyTorch. Working, yet not very efficient.
The code closely follows the TensorFlow NCE loss source code, with this being my attempt to adapt parts of it for pyTorch.
Assumes the training data follows a Zipfian distribution, so this version is best used for training language models or word embeddings.


The following papers provide the necessary theoretical background:

Gutmann, Michael, and Aapo Hyvärinen. "Noise-contrastive estimation: A new estimation principle for unnormalized statistical models." AISTATS. Vol. 1. No. 2. 2010.

Mnih, Andriy, and Yee Whye Teh. "A fast and simple algorithm for training neural probabilistic language models." arXiv preprint arXiv:1206.6426 (2012).
