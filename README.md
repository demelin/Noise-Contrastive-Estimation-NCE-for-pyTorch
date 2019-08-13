# Noise Contrastive Estimation for pyTorch

## Overview
This repository contains a re-implementation of the [Noise Contrastive Estimation algorithm](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf), implemented in pyTorch. While the algorithm is fully functional, it is not very efficient. 

## Implementation details
As provided, the implementation assumes that its input data follows a Zipfian distribution, making it particularly suitable for training language models or word embeddings. In case the built-in (Zipfian) sampler is used to obtain the distractor items, indices representing the data classes have to be sorted in the order of descending frequency, i.e. index 0 should correspond to the most frequent word in the input data.

## Acknowledgement
The provided code closely follows the [TensorFlow](https://github.com/tensorflow/tensorflow) NCE-loss implementation. As such, this project should be seen as an attempt to adopt the TF code for use within pyTorch.

## Note
This re-implementation was completed with personal use in mind and is, as such, not actively maintained. You are, however, very welcome to extend or adjust it according to your own needs, should you find it useful. Happy coding :) .
