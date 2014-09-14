This is my fork of the ``cuda-convnet`` convolutional neural network
implementation written by Alex Krizhevsky, and modified by Daniel Nouri.

For the documentation of ``cuda-convnet`` Please refer to  
`MAIN DOCUMENTATION HERE <http://code.google.com/p/cuda-convnet/>`_.

Additional Features
===================
Added a random feature layer that generates random features incrementally.

For details, please refer to the paper
(http://arxiv.org/abs/1407.5599)

Compile
===================
Please modified the path variables in ``build.sh`` according to your own machine setup.

This software also requires MAGMA, which can be found at
http://icl.cs.utk.edu/magma/index.html.

To compile, run ``./build.sh``

Dataset
===================
Please download the MNIST 8M dataset and preprocess it according to the description in ``../matlab/README``

