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

After all 82 data batches are created, run ``python calc_mean_mnist8m.py`` to compute a batches.meta file.
Please modify the paths inside ``calc_mean_mnist8m.py``.

A script ``start_training_rand_mnist8m.sh`` is provided to start the experiement. Please modify the paths accordingly.

After about 40 batches, the test error reaches about 0.54%.
