#!/bin/sh

# Fill in these environment variables.

# This file and others have been updated to work with CUDA 5.0.

# Only use Fermi-generation cards. Older cards won't work.

# If you're not sure what these paths should be, 
# you can use the find command to try to locate them.
# For example, NUMPY_INCLUDE_PATH contains the file
# arrayobject.h. So you can search for it like this:
# 
# find /usr -name arrayobject.h
# 
# (it'll almost certainly be under /usr)

# CUDA toolkit installation directory.
export CUDA_INSTALL_PATH=/usr/local/packages/cuda/5.0.35

# CUDA SDK installation directory.
export CUDA_SDK_PATH=$CUDA_INSTALL_PATH

# Python include directory. This should contain the file Python.h, among others.
export PYTHON_INCLUDE_PATH=/usr/local/packages/python/2.7.2/gcc-4.4.5/include/python2.7/

# Numpy include directory. This should contain the file arrayobject.h, among others.
export NUMPY_INCLUDE_PATH=/usr/local/packages/python/2.7.2/gcc-4.4.5/lib/python2.7/site-packages/numpy/core/include/numpy

# Intentionally set to blank to avoid a bug in CUDA
export CPLUS_INCLUDE_PATH=

# ATLAS library directory. This should contain the file libcblas.so, among others.
export ATLAS_LIB_PATH=/usr/local/packages/atlas/3.8.4/gcc-4.4.5/lib/

export MKL_LIB_PATH=/usr/local/packages/intel/mkl/10.3/lib/intel64

export INTEL_LIB_PATH=/usr/local/packages/intel/compiler/12.0.0.084/lib/intel64

export MAGMA_INCLUDE_PATH=~/Downloads/include

export MAGMA_LIB_PATH=~/Downloads/lib

make $*

