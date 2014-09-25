import numpy as n
from gpumodel import *
from scipy.io import loadmat

checkpoint_file = 'pretrained_models/imagenet'
layers_ = IGPUModel.load_checkpoint(checkpoint_file)['model_state']['layers']
layers = {}
for l in layers_:
  layers[l['name']] = l

pca_file = 'pca_imagenet.mat'
pca_proj_mat = n.require(loadmat(pca_file)['v'], dtype=n.single,
                         requirements=['A', 'O', 'W', 'F'])

def makew(name, idx, shape, params=None):
  layer_name = str(params[0])
  if layer_name == 'pca':
    return pca_proj_mat
  else:
    return layers[layer_name]['weights'][idx]


def makeb(name, shape, params=None):
  layer_name = str(params[0])
  return layers[layer_name]['biases']
