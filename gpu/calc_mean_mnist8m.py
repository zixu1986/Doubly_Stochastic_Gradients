from scipy.io import loadmat
import numpy as n
from glob import glob
import cPickle as pickle

file_idx = range(1, 82)
# Please modify these paths according to your location setting
src_pattern = '/nv/hcoc1/bxie33/data/mnist8m_dataset/data_batch_%i.mat'
output_file = '/nv/hcoc1/bxie33/data/mnist8m_dataset/batches.meta'

img_mean = n.zeros((1, 784), dtype=n.single, order='C')
num_img = 0.0
for idx in file_idx:
  fpath = src_pattern % idx
  mat_dic = loadmat(fpath)
  data = n.require(mat_dic['data'], dtype=n.single,
                   requirements='C')
  img_mean += n.sum(data, axis=1)
  num_img += data.shape[1]

img_mean /= num_img
img_mean.reshape(784, 1)
dic = {}
dic['num_vis'] = 784
dic['data_mean'] = img_mean

f = open(output_file, 'wb')
pickle.dump(dic, f, -1)
