# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from data import *
import numpy.random as nr
import numpy as n
import random as r
from time import time
from scipy.io import loadmat

class CIFARDataProvider(LabeledMemoryDataProvider):
  def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
    LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
    self.data_mean = self.batch_meta['data_mean']
    self.num_colors = 3
    self.img_size = 32
    # Subtract the mean from the data and make sure that both data and
    # labels are in single-precision floating point.
    for d in self.data_dic:
      # This converts the data matrix to single precision and makes sure that it is C-ordered
      d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
      d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

  def get_next_batch(self):
    epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
    return epoch, batchnum, [datadic['data'], datadic['labels']]

  # Returns the dimensionality of the two data matrices returned by get_next_batch
  # idx is the index of the matrix. 
  def get_data_dims(self, idx=0):
    return self.img_size**2 * self.num_colors if idx == 0 else 1

  # Takes as input an array returned by get_next_batch
  # Returns a (numCases, imgSize, imgSize, 3) array which can be
  # fed to pylab for plotting.
  # This is used by shownet.py to plot test case predictions.
  def get_plottable_data(self, data):
    return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)


class CroppedCIFARDataProvider(LabeledMemoryDataProvider):
  def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
    LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

    self.border_size = dp_params['crop_border']
    self.inner_size = 32 - self.border_size*2
    self.multiview = dp_params['multiview_test'] and test
    self.num_views = 5*2
    self.data_mult = self.num_views if self.multiview else 1
    self.num_colors = 3

    for d in self.data_dic:
      d['data'] = n.require(d['data'], requirements='C')
      d['labels'] = n.require(n.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')

    self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

    self.batches_generated = 0
    self.data_mean = self.batch_meta['data_mean'].reshape((3,32,32))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

  def get_next_batch(self):
    epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

    cropped = self.cropped_data[self.batches_generated % 2]

    self.__trim_borders(datadic['data'], cropped)
    cropped -= self.data_mean
    self.batches_generated += 1
    return epoch, batchnum, [cropped, datadic['labels']]

  def get_data_dims(self, idx=0):
    return self.inner_size**2 * 3 if idx == 0 else 1

  def get_curr_data(self, n_copy):
    curr_batch_idx = (self.batch_idx + len(self.batch_range) - 1) % len(self.batch_range)
    data = self.data_dic[curr_batch_idx]
    n_data = data['data'].shape[1]
    copies = n.zeros((self.get_data_dims(), n_data * n_copy), dtype=n.single)
    for i in range(n_copy):
      self.__trim_borders(data['data'], copies[:, i*n_data:(i+1)*n_data])
    copies -= self.data_mean
    return copies

  # Takes as input an array returned by get_next_batch
  # Returns a (numCases, imgSize, imgSize, 3) array which can be
  # fed to pylab for plotting.
  # This is used by shownet.py to plot test case predictions.
  def get_plottable_data(self, data):
    return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

  def __trim_borders(self, x, target):
    y = x.reshape(3, 32, 32, x.shape[1])

    if self.test: # don't need to loop over cases
      if self.multiview:
        start_positions = [(0,0),  (0, self.border_size*2),
                           (self.border_size, self.border_size),
                           (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
        end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
        for i in xrange(self.num_views/2):
          pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
          target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
          target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
      else:
        pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
        target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
    else:
      for c in xrange(x.shape[1]): # loop over cases
        startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
        endY, endX = startY + self.inner_size, startX + self.inner_size
        pic = y[:,startY:endY,startX:endX, c]
        if nr.randint(2) == 0: # also flip the image with 50% probability
          pic = pic[:,:,::-1]
        target[:,c] = pic.reshape((self.get_data_dims(),))


class DummyConvNetDataProvider(LabeledDummyDataProvider):
  def __init__(self, data_dim):
    LabeledDummyDataProvider.__init__(self, data_dim)

  def get_next_batch(self):
    epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)

    dic['data'] = n.require(dic['data'].T, requirements='C')
    dic['labels'] = n.require(dic['labels'].T, requirements='C')

    return epoch, batchnum, [dic['data'], dic['labels']]

  # Returns the dimensionality of the two data matrices returned by get_next_batch
  def get_data_dims(self, idx=0):
    return self.batch_meta['num_vis'] if idx == 0 else 1


class ImagetNetDataProvider(LabeledDataProvider):
  def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
    LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
    self.data_mean = self.batch_meta['data_mean']
    self.num_colors = 3
    self.img_size = 256

  def get_next_batch(self):
    epoch, batchnum, datadic = LabeledDataProvider.get_next_batch(self)
    # Subtract the mean from the data and make sure that both data and
    # labels are in single-precision floating point.
    # This converts the data matrix to single precision and makes sure that it is C-ordered
    datadic['data'] = n.require((datadic['data'] - self.data_mean),
                                dtype=n.single, requirements='C')
    labels = n.asarray(datadic['labels'], dtype=n.single)
    datadic['labels'] = n.require(labels.reshape((1, datadic['data'].shape[1])),
                                  dtype=n.single, requirements='C')
    return epoch, batchnum, [datadic['data'], datadic['labels']]

  # Returns the dimensionality of the two data matrices returned by get_next_batch
  # idx is the index of the matrix. 
  def get_data_dims(self, idx=0):
    return self.img_size**2 * self.num_colors if idx == 0 else 1

  # Takes as input an array returned by get_next_batch
  # Returns a (numCases, imgSize, imgSize, 3) array which can be
  # fed to pylab for plotting.
  # This is used by shownet.py to plot test case predictions.
  def get_plottable_data(self, data):
    return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)


class CroppedImagetNetDataProvider(LabeledDataProvider):
  def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
    LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
    self.num_colors = 3
    self.img_size = 256
    self.border_size = dp_params['crop_border']
    self.inner_size = self.img_size - self.border_size*2
    self.multiview = dp_params['multiview_test'] and test
    self.num_views = 5*2
    self.data_mult = self.num_views if self.multiview else 1
    self.maximum_batch_size = self.batch_meta['maximum_batch_size']

    self.cropped_data = [n.zeros((self.get_data_dims(), self.maximum_batch_size*self.data_mult), dtype=n.single) for x in xrange(2)]

    self.batches_generated = 0
    #self.data_mean = self.batch_meta['data_mean'].reshape((3,256,256))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))
    self.data_mean = self.batch_meta['data_mean']

  def get_next_batch(self):
    epoch, batchnum, datadic = LabeledDataProvider.get_next_batch(self)

    curr_batch_size = datadic['data'].shape[1]
    datadic['data'] = n.require(datadic['data'],
                                dtype=n.single, requirements='C')
    labels = n.asarray(datadic['labels'], dtype=n.single)
    datadic['labels'] = n.require(n.tile(labels.reshape((1, curr_batch_size)), (1, self.data_mult)), requirements='C')

    cropped = self.cropped_data[self.batches_generated % 2][:, :curr_batch_size * self.data_mult]

    #self.__trim_borders(datadic['data'], cropped)
    #cropped -= self.data_mean
    self.__trim_borders(datadic['data']-self.data_mean, cropped)
    self.batches_generated += 1
    return epoch, batchnum, [cropped, datadic['labels']]

  def get_data_dims(self, idx=0):
    return self.inner_size**2 * 3 if idx == 0 else 1

  # Takes as input an array returned by get_next_batch
  # Returns a (numCases, imgSize, imgSize, 3) array which can be
  # fed to pylab for plotting.
  # This is used by shownet.py to plot test case predictions.
  def get_plottable_data(self, data):
    data_mean = self.data_mean.reshape((3,256,256))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))
    return n.require((data + data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

  def __trim_borders(self, x, target):
    y = x.reshape(3, 256, 256, x.shape[1])

    if self.test: # don't need to loop over cases
      if self.multiview:
        start_positions = [(0,0),  (0, self.border_size*2),
                           (self.border_size, self.border_size),
                           (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
        end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
        for i in xrange(self.num_views/2):
          pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
          target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
          target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
      else:
        pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
        target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
    else:
      for c in xrange(x.shape[1]): # loop over cases
        startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
        endY, endX = startY + self.inner_size, startX + self.inner_size
        pic = y[:,startY:endY,startX:endX, c]
        if nr.randint(2) == 0: # also flip the image with 50% probability
          pic = pic[:,:,::-1]
        target[:,c] = pic.reshape((self.get_data_dims(),))


class MNIST8MDataProvider(LabeledDataProvider):
  def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
    LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
    self.data_mean = self.batch_meta['data_mean']
    self.num_colors = 1
    self.img_size = 28

  def get_batch(self, batch_num):
    old_dic = loadmat(self.get_data_file_name(batch_num))
    dic = {}
    dic['data'] = n.require(old_dic['data'], dtype=n.single,
                            requirements='C')
    dic['labels'] = old_dic['label'].T
    return dic

  def get_next_batch(self):
    epoch, batchnum, datadic = LabeledDataProvider.get_next_batch(self)
    # Subtract the mean from the data and make sure that both data and
    # labels are in single-precision floating point.
    # This converts the data matrix to single precision and makes sure that it is C-ordered

    #datadic['data'] = n.require((datadic['data'] - self.data_mean),
    #                            dtype=n.single, requirements='C')
    datadic['labels'] = n.require(datadic['labels'], dtype=n.single, requirements='C')
    return epoch, batchnum, [datadic['data'], datadic['labels']]

  # Returns the dimensionality of the two data matrices returned by get_next_batch
  # idx is the index of the matrix. 
  def get_data_dims(self, idx=0):
    return self.img_size**2 * self.num_colors if idx == 0 else 1

  # Takes as input an array returned by get_next_batch
  # Returns a (numCases, imgSize, imgSize, 3) array which can be
  # fed to pylab for plotting.
  # This is used by shownet.py to plot test case predictions.
  def get_plottable_data(self, data):
    #return n.require((data + self.data_mean).T.reshape(data.shape[1], 1, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    return n.require((data).T.reshape(data.shape[1], 1, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
