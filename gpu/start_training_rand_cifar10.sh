#!/bin/bash

data_path="../datasets/cifar10"
# Modify save_path
save_path="/nv/hcoc1/bxie33/scratch/cifar10/rand_caches"
test_range="6"
train_range="1-5"
layer_def="./example-layers/layers-rand-11pct.cfg"
layer_params="./example-layers/layer-params-rand-11pct.cfg"
data_provider="cifar-cropped"
test_freq="13"
mini="512"
epochs="350"
crop_border="4"

python convnet.py \
  --data-path=$data_path \
  --save-path=$save_path \
  --test-range=$test_range --train-range=$train_range \
  --layer-def=$layer_def \
  --layer-params=$layer_params \
  --data-provider=$data_provider \
  --test-freq=$test_freq \
  --mini=$mini \
  --epochs=$epochs \
  --crop-border=$crop_border \
  --run-id=0
