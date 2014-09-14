#!/bin/bash

data_path="/nv/hcoc1/bxie33/data/mnist/mnist8m"
save_path="/nv/hcoc1/bxie33/scratch/mnist8m/rand_caches"
test_range="82"
train_range="1-81"
layer_def="./example-layers/layers-rand-mnist8m.cfg"
layer_params="./example-layers/layer-params-rand-mnist8m.cfg"
data_provider="mnist"
test_freq="1"
gpu="3"
mini="512"
epochs="5"

python convnet.py \
  --data-path=$data_path \
  --save-path=$save_path \
  --test-range=$test_range --train-range=$train_range \
  --layer-def=$layer_def \
  --layer-params=$layer_params \
  --data-provider=$data_provider \
  --test-freq=$test_freq \
  --gpu=$gpu \
  --mini=$mini \
  --epochs=$epochs \
  --run-id=0
